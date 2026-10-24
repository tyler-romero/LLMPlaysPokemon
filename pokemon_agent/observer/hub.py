"""In-process event hub for the live observer page.

Agents and the emulator publish events here; the observer server streams them to browsers. Publishing never blocks
the agent: slow subscribers drop media and, at worst, are disconnected.
"""

import base64
import io
import json
import os
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

FRAME_FPS = 30
AUDIO_DOWNSAMPLE = 2  # 48 kHz from PyBoy -> 24 kHz to the browser
LOG_HISTORY = 400
INPUT_HISTORY = 200
SUBSCRIBER_QUEUE_SIZE = 2000


class Subscriber:
    def __init__(self):
        self.queue: queue.Queue = queue.Queue(maxsize=SUBSCRIBER_QUEUE_SIZE)
        self.dropped = threading.Event()


class ObserverHub:
    """Collects run events for the observer page.

    With `event_log`, events (except media) are appended to a JSONL file so a resumed run keeps its history;
    `resume_seq` replays the file up to that event and drops anything later (events after the checkpoint).
    """

    def __init__(self, event_log: Path | None = None, resume_seq: int | None = None):
        self._lock = threading.Lock()
        self._subscribers: list[Subscriber] = []
        self._seq = 0
        self.run: dict[str, Any] = {}
        self.state: dict[str, Any] = {}
        self.memory_listing: str = ""
        self.log: deque[dict] = deque(maxlen=LOG_HISTORY)
        self.inputs: deque[dict] = deque(maxlen=INPUT_HISTORY)
        self.input_count = 0
        self.step = 0
        self.audio_sample_rate = 48_000 // AUDIO_DOWNSAMPLE

        self._frame: np.ndarray | None = None
        self._frame_dirty = False
        self._last_frame_b64: str | None = None
        self._audio_chunks: list[np.ndarray] = []
        self._event_log = None
        if event_log:
            if resume_seq is not None and event_log.exists():
                self._replay(event_log, resume_seq)
            self._event_log = open(event_log, "a", buffering=1)

        self._pump_stop = threading.Event()
        self._pump = threading.Thread(target=self._pump_media, name="observer-media", daemon=True)
        self._pump.start()

    @property
    def seq(self) -> int:
        return self._seq

    def _replay(self, path: Path, max_seq: int):
        kept = []
        for line in path.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut off by a crash
            if event.get("seq", 0) <= max_seq:
                kept.append(event)
                self._apply(event)
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(e, default=str) + "\n" for e in kept))
        os.replace(tmp, path)
        self._seq = max((e["seq"] for e in kept), default=0)

    # ----- publishing -----------------------------------------------------------------------------------------------

    def publish(self, kind: str, **data):
        """Record an event and broadcast it to every connected page."""
        with self._lock:
            self._seq += 1
            event = {"seq": self._seq, "t": time.time(), "kind": kind, **data}
            self._apply(event)
            if self._event_log:
                self._event_log.write(json.dumps(event, default=str) + "\n")
            self._broadcast(event)

    def _apply(self, event: dict):
        """Fold an event into the hub's current state and history."""
        kind = event["kind"]
        data = {k: v for k, v in event.items() if k not in ("seq", "t", "kind")}
        if kind == "run":
            self.run = {**self.run, **data}
        elif kind == "run_start":
            # A resumed run keeps its original start time.
            started = self.run.get("started_at") or event["t"]
            self.run = {**self.run, "started_at": started, "steps": data["steps"], "ended_at": None}
        elif kind == "run_end":
            self.run = {**self.run, "ended_at": event["t"]}
        elif kind == "usage":
            self.run = {**self.run, "input_tokens": data["input"]}
        elif kind == "state":
            self.state = data["state"]
        elif kind == "memory":
            self.memory_listing = data["listing"]
        elif kind == "input":
            self.input_count += 1
            self.inputs.append(event)
        elif kind == "step":
            self.step = data["step"]
        if kind in LOGGED_KINDS:
            self.log.append(event)

    def on_frame(self, screen: np.ndarray, sound: np.ndarray | None):
        """Called by the emulator after every frame; must stay cheap."""
        with self._lock:
            self._frame = screen.copy()
            self._frame_dirty = True
            self._last_frame_b64 = None
            if sound is not None and len(sound):
                self._audio_chunks.append(sound[::AUDIO_DOWNSAMPLE].copy())

    def on_button(self, button: str, wait: bool):
        self.publish("input", button=button, wait=wait, step=self.step + 1)

    # ----- subscribers ----------------------------------------------------------------------------------------------

    def subscribe(self) -> tuple[Subscriber, dict]:
        """Register a page; returns its subscriber and a snapshot to render before live events."""
        subscriber = Subscriber()
        with self._lock:
            snapshot = {
                "kind": "snapshot",
                "seq": self._seq,
                "t": time.time(),
                "run": self.run,
                "state": self.state,
                "memory": self.memory_listing,
                "log": list(self.log),
                "inputs": list(self.inputs),
                "inputCount": self.input_count,
                "step": self.step,
                "frame": self._last_frame_b64 or _encode_frame(self._frame),
                "audioSampleRate": self.audio_sample_rate,
            }
            self._subscribers.append(subscriber)
        return subscriber, snapshot

    def unsubscribe(self, subscriber: Subscriber):
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def close(self):
        self._pump_stop.set()
        with self._lock:
            if self._event_log:
                self._event_log.close()
                self._event_log = None

    def _broadcast(self, event: dict, droppable: bool = False):
        for subscriber in list(self._subscribers):
            try:
                subscriber.queue.put_nowait(event)
            except queue.Full:
                if not droppable:
                    # The page fell far behind; drop it so it reconnects and resyncs from a fresh snapshot.
                    self._subscribers.remove(subscriber)
                    subscriber.dropped.set()

    # ----- media pump -----------------------------------------------------------------------------------------------

    def _pump_media(self):
        interval = 1 / FRAME_FPS
        while not self._pump_stop.wait(interval):
            with self._lock:
                frame = self._frame if self._frame_dirty else None
                self._frame_dirty = False
                audio = np.concatenate(self._audio_chunks) if self._audio_chunks else None
                self._audio_chunks = []
                has_subscribers = bool(self._subscribers)
            if not has_subscribers or (frame is None and audio is None):
                continue
            event: dict[str, Any] = {"kind": "media", "t": time.time()}
            if frame is not None:
                frame_b64 = _encode_frame(frame)
                event["frame"] = frame_b64
                with self._lock:
                    self._last_frame_b64 = frame_b64
            if audio is not None:
                event["audio"] = base64.b64encode(audio.astype(np.int8).tobytes()).decode()
            with self._lock:
                self._broadcast(event, droppable=True)


def _encode_frame(frame: np.ndarray | None) -> str | None:
    if frame is None:
        return None
    buffer = io.BytesIO()
    Image.fromarray(frame[:, :, :3]).save(buffer, format="PNG", optimize=False, compress_level=1)
    return base64.b64encode(buffer.getvalue()).decode()


LOGGED_KINDS = {"thought", "tool", "tool_result", "summary", "notice", "step", "run_start", "run_end", "checkpoint"}
