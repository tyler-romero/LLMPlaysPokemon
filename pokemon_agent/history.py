"""Read-only, eventually consistent index over the Copilot SDK session log (events.jsonl), for the `history` tool.

The index exposes three kinds of items, each with a short unique ID (a suffix of its event ID):
* prompt: a turn's message to the model (game state text) and its screenshot
* response: the model's reasoning summary, visible text, and requested tool calls
* tool: one tool call with its arguments, result, success, and any images it returned

Compaction summaries are kept apart, in `summaries` (the model's own conclusions when its conversation was condensed,
not a record of what happened). Hooks, usage, system messages, and other bookkeeping events are not indexed. The log is
read
incrementally: each refresh only parses complete lines appended since the last one, so the turn in progress may not be
visible yet. Images are not kept in memory; their location in the log is recorded and they are read on demand.
"""

import base64
import json
import re
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pokemon_agent.tools.base import TOOL_FAILURE_PREFIXES

MIN_ID_CHARS = 4
SNIPPET_CHARS = 70
LINE_CHARS = 160
MAX_ITEM_CHARS = 20_000


@dataclass
class Summary:
    """One compaction summary: what the conversation was condensed into at that turn."""

    number: int  # the SDK's checkpoint number, counting from 1
    turn: int | None
    title: str
    text: str


@dataclass
class HistoryItem:
    event_id: str
    kind: str  # prompt | response | tool
    time: float
    text: str  # searchable / displayed text
    summary: str  # one-line description for windows and results
    turn: int | None = None
    tool: str | None = None
    ok: bool | None = None
    location: str | None = None
    images: list = field(default_factory=list)  # ("asset", asset_id) or ("inline", line_offset, index)


def _epoch(timestamp: str) -> float:
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()


def _one_line(text: str, limit: int = LINE_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _reasoning_text(data: dict) -> str:
    """The response's reasoning summary, with each part's title on its own line."""
    parts = []
    for block in (data.get("reasoningBlocks") or {}).get("blocks") or []:
        if not isinstance(block, dict):
            continue
        for summary in block.get("summary") or []:
            if text := (summary.get("text") or "").strip():
                parts.append(text)
    return "\n\n".join(parts) or (data.get("reasoningText") or "").strip()


class HistoryIndex:
    """Index of one SDK session log; turns are numbered from the observer's step log when one is given."""

    def __init__(self, events_path: Path, steps_log: Path | None = None):
        self.events_path = Path(events_path)
        self.steps_log = Path(steps_log) if steps_log else None
        self.items: list[HistoryItem] = []
        self.summaries: list[Summary] = []
        self._by_call: dict[str, HistoryItem] = {}
        self._assets: dict[str, tuple[int, int]] = {}  # asset id -> (line offset, line length)
        self._offset = 0
        self._step_offset = 0
        self._step_times: list[float] = []  # completion time of each step, in order
        self._step_numbers: list[int] = []
        self._short_ids: dict[str, str] = {}

    # ----- loading ------------------------------------------------------------------------------------------------

    def refresh(self):
        self._refresh_steps()
        if not self.events_path.exists():
            return
        new_items = False
        with open(self.events_path, "rb") as f:
            f.seek(self._offset)
            while True:
                start = f.tell()
                line = f.readline()
                if not line.endswith(b"\n"):
                    break  # a line still being written; read it next time
                self._offset = f.tell()
                new_items |= self._ingest(line, start)
        if new_items:
            self._short_ids = self._shortest_ids()

    def _refresh_steps(self):
        if not self.steps_log or not self.steps_log.exists():
            return
        with open(self.steps_log, "rb") as f:
            f.seek(self._step_offset)
            for line in f:
                if not line.endswith(b"\n"):
                    break
                self._step_offset += len(line)
                if b'"kind": "step"' not in line:
                    continue
                event = json.loads(line)
                self._step_times.append(event["t"])
                self._step_numbers.append(event["step"])

    def _turn_at(self, time: float) -> int | None:
        if not self._step_times:
            return None
        index = bisect_right(self._step_times, time)
        return (self._step_numbers[index - 1] if index else 0) + 1

    def _ingest(self, line: bytes, start: int) -> bool:
        kind = re.match(rb'\{"type":\s*"([a-z_.]+)"', line)
        event_type = kind[1].decode() if kind else ""
        if event_type == "session.binary_asset":
            asset = re.search(rb'"assetId":\s*"([^"]+)"', line)
            if asset:
                self._assets[asset[1].decode()] = (start, len(line))
            return False
        if event_type not in (
            "user.message", "assistant.message", "tool.execution_start", "tool.execution_complete",
            "session.compaction_complete",
        ):
            return False
        event = json.loads(line)
        data = event.get("data") or {}
        time = _epoch(event["timestamp"])
        if event_type == "session.compaction_complete":
            self._add_summary(data, time)
            return False
        if event_type == "user.message":
            content = data.get("content") or ""
            location = re.search(r'"location": "([^"]+)"', content)
            coords = re.search(r'"map_coordinates": \[\s*(-?\d+),\s*(-?\d+)\s*\]', content)
            dialog = re.search(r'"dialog": (null|"(?:[^"\\]|\\.)*")', content)
            place = location[1] if location else "no location"
            if coords:
                place += f" ({coords[1]},{coords[2]})"
            dialog_text = json.loads(dialog[1]) if dialog else None
            summary = place + (f" · dialog: {_one_line(dialog_text, 60)}" if dialog_text else "")
            images = [("asset", a["assetId"]) for a in data.get("attachments") or [] if a.get("assetId")]
            prompt = HistoryItem(event["id"], "prompt", time, content, summary, images=images)
            prompt.location = location[1] if location else None
            self._add(prompt)
        elif event_type == "assistant.message":
            reasoning = _reasoning_text(data)
            content = (data.get("content") or "").strip()
            requests = data.get("toolRequests") or []
            calls = [f"{r.get('name')} {json.dumps(r.get('arguments'), ensure_ascii=False)}" for r in requests]
            text = "\n\n".join(filter(None, [reasoning, content, *(f"→ {c}" for c in calls)]))
            if not text:
                return False
            gist = _one_line(content or reasoning, 110)
            names = ", ".join(r.get("name") or "?" for r in requests)
            self._add(HistoryItem(event["id"], "response", time, text, gist + (f" → {names}" if names else "")))
        elif event_type == "tool.execution_start":
            name = data.get("toolName") or "?"
            arguments = json.dumps(data.get("arguments"), ensure_ascii=False)
            summary = f"{name} {_one_line(arguments, 70)}"
            item = HistoryItem(event["id"], "tool", time, f"{name} {arguments}", summary, tool=name)
            self._add(item)
            self._by_call[data.get("toolCallId")] = item
        else:  # tool.execution_complete
            item = self._by_call.pop(data.get("toolCallId"), None)
            if item is None:
                return False
            result = data.get("result") or {}
            output = result.get("content") if isinstance(result, dict) else str(result)
            item.ok = bool(data.get("success")) and not (output or "").lower().startswith(TOOL_FAILURE_PREFIXES)
            item.text += f"\n\n{output or ''}"
            item.summary += f" {'✓' if item.ok else '✗'} {_one_line(output or '', 70)}"
            for index, binary in enumerate(result.get("binaryResultsForLlm") or [] if isinstance(result, dict) else []):
                if binary.get("assetId"):
                    item.images.append(("asset", binary["assetId"]))
                else:
                    item.images.append(("inline", start, index))
        return True

    def _add_summary(self, data: dict, time: float):
        text = (data.get("summaryContent") or "").strip()
        if not data.get("success") or not text:
            return
        title = re.search(r"<checkpoint_title>\s*(.*?)\s*</checkpoint_title>", text, re.DOTALL)
        number = data.get("checkpointNumber") or len(self.summaries) + 1
        title_text = _one_line(title[1] if title else text, 80)
        self.summaries.append(Summary(int(number), self._turn_at(time), title_text, text))

    def _add(self, item: HistoryItem):
        item.turn = self._turn_at(item.time)
        self.items.append(item)

    def _shortest_ids(self) -> dict[str, str]:
        ids = [item.event_id.replace("-", "") for item in self.items]
        length = MIN_ID_CHARS
        while len({i[-length:] for i in ids}) < len(ids) and length < 32:
            length += 1
        return {item.event_id: plain[-length:] for item, plain in zip(self.items, ids)}

    def short_id(self, item: HistoryItem) -> str:
        return self._short_ids.get(item.event_id, item.event_id[-MIN_ID_CHARS:])

    def find(self, suffix: str) -> HistoryItem | None:
        suffix = suffix.lstrip("#").replace("-", "").lower()
        matches = [item for item in self.items if item.event_id.replace("-", "").lower().endswith(suffix)]
        return matches[0] if len(matches) == 1 else None

    def image_b64(self, item: HistoryItem, index: int) -> str | None:
        reference = item.images[index]
        with open(self.events_path, "rb") as f:
            if reference[0] == "asset":
                location = self._assets.get(reference[1])
                if not location:
                    return None
                f.seek(location[0])
                return json.loads(f.read(location[1]))["data"]["data"]
            f.seek(reference[1])
            binary = json.loads(f.readline())["data"]["result"]["binaryResultsForLlm"][reference[2]]
            data = binary.get("data")
            return data if isinstance(data, str) else base64.standard_b64encode(data).decode()

    # ----- formatting ---------------------------------------------------------------------------------------------

    def line(self, item: HistoryItem) -> str:
        turn = f"turn {item.turn}" if item.turn is not None else "turn ?"
        return f"#{self.short_id(item)}  {turn:<10} {item.kind:<8} {item.summary}"
