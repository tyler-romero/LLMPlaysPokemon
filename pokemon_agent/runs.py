"""Durable runs: a run folder holding atomic checkpoints of the game, memory, conversation, and counters.

Layout:
    runs/<run-id>/
        run.json                  # agent, model, and settings the run was started with
        latest                    # name of the newest complete checkpoint
        events.jsonl              # observer event history
        memories/                 # live /memories directory
        checkpoints/<step>/       # game.state, memories/, agent/, checkpoint.json
"""

import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

CHECKPOINTS = "checkpoints"
LATEST = "latest"


def _write_json(path: Path, data: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, path)


class RunStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.checkpoints = self.root / CHECKPOINTS

    @property
    def run_id(self) -> str:
        return self.root.name

    @property
    def memory_dir(self) -> Path:
        return self.root / "memories"

    @property
    def events_path(self) -> Path:
        return self.root / "events.jsonl"

    @classmethod
    def create(cls, runs_dir: str, meta: dict) -> "RunStore":
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", f"{meta.get('agent', 'run')}-{meta.get('model') or 'default'}").strip("-")
        root = Path(runs_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug[:48]}"
        root.mkdir(parents=True)
        (root / CHECKPOINTS).mkdir()
        store = cls(root)
        _write_json(root / "run.json", {**meta, "created_at": time.time()})
        logger.info(f"Run folder: {root}")
        return store

    @classmethod
    def open(cls, path: str) -> "RunStore":
        root = Path(path)
        if not (root / "run.json").is_file():
            raise FileNotFoundError(f"{path} is not a run folder (no run.json)")
        return cls(root)

    @property
    def meta(self) -> dict:
        return json.loads((self.root / "run.json").read_text())

    def update_meta(self, **changes):
        _write_json(self.root / "run.json", {**self.meta, **changes})

    def latest_checkpoint(self) -> Path | None:
        """The checkpoint `latest` names, or, if it's missing or incomplete (a crash while saving), the newest complete
        one."""
        pointer = self.root / LATEST
        if pointer.is_file():
            checkpoint = self.checkpoints / pointer.read_text().strip()
            if (checkpoint / "checkpoint.json").is_file():
                return checkpoint
        complete = [path for path in self._complete() if (path / "checkpoint.json").is_file()]
        if complete:
            logger.warning(f"{pointer} doesn't name a complete checkpoint; using {complete[-1].name}")
        return complete[-1] if complete else None

    def _complete(self) -> list[Path]:
        """Checkpoint directories in step order (hidden temp and replaced ones excluded)."""
        if not self.checkpoints.is_dir():
            return []
        return sorted(p for p in self.checkpoints.iterdir() if p.is_dir() and not p.name.startswith("."))

    def checkpoint(self, step: int, write: Callable[[Path], dict], keep_last: int = 5, keep_every: int = 100) -> Path:
        """Write a checkpoint atomically: `write(dir)` fills a temp dir and returns extra metadata; the directory is
        renamed into place and `latest` updated only after everything is on disk."""
        name = f"{step:06d}"
        tmp = self.checkpoints / f".tmp-{name}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        info = write(tmp) or {}
        _write_json(tmp / "checkpoint.json", {"step": step, "saved_at": time.time(), **info})
        final = self.checkpoints / name
        # Re-saving a step (e.g. the final save after a periodic one) moves the old copy aside rather than deleting it
        # first, so a crash leaves at most a missing `latest` target, which latest_checkpoint() recovers from.
        replaced = self.checkpoints / f".replaced-{name}"
        if replaced.exists():
            shutil.rmtree(replaced)
        if final.exists():
            os.replace(final, replaced)
        os.replace(tmp, final)
        latest_tmp = self.root / f"{LATEST}.tmp"
        latest_tmp.write_text(name)
        os.replace(latest_tmp, self.root / LATEST)
        shutil.rmtree(replaced, ignore_errors=True)
        self._prune(keep_last, keep_every)
        logger.info(f"Checkpoint saved: {final}")
        return final

    def _prune(self, keep_last: int, keep_every: int):
        complete = self._complete()
        recent = set(complete[-keep_last:])
        for path in complete:
            if path not in recent and (keep_every <= 0 or int(path.name) % keep_every != 0):
                shutil.rmtree(path, ignore_errors=True)


def copy_tree(source: Path, destination: Path, ignore_patterns: tuple[str, ...] = ()):
    """Copy a directory, replacing destination; symlinks and ignored patterns are skipped."""
    if destination.exists():
        shutil.rmtree(destination)

    def ignore(directory, names):
        skipped = set(shutil.ignore_patterns(*ignore_patterns)(directory, names)) if ignore_patterns else set()
        return skipped | {n for n in names if os.path.islink(os.path.join(directory, n))}

    shutil.copytree(source, destination, ignore=ignore)
