"""File-based memory exposed through tools shaped like GitHub Copilot's built-in file tools.

The model sees a virtual `/memories` directory and uses `view`, `create`, and `edit` (same names and parameters as the
Copilot built-ins) plus `insert` (like the Copilot editor's insert command), `list`, `delete`, and `rename`. Files
live under MEMORY_DIR on disk; every path is resolved and checked so the model cannot reach anything outside it.
`save_screenshot` stores the turn's screenshot as a .png, and `view` returns a .png as an image. The turn each file was
last written (and where a screenshot was taken) is kept in a hidden META_FILE in the same directory, so checkpoints
carry it; hidden paths are off limits to the tools.
"""

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from pokemon_agent.tools.base import ToolError

MEMORY_DIR = "memories"
VIRTUAL_ROOT = "/memories"
MAX_FILE_BYTES = 400_000  # per text file, UTF-8 encoded
# One `view` shows at most this much; past it, the Copilot SDK would write the result to a temp file the model can't
# open (LargeToolOutputConfig: 50KB by default), so long files are read in parts with view_range.
MAX_VIEW_CHARS = 20_000
META_FILE = ".meta.json"
MAX_SCREENSHOTS = 30

_PATHS_NOTE = f"* Paths must start with {VIRTUAL_ROOT}"

MEMORY_TOOL_SPECS: dict[str, dict] = {
    "view": {
        "description": (
            "View the contents of a file in your memory directory.\n"
            f"* Your notes in {VIRTUAL_ROOT} persist while your conversation history is periodically condensed.\n"
            "* Returns a text file's text, or shows a saved screenshot (.png) as an image. To see which files exist, "
            "use `list`.\n"
            f"{_PATHS_NOTE}\n"
            f"* A long file is shown {MAX_VIEW_CHARS:,} characters at a time, cut at a line break; the result says "
            "which lines it showed and where to continue with `view_range`."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": f"Existing file's path under {VIRTUAL_ROOT}."},
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional 1-based inclusive line range. [start,-1] reads through EOF. "
                    "Use it to read a long file in parts.",
                },
            },
            "required": ["path"],
        },
    },
    "list": {
        "description": (
            "List the files and directories in your memory directory, like `ls`.\n"
            "* Shows every entry under the path (all levels): text files with their size in characters and the turn "
            "they were last updated, screenshots with the turn and place they were saved. Use it to spot missing or "
            "stale notes.\n"
            f"{_PATHS_NOTE}"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": f"Directory under {VIRTUAL_ROOT} to list; defaults to {VIRTUAL_ROOT}.",
                },
            },
        },
    },
    "save_screenshot": {
        "description": (
            "Save this turn's screenshot, exactly as you see it (grid and labels included), to a .png file in your "
            "memory directory. View it later with `view` to see it again.\n"
            "* The location and coordinates where it was taken are recorded with it, since grid labels belong to "
            "that map\n"
            "* Cannot be used if the path already exists; delete or rename the old file first\n"
            "* Missing parent directories are created automatically\n"
            f"{_PATHS_NOTE}, and must end in .png\n"
            f"* At most {MAX_SCREENSHOTS} screenshots can be kept; delete ones you no longer need\n"
            "* Before calling, consider `list` to see the screenshots you already have and choose an unused path"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": f"Path under {VIRTUAL_ROOT} ending in .png."},
            },
            "required": ["path"],
        },
    },
    "create": {
        "description": (
            "Tool for creating new memory files. Record what you learn as soon as you learn it; to add to an existing "
            "file, use `insert`.\n"
            "* Creates a new file with the specified content at the given path\n"
            "* Cannot be used if the specified path already exists; use `edit` to change an existing file\n"
            "* Missing parent directories are created automatically\n"
            f"{_PATHS_NOTE}\n"
            f"* Keep notes concise and organized: each file holds at most {MAX_FILE_BYTES:,} bytes, and `view` shows a "
            f"long file {MAX_VIEW_CHARS:,} characters at a time, so split long files by topic\n"
            "* Before calling, consider `list` (and `view`): the file may already exist, or another file may already "
            "cover this"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": f"Path under {VIRTUAL_ROOT} of the file to create. File MUST not exist before "
                    "creating.",
                },
                "file_text": {"type": "string", "description": "The content of the file to be created."},
            },
            "required": ["path", "file_text"],
        },
    },
    "edit": {
        "description": (
            "Tool for making string replacements in memory files. Use it constantly to keep your notes current: "
            "update or remove outdated information instead of adding contradictions. To add new notes without "
            "changing existing text, use `insert` instead.\n"
            "* Replaces exactly one occurrence of `old_str` with `new_str` in the specified file\n"
            "* When called multiple times in a single response, edits are independently made in the order calls are "
            "specified\n"
            "* The `old_str` parameter must match EXACTLY one or more consecutive lines from the original file\n"
            "* If `old_str` is not unique in the file, replacement will not be performed\n"
            "* Make sure to include enough context in `old_str` to make it unique\n"
            "* Before calling, `view` the file (or the lines you are changing) and copy `old_str` from what it says "
            "now. If the match fails, the error says where it broke down\n"
            f"{_PATHS_NOTE}"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": f"Path under {VIRTUAL_ROOT} of the file to edit. File MUST exist to edit.",
                },
                "old_str": {
                    "type": "string",
                    "description": "The string in the file to replace. Leading and ending whitespaces from file "
                    "content should be preserved!",
                },
                "new_str": {
                    "type": "string",
                    "description": "The new string to replace old_str with. Use an empty string to delete old_str.",
                },
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
    "insert": {
        "description": (
            "Insert text into an existing memory file without having to match its current text. Use it whenever "
            "you have something new to record: a discovery, a test and its result, a change of plan.\n"
            "* `insert_line` -1 (the default) adds `new_str` at the end of the file, 0 at the top, and N after line "
            "N (line 1 is the first line; use `view` with `view_range` to find line numbers)\n"
            "* `new_str` is inserted as whole lines\n"
            "* Before calling, consider `view` on the file, so the new lines don't repeat or contradict what is "
            "already there and land where you mean\n"
            f"{_PATHS_NOTE}"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": f"Path under {VIRTUAL_ROOT} of an existing file."},
                "new_str": {"type": "string", "description": "The text to insert, as one or more lines."},
                "insert_line": {
                    "type": "integer",
                    "description": "Insert after this line: -1 for the end of the file (default), 0 for the top.",
                },
            },
            "required": ["path", "new_str"],
        },
    },
    "delete": {
        "description": (
            "Delete a memory file or directory that is no longer useful.\n"
            f"{_PATHS_NOTE}\n"
            f"* {VIRTUAL_ROOT} itself cannot be deleted\n"
            "* Before calling, consider `view` (or `list` for a directory) to make sure it is the right path and "
            "nothing in it is still needed"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": f"Path under {VIRTUAL_ROOT} to delete."},
            },
            "required": ["path"],
        },
    },
    "rename": {
        "description": (
            "Move or rename a memory file or directory.\n"
            f"{_PATHS_NOTE}\n"
            "* The destination must not already exist; missing parent directories are created automatically\n"
            "* Before calling, consider `list` to check the current path and that the new one is free"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "old_path": {"type": "string", "description": f"Current path under {VIRTUAL_ROOT}."},
                "new_path": {"type": "string", "description": f"New path under {VIRTUAL_ROOT}."},
            },
            "required": ["old_path", "new_path"],
        },
    },
}


class MemoryCommandError(Exception):
    pass


def _find_all(text: str, needle: str):
    start = text.find(needle)
    while start != -1:
        yield start
        start = text.find(needle, start + 1)


def _clip(line: str, limit: int = 140) -> str:
    return repr(line if len(line) <= limit else line[: limit - 1] + "…")


def _mismatch(text: str, old: str) -> str:
    """Where an edit's old_str stops matching the file, so the model can fix it without guessing."""
    file_lines, old_lines = text.splitlines(), old.splitlines()
    if not any(line.strip() for line in old_lines):
        return "old_str has no text, only spaces or line breaks; copy a line from the file to anchor the edit."
    first = old_lines[0]
    # old_str may start partway through a file line and end partway through another.
    starts = [i for i, line in enumerate(file_lines) if first and line.endswith(first)] or [
        i for i, line in enumerate(file_lines) if first.strip() and first.strip() in line
    ]
    if not starts:
        found = next((j for j, line in enumerate(old_lines) if line.strip() and line.strip() in text), None)
        if found is None:
            return "None of old_str's lines appear in the file; view it and copy old_str from what it says now."
        return (
            f"Its first line isn't in the file, though its line {found + 1} is. View the file and copy old_str from "
            "what it says now."
        )
    best_k, best_i = -1, starts[0]
    for i in starts:
        k = 0
        while k < len(old_lines) and i + k < len(file_lines):
            expected, actual = old_lines[k], file_lines[i + k]
            if k == 0:
                same = actual.endswith(expected)
            elif k == len(old_lines) - 1:
                same = actual.startswith(expected)
            else:
                same = actual == expected
            if not same:
                break
            k += 1
        if k > best_k:
            best_k, best_i = k, i
    k, i = best_k, best_i
    if k == len(old_lines):
        # Every line matched on its own, so the difference is where old_str starts or ends within a line.
        ends = " It ends with a line break, but the file's line continues there." if old.endswith("\n") else ""
        return (
            f"old_str's lines each match lines {i + 1} through {i + k} of the file, but not as one exact piece: it "
            f"starts or ends partway through a line differently from the file.{ends} Line {i + k} of the file is: "
            f"{_clip(file_lines[i + k - 1])}"
        )
    if i + k >= len(file_lines):
        return f"old_str matches from line {i + 1} to the end of the file, but continues past it."
    actual, expected = file_lines[i + k], old_lines[k]
    note = " (they differ only in spacing)" if actual.strip() == expected.strip() else ""
    return (
        f"old_str matches the file from line {i + 1} through line {i + k}, but its line {k + 1} differs from line "
        f"{i + k + 1} of the file{note}:\n  file:    {_clip(actual)}\n  old_str: {_clip(expected)}"
    )


@dataclass
class MemoryImage:
    """A screenshot read from memory: a caption for the model and the PNG bytes."""

    caption: str
    png: bytes


def _is_screenshot(path: Path) -> bool:
    return path.suffix.lower() == ".png"


class MemoryStore:
    """Executes memory tool calls against a directory on disk."""

    def __init__(self, root: str = MEMORY_DIR):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        # Set by the agent each turn: the turn (step) number, which writes record so `list` can say when each file was
        # updated, and the turn's screenshot (PNG bytes, as the model saw it) and place, for save_screenshot.
        self.turn: int | None = None
        self.screenshot: bytes | None = None
        self.place: str | None = None

    def execute(self, tool_name: str, tool_input: dict, **options) -> "str | MemoryImage":
        """Run a memory tool. `options` are for the harness, not the model (view: full=True skips the size cap)."""
        if tool_name not in MEMORY_TOOL_SPECS:
            return ToolError(f"Error: unknown memory tool {tool_name!r}")
        try:
            return getattr(self, tool_name)(tool_input, **options)
        except MemoryCommandError as e:
            return ToolError(str(e))
        except OSError as e:
            # e.g. creating a file under a path that is itself a file
            return ToolError(f"Error: {e.strerror or e}")

    def _resolve(self, path: str | None) -> Path:
        if not path:
            raise MemoryCommandError("Error: path is required")
        if path != VIRTUAL_ROOT and not path.startswith(VIRTUAL_ROOT + "/"):
            raise MemoryCommandError(f"Error: path must start with {VIRTUAL_ROOT}: {path}")
        real = (self.root / path[len(VIRTUAL_ROOT) :].lstrip("/")).resolve()
        if real != self.root and self.root not in real.parents:
            raise MemoryCommandError(f"Error: path escapes {VIRTUAL_ROOT}: {path}")
        if any(part.startswith(".") for part in real.relative_to(self.root).parts):
            raise MemoryCommandError(f"Error: hidden paths are not allowed: {path}")
        return real

    def _virtual(self, real: Path) -> str:
        relative = real.relative_to(self.root).as_posix()
        return VIRTUAL_ROOT if relative == "." else f"{VIRTUAL_ROOT}/{relative}"

    def _make_parents(self, real: Path):
        for parent in reversed(real.parents):
            if (parent == self.root or self.root in parent.parents) and parent.is_file():
                raise MemoryCommandError(f"Error: {self._virtual(parent)} is a file, not a directory")
        real.parent.mkdir(parents=True, exist_ok=True)

    def _check_size(self, text: str, real: Path):
        size = len(text.encode())
        if size > MAX_FILE_BYTES:
            raise MemoryCommandError(
                f"Error: {self._virtual(real)} would be {size:,} bytes; each file is limited to {MAX_FILE_BYTES:,} "
                "bytes, so nothing was written. Make room first: delete or condense outdated sections with 'edit', "
                "or move some notes into another file."
            )

    def listing(self, real: Path | None = None, depth: int = 2) -> str:
        """Directory listing with sizes, skipping hidden files and symlinks."""
        real = real or self.root
        lines = []

        def walk(directory: Path, level: int):
            for child in sorted(directory.iterdir()):
                if child.name.startswith(".") or child.is_symlink():
                    continue
                if child.is_file():
                    files = [child]
                else:
                    files = [f for f in child.rglob("*") if f.is_file() and not f.is_symlink()]
                size = sum(f.stat().st_size for f in files)
                lines.append(f"{size:>7}\t{self._virtual(child)}{'/' if child.is_dir() else ''}")
                if child.is_dir() and level < depth:
                    walk(child, level + 1)

        walk(real, 1)
        return "\n".join(lines)

    def file_times(self) -> dict[str, float]:
        """Last-modified time (epoch seconds) of each memory file, by virtual path."""
        return {
            self._virtual(path): path.stat().st_mtime
            for path in sorted(self.root.rglob("*"))
            if path.is_file() and not path.is_symlink()
            and not any(part.startswith(".") for part in path.relative_to(self.root).parts)
        }

    # ----- per-file metadata: last-updated turn, and where a screenshot was taken -----------------------------------

    def _load_meta(self) -> dict[str, dict]:
        try:
            return json.loads((self.root / META_FILE).read_text())
        except (OSError, ValueError):
            return {}

    def note_turns(self) -> dict[str, int]:
        """The turn each text file (not screenshot) was last written, by path, for files that still exist."""
        turns = {}
        for path, entry in self._load_meta().items():
            real = self.root / path.removeprefix(VIRTUAL_ROOT + "/")
            if entry.get("turn") is not None and not _is_screenshot(real) and real.is_file():
                turns[path] = entry["turn"]
        return turns

    def _save_meta(self, meta: dict[str, dict]):
        (self.root / META_FILE).write_text(json.dumps(meta, indent=0, sort_keys=True))

    def _record_write(self, real: Path, **extra):
        meta = self._load_meta()
        entry = {**extra}
        if self.turn is not None:
            entry["turn"] = self.turn
        meta[self._virtual(real)] = entry
        self._save_meta(meta)

    def _move_meta(self, old: Path, new: Path | None):
        """Carry (or, when new is None, drop) the metadata recorded for old and everything under it."""
        prefix = self._virtual(old)
        meta = self._load_meta()
        moved = {}
        for path in list(meta):
            if path == prefix or path.startswith(prefix + "/"):
                entry = meta.pop(path)
                if new is not None:
                    moved[self._virtual(new) + path[len(prefix) :]] = entry
        meta.update(moved)
        self._save_meta(meta)

    def _ago(self, turn: int | None) -> str:
        if turn is None:
            return "unknown"
        if self.turn is None:
            return f"turn {turn}"
        ago = self.turn - turn
        if ago <= 0:
            return f"turn {turn} (this turn)"
        return f"turn {turn} ({ago} turn{'' if ago == 1 else 's'} ago)"

    def list(self, tool_input: dict) -> str:
        real = self._resolve(tool_input.get("path") or VIRTUAL_ROOT)
        if not real.is_dir():
            raise MemoryCommandError(f"Error: {tool_input.get('path')} is not an existing directory")
        meta = self._load_meta()
        lines = []
        for child in sorted(real.rglob("*")):
            relative = child.relative_to(self.root).parts
            if child.is_symlink() or any(part.startswith(".") for part in relative):
                continue
            path = self._virtual(child)
            entry = meta.get(path, {})
            if child.is_dir():
                lines.append(f"{path}/")
            elif _is_screenshot(child):
                place = f" at {entry['place']}" if entry.get("place") else ""
                lines.append(f"{path}  screenshot  saved {self._ago(entry.get('turn'))}{place}")
            else:
                chars = len(child.read_text(errors="replace"))
                lines.append(f"{path}  {chars} chars  updated {self._ago(entry.get('turn'))}")
        current = f"Current turn: {self.turn}\n" if self.turn is not None else ""
        return current + ("\n".join(lines) or f"(empty directory: {self._virtual(real)})")

    def view(self, tool_input: dict, full: bool = False) -> "str | MemoryImage":
        real = self._resolve(tool_input.get("path"))
        if real.is_dir():
            raise MemoryCommandError(f"Error: {self._virtual(real)} is a directory; use list to see its contents")
        if not real.is_file():
            raise MemoryCommandError(f"Error: {tool_input.get('path')} does not exist")
        if _is_screenshot(real):
            entry = self._load_meta().get(self._virtual(real), {})
            place = f" at {entry['place']}" if entry.get("place") else ""
            caption = f"Screenshot {self._virtual(real)}, saved {self._ago(entry.get('turn'))}{place}."
            return MemoryImage(caption=caption, png=real.read_bytes())
        text = real.read_text()
        view_range = tool_input.get("view_range")
        content = self._slice_lines(text, view_range) if view_range else text
        if len(content) <= MAX_VIEW_CHARS or full:
            return content
        return self._cut(content, first_line=view_range[0] if view_range else 1, file_text=text)

    @staticmethod
    def _cut(content: str, first_line: int, file_text: str) -> str:
        """The first MAX_VIEW_CHARS of `content`, cut at a line break, saying which lines it shows and where to go on."""
        shown, size = [], 0
        for line in content.splitlines(keepends=True):
            if size + len(line) > MAX_VIEW_CHARS:
                break
            shown.append(line)
            size += len(line)
        total_lines = len(file_text.splitlines())
        if not shown:  # one line longer than the cap
            return (
                content[:MAX_VIEW_CHARS]
                + f"\n... (cut off: line {first_line} alone is longer than {MAX_VIEW_CHARS:,} characters, so only its "
                f"start is shown. The file has {total_lines} line{'s' if total_lines != 1 else ''} and "
                f"{len(file_text):,} characters.)"
            )
        last_line = first_line + len(shown) - 1
        return (
            "".join(shown).removesuffix("\n")
            + f"\n... (cut off here: this shows lines {first_line}-{last_line} of {total_lines}, {size:,} of the "
            f"file's {len(file_text):,} characters. To keep reading, use view_range starting at line "
            f"{last_line + 1}.)"
        )

    @staticmethod
    def _slice_lines(text: str, view_range: list) -> str:
        """Apply a 1-based inclusive line range the way Copilot's built-in `view` does."""
        lines = text.splitlines(keepends=True)
        if len(view_range) < 2 or not all(isinstance(n, int) for n in view_range[:2]):
            raise MemoryCommandError("Invalid view_range")
        start, end = view_range[0], view_range[1]
        if start < 1 or end == 0 or end < -1 or (end != -1 and end < start):
            raise MemoryCommandError("Invalid view_range")
        if start > len(lines):
            raise MemoryCommandError("view_range out of bounds")
        if end == -1 or end > len(lines):
            return "".join(lines[start - 1 :])
        return "".join(lines[start - 1 : end]).removesuffix("\n")

    def create(self, tool_input: dict) -> str:
        real = self._resolve(tool_input.get("path"))
        if _is_screenshot(real):
            raise MemoryCommandError("Error: .png files are screenshots; use save_screenshot to save one")
        if real.exists():
            raise MemoryCommandError(f"Path already exists: {tool_input.get('path')}; use edit to change it")
        text = tool_input.get("file_text", "")
        self._check_size(text, real)
        self._make_parents(real)
        real.write_text(text)
        self._record_write(real)
        return f"Created file {self._virtual(real)} with {len(text)} characters"

    def edit(self, tool_input: dict) -> str:
        real = self._resolve(tool_input.get("path"))
        if _is_screenshot(real):
            raise MemoryCommandError("Error: screenshots can't be edited; delete it and save a new one instead")
        if not real.is_file():
            raise MemoryCommandError(f"Error: {tool_input.get('path')} is not an existing file")
        if "new_str" not in tool_input:
            raise MemoryCommandError(
                "Error: edit needs new_str (use an empty string to delete old_str); no changes made"
            )
        text = real.read_text()
        old = tool_input.get("old_str") or ""
        if not old:
            raise MemoryCommandError("Error: old_str is empty; to add text without replacing any, use insert")
        count = text.count(old)
        if count > 1:
            starts = [text.count("\n", 0, i) + 1 for i in _find_all(text, old)]
            raise MemoryCommandError(
                f"Multiple matches found for old_str in {self._virtual(real)} (starting on lines "
                f"{', '.join(map(str, starts[:10]))}); no changes made. Include more surrounding text to make it "
                "unique."
            )
        if count == 0:
            raise MemoryCommandError(
                f"No match found for old_str in {self._virtual(real)}; no changes made. {_mismatch(text, old)}"
                f"{self._changed_this_turn(real)}"
            )
        updated = text.replace(old, tool_input["new_str"], 1)
        self._check_size(updated, real)
        real.write_text(updated)
        self._record_write(real)
        return f"File {self._virtual(real)} updated with changes."

    def insert(self, tool_input: dict) -> str:
        real = self._resolve(tool_input.get("path"))
        if _is_screenshot(real):
            raise MemoryCommandError("Error: screenshots can't be edited; delete it and save a new one instead")
        if not real.is_file():
            raise MemoryCommandError(f"Error: {tool_input.get('path')} is not an existing file; use create to make it")
        new = tool_input.get("new_str")
        if not isinstance(new, str) or not new.strip():
            raise MemoryCommandError("Error: insert needs new_str, the text to insert; no changes made")
        lines = real.read_text().splitlines(keepends=True)
        try:
            at = int(tool_input.get("insert_line", -1))
        except (TypeError, ValueError):
            at = None
        if at == -1:
            at = len(lines)
        if at is None or not 0 <= at <= len(lines):
            raise MemoryCommandError(
                f"Error: insert_line must be from 0 to {len(lines)} ({self._virtual(real)} has {len(lines)} lines), "
                "or -1 for the end; no changes made"
            )
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.insert(at, new if new.endswith("\n") else new + "\n")
        updated = "".join(lines)
        self._check_size(updated, real)
        real.write_text(updated)
        self._record_write(real)
        if at == 0:
            where = "at the top"
        elif at == len(lines) - 1:
            where = "at the end"
        else:
            where = f"after line {at}"
        return f"Inserted {len(new.splitlines())} line(s) into {self._virtual(real)} {where}."

    def _changed_this_turn(self, real: Path) -> str:
        entry = self._load_meta().get(self._virtual(real), {})
        if self.turn is not None and entry.get("turn") == self.turn:
            return (
                " This file was already changed this turn (by another edit or insert, perhaps in the same response), "
                "so it may no longer read the way you last saw it; view it again before retrying."
            )
        return ""

    def save_screenshot(self, tool_input: dict) -> str:
        real = self._resolve(tool_input.get("path"))
        if not _is_screenshot(real):
            raise MemoryCommandError("Error: screenshot paths must end in .png")
        if real.exists():
            raise MemoryCommandError(f"Path already exists: {tool_input.get('path')}; delete or rename it first")
        if self.screenshot is None:
            raise MemoryCommandError("Error: there is no screenshot to save yet")
        count = sum(1 for path in self.root.rglob("*.png") if not path.is_symlink())
        if count >= MAX_SCREENSHOTS:
            raise MemoryCommandError(
                f"Error: memory already holds {count} screenshots (the limit is {MAX_SCREENSHOTS}); delete some first"
            )
        self._make_parents(real)
        real.write_bytes(self.screenshot)
        self._record_write(real, place=self.place)
        place = f" at {self.place}" if self.place else ""
        return f"Saved this turn's screenshot{place} to {self._virtual(real)}"

    def delete(self, tool_input: dict) -> str:
        real = self._resolve(tool_input.get("path"))
        if real == self.root:
            raise MemoryCommandError(f"Error: cannot delete {VIRTUAL_ROOT} itself")
        if not real.exists():
            raise MemoryCommandError(f"Error: {tool_input.get('path')} does not exist")
        shutil.rmtree(real) if real.is_dir() else real.unlink()
        self._move_meta(real, None)
        return f"Deleted {self._virtual(real)}"

    def rename(self, tool_input: dict) -> str:
        old, new = self._resolve(tool_input.get("old_path")), self._resolve(tool_input.get("new_path"))
        if old == self.root or not old.exists():
            raise MemoryCommandError(f"Error: {tool_input.get('old_path')} does not exist")
        if new.exists():
            raise MemoryCommandError(f"Error: {tool_input.get('new_path')} already exists")
        if new == old or old in new.parents:
            raise MemoryCommandError("Error: cannot move a directory into itself")
        self._make_parents(new)
        old.rename(new)
        self._move_meta(old, new)
        return f"Renamed {self._virtual(old)} to {self._virtual(new)}"


def _skip_symlinks(directory: str, names: list[str]) -> list[str]:
    return [name for name in names if os.path.islink(os.path.join(directory, name))]


def snapshot_memory(destination: str, root: str = MEMORY_DIR):
    """Copy the memory directory to destination, replacing it. Symlinks are skipped, never followed."""
    if os.path.exists(destination):
        shutil.rmtree(destination)
    shutil.copytree(root, destination, ignore=_skip_symlinks)


def reset_memory(restore_from: str | None = None, root: str = MEMORY_DIR):
    """Start a run's memory: restore a snapshot if given, otherwise begin empty (keeping the previous run's as .old).

    A snapshot is copied to a staging directory first, so a failed copy leaves the current memory untouched.
    """
    staging = f"{root}.restoring"
    if os.path.exists(staging):
        shutil.rmtree(staging)
    if restore_from:
        shutil.copytree(restore_from, staging, ignore=_skip_symlinks)
    else:
        os.makedirs(staging)
    old = f"{root}.old"
    if os.path.exists(root):
        if os.path.exists(old):
            shutil.rmtree(old)
        os.rename(root, old)
    os.rename(staging, root)
