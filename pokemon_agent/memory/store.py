"""File-based memory exposed through tools shaped like GitHub Copilot's built-in file tools.

The model sees a virtual `/memories` directory and uses `view`, `create`, and `edit` (same names and parameters as the
Copilot built-ins) plus `delete` and `rename`. Files live under MEMORY_DIR on disk; every path is resolved and checked
so the model cannot reach anything outside it.
"""

import os
import shutil
from pathlib import Path

MEMORY_DIR = "memories"
VIRTUAL_ROOT = "/memories"
MAX_FILE_CHARS = 20_000
MAX_VIEW_CHARS = 20_000

_PATHS_NOTE = f"* Paths must start with {VIRTUAL_ROOT}"

MEMORY_TOOL_SPECS: dict[str, dict] = {
    "view": {
        "description": (
            "View files or directories in your memory directory.\n"
            f"* Your notes in {VIRTUAL_ROOT} persist while your conversation history is periodically condensed.\n"
            "* Text files return their content.\n"
            "* Directories list non-hidden entries up to 2 levels deep.\n"
            f"{_PATHS_NOTE}\n"
            "* Files over 20KB are truncated; use `view_range` for sections."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": f"Existing file or directory's path under {VIRTUAL_ROOT}."},
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional 1-based inclusive line range. [start,-1] reads through EOF. "
                    "Prefer for files over 20KB, which are otherwise truncated.",
                },
                "forceReadLargeFiles": {
                    "type": "boolean",
                    "description": "Read an entire large file despite the size limit; default false.",
                },
            },
            "required": ["path"],
        },
    },
    "create": {
        "description": (
            "Tool for creating new memory files.\n"
            "* Creates a new file with the specified content at the given path\n"
            "* Cannot be used if the specified path already exists; use `edit` to change an existing file\n"
            "* Missing parent directories are created automatically\n"
            f"{_PATHS_NOTE}\n"
            f"* Files are limited to {MAX_FILE_CHARS} characters; keep notes concise and organized"
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
            "Tool for making string replacements in memory files. Use it to keep notes current: update or remove "
            "outdated information instead of adding contradictions.\n"
            "* Replaces exactly one occurrence of `old_str` with `new_str` in the specified file\n"
            "* When called multiple times in a single response, edits are independently made in the order calls are "
            "specified\n"
            "* The `old_str` parameter must match EXACTLY one or more consecutive lines from the original file\n"
            "* If `old_str` is not unique in the file, replacement will not be performed\n"
            "* Make sure to include enough context in `old_str` to make it unique\n"
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
                "new_str": {"type": "string", "description": "The new string to replace old_str with."},
            },
            "required": ["path"],
        },
    },
    "delete": {
        "description": (
            "Delete a memory file or directory that is no longer useful.\n"
            f"{_PATHS_NOTE}\n"
            f"* {VIRTUAL_ROOT} itself cannot be deleted"
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
            "* The destination must not already exist; missing parent directories are created automatically"
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


class MemoryStore:
    """Executes memory tool calls against a directory on disk."""

    def __init__(self, root: str = MEMORY_DIR):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def execute(self, tool_name: str, tool_input: dict) -> str:
        if tool_name not in MEMORY_TOOL_SPECS:
            return f"Error: unknown memory tool {tool_name!r}"
        try:
            return getattr(self, tool_name)(tool_input)
        except MemoryCommandError as e:
            return str(e)
        except OSError as e:
            # e.g. creating a file under a path that is itself a file
            return f"Error: {e.strerror or e}"

    def _resolve(self, path: str | None) -> Path:
        if not path:
            raise MemoryCommandError("Error: path is required")
        if path != VIRTUAL_ROOT and not path.startswith(VIRTUAL_ROOT + "/"):
            raise MemoryCommandError(f"Error: path must start with {VIRTUAL_ROOT}: {path}")
        real = (self.root / path[len(VIRTUAL_ROOT) :].lstrip("/")).resolve()
        if real != self.root and self.root not in real.parents:
            raise MemoryCommandError(f"Error: path escapes {VIRTUAL_ROOT}: {path}")
        return real

    def _virtual(self, real: Path) -> str:
        relative = real.relative_to(self.root).as_posix()
        return VIRTUAL_ROOT if relative == "." else f"{VIRTUAL_ROOT}/{relative}"

    def _make_parents(self, real: Path):
        for parent in reversed(real.parents):
            if (parent == self.root or self.root in parent.parents) and parent.is_file():
                raise MemoryCommandError(f"Error: {self._virtual(parent)} is a file, not a directory")
        real.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _check_size(text: str):
        if len(text) > MAX_FILE_CHARS:
            raise MemoryCommandError(f"Error: file would be {len(text)} characters; the limit is {MAX_FILE_CHARS}")

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

    def view(self, tool_input: dict) -> str:
        real = self._resolve(tool_input.get("path"))
        if real.is_dir():
            entries = [
                f"{child.relative_to(real).as_posix()}{'/' if child.is_dir() else ''}"
                for child in sorted(real.rglob("*"))
                if len(child.relative_to(real).parts) <= 2
                and not any(part.startswith(".") for part in child.relative_to(real).parts)
                and not child.is_symlink()
            ]
            return "\n".join(entries) or "(empty directory)"
        if not real.is_file():
            raise MemoryCommandError(f"Error: {tool_input.get('path')} does not exist")
        text = real.read_text()
        view_range = tool_input.get("view_range")
        content = self._slice_lines(text, view_range) if view_range else text
        if len(content) > MAX_VIEW_CHARS and not tool_input.get("forceReadLargeFiles"):
            content = content[:MAX_VIEW_CHARS] + "\n... (truncated; use view_range to see more)"
        return content

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
        if real.exists():
            raise MemoryCommandError(f"Path already exists: {tool_input.get('path')}; use edit to change it")
        text = tool_input.get("file_text", "")
        self._check_size(text)
        self._make_parents(real)
        real.write_text(text)
        return f"Created file {self._virtual(real)} with {len(text)} characters"

    def edit(self, tool_input: dict) -> str:
        real = self._resolve(tool_input.get("path"))
        if not real.is_file():
            raise MemoryCommandError(f"Error: {tool_input.get('path')} is not an existing file")
        text = real.read_text()
        old = tool_input.get("old_str") or ""
        count = text.count(old) if old else 0
        if count != 1:
            problem = "Multiple matches found" if count > 1 else "No match found"
            raise MemoryCommandError(f"{problem} for old_str in {self._virtual(real)}; no changes made")
        updated = text.replace(old, tool_input.get("new_str", ""), 1)
        self._check_size(updated)
        real.write_text(updated)
        return f"File {self._virtual(real)} updated with changes."

    def delete(self, tool_input: dict) -> str:
        real = self._resolve(tool_input.get("path"))
        if real == self.root:
            raise MemoryCommandError(f"Error: cannot delete {VIRTUAL_ROOT} itself")
        if not real.exists():
            raise MemoryCommandError(f"Error: {tool_input.get('path')} does not exist")
        shutil.rmtree(real) if real.is_dir() else real.unlink()
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
