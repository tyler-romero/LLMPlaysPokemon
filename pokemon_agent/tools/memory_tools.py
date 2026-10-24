"""Tools over the /memories directory (see pokemon_agent/memory/store.py)."""

import logging

from pokemon_agent.game.emulator import Emulator
from pokemon_agent.memory.store import MEMORY_TOOL_SPECS, MemoryStore
from pokemon_agent.tools.base import Tool

logger = logging.getLogger(__name__)


class MemoryFileTool(Tool):
    """Base for the file tools over /memories; subclasses set `name`."""

    def __init__(self, emulator: Emulator, memory: MemoryStore | None = None):
        super().__init__(emulator)
        self.store = memory or MemoryStore()
        self.description = MEMORY_TOOL_SPECS[self.name]["description"]
        self.parameters = MEMORY_TOOL_SPECS[self.name]["parameters"]

    def execute(self, tool_input: dict) -> str:
        result = self.store.execute(self.name, tool_input)
        target = tool_input.get("path") or tool_input.get("old_path")
        logger.info(f"[Memory] {self.name} {target}: {(result.splitlines() or [''])[0][:120]}")
        return result


class MemoryViewTool(MemoryFileTool):
    name = "view"


class MemoryCreateTool(MemoryFileTool):
    name = "create"


class MemoryEditTool(MemoryFileTool):
    name = "edit"


class MemoryDeleteTool(MemoryFileTool):
    name = "delete"


class MemoryRenameTool(MemoryFileTool):
    name = "rename"
