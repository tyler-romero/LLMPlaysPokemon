"""Tools over the /memories directory (see pokemon_agent/memory/store.py)."""

import base64
import logging

from pokemon_agent.game.emulator import Emulator
from pokemon_agent.game.observation import at_model_scale
from pokemon_agent.memory.store import MEMORY_TOOL_SPECS, MemoryImage, MemoryStore
from pokemon_agent.tools.base import Tool, ToolOutput

logger = logging.getLogger(__name__)


class MemoryFileTool(Tool):
    """Base for the file tools over /memories; subclasses set `name`."""

    def __init__(self, emulator: Emulator, memory: MemoryStore | None = None):
        super().__init__(emulator)
        self.store = memory or MemoryStore()
        self.description = MEMORY_TOOL_SPECS[self.name]["description"]
        self.parameters = MEMORY_TOOL_SPECS[self.name]["parameters"]

    def execute(self, tool_input: dict) -> str | ToolOutput:
        result = self.store.execute(self.name, tool_input)
        if isinstance(result, MemoryImage):
            png = at_model_scale(result.png)
            result = ToolOutput(text=result.caption, images=[base64.standard_b64encode(png).decode()])
        text = result.text if isinstance(result, ToolOutput) else result
        target = tool_input.get("path") or tool_input.get("old_path")
        logger.info(f"[Memory] {self.name} {target}: {(text.splitlines() or [''])[0][:120]}")
        return result


class MemoryViewTool(MemoryFileTool):
    name = "view"


class MemoryListTool(MemoryFileTool):
    name = "list"


class MemorySaveScreenshotTool(MemoryFileTool):
    name = "save_screenshot"


class MemoryCreateTool(MemoryFileTool):
    name = "create"


class MemoryEditTool(MemoryFileTool):
    name = "edit"


class MemoryInsertTool(MemoryFileTool):
    name = "insert"


class MemoryDeleteTool(MemoryFileTool):
    name = "delete"


class MemoryRenameTool(MemoryFileTool):
    name = "rename"
