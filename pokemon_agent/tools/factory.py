import logging

from pokemon_agent.tools.base import Tool, ToolError, ToolOutput
from pokemon_agent.tools.compaction_tool import TriggerCompactionTool
from pokemon_agent.tools.game_tools import (
    OnScreenKeyboardTool,
    PressButtonsTool,
)
from pokemon_agent.tools.history_tool import HistoryTool
from pokemon_agent.tools.objective_tool import ObjectiveTool
from pokemon_agent.tools.subagent_tool import SubagentTool
from pokemon_agent.memory.store import MemoryStore
from pokemon_agent.tools.memory_tools import (
    MemoryCreateTool,
    MemoryFileTool,
    MemoryDeleteTool,
    MemoryEditTool,
    MemoryInsertTool,
    MemoryListTool,
    MemoryRenameTool,
    MemorySaveScreenshotTool,
    MemoryViewTool,
)

logger = logging.getLogger(__name__)

TOOL_CLASSES: dict[str, type[Tool]] = {
    tool_class.name: tool_class
    for tool_class in [
        PressButtonsTool,
        OnScreenKeyboardTool,
        HistoryTool,
        SubagentTool,
        ObjectiveTool,
        TriggerCompactionTool,
        MemoryViewTool,
        MemoryListTool,
        MemorySaveScreenshotTool,
        MemoryCreateTool,
        MemoryEditTool,
        MemoryInsertTool,
        MemoryDeleteTool,
        MemoryRenameTool,
    ]
}

DEFAULT_TOOL_NAMES = [
    "press_buttons",
    "on_screen_keyboard_input",
    "view",
    "list",
    "save_screenshot",
    "create",
    "edit",
    "insert",
    "delete",
    "rename",
]

# A helper (see SubagentTool) only reads: memory files and the main agent's history. It can't play or write.
HELPER_TOOL_NAMES = ["view", "list", "history"]


class ToolFactory:
    """Factory for creating tool instances."""

    def __init__(self, emulator, tool_names: list[str] | None = None, memory: MemoryStore | None = None):
        """Initialize the selected tools (DEFAULT_TOOL_NAMES if None); memory tools use the given memory store."""
        self.emulator = emulator
        tool_names = DEFAULT_TOOL_NAMES if tool_names is None else tool_names
        unknown = set(tool_names) - TOOL_CLASSES.keys()
        if unknown:
            raise ValueError(f"Unknown tools: {sorted(unknown)}; available: {sorted(TOOL_CLASSES)}")
        self.tools: dict[str, Tool] = {
            name: TOOL_CLASSES[name](emulator, memory)
            if issubclass(TOOL_CLASSES[name], MemoryFileTool)
            else TOOL_CLASSES[name](emulator)
            for name in tool_names
        }

        logger.info("Available tools:")
        for tool in self.tools.values():
            logger.info(f"- {tool.name}: {tool.description}")

    def execute_tool(self, tool_name: str, tool_input: dict) -> str | ToolOutput:
        logger.info(f"Processing tool call: {tool_name}")
        tool = self.tools.get(tool_name)
        if tool is None:
            return ToolError(f"Error: unknown tool {tool_name!r}; available tools: {', '.join(self.tools)}")
        return tool.execute(tool_input)

    def get_available_tools(self):
        return [tool.get_tool_definition() for tool in self.tools.values()]
