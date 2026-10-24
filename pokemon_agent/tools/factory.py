import logging

from pokemon_agent.tools.base import Tool
from pokemon_agent.tools.game_tools import NavigateToScreenCoordinatesTool, OnScreenKeyboardTool, PressButtonsTool
from pokemon_agent.tools.memory_tools import (
    MemoryCreateTool,
    MemoryDeleteTool,
    MemoryEditTool,
    MemoryRenameTool,
    MemoryViewTool,
)

logger = logging.getLogger(__name__)

TOOL_CLASSES: dict[str, type[Tool]] = {
    tool_class.name: tool_class
    for tool_class in [
        PressButtonsTool,
        NavigateToScreenCoordinatesTool,
        OnScreenKeyboardTool,
        MemoryViewTool,
        MemoryCreateTool,
        MemoryEditTool,
        MemoryDeleteTool,
        MemoryRenameTool,
    ]
}

DEFAULT_TOOL_NAMES = [
    "press_buttons",
    "navigate_to_screen_coordinates",
    "on_screen_keyboard_input",
    "view",
    "create",
    "edit",
    "delete",
    "rename",
]


class ToolFactory:
    """Factory for creating tool instances."""

    def __init__(self, emulator, tool_names: list[str] | None = None):
        """Initialize the selected tools (DEFAULT_TOOL_NAMES if None) with an emulator instance."""
        self.emulator = emulator
        tool_names = DEFAULT_TOOL_NAMES if tool_names is None else tool_names
        unknown = set(tool_names) - TOOL_CLASSES.keys()
        if unknown:
            raise ValueError(f"Unknown tools: {sorted(unknown)}; available: {sorted(TOOL_CLASSES)}")
        self.tools: dict[str, Tool] = {name: TOOL_CLASSES[name](emulator) for name in tool_names}

        logger.info("Available tools:")
        for tool in self.tools.values():
            logger.info(f"- {tool.name}: {tool.description}")

    def execute_tool(self, tool_name: str, tool_input: dict) -> str:
        logger.info(f"Processing tool call: {tool_name}")
        return self.tools[tool_name].execute(tool_input)

    def get_available_tools(self):
        return [tool.get_tool_definition() for tool in self.tools.values()]
