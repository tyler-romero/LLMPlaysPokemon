import json
import logging

from openai.types.chat import ChatCompletionMessageToolCall, ChatCompletionToolMessageParam

from agent.emulator import Emulator
from agent.memory import MEMORY_TOOL_SPECS, MemoryStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class Tool:
    """Base class for all tools."""

    name: str = "base_tool"
    description: str = "Base tool description"
    parameters: dict = {}
    # True for tools that act on the game. Agents that let the model chain tool calls on their own (CopilotAgent) end
    # the model's turn after these, so it sees a fresh screenshot before acting again.
    ends_turn: bool = False

    def __init__(self, emulator: Emulator):
        """Initialize the tool with an emulator instance."""
        self.emulator = emulator

    def execute(self, tool_input: dict) -> str:
        """Run the tool with parsed arguments and return the result text for the model."""
        raise NotImplementedError

    def get_tool_definition(self):
        """Generate the tool definition for the OpenAI API."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class PressButtonTool(Tool):
    """Tool for pressing a button on the Game Boy."""

    name = "press_button"
    description = "Press a button on the Game Boy. In Pokemon Red, the arrow keys are used to move the player or cursor. The 'a' button is used to confirm decision, talk or read, advance dialogue, etc. The 'b' button is used to cancel decisions, close menus, etc. The 'start' button is used to open the overworld menu."
    ends_turn = True
    parameters = {
        "type": "object",
        "properties": {
            "button": {
                "type": "string",
                "enum": [
                    "a",
                    "b",
                    "up",
                    "down",
                    "left",
                    "right",
                    "start",
                    "select",
                ],
                "description": "Button to press: 'a', 'b', 'up', 'down', 'left', 'right', 'start', or 'select'.",
            },
            "wait": {
                "type": "boolean",
                "description": "Whether to let the game run for about 2 seconds after the press so walking, menu, and text animations finish before you see the next screenshot. Defaults to true. Set to false only for rapid presses whose intermediate result you don't need to see (the next screenshot may then show the game mid-animation).",
            },
        },
        "required": ["button"],
        "additionalProperties": False,
    }

    def execute(self, tool_input: dict) -> str:
        """Process a press_button tool call."""
        button = tool_input["button"]
        wait = tool_input.get("wait", True)

        logger.info(f"[Button] Pressing: {button} (wait={wait})")
        self.emulator.press_buttons([button], wait)

        return f"Pressed button: {button}"


class NavigateToScreenCoordinatesTool(Tool):
    """Tool for navigating to a position on the map grid."""

    name = "navigate_to_screen_coordinates"
    description = "Automatically navigate to the specified map coordinates on the current screen. The screen is divided into a grid and the available coordinates are rendered in the provided screenshot. This tool is only available in the overworld."
    ends_turn = True
    parameters = {
        "type": "object",
        "properties": {
            "col": {"type": "integer", "description": "The column coordinate to navigate to."},
            "row": {"type": "integer", "description": "The row coordinate to navigate to."},
        },
        "required": ["col", "row"],
        "additionalProperties": False,
    }

    def execute(self, tool_input: dict) -> str:
        """Process a navigate_to tool call."""
        row = tool_input["row"]
        col = tool_input["col"]

        player_x, player_y = self.emulator.get_coordinates()

        # Convert absolute coordinates to screen-relative coordinates
        # In the screen grid, player is always at (4,4)
        # We need to convert the target absolute coordinates to screen-relative
        target_screen_row = row - player_y + 4
        target_screen_col = col - player_x + 4

        logger.info(f"[Navigation] Navigating to: ({row}, {col})")

        if not (0 <= target_screen_row < 9 and 0 <= target_screen_col < 10):
            return f"Navigation failed: Target coordinates ({row}, {col}) are outside the visible screen area."

        # Execute the navigation
        status, path = self.emulator.find_path(target_screen_row, target_screen_col)
        if path:
            for direction in path:
                self.emulator.press_buttons([direction], wait=True)  # TODO: pass status in partial success case
            result = f"Navigation successful: followed path with {len(path)} steps"
        else:
            result = f"Navigation failed: {status}"

        # Return tool result as a dictionary
        return result


class OnScreenKeyboardTool(Tool):
    """Tool for navigating and selecting characters on an on-screen keyboard."""

    name = "on_screen_keyboard_input"
    description = (
        "Navigate and select characters on an on-screen keyboard. You should only use this tool when an on-screen "
        "keyboard is visible. This tool helps with text input by automatically navigating to and selecting characters."
    )
    ends_turn = True
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The text to input using the on-screen keyboard."},
            "end_input": {
                "type": "boolean",
                "description": "Whether to select 'END' after inputting the text to complete the input process.",
                "default": True,
            },
        },
        "required": ["text"],
        "additionalProperties": False,
    }

    # Keyboard layout mapping (position coordinates to character)
    KEYBOARD_LAYOUT = {
        # Row 1: A-I
        (0, 0): "A",
        (1, 0): "B",
        (2, 0): "C",
        (3, 0): "D",
        (4, 0): "E",
        (5, 0): "F",
        (6, 0): "G",
        (7, 0): "H",
        (8, 0): "I",
        # Row 2: J-R
        (0, 1): "J",
        (1, 1): "K",
        (2, 1): "L",
        (3, 1): "M",
        (4, 1): "N",
        (5, 1): "O",
        (6, 1): "P",
        (7, 1): "Q",
        (8, 1): "R",
        # Row 3: S-Z
        (0, 2): "S",
        (1, 2): "T",
        (2, 2): "U",
        (3, 2): "V",
        (4, 2): "W",
        (5, 2): "X",
        (6, 2): "Y",
        (7, 2): "Z",
        # Row 4: Special characters
        (0, 3): "×",
        (1, 3): "(",
        (2, 3): ")",
        (3, 3): ":",
        (4, 3): ";",
        (5, 3): "[",
        (6, 3): "]",
        (7, 3): "Pk",
        (8, 3): "Mn",
        # Row 5: More special characters and END
        (0, 4): "-",
        (1, 4): "?",
        (2, 4): "!",
        (3, 4): "♂",
        (4, 4): "♀",
        (5, 4): "/",
        (6, 4): ".",
        (7, 4): ",",
        (8, 4): "ED",  # ED is END
    }

    # Reverse mapping to find coordinates for a character
    CHAR_TO_COORDS = {char: coords for coords, char in KEYBOARD_LAYOUT.items()}

    def execute(self, tool_input: dict) -> str:
        """Process a keyboard_input tool call."""
        text = tool_input["text"]
        end_input = tool_input.get("end_input", True)

        logger.info(f"[Keyboard] Inputting text: '{text}' (end_input={end_input})")

        result_parts = []
        current_pos = (0, 0)  # Start at 'A'

        for char in text.upper():
            if char not in self.CHAR_TO_COORDS:
                result_parts.append(f"Character '{char}' not found on keyboard, skipping")
                continue

            target_pos = self.CHAR_TO_COORDS[char]

            # Navigate to the target character
            while current_pos != target_pos:
                # Move horizontally
                if current_pos[0] < target_pos[0]:
                    self.emulator.press_buttons(["right"], wait=True)
                    current_pos = (current_pos[0] + 1, current_pos[1])
                elif current_pos[0] > target_pos[0]:
                    self.emulator.press_buttons(["left"], wait=True)
                    current_pos = (current_pos[0] - 1, current_pos[1])

                # Move vertically
                elif current_pos[1] < target_pos[1]:
                    self.emulator.press_buttons(["down"], wait=True)
                    current_pos = (current_pos[0], current_pos[1] + 1)
                elif current_pos[1] > target_pos[1]:
                    self.emulator.press_buttons(["up"], wait=True)
                    current_pos = (current_pos[0], current_pos[1] - 1)

            # Press A to select the character
            self.emulator.press_buttons(["a"], wait=True)
            result_parts.append(f"Selected character '{char}' at {current_pos}")

        # If end_input is True, navigate to END and press A
        if end_input:
            end_pos = self.CHAR_TO_COORDS["ED"]

            # Navigate to END
            while current_pos != end_pos:
                if current_pos[0] < end_pos[0]:
                    self.emulator.press_buttons(["right"], wait=True)
                    current_pos = (current_pos[0] + 1, current_pos[1])
                elif current_pos[0] > end_pos[0]:
                    self.emulator.press_buttons(["left"], wait=True)
                    current_pos = (current_pos[0] - 1, current_pos[1])
                elif current_pos[1] < end_pos[1]:
                    self.emulator.press_buttons(["down"], wait=True)
                    current_pos = (current_pos[0], current_pos[1] + 1)
                elif current_pos[1] > end_pos[1]:
                    self.emulator.press_buttons(["up"], wait=True)
                    current_pos = (current_pos[0], current_pos[1] - 1)

            # Press A to select END
            self.emulator.press_buttons(["a"], wait=True)
            result_parts.append("Selected END to complete input")

        return "\n".join(result_parts)


class MemoryFileTool(Tool):
    """Base for the file tools over /memories (see agent/memory.py); subclasses set `name`."""

    def __init__(self, emulator: Emulator):
        super().__init__(emulator)
        self.store = MemoryStore()
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


TOOL_CLASSES: dict[str, type[Tool]] = {
    tool_class.name: tool_class
    for tool_class in [
        PressButtonTool,
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
    "press_button",
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

    def process_tool_call(self, tool_call: ChatCompletionMessageToolCall) -> ChatCompletionToolMessageParam:
        """Execute an OpenAI-format tool call and return the tool result message."""
        result = self.execute_tool(tool_call.function.name, json.loads(tool_call.function.arguments))
        return {"role": "tool", "tool_call_id": tool_call.id, "content": result}

    def get_available_tools(self):
        return [tool.get_tool_definition() for tool in self.tools.values()]
