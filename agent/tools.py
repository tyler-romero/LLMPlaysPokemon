import json
import logging

from openai.types.chat import ChatCompletionToolMessageParam

from agent.emulator import Emulator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class Tool:
    """Base class for all tools."""

    name: str = "base_tool"
    description: str = "Base tool description"
    parameters: dict = {}

    def __init__(self, emulator: Emulator):
        """Initialize the tool with an emulator instance."""
        self.emulator = emulator

    def process(self, tool_call) -> ChatCompletionToolMessageParam:
        """Process the tool call and return results."""
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
    """Tool for pressing buttons on the Game Boy."""

    name = "press_button"
    description = "Press a button on the Game Boy."
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
                "description": "Button to press. Valid buttons: 'a', 'b', 'up', 'down', 'left', 'right', 'start', 'select'",
            },
            "wait": {
                "type": "boolean",
                "description": "Whether to wait for a brief period after pressing button. Defaults to true.",
            },
        },
        "required": ["button"],
        "additionalProperties": False,
    }

    def process(self, tool_call) -> ChatCompletionToolMessageParam:
        """Process a press_button tool call."""
        tool_input = json.loads(tool_call.function.arguments)
        button = tool_input["button"]
        wait = tool_input.get("wait", True)

        logger.info(f"[Button] Pressing: {button} (wait={wait})")
        self.emulator.press_buttons([button], wait)

        # Return tool result as a dictionary
        return {"role": "tool", "tool_call_id": tool_call.id, "content": f"Pressed button: {button}"}


class NavigateToCoordinatesTool(Tool):
    """Tool for navigating to a position on the map grid."""

    name = "navigate_to_coordinates"
    description = "Automatically navigate to a position on the map grid. The screen is divided into a 10x9 grid, with the top-left corner as (0, 0) and the bottom-right corner as (9, 8). This tool is only available in the overworld."
    parameters = {
        "type": "object",
        "properties": {
            "col": {"type": "integer", "description": "The column coordinate to navigate to (0-9)."},
            "row": {"type": "integer", "description": "The row coordinate to navigate to (0-8)."},
        },
        "required": ["col", "row"],
        "additionalProperties": False,
    }

    def process(self, tool_call) -> ChatCompletionToolMessageParam:
        """Process a navigate_to tool call."""
        tool_input = json.loads(tool_call.function.arguments)
        row = tool_input["row"]
        col = tool_input["col"]

        logger.info(f"[Navigation] Navigating to: ({row}, {col})")

        # Execute the navigation
        status, path = self.emulator.find_path(row, col)
        if path:
            for direction in path:
                self.emulator.press_buttons([direction], wait=True)  # TODO: pass status in partial success case
            result = f"Navigation successful: followed path with {len(path)} steps"
        else:
            result = f"Navigation failed: {status}"

        # Return tool result as a dictionary
        return {"role": "tool", "tool_call_id": tool_call.id, "content": result}


class DeadReckoningNavigationTool(Tool):
    """Tool for navigating using dead reckoning (relative movement from current position)."""

    name = "navigate_using_dead_reckoning"
    description = "Navigate using relative directions from the current position. This tool allows you to specify a sequence of movements (up, down, left, right) to execute in order."
    parameters = {
        "type": "object",
        "properties": {
            "directions": {
                "type": "array",
                "description": "A sequence of directions to move in ('up', 'down', 'left', 'right'). For example, ['up', 'right', 'down'] would move up, then right, then down.",
                "items": {"type": "string", "enum": ["up", "down", "left", "right"]},
            },
            "wait_after_each": {
                "type": "boolean",
                "description": "Whether to wait after each button press for the movement to complete.",
                "default": True,
            },
        },
        "required": ["directions"],
        "additionalProperties": False,
    }

    def process(self, tool_call) -> ChatCompletionToolMessageParam:
        """Process a dead reckoning navigation tool call."""
        tool_input = json.loads(tool_call.function.arguments)
        directions = tool_input["directions"]
        wait_after_each = tool_input.get("wait_after_each", True)

        logger.info(f"[Dead Reckoning] Executing {len(directions)} movements: {directions}")

        result_parts = []
        for i, direction in enumerate(directions):
            result = self.emulator.press_buttons([direction], wait=wait_after_each)
            result_parts.append(f"\tStep {i + 1}: Moved {direction} ({result=})")

        result = "\n".join(result_parts)
        return {"role": "tool", "tool_call_id": tool_call.id, "content": result}


class OnScreenKeyboardTool(Tool):
    """Tool for navigating and selecting characters on an on-screen keyboard."""

    name = "on_screen_keyboard_input"
    description = "Navigate and select characters on an on-screen keyboard. This tool helps with text input by automatically navigating to and selecting characters."
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

    def process(self, tool_call) -> ChatCompletionToolMessageParam:
        """Process a keyboard_input tool call."""
        tool_input = json.loads(tool_call.function.arguments)
        text = tool_input["text"]
        end_input = tool_input.get("end_input", False)

        logger.info(f"[Keyboard] Inputting text: '{text}' (end_input={end_input})")

        result_parts = []
        current_pos = (0, 0)  # Start at 'A'

        for char in text.upper():
            if char == " ":
                # Handle space character (not explicitly shown in the layout)
                result_parts.append("Skipping space character")
                continue

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
                    result_parts.append(f"Moved right to {current_pos}")
                elif current_pos[0] > target_pos[0]:
                    self.emulator.press_buttons(["left"], wait=True)
                    current_pos = (current_pos[0] - 1, current_pos[1])
                    result_parts.append(f"Moved left to {current_pos}")

                # Move vertically
                elif current_pos[1] < target_pos[1]:
                    self.emulator.press_buttons(["down"], wait=True)
                    current_pos = (current_pos[0], current_pos[1] + 1)
                    result_parts.append(f"Moved down to {current_pos}")
                elif current_pos[1] > target_pos[1]:
                    self.emulator.press_buttons(["up"], wait=True)
                    current_pos = (current_pos[0], current_pos[1] - 1)
                    result_parts.append(f"Moved up to {current_pos}")

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
                    result_parts.append(f"Moved right to {current_pos}")
                elif current_pos[0] > end_pos[0]:
                    self.emulator.press_buttons(["left"], wait=True)
                    current_pos = (current_pos[0] - 1, current_pos[1])
                    result_parts.append(f"Moved left to {current_pos}")
                elif current_pos[1] < end_pos[1]:
                    self.emulator.press_buttons(["down"], wait=True)
                    current_pos = (current_pos[0], current_pos[1] + 1)
                    result_parts.append(f"Moved down to {current_pos}")
                elif current_pos[1] > end_pos[1]:
                    self.emulator.press_buttons(["up"], wait=True)
                    current_pos = (current_pos[0], current_pos[1] - 1)
                    result_parts.append(f"Moved up to {current_pos}")

            # Press A to select END
            self.emulator.press_buttons(["a"], wait=True)
            result_parts.append("Selected END to complete input")

        return {"role": "tool", "tool_call_id": tool_call.id, "content": "\n".join(result_parts)}


class ToolFactory:
    """Factory for creating tool instances."""

    def __init__(self, emulator):
        """Initialize the factory with an emulator instance."""
        self.emulator = emulator
        self.tools = {
            "press_button": PressButtonTool(emulator),
            "navigate_to_coordinates": NavigateToCoordinatesTool(emulator),
            "navigate_using_dead_reckoning": DeadReckoningNavigationTool(emulator),
            "on_screen_keyboard_input": OnScreenKeyboardTool(emulator),
        }

        logger.info("Available tools:")
        for tool in self.get_available_tools():
            logger.info(f"- {tool['function']['name']}: {tool['function']['description']}")

    def process_tool_call(self, tool_call) -> ChatCompletionToolMessageParam:
        tool_name = tool_call.function.name
        logger.info(f"Processing tool call: {tool_name}")
        tool = self.tools[tool_name]
        return tool.process(tool_call)

    def get_available_tools(self):
        available_tools = []
        for _, tool in self.tools.items():
            available_tools.append(tool.get_tool_definition())
        return available_tools
