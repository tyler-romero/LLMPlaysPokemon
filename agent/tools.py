import json
import logging
import os
from datetime import datetime

from openai.types.chat import ChatCompletionMessageToolCall, ChatCompletionToolMessageParam

from agent.emulator import Emulator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MEMORY_FILE = "agent_memory.txt"


class Tool:
    """Base class for all tools."""

    name: str = "base_tool"
    description: str = "Base tool description"
    parameters: dict = {}

    def __init__(self, emulator: Emulator):
        """Initialize the tool with an emulator instance."""
        self.emulator = emulator

    def process(self, tool_call: ChatCompletionMessageToolCall) -> ChatCompletionToolMessageParam:
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


class PressButtonsTool(Tool):
    """Tool for pressing buttons on the Game Boy."""

    name = "press_buttons"
    description = "Press one or more buttons on the Game Boy in sequence."
    parameters = {
        "type": "object",
        "properties": {
            "buttons": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "a",
                        "b",
                        "↑",
                        "↓",
                        "←",
                        "→",
                        "start",
                        "select",
                    ],
                },
                "description": "Sequence of buttons to press. Valid buttons: 'a', 'b', '↑', '↓', '←', '→', 'start', 'select'",
            },
            "wait": {
                "type": "boolean",
                "description": "Whether to wait for a brief period after pressing each button. Defaults to true.",
            },
        },
        "required": ["buttons"],
        "additionalProperties": False,
    }

    def process(self, tool_call: ChatCompletionMessageToolCall) -> ChatCompletionToolMessageParam:
        """Process a press_buttons tool call."""
        tool_input = json.loads(tool_call.function.arguments)
        buttons = tool_input["buttons"]
        wait = tool_input.get("wait", True)

        if not buttons:
            return {"role": "tool", "tool_call_id": tool_call.id, "content": "No buttons specified to press"}

        results = []
        for button in buttons:
            logger.info(f"[Button] Pressing: {button} (wait={wait})")
            self.emulator.press_buttons([button], wait)
            results.append(f"Pressed button: {button}")

        # Return tool result as a dictionary
        return {"role": "tool", "tool_call_id": tool_call.id, "content": "\n".join(results)}


class NavigateToScreenCoordinatesTool(Tool):
    """Tool for navigating to a position on the map grid."""

    name = "navigate_to_screen_coordinates"
    description = "Automatically navigate to a position on the current screen. The screen is divided into a 10x9 grid, with the left-top corner as (0, 0) and the right-bottom corner as (9, 8). The player is always at (4, 4) and the map moves around them. This tool is only available in the overworld."
    parameters = {
        "type": "object",
        "properties": {
            "coordinates": {
                "type": "array",
                "description": "A tuple of [col, row] coordinates to navigate to. Column ranges from 0-9, row ranges from 0-8.",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
            }
        },
        "required": ["coordinates"],
        "additionalProperties": False,
    }

    def process(self, tool_call) -> ChatCompletionToolMessageParam:
        """Process a navigate_to tool call."""
        tool_input = json.loads(tool_call.function.arguments)
        coordinates = tool_input["coordinates"]
        row = coordinates[1]
        col = coordinates[0]

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

    def process(self, tool_call: ChatCompletionMessageToolCall) -> ChatCompletionToolMessageParam:
        """Process a keyboard_input tool call."""
        tool_input = json.loads(tool_call.function.arguments)
        text = tool_input["text"]
        end_input = tool_input.get("end_input", False)

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

        return {"role": "tool", "tool_call_id": tool_call.id, "content": "\n".join(result_parts)}


class WriteToMemoryTool(Tool):
    """Tool for tracking memory in a text file."""

    name = "write_to_memory"
    description = "Save information to a memory file for later reference. Useful for tracking important game events, locations, or other information you want to remember."
    parameters = {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The information to save to the memory file."},
            "category": {
                "type": "string",
                "description": "Optional category to organize the memory (e.g., 'badges', 'pokemon_team', 'items', 'key_npcs', 'gym_leaders', 'routes', 'towns', 'pokedex').",
                "default": "general",
            },
        },
        "required": ["content"],
        "additionalProperties": False,
    }

    def __init__(self, emulator):
        """Initialize the memory tool."""
        super().__init__(emulator)
        # If the file exists, back it up first
        if os.path.exists(MEMORY_FILE):
            os.rename(MEMORY_FILE, f"{MEMORY_FILE}.old")
        # Create a new file
        with open(MEMORY_FILE, "w") as f:
            f.write("# Agent Memory File\n\n")

    def process(self, tool_call: ChatCompletionMessageToolCall) -> ChatCompletionToolMessageParam:
        """Process a memory_tracker tool call."""
        tool_input = json.loads(tool_call.function.arguments)
        content = tool_input["content"]
        category = tool_input.get("category", "general")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        memory_entry = f"[{timestamp}] [{category}] {content}\n"

        # Append to the memory file
        with open(MEMORY_FILE, "a") as f:
            f.write(memory_entry)

        logger.info(f"[Memory] Added: [{category}] {content}")
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": f"Memory saved: {content} (category: {category})",
        }


def read_memory() -> str:
    """Read the entire memory file and return its contents without comments."""
    if not os.path.exists(MEMORY_FILE):
        return "Memory file does not exist or is empty."
    with open(MEMORY_FILE, "r") as f:
        lines = f.readlines()
    memory_content = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    return "\n".join(memory_content)


class NavigationalHintsTool(Tool):
    """Tool for getting navigational hints based on the current location."""

    name = "get_navigational_hints"
    description = "Get navigational hints and information about the current location to help with exploration and navigation. Call this when stuck. It is recommended to record hints in your memory."
    parameters = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The current location name (e.g., 'PLAYERS HOUSE 2F', 'VIRIDIAN CITY'). If not provided, the tool will read the current location from memory.",
                "default": None,
            },
        },
        "additionalProperties": False,
    }

    # Location-based navigation hints
    LOCATION_HINTS = {
        # Specifically provided hints
        "PLAYERS HOUSE 2F": "This is your bedroom. You can use the PC to access your storage. The stairs are in the top-right corner of the room.",
        # Generic descriptions for locations
        "PLAYERS HOUSE 1F": "This is the ground floor of your house. Your mom is usually here. The door leads outside to PALLET TOWN.",
        "PALLET TOWN": "A small town with two houses and Professor Oak's Lab to the south. Route 1 is to the north.",
        "OAKS LAB": "Professor Oak's research lab. This is where you can get your starter Pokémon and Pokédex. The door is at the bottom of the room.",
        "VIRIDIAN CITY": "The first major city north of Pallet Town. Has a Pokémon Center, Poké Mart, and Gym (which opens later).",
        "VIRIDIAN FOREST": "A dense forest maze with many Bug Catchers and bug-type Pokémon. Leads to Pewter City.",
        "PEWTER CITY": "Home to the first Gym led by Brock who specializes in Rock-type Pokémon.",
        "PEWTER GYM": "Brock's Gym specializing in Rock-type Pokémon. Defeat him to earn the Boulder Badge.",
        "POKEMON CENTER": "Heal your Pokémon here for free. Also contains a PC to access storage and a link cable station.",
        "POKEMART": "Shop selling Poké Balls, Potions, and other useful items.",
        "ROUTE 1": "Connects Pallet Town to Viridian City. Contains wild Pidgey and Rattata.",
        "ROUTE 2": "Connects Viridian City to Pewter City through Viridian Forest.",
        "ROUTE 3": "East of Pewter City, leads to Mt. Moon. Many trainers here.",
        "MT MOON": "A large cave system with multiple floors. Home to many Zubat and the rare Clefairy.",
        "CERULEAN CITY": "City east of Mt. Moon. Has the second Gym led by Misty who specializes in Water-type Pokémon.",
        "CERULEAN GYM": "Misty's Gym specializing in Water-type Pokémon. Defeat her to earn the Cascade Badge.",
        "BILLS HOUSE": "Home of Bill, the Pokémon storage system developer. Located north of Cerulean City on Route 25.",
        "VERMILION CITY": "Port city with the S.S. Anne and the third Gym led by Lt. Surge.",
        "SS ANNE": "A luxury cruise ship docked in Vermilion City. Contains many trainers and a key item.",
        "VERMILION GYM": "Lt. Surge's Gym specializing in Electric-type Pokémon. Defeat him to earn the Thunder Badge.",
    }

    def process(self, tool_call: ChatCompletionMessageToolCall) -> ChatCompletionToolMessageParam:
        """Process a get_navigational_hints tool call."""
        tool_input = json.loads(tool_call.function.arguments)
        location = tool_input.get("location")

        # If location not provided, read from memory
        if not location:
            memory_dict, location, _ = self.emulator.get_state_from_memory()
            if not location:
                return {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": "Could not determine current location.",
                }

        # Normalize location name
        location = location.upper().strip()

        # Get hint for the location
        hint = self.LOCATION_HINTS.get(location)
        result = f"Hint for {location}:\n{hint}" if hint else f"No hints available for '{location}'."
        logger.info(f"[Navigation Hint] For location: {location}")
        return {"role": "tool", "tool_call_id": tool_call.id, "content": result}


class ToolFactory:
    """Factory for creating tool instances."""

    def __init__(self, emulator):
        """Initialize the factory with an emulator instance."""
        self.emulator = emulator
        self.tools = {
            "press_buttons": PressButtonsTool(emulator),
            "navigate_to_screen_coordinates": NavigateToScreenCoordinatesTool(emulator),
            "on_screen_keyboard_input": OnScreenKeyboardTool(emulator),
            "write_to_memory": WriteToMemoryTool(emulator),
            "get_navigational_hints": NavigationalHintsTool(emulator),
        }

        logger.info("Available tools:")
        for tool in self.get_available_tools():
            logger.info(f"- {tool['function']['name']}: {tool['function']['description']}")

    def process_tool_call(self, tool_call: ChatCompletionMessageToolCall) -> ChatCompletionToolMessageParam:
        tool_name = tool_call.function.name
        logger.info(f"Processing tool call: {tool_name}")
        tool = self.tools[tool_name]
        return tool.process(tool_call)

    def get_available_tools(self):
        available_tools = []
        for _, tool in self.tools.items():
            available_tools.append(tool.get_tool_definition())
        return available_tools
