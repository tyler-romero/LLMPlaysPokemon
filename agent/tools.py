import json
import logging
import os
from datetime import datetime

from openai.types.chat import ChatCompletionMessageToolCall, ChatCompletionToolMessageParam

from agent.emulator import Emulator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_FILE = "agent_knowledge_base.txt"


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


class PressButtonTool(Tool):
    """Tool for pressing a button on the Game Boy."""

    name = "press_button"
    description = "Press a button on the Game Boy. In Pokemon Red, the arrow keys are used move the player or cursor. The 'a' button is used to confirm decision, talk or read, advance dialogue, etc. The 'b' button is used to cancel decisions, close menus, etc. The 'start' button is used to open the overworld menu."
    parameters = {
        "type": "object",
        "properties": {
            "button": {
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
                "description": "Button to press. Valid buttons: 'a', 'b', '↑', '↓', '←', '→', 'start', 'select'",
            },
            "wait": {
                "type": "boolean",
                "description": "Whether to wait for a brief period after pressing the button. Defaults to true.",
            },
        },
        "required": ["button"],
        "additionalProperties": False,
    }

    def process(self, tool_call: ChatCompletionMessageToolCall) -> ChatCompletionToolMessageParam:
        """Process a press_button tool call."""
        tool_input = json.loads(tool_call.function.arguments)
        button = tool_input["button"]
        wait = tool_input.get("wait", True)

        logger.info(f"[Button] Pressing: {button} (wait={wait})")
        self.emulator.press_buttons([button], wait)

        return {"role": "tool", "tool_call_id": tool_call.id, "content": f"Pressed button: {button}"}


class NavigateToScreenCoordinatesTool(Tool):
    """Tool for navigating to a position on the map grid."""

    name = "navigate_to_screen_coordinates"
    description = "Automatically navigate to the specified map coordinates on the current screen. The screen is divided into a grid and the available coordinates are rendered in the provided screenshot. This tool is only available in the overworld."
    parameters = {
        "type": "object",
        "properties": {
            "col": {"type": "integer", "description": "The column coordinate to navigate to."},
            "row": {"type": "integer", "description": "The row coordinate to navigate to."},
        },
        "required": ["col", "row"],
        "additionalProperties": False,
    }

    def process(self, tool_call) -> ChatCompletionToolMessageParam:
        """Process a navigate_to tool call."""
        tool_input = json.loads(tool_call.function.arguments)
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
            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": f"Navigation failed: Target coordinates ({row}, {col}) are outside the visible screen area.",
            }

        # Execute the navigation
        status, path = self.emulator.find_path(target_screen_row, target_screen_col)
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
    description = (
        "Navigate and select characters on an on-screen keyboard. You should only use this tool when an on-screen "
        "keyboard is visible. This tool helps with text input by automatically navigating to and selecting characters."
    )
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


class AddToKnowledgeBaseTool(Tool):
    """Tool for adding information to the knowledge base."""

    name = "add_to_knowledge_base"
    description = "Add information to the knowledge base. Useful for tracking important game events, locations, or other information you want to remember."
    parameters = {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The information to save to the knowledge base."},
            "category": {
                "type": "string",
                "description": "Optional category to organize the knowledge base (e.g., 'badges', 'pokemon_team', 'items', 'key_npcs', 'gym_leaders', 'routes', 'towns', 'pokedex').",
                "default": "general",
            },
        },
        "required": ["content"],
        "additionalProperties": False,
    }

    def __init__(self, emulator):
        """Initialize the knowledge base tool."""
        super().__init__(emulator)
        # If the file exists, back it up first
        if os.path.exists(KNOWLEDGE_BASE_FILE):
            os.rename(KNOWLEDGE_BASE_FILE, f"{KNOWLEDGE_BASE_FILE}.old")
        # Create a new file
        with open(KNOWLEDGE_BASE_FILE, "w") as f:
            f.write("# Agent Knowledge Base\n\n")

    def process(self, tool_call: ChatCompletionMessageToolCall) -> ChatCompletionToolMessageParam:
        """Process a add_to_knowledge_base tool call."""
        tool_input = json.loads(tool_call.function.arguments)
        content = tool_input["content"]
        category = tool_input.get("category", "general")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        knowledge_entry = f"[{timestamp}] [{category}] {content}\n"

        # Append to the memory file
        with open(KNOWLEDGE_BASE_FILE, "a") as f:
            f.write(knowledge_entry)

        logger.info(f"[Knowledge Base] Added: [{category}] {content}")
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": f"Knowledge base updated: {content} (category: {category})",
        }


def read_knowledge_base() -> str:
    """Read the entire knowledge base file and return its contents without comments."""
    if not os.path.exists(KNOWLEDGE_BASE_FILE):
        return "Knowledge base file does not exist or is empty."
    with open(KNOWLEDGE_BASE_FILE, "r") as f:
        lines = f.readlines()
    knowledge_base_content = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    return "\n".join(knowledge_base_content)


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
        "OAKS LAB": "Professor Oak's research lab. This is where you can get your starter Pokémon and Pokédex. The door is at the bottom of the room.",
        # Generic descriptions for locations
        "PLAYERS HOUSE 1F": "This is the ground floor of your house. Your mom is usually here. The door leads outside to PALLET TOWN.",
        "PALLET TOWN": "A small town with two houses and Professor Oak's Lab to the south. Route 1 is to the north.",
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
            "press_button": PressButtonTool(emulator),
            "navigate_to_screen_coordinates": NavigateToScreenCoordinatesTool(emulator),
            "on_screen_keyboard_input": OnScreenKeyboardTool(emulator),
            "add_to_knowledge_base": AddToKnowledgeBaseTool(emulator),
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
