"""Pokemon Red game-control tools."""

import logging

from pokemon_agent.tools.base import Tool

logger = logging.getLogger(__name__)


BUTTONS = ["a", "b", "up", "down", "left", "right", "start", "select"]
MAX_BUTTONS_PER_CALL = 20


class PressButtonsTool(Tool):
    """Tool for pressing a sequence of buttons on the Game Boy."""

    name = "press_buttons"
    description = (
        "Press one or more buttons on the Game Boy, in order. In Pokemon Red, the arrow keys are used to move the "
        "player or cursor. The 'a' button is used to confirm decisions, talk or read, advance dialogue, etc. The 'b' "
        "button is used to cancel decisions, close menus, etc. The 'start' button is used to open the overworld menu."
    )
    ends_turn = True
    parameters = {
        "type": "object",
        "properties": {
            "buttons": {
                "type": "array",
                "items": {"type": "string", "enum": BUTTONS},
                "minItems": 1,
                "maxItems": MAX_BUTTONS_PER_CALL,
                "description": "Buttons to press in order, e.g. [\"a\"] or [\"up\", \"up\", \"a\"]. Valid buttons: "
                "'a', 'b', 'up', 'down', 'left', 'right', 'start', 'select'.",
            },
            "wait": {
                "type": "boolean",
                "description": "Whether to let the game run for about 2 seconds after each press so walking, menu, and "
                "text animations finish before the next press and the next screenshot. Defaults to true. Set to false "
                "only for rapid presses whose intermediate results you don't need to see (the next screenshot may then "
                "show the game mid-animation).",
            },
        },
        "required": ["buttons"],
        "additionalProperties": False,
    }

    def execute(self, tool_input: dict) -> str:
        """Process a press_buttons tool call."""
        buttons = tool_input.get("buttons") or []
        wait = tool_input.get("wait", True)
        if not buttons:
            return "Error: buttons must contain at least one button."
        if len(buttons) > MAX_BUTTONS_PER_CALL:
            return f"Error: at most {MAX_BUTTONS_PER_CALL} buttons per call; got {len(buttons)}."
        invalid = [button for button in buttons if button not in BUTTONS]
        if invalid:
            return f"Error: invalid buttons {invalid}; valid buttons are {BUTTONS}."

        logger.info(f"[Button] Pressing: {buttons} (wait={wait})")
        self.emulator.press_buttons(buttons, wait)

        return f"Pressed buttons: {', '.join(buttons)}"


class NavigateToScreenCoordinatesTool(Tool):
    """Tool for walking to a visible cell of the screenshot's coordinate grid."""

    name = "navigate_to_screen_coordinates"
    description = (
        "Walk the player to a cell that is visible on the current screen, following a path around obstacles. "
        "Outside of battle the screenshot is overlaid with a grid; each cell is labeled (x,y) with its absolute map "
        "coordinates, the same coordinates as map_coordinates in the game state. Pass the x and y from a labeled "
        "cell. The player stands in the unlabeled cell at the center of the grid, and only the cells on screen "
        "(x from your x-4 to x+5, y from your y-4 to y+4) can be targeted. Overworld only."
    )
    ends_turn = True
    parameters = {
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "Target x: the first number of the cell's (x,y) label."},
            "y": {"type": "integer", "description": "Target y: the second number of the cell's (x,y) label."},
        },
        "required": ["x", "y"],
        "additionalProperties": False,
    }

    def execute(self, tool_input: dict) -> str:
        """Process a navigate_to_screen_coordinates tool call."""
        x, y = tool_input["x"], tool_input["y"]
        player_x, player_y = self.emulator.get_coordinates()

        # The player is always at screen cell (row 4, col 4) of the 9x10 grid.
        target_screen_row = y - player_y + 4
        target_screen_col = x - player_x + 4

        logger.info(f"[Navigation] Navigating to: ({x}, {y}) from ({player_x}, {player_y})")

        if not (0 <= target_screen_row < 9 and 0 <= target_screen_col < 10):
            return (
                f"Navigation failed: ({x}, {y}) is not on screen. You are at ({player_x}, {player_y}); visible cells "
                f"span x {player_x - 4} to {player_x + 5} and y {player_y - 4} to {player_y + 4}."
            )

        status, path = self.emulator.find_path(target_screen_row, target_screen_col)
        if path:
            for direction in path:
                self.emulator.press_buttons([direction], wait=True)  # TODO: pass status in partial success case
            return f"Navigation successful: followed path with {len(path)} steps"
        return f"Navigation failed: {status}"


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
