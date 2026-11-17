"""Pokemon Red game-control tools."""

import logging

from pokemon_agent.game.emulator import Press
from pokemon_agent.tools.base import Tool, ToolError

logger = logging.getLogger(__name__)


BUTTONS = ["a", "b", "up", "down", "left", "right", "start", "select"]
DIRECTIONS = {"up", "down", "left", "right"}
MAX_BUTTONS_PER_CALL = 20

# Every game action carries the model's narration for viewers; agents show it as the model's dialogue before the action.
NARRATION_PARAMETER = {
    "type": "string",
    "description": "One or two plain sentences for the people watching: what you are doing and why. Shown to viewers "
    "as you speaking, before the action.",
}


def missing_narration(tool_input: dict) -> ToolError | None:
    if isinstance(tool_input.get("narration"), str) and tool_input["narration"].strip():
        return None
    return ToolError(
        "Error: 'narration' is required: one or two plain sentences for the people watching, saying what you are "
        "doing and why. Nothing was pressed."
    )


def describe_presses(presses: list[Press]) -> str:
    """One clause per press saying where it left the player, or nothing if every press went to text, a menu, or a
    battle and no map changed (the next screenshot shows those)."""
    if not any(
        press.before[0] != press.after[0] or (not press.in_menu and (press.button in DIRECTIONS or press.event))
        for press in presses
    ):
        return ""
    clauses = []
    for press in presses:
        (old_map, old_xy), (new_map, new_xy) = press.before, press.after
        if new_map != old_map:
            what = f"you left {old_map} and are now in {new_map} at ({new_xy[0]}, {new_xy[1]})"
        elif press.in_menu:
            what = "pressed with text, a menu, or a battle on screen"
        elif new_xy != old_xy:
            what = f"moved to ({new_xy[0]}, {new_xy[1]})"
        elif press.button in DIRECTIONS:
            what = f"didn't move you; still at ({new_xy[0]}, {new_xy[1]})"
        else:
            what = f"you stayed at ({new_xy[0]}, {new_xy[1]})"
        if press.event:
            what += f", and {press.event}"
        if press.busy_before:
            what += " (the game was still busy with a scripted event, so this press may have been ignored)"
        clauses.append(f"{press.button}: {what}.")
    return " Each press: " + " ".join(clauses)


class PressButtonsTool(Tool):
    """Tool for pressing a sequence of buttons on the Game Boy."""

    name = "press_buttons"
    description = (
        "Press one or more buttons on the Game Boy, in order. In Pokemon Red, the arrow keys are used to move the "
        "player or cursor. The 'a' button is used to confirm decisions, talk or read, advance dialogue, etc. The 'b' "
        "button is used to cancel decisions, close menus, etc. The 'start' button is used to open the overworld menu. "
        "When a press moves you or changes the map, the result says where each press left you, so you can see which "
        "press was blocked.\n"
        "Arguments: 'narration' (a string) and 'buttons' (a JSON array of button names). Each button name is a plain "
        "lowercase word with no quote marks inside it: a, b, up, down, left, right, start, or select. A complete "
        "call's arguments look like {\"narration\": \"Walking up to the door.\", \"buttons\": [\"up\", \"up\", \"a\"]}."
    )
    ends_turn = True
    parameters = {
        "type": "object",
        "properties": {
            "narration": NARRATION_PARAMETER,
            "buttons": {
                "type": "array",
                "items": {"type": "string", "enum": BUTTONS},
                "minItems": 1,
                "maxItems": MAX_BUTTONS_PER_CALL,
                "description": "Buttons to press in order: a JSON array of plain lowercase names, e.g. [\"a\"] or "
                "[\"up\", \"up\", \"a\"]. Each name is exactly one of a, b, up, down, left, right, start, select, with "
                "no quote marks inside it.",
            },
            "wait": {
                "type": "boolean",
                "description": "Whether to let the game run for about 2 seconds after each press (longer if a "
                "scripted event is still playing out) so walking, menu, and text animations finish before the next "
                "press and the next screenshot. Defaults to true. Set to false "
                "only for rapid presses whose intermediate results you don't need to see (the next screenshot may then "
                "show the game mid-animation).",
            },
        },
        "required": ["narration", "buttons"],
        "additionalProperties": False,
    }

    def execute(self, tool_input: dict) -> str:
        """Process a press_buttons tool call."""
        if error := missing_narration(tool_input):
            return error
        buttons = tool_input.get("buttons") or []
        wait = tool_input.get("wait", True)
        if not buttons:
            return ToolError("Error: buttons must contain at least one button.")
        if len(buttons) > MAX_BUTTONS_PER_CALL:
            return ToolError(f"Error: at most {MAX_BUTTONS_PER_CALL} buttons per call; got {len(buttons)}.")
        invalid = [button for button in buttons if button not in BUTTONS]
        if invalid:
            return ToolError(f"Error: invalid buttons {invalid}; valid buttons are {BUTTONS}.")

        logger.info(f"[Button] Pressing: {buttons} (wait={wait})")
        presses = self.emulator.press_buttons(buttons, wait)

        return f"Pressed buttons: {', '.join(buttons)}.{describe_presses(presses)}"


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
            "narration": NARRATION_PARAMETER,
            "text": {"type": "string", "description": "The text to input using the on-screen keyboard."},
            "end_input": {
                "type": "boolean",
                "description": "Whether to select 'END' after inputting the text to complete the input process.",
                "default": True,
            },
        },
        "required": ["narration", "text"],
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
        if error := missing_narration(tool_input):
            return error
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
