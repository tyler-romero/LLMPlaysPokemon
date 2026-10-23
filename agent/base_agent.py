import base64
import io
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from openai.types.chat import ChatCompletionUserMessageParam
from PIL import Image, ImageDraw, ImageFont

from agent.emulator import Emulator
from agent.memory import MemoryStore, reset_memory, snapshot_memory
from agent.tools import ToolFactory

logger = logging.getLogger(__name__)


def get_screenshot_base64(emulator: Emulator, screenshot: Image.Image, upscale: int = 3):
    """Convert PIL image to base64 string."""
    if upscale > 1:
        new_size = (screenshot.width * upscale, screenshot.height * upscale)
        logger.info(f"[Screenshot] Upscaling from {screenshot.size} to {new_size}")
        screenshot = screenshot.resize(new_size)

    if not emulator.get_in_combat():
        shape = screenshot.size
        player_x, player_y = emulator.get_coordinates()  # absolute coordinates

        cell_width = shape[0] // 10
        cell_height = shape[1] // 9

        # Draw grid lines
        for x in range(0, shape[0], cell_width):
            ImageDraw.Draw(screenshot).line(((x, 0), (x, shape[1] - 1)), fill=(255, 0, 0))
        for y in range(0, shape[1], cell_height):
            ImageDraw.Draw(screenshot).line(((0, y), (shape[0] - 1, y)), fill=(255, 0, 0))

        # Add coordinates to every other tile
        font = ImageFont.load_default()
        for grid_y in range(9):
            for grid_x in range(10):
                # Calculate the relative coordinates
                rel_x = grid_x - 4 + player_x
                rel_y = grid_y - 4 + player_y

                if not (grid_x == 4 and grid_y == 4):
                    # Calculate position for text
                    text_x = grid_x * cell_width + 2
                    text_y = grid_y * cell_height + 2

                    # Create coordinate text
                    coord_text = f"({rel_x},{rel_y})"

                    # Add yellow background for text
                    text_width, text_height = (
                        font.getsize(coord_text)
                        if hasattr(font, "getsize")
                        else ImageDraw.Draw(screenshot).textbbox((0, 0), coord_text, font=font)[2:4]
                    )
                    ImageDraw.Draw(screenshot).rectangle(
                        [(text_x, text_y), (text_x + text_width, text_y + text_height)], fill=(255, 255, 0)
                    )

                    # Draw text
                    ImageDraw.Draw(screenshot).text((text_x, text_y), coord_text, fill=(0, 0, 0), font=font)

    screenshot.save("screenshot_test.png")
    buffered = io.BytesIO()
    screenshot.save(buffered, format="PNG")
    return base64.standard_b64encode(buffered.getvalue()).decode()


def create_user_message_with_game_state(
    prefix_content: str | None = None,
    screenshot_b64: str | None = None,
    memory_info: dict | None = None,
    collision_map: str | None = None,
    memory_listing: str | None = None,
    suffix_content: str | None = None,
) -> ChatCompletionUserMessageParam:
    """Create a user message with game state data, ensuring no empty content parts."""
    content_parts = []

    if prefix_content:
        content_parts.append({"type": "text", "text": prefix_content})

    if screenshot_b64:
        content_parts.append({"type": "text", "text": "Here is a screenshot of the screen:\n"})
        content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}})

    if memory_info:
        content_parts.append(
            {
                "type": "text",
                "text": f"\nGame state information from memory:\n{json.dumps(memory_info, indent=2)}",
            }
        )

    if memory_listing is not None:
        content_parts.append(
            {
                "type": "text",
                "text": "\nFiles in your memory directory (read them with the view tool):\n"
                + (memory_listing or "(empty)"),
            }
        )

    if collision_map:
        content_parts.append(
            {
                "type": "text",
                "text": f"\nText-based map:\n{collision_map}",
            }
        )

    if suffix_content:
        content_parts.append({"type": "text", "text": suffix_content})

    return {"role": "user", "content": content_parts}


@dataclass
class GameObservation:
    screenshot_b64: str
    memory_info: dict
    collision_map: str | None


class BaseAgent(ABC):
    """Model-independent agent plumbing: emulator lifecycle, tools, observations, and saving.

    Subclasses implement `run` and choose their tools with `tool_names` (None selects DEFAULT_TOOL_NAMES).
    """

    tool_names: list[str] | None = None

    def __init__(
        self,
        rom_path,
        headless=True,
        sound=False,
        save_every=100,
        load_state=None,
        web_output_dir=None,
    ):
        """Initialize the emulator and tools.

        Args:
            rom_path: Path to the ROM file
            headless: Whether to run without display
            sound: Whether to enable sound
            save_every: Number of steps between save_state() calls
            load_state: Path to a saved state file to load; its memory snapshot is restored if one exists
            web_output_dir: Directory to save latest.png for web viewing
        """
        self.emulator = Emulator(rom_path, headless, sound=sound)
        self.emulator.initialize()

        self.running = True
        self.save_every = save_every
        self.web_output_dir = web_output_dir

        if load_state:
            logger.warning(f"Loading saved state from {load_state}")
            try:
                self.emulator.load_state(load_state)
                logger.info(f"Successfully loaded state from {load_state}")
            except Exception as e:
                logger.error(f"Failed to load state from {load_state}: {e}")
                raise RuntimeError(f"Could not load saved state: {e}")

        # Only replace memory once the saved state has loaded, so a failed start leaves existing memory untouched.
        self._start_memory(load_state)
        self.memory = MemoryStore()
        self.tool_factory = ToolFactory(self.emulator, self.tool_names)

        self._save_web_screenshot()

    @staticmethod
    def _start_memory(load_state: str | None):
        snapshot = f"{load_state}.memories" if load_state else None
        if snapshot and not os.path.isdir(snapshot):
            logger.warning(f"No memory snapshot at {snapshot}; starting with empty memory")
            snapshot = None
        reset_memory(restore_from=snapshot)
        if snapshot:
            logger.info(f"Restored memory from {snapshot}")

    def save_state(self):
        """Save the emulator state, with a snapshot of the memory directory next to each state file."""
        for state_path in self.emulator.save_state():
            snapshot_memory(f"{state_path}.memories")

    def memory_listing(self) -> str:
        return self.memory.listing()

    @abstractmethod
    def run(self, num_steps=1) -> int:
        """Run the agent for up to num_steps steps and return the number of steps completed."""

    def observe(self) -> GameObservation:
        """Get the annotated screenshot and game state after an action."""
        collision_map = self.emulator.get_collision_map()
        screenshot = self.emulator.get_screenshot()
        screenshot_b64 = get_screenshot_base64(self.emulator, screenshot)
        memory_info, location, coords = self.emulator.get_state_from_memory()

        logger.info("[Memory State after action]")
        logger.info(memory_info)
        if collision_map:
            logger.info(f"[Collision Map after action]\n{collision_map}")
            with open("collision_map.txt", "w") as f:
                f.write(collision_map)
            logger.debug("Collision map saved to collision_map.txt")
        return GameObservation(screenshot_b64=screenshot_b64, memory_info=memory_info, collision_map=collision_map)

    def _save_web_screenshot(self):
        """Save the current screenshot to web_output_dir/latest.png if web output is enabled."""
        if not self.web_output_dir:
            return  # Web output not enabled

        try:
            # Get screenshot and save to file
            screenshot = self.emulator.get_screenshot()
            if screenshot:
                output_file = os.path.join(self.web_output_dir, "latest.png")
                screenshot.save(output_file)
                logger.debug(f"Saved screenshot to {output_file}")
            else:
                logger.warning("Could not get screenshot for web output")
        except Exception as e:
            logger.error(f"Error saving screenshot for web output: {e}")

    def _on_step_complete(self, steps_completed: int, num_steps: int):
        """Log progress, refresh the web screenshot, and periodically save emulator state and memory."""
        logger.info(f"Completed step {steps_completed}/{num_steps}")
        self._save_web_screenshot()
        if steps_completed % self.save_every == 0 and steps_completed > 0:
            self.save_state()

    def stop(self):
        """Stop the agent."""
        self.running = False
        self.emulator.stop()
