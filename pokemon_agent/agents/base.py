import json
import logging
import os
from abc import ABC, abstractmethod

from pokemon_agent.game.emulator import Emulator
from pokemon_agent.game.observation import GameObservation, get_screenshot_base64
from pokemon_agent.memory.store import MemoryStore, reset_memory, snapshot_memory
from pokemon_agent.tools.factory import ToolFactory

logger = logging.getLogger(__name__)


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

    @staticmethod
    def game_state_sections(observation: GameObservation, memory_listing: str | None = None) -> list[str]:
        """Text describing the game state and memory, shared by every agent's step message."""
        sections = []
        if observation.memory_info:
            sections.append(f"\nGame state information from memory:\n{json.dumps(observation.memory_info, indent=2)}")
        if memory_listing is not None:
            sections.append(
                "\nFiles in your memory directory (read them with the view tool):\n" + (memory_listing or "(empty)")
            )
        if observation.collision_map:
            sections.append(f"\nText-based map:\n{observation.collision_map}")
        return sections

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
