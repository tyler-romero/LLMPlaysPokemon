import base64
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from pokemon_agent.agents.prompts import (
    FAILED_WRITE_NOTICE_TURNS,
    NOTE_WRITE_TOOLS,
    NOTES_STALE_EVERY,
    NOTES_STALE_TURNS,
    NOTICE_FAILED_WRITE,
    NOTICE_NO_NOTES,
    NOTICE_STALE,
    NOTICE_UNVIEWED,
)
from pokemon_agent.game.emulator import Emulator
from pokemon_agent.game.observation import GameObservation, get_screenshot_base64
from pokemon_agent.memory.store import MEMORY_DIR, MEMORY_TOOL_SPECS, MemoryStore, reset_memory, snapshot_memory
from pokemon_agent.observer.hub import ObserverHub
from pokemon_agent.runs import RunStore, copy_tree
from pokemon_agent.tools.base import ToolError, ToolOutput
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
        observer: ObserverHub | None = None,
        run: RunStore | None = None,
        resume: bool = False,
        text_map: bool = True,
    ):
        """Initialize the emulator and tools.

        Args:
            rom_path: Path to the ROM file
            headless: Whether to run without display
            sound: Whether to enable sound
            save_every: Number of steps between save_state() calls
            load_state: Path to a saved state file to load; its memory snapshot is restored if one exists
            observer: Hub that streams the run to the observer page, if enabled
            run: Run folder for durable checkpoints (game, memory, conversation); None saves to saved_games/
            resume: Restore the run's latest checkpoint instead of starting fresh
            text_map: Include the text-based map of the screen in each step's game state
        """
        self.observer = observer
        self.run_store = run
        self.text_map = text_map
        self._tool_calls = 0
        self.steps_completed = 0
        # For the turn message's notices (main agent only): failed note writes not yet redone, path -> (turn, tool),
        # and the turn the conversation was last condensed while the notes haven't been viewed since (None: viewed).
        self._failed_writes: dict[str, tuple[int, str]] = {}
        self._condensed_unviewed: int | None = None
        self.resume_checkpoint: Path | None = None
        # Who is acting, added to observer events: None for the main agent, "helper" while a subagent helper works.
        self.actor: str | None = None
        self.emulator = Emulator(rom_path, headless, sound=sound)
        self.emulator.initialize()

        self.running = True
        self.save_every = save_every
        memory_root = str(run.memory_dir) if run else MEMORY_DIR

        if resume:
            if not run or not (checkpoint := run.latest_checkpoint()):
                raise RuntimeError(f"Nothing to resume: {run.root if run else 'no run'} has no complete checkpoint")
            info = json.loads((checkpoint / "checkpoint.json").read_text())
            self.emulator.load_state(checkpoint / "game.state", validate=False)
            reset_memory(restore_from=str(checkpoint / "memories"), root=memory_root)
            self.steps_completed = info["step"]
            self._tool_calls = info.get("tool_calls", 0)
            self.resume_checkpoint = checkpoint
            logger.info(f"Resumed {run.run_id} from step {self.steps_completed} ({checkpoint})")
        elif load_state:
            logger.warning(f"Loading saved state from {load_state}")
            try:
                self.emulator.load_state(load_state)
                logger.info(f"Successfully loaded state from {load_state}")
            except Exception as e:
                logger.error(f"Failed to load state from {load_state}: {e}")
                raise RuntimeError(f"Could not load saved state: {e}")

        if not resume:
            # Only replace memory once the saved state has loaded, so a failed start leaves existing memory untouched.
            self._start_memory(load_state, memory_root)
        self.memory = MemoryStore(memory_root)
        self.tool_factory = ToolFactory(self.emulator, self.tool_names, memory=self.memory)
        for tool in self.tool_factory.tools.values():
            tool.emit = self.emit

        if observer:
            self.emulator.frame_listener = observer.on_frame

    def emit(self, kind: str, **data):
        """Publish an event to the observer page, if one is attached."""
        if self.observer:
            if self.actor:
                data.setdefault("actor", self.actor)
            self.observer.publish(kind, **data)

    def run_tool(self, name: str, tool_input: dict, factory: ToolFactory | None = None) -> str | ToolOutput:
        """Execute a tool (from the agent's own tools unless `factory` is given), reporting it to the observer."""
        self._tool_calls += 1
        call_id = self._tool_calls
        shown = tool_input
        if isinstance(narration := tool_input.get("narration"), str):
            # A game action's narration is the model speaking to viewers: show it as its dialogue, before the action.
            if narration.strip():
                self.emit("thought", text=narration.strip(), narration=True, step=self.steps_completed + 1)
            shown = {key: value for key, value in tool_input.items() if key != "narration"}
        self.emit("tool", id=call_id, name=name, args=shown)
        try:
            result = (factory or self.tool_factory).execute_tool(name, tool_input)
        except Exception as e:
            self.emit("tool_result", id=call_id, result=f"{type(e).__name__}: {e}", ok=False)
            raise
        text = result.text if isinstance(result, ToolOutput) else result
        # The observer shows any images exactly as the model receives them (base64 PNGs).
        images = {"images": result.images} if isinstance(result, ToolOutput) and result.images else {}
        self.emit("tool_result", id=call_id, result=text, ok=not isinstance(result, ToolError), **images)
        if name in MEMORY_TOOL_SPECS and name not in ("view", "list"):
            self.publish_memory()
        if self.actor is None:
            self._track_notes(name, tool_input, ok=not isinstance(result, ToolError))
        return result

    def _track_notes(self, name: str, tool_input: dict, ok: bool):
        path = tool_input.get("path")
        if not isinstance(path, str):
            return
        if name == "view" and ok and not path.lower().endswith(".png"):
            self._condensed_unviewed = None
        elif name in NOTE_WRITE_TOOLS:
            if ok:
                self._failed_writes.pop(path, None)
            else:
                self._failed_writes[path] = (self.steps_completed + 1, name)

    def notes_condensed(self):
        """The conversation was just condensed: until a note is viewed, the turn message says so."""
        self._condensed_unviewed = self.steps_completed + 1

    def notices(self) -> list[str]:
        """Short facts about the model's notes for the turn message, each only when it applies."""
        turn = self.steps_completed + 1
        lines = []
        note_turns = self.memory.note_turns()
        if self._condensed_unviewed is not None and note_turns:
            lines.append(NOTICE_UNVIEWED.format(turn=self._condensed_unviewed))
        for path, (failed_turn, tool) in sorted(self._failed_writes.items(), key=lambda item: item[1]):
            if turn - failed_turn > FAILED_WRITE_NOTICE_TURNS:
                del self._failed_writes[path]  # (a write the model chose not to redo shouldn't nag forever)
                continue
            lines.append(NOTICE_FAILED_WRITE.format(tool=tool, path=path, turn=failed_turn))
        newest = max(note_turns.values(), default=None)
        ago = turn - newest if newest is not None else turn - 1
        if ago >= NOTES_STALE_TURNS and ago % NOTES_STALE_EVERY == 0:
            if newest is None:
                lines.append(NOTICE_NO_NOTES.format(ago=ago))
            else:
                latest = max(note_turns, key=note_turns.get)
                lines.append(NOTICE_STALE.format(ago=ago, path=latest, turn=newest))
        return lines

    def report_not_run(self, name: str, args, reason: str):
        """Show the observer a tool call the agent refused to run (the model gets `reason` as its result)."""
        self._tool_calls += 1
        self.emit("tool", id=self._tool_calls, name=name, args=args or {})
        self.emit("tool_result", id=self._tool_calls, result=reason, ok=False)
        logger.info(f"[Tool] Not run {name}: {reason}")

    @staticmethod
    def _start_memory(load_state: str | None, memory_root: str):
        snapshot = None
        if load_state:
            # A run checkpoint's game.state keeps its memory beside it; legacy saves use <state>.memories.
            candidates = [Path(load_state).parent / "memories", Path(f"{load_state}.memories")]
            snapshot = next((str(c) for c in candidates if c.is_dir()), None)
            if not snapshot:
                logger.warning(f"No memory snapshot for {load_state}; starting with empty memory")
        reset_memory(restore_from=snapshot, root=memory_root)
        if snapshot:
            logger.info(f"Restored memory from {snapshot}")

    def save_state(self):
        """Checkpoint the run (game, memory, conversation, counters), or save to saved_games/ without a run."""
        if not self.run_store:
            for state_path in self.emulator.save_state():
                snapshot_memory(f"{state_path}.memories")
            return

        def write(directory: Path) -> dict:
            self.emulator.save_state_to(directory / "game.state")
            copy_tree(self.run_store.memory_dir, directory / "memories")
            (directory / "agent").mkdir()
            self.write_agent_state(directory / "agent")
            return {
                "tool_calls": self._tool_calls,
                "observer_seq": self.observer.seq if self.observer else None,
            }

        self.run_store.checkpoint(self.steps_completed, write)
        self.emit("checkpoint", step=self.steps_completed)

    def save_state_after_failure(self):
        """Best-effort checkpoint after a crash, so a durable run can resume from the last completed step.

        A clean checkpoint already taken at that step is kept rather than replaced by a mid-step one.
        """
        if self.run_store and (self.run_store.checkpoints / f"{self.steps_completed:06d}").is_dir():
            logger.info(f"Keeping the existing checkpoint for step {self.steps_completed}")
            return
        try:
            self.save_state()
        except Exception:
            logger.error("Could not save a checkpoint after the failure", exc_info=True)

    def write_agent_state(self, directory: Path):
        """Write the agent's conversation state into a checkpoint; subclasses override."""

    def restore_agent_state(self, directory: Path):
        """Restore the agent's conversation state from a checkpoint; subclasses override."""

    def publish_memory(self):
        """Show the observer the current memory files."""
        self.emit("memory", listing=self.memory.listing(), updated=self.memory.file_times())

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
        self.memory.turn = self.steps_completed + 1
        self.publish_memory()
        self.emulator.wait_for_screen_wipe()
        collision_map = self.emulator.get_collision_map() if self.text_map else None
        screenshot = self.emulator.get_screenshot()
        screenshot_b64 = get_screenshot_base64(self.emulator, screenshot)
        memory_info, location, coords = self.emulator.get_state_from_memory()
        self.emit("state", state=memory_info)
        # save_screenshot stores this turn's screenshot exactly as the model sees it, with where it was taken.
        self.memory.screenshot = base64.standard_b64decode(screenshot_b64)
        self.memory.place = f"{location} ({coords[0]},{coords[1]})" if location and coords else location

        logger.info("[Memory State after action]")
        logger.info(memory_info)
        if collision_map:
            logger.info(f"[Collision Map after action]\n{collision_map}")
        return GameObservation(screenshot_b64=screenshot_b64, memory_info=memory_info, collision_map=collision_map)

    def _on_step_complete(self, steps_completed: int, num_steps: int):
        """Log progress and periodically save emulator state and memory."""
        self.steps_completed = steps_completed
        logger.info(f"Completed step {steps_completed}/{num_steps}")
        self.emit("step", step=steps_completed, total=num_steps)
        if self.save_every and steps_completed % self.save_every == 0 and steps_completed > 0:
            self.save_state()

    def stop(self):
        """Stop the agent."""
        self.running = False
        self.emulator.stop()
