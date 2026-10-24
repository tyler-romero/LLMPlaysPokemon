"""Agent built on the GitHub Copilot SDK, which runs the model/tool loop and compacts context on its own."""

import asyncio
import logging
from pathlib import Path

from copilot import CopilotClient, Tool, ToolInvocation, ToolResult, ToolSet
from copilot.session import PermissionHandler
from copilot.session_events import (
    AssistantMessageData,
    AssistantReasoningData,
    AssistantUsageData,
    SessionCompactionCompleteData,
    SessionErrorData,
    SessionEvent,
    SessionWarningData,
    ToolExecutionStartData,
)

from pokemon_agent.agents.base import BaseAgent
from pokemon_agent.agents.copilot_config import CopilotAgentConfig
from pokemon_agent.agents.prompts import system_prompt
from pokemon_agent.runs import copy_tree

logger = logging.getLogger(__name__)

# Runtime lock files inside a session folder; never copied into or out of checkpoints.
SESSION_RUNTIME_FILES = ("*.lock", "*.hold")

# Our memory tools reuse these Copilot built-in names; the SDK requires an explicit override to register them.
COPILOT_BUILTIN_TOOL_NAMES = {"view", "create", "edit"}



def _quiet_unknown_event_errors(loop: asyncio.AbstractEventLoop, context: dict):
    # The bundled runtime can emit event fields newer than the SDK's parser accepts; those events are dropped either
    # way, so log one line instead of a full traceback.
    if isinstance(context.get("exception"), AssertionError) and "handle_notification" in repr(context.get("handle")):
        logger.debug(f"Ignored unparseable Copilot session event: {context.get('handle')!r:.300}")
        return
    loop.default_exception_handler(context)


class CopilotAgent(BaseAgent):
    """Agent that delegates the model/tool loop and context management to a Copilot SDK session.

    Each step sends the current game state and screenshot as one message. The model can call tools freely within the
    step; tools with `ends_turn` (game actions) end it, so the model always acts on a fresh screenshot.
    """


    def __init__(self, rom_path, config: CopilotAgentConfig, **kwargs):
        """Initialize the Copilot SDK agent.

        Args:
            rom_path: Path to the ROM file
            config: Copilot SDK model and session settings
            **kwargs: BaseAgent options (headless, sound, save_every, load_state, web_output_dir)
        """
        super().__init__(rom_path, **kwargs)
        self.config = config
        self.system_prompt = system_prompt(config.model)
        # In a durable run, the SDK's session lives inside the run folder under a stable ID so it can be resumed.
        self.copilot_home = self.run_store.root / "copilot-sdk" if self.run_store else Path(config.copilot_home)
        self.session_id = self.run_store.run_id if self.run_store else None
        self.resume_session = False
        if self.resume_checkpoint:
            self.restore_agent_state(self.resume_checkpoint / "agent")
        logger.info(f"Using Copilot SDK with GitHub Copilot model {config.model or 'default'!r}")
        self.emit("run", agent="copilot", model=config.model or "Copilot default", provider="GitHub Copilot")

    def _sdk_tools(self) -> list[Tool]:
        def make_handler(name: str):
            def handler(invocation: ToolInvocation) -> ToolResult:
                result = self.run_tool(name, invocation.arguments or {})
                self._save_web_screenshot()
                return ToolResult(text_result_for_llm=result)

            return handler

        return [
            Tool(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                handler=make_handler(tool.name),
                skip_permission=True,
                is_terminal=tool.ends_turn,
                overrides_built_in_tool=tool.name in COPILOT_BUILTIN_TOOL_NAMES,
            )
            for tool in self.tool_factory.tools.values()
        ]

    def _step_message(self, prefix_content: str) -> tuple[str, list[dict]]:
        """Build the step prompt text and screenshot attachment from the current game state."""
        observation = self.observe()
        prompt = "\n".join([prefix_content, *self.game_state_sections(observation, self.memory_listing())])
        attachments = [
            {"type": "blob", "data": observation.screenshot_b64, "mimeType": "image/png", "displayName": "screenshot"}
        ]
        return prompt, attachments

    def _log_event(self, event: SessionEvent):
        match event.data:
            case AssistantMessageData(content=content) if content:
                logger.info(f"[Text] {content}")
                self.emit("thought", text=content, step=self.steps_completed + 1)
            case AssistantReasoningData(content=content) if content:
                self.emit("thought", text=content, reasoning=True, step=self.steps_completed + 1)
            case ToolExecutionStartData(tool_name=name, arguments=arguments):
                logger.info(f"[Tool] Using tool {name} with args {arguments}")
            case AssistantUsageData() as usage:
                logger.info(f"Agent response usage: input={usage.input_tokens} output={usage.output_tokens}")
                self.emit("usage", input=usage.input_tokens, output=usage.output_tokens)
            case SessionCompactionCompleteData() as compaction:
                logger.info(f"[Agent] Context compacted: success={compaction.success}")
                self.emit(
                    "summary",
                    text=compaction.summary_content or "",
                    before=compaction.pre_compaction_tokens,
                    after=compaction.post_compaction_tokens,
                )
            case SessionWarningData(message=message):
                logger.warning(f"[Copilot] {message}")
            case SessionErrorData(message=message):
                logger.error(f"[Copilot] {message}")

    # ----- durable conversation -----------------------------------------------------------------------------------

    def _session_dir(self) -> Path:
        return self.copilot_home / "session-state" / str(self.session_id)

    def write_agent_state(self, directory: Path):
        if self.session_id and self._session_dir().is_dir():
            copy_tree(self._session_dir(), directory / "copilot-session", ignore_patterns=SESSION_RUNTIME_FILES)

    def restore_agent_state(self, directory: Path):
        snapshot = directory / "copilot-session"
        if not snapshot.is_dir():
            logger.warning("Checkpoint has no Copilot session; starting a new conversation")
            return
        copy_tree(snapshot, self._session_dir(), ignore_patterns=SESSION_RUNTIME_FILES)
        self.resume_session = True
        logger.info(f"Restored Copilot session {self.session_id}")

    async def _run_async(self, num_steps: int):
        asyncio.get_running_loop().set_exception_handler(_quiet_unknown_event_errors)

        session_options = dict(
            on_permission_request=PermissionHandler.approve_all,
            model=self.config.model,
            system_message={"mode": "replace", "content": self.system_prompt},
            available_tools=ToolSet().add_custom("*"),
            tools=self._sdk_tools(),
            reasoning_effort=self.config.reasoning_effort,
        )
        async with CopilotClient(mode="empty", base_directory=str(self.copilot_home)) as client:
            if self.resume_session:
                open_session = client.resume_session(self.session_id, **session_options)
                prefix_content = "Resumed from a saved checkpoint. Time to take your next action!"
            else:
                open_session = client.create_session(session_id=self.session_id, **session_options)
                prefix_content = "You may now begin playing.\n"
            async with await open_session as session:
                session.on(self._log_event)
                while self.running and self.steps_completed < num_steps:
                    prompt, attachments = self._step_message(prefix_content)
                    await session.send_and_wait(
                        prompt, attachments=attachments, timeout=self.config.turn_timeout_seconds
                    )
                    self.steps_completed += 1
                    self._on_step_complete(self.steps_completed, num_steps)
                    prefix_content = "Time to take your next action!"

    def run(self, num_steps=1):
        """Main agent loop.

        Args:
            num_steps: Number of steps to run for
        """
        logger.info(f"Running until step {num_steps} (starting at {self.steps_completed})")
        self.emit("run_start", steps=num_steps, resumed_from=self.steps_completed or None)
        try:
            asyncio.run(self._run_async(num_steps))
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, stopping")
            self.running = False
        except Exception as e:
            logger.error("Error in agent loop.", exc_info=True)
            self.save_state_after_failure()
            raise e

        self.save_state()
        self.emit("run_end", steps=self.steps_completed)
        return self.steps_completed
