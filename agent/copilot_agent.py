"""Agent built on the GitHub Copilot SDK, which runs the model/tool loop and compacts context on its own."""

import asyncio
import logging
from dataclasses import dataclass

from copilot import CopilotClient, Tool, ToolInvocation, ToolResult, ToolSet
from copilot.session import PermissionHandler
from copilot.session_events import (
    AssistantMessageData,
    AssistantUsageData,
    SessionCompactionCompleteData,
    SessionErrorData,
    SessionEvent,
    SessionWarningData,
    ToolExecutionStartData,
)

from agent.base_agent import BaseAgent, create_user_message_with_game_state
from agent.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Our memory tools reuse these Copilot built-in names; the SDK requires an explicit override to register them.
COPILOT_BUILTIN_TOOL_NAMES = {"view", "create", "edit"}

@dataclass(frozen=True)
class CopilotAgentConfig:
    """Settings for CopilotAgent, which uses GitHub Copilot-hosted models with your Copilot login.

    model: GitHub Copilot model ID (the model must support image input); None uses Copilot's default model. Usually set
        per run with `main.py --model`.
    reasoning_effort: "low", "medium", "high", or "xhigh"; None uses the model's default.
    turn_timeout_seconds: Maximum time for one step (model calls plus tool execution).
    copilot_home: Directory for Copilot session state and logs.
    """

    model: str | None = None
    reasoning_effort: str | None = None
    turn_timeout_seconds: float = 600.0
    copilot_home: str = ".copilot-sdk"


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

    system_prompt = SYSTEM_PROMPT

    def __init__(self, rom_path, config: CopilotAgentConfig, **kwargs):
        """Initialize the Copilot SDK agent.

        Args:
            rom_path: Path to the ROM file
            config: Copilot SDK model and session settings
            **kwargs: BaseAgent options (headless, sound, save_every, load_state, web_output_dir)
        """
        super().__init__(rom_path, **kwargs)
        self.config = config
        self.steps_completed = 0
        logger.info(f"Using Copilot SDK with GitHub Copilot model {config.model or 'default'!r}")

    def _sdk_tools(self) -> list[Tool]:
        def make_handler(name: str):
            def handler(invocation: ToolInvocation) -> ToolResult:
                result = self.tool_factory.execute_tool(name, invocation.arguments or {})
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
        message = create_user_message_with_game_state(
            prefix_content=prefix_content,
            memory_info=observation.memory_info,
            collision_map=observation.collision_map,
            memory_listing=self.memory_listing(),
        )
        prompt = "\n".join(part["text"] for part in message["content"])
        attachments = [
            {"type": "blob", "data": observation.screenshot_b64, "mimeType": "image/png", "displayName": "screenshot"}
        ]
        return prompt, attachments

    def _log_event(self, event: SessionEvent):
        match event.data:
            case AssistantMessageData(content=content) if content:
                logger.info(f"[Text] {content}")
            case ToolExecutionStartData(tool_name=name, arguments=arguments):
                logger.info(f"[Tool] Using tool {name} with args {arguments}")
            case AssistantUsageData() as usage:
                logger.info(f"Agent response usage: input={usage.input_tokens} output={usage.output_tokens}")
            case SessionCompactionCompleteData() as compaction:
                logger.info(f"[Agent] Context compacted: success={compaction.success}")
            case SessionWarningData(message=message):
                logger.warning(f"[Copilot] {message}")
            case SessionErrorData(message=message):
                logger.error(f"[Copilot] {message}")

    async def _run_async(self, num_steps: int):
        asyncio.get_running_loop().set_exception_handler(_quiet_unknown_event_errors)

        async with CopilotClient(mode="empty", base_directory=self.config.copilot_home) as client:
            async with await client.create_session(
                on_permission_request=PermissionHandler.approve_all,
                model=self.config.model,
                system_message={"mode": "replace", "content": self.system_prompt},
                available_tools=ToolSet().add_custom("*"),
                tools=self._sdk_tools(),
                reasoning_effort=self.config.reasoning_effort,
            ) as session:
                session.on(self._log_event)
                prefix_content = "You may now begin playing.\n"
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
        logger.info(f"Starting agent loop for {num_steps} steps")
        try:
            asyncio.run(self._run_async(num_steps))
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, stopping")
            self.running = False
        except Exception as e:
            logger.error("Error in agent loop.", exc_info=True)
            raise e  # skips saving state, handled in main.py

        self.save_state()
        return self.steps_completed
