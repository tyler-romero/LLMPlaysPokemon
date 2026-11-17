import copy
import json
import logging
from pathlib import Path

import openai
from openai import NotGiven
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from pokemon_agent.agents.base import BaseAgent
from pokemon_agent.agents.prompts import (
    AFTER_FAILURE_REFUSAL,
    AFTER_GAME_ACTION_REFUSAL,
    GAME_ACTION_NOT_LAST_REFUSAL,
    RESUME_STEP_PREFIX,
    START_STEP_PREFIX,
    STEP_PREFIX,
    SUMMARY_PROMPT,
    system_prompt,
)
from pokemon_agent.game.observation import GameObservation
from pokemon_agent.llm.client import LLMConfig, LLMResponse, create_llm_adapter
from pokemon_agent.tools.base import ToolError, ToolOutput

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class SimpleAgent(BaseAgent):
    """Agent that drives an OpenAI-compatible LLM (Chat Completions or Responses) with its own history and summaries."""

    summary_prompt = SUMMARY_PROMPT

    def __init__(self, rom_path, llm_config: LLMConfig, max_history=60, **kwargs):
        """Initialize the simple agent.

        Args:
            rom_path: Path to the ROM file
            llm_config: LLM endpoint and sampling settings
            max_history: Maximum number of messages in history before summarization
            **kwargs: BaseAgent options (headless, sound, save_every, load_state, observer, run)
        """
        super().__init__(rom_path, **kwargs)
        self.llm_config = llm_config
        self.system_prompt = system_prompt(llm_config.model, agent="simple")
        self.llm = create_llm_adapter(llm_config)
        self.max_history = max_history
        logger.info(
            f"Using provider {llm_config.provider!r} with model {llm_config.model!r} via the {llm_config.api!r} API"
        )
        self.emit(
            "run",
            agent="simple",
            model=llm_config.model,
            provider=llm_config.provider,
            api=llm_config.api,
            tools=self.tool_factory.get_available_tools(),
        )

        self._history_mark = 0
        self._failed = False
        if self.resume_checkpoint:
            self.restore_agent_state(self.resume_checkpoint / "agent")
        else:
            initial_message = self._game_state_message(self.observe(), prefix_content=START_STEP_PREFIX)
            self.message_history = [initial_message]

    # ----- durable conversation -----------------------------------------------------------------------------------

    def write_agent_state(self, directory: Path):
        # After a crash mid-step, keep only whole steps; the next observation shows whatever the partial step did.
        history = self.message_history[: self._history_mark] if self._failed else self.message_history
        items = [_jsonable(item) for item in history]
        # Screenshots dominate the size; keep only the latest one.
        items = [_without_images(item) for item in items[:-1]] + items[-1:]
        (directory / "conversation.json").write_text(json.dumps({"message_history": items}, default=str))

    def restore_agent_state(self, directory: Path):
        self.message_history = json.loads((directory / "conversation.json").read_text())["message_history"]
        resumed = self._game_state_message(
            self.observe(),
            prefix_content=RESUME_STEP_PREFIX,
        )
        # Replace the checkpoint's last observation with a fresh one (or add one after a partial step).
        if self.message_history and self.message_history[-1].get("role") == "user":
            self.message_history[-1] = resumed
        else:
            self.message_history.append(resumed)
        logger.info(f"Restored {len(self.message_history)} conversation messages")

    @staticmethod
    def _game_state_message(
        observation: GameObservation,
        prefix_content: str | None = None,
        suffix_content: str | None = None,
    ) -> ChatCompletionUserMessageParam:
        """Build a Chat Completions user message with the screenshot and game state."""
        texts = [prefix_content] if prefix_content else []
        content_parts = [{"type": "text", "text": text} for text in texts]
        content_parts.append({"type": "text", "text": "Here is a screenshot of the screen:\n"})
        content_parts.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{observation.screenshot_b64}"}}
        )
        for section in BaseAgent.game_state_sections(observation):
            content_parts.append({"type": "text", "text": section})
        if suffix_content:
            content_parts.append({"type": "text", "text": suffix_content})
        return {"role": "user", "content": content_parts}

    def _execute_tool_call(
        self, tool_call: ChatCompletionMessageToolCall
    ) -> tuple[ChatCompletionToolMessageParam, bool]:
        """Execute an OpenAI-format tool call; return the tool result message and whether the call failed."""
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError as e:
            result = f"Error: tool arguments are not valid JSON ({e}); no action was taken."
            return {"role": "tool", "tool_call_id": tool_call.id, "content": result}, True
        result = self.run_tool(tool_call.function.name, arguments)
        failed = isinstance(result, ToolError)
        if isinstance(result, ToolOutput):
            # Tool messages here can only carry text, so screenshots can't be shown to this agent.
            result = f"{result.text}\n(Images can't be shown in tool results for this agent.)"
        return {"role": "tool", "tool_call_id": tool_call.id, "content": str(result)}, failed

    def _refusal(self, tool_calls: list, index: int, failed: str | None) -> str | None:
        """Why tool_calls[index] must not run, or None: calls run in order, the first failure stops the rest, and a game
        action must be a response's last call."""
        game_actions = {name for name, tool in self.tool_factory.tools.items() if tool.ends_turn}
        if any(call.function.name in game_actions for call in tool_calls[:index]):
            return AFTER_GAME_ACTION_REFUSAL
        if failed:
            return AFTER_FAILURE_REFUSAL.format(failed=failed)
        if tool_calls[index].function.name in game_actions and index != len(tool_calls) - 1:
            return GAME_ACTION_NOT_LAST_REFUSAL
        return None

    def _emit_usage(self, usage):
        input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)
        if input_tokens is not None:
            self.emit("usage", input=input_tokens, output=output_tokens)

    def _system_message(self) -> ChatCompletionSystemMessageParam:
        return {"role": "system", "content": self.system_prompt}

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(max=60),
        retry=retry_if_exception_type(
            (openai.InternalServerError, openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)
        ),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            f"{retry_state.outcome.exception()}: Retrying LLM request "  # type: ignore
            f"in {retry_state.next_action.sleep} seconds"  # type: ignore
        ),
    )
    def _generate(
        self, messages: list[ChatCompletionMessageParam], tools: list[ChatCompletionToolParam] | NotGiven = NotGiven()
    ) -> LLMResponse:
        """Get a model response from the configured API, retrying on transient errors up to 5 times."""
        return self.llm.create(messages, tools)

    def run(self, num_steps=1):
        """Main agent loop.

        Args:
            num_steps: Number of steps to run for
        """
        logger.info(f"Running until step {num_steps} (starting at {self.steps_completed})")
        self.emit("run_start", steps=num_steps, resumed_from=self.steps_completed or None)
        steps_completed = self.steps_completed
        while self.running and steps_completed < num_steps:
            self._history_mark = len(self.message_history)
            try:
                messages: list[ChatCompletionMessageParam] = [self._system_message()] + copy.deepcopy(
                    self.message_history
                )  # type: ignore
                response = self._generate(messages, self.tool_factory.get_available_tools())
                logger.info(f"Agent response usage: {response.usage}")
                self._emit_usage(response.usage)

                # Add response to message history
                self.message_history.extend(response.history_items)

                # Display the model's reasoning
                if response.text:
                    logger.info(f"[Text] {response.text}")
                    self.emit("thought", text=response.text, step=steps_completed + 1)
                else:
                    logger.info("[No Text] No text response from model")

                # Process tool calls
                tool_calls = response.tool_calls
                failed = None
                for index, tool_call in enumerate(tool_calls):
                    name = tool_call.function.name
                    logger.info(f"[Tool] Using tool {name} with args {tool_call.function.arguments}")
                    if refusal := self._refusal(tool_calls, index, failed):
                        self.report_not_run(name, tool_call.function.arguments, refusal)
                        tool_call_result = {"role": "tool", "tool_call_id": tool_call.id, "content": refusal}
                        call_failed = True
                    else:
                        tool_call_result, call_failed = self._execute_tool_call(tool_call)
                    failed = failed or (name if call_failed else None)
                    self.message_history.append(tool_call_result)  # type: ignore
                if not tool_calls:
                    logger.info("[No Tool] No tool calls made")

                # Add new user message with screenshot and memory info
                user_message = self._game_state_message(
                    self.observe(),
                    prefix_content=STEP_PREFIX,
                )
                self.message_history.append(user_message)

                # Check if we need to summarize the history
                if len(self.message_history) >= self.max_history:
                    self.summarize_history()

                steps_completed += 1
                self._on_step_complete(steps_completed, num_steps)

            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, stopping")
                self.running = False
                self._failed = True  # the step was cut short; checkpoint only whole steps
            except Exception as e:
                logger.error("Error in agent loop.", exc_info=True)
                self._failed = True
                self.save_state_after_failure()
                raise e

        self.save_state()
        self.emit("run_end", steps=steps_completed)
        return steps_completed

    def summarize_history(self):
        """Generate a summary of the conversation history and replace the history with just the summary."""
        logger.info("[Agent] Generating conversation summary...")

        messages: list[ChatCompletionMessageParam] = copy.deepcopy(self.message_history)  # type: ignore

        # Add summary prompt with validation to ensure no empty content
        summary_prompt = self.summary_prompt.strip()
        if summary_prompt:  # Make sure we don't add empty prompt
            messages.append({"role": "user", "content": [{"type": "text", "text": summary_prompt}]})

        summary_api_messages = [self._system_message()] + messages

        logger.info(
            f"Calling model {self.llm_config.model} for summarization with {len(summary_api_messages)} messages"
        )
        response = self._generate(summary_api_messages)

        logger.info(f"Summary response usage: {response.usage}")
        summary_text = response.text or "Summary generation failed."
        self.emit("summary", text=summary_text, messages=len(self.message_history))
        logger.info("[Agent] Game Progress Summary:")
        logger.info(f"{summary_text}")

        prefix_content = (
            f"CONVERSATION HISTORY SUMMARY (representing {self.max_history} previous messages): {summary_text}"
        )
        suffix_content = (
            "\nYou were just asked to summarize your playthrough so far, which is the summary you see above. "
            "You may now continue playing by selecting your next action."
        )
        user_message = self._game_state_message(
            self.observe(),
            prefix_content=prefix_content,
            suffix_content=suffix_content,
        )

        # Seed the new history after summarization
        self.message_history = [user_message]

        logger.info("[Agent] Message history condensed into summary.")


def _jsonable(item):
    return item.model_dump(mode="json", exclude_none=True) if hasattr(item, "model_dump") else item


def _without_images(item):
    if not isinstance(item, dict) or not isinstance(item.get("content"), list):
        return item
    parts = [
        {"type": "text", "text": "[earlier screenshot omitted]"} if part.get("type") == "image_url" else part
        for part in item["content"]
    ]
    return {**item, "content": parts}
