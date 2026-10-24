import copy
import json
import logging

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
from pokemon_agent.agents.prompts import SUMMARY_PROMPT, system_prompt
from pokemon_agent.game.observation import GameObservation
from pokemon_agent.llm.client import LLMConfig, LLMResponse, create_llm_adapter

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
            **kwargs: BaseAgent options (headless, sound, save_every, load_state, web_output_dir)
        """
        super().__init__(rom_path, **kwargs)
        self.llm_config = llm_config
        self.system_prompt = system_prompt(llm_config.model)
        self.llm = create_llm_adapter(llm_config)
        self.max_history = max_history
        logger.info(
            f"Using provider {llm_config.provider!r} with model {llm_config.model!r} via the {llm_config.api!r} API"
        )
        self.emit("run", agent="simple", model=llm_config.model, provider=llm_config.provider, api=llm_config.api)

        initial_message = self._game_state_message(
            self.observe(), prefix_content="You may now begin playing.\n", memory_listing=self.memory_listing()
        )
        self.message_history = [initial_message]

    @staticmethod
    def _game_state_message(
        observation: GameObservation,
        prefix_content: str | None = None,
        memory_listing: str | None = None,
        suffix_content: str | None = None,
    ) -> ChatCompletionUserMessageParam:
        """Build a Chat Completions user message with the screenshot and game state."""
        texts = [prefix_content] if prefix_content else []
        content_parts = [{"type": "text", "text": text} for text in texts]
        content_parts.append({"type": "text", "text": "Here is a screenshot of the screen:\n"})
        content_parts.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{observation.screenshot_b64}"}}
        )
        for section in BaseAgent.game_state_sections(observation, memory_listing):
            content_parts.append({"type": "text", "text": section})
        if suffix_content:
            content_parts.append({"type": "text", "text": suffix_content})
        return {"role": "user", "content": content_parts}

    def _execute_tool_call(self, tool_call: ChatCompletionMessageToolCall) -> ChatCompletionToolMessageParam:
        """Execute an OpenAI-format tool call and return the tool result message."""
        result = self.run_tool(tool_call.function.name, json.loads(tool_call.function.arguments))
        return {"role": "tool", "tool_call_id": tool_call.id, "content": result}

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
        logger.info(f"Starting agent loop for {num_steps} steps")
        self.emit("run_start", steps=num_steps)
        steps_completed = 0
        while self.running and steps_completed < num_steps:
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
                for tool_call in tool_calls:
                    logger.info(f"[Tool] Using tool {tool_call.function.name} with args {tool_call.function.arguments}")
                    tool_call_result = self._execute_tool_call(tool_call)
                    self.message_history.append(tool_call_result)  # type: ignore
                    self._save_web_screenshot()
                if not tool_calls:
                    logger.info("[No Tool] No tool calls made")

                # Add new user message with screenshot and memory info
                user_message = self._game_state_message(
                    self.observe(),
                    prefix_content="Time to take your next action!",
                    memory_listing=self.memory_listing(),
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
            except Exception as e:
                logger.error("Error in agent loop.", exc_info=True)
                raise e  # skips saving state, handled in main.py

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
            memory_listing=self.memory_listing(),
            suffix_content=suffix_content,
        )

        # Seed the new history after summarization
        self.message_history = [user_message]

        logger.info("[Agent] Message history condensed into summary.")
        self._save_web_screenshot()
