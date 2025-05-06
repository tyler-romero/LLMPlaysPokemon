import base64
import copy
import io
import json
import logging
import os

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
)
from PIL import Image, ImageDraw

from agent.emulator import Emulator
from agent.prompts import SUMMARY_PROMPT, SYSTEM_PROMPT
from agent.tools import ToolFactory
from config import MAX_TOKENS, MODEL_NAME, TEMPERATURE

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def pretty_print_messages(messages: list[ChatCompletionMessageParam]):
    for message in messages:
        if isinstance(message, dict):
            # Handle dictionary-style messages
            if message["role"] == "user":
                content = message["content"]
                if isinstance(content, list):
                    for part in content:
                        if part.get("type") == "text":
                            logger.info(f"[User] {part['text']}")
                        elif part.get("type") == "image_url":
                            logger.info("[User] [Screenshot]")
                else:
                    logger.info(f"[User] {content}")
            elif message["role"] == "assistant":
                logger.info(f"[Assistant] {message.get('content', '')}")
                if tool_calls := message.get("tool_calls"):
                    for tool_call in tool_calls:
                        tool_name = tool_call.get("function", {}).get("name", "unknown")
                        tool_args = tool_call.get("function", {}).get("arguments", "{}")
                        logger.info(f"[Tool Call] {tool_name}({tool_args})")
            elif message["role"] == "tool":
                logger.info(f"[Tool Response] {message.get('content', '')}")
            else:
                logger.info(f"[{message['role'].capitalize()}] {message.get('content', '')}")
        else:
            # Handle ChatCompletionMessage objects
            if message.role == "user":
                content = message.content
                if isinstance(content, list):
                    for part in content:
                        if part.type == "text":
                            logger.info(f"[User] {part.text}")
                        elif part.type == "image_url":
                            logger.info("[User] [Screenshot]")
                else:
                    logger.info(f"[User] {content}")
            elif message.role == "assistant":
                logger.info(f"[Assistant] {message.content or ''}")
                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name if hasattr(tool_call, "function") else "unknown"
                        tool_args = tool_call.function.arguments if hasattr(tool_call, "function") else "{}"
                        logger.info(f"[Tool Call] {tool_name}({tool_args})")
            elif message.role == "tool":
                logger.info(f"[Tool Response] {message.content or ''}")
            else:
                logger.info(f"[{message.role.capitalize()}] {message.content or ''}")


def get_screenshot_base64(emulator: Emulator, screenshot: Image.Image, upscale=1):
    """Convert PIL image to base64 string."""
    if upscale > 1:
        new_size = (screenshot.width * upscale, screenshot.height * upscale)
        screenshot = screenshot.resize(new_size)

    if not emulator.get_in_combat():
        shape = screenshot.size
        # draw grid lines
        for x in range(0, shape[0], shape[0] // 10):
            ImageDraw.Draw(screenshot).line(((x, 0), (x, shape[1] - 1)), fill=(255, 0, 0))
        for y in range(0, shape[1], shape[1] // 9):
            ImageDraw.Draw(screenshot).line(((0, y), (shape[0] - 1, y)), fill=(255, 0, 0))

        # # add coordinate labels
        # # Add grid coordinates with nested loops (10 columns x 9 rows)
        # for col in range(10):
        #     x = col * (shape[0] // 10)
        #     for row in range(9):
        #         y = row * (shape[1] // 9)
        #         # Draw (col, row) at every cell
        #         ImageDraw.Draw(screenshot).text((x + 2, y + 2), f"({col}, {row})", fill=(255, 0, 0))

    screenshot.save("screenshot_test.png")
    buffered = io.BytesIO()
    screenshot.save(buffered, format="PNG")
    return base64.standard_b64encode(buffered.getvalue()).decode()


class SimpleAgent:
    def __init__(self, rom_path, headless=True, sound=False, max_history=60, load_state=None, web_output_dir=None):
        """Initialize the simple agent.

        Args:
            rom_path: Path to the ROM file
            headless: Whether to run without display
            sound: Whether to enable sound
            max_history: Maximum number of messages in history before summarization
            load_state: Path to a saved state file to load
            web_output_dir: Directory to save latest.png for web viewing
        """
        self.emulator = Emulator(rom_path, headless, sound)
        self.emulator.initialize()  # Initialize the emulator
        self.tool_factory = ToolFactory(self.emulator)

        if MODEL_NAME.startswith("gemini-"):
            self.client = OpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.getenv("GEMINI_API_KEY")
            )
        elif MODEL_NAME.startswith("gpt-"):
            self.client = OpenAI()
        else:
            # Default to local vLLM server for other models
            self.client = OpenAI(base_url="http://0.0.0.0:8000/v1", api_key="dummy")

        self.running = True
        self.max_history = max_history
        self.web_output_dir = web_output_dir

        if load_state:
            logger.warning(f"Loading saved state from {load_state}")
            try:
                self.emulator.load_state(load_state)
                logger.info(f"Successfully loaded state from {load_state}")
            except Exception as e:
                logger.error(f"Failed to load state from {load_state}: {e}")
                raise RuntimeError(f"Could not load saved state: {e}")

        self._save_web_screenshot()

        screenshot_b64, memory_info, collision_map = self._get_post_action_data()
        self.message_history = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You may now begin playing.\n"},
                    {"type": "text", "text": "Here is a screenshot of the screen:\n"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                    {
                        "type": "text",
                        "text": f"\nGame state information from memory:\n{json.dumps(memory_info, indent=2)}",
                    },
                ],
            }
        ]

    def _get_post_action_data(self):
        """Get screenshot and game state after an action."""
        screenshot = self.emulator.get_screenshot()
        screenshot_b64 = get_screenshot_base64(self.emulator, screenshot, upscale=2)
        memory_info, location, coords = self.emulator.get_state_from_memory()
        collision_map = self.emulator.get_collision_map()

        logger.info("[Memory State after action]")
        logger.info(memory_info)
        if collision_map:
            logger.info(f"[Collision Map after action]\n{collision_map}")

        return screenshot_b64, memory_info, collision_map

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

    def run(self, num_steps=1):
        """Main agent loop.

        Args:
            num_steps: Number of steps to run for
        """
        logger.info(f"Starting agent loop for {num_steps} steps")
        steps_completed = 0
        while self.running and steps_completed < num_steps:
            try:
                messages: list[ChatCompletionMessageParam] = copy.deepcopy(self.message_history)

                logger.info(f"---------- Messages ({len(messages)}) -----------")
                pretty_print_messages(messages)
                logger.info("-------------------------------------------")

                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    max_tokens=MAX_TOKENS,
                    messages=messages,
                    tools=self.tool_factory.get_available_tools(),
                    temperature=TEMPERATURE,
                )
                logger.info(f"Agent response usage: {response.usage}")

                # Add the assistant's message to the message history
                assistant_message: ChatCompletionMessage = response.choices[0].message
                self.message_history.append(assistant_message)  # type: ignore

                # Display the model's reasoning
                if assistant_message.content:
                    logger.info(f"[Text] {assistant_message.content}")
                else:
                    logger.info("[No Text] No text response from model")

                # Extract tool calls
                tool_calls = assistant_message.tool_calls or []
                for tool_call in tool_calls:
                    logger.info(f"[Tool] Using tool {tool_call.function.name} with args {tool_call.function.arguments}")
                    tool_call_result = self.tool_factory.process_tool_call(tool_call)
                    self.message_history.append(tool_call_result)
                    self._save_web_screenshot()
                if not tool_calls:
                    logger.info("[No Tool] No tool calls made")

                # Add user message with screenshot and memory info
                screenshot_b64, memory_info, collision_map = self._get_post_action_data()
                self.message_history.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Time to take your next action!\n"},
                            {"type": "text", "text": "Here is a screenshot of the screen:\n"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                            {
                                "type": "text",
                                "text": f"\nGame state information from memory:\n{json.dumps(memory_info, indent=2)}",
                            },
                        ],
                    }
                )

                # Check if we need to summarize the history
                if len(self.message_history) >= self.max_history:
                    self.summarize_history()

                steps_completed += 1
                logger.info(f"Completed step {steps_completed}/{num_steps}")

            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, stopping")
                self.running = False
            except Exception as e:
                logger.error("Error in agent loop.", exc_info=True)
                raise e

        if not self.running:
            self.emulator.save_state()
            self.emulator.stop()
        else:
            self._save_web_screenshot()

        return steps_completed

    def summarize_history(self):
        """Generate a summary of the conversation history and replace the history with just the summary."""
        logger.info("[Agent] Generating conversation summary...")

        messages: list[ChatCompletionMessageParam] = copy.deepcopy(self.message_history)
        messages.append({"role": "user", "content": [{"type": "text", "text": SUMMARY_PROMPT}]})
        system_message: ChatCompletionSystemMessageParam = {"role": "system", "content": SYSTEM_PROMPT}
        summary_api_messages = [system_message] + messages

        logger.info(f"Calling model {MODEL_NAME} for summarization with {len(summary_api_messages)} messages")
        response = self.client.chat.completions.create(
            model=MODEL_NAME, max_tokens=MAX_TOKENS, messages=summary_api_messages, temperature=TEMPERATURE
        )

        logger.info(f"Summary response usage: {response.usage}")
        summary_text = response.choices[0].message.content or "Summary generation failed."
        logger.info("[Agent] Game Progress Summary:")
        logger.info(f"{summary_text}")

        summary_screenshot = self.emulator.get_screenshot()
        summary_screenshot_b64 = get_screenshot_base64(self.emulator, summary_screenshot, upscale=2)

        # Seed the new history after summarization
        self.message_history: list[ChatCompletionMessageParam] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"CONVERSATION HISTORY SUMMARY (representing {self.max_history} previous messages): {summary_text}",
                    },
                    {"type": "text", "text": "\nCurrent game screenshot for reference:\n"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{summary_screenshot_b64}"}},
                    {
                        "type": "text",
                        "text": "\nYou were just asked to summarize your playthrough so far, which is the summary you see above. You may now continue playing by selecting your next action.",
                    },
                ],
            }
        ]

        logger.info("[Agent] Message history condensed into summary.")
        self._save_web_screenshot()

    def stop(self):
        """Stop the agent."""
        self.running = False
        self.emulator.stop()


if __name__ == "__main__":
    # Get the ROM path relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rom_path = os.path.join(os.path.dirname(current_dir), "pokemon.gb")

    # Create and run agent
    agent = SimpleAgent(rom_path)

    try:
        steps_completed = agent.run(num_steps=10)
        logger.info(f"Agent completed {steps_completed} steps")
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, stopping")
    finally:
        agent.stop()
