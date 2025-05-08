import base64
import copy
import io
import json
import logging
import os

import openai
from openai import NotGiven, OpenAI
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from PIL import Image, ImageDraw, ImageFont
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agent.emulator import Emulator
from agent.prompts import SUMMARY_PROMPT, SYSTEM_PROMPT
from agent.tools import ToolFactory, read_knowledge_base
from config import MAX_TOKENS, MODEL_NAME, TEMPERATURE

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
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
    knowledge_base: str | None = None,
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

    if knowledge_base:
        content_parts.append(
            {
                "type": "text",
                "text": f"\nYour recorded knowledge base:\n{knowledge_base}",
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


class SimpleAgent:
    def __init__(
        self,
        rom_path,
        headless=True,
        sound=False,
        max_history=60,
        save_every=100,
        load_state=None,
        web_output_dir=None,
    ):
        """Initialize the simple agent.

        Args:
            rom_path: Path to the ROM file
            headless: Whether to run without display
            sound: Whether to enable sound
            max_history: Maximum number of messages in history before summarization
            save_every: Number of steps between save_state() calls
            load_state: Path to a saved state file to load
            web_output_dir: Directory to save latest.png for web viewing
        """
        self.emulator = Emulator(rom_path, headless, sound=sound)
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

        self._save_web_screenshot()

        screenshot_b64, memory_info, collision_map = self._get_post_action_data()

        initial_message = create_user_message_with_game_state(
            prefix_content="You may now begin playing.\n",
            screenshot_b64=screenshot_b64,
            memory_info=memory_info,
            collision_map=collision_map,
        )
        self.message_history = [initial_message]

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(max=60),
        retry=retry_if_exception_type(
            (openai.InternalServerError, openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)
        ),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            f"{retry_state.outcome.exception()}: Retrying request to /chat/completions "  # type: ignore
            f"in {retry_state.next_action.sleep} seconds"  # type: ignore
        ),
    )
    def _get_chat_completion(
        self, messages: list[ChatCompletionMessageParam], tools: list[ChatCompletionToolParam] | NotGiven = NotGiven()
    ):
        """Get a chat completion from the OpenAI API, retrying on 5xx errors up to 5 times."""
        return self.client.chat.completions.create(
            messages=messages, tools=tools, model=MODEL_NAME, max_tokens=MAX_TOKENS, temperature=TEMPERATURE
        )

    def _get_post_action_data(self):
        """Get screenshot and game state after an action."""
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
                messages: list[ChatCompletionMessageParam] = copy.deepcopy(self.message_history)  # type: ignore
                response = self._get_chat_completion(messages, self.tool_factory.get_available_tools())
                logger.info(f"Agent response usage: {response.usage}")

                # Add response to message history
                assistant_message: ChatCompletionMessage = response.choices[0].message
                self.message_history.append(assistant_message)  # type: ignore

                # Display the model's reasoning
                if assistant_message.content:
                    logger.info(f"[Text] {assistant_message.content}")
                else:
                    logger.info("[No Text] No text response from model")

                # Process tool calls
                tool_calls = assistant_message.tool_calls or []
                for tool_call in tool_calls:
                    logger.info(f"[Tool] Using tool {tool_call.function.name} with args {tool_call.function.arguments}")
                    tool_call_result = self.tool_factory.process_tool_call(tool_call)
                    self.message_history.append(tool_call_result)  # type: ignore
                    self._save_web_screenshot()
                if not tool_calls:
                    logger.info("[No Tool] No tool calls made")

                # Add new user message with screenshot and memory info
                screenshot_b64, memory_info, collision_map = self._get_post_action_data()
                knowledge_base = read_knowledge_base()
                user_message = create_user_message_with_game_state(
                    prefix_content="Time to take your next action!",
                    screenshot_b64=screenshot_b64,
                    memory_info=memory_info,
                    collision_map=collision_map,
                    knowledge_base=knowledge_base,
                )

                self.message_history.append(user_message)

                # Check if we need to summarize the history
                if len(self.message_history) >= self.max_history:
                    self.summarize_history()

                steps_completed += 1
                logger.info(f"Completed step {steps_completed}/{num_steps}")
                self._save_web_screenshot()

                if steps_completed % self.save_every == 0 and steps_completed > 0:
                    self.emulator.save_state()

            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, stopping")
                self.running = False
            except Exception as e:
                logger.error("Error in agent loop.", exc_info=True)
                raise e  # skips saving state, handled in main.py

        self.emulator.save_state()
        return steps_completed

    def summarize_history(self):
        """Generate a summary of the conversation history and replace the history with just the summary."""
        logger.info("[Agent] Generating conversation summary...")

        messages: list[ChatCompletionMessageParam] = copy.deepcopy(self.message_history)  # type: ignore

        # Add summary prompt with validation to ensure no empty content
        summary_prompt = SUMMARY_PROMPT.strip()
        if summary_prompt:  # Make sure we don't add empty prompt
            messages.append({"role": "user", "content": [{"type": "text", "text": summary_prompt}]})

        system_message: ChatCompletionSystemMessageParam = {"role": "system", "content": SYSTEM_PROMPT}
        summary_api_messages = [system_message] + messages

        logger.info(f"Calling model {MODEL_NAME} for summarization with {len(summary_api_messages)} messages")
        response = self._get_chat_completion(summary_api_messages)

        logger.info(f"Summary response usage: {response.usage}")
        summary_text = response.choices[0].message.content or "Summary generation failed."
        logger.info("[Agent] Game Progress Summary:")
        logger.info(f"{summary_text}")

        screenshot_b64, memory_info, collision_map = self._get_post_action_data()
        knowledge_base = read_knowledge_base()
        prefix_content = (
            f"CONVERSATION HISTORY SUMMARY (representing {self.max_history} previous messages): {summary_text}"
        )
        suffix_content = (
            "\nYou were just asked to summarize your playthrough so far, which is the summary you see above. "
            "You may now continue playing by selecting your next action."
        )
        user_message = create_user_message_with_game_state(
            prefix_content=prefix_content,
            screenshot_b64=screenshot_b64,
            memory_info=memory_info,
            collision_map=collision_map,
            knowledge_base=knowledge_base,
            suffix_content=suffix_content,
        )

        # Seed the new history after summarization
        self.message_history = [user_message]

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
