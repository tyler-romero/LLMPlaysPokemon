import base64
import copy
import io
import json
import logging
import os
import typing

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)

from agent.emulator import Emulator
from config import MAX_TOKENS, MODEL_NAME, TEMPERATURE, USE_NAVIGATOR

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_screenshot_base64(screenshot, upscale=1):
    """Convert PIL image to base64 string."""
    # Resize if needed
    if upscale > 1:
        new_size = (screenshot.width * upscale, screenshot.height * upscale)
        screenshot = screenshot.resize(new_size)

    # Convert to base64
    buffered = io.BytesIO()
    screenshot.save(buffered, format="PNG")
    return base64.standard_b64encode(buffered.getvalue()).decode()


SYSTEM_PROMPT = """You are playing Pokemon Red. You can see the game screen and control the game by executing emulator commands.

Your goal is to play through Pokemon Red and eventually defeat the Elite Four. Make decisions based on what you see on the screen.

Before each action, explain your reasoning briefly, then use the emulator tool to execute your chosen commands.

The conversation history may occasionally be summarized to save context space. If you see a message labeled "CONVERSATION HISTORY SUMMARY", this contains the key information about your progress so far. Use this information to maintain continuity in your gameplay."""

SUMMARY_PROMPT = """I need you to create a detailed summary of our conversation history up to this point. This summary will replace the full conversation history to manage the context window.

Please include:
1. Key game events and milestones you've reached
2. Important decisions you've made
3. Current objectives or goals you're working toward
4. Your current location and Pokémon team status
5. Any strategies or plans you've mentioned

The summary should be comprehensive enough that you can continue gameplay without losing important context about what has happened so far."""


AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "press_buttons",
            "description": "Press a sequence of buttons on the Game Boy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "buttons": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "a",
                                "b",
                                "start",
                                "select",
                                "up",
                                "down",
                                "left",
                                "right",
                            ],
                        },
                        "description": "List of buttons to press in sequence. Valid buttons: 'a', 'b', 'start', 'select', 'up', 'down', 'left', 'right'",
                    },
                    "wait": {
                        "type": "boolean",
                        "description": "Whether to wait for a brief period after pressing each button. Defaults to true.",
                    },
                },
                "required": ["buttons"],
                "additionalProperties": False,
            },
        },
    }
]

if USE_NAVIGATOR:
    AVAILABLE_TOOLS.append(
        {
            "type": "function",
            "function": {
                "name": "navigate_to",
                "description": "Automatically navigate to a position on the map grid. The screen is divided into a 9x10 grid, with the top-left corner as (0, 0). This tool is only available in the overworld.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "row": {
                            "type": "integer",
                            "description": "The row coordinate to navigate to (0-8).",
                        },
                        "col": {
                            "type": "integer",
                            "description": "The column coordinate to navigate to (0-9).",
                        },
                    },
                    "required": ["row", "col"],
                    "additionalProperties": False,
                },
            },
        }
    )


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
        # self.client = OpenAI(base_url="http://0.0.0.0:8000/v1", api_key="dummy")
        self.client = OpenAI()
        self.running = True
        self.message_history = [{"role": "user", "content": "You may now begin playing."}]
        self.max_history = max_history
        self.web_output_dir = web_output_dir
        if load_state:
            logger.info(f"Loading saved state from {load_state}")
            self.emulator.load_state(load_state)

        # Save initial screenshot if web output is enabled
        if self.web_output_dir:
            self._save_web_screenshot()

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

    def process_tool_call(self, tool_call):
        """Process a single tool call."""
        tool_name = tool_call.function.name
        tool_input = json.loads(tool_call.function.arguments)
        logger.info(f"Processing tool call: {tool_name}")

        if tool_name == "press_buttons":
            buttons = tool_input["buttons"]
            wait = tool_input.get("wait", True)
            logger.info(f"[Buttons] Pressing: {buttons} (wait={wait})")

            result = self.emulator.press_buttons(buttons, wait)

            # Get a fresh screenshot after executing the buttons
            screenshot = self.emulator.get_screenshot()
            screenshot_b64 = get_screenshot_base64(screenshot, upscale=2)

            # Save screenshot for web viewing if enabled
            if self.web_output_dir:
                self._save_web_screenshot()

            # Get game state from memory after the action
            memory_info = self.emulator.get_state_from_memory()

            # Log the memory state after the tool call
            logger.info("[Memory State after action]")
            logger.info(memory_info)

            collision_map = self.emulator.get_collision_map()
            if collision_map:
                logger.info(f"[Collision Map after action]\n{collision_map}")

            # Return tool result as a dictionary
            return {
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": [
                    {"type": "text", "text": f"Pressed buttons: {', '.join(buttons)}"},
                    {
                        "type": "text",
                        "text": "\nHere is a screenshot of the screen after your button presses:",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                    },
                    {
                        "type": "text",
                        "text": f"\nGame state information from memory after your action:\n{memory_info}",
                    },
                ],
            }
        elif tool_name == "navigate_to":
            row = tool_input["row"]
            col = tool_input["col"]
            logger.info(f"[Navigation] Navigating to: ({row}, {col})")

            status, path = self.emulator.find_path(row, col)
            if path:
                for direction in path:
                    self.emulator.press_buttons([direction], True)
                result = f"Navigation successful: followed path with {len(path)} steps"
            else:
                result = f"Navigation failed: {status}"

            # Get a fresh screenshot after executing the navigation
            screenshot = self.emulator.get_screenshot()
            screenshot_b64 = get_screenshot_base64(screenshot, upscale=2)

            # Save screenshot for web viewing if enabled
            if self.web_output_dir:
                self._save_web_screenshot()

            # Get game state from memory after the action
            memory_info = self.emulator.get_state_from_memory()

            # Log the memory state after the tool call
            logger.info("[Memory State after action]")
            logger.info(memory_info)

            collision_map = self.emulator.get_collision_map()
            if collision_map:
                logger.info(f"[Collision Map after action]\n{collision_map}")

            # Return tool result as a dictionary
            return {
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": [
                    {"type": "text", "text": f"Navigation result: {result}"},
                    {
                        "type": "text",
                        "text": "\nHere is a screenshot of the screen after navigation:",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                    },
                    {
                        "type": "text",
                        "text": f"\nGame state information from memory after your action:\n{memory_info}",
                    },
                ],
            }
        else:
            logger.error(f"Unknown tool called: {tool_name}")
            return {
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": [{"type": "text", "text": f"Error: Unknown tool '{tool_name}'"}],
            }

    def run(self, num_steps=1):
        """Main agent loop.

        Args:
            num_steps: Number of steps to run for
        """
        logger.info(f"Starting agent loop for {num_steps} steps")

        steps_completed = 0
        while self.running and steps_completed < num_steps:
            try:
                messages: typing.List[ChatCompletionMessageParam] = copy.deepcopy(self.message_history)

                # Mark screenshots as ephemeral to save context space
                if len(messages) >= 3:
                    last_msg = messages[-1]
                    if last_msg["role"] == "user" and isinstance(last_msg["content"], list) and last_msg["content"]:
                        content_list = last_msg["content"]
                        if content_list and isinstance(content_list[-1], dict):
                            content_list[-1]["cache_control"] = {"type": "ephemeral"}

                    if len(messages) >= 5:
                        third_last_msg = messages[-3]
                        if (
                            third_last_msg["role"] == "user"
                            and isinstance(third_last_msg["content"], list)
                            and third_last_msg["content"]
                        ):
                            content_list = third_last_msg["content"]
                            if content_list and isinstance(content_list[-1], dict):
                                content_list[-1]["cache_control"] = {"type": "ephemeral"}

                # Cast tools before sending to API
                api_tools = typing.cast(typing.List[ChatCompletionToolParam], AVAILABLE_TOOLS)

                # logger.info(f"Calling model {MODEL_NAME} with {len(messages)} messages")
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    max_tokens=MAX_TOKENS,
                    messages=messages,
                    tools=api_tools,
                    temperature=TEMPERATURE,
                )

                logger.info(f"Run response usage: {response.usage}")

                # Extract tool calls
                tool_calls = []
                if response.choices and response.choices[0].message.tool_calls:
                    tool_calls = response.choices[0].message.tool_calls

                # Display the model's reasoning
                if response.choices and response.choices[0].message.content:
                    logger.info(f"[Text] {response.choices[0].message.content}")

                # Log tool calls if present
                if tool_calls:
                    for tool_call in tool_calls:
                        logger.info(f"[Tool] Using tool: {tool_call.function.name}")
                else:
                    logger.info("[No Tool] No tool calls made")

                # Process tool calls
                if tool_calls:
                    # 1. Add the assistant message correctly (with potential tool_calls attribute)
                    assistant_response_message = response.choices[0].message
                    assistant_message_dict: typing.Dict[str, typing.Any] = {"role": "assistant"}
                    if assistant_response_message.content:
                        assistant_message_dict["content"] = assistant_response_message.content
                    if assistant_response_message.tool_calls:
                        assistant_message_dict["tool_calls"] = assistant_response_message.tool_calls

                    self.message_history.append(
                        typing.cast(ChatCompletionAssistantMessageParam, assistant_message_dict)
                    )

                    # 2. Process tool calls and add 'role: tool' messages
                    rich_results_for_next_user_message = []
                    for tool_call in tool_calls:
                        tool_result_dict = self.process_tool_call(tool_call)
                        self._save_web_screenshot()

                        tool_output_text = "Tool execution finished."
                        result_content_list = tool_result_dict.get("content", [])
                        if (
                            result_content_list
                            and isinstance(result_content_list, list)
                            and len(result_content_list) > 0
                            and result_content_list[0].get("type") == "text"
                        ):
                            tool_output_text = result_content_list[0]["text"] or tool_output_text

                        tool_message: ChatCompletionToolMessageParam = {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_output_text,
                        }
                        self.message_history.append(tool_message)

                        rich_results_for_next_user_message.extend(tool_result_dict.get("content", []))

                    # 3. Add the rich results (screenshots, memory state) as a single 'user' message
                    if rich_results_for_next_user_message:
                        user_message: ChatCompletionUserMessageParam = {
                            "role": "user",
                            "content": rich_results_for_next_user_message,
                        }
                        self.message_history.append(user_message)

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
            self.emulator.stop()
        else:
            self._save_web_screenshot()

        return steps_completed

    def summarize_history(self):
        """Generate a summary of the conversation history and replace the history with just the summary."""
        logger.info("[Agent] Generating conversation summary...")

        messages: list[ChatCompletionMessageParam] = copy.deepcopy(self.message_history)

        # Mark screenshots ephemeral (logic copied from run method, adapt if needed)
        if len(messages) >= 3:
            last_msg = messages[-1]
            if last_msg["role"] == "user" and isinstance(last_msg["content"], list) and last_msg["content"]:
                content_list = last_msg["content"]
                if content_list and isinstance(content_list[-1], dict):
                    content_list[-1]["cache_control"] = {"type": "ephemeral"}
            if len(messages) >= 5:
                third_last_msg = messages[-3]
                if (
                    third_last_msg["role"] == "user"
                    and isinstance(third_last_msg["content"], list)
                    and third_last_msg["content"]
                ):
                    content_list = third_last_msg["content"]
                    if content_list and isinstance(content_list[-1], dict):
                        content_list[-1]["cache_control"] = {"type": "ephemeral"}

        messages.append(
            typing.cast(
                ChatCompletionUserMessageParam,
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": SUMMARY_PROMPT,
                        }
                    ],
                },
            )
        )

        # Construct the system message and combine with others
        system_message: ChatCompletionMessageParam = {"role": "system", "content": SYSTEM_PROMPT}
        summary_api_messages = [system_message] + messages

        logger.info(f"Calling model {MODEL_NAME} for summarization with {len(summary_api_messages)} messages")
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            messages=summary_api_messages,
            temperature=TEMPERATURE,
        )

        logger.info(f"Summary response usage: {response.usage}")
        summary_text = response.choices[0].message.content or "Summary generation failed."
        logger.info("[Agent] Game Progress Summary:")
        logger.info(f"{summary_text}")

        # Get current screenshot B64 for the new history
        summary_screenshot = self.emulator.get_screenshot()
        summary_screenshot_b64 = get_screenshot_base64(summary_screenshot, upscale=2)

        # Construct the new history after summarization
        self.message_history = [
            typing.cast(
                ChatCompletionUserMessageParam,
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"CONVERSATION HISTORY SUMMARY (representing {self.max_history} previous messages): {summary_text}",
                        },
                        {
                            "type": "text",
                            "text": "\n\nCurrent game screenshot for reference:",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{summary_screenshot_b64}"},
                        },
                        {
                            "type": "text",
                            "text": "You were just asked to summarize your playthrough so far, which is the summary you see above. You may now continue playing by selecting your next action.",
                        },
                    ],
                },
            )
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
