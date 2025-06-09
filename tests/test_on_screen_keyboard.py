import json
import os
import sys
import types
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

openai_module = types.ModuleType("openai")
openai_types = types.ModuleType("openai.types")
openai_chat = types.ModuleType("openai.types.chat")

class ChatCompletionMessageToolCall:
    def __init__(self, id, function):
        self.id = id
        self.function = function


ChatCompletionToolMessageParam = dict

openai_chat.ChatCompletionMessageToolCall = ChatCompletionMessageToolCall
openai_chat.ChatCompletionToolMessageParam = ChatCompletionToolMessageParam
openai_types.chat = openai_chat
openai_module.types = openai_types
sys.modules.setdefault("openai", openai_module)
sys.modules.setdefault("openai.types", openai_types)
sys.modules.setdefault("openai.types.chat", openai_chat)

emulator_module = types.ModuleType("agent.emulator")
class Emulator:
    pass
emulator_module.Emulator = Emulator
sys.modules.setdefault("agent.emulator", emulator_module)

from agent.tools import OnScreenKeyboardTool


class DummyEmulator:
    def __init__(self):
        self.presses = []

    def press_buttons(self, buttons, wait=True):
        self.presses.append((list(buttons), wait))


def make_tool_call(arguments: dict):
    return SimpleNamespace(
        id="test_call",
        function=SimpleNamespace(arguments=json.dumps(arguments)),
    )


def test_keyboard_default_end():
    emulator = DummyEmulator()
    tool = OnScreenKeyboardTool(emulator)
    tool_call = make_tool_call({"text": "A"})

    tool.process(tool_call)

    # expected sequence: select 'A', move to END, press 'A' again
    expected = []
    # select 'A'
    expected.append((["a"], True))
    # move right 8 times to column 8
    expected.extend(((["right"], True) for _ in range(8)))
    # move down 4 times to row 4
    expected.extend(((["down"], True) for _ in range(4)))
    # press 'a' to select END
    expected.append((["a"], True))

    assert emulator.presses == expected
