from dataclasses import dataclass, field

from pokemon_agent.game.emulator import Emulator


class ToolError(str):
    """A tool result reporting that the call failed; its text is what the model reads. Tools return one instead of
    plain text whenever a call fails, so agents never have to guess failure from the wording. A game action returns
    one only when nothing happened in the game (nothing was pressed or moved), so the model can simply retry."""


# Results in logs written before tools returned ToolError began with one of these when the call failed (history.py
# reads such logs).
TOOL_FAILURE_PREFIXES = (
    "error",
    "navigation failed",
    "navigation stopped",
    "invalid",
    "no match found",
    "multiple matches found",
    "path already exists",
    "not run",
)


@dataclass
class ToolOutput:
    """Text for the model plus base64 PNG images to show with it (most tools return plain text instead)."""

    text: str
    images: list[str] = field(default_factory=list)


class Tool:
    """Base class for all tools."""

    name: str = "base_tool"
    description: str = "Base tool description"
    parameters: dict = {}
    # True for tools that act on the game. Agents that let the model chain tool calls on their own (CopilotAgent) end
    # the model's turn after these, so it sees a fresh screenshot before acting again.
    ends_turn: bool = False

    def __init__(self, emulator: Emulator):
        """Initialize the tool with an emulator instance."""
        self.emulator = emulator
        # Set by the agent to publish observer events (kind, **data); a no-op without an observer.
        self.emit = lambda kind, **data: None

    def execute(self, tool_input: dict) -> "str | ToolOutput":
        """Run the tool with parsed arguments and return the result for the model: text, or a ToolOutput with images."""
        raise NotImplementedError

    def get_tool_definition(self):
        """Generate the tool definition in OpenAI function-tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
