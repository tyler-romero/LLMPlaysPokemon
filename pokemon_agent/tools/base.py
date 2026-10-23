from pokemon_agent.game.emulator import Emulator


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

    def execute(self, tool_input: dict) -> str:
        """Run the tool with parsed arguments and return the result text for the model."""
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
