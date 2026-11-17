"""The `trigger_compaction` tool: the model asks for its conversation to be condensed now (run by CopilotAgent)."""

from pokemon_agent.tools.base import Tool, ToolError


class TriggerCompactionTool(Tool):
    name = "trigger_compaction"
    description = (
        "Condense your conversation history now instead of waiting for it to happen automatically. It takes effect "
        "when this turn ends, and your next turn starts from the summary. How it works: the same kind of model as you "
        "reads the whole conversation and writes a structured summary of it (an overview, what happened, the work "
        "done, technical details, important files, and next steps), and that summary replaces the earlier messages, "
        "including their screenshots. Details it leaves out drop out of your context. Your instructions, tools, and "
        "memory files are unchanged, and 'history' still holds every past turn. Automatic compaction works the same "
        "way when your context reaches the threshold shown in each turn's message. Write anything you need to keep to "
        "memory first. It doesn't end your turn."
    )
    parameters = {
        "type": "object",
        "properties": {
            "custom_instructions": {
                "type": "string",
                "description": "What the summary should focus on or must keep, e.g. 'keep the exact results of "
                "every approach I have tried for my current goal'.",
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, emulator):
        super().__init__(emulator)
        # Set by the agent: called with the custom instructions (or None) to schedule a compaction after the turn.
        self.on_request = None

    def execute(self, tool_input: dict) -> str:
        if self.on_request is None:
            return ToolError("Error: compaction isn't available for this agent.")
        instructions = (tool_input.get("custom_instructions") or "").strip() or None
        self.on_request(instructions)
        focus = " following your instructions" if instructions else ""
        return f"Compaction requested: your conversation will be condensed{focus} when this turn ends."
