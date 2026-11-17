"""The `subagent` tool: ask a read-only helper (a fresh model session) to analyze something for the main agent."""

from pokemon_agent.tools.base import Tool, ToolError


class SubagentTool(Tool):
    name = "subagent"
    description = (
        "Ask a helper to analyze something for you; its answer comes back as this tool's result, within your turn. "
        "The helper is the same kind of model as you but starts fresh: it gets only your task, the current "
        "screenshot and game state, and your memory file list. It can read your memory files and search your "
        "history (including old screenshots), but it can't play the game or change anything. It doesn't end your "
        "turn and uses no game turns, so it is cheap compared with game actions, and only its answer enters your "
        "context, however much it reads. Use it often: whenever you aren't making clear progress, before a long or "
        "risky plan, after a surprising result, or when unsure what to do next. "
        "Write the task as a question or goal."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The question or goal for the helper.",
            },
        },
        "required": ["task"],
        "additionalProperties": False,
    }

    def execute(self, tool_input: dict) -> str:
        # Agents that support helpers (CopilotAgent) run this tool themselves, since the helper is a model session.
        return ToolError("Error: no helper is available for this agent.")
