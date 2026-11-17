"""The `objective` tool: the model states what it's working toward (long, medium, and short term), shown to viewers."""

from pokemon_agent.tools.base import Tool, ToolError


class ObjectiveTool(Tool):
    name = "objective"
    description = (
        "Record what you are working toward: your long-term goal, your medium-term milestone, and your short-term "
        "plan. They are shown to the human fans watching you play, so write them for a spectator: plain words someone "
        "following along can understand, not shorthand notes to yourself. Call it whenever any of them changes. It "
        "doesn't end your turn."
    )
    parameters = {
        "type": "object",
        "properties": {
            "long_term": {
                "type": "string",
                "description": "The bigger goal you are working toward, in one sentence for spectators.",
            },
            "medium_term": {
                "type": "string",
                "description": "The milestone you are working on now, which will take many turns, in one sentence for "
                "spectators.",
            },
            "short_term": {
                "type": "string",
                "description": "What you are doing right now and your next few steps toward the milestone, briefly, "
                "for spectators.",
            },
        },
        "required": ["long_term", "medium_term", "short_term"],
        "additionalProperties": False,
    }

    def __init__(self, emulator):
        super().__init__(emulator)
        # Set by the agent: called after an objective is recorded (to lift that turn's requirement).
        self.on_set = lambda: None

    def execute(self, tool_input: dict) -> str:
        terms = {term: (tool_input.get(term) or "").strip() for term in ("long_term", "medium_term", "short_term")}
        if missing := [term for term, text in terms.items() if not text]:
            return ToolError(f"Error: objective needs 'long_term', 'medium_term', and 'short_term'; missing {missing}.")
        self.emit("objective", **terms)
        self.on_set()
        return "Objective recorded."
