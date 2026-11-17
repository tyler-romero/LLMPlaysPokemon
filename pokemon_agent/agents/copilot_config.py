from dataclasses import dataclass


@dataclass(frozen=True)
class CopilotAgentConfig:
    """Settings for CopilotAgent, which uses GitHub Copilot-hosted models with your Copilot login.

    model: GitHub Copilot model ID (the model must support image input); None uses Copilot's default model. Usually set
        per run with `--model`.
    reasoning_effort: "low", "medium", "high", or "xhigh"; None uses the model's default.
    turn_timeout_seconds: Maximum time for one step (model calls plus tool execution).
    step_retries: How many times to retry a step whose model request fails (the SDK already retries each request), so
        a model service outage doesn't end the run.
    step_retry_wait_seconds: Wait before the first retry; each later retry waits that much longer.
    helper_timeout_seconds: Time limit for one `subagent` helper's analysis, which runs inside a step.
    helper_max_tool_calls: Most tool calls one helper may make before it must answer.
    objective_every_turns: Every this many turns, the model must call `objective` before its game action (0: never).
    copilot_home: Directory for Copilot session state and logs.
    """

    model: str | None = None
    reasoning_effort: str | None = None
    turn_timeout_seconds: float = 1800.0
    step_retries: int = 5
    step_retry_wait_seconds: float = 60.0
    helper_timeout_seconds: float = 900.0
    helper_max_tool_calls: int = 100
    objective_every_turns: int = 20
    copilot_home: str = ".copilot-sdk"
