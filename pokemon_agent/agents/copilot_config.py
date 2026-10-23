from dataclasses import dataclass


@dataclass(frozen=True)
class CopilotAgentConfig:
    """Settings for CopilotAgent, which uses GitHub Copilot-hosted models with your Copilot login.

    model: GitHub Copilot model ID (the model must support image input); None uses Copilot's default model. Usually set
        per run with `--model`.
    reasoning_effort: "low", "medium", "high", or "xhigh"; None uses the model's default.
    turn_timeout_seconds: Maximum time for one step (model calls plus tool execution).
    copilot_home: Directory for Copilot session state and logs.
    """

    model: str | None = None
    reasoning_effort: str | None = None
    turn_timeout_seconds: float = 600.0
    copilot_home: str = ".copilot-sdk"
