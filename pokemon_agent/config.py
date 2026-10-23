# Configuration for the application
from pokemon_agent.agents.copilot_config import CopilotAgentConfig
from pokemon_agent.llm.client import LLMConfig

# LLM used by the "simple" agent. See LLMConfig in pokemon_agent/llm/client.py for all options, including
# provider ("gemini", "openai", "vllm") and api ("chat_completions", "responses").
SIMPLE_AGENT_LLM = LLMConfig(
    provider="gemini",
    model="gemini-2.0-flash",
    api="chat_completions",
    temperature=1.0,
    max_tokens=4096,
)

# Other examples:
# LLMConfig(provider="openai", model="gpt-4.1-nano", api="responses")
# LLMConfig(provider="vllm", model="Qwen/Qwen2.5-VL-7B-Instruct")

# Settings for the "copilot" agent (GitHub Copilot SDK with a Copilot-hosted model and your Copilot login).
# See CopilotAgentConfig in pokemon_agent/agents/copilot_config.py for all options. The model defaults to Copilot's
# default model; choose one per run with `--model <copilot-model-id>`.
COPILOT_AGENT = CopilotAgentConfig()
