"""LLM client construction and API adapters (Chat Completions and Responses).

The agent keeps its history in Chat Completions format for user and tool messages. Assistant turns are stored in
whatever native format the selected API returns, so they can be replayed to that API without loss.
"""

import os
from dataclasses import dataclass
from typing import Any

from openai import NotGiven, OpenAI
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
PROVIDERS = ("gemini", "openai", "vllm")
APIS = ("chat_completions", "responses")


@dataclass(frozen=True)
class LLMConfig:
    """Settings for one OpenAI-compatible LLM endpoint.

    provider: "gemini", "openai", or "vllm"
    api: "chat_completions" or "responses" (Gemini supports only "chat_completions")
    model: Sent as the request's `model`.
    """

    provider: str
    model: str
    api: str = "chat_completions"
    temperature: float = 1.0
    max_tokens: int = 4096
    vllm_base_url: str = "http://0.0.0.0:8000/v1"

    def __post_init__(self):
        if self.provider not in PROVIDERS:
            raise ValueError(f"Unknown provider {self.provider!r}; expected one of {PROVIDERS}")
        if self.api not in APIS:
            raise ValueError(f"Unknown api {self.api!r}; expected one of {APIS}")
        if self.provider == "gemini" and self.api == "responses":
            raise ValueError("Gemini's OpenAI-compatible endpoint does not support the Responses API")


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Environment variable {name} must be set")
    return value


def create_openai_client(config: LLMConfig) -> OpenAI:
    """Create an OpenAI SDK client pointed at the configured provider."""
    if config.provider == "gemini":
        return OpenAI(base_url=GEMINI_BASE_URL, api_key=_require_env("GEMINI_API_KEY"))
    if config.provider == "openai":
        return OpenAI()
    return OpenAI(base_url=config.vllm_base_url, api_key="dummy")


@dataclass
class LLMResponse:
    text: str | None
    tool_calls: list[ChatCompletionMessageToolCall]
    usage: Any
    history_items: list[Any]  # Assistant items to append to the message history


class ChatCompletionsAdapter:
    def __init__(self, client: OpenAI, config: LLMConfig):
        self.client = client
        self.config = config

    def create(self, messages: list, tools: list | NotGiven = NotGiven()) -> LLMResponse:
        response = self.client.chat.completions.create(
            messages=messages,
            tools=tools,
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        message = response.choices[0].message
        return LLMResponse(
            text=message.content,
            tool_calls=list(message.tool_calls or []),
            usage=response.usage,
            history_items=[message],
        )


def _to_responses_content_part(part: dict) -> dict:
    if part["type"] == "text":
        return {"type": "input_text", "text": part["text"]}
    if part["type"] == "image_url":
        return {"type": "input_image", "image_url": part["image_url"]["url"], "detail": "auto"}
    raise ValueError(f"Unsupported content part type for Responses API: {part['type']!r}")


def _to_responses_input_item(item: dict) -> dict:
    """Convert a Chat Completions-format history item into a Responses API input item.

    Assistant output items stored from previous Responses turns are passed through unchanged.
    """
    if item.get("role") == "tool":
        return {"type": "function_call_output", "call_id": item["tool_call_id"], "output": item["content"]}
    if item.get("role") in ("user", "system", "developer") and isinstance(item.get("content"), list):
        return {"role": item["role"], "content": [_to_responses_content_part(p) for p in item["content"]]}
    return item


def _to_responses_tool(tool: dict) -> dict:
    function = tool["function"]
    # Responses function tools default to strict schemas, which reject optional parameters; match Chat Completions.
    return {
        "type": "function",
        "name": function["name"],
        "description": function.get("description"),
        "parameters": function.get("parameters"),
        "strict": False,
    }


class ResponsesAdapter:
    def __init__(self, client: OpenAI, config: LLMConfig):
        self.client = client
        self.config = config

    def create(self, messages: list, tools: list | NotGiven = NotGiven()) -> LLMResponse:
        response = self.client.responses.create(
            input=[_to_responses_input_item(m) for m in messages],
            tools=[_to_responses_tool(t) for t in tools] if tools else NotGiven(),
            model=self.config.model,
            max_output_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        tool_calls = [
            ChatCompletionMessageToolCall(
                id=item.call_id, type="function", function=Function(name=item.name, arguments=item.arguments)
            )
            for item in response.output
            if item.type == "function_call"
        ]
        return LLMResponse(
            text=response.output_text or None,
            tool_calls=tool_calls,
            usage=response.usage,
            history_items=[item.model_dump(exclude_none=True) for item in response.output],
        )


def create_llm_adapter(config: LLMConfig) -> ChatCompletionsAdapter | ResponsesAdapter:
    client = create_openai_client(config)
    if config.api == "responses":
        return ResponsesAdapter(client, config)
    return ChatCompletionsAdapter(client, config)
