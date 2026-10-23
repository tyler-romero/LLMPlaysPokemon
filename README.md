# LLM Plays Pokemon

A minimal implementation of an LLM playing Pokemon Red using the PyBoy emulator. It includes:

- A simple agent that uses an LLM to play Pokemon Red
- Memory reading functionality to extract game state information
- Basic emulator control through the LLM's tool calling

## Setup

1. Clone this repository
2. Install the dependencies with [uv](https://docs.astral.sh/uv/):
   ```
   uv sync
   ```
3. Choose a provider, model, and API by editing `SIMPLE_AGENT_LLM` in `config.py` (see [Supported LLM endpoints](#supported-llm-endpoints))
4. Place your Pokemon Red ROM file in the root directory as `pokemon.gb` (you need to provide your own ROM)

## Supported LLM Endpoints

All models are called through the OpenAI Python SDK with tool calling. Each agent's LLM is an `LLMConfig`
(`agent/llm_client.py`); select the endpoint with `provider` and the API with `api`:

| `provider` | Endpoint | Supported `api` values | Required setup |
|---|---|---|---|
| `gemini` | Gemini's OpenAI-compatible API | `chat_completions` | `export GEMINI_API_KEY=your_api_key_here` |
| `openai` | OpenAI API | `chat_completions`, `responses` | `export OPENAI_API_KEY=your_api_key_here` |
| `vllm` | Local vLLM server at `vllm_base_url` | `chat_completions`, `responses` | Start the server with `start_vllm.sh` |

`model` is sent as the request's `model`. `temperature` and `max_tokens` apply to both APIs.

### vLLM

To serve a local model with vLLM:

```
./start_vllm.sh --model Qwen/Qwen2.5-VL-7B-Instruct-AWQ --gpus 2
```

Run `./start_vllm.sh --help` for all options. `model` must match the model served by vLLM.

## Agents

### Simple agent (`--agent simple`)

Calls the configured LLM directly, keeps its own message history, and summarizes it every `--max-history` messages.
Configure its LLM with `SIMPLE_AGENT_LLM` in `config.py`.

### Copilot SDK agent (`--agent copilot`)

Runs on a [GitHub Copilot SDK](https://github.com/github/copilot-sdk) session, which handles the model/tool loop and
compacts context automatically. Each step sends the game state with the screenshot attached; the model can call
tools within a step, and the step ends after a game action (`press_button`, `navigate_to_screen_coordinates`,
`on_screen_keyboard_input`) so the model always acts on a fresh screenshot. Only the game tools are available to the
model; the SDK's built-in tools are disabled.

It uses a GitHub Copilot-hosted model with your existing Copilot login; no API key is needed. Configure it with
`COPILOT_AGENT` in `config.py`. Choose the model per run with `--model <copilot-model-id>`; without it, Copilot's default
model is used. The model must support image input.

The SDK downloads its runtime on first use (or run `uv run python -m copilot download-runtime` ahead of time).
Session state and logs are written to `.copilot-sdk/`.

## Memory

Both agents get memory tools shaped like GitHub Copilot's built-in file tools: `view`, `create`, and `edit` use the
same names and parameters as the built-ins (with memory-focused descriptions), plus `delete` and `rename`. They operate
on a virtual `/memories` directory stored in `memories/`, and every path is checked so the model can't read or write
outside it. Each step includes a listing of the memory files (not their contents); the model reads what it needs.

A new run starts with empty memory (the previous run's is kept in `memories.old/`). Each saved state gets a snapshot
of the memory directory next to it (`<state file>.memories/`), and `--load-state` restores it.

## Usage

Run the main script:

```
uv run python main.py
```

Optional arguments:
- `--rom`: Path to the Pokemon ROM file (default: `pokemon.gb` in the root directory)
- `--steps`: Number of agent steps to run (default: 250)
- `--display`: Run with display (not headless)
- `--sound`: Enable sound (only applicable with display)
- `--agent`: Agent to run: `simple` or `copilot` (default: `simple`)
- `--model`: Model to use, overriding the agent's model in `config.py`
- `--max-history`: Maximum number of messages in history before summarization, for the `simple` agent (default: 30)
- `--save-every`: Number of steps between save states (default: 100)
- `--load-state`: Path to a saved state to load
- `--web-output`: Save latest screenshots to `./web_output/` for web viewing

Example:
```
uv run python main.py --rom pokemon.gb --steps 20 --display --sound
```

## Implementation Details

### Components

- `agent/base_agent.py`: `BaseAgent` with shared emulator, tool, observation, and save-state handling
- `agent/simple_agent.py`: `SimpleAgent`, which manages its own history and summaries over an OpenAI-compatible API
- `agent/copilot_agent.py`: `CopilotAgent`, which runs on a GitHub Copilot SDK session
- `agent/tools.py`: Tool definitions exposed to the LLM
- `agent/memory.py`: File-based memory store and memory tool definitions
- `agent/prompts.py`: System and summarization prompts
- `agent/emulator.py`: Wrapper around PyBoy with helper functions
- `agent/memory_reader.py`: Extracts game state information from emulator memory
- `agent/llm_client.py`: Provider clients and Chat Completions/Responses API adapters
- `config.py`: Per-agent LLM configuration
- `main.py`: CLI and the `AGENT_BUILDERS` registry

### How It Works

1. The agent captures a screenshot from the emulator
2. It reads the game state information from memory
3. It sends the screenshot and game state to the LLM
4. The LLM responds with explanations and tool calls
5. The agent executes the tool calls and repeats the process

### Adding an Agent

1. Subclass `BaseAgent` and implement `run(num_steps) -> int`. Use `self.observe()` for the annotated screenshot and
   game state, `self.tool_factory` to run tools, and `self._on_step_complete(...)` after each step.
2. Optionally set `tool_names` on the class to choose its tools (defaults to `DEFAULT_TOOL_NAMES` in
   `agent/tools.py`). Tools implement `execute(tool_input: dict) -> str` and don't depend on an LLM SDK.
3. Add its settings (e.g. an `LLMConfig`) to `config.py` and a builder to `AGENT_BUILDERS` in `main.py`, then run it
   with `--agent <name>`.
