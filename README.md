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
3. Choose a provider, model, and API by editing `SIMPLE_AGENT_LLM` in `pokemon_agent/config.py` (see [Supported LLM endpoints](#supported-llm-endpoints))
4. Place your Pokemon Red ROM file in the root directory as `pokemon.gb` (you need to provide your own ROM)

## Supported LLM Endpoints

All models are called through the OpenAI Python SDK with tool calling. Each agent's LLM is an `LLMConfig`
(`pokemon_agent/llm/client.py`); select the endpoint with `provider` and the API with `api`:

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
Configure its LLM with `SIMPLE_AGENT_LLM` in `pokemon_agent/config.py`.

### Copilot SDK agent (`--agent copilot`)

Runs on a [GitHub Copilot SDK](https://github.com/github/copilot-sdk) session, which handles the model/tool loop and
compacts context automatically. Each step sends the game state with the screenshot attached; the model can call
tools within a step, and the step ends after a game action (`press_buttons`, `navigate_to_screen_coordinates`,
`on_screen_keyboard_input`) so the model always acts on a fresh screenshot. Only the game tools are available to the
model; the SDK's built-in tools are disabled.

It uses a GitHub Copilot-hosted model with your existing Copilot login; no API key is needed. Configure it with
`COPILOT_AGENT` in `pokemon_agent/config.py`. Choose the model per run with `--model <copilot-model-id>`; without it, Copilot's default
model is used. The model must support image input.

The SDK downloads its runtime on first use (or run `uv run python -m copilot download-runtime` ahead of time).
Session state and logs are written to `.copilot-sdk/`.

## Observer

Run with `--observe` to watch a run in your browser:

```
uv run pokemon-agent --agent copilot --observe
```

Open the printed URL (default `http://127.0.0.1:8765/`). The page shows the live game screen with optional game
audio (off until you turn it on), the model's dialogue and tool calls per step with timings, a battle-text narration of
each action, the button-press history, the party, location and badges, and the agent's `/memories` files. The memory
reader follows the files the model reads and writes until you click a file. The page is read-only, desktop-only, and
stays up after the run ends until you press Ctrl-C. The pixel font loads from Google Fonts; offline it falls back to a
system monospace.

## Memory

Both agents get memory tools shaped like GitHub Copilot's built-in file tools: `view`, `create`, and `edit` use the
same names and parameters as the built-ins (with memory-focused descriptions), plus `delete` and `rename`. They operate
on a virtual `/memories` directory stored in the run folder's `memories/`, and every path is checked so the model can't read or write
outside it. Each step includes a listing of the memory files (not their contents); the model reads what it needs.

Each run keeps its memory in its own run folder (see [Durable runs](#durable-runs)) and starts empty unless it resumes
a checkpoint or loads a state with a memory snapshot.

## Durable runs

Every run gets a folder under `runs/` (change it with `--runs-dir`), and is checkpointed every `--save-every` steps and
when it ends or crashes. A checkpoint holds everything needed to continue: the game state, the `/memories` files, and
the conversation (the `simple` agent's message history, or the Copilot SDK session). Resume a run with:

```
uv run pokemon-agent --resume runs/<run id> --steps 1000
```

`--steps` is the total step count for the run, so the example continues until step 1000. The agent and model come from
the run, and passing a different `--agent` or `--model` is an error. The observer replays the run's earlier history.

```
runs/<YYYYmmdd-HHMMSS>-<agent>-<model>/
  run.json          # agent, model, and creation time
  events.jsonl      # observer event log (replayed on resume)
  memories/         # live /memories files
  copilot-sdk/      # Copilot SDK home (copilot agent only)
  checkpoints/<step>/
    game.state  memories/  agent/  checkpoint.json
  latest            # name of the newest complete checkpoint
```

Checkpoints are written atomically, so an interrupted save never replaces the last good one. The last 5 checkpoints and
every 100th step are kept. `--load-state` starts a new run from a PyBoy state file (restoring the memory snapshot next
to it, `<state file>.memories/`, if there is one).

## Usage

Run the CLI (or `uv run python main.py`, which is equivalent):

```
uv run pokemon-agent
```

Optional arguments:
- `--rom`: Path to the Pokemon ROM file (default: `pokemon.gb` in the root directory)
- `--steps`: Number of agent steps to run (default: 250)
- `--display`: Run with display (not headless)
- `--sound`: Enable sound (only applicable with display)
- `--agent`: Agent to run: `simple` or `copilot` (default: `simple`)
- `--model`: Model to use, overriding the agent's model in `pokemon_agent/config.py`
- `--observe`: Serve the live observer page while the agent runs (see [Observer](#observer))
- `--observe-port`: Port for the observer page (default: 8765)
- `--max-history`: Maximum number of messages in history before summarization, for the `simple` agent (default: 30)
- `--save-every`: Number of steps between checkpoints (default: 10)
- `--load-state`: Path to a saved state to start a new run from
- `--resume`: Run folder to resume from its latest checkpoint (see [Durable runs](#durable-runs))
- `--runs-dir`: Directory for new run folders (default: `runs`)
- `--web-output`: Save latest screenshots to `./web_output/` for web viewing

Example:
```
uv run pokemon-agent --rom pokemon.gb --steps 20 --display --sound
```

## Implementation Details

### Components

Everything lives in the `pokemon_agent` package. Dependencies flow downward: agents use tools, memory, and LLM
clients; tools use the game; the game knows nothing about LLMs.

- `game/`: The Pokemon Red environment
  - `emulator.py`: Wrapper around PyBoy with helper functions
  - `memory_reader.py`: Extracts game state information from emulator memory
  - `observation.py`: Screenshot grid annotation and the per-step `GameObservation`
- `tools/`: Tools exposed to the model; each implements `execute(tool_input: dict) -> str` with no LLM SDK types
  - `base.py`: `Tool` base class
  - `game_tools.py`: `press_buttons`, `navigate_to_screen_coordinates`, `on_screen_keyboard_input`
  - `memory_tools.py`: `view`, `create`, `edit`, `delete`, `rename` over `/memories`
  - `factory.py`: Tool registry and `ToolFactory`
- `memory/store.py`: File-based memory store, tool schemas, and save-state snapshots
- `llm/client.py`: `LLMConfig` and Chat Completions/Responses API adapters
- `agents/`: Agents and prompts
  - `base.py`: `BaseAgent` with shared emulator, tool, observation, memory, and save-state handling
  - `simple.py`: `SimpleAgent`, which manages its own history and summaries over an OpenAI-compatible API
  - `copilot.py`, `copilot_config.py`: `CopilotAgent`, which runs on a GitHub Copilot SDK session, and its settings
  - `prompts.py`: System and summarization prompts
- `config.py`: Per-agent configuration
- `observer/`: Live observer page: event hub, local HTTP/SSE server, and the static page
- `cli.py`: CLI and the `AGENT_BUILDERS` registry

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
   `pokemon_agent/tools/factory.py`).
3. Add its settings (e.g. an `LLMConfig`) to `pokemon_agent/config.py` and a builder to `AGENT_BUILDERS` in
   `pokemon_agent/cli.py`, then run it with `--agent <name>`.
