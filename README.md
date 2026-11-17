# LLM Plays Pokemon

An LLM playing Pokemon Red in the PyBoy emulator. It includes:

- Two agents: a simple agent for any OpenAI-compatible endpoint, and an agent built on the GitHub Copilot SDK
- Game state read from emulator RAM, a text map of the screen, and the moves the screen allows
- File-based `/memories` the model curates, a searchable history of its own turns, and read-only helper subagents
- A live browser observer and durable runs that can be paused and resumed from checkpoints

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
compacts context automatically. It uses a GitHub Copilot-hosted model with your existing Copilot login; no API key is
needed. Configure it with `COPILOT_AGENT` in `pokemon_agent/config.py`, and choose the model per run with
`--model <copilot-model-id>` (without it, Copilot's default model is used). The model must support image input.

#### Turns and tool calls

Each step sends the game state with the screenshot attached; the model can call
tools within a step, and the step ends after a game action (`press_buttons`, `on_screen_keyboard_input`) so the model
always acts on a fresh screenshot. The model may list several tool calls in one response; they run one at a time in the
order listed (every handler takes a per-session lock), and the first failure stops the rest, which return `Not run`
(the turn continues). A response may hold one game action, as its last call, after memory writes and `objective` but
not after `view`, `list`, `history`, or `subagent`, whose results the model must read before acting. These rules are
checked in the handlers (`CopilotAgent._refusal`), since every call runs in order. A call fails when its tool returns
a `ToolError` (tools never signal failure by wording alone) or raises. When a turn times out, its request (and any
helper's) is aborted and its remaining calls refused before the step is retried. `SimpleAgent` follows the same order,
fail-fast, and last-game-action rules, with a prompt variant (`system_prompt(model, agent="simple")`) that leaves out
the tools it doesn't have.

#### History, helpers, compaction, and objectives

The Copilot agent gets a read-only `history` tool over its own SDK session log
(`copilot-sdk/session-state/<id>/events.jsonl`): browse turns (`window`), open an item (`item`), search (a small query syntax: all words in any order, `"phrases"`, `OR`, `-exclude`, `prefix*`, `~fuzzy`, `/regex/`; whole words, ignoring case, punctuation, and spacing; newest first with `page`d results) over
prompts (including the game state), reasoning, and tool calls (`search`, with consecutive matching prompts grouped into
turn ranges), per-tool call counts (`tools`), old screenshots (`image`), and one map stitched together from every
text map the model saw on it and every cell it stood on (`map`, in `history_map.py`: each cell as last seen, `?` where never seen or too dark, level boundaries marked in the cell's own
row (`|` before it, `_` after it), with the exits
used and their destinations from button-press results, and doors seen but never used; `x_range`/`y_range` crop it and
`to_turn` shows it as of an earlier turn), and the summaries its conversation was condensed into (`summary`: numbered and
titled, newest first, searchable with the same query syntax, opened by `number`; read from the log's
`session.compaction_complete` events, so they follow the run's timeline). Items have short unique ID suffixes; compaction
summaries are kept out of the items, bookkeeping events are hidden, and the index reads new log lines incrementally. The
turn after each compaction lists the titles of the 8 summaries before it. Only the game and memory tools are available to the model; the SDK's built-in tools are
disabled.

The `subagent` tool asks a read-only helper for an analysis: a second SDK session with a fresh context that gets the
task, the current screenshot and game state, and the memory file list. The helper can only read: `view` and `list` over
`/memories`, and `history` over the main agent's turns (including old screenshots). It can't play or write memory. It
runs inside the tool call, so it uses no game turns and doesn't end the main agent's turn; its final reply comes back as
the tool result. It's limited to `helper_max_tool_calls` tool calls and `helper_timeout_seconds` (see
`CopilotAgentConfig`), and its own turns aren't in `history`.

Each step message also says how full the model's context is and how far it is from automatic compaction (from the
SDK's `metadata.context_info`). The `trigger_compaction` tool lets the model condense its conversation itself: it only
sets a flag, and once the turn ends the agent calls `history.compact` before the next step message.

The `objective` tool records the model's long-term goal, medium-term milestone, and short-term plan, written for
spectators; the observer shows the latest medium and short term above the party (not the long term). The model can call it anytime, and every `objective_every_turns` turns (default 20) it
must: that turn's message asks for it, and game actions are refused until it's called (earlier in the same response
counts).

A turn's message can also carry Notices about the model's notes (`BaseAgent.notices`, Copilot agent only), each only
while it applies: the conversation was condensed and no note has been viewed since; a `create`/`edit`/`insert` failed
and no later write to that file succeeded (for up to 20 turns); and nothing has been written for 50 turns (repeated
every 10).

#### Game state and movement

Each step's game state comes from RAM. `valid_moves` lists the directions the player can move from their cell, judged
from the current screen, and marks one-way ledge jumps (`down (ledge jump, one way)`) and moves onto another map, off the
map's edge or through the door the player stands on (`up (leaves this map)`). On a warp, each direction is also tried
from a saved emulator state (`Emulator._warp_exits`, restored afterwards and hidden from the observer), since some exits,
like a ship's gangway, lead off by pressing into a blocked tile mid-map. The model walks with `press_buttons`,
which reports where each press left the player when any press moved them or changed the map (`left: didn't move you;
still at (20, 16)`), so a blocked press in a sequence is visible. Before each observation and after each press, the
emulator waits (up to 900 frames) while the game is busy with a scripted overworld event that would ignore input: a
pending escape-rope or fly warp, a trainer walking over, scripted movement, or a step or ledge jump still animating (RAM
flags in `BUSY_FLAGS`). Presses run in real time, but every wait (after a press, for a busy game, for a screen
transition) runs unthrottled (`Emulator.fast`, about 180x real time), so the observer sees results
sooner; the sound of fast-forwarded frames is dropped. A game action that changed nothing (an `Error` result) doesn't
end the turn: the SDK gets a failed tool result, so the model can retry right away. Step messages don't list the memory
files (the model calls `list`); only a helper's single message includes that list. When a tool returns images (e.g.
`history` with `image`), the observer shows them under the tool's row, exactly as the model received them. Game actions
require a `narration` field (what the model is doing and why, for viewers), which the observer shows as the model's
dialogue before the action; each turn's text input is also published (`prompt` events) and shown collapsed.

The SDK downloads its runtime on first use (or run `uv run python -m copilot download-runtime` ahead of time).
Session state and logs are written to `.copilot-sdk/`.

## Observer

Run with `--observe` to watch a run in your browser:

```
uv run pokemon-agent --agent copilot --observe
```

Open the printed URL (default `http://127.0.0.1:8765/`). The page shows the live game screen with optional game
audio (off until you turn it on), the model's dialogue and tool calls per step with timings, a battle-text narration of
each action, per-tool call stats (calls, failure rate, turns since the last call), the model's stated objective, the party, location and badges, and the agent's `/memories` files. The memory
reader follows the files the model reads and writes until you click a file. The page is read-only, desktop-only, and
stays up after the run ends until you press Ctrl-C. The pixel font loads from Google Fonts; offline it falls back to a
system monospace.

## Memory

Both agents get memory tools shaped like GitHub Copilot's built-in file tools: `view`, `create`, and `edit` use the
same names and parameters as the built-ins (with memory-focused descriptions, and `view` reads files only), plus
`list`, `save_screenshot`, `delete`, and `rename`. `list` works like `ls`: every file with its size and the turn it was
last updated. `save_screenshot` saves the turn's screenshot (as the model saw it, grid included) as a `.png`, up to 30;
`view` shows a saved screenshot to the Copilot agent as an image (the simple agent's API can't carry images in tool
results). Turns and screenshot locations are kept in a hidden `.meta.json` in the memory folder, which the tools can't
reach. They operate
on a virtual `/memories` directory stored in the run folder's `memories/`, and every path is checked so the model can't read or write
outside it. Each text file holds at most 400,000 bytes (`MAX_FILE_BYTES`; a write past it is refused, saying how to make
room), and `view` shows a long file 20,000 characters at a time, cut at a line break, with where to continue. The model
calls `list` to see its files and reads what it needs.

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
  run.json          # agent, model, settings, and creation time
  events.jsonl      # observer event log (replayed on resume)
  memories/         # live /memories files
  copilot-sdk/      # Copilot SDK home (copilot agent only)
  checkpoints/<step>/
    game.state  memories/  agent/  checkpoint.json
  latest            # name of the newest complete checkpoint
```

Checkpoints are written atomically, so an interrupted save never replaces the last good one (re-saving a step moves
the old copy aside first, and resume falls back to the newest complete checkpoint if `latest` names a missing one). The last 5 checkpoints and
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
- `--text-map` / `--no-text-map`: Include a text-based map of the screen in each step's game state (default: on; a
  resumed run keeps its setting unless one of these is given). The map (`Emulator.get_collision_map`) uses the same screen model as `valid_moves`:
  `@` you, `.` walkable, `#` blocked, `v < >` one-way ledges, `T` trees Cut can remove (the tile UsedCut checks: 0x3D overworld, 0x50 gym), `≈` surfable water (the tiles
  IsNextTileShoreOrWater checks; walkable for `valid_moves` while surfing), `W` exits, `S` sprites, `~` past the map's edge, and
  `|`/`-` between two walkable cells the game won't let you step directly between (cave and forest ground levels). It is left out during battles. In a dark cave (wMapPalOffset, D35D, nonzero until the area is lit) the screen is black
  except for faint outlines, so only cells with something visible in the screenshot that are blocked (`#`) or exits (`W`)
  are mapped (`Emulator._visible_cells`); everything else, walkable ground included, shows `?`; the stitched map treats `?` as not seen.

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
  - `game_tools.py`: `press_buttons` and `on_screen_keyboard_input`
  - `memory_tools.py`: `view`, `list`, `save_screenshot`, `create`, `edit`, `insert`, `delete`, `rename` over `/memories`
  - `history_tool.py`: `history`, read-only search over the Copilot agent's session log (index in `history.py`)
  - `subagent_tool.py`: `subagent`, a read-only helper that analyzes memory and history (run by `CopilotAgent`)
  - `compaction_tool.py`: `trigger_compaction`, the model condensing its own conversation (run by `CopilotAgent`)
  - `objective_tool.py`: `objective`, the model's long-, medium-, and short-term goals (required every N turns by `CopilotAgent`)
  - `factory.py`: Tool registry and `ToolFactory`
- `memory/store.py`: File-based memory store, tool schemas, and save-state snapshots
- `llm/client.py`: `LLMConfig` and Chat Completions/Responses API adapters
- `agents/`: Agents and prompts
  - `base.py`: `BaseAgent` with shared emulator, tool, observation, memory, and save-state handling
  - `simple.py`: `SimpleAgent`, which manages its own history and summaries over an OpenAI-compatible API
  - `copilot.py`, `copilot_config.py`: `CopilotAgent`, which runs on a GitHub Copilot SDK session, and its settings
  - `prompts.py`: System prompts (main agent and helper), step messages, and the summarization prompt
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
