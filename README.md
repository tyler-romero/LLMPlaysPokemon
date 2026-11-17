# LLM Plays Pokemon

A harness that lets a language model play Pokemon Red. The game runs in the
[PyBoy](https://github.com/Baekalfen/PyBoy) emulator, and the model plays by calling tools that press buttons, read
and write its own notes, and look back over what it has already done. You can watch a run live in your browser,
and you can stop a long run and pick it up again later.

## Quick start

You need [uv](https://docs.astral.sh/uv/) and your own copy of the Pokemon Red ROM.

```
uv sync
cp /path/to/your/rom.gb pokemon.gb
uv run pokemon-agent --agent copilot --observe
```

Open the URL that it prints (by default `http://127.0.0.1:8765/`) to watch the run.

## Agents

There are two agents, and you choose one with `--agent`.

**`copilot`** runs on the [GitHub Copilot SDK](https://github.com/github/copilot-sdk). It uses a model hosted by
GitHub Copilot and signs in with your existing Copilot login, so you don't need an API key. Pass `--model` to pick a
model; it must accept images. Without `--model`, Copilot's default model is used. This is the agent with the most
tools, described below. The SDK downloads its runtime the first time it runs.

**`simple`** calls any OpenAI-compatible endpoint directly and keeps its own message history. Set the endpoint in
`SIMPLE_AGENT_LLM` in `pokemon_agent/config.py`:

| `provider` | Endpoint | `api` | Setup |
|---|---|---|---|
| `gemini` | Gemini's OpenAI-compatible API | `chat_completions` | `export GEMINI_API_KEY=...` |
| `openai` | OpenAI API | `chat_completions` or `responses` | `export OPENAI_API_KEY=...` |
| `vllm` | A local vLLM server | `chat_completions` or `responses` | `./start_vllm.sh --model <model>` |

## What the model gets

On every turn, the model sees a screenshot with a coordinate grid and a game state that is read from the emulator's
memory: its location and position, its party, the moves it can make from where it stands, and any text on screen.
It also gets a text map of the screen, with symbols for walls, ledges, doors, water, and trees that Cut can clear.
In dark caves, this map only shows what the screen actually shows.

It plays with these tools:

- **`press_buttons`** presses a sequence of buttons and reports where each press left the player.
- **`on_screen_keyboard_input`** types a name when the game asks for one.
- **Memory tools** (`view`, `list`, `create`, `edit`, `insert`, `delete`, `rename`, `save_screenshot`) work on a
  private `/memories` folder that the model organizes however it likes.
- **`history`** searches the model's own past turns. It can also show old screenshots, list the summaries that the
  conversation was condensed into, and stitch every screen it has seen of a place into one map.
- **`subagent`** asks a helper to analyze the notes and history and report back. The helper can read but not play.
- **`objective`** records the model's current goals for people watching.
- **`trigger_compaction`** lets the model condense its own conversation when it gets cluttered.

The model must narrate each button press for viewers. A turn ends after a game action, so the model always looks
at a fresh screenshot before it acts again.

## Watching a run

With `--observe`, the agent serves a page that shows the game with optional sound, the model's narration and tool
calls, its goals and party, and its memory files as they change. The page stays up after the run ends until you
press Ctrl-C.

## Pausing and resuming

Every run gets a folder under `runs/` and saves a checkpoint every `--save-every` steps, and also when it stops. A
checkpoint holds the game, the memory folder, and the conversation. To continue a run:

```
uv run pokemon-agent --resume runs/<run folder> --steps 5000
```

`--steps` is the total for the whole run, so this example keeps going until step 5000.

## Options

| Option | What it does |
|---|---|
| `--agent` | `simple` (the default) or `copilot` |
| `--model` | Overrides the model set in `pokemon_agent/config.py` |
| `--steps` | How many steps to run (default 250) |
| `--observe`, `--observe-port` | Serve the live page, by default on port 8765 |
| `--resume` | Continue a run from its latest checkpoint |
| `--save-every` | Steps between checkpoints (default 10) |
| `--load-state` | Start a new run from a saved PyBoy state |
| `--no-text-map` | Leave the text map out of the game state |
| `--display`, `--sound` | Show the emulator window, with sound |
| `--rom` | Path to the ROM (default `pokemon.gb`) |

Run `uv run pokemon-agent --help` to see every option.

## Code layout

Everything is in the `pokemon_agent` package:

- `game/` wraps the emulator and reads the game state from memory. It knows nothing about language models.
- `tools/` holds the tools that the model can call.
- `memory/` stores the `/memories` folder.
- `agents/` has the two agents and all the prompts.
- `history.py` and `history_map.py` index a run's log for the `history` tool.
- `observer/` serves the live page.
- `runs.py` writes and reads checkpoints.
- `cli.py` is the command line, and `config.py` holds each agent's settings.

To add an agent, subclass `BaseAgent` in `agents/base.py`, implement `run`, and register a builder in
`AGENT_BUILDERS` in `cli.py`.
