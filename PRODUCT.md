# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Python server inside the harness (part of the `pokemon_agent` package) serving a static HTML/CSS/JS page with no build
step and no Node toolchain. Chosen by the user after comparing it with a JS framework + build step.

## Users

The researcher running the harness (one person, locally): they start an agent run and want to watch it play, check
what the model is thinking and doing, and spot when it is stuck or misbehaving. Secondary: anyone they share a screen
or stream with.

## Product Purpose

LLM Plays Pokemon is a research harness where LLM agents (an OpenAI-compatible "simple" agent and a GitHub Copilot SDK
agent) play Pokemon Red through the PyBoy emulator using tools (button presses, navigation, on-screen keyboard) and a
file-based `/memories` notebook. The observer page makes a live run legible: the game itself, the model's reasoning
and tool calls, the inputs it sent, and the game state that matters. Success is being able to tell at a glance what
the agent is doing, why, and whether it is making progress.

## Operating Context

- Runs locally next to an agent run started with `uv run pokemon-agent`; viewed in a desktop browser, often beside a
  terminal and the code.
- Runs are long (hundreds to thousands of steps); the page stays open for the whole run.
- Live only for now: it shows the current run; replay of past runs is explicitly out of scope until requested.

## Capabilities and Constraints

- Shows: the live game screen; the model's text output and tool calls per step; button/input history; the current
  Pokemon party (species, nickname, level, HP, types, status); run progress (step count, agent, model, badges,
  location, money); and the memory files the agent maintains.
- Game audio: real emulator audio streamed to the browser, off by default with a mute/unmute toggle (browsers require
  a click before playing sound).
- Data available in-process: emulator frames, emulator audio buffer, RAM-derived game state
  (`pokemon_agent/game/emulator.py`, `memory_reader.py`), agent events (text, tool calls, usage, compaction), and
  `/memories`.
- Must not slow the agent loop noticeably; the page is an observer and never controls the game.

## Brand Commitments

- Name: "LLM Plays Pokemon".
- The user shared a screenshot of the "Claude Plays Pokemon" stream layout (model log beside the game screen, current
  team strip along the bottom) as a reference for the kind of content to show. It is a content reference, not a
  binding visual style.

## Evidence on Hand

- Real game data comes from the running emulator; no sample runs, logs, or recordings are stored in the repo.
- No logo or brand assets exist.

## Product Principles

- The game is the subject; everything else explains what the agent is doing to it.
- Make the agent's reasoning and actions legible in time order, so a stuck or looping agent is obvious.
- Show facts read from the game and agent, never invented or decorative stats.
- Observing must be safe: read-only and cheap enough not to disturb the run.
