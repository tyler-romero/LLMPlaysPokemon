import argparse
import dataclasses
import json
import logging
import os
import sys
import time

from pokemon_agent.agents.base import BaseAgent
from pokemon_agent.agents.copilot import CopilotAgent
from pokemon_agent.agents.simple import SimpleAgent
from pokemon_agent.config import COPILOT_AGENT, SIMPLE_AGENT_LLM
from pokemon_agent.observer.hub import ObserverHub
from pokemon_agent.observer.server import ObserverServer
from pokemon_agent.runs import RunStore

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def build_simple_agent(args: argparse.Namespace, **common) -> BaseAgent:
    llm_config = dataclasses.replace(SIMPLE_AGENT_LLM, model=args.model) if args.model else SIMPLE_AGENT_LLM
    return SimpleAgent(llm_config=llm_config, max_history=args.max_history, **common)


def build_copilot_agent(args: argparse.Namespace, **common) -> BaseAgent:
    config = dataclasses.replace(COPILOT_AGENT, model=args.model) if args.model else COPILOT_AGENT
    return CopilotAgent(config=config, **common)


# Agent name -> builder taking parsed CLI args and the shared BaseAgent options.
AGENT_BUILDERS = {
    "simple": build_simple_agent,
    "copilot": build_copilot_agent,
}


def main():
    parser = argparse.ArgumentParser(description="LLM Plays Pokemon")
    parser.add_argument("--agent", choices=sorted(AGENT_BUILDERS), default=None, help="Agent to run (default: simple)")
    parser.add_argument(
        "--model", type=str, default=None, help="Model to use, overriding the agent's model in config.py"
    )
    parser.add_argument(
        "--rom", type=str, default="pokemon.gb", help="Path to the Pokemon ROM file (relative to the current directory)"
    )
    parser.add_argument(
        "--steps", type=int, default=None, help="Run until this many total steps (default: 250, or the run's target)"
    )
    parser.add_argument("--display", action="store_true", help="Run with display (not headless)")
    parser.add_argument(
        "--sound",
        action="store_true",
        help="Enable sound (only applicable with display)",
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=30,
        help="Maximum number of messages in history before summarization (simple agent)",
    )
    parser.add_argument("--save-every", type=int, default=10, help="Steps between checkpoints (default: 10)")
    parser.add_argument(
        "--load-state", type=str, default=None, help="Start a new run from a saved game state (and its memory)"
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume a run folder from its latest checkpoint")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Where new run folders go (default: runs)")
    parser.add_argument(
        "--text-map",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Give the model a text-based map of the screen each step (default: on, or a resumed run's setting)",
    )

    parser.add_argument(
        "--observe",
        action="store_true",
        help="Serve a live observer page (game, audio, model log, tool stats, party, memory) while the agent runs",
    )
    parser.add_argument("--observe-port", type=int, default=8765, help="Port for the observer page (default: 8765)")

    args = parser.parse_args()

    if args.resume and args.load_state:
        parser.error("--resume and --load-state can't be combined")

    rom_path = os.path.abspath(args.rom)

    # Check if ROM exists
    if not os.path.exists(rom_path):
        logger.error(f"ROM file not found: {rom_path}")
        print("\nYou need to provide a Pokemon Red ROM file to run this program.")
        print("Place the ROM in the root directory or specify its path with --rom.")
        return

    # Create and run agent
    agent = AGENT_BUILDERS[args.agent](
        args,
        rom_path=rom_path,
        headless=not args.display,
        sound=args.sound if args.display else False,
        save_every=args.save_every,
        load_state=args.load_state,
        observer=observer,
        run=run,
        resume=bool(args.resume),
        text_map=args.text_map,
    )
    print(f"\nRun folder: {run.root} (resume with --resume {run.root})\n", flush=True)

    server = None
    if args.observe:
        try:
            server = ObserverServer(observer, agent.memory, port=args.observe_port).start()
        except OSError as e:
            logger.error(f"Could not start the observer page on port {args.observe_port}: {e}")
            agent.stop()
            sys.exit(1)
        print(f"\nObserver page: {server.url}\n", flush=True)

    try:
        logger.info(f"Starting agent for {args.steps} steps")
        steps_completed = agent.run(num_steps=args.steps)
        logger.info(f"Agent completed {steps_completed} steps")
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, stopping")
    except Exception as e:
        logger.error(f"Error running agent: {e}")
    finally:
        logger.info("Stopping agent")
        agent.stop()
        if server:
            _keep_observer_open(server)
        observer.close()


def _effective_model(args: argparse.Namespace) -> str | None:
    """The model the run will use, recorded so a resumed run can't silently switch models."""
    if args.model:
        return args.model
    return SIMPLE_AGENT_LLM.model if args.agent == "simple" else COPILOT_AGENT.model


def _keep_observer_open(server: ObserverServer):
    """Leave the observer page up after the run so its final state can be inspected."""
    print(f"\nRun finished. The observer page is still at {server.url} — press Ctrl-C to exit.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
