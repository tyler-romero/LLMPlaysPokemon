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
        "--web-output", action="store_true", help="Save latest screenshots to ./web_output/ for web viewing"
    )

    parser.add_argument(
        "--observe",
        action="store_true",
        help="Serve a live observer page (game, audio, model log, inputs, party, memory) while the agent runs",
    )
    parser.add_argument("--observe-port", type=int, default=8765, help="Port for the observer page (default: 8765)")

    args = parser.parse_args()

    # Validate conflicting arguments
    if args.web_output and args.display:
        logger.error("Cannot use both --display and --web-output flags simultaneously.")
        sys.exit(1)

    if args.resume and args.load_state:
        parser.error("--resume and --load-state can't be combined")

    rom_path = os.path.abspath(args.rom)

    # Check if ROM exists
    if not os.path.exists(rom_path):
        logger.error(f"ROM file not found: {rom_path}")
        print("\nYou need to provide a Pokemon Red ROM file to run this program.")
        print("Place the ROM in the root directory or specify its path with --rom.")
        return

    # Set up web output directory if needed
    web_output_dir = None
    if args.web_output:
        web_output_dir = "web_output"
        try:
            os.makedirs(web_output_dir, exist_ok=True)
            logger.info(f"Web output mode enabled. Screenshots will be saved to {web_output_dir}/latest.png")
        except OSError as e:
            logger.error(f"Failed to create web output directory {web_output_dir}: {e}")
            sys.exit(1)

    # Every run is durable: it gets a run folder with checkpoints, and --resume continues it.
    if args.resume:
        try:
            run = RunStore.open(args.resume)
        except FileNotFoundError as e:
            parser.error(str(e))
        meta = run.meta
        if args.agent and args.agent != meta["agent"]:
            parser.error(f"{args.resume} is a {meta['agent']!r} run; it can't be resumed as {args.agent!r}")
        if args.model and args.model != meta.get("model"):
            parser.error(f"{args.resume} uses model {meta.get('model')!r}; it can't be resumed with {args.model!r}")
        args.agent, args.model = meta["agent"], meta.get("model")
        args.steps = args.steps or meta.get("steps", 250)
        run.update_meta(steps=args.steps)
        checkpoint = run.latest_checkpoint()
        if not checkpoint:
            parser.error(f"{args.resume} has no complete checkpoint to resume")
        resume_seq = json.loads((checkpoint / "checkpoint.json").read_text()).get("observer_seq")
    else:
        args.agent = args.agent or "simple"
        args.steps = args.steps or 250
        run = RunStore.create(
            args.runs_dir,
            {"agent": args.agent, "model": _effective_model(args), "steps": args.steps, "load_state": args.load_state},
        )
        resume_seq = None

    # The hub records the run's event history even without --observe, so a later --observe shows the whole run.
    observer = ObserverHub(event_log=run.events_path, resume_seq=resume_seq)

    # Create and run agent
    agent = AGENT_BUILDERS[args.agent](
        args,
        rom_path=rom_path,
        headless=not args.display,
        sound=args.sound if args.display else False,
        save_every=args.save_every,
        load_state=args.load_state,
        web_output_dir=web_output_dir,
        observer=observer,
        run=run,
        resume=bool(args.resume),
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
