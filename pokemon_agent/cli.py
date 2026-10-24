import argparse
import dataclasses
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
    parser.add_argument("--agent", choices=sorted(AGENT_BUILDERS), default="simple", help="Agent to run")
    parser.add_argument(
        "--model", type=str, default=None, help="Model to use, overriding the agent's model in config.py"
    )
    parser.add_argument(
        "--rom", type=str, default="pokemon.gb", help="Path to the Pokemon ROM file (relative to the current directory)"
    )
    parser.add_argument("--steps", type=int, default=250, help="Number of agent steps to run")
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
    parser.add_argument("--save-every", type=int, default=100, help="Number of steps between save_state() calls")
    parser.add_argument("--load-state", type=str, default=None, help="Path to a saved state to load")
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

    observer = ObserverHub() if args.observe else None

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
    )

    server = None
    if observer:
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
