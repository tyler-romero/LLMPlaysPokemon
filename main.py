import argparse
import logging
import os
import sys

from agent.simple_agent import SimpleAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Claude Plays Pokemon - Starter Version")
    parser.add_argument("--rom", type=str, default="pokemon.gb", help="Path to the Pokemon ROM file")
    parser.add_argument("--steps", type=int, default=10, help="Number of agent steps to run")
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
        help="Maximum number of messages in history before summarization",
    )
    parser.add_argument("--load-state", type=str, default=None, help="Path to a saved state to load")
    parser.add_argument(
        "--web-output", action="store_true", help="Save latest screenshots to ./web_output/ for web viewing"
    )

    args = parser.parse_args()

    # Validate conflicting arguments
    if args.web_output and args.display:
        logger.error("Cannot use both --display and --web-output flags simultaneously.")
        sys.exit(1)

    # Get absolute path to ROM
    if not os.path.isabs(args.rom):
        rom_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.rom)
    else:
        rom_path = args.rom

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

    # Create and run agent
    agent = SimpleAgent(
        rom_path=rom_path,
        headless=not args.display,
        sound=args.sound if args.display else False,
        max_history=args.max_history,
        load_state=args.load_state,
        web_output_dir=web_output_dir,  # Pass web output directory to agent
    )

    try:
        logger.info(f"Starting agent for {args.steps} steps")
        steps_completed = agent.run(num_steps=args.steps)
        logger.info(f"Agent completed {steps_completed} steps")
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, stopping")
    except Exception as e:
        logger.error(f"Error running agent: {e}")
    finally:
        agent.stop()


if __name__ == "__main__":
    main()
