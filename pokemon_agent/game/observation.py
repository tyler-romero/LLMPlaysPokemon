"""Screenshot annotation and the per-step game observation."""

import base64
import io
import logging
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

from pokemon_agent.game.emulator import Emulator

logger = logging.getLogger(__name__)


def get_screenshot_base64(emulator: Emulator, screenshot: Image.Image, upscale: int = 3):
    """Convert PIL image to base64 string."""
    if upscale > 1:
        new_size = (screenshot.width * upscale, screenshot.height * upscale)
        logger.info(f"[Screenshot] Upscaling from {screenshot.size} to {new_size}")
        screenshot = screenshot.resize(new_size)

    if not emulator.get_in_combat():
        shape = screenshot.size
        player_x, player_y = emulator.get_coordinates()  # absolute coordinates

        cell_width = shape[0] // 10
        cell_height = shape[1] // 9

        # Draw grid lines
        for x in range(0, shape[0], cell_width):
            ImageDraw.Draw(screenshot).line(((x, 0), (x, shape[1] - 1)), fill=(255, 0, 0))
        for y in range(0, shape[1], cell_height):
            ImageDraw.Draw(screenshot).line(((0, y), (shape[0] - 1, y)), fill=(255, 0, 0))

        # Add coordinates to every other tile
        font = ImageFont.load_default()
        for grid_y in range(9):
            for grid_x in range(10):
                # Calculate the relative coordinates
                rel_x = grid_x - 4 + player_x
                rel_y = grid_y - 4 + player_y

                if not (grid_x == 4 and grid_y == 4):
                    # Calculate position for text
                    text_x = grid_x * cell_width + 2
                    text_y = grid_y * cell_height + 2

                    # Create coordinate text
                    coord_text = f"({rel_x},{rel_y})"

                    # Add yellow background for text
                    text_width, text_height = (
                        font.getsize(coord_text)
                        if hasattr(font, "getsize")
                        else ImageDraw.Draw(screenshot).textbbox((0, 0), coord_text, font=font)[2:4]
                    )
                    ImageDraw.Draw(screenshot).rectangle(
                        [(text_x, text_y), (text_x + text_width, text_y + text_height)], fill=(255, 255, 0)
                    )

                    # Draw text
                    ImageDraw.Draw(screenshot).text((text_x, text_y), coord_text, fill=(0, 0, 0), font=font)

    screenshot.save("screenshot_test.png")
    buffered = io.BytesIO()
    screenshot.save(buffered, format="PNG")
    return base64.standard_b64encode(buffered.getvalue()).decode()


@dataclass
class GameObservation:
    screenshot_b64: str
    memory_info: dict
    collision_map: str | None
