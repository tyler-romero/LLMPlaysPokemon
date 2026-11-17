"""Screenshot annotation and the per-step game observation."""

import base64
import io
import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pokemon_agent.game.emulator import Emulator

logger = logging.getLogger(__name__)


GRID_COLUMNS, GRID_ROWS = 10, 9
AXIS_LEFT, AXIS_TOP = 52, 36  # margins (pixels) for the y labels on the left and x labels on the top
# Every screenshot the model sees is the Game Boy screen (160x144) scaled up this much. In a model evaluation of reading
# walls, exits, and reachability, screenshots alone did best at 5x-6.5x, but with the text map and valid_moves the
# image size made no difference (3x-6x all ~94%), so 4x keeps the cost down (about 600 tokens; 5x and up cost ~855).
MODEL_IMAGE_SCALE = 4
GB_WIDTH, GB_HEIGHT = 160, 144
GRID_COLOR = (255, 0, 0)
INK = (0, 0, 0)


def annotate_grid(screenshot: Image.Image, player_x: int, player_y: int) -> Image.Image:
    """Add a coordinate grid: red cell lines over the screen, x labels along the top, y labels down the left.

    The labels are absolute map coordinates; the player is always in column 4 and row 4 (counting from 0), and their
    own x and y labels are circled. The corner is split diagonally like a table header: x above, y below.
    """
    width, height = screenshot.size
    cell_width, cell_height = width // GRID_COLUMNS, height // GRID_ROWS
    canvas = Image.new("RGB", (width + AXIS_LEFT, height + AXIS_TOP), (255, 255, 255))
    canvas.paste(screenshot.convert("RGB"), (AXIS_LEFT, AXIS_TOP))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=22)

    for column in range(GRID_COLUMNS + 1):
        x = AXIS_LEFT + min(column * cell_width, width - 1)
        draw.line(((x, AXIS_TOP), (x, AXIS_TOP + height - 1)), fill=GRID_COLOR)
    for row in range(GRID_ROWS + 1):
        y = AXIS_TOP + min(row * cell_height, height - 1)
        draw.line(((AXIS_LEFT, y), (AXIS_LEFT + width - 1, y)), fill=GRID_COLOR)

    def label(text: str, center: tuple[float, float], circled: bool):
        draw.text(center, text, fill=INK, font=font, anchor="mm")
        if circled:
            left, top, right, bottom = draw.textbbox(center, text, font=font, anchor="mm")
            rx, ry = max((right - left) / 2 + 8, 16), 16
            draw.ellipse((center[0] - rx, center[1] - ry, center[0] + rx, center[1] + ry), outline=INK, width=2)

    for column in range(GRID_COLUMNS):
        center = (AXIS_LEFT + column * cell_width + cell_width / 2, AXIS_TOP / 2)
        label(str(player_x - 4 + column), center, circled=column == 4)
    for row in range(GRID_ROWS):
        center = (AXIS_LEFT / 2, AXIS_TOP + row * cell_height + cell_height / 2)
        label(str(player_y - 4 + row), center, circled=row == 4)

    corner_font = ImageFont.load_default(size=18)
    draw.line(((0, 0), (AXIS_LEFT - 1, AXIS_TOP - 1)), fill=INK, width=1)
    draw.text((AXIS_LEFT * 0.72, AXIS_TOP * 0.32), "x", fill=INK, font=corner_font, anchor="mm")
    draw.text((AXIS_LEFT * 0.28, AXIS_TOP * 0.68), "y", fill=INK, font=corner_font, anchor="mm")
    return canvas


def get_screenshot_base64(emulator: Emulator, screenshot: Image.Image, upscale: int = MODEL_IMAGE_SCALE):
    """Upscale the screenshot, add the coordinate grid outside of battle, and return it as base64 PNG."""
    if upscale > 1:
        new_size = (screenshot.width * upscale, screenshot.height * upscale)
        logger.info(f"[Screenshot] Upscaling from {screenshot.size} to {new_size}")
        screenshot = screenshot.resize(new_size, Image.NEAREST)

    if emulator.game_started() and not emulator.get_in_combat():
        player_x, player_y = emulator.get_coordinates()
        screenshot = annotate_grid(screenshot, player_x, player_y)

    return base64.standard_b64encode(encode_png(screenshot)).decode()


def at_model_scale(png: bytes) -> bytes:
    """A saved screenshot as the model should see it now: one taken at a smaller scale (with or without the grid
    margins) is enlarged to MODEL_IMAGE_SCALE, nearest-neighbor, so old screenshots look like new ones. Other images
    are returned unchanged."""
    image = Image.open(io.BytesIO(png))
    width, height = image.size
    for margin_x, margin_y in ((AXIS_LEFT, AXIS_TOP), (0, 0)):
        scale = (width - margin_x) / GB_WIDTH
        if 1 <= scale < MODEL_IMAGE_SCALE and scale == (height - margin_y) / GB_HEIGHT:
            factor = MODEL_IMAGE_SCALE / scale
            return encode_png(image.resize((round(width * factor), round(height * factor)), Image.NEAREST))
    return png


def encode_png(image: Image.Image) -> bytes:
    """PNG bytes, as an indexed-color PNG when that is pixel-identical (the screen plus the grid use well under 256
    colors), which is about 2.5x smaller; otherwise as RGB. The model sees exactly the same pixels either way."""
    rgb = image.convert("RGB")
    buffered = io.BytesIO()
    if rgb.getcolors(256) is not None:
        indexed = rgb.quantize(colors=256, dither=Image.Dither.NONE)
        if np.array_equal(np.asarray(indexed.convert("RGB")), np.asarray(rgb)):
            indexed.save(buffered, format="PNG", optimize=True)
            return buffered.getvalue()
    rgb.save(buffered, format="PNG")
    return buffered.getvalue()


@dataclass
class GameObservation:
    screenshot_b64: str
    memory_info: dict
    collision_map: str | None
