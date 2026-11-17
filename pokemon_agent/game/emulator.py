import datetime
import io
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass

from PIL import Image
from pyboy import PyBoy

from pokemon_agent.game.memory_reader import PokemonRedReader, StatusCondition

logger = logging.getLogger(__name__)

# (row, col) steps on the screen's 9x10 cell grid.
DIRECTION_DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

# One-way ledge jumps, from pokered's data/tilesets/ledge_tiles.asm: (direction pressed, tile stood on, ledge tile in
# front). Tiles are the bottom-left 8x8 tile of a cell, offset by 0x100 as PyBoy's background tilemap reports them.
# Ledges exist only in the OVERWORLD tileset; the collision map marks them as walls, and a jump lands two cells away.
LEDGE_JUMPS = {
    ("down", 0x12C, 0x137),
    ("down", 0x139, 0x136),
    ("down", 0x139, 0x137),
    ("left", 0x12C, 0x127),
    ("left", 0x139, 0x127),
    ("right", 0x12C, 0x10D),
    ("right", 0x12C, 0x11D),
    ("right", 0x139, 0x10D),
}

# Pairs of tiles the player can't step directly between (pokered's data/tilesets/pair_collision_tile_ids.asm, land and
# water), as (tileset, tile, tile) with tiles offset by 0x100 like LEDGE_JUMPS. In caves these separate the floor from
# raised ground.
TILE_PAIR_COLLISIONS = [
    ("CAVERN", 0x120, 0x105),
    ("CAVERN", 0x141, 0x105),
    ("FOREST", 0x130, 0x12E),
    ("CAVERN", 0x12A, 0x105),
    ("CAVERN", 0x105, 0x121),
    ("FOREST", 0x152, 0x12E),
    ("FOREST", 0x155, 0x12E),
    ("FOREST", 0x156, 0x12E),
    ("FOREST", 0x120, 0x12E),
    ("FOREST", 0x15E, 0x12E),
    ("FOREST", 0x15F, 0x12E),
    ("FOREST", 0x114, 0x12E),
    ("FOREST", 0x148, 0x12E),
    ("CAVERN", 0x114, 0x105),
]

# Text map symbols for a ledge you jump across in each direction.
TEXT_MAP_LEDGES = {"down": "v", "left": "<", "right": ">"}
# The tiles the game's Cut checks for (UsedCut: the tile in front of the player), by tileset. Shown as T in text maps.
CUT_TREE_TILES = {"OVERWORLD": 0x3D, "GYM": 0x50}
# Water the game lets you surf on (IsNextTileShoreOrWater): tile $14 in these tilesets, plus the shore tiles $32 and $48.
# The game accepts the shore tiles in every one of these tilesets but the ship port; they're only shore outdoors, though
# (in the gym tileset $32 is a statue's base), so only the outdoor tilesets count them. Shown as ≈ in text maps;
# while surfing it is walkable.
WATER_TILESETS = {"OVERWORLD", "FOREST", "DOJO", "GYM", "SHIP", "SHIP PORT", "CAVERN", "FACILITY", "PLATEAU"}
SHORE_TILESETS = {"OVERWORLD", "FOREST"}
SURFING = 2  # wWalkBikeSurfState (D700): 0 walking, 1 biking, 2 surfing
# wMapPalOffset (D35D) is nonzero in a dark cave until it is lit. The screen is then black except for faint outlines
# (rock and wall edges, ladders, some textured ground), so the text map shows only visible cells that are blocked (#)
# or exits (W); everything else, ground included, is ? since an outline doesn't show whether it can be walked on.
DARK_UNSEEN = "?"
DARK_PIXEL_LEVEL = 60  # a cell is visible in the dark if any pixel's R+G+B exceeds this (black is 0)


def is_water_tile(tile: int, tileset: str) -> bool:
    tile &= 0xFF
    if tileset not in WATER_TILESETS:
        return False
    return tile == 0x14 or (tileset in SHORE_TILESETS and tile in (0x32, 0x48))

# wSpritePlayerStateData1FacingDirection (C109) values.
FACING = {0x0: "down", 0x4: "up", 0x8: "left", 0xC: "right"}

# RAM flags that mean the game is busy in the overworld and will ignore or override input, as (address, bit mask):
# wJoyIgnore (CD6B) is set while scripts block input; wWalkCounter (CFC5) counts down while a step animates; wd732
# (D732) bits 3-4 mark a pending fly/escape-rope or dungeon warp (it starts on its own a few seconds later);
# wFlags_0xcd60 (CD60) bit 0 is set while a trainer who spotted the player walks over; wd730 (D730) bit 0 is set while
# a script moves a sprite and bit 7 while it drives the player; wd736 (D736) bit 6 is set during a ledge jump.
BUSY_FLAGS = [(0xCD6B, 0xFF), (0xCFC5, 0xFF), (0xD732, 0x18), (0xCD60, 0x01), (0xD730, 0x81), (0xD736, 0x40)]
SETTLE_FRAME_LIMIT = 900


@dataclass
class Press:
    """What one button press did: where the player was before and after, and what interrupted them."""

    button: str
    before: tuple  # (location, (x, y))
    after: tuple
    busy_before: bool  # the game was still running a scripted event, so the press may have been ignored
    in_menu: bool  # text, a menu, or a battle was on screen, so a direction moved a cursor rather than the player
    event: str | None  # "a battle started" or "text appeared", when the press caused it


@dataclass
class ScreenGrid:
    """What the visible 9x10 cell grid allows: terrain (nonzero = walkable), sprite cells as (col, row), and the
    background tilemap for tile-pair rules and ledges. Cells are (row, col); the player is always at (4, 4)."""

    terrain: list
    sprites: set
    tiles: list
    tileset: str
    exits: frozenset = frozenset()  # (row, col) cells of the map's warps (doors, stairs, ladders, holes) on screen
    water: frozenset = frozenset()  # (row, col) cells of surfable water (walkable in terrain while surfing)

    def inside(self, cell) -> bool:
        return 0 <= cell[0] < 9 and 0 <= cell[1] < 10

    def tile(self, cell) -> int:
        return self.tiles[cell[0] * 2 + 1][cell[1] * 2]

    def walkable(self, cell) -> bool:
        return self.terrain[cell[0]][cell[1]] != 0 and (cell[1], cell[0]) not in self.sprites

    def moves_from(self, cell) -> list[tuple[str, tuple[int, int], str]]:
        """(direction, landing cell, "walk" or "jump") for each move that stays on the visible grid."""
        moves = []
        for direction, (dr, dc) in DIRECTION_DELTAS.items():
            front = (cell[0] + dr, cell[1] + dc)
            if not self.inside(front):
                continue
            passable = Emulator.can_move_between_tiles(self.tile(cell), self.tile(front), self.tileset)
            if self.walkable(front) and passable:
                moves.append((direction, front, "walk"))
                continue
            landing = (cell[0] + 2 * dr, cell[1] + 2 * dc)
            ledge = self.tileset == "OVERWORLD" and (direction, self.tile(cell), self.tile(front)) in LEDGE_JUMPS
            if ledge and self.inside(landing) and self.walkable(landing):
                moves.append((direction, landing, "jump"))
        return moves


class Emulator:
    def __init__(self, rom_path, headless=True, sound=False):
        if headless:
            self.pyboy = PyBoy(
                rom_path,
                window="null",
                cgb=True,
            )
        else:
            self.pyboy = PyBoy(
                rom_path,
                cgb=True,
                sound=sound,
                sound_volume=50 if sound else 0,
            )
        # Optional observer: frame_listener(screen_rgba, stereo_int8_samples) after every frame.
        self.frame_listener = None
        self._sound_emulated = hasattr(self.pyboy.sound.raw_ndarray, "shape")
        self._fast_depth = 0

    def tick(self, frames):
        """Advance the emulator by the specified number of frames."""
        for _ in range(frames):
            self.pyboy.tick()
            if self.frame_listener:
                sound = self.pyboy.sound.ndarray if self._sound_emulated and not self._fast_depth else None
                self.frame_listener(self.pyboy.screen.ndarray, sound)

    @contextmanager
    def fast(self):
        """Run frames as fast as the host allows instead of in real time, for waits whose frames nobody needs to watch
        (letting the screen settle). A viewer sees the result sooner; the sound of those frames is dropped, since it
        would arrive far faster than it plays."""
        self._fast_depth += 1
        if self._fast_depth == 1:
            self.pyboy.set_emulation_speed(0)
        try:
            yield
        finally:
            self._fast_depth -= 1
            if not self._fast_depth:
                self.pyboy.set_emulation_speed(1)

    def initialize(self):
        """Initialize the emulator."""
        # Run the emulator for a short time to make sure it's ready
        self.pyboy.set_emulation_speed(0)
        for _ in range(60):
            self.tick(60)
        self.pyboy.set_emulation_speed(1)

    def get_screenshot(self):
        """Get the current screenshot."""
        return Image.fromarray(self.pyboy.screen.ndarray)

    def save_state_to(self, path):
        """Save the emulator state to one file."""
        with open(path, "wb") as f:
            self.pyboy.save_state(f)

    def load_state(self, state_filename, validate=True):
        """
        Load a state from a pickled file into the emulator.
        The pickled file should contain a dictionary with a 'pyboy_state' key.

        Args:
            state_filename: Path to the state file
        """
        with open(state_filename, "rb") as f:
            self.pyboy.load_state(f)
        if not validate:
            return

        memory_reader = PokemonRedReader(self.pyboy.memory)
        player_name = memory_reader.read_player_name()
        if player_name.strip() in ["", "NINTEN", "Not yet set"]:
            raise ValueError("Loaded state appears to be at the beginning of the game or title screen")

    def save_state(self, state_dir="saved_games") -> list[str]:
        """
        Save the current emulator state to a pickled file.

        Args:
            state_dir: Directory path where state files will be saved

        Returns:
            Paths of the state files written
        """
        os.makedirs(state_dir, exist_ok=True)

        # Save a standard "pokemon_red_last_session.state" file
        last_session_path = os.path.join(state_dir, "pokemon_red_last_session.state")
        if os.path.exists(last_session_path):
            os.remove(last_session_path)
        with open(last_session_path, "wb") as f:
            self.pyboy.save_state(f)

        # Save a timestamped state file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_path = os.path.join(state_dir, f"pokemon_red_{timestamp}.state")
        with open(timestamped_path, "wb") as f:
            self.pyboy.save_state(f)

        logger.info(f"Saved game state to {timestamped_path} and {last_session_path}")
        return [last_session_path, timestamped_path]

    def _place(self) -> tuple:
        reader = PokemonRedReader(self.pyboy.memory)
        return reader.read_location(), reader.read_coordinates()

    def press_buttons(self, buttons, wait=True) -> list[Press]:
        """Press a sequence of buttons on the Game Boy, returning what each press did.

        Args:
            buttons (list[str]): List of buttons to press in sequence
            wait (bool): Whether to wait after each button press (about 2 seconds, then longer if the game is still busy
                with a scripted event; see is_busy)
        """
        presses = []

        for button in buttons:
            if button not in [
                "a",
                "b",
                "start",
                "select",
                "up",
                "down",
                "left",
                "right",
            ]:
                raise ValueError(f"Invalid button: {button}")

            reader = PokemonRedReader(self.pyboy.memory)
            before = self._place()
            busy_before = self.is_busy()
            in_menu = bool(reader.read_dialog()) or reader.read_in_combat()
            self.pyboy.button_press(button)
            self.tick(10)  # Press briefly
            self.pyboy.button_release(button)

            if wait:
                with self.fast():
                    self.tick(120)  # Wait longer after button release
                    self.wait_until_idle()
            else:
                self.tick(10)  # Brief pause between button presses

            event = None
            if not in_menu and reader.read_in_combat():
                event = "a battle started"
            elif not in_menu and reader.read_dialog():
                event = "text appeared"
            presses.append(Press(button, before, self._place(), busy_before, in_menu, event))

        return presses

    def walk_step(self, direction: str, max_frames: int = 120):
        """Press a direction and run only until the resulting step, turn, or bump finishes.

        The warp-exit check (_warp_exits) uses this instead of press_buttons' fixed 2-second wait. The walk counter (wWalkCounter, CFC5)
        counts down while a step animates, and wJoyIgnore (CD6B) is set while the game ignores input (for example
        during a ledge jump). Landing on a warp (door, stairs) gets the full wait so the map transition completes.
        """
        self.pyboy.button_press(direction)
        self.tick(10)
        self.pyboy.button_release(direction)
        memory = self.pyboy.memory
        with self.fast():
            for _ in range(max_frames):
                if not memory[0xCFC5] and not memory[0xCD6B]:
                    break
                self.tick(1)
            self.tick(6)
            reader = PokemonRedReader(memory)
            if reader.read_coordinates() in reader.get_warps():
                self.tick(120)
            self.wait_until_idle()

    def is_busy(self) -> bool:
        """Whether the game is running a scripted event in the overworld (a warp about to happen, a trainer walking
        over, scripted movement, a step or ledge jump still animating) and so will ignore or override input. Text and
        battles aren't counted: the player is expected to act on them."""
        reader = PokemonRedReader(self.pyboy.memory)
        if not reader.read_game_started() or reader.read_in_combat() or reader.read_dialog():
            return False
        return any(self.pyboy.memory[address] & mask for address, mask in BUSY_FLAGS)

    def wait_until_idle(self, limit: int = SETTLE_FRAME_LIMIT) -> bool:
        """Let a scripted overworld event play out (see is_busy), up to `limit` frames; return whether it's idle."""
        with self.fast():
            for _ in range(0, limit, 10):
                if not self.is_busy():
                    return True
                self.tick(10)
        return not self.is_busy()

    def get_coordinates(self):
        """
        Returns the player's current map coordinates from game memory.
        Note: Pokemon Red specific.

        Returns:
            tuple[int, int]: (x, y) coordinates
        """
        reader = PokemonRedReader(self.pyboy.memory)
        return reader.read_coordinates()

    def game_started(self) -> bool:
        """Whether the player is in the game world (past the title screen and intro)."""
        return PokemonRedReader(self.pyboy.memory).read_game_started()

    def wait_for_screen_wipe(self, max_waits: int = 10, frames: int = 30):
        """Let screen transitions finish so the next screenshot shows a settled screen.

        First lets any scripted overworld event play out (see is_busy). Then waits while the battle-start wipe fills the
        screen with black tiles, while the screen is a single solid color, while a battle has started but its first text
        hasn't appeared yet, or while on-screen text is still printing (up to max_waits * frames frames).
        """
        self.wait_until_idle()
        reader = PokemonRedReader(self.pyboy.memory)
        previous_dialog = None
        for _ in range(max_waits):
            screen = self.pyboy.screen.ndarray[:, :, :3]
            blank = (screen == screen[0, 0]).all()
            dialog = reader.read_dialog()
            printing = bool(dialog) and dialog != previous_dialog
            previous_dialog = dialog
            if not (
                reader.read_screen_wipe_fraction() >= 0.25
                or blank
                or (reader.read_in_combat() and not dialog)
                or printing
            ):
                return
            with self.fast():
                self.tick(frames)

    def get_active_dialog(self):
        """
        Returns the active dialog text from game memory.
        Returns:
            str: Dialog text
        """
        reader = PokemonRedReader(self.pyboy.memory)
        dialog = reader.read_dialog()
        if dialog:
            return dialog
        return None

    def get_location(self):
        """
        Returns the player's current location name from game memory.
        Returns:
            str: Location name
        """
        reader = PokemonRedReader(self.pyboy.memory)
        return reader.read_location()

    def _downsample_array(self, arr):
        """Downsample an 18x20 array to 9x10 by averaging 2x2 blocks."""
        # Ensure input array is 18x20
        if arr.shape != (18, 20):
            raise ValueError("Input array must be 18x20")

        # Reshape to group 2x2 blocks and take mean
        return arr.reshape(9, 2, 10, 2).mean(axis=(1, 3))

    def get_collision_map(self) -> str | None:
        """A text map of the visible 9x10 cells, labeled with absolute map coordinates like the screenshot's grid, or
        None before the game starts and during battles.

        Built from the same screen model as valid_moves, using only what is on screen:
        walkable or blocked terrain, one-way ledges, exits (the map's warps), sprites, and cells past the map's edge.
        A `|` or `-` between two walkable cells marks a step the game refuses (TILE_PAIR_COLLISIONS: in caves and
        forests, ground at a different level can't be stepped onto directly); both cells stay walkable, since the other
        ground may be reachable another way, possibly off screen.
        """
        reader = PokemonRedReader(self.pyboy.memory)
        if not reader.read_game_started() or reader.read_in_combat():
            return None  # (in battle the screen shows the battle, not the map)
        x, y = reader.read_coordinates()
        grid = self.screen_grid()
        width, height = self.map_size()
        visible = self._visible_cells() if self.pyboy.memory[0xD35D] else None

        def on_map(cell) -> bool:
            return 0 <= x + cell[1] - 4 < width and 0 <= y + cell[0] - 4 < height

        def symbol(cell) -> str:
            if cell == (4, 4):
                return "@"
            if visible is not None and cell not in visible:
                return DARK_UNSEEN
            if not on_map(cell):
                return "~"
            if (cell[1], cell[0]) in grid.sprites:
                return "S"
            if cell in grid.exits:
                return "W"
            if cell in grid.water:
                return "≈"
            if grid.terrain[cell[0]][cell[1]] != 0:
                # In the dark a faint outline doesn't show whether the ground there can be walked on.
                return DARK_UNSEEN if visible is not None else "."
            if (grid.tile(cell) & 0xFF) == CUT_TREE_TILES.get(grid.tileset):
                return "T"
            if grid.tileset == "OVERWORLD":
                for direction, (dr, dc) in DIRECTION_DELTAS.items():
                    behind = (cell[0] - dr, cell[1] - dc)
                    if grid.inside(behind) and (direction, grid.tile(behind), grid.tile(cell)) in LEDGE_JUMPS:
                        return TEXT_MAP_LEDGES[direction]
            return "#"

        def refused(a, b) -> bool:
            """Whether the game refuses a step between two walkable, on-map cells (a change in ground level)."""
            if visible is not None:
                return False  # (both cells are walkable ground, which the dark map doesn't show)
            walkable = all(on_map(c) and grid.terrain[c[0]][c[1]] != 0 for c in (a, b))
            return walkable and not self.can_move_between_tiles(grid.tile(a), grid.tile(b), grid.tileset)

        xs = [x - 4 + col for col in range(10)]
        lines = ["y\\x" + "".join(f"{label:>4}" for label in xs)]
        boundaries = False
        for row in range(9):
            line = f"{y - 4 + row:>3}"
            for col in range(10):
                edge = "|" if col > 0 and refused((row, col - 1), (row, col)) else " "
                line += "  " + edge + symbol((row, col))
            lines.append(line)
            boundaries = boundaries or "|" in line
            if row < 8:
                marks = ["-" if refused((row, col), (row + 1, col)) else " " for col in range(10)]
                if "-" in marks:
                    lines.append("   " + "".join(f"   {mark}" for mark in marks).rstrip())
                    boundaries = True
        lines += [
            "",
            f"Legend: @ you (facing {self.get_facing()})   . walkable   # blocked   v < > ledge: jump across it in "
            "that direction, one way   T small tree: blocks the way, but can be cut down   ≈ water: can't be walked on, "
            "but can be crossed by surfing   W door, stairs, ladder, or hole   S character or object   ~ past this map's "
            "edge",
        ]
        if visible is not None:
            lines.append(
                f"It's dark here: only the faint outlines on screen are mapped, the rock and walls (#) and ladders or "
                f"other exits (W). Everything else is {DARK_UNSEEN}: too dark to tell whether it's ground or rock."
            )
        if boundaries:
            lines.append(
                "- | between two cells: the game won't let you step directly across that edge (the ground changes "
                "level); that ground may be reachable another way."
            )
        return "\n".join(lines)

    def _visible_cells(self) -> set[tuple[int, int]]:
        """(row, col) cells of the 9x10 grid with anything visible on screen, for a dark cave's mostly black screen."""
        pixels = self.pyboy.screen.ndarray[:, :, :3].astype(int).sum(axis=2)
        return {
            (row, col)
            for row in range(9)
            for col in range(10)
            if pixels[row * 16 : row * 16 + 16, col * 16 : col * 16 + 16].max() > DARK_PIXEL_LEVEL
        }

    def screen_grid(self) -> ScreenGrid:
        tileset = PokemonRedReader(self.pyboy.memory).read_tileset()
        terrain = self._downsample_array(self.pyboy.game_wrapper.game_area_collision())  # type: ignore
        tiles = self.pyboy.game_wrapper._get_screen_background_tilemap()  # type: ignore
        reader = PokemonRedReader(self.pyboy.memory)
        x, y = reader.read_coordinates()
        exits = frozenset((wy - y + 4, wx - x + 4) for wx, wy in reader.get_warps())
        water = frozenset(
            (row, col) for row in range(9) for col in range(10) if is_water_tile(tiles[row * 2 + 1][col * 2], tileset)
        )
        if water and self.pyboy.memory[0xD700] == SURFING:
            terrain = [list(line) for line in terrain]
            for row, col in water:
                terrain[row][col] = 1
        return ScreenGrid(terrain, self.get_sprites(), tiles, tileset, exits, water)

    def map_size(self) -> tuple[int, int]:
        """The current map's (width, height) in cells: wCurMapWidth (D369) and wCurMapHeight (D368) count 2x2 blocks."""
        return self.pyboy.memory[0xD369] * 2, self.pyboy.memory[0xD368] * 2

    def get_facing(self) -> str:
        return FACING.get(self.pyboy.memory[0xC109] & 0xC, "down")

    def get_valid_moves(self) -> list[str]:
        """Directions the player can move from here, judged from the current screen.

        A plain direction is a normal step. "(ledge jump, one way)" marks a jump down a ledge, which lands two cells
        away. "(leaves this map)" marks a step off the map's edge onto a connected map, or through the door or exit the
        player is standing on. Some exits aren't at the map's edge (a gangway, for example, leads off by pressing into
        what looks like a wall), so on a warp each direction is tried in the emulator to see which ones leave.
        """
        reader = PokemonRedReader(self.pyboy.memory)
        x, y = reader.read_coordinates()
        width, height = self.map_size()
        moves = {direction: kind for direction, _, kind in self.screen_grid().moves_from((4, 4))}

        def off_map(direction):
            dr, dc = DIRECTION_DELTAS[direction]
            return not (0 <= x + dc < width and 0 <= y + dr < height)

        leaving = set()
        if (x, y) in reader.get_warps():
            leaving = {direction for direction in DIRECTION_DELTAS if off_map(direction)} | self._warp_exits()
            for direction in leaving:
                moves.setdefault(direction, "walk")
        result = []
        for direction in DIRECTION_DELTAS:
            if direction not in moves:
                continue
            if off_map(direction) or direction in leaving:
                result.append(f"{direction} (leaves this map)")
            elif moves[direction] == "jump":
                result.append(f"{direction} (ledge jump, one way)")
            else:
                result.append(direction)
        return result

    def _warp_exits(self) -> set[str]:
        """Directions that take the player off this map from the warp they stand on, found by trying each one from a
        saved state and restoring it afterwards (the observer doesn't see the tries)."""
        if not PokemonRedReader(self.pyboy.memory).read_game_started() or self.get_in_combat():
            return set()
        memory = self.pyboy.memory
        here = memory[0xD35E]  # wCurMap
        snapshot = io.BytesIO()
        self.pyboy.save_state(snapshot)
        listener, self.frame_listener = self.frame_listener, None
        found = set()
        try:
            with self.fast():
                for direction in DIRECTION_DELTAS:
                    snapshot.seek(0)
                    self.pyboy.load_state(snapshot)
                    self.walk_step(direction)
                    for _ in range(12):  # the map changes after the screen fades out
                        if memory[0xD35E] != here:
                            found.add(direction)
                            break
                        self.tick(10)
        finally:
            snapshot.seek(0)
            self.pyboy.load_state(snapshot)
            self.frame_listener = listener
        return found

    @staticmethod
    def can_move_between_tiles(tile1: int, tile2: int, tileset: str) -> bool:
        """
        Check if movement between two tiles is allowed based on tile pair collision data.

        Args:
            tile1: The tile being moved from
            tile2: The tile being moved to
            tileset: The current tileset name

        Returns:
            bool: True if movement is allowed, False if blocked
        """
        for ts, t1, t2 in TILE_PAIR_COLLISIONS:
            # Collisions are bidirectional.
            if ts == tileset and ((tile1 == t1 and tile2 == t2) or (tile1 == t2 and tile2 == t1)):
                return False
        return True

    def get_warps(self):
        reader = PokemonRedReader(self.pyboy.memory)
        return reader.get_warps()

    def get_sprites(self) -> set[tuple[int, int]]:
        """Screen cells (column, row) of the 10x9 grid occupied by NPC and object sprites (from the sprite table)."""
        reader = PokemonRedReader(self.pyboy.memory)
        player_x, player_y = reader.read_coordinates()
        cells = set()
        for x, y in reader.read_sprite_cells():
            col, row = x - player_x + 4, y - player_y + 4
            if 0 <= col < 10 and 0 <= row < 9 and (col, row) != (4, 4):
                cells.add((col, row))
        return cells

    def get_in_combat(self) -> bool:
        reader = PokemonRedReader(self.pyboy.memory)
        return reader.read_in_combat()

    def get_state_from_memory(self) -> tuple[dict, str, tuple[int, int]]:
        """
        Reads the game state from memory and returns a dictionary representation of it.
        """
        reader = PokemonRedReader(self.pyboy.memory)
        memory_dict = {}

        name = reader.read_player_name()
        if name == "NINTEN" or name == "AAAAAAA":
            name = "Not yet set"
        rival_name = reader.read_rival_name()
        if rival_name == "SONY" or rival_name == "AAAAAAA":
            rival_name = "Not yet set"

        # Get valid moves
        valid_moves = self.get_valid_moves()

        location = reader.read_location()
        coords = reader.read_coordinates()  # This comes out col, row (x, y)
        dialog = reader.read_dialog()

        memory_dict["dialog"] = dialog if dialog else None
        memory_dict["player"] = name
        memory_dict["rival"] = rival_name
        memory_dict["money"] = f"${reader.read_money()}"
        # Before the intro ends, the map and position in memory aren't where the screen is.
        if reader.read_game_started():
            memory_dict["location"] = location
            memory_dict["map_coordinates"] = coords
            memory_dict["valid_moves"] = valid_moves if valid_moves and not dialog else []
        else:
            location, coords = None, None
        memory_dict["badges"] = reader.read_badges()

        # Inventory
        memory_dict["inventory"] = {}
        for item, qty in reader.read_items():
            memory_dict["inventory"][item] = qty

        # Party Pokemon
        memory_dict["pokemon_party"] = []
        for pokemon in reader.read_party_pokemon():
            pokemon_dict = {
                "nickname": pokemon.nickname,
                "species_name": pokemon.species_name,
                "level": pokemon.level,
                "hp": f"{pokemon.current_hp}/{pokemon.max_hp}",
                "types": f"{pokemon.type1.name}{', ' + pokemon.type2.name if pokemon.type2 else ''}",
                "moves": [],
                "status": pokemon.status.get_status_name() if pokemon.status != StatusCondition.NONE else None,
            }

            for move, pp in zip(pokemon.moves, pokemon.move_pp, strict=True):
                pokemon_dict["moves"].append({"name": move, "pp": pp})

            memory_dict["pokemon_party"].append(pokemon_dict)

        return memory_dict, location, coords

    def stop(self):
        self.pyboy.stop()
