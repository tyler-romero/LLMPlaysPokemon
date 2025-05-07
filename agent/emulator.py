import datetime
import heapq
import logging
import os
import random
import time

from PIL import Image
from pyboy import PyBoy

from agent.memory_reader import PokemonRedReader, StatusCondition

logger = logging.getLogger(__name__)


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
            )

    def tick(self, frames):
        """Advance the emulator by the specified number of frames."""
        for _ in range(frames):
            self.pyboy.tick()

    def initialize(self):
        """Initialize the emulator."""
        # Run the emulator for a short time to make sure it's ready
        self.pyboy.set_emulation_speed(0)
        for _ in range(60):
            self.tick(60)
        self.pyboy.set_emulation_speed(1)

    def get_screenshot(self):
        """Get the current screenshot."""
        time.sleep(0.2 + random.random())
        return Image.fromarray(self.pyboy.screen.ndarray)

    def load_state(self, state_filename):
        """
        Load a state from a pickled file into the emulator.
        The pickled file should contain a dictionary with a 'pyboy_state' key.

        Args:
            state_filename: Path to the state file
        """
        with open(state_filename, "rb") as f:
            self.pyboy.load_state(f)

        memory_reader = PokemonRedReader(self.pyboy.memory)
        player_name = memory_reader.read_player_name()
        if player_name.strip() in ["", "NINTEN", "Not yet set"]:
            raise ValueError("Loaded state appears to be at the beginning of the game or title screen")

    def save_state(self, state_dir="saved_games"):
        """
        Save the current emulator state to a pickled file.

        Args:
            state_dir: Directory path where state files will be saved
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

    def press_buttons(self, buttons, wait=True) -> str:
        """Press a sequence of buttons on the Game Boy.

        Args:
            buttons (list[str]): List of buttons to press in sequence
            wait (bool): Whether to wait after each button press

        Returns:
            str: Result of the button presses
        """
        results = []

        for button in buttons:
            if button == "↑":
                button = "up"
            elif button == "↓":
                button = "down"
            elif button == "←":
                button = "left"
            elif button == "→":
                button = "right"

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
                results.append(f"Invalid button: {button}")
                raise ValueError(f"Invalid button: {button}")

            self.pyboy.button_press(button)
            self.tick(10)  # Press briefly
            self.pyboy.button_release(button)

            if wait:
                self.tick(120)  # Wait longer after button release
            else:
                self.tick(10)  # Brief pause between button presses

            results.append(f"Pressed {button}")

        return "\n".join(results)

    def get_coordinates(self):
        """
        Returns the player's current map coordinates from game memory.
        Note: Pokemon Red specific.

        Returns:
            tuple[int, int]: (x, y) coordinates
        """
        reader = PokemonRedReader(self.pyboy.memory)
        return reader.read_coordinates()

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

    def _get_direction(self, array):
        """Determine the player's facing direction from the sprite pattern."""
        # Look through the array for any 2x2 grid containing numbers 0-3
        rows, cols = array.shape

        for i in range(rows - 1):
            for j in range(cols - 1):
                # Extract 2x2 grid
                grid = array[i : i + 2, j : j + 2].flatten()

                # Check for each direction pattern
                if list(grid) == [0, 1, 2, 3]:
                    return "down"
                elif list(grid) == [4, 5, 6, 7]:
                    return "up"
                elif list(grid) == [9, 8, 11, 10]:
                    return "right"
                elif list(grid) == [8, 9, 10, 11]:
                    return "left"

        return "no direction found"

    def _downsample_array(self, arr):
        """Downsample an 18x20 array to 9x10 by averaging 2x2 blocks."""
        # Ensure input array is 18x20
        if arr.shape != (18, 20):
            raise ValueError("Input array must be 18x20")

        # Reshape to group 2x2 blocks and take mean
        return arr.reshape(9, 2, 10, 2).mean(axis=(1, 3))

    def get_collision_map(self):
        """
        Creates a simple ASCII map showing player position, direction, terrain, warps, and sprites.
        Returns:
            str: A string representation of the ASCII map with legend
        """
        # Get the terrain and movement data
        full_map = self.pyboy.game_wrapper.game_area()
        collision_map = self.pyboy.game_wrapper.game_area_collision()  # type: ignore
        downsampled_terrain = self._downsample_array(collision_map)
        sprite_locations = self.get_sprites()
        reader = PokemonRedReader(self.pyboy.memory)
        warp_coords = reader.get_warps()  # absolute coordinates
        player_absolute_coords = reader.read_coordinates()

        relative_warp_coords = []
        for warp_x, warp_y in warp_coords:
            # Calculate relative position (player is always at (4,4) in our map)
            rel_x = warp_x - player_absolute_coords[0] + 4
            rel_y = warp_y - player_absolute_coords[1] + 4

            # Only include warps that would appear on our 9x10 map
            if 0 <= rel_x < 10 and 0 <= rel_y < 9:
                relative_warp_coords.append((rel_x, rel_y))

        # Get character direction from the full map
        direction = self._get_direction(full_map)
        if direction == "no direction found":
            return None

        # Direction symbols
        direction_chars = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
        player_char = direction_chars.get(direction, "P")

        # Create the ASCII map (Note that this is relative to the player's position, which is always (4,4))
        horizontal_border = "+" + "-" * 10 + "+"
        lines = [horizontal_border]

        # Create each row
        for i in range(9):
            row = "|"
            for j in range(10):
                if i == 4 and j == 4:
                    # Player position with direction
                    row += player_char
                elif (j, i) in sprite_locations:
                    # Sprite position
                    row += "S"
                elif (j, i) in relative_warp_coords:
                    # Warp position
                    row += "W"
                else:
                    # Terrain representation
                    if downsampled_terrain[i][j] == 0:
                        row += "█"  # Wall
                    else:
                        row += "·"  # Path
            row += "|"
            lines.append(row)

        # Add bottom border
        lines.append(horizontal_border)

        # Add legend
        lines.extend(
            [
                "",
                "Legend:",
                "█ - Wall/Obstacle",
                "· - Path/Walkable",
                "S - Sprite",
                "W - Warp/Door",
                f"{direction_chars['up']}/{direction_chars['down']}/{direction_chars['left']}/{direction_chars['right']} - Player (facing direction)",
            ]
        )

        # Join all lines with newlines
        return "\n".join(lines)

    def get_valid_moves(self):
        """
        Returns a list of valid moves (up, down, left, right) based on the collision map.
        Returns:
            list[str]: List of valid movement directions
        """
        # Get collision map
        collision_map = self.pyboy.game_wrapper.game_area_collision()  # type: ignore
        terrain = self._downsample_array(collision_map)
        sprites = self.get_sprites()
        reader = PokemonRedReader(self.pyboy.memory)
        warp_coords = reader.get_warps()

        # Player is always at position (4,4) in the 9x10 downsampled map
        valid_moves = []

        player_coords = reader.read_coordinates()
        if player_coords in warp_coords:
            if player_coords[0] and player_coords[1]:  # They're both not 9
                # just yield all moves...
                return ["↑", "↓", "←", "→"]
            if not player_coords[0]:
                valid_moves.append("←")
            if not player_coords[
                1
            ]:  # there is a literal corner case where both are 0, but that never happens in Pokemon Red.
                valid_moves.append("↑")
        # Check each direction
        if terrain[3][4] != 0 and (4, 3) not in sprites:  # Up
            valid_moves.append("↑")
        if terrain[5][4] != 0 and (4, 5) not in sprites:  # Down
            valid_moves.append("↓")
        if terrain[4][3] != 0 and (3, 4) not in sprites:  # Left
            valid_moves.append("←")
        if terrain[4][5] != 0 and (5, 4) not in sprites:  # Right
            valid_moves.append("→")

        return valid_moves

    def _can_move_between_tiles(self, tile1: int, tile2: int, tileset: str) -> bool:
        """
        Check if movement between two tiles is allowed based on tile pair collision data.

        Args:
            tile1: The tile being moved from
            tile2: The tile being moved to
            tileset: The current tileset name

        Returns:
            bool: True if movement is allowed, False if blocked
        """
        # Tile pair collision data
        TILE_PAIR_COLLISIONS_LAND = [
            ("CAVERN", 288, 261),
            ("CAVERN", 321, 261),
            ("FOREST", 304, 302),
            ("CAVERN", 298, 261),
            ("CAVERN", 261, 289),
            ("FOREST", 338, 302),
            ("FOREST", 341, 302),
            ("FOREST", 342, 302),
            ("FOREST", 288, 302),
            ("FOREST", 350, 302),
            ("FOREST", 351, 302),
        ]

        TILE_PAIR_COLLISIONS_WATER = [
            ("FOREST", 276, 302),
            ("FOREST", 328, 302),
            ("CAVERN", 276, 261),
        ]

        # Check both land and water collisions
        for ts, t1, t2 in TILE_PAIR_COLLISIONS_LAND + TILE_PAIR_COLLISIONS_WATER:
            if ts == tileset:
                # Check both directions since collisions are bidirectional
                if (tile1 == t1 and tile2 == t2) or (tile1 == t2 and tile2 == t1):
                    return False

        return True

    def get_warps(self):
        reader = PokemonRedReader(self.pyboy.memory)
        return reader.get_warps()

    def get_sprites(self, debug=False):
        """
        Get the location of all of the sprites on the screen.
        returns set of coordinates that are (column, row)
        """
        # Group sprites by their exact Y coordinate
        sprites_by_y = {}

        for i in range(40):
            sp = self.pyboy.get_sprite(i)
            if sp.on_screen:
                x = int(sp.x / 160 * 10)
                y = int(sp.y / 144 * 9)
                orig_y = sp.y

                if orig_y not in sprites_by_y:
                    sprites_by_y[orig_y] = []
                sprites_by_y[orig_y].append((x, y, i))

        # Sort Y coordinates
        y_positions = sorted(sprites_by_y.keys())
        bottom_sprite_tiles = set()

        if debug:
            print("\nSprites grouped by original Y:")
            for orig_y in y_positions:
                sprites = sprites_by_y[orig_y]
                print(f"Y={orig_y}:")
                for x, grid_y, i in sprites:
                    print(f"  Sprite {i}: x={x}, grid_y={grid_y}")

        SPRITE_HEIGHT = 8

        # First, group sprites by X coordinate for each Y level
        for i in range(len(y_positions) - 1):
            y1 = y_positions[i]
            y2 = y_positions[i + 1]

            if y2 - y1 == SPRITE_HEIGHT:
                # Group sprites by X coordinate at each Y level
                sprites_at_y1 = {s[0]: s for s in sprites_by_y[y1]}  # x -> sprite info
                sprites_at_y2 = {s[0]: s for s in sprites_by_y[y2]}

                # Only match sprites that share the same X coordinate
                for x in sprites_at_y2:
                    if x in sprites_at_y1:  # If there's a matching top sprite at this X
                        bottom_sprite = sprites_at_y2[x]
                        bottom_sprite_tiles.add((x, bottom_sprite[1]))
                        if debug:
                            print(f"\nMatched sprites at x={x}, Y1={y1}, Y2={y2}")

        return bottom_sprite_tiles

    def find_path(self, target_row: int, target_col: int) -> tuple[str, list[str]]:
        """
        Finds the most efficient path from the player's current position (4,4) to the target position.
        If the target is unreachable, finds path to nearest accessible spot.
        Allows ending on a wall tile if that's the target.
        Takes into account terrain, sprite collisions, and tile pair collisions.

        Args:
            target_row: Row index in the 9x10 downsampled map (0-8)
            target_col: Column index in the 9x10 downsampled map (0-9)

        Returns:
            tuple[str, list[str]]: Status message and sequence of movements
        """
        # Get collision map, terrain, and sprites
        collision_map = self.pyboy.game_wrapper.game_area_collision()
        terrain = self._downsample_array(collision_map)
        sprite_locations = self.get_sprites()

        # Get full map for tile values and current tileset
        full_map = self.pyboy.game_wrapper._get_screen_background_tilemap()
        reader = PokemonRedReader(self.pyboy.memory)
        tileset = reader.read_tileset()

        # Start at player position (always 4,4 in the 9x10 grid)
        start = (4, 4)
        end = (target_row, target_col)

        # Validate target position
        if not (0 <= target_row < 9 and 0 <= target_col < 10):
            return "Invalid target coordinates", []

        # A* algorithm
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, end)}

        # Track closest reachable point
        closest_point = start
        min_distance = heuristic(start, end)

        def reconstruct_path(current):
            path = []
            while current in came_from:
                prev = came_from[current]
                if prev[0] < current[0]:
                    path.append("down")
                elif prev[0] > current[0]:
                    path.append("up")
                elif prev[1] < current[1]:
                    path.append("right")
                else:
                    path.append("left")
                current = prev
            path.reverse()
            return path

        while open_set:
            _, current = heapq.heappop(open_set)

            # Check if we've reached target
            if current == end:
                path = reconstruct_path(current)
                is_wall = terrain[end[0]][end[1]] == 0
                if is_wall:
                    return (
                        "Partial Success: Your target location is a wall. In case this is intentional, attempting to navigate there.",
                        path,
                    )
                else:
                    return (
                        f"Success: Found path to target at ({target_row}, {target_col}).",
                        path,
                    )

            # Track closest point
            current_distance = heuristic(current, end)
            if current_distance < min_distance:
                closest_point = current
                min_distance = current_distance

            # If we're next to target and target is a wall, we can end here
            if (abs(current[0] - end[0]) + abs(current[1] - end[1])) == 1 and terrain[end[0]][end[1]] == 0:
                path = reconstruct_path(current)
                # Add final move onto wall
                if end[0] > current[0]:
                    path.append("down")
                elif end[0] < current[0]:
                    path.append("up")
                elif end[1] > current[1]:
                    path.append("right")
                else:
                    path.append("left")
                return (
                    f"Success: Found path to position adjacent to wall at ({target_row}, {target_col}).",
                    path,
                )

            # Check all four directions
            for dr, dc, direction in [
                (1, 0, "down"),
                (-1, 0, "up"),
                (0, 1, "right"),
                (0, -1, "left"),
            ]:
                neighbor = (current[0] + dr, current[1] + dc)

                # Check bounds
                if not (0 <= neighbor[0] < 9 and 0 <= neighbor[1] < 10):
                    continue
                # Skip walls unless it's the final destination
                if terrain[neighbor[0]][neighbor[1]] == 0 and neighbor != end:
                    continue
                # Skip sprites unless it's the final destination
                if (neighbor[1], neighbor[0]) in sprite_locations and neighbor != end:
                    continue

                # Check tile pair collisions
                # Get bottom-left tile of each 2x2 block
                current_tile = full_map[current[0] * 2 + 1][current[1] * 2]  # Bottom-left tile of current block
                neighbor_tile = full_map[neighbor[0] * 2 + 1][neighbor[1] * 2]  # Bottom-left tile of neighbor block
                if not self._can_move_between_tiles(current_tile, neighbor_tile, tileset):
                    continue

                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # If target unreachable, return path to closest point
        if closest_point != start:
            path = reconstruct_path(closest_point)
            return (
                "Partial Success: Could not reach the exact target, but found a path to the closest reachable point.",
                path,
            )

        return (
            "Failure: No path is visible to the chosen location. You may need to explore a totally different path to get where you're trying to go.",
            [],
        )

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
        memory_dict["location"] = location
        # memory_dict["map_coordinates"] = coords
        memory_dict["valid_moves"] = valid_moves if valid_moves and not dialog else []
        memory_dict["badges"] = reader.read_badges()

        # Inventory
        memory_dict["inventory"] = []
        for item, qty in reader.read_items():
            memory_dict["inventory"].append({"item": item, "quantity": qty})

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

            memory_dict["party"].append(pokemon_dict)

        return memory_dict, location, coords

    def stop(self):
        self.pyboy.stop()
