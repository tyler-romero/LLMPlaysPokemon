"""Stitch the per-turn text maps in the history into one map of a location, for the `history` tool's `map` action.

Everything comes from what the model was shown: each turn's text map (see Emulator.get_collision_map) and its
press_buttons results. Where views of a cell disagree, the latest one wins; cells never on screen are unknown.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field

from pokemon_agent.history import HistoryItem

MAP_MARKER = "Text-based map:"
# A turn's text map in a dark cave: the current outline-only map, or (older runs) a note that it was withheld.
DARK_MARKERS = ("It's dark here", "too dark here to see")
COORDINATES = re.compile(r'"map_coordinates": \[\s*(-?\d+),\s*(-?\d+)\s*\]')
DIRECTIONS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
# describe_presses clauses: "right: moved to (3, 4)." / "down: you left A and are now in B at (3, 4)."
PRESS = re.compile(
    r"(\w+): (?:(?:moved to|didn't move you; still at|you stayed at) \((-?\d+), (-?\d+)\)"
    r"|you left (.+?) and are now in (.+?) at \((-?\d+), (-?\d+)\))"
)
DOOR = "W"
UNSEEN = "?"


def parse_text_map(text: str):
    """The cells of one turn's text map, as {(x, y): symbol}, plus its level boundaries: vertical ones as the cell
    to their right, horizontal ones as the cell above them. Cells past the map's edge (~) are left out."""
    lines = text.split(MAP_MARKER, 1)[1].strip("\n").split("\n")
    header = lines[0].split()
    if not header or header[0] != "y\\x":
        return {}, set(), set()
    xs = [int(value) for value in header[1:]]
    cells, vertical, horizontal = {}, set(), set()
    y, columns = None, []
    for line in lines[1:]:
        if not line.strip() or line.startswith("Legend"):
            break
        if row := re.match(r"\s*(-?\d+)", line):
            y = int(row[1])
            found = list(re.finditer(r"(\|?)(\S)", line[row.end():]))
            columns = [row.end() + match.start(2) for match in found]
            for x, match in zip(xs, found):
                if match[2] not in "~?":  # past the map's edge, or too dark to see
                    cells[(x, y)] = match[2]
                if match[1]:
                    vertical.add((x, y))
        elif y is not None and columns:
            for position, char in enumerate(line):
                if char == "-":
                    nearest = min(range(len(columns)), key=lambda i: abs(columns[i] - position))
                    horizontal.add((xs[nearest], y))
    return cells, vertical, horizontal


@dataclass
class SeenMap:
    location: str
    cells: dict = field(default_factory=dict)
    vertical: set = field(default_factory=set)
    horizontal: set = field(default_factory=set)
    turns: list = field(default_factory=list)  # turns whose text map was on this location
    you: tuple | None = None  # where the player was on the last turn included, if it was on this location
    # (cell the player stood on, direction) -> [(turn, "MAP (x, y)")] for each press that left this location
    exits: dict = field(default_factory=lambda: defaultdict(list))
    visits: list = field(default_factory=list)  # every turn that started on this location, with or without a text map
    stood: set = field(default_factory=set)  # cells the player stood on (turn starts and press results): walkable
    dark_turns: int = 0  # turns whose text map was taken in the dark (only rock outlines and exits shown)

    def add_view(self, turn: int, text: str):
        cells, vertical, horizontal = parse_text_map(text)
        if not cells:
            return
        self.turns.append(turn)
        # The player's own cell shows @, which hides what is under it (often a door they just came through).
        for cell, symbol in list(cells.items()):
            if symbol == "@":
                if cell in self.cells:
                    del cells[cell]
                else:
                    cells[cell] = "."
        # The latest view decides, including about boundaries it shows both sides of.
        self.vertical -= {(x, y) for x, y in cells if (x - 1, y) in cells}
        self.horizontal -= {(x, y) for x, y in cells if (x, y + 1) in cells}
        self.cells.update(cells)
        self.vertical |= vertical
        self.horizontal |= horizontal

    def add_presses(self, turn: int, start: tuple, result: str):
        position = start
        for press in PRESS.finditer(result.split("Each press:", 1)[-1]):
            button, x, y, old, new, new_x, new_y = press.groups()
            if old is None:
                position = (int(x), int(y))
                self.stood.add(position)
                continue
            if old == self.location and position and button in DIRECTIONS:
                self.exits[(position, button)].append((turn, f"{new} ({new_x}, {new_y})"))
            return

    def all_cells(self) -> dict:
        """Cells as last seen, plus every cell the player stood on (walkable), even where the screen didn't show it."""
        cells = dict(self.cells)
        for cell in self.stood:
            cells.setdefault(cell, ".")
        return cells

    def render(self, x_range=None, y_range=None, as_of: int | None = None) -> str:
        cells = self.all_cells()
        if not cells:
            return f"No record of {self.location} in your history."
        xs = [x for x, _ in cells]
        ys = [y for _, y in cells]
        x0, x1 = x_range or (min(xs), max(xs))
        y0, y1 = y_range or (min(ys), max(ys))
        visits = self.visits or self.turns
        span = f"turns {visits[0]}-{visits[-1]}" if len(visits) > 1 else f"turn {visits[0]}"
        dark = self.dark_turns > 0
        unseen = "never on your screen, or too dark to see" if dark else "never on your screen"
        intro = f"{self.location}: what you have seen of it over your {len(visits)} turns here ({span})"
        intro += f", from the text maps of {len(self.turns)} of them" if self.turns else ""
        intro += (
            ", and every cell you have stood on, which is walkable (.) even where the screen didn't show the ground. "
        )
        if dark:
            intro += (
                "It was dark on some of those turns, so there the screen showed only rock outlines (#) and exits (W), "
                "and ground you haven't stood on is ?. "
            )
        lines = [
            intro + f"Each cell is as you last saw it, and {UNSEEN} marks cells {unseen}. Characters (S) move, so "
            "they are shown where you last saw them.",
            "",
            "y\\x " + "".join(f"{x:>3} " for x in range(x0, x1 + 1)),
        ]
        boundaries = False
        for y in range(y0, y1 + 1):
            row = ""
            for x in range(x0, x1 + 1):
                symbol = "@" if self.you == (x, y) else cells.get((x, y), UNSEEN)
                left = "|" if (x, y) in self.vertical else " "
                below = "_" if (x, y) in self.horizontal else " "
                boundaries = boundaries or left != " " or below != " "
                row += f" {left}{symbol}{below}"
            lines.append(f"{y:>3} {row}".rstrip())
        you = ""
        if self.you:
            you = f"   @ you{f' at turn {as_of}' if as_of else ' now'}"
        lines += [
            "",
            "Legend: . walkable   # blocked   v < > ledge: jump across it in that direction, one way   T small tree: "
            "blocks the way, but can be cut down   ≈ water: can't be walked on, but can be crossed by surfing   W door, "
            "stairs, ladder, or hole   S character or object"
            f"{you}   {UNSEEN} {unseen}",
        ]
        if boundaries:
            # Edges below a cell are marked in its own row: in evals, a separate line of '-' marks was misread.
            lines.append(
                "| just before a cell: the game won't let you step directly between it and the cell to its left. "
                "_ just after a cell: the game won't let you step directly between it and the cell below it. (The "
                "ground changes level there; that ground may be reachable another way.)"
            )
        lines.append("")
        lines += self._exit_lines(unseen)
        return "\n".join(lines)

    def _exit_lines(self, unseen: str = "never on your screen") -> list[str]:
        used: dict[str, list] = defaultdict(list)
        for ((x, y), direction), uses in self.exits.items():
            dx, dy = DIRECTIONS[direction]
            if self.cells.get((x, y)) == DOOR:  # walked off the door or exit the player stood on
                where = f"{DOOR} at ({x}, {y})"
            elif self.cells.get((x + dx, y + dy)) == DOOR:
                where = f"{DOOR} at ({x + dx}, {y + dy})"
            else:
                where = f"pressing {direction} at ({x}, {y})"
            used[where] += uses
        lines = ["Ways off this map you have used (from your button-press results):"]
        for where, uses in sorted(used.items(), key=lambda entry: min(turn for turn, _ in entry[1])):
            turns = sorted(turn for turn, _ in uses)
            shown = ", ".join(map(str, turns[:6])) + (f", ... ({len(turns)} times)" if len(turns) > 6 else "")
            destinations = " / ".join(sorted({destination for _, destination in uses}))
            lines.append(f"* {where} -> {destinations} (turns {shown})")
        if len(lines) == 1:
            lines.append("* none yet")
        unused = sorted(
            (cell for cell, symbol in self.cells.items() if symbol == DOOR and f"{DOOR} at {cell}" not in used),
            key=lambda cell: (cell[1], cell[0]),
        )
        listed = ", ".join(f"({x}, {y})" for x, y in unused) or "none"
        lines.append(
            f"{DOOR} cells you have seen here but never used: {listed}. Cells marked {UNSEEN} were {unseen}, so "
            "there may be more you haven't seen."
        )
        return lines


def stitch(items: list[HistoryItem], location: str, to_turn: int | None = None,
           from_turn: int | None = None) -> SeenMap:
    """Build the SeenMap of `location` from history items (in order), between from_turn and to_turn."""
    seen = SeenMap(location)
    here, start = None, None
    for item in items:
        turn = item.turn or 0
        if to_turn is not None and turn > to_turn:
            break
        if item.kind == "prompt":
            here = item.location
            coordinates = COORDINATES.search(item.text)
            start = (int(coordinates[1]), int(coordinates[2])) if coordinates else None
            seen.you = start if here == location else None
            if here == location and (from_turn is None or turn >= from_turn):
                seen.visits.append(turn)
                if start:
                    seen.stood.add(start)
                if MAP_MARKER in item.text:
                    seen.dark_turns += any(marker in item.text for marker in DARK_MARKERS)
                    seen.add_view(turn, item.text)
        elif item.kind == "tool" and item.tool == "press_buttons" and item.ok and here == location:
            if from_turn is None or turn >= from_turn:
                seen.add_presses(turn, start, item.text)
    return seen
