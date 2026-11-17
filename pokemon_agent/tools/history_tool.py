"""The `history` tool: read-only search and browsing of the model's own session history (Copilot agent)."""

import base64
import logging
import re
from collections import Counter

from pokemon_agent.game.observation import at_model_scale
from pokemon_agent.history import MAX_ITEM_CHARS, SNIPPET_CHARS, HistoryIndex, HistoryItem
from pokemon_agent.history_map import stitch
from pokemon_agent.tools.base import Tool, ToolError, ToolOutput

logger = logging.getLogger(__name__)

KINDS = ["prompt", "response", "tool"]
ACTIONS = ("window", "item", "search", "tools", "image", "map", "summary")
# Punctuation, underscores, and whitespace all separate words, so they're interchangeable when searching.
SEPARATOR = r"[\W_]+"
WORD = re.compile(r"[A-Za-z0-9]+")
# Words match whole words: "5" doesn't match inside "15", and "ladder" doesn't match "ladders" (use "ladder*").
START, END = r"(?<![A-Za-z0-9])", r"(?![A-Za-z0-9])"
# A query's terms: an optional "-", then a "quoted phrase", a /regex/, or a bare word.
TERM = re.compile(r'\s*(-?)(?:"([^"]*)"|/((?:\\.|[^/\\])+)/|(\S+))')
NEVER = re.compile(r"(?!)")


class QueryError(ValueError):
    pass


def _edits_within(a: str, b: str, limit: int) -> bool:
    """Whether a and b are at most `limit` single-character edits apart (Levenshtein, with an early exit)."""
    if abs(len(a) - len(b)) > limit:
        return False
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b)))
        if min(current) > limit:
            return False
        previous = current
    return previous[-1] <= limit


class Query:
    """A search query: every group must match (a group is one term, or terms joined by OR) and no excluded term may.

    Syntax: bare words must all appear, in any order; "quoted words" must appear in that order; `a OR b` matches
    either; `-term` excludes; `word*` matches words starting with it; `~word` allows about one typo per five letters;
    /regex/ is a case-insensitive regular expression. Case, punctuation, and spacing are ignored except in a regex.
    """

    def __init__(self, text: str, vocabulary=None):
        self.text = text
        self.groups: list[list[re.Pattern]] = []
        self.excluded: list[re.Pattern] = []
        self._vocabulary = vocabulary  # () -> set of lowercase words, for ~fuzzy terms
        join_next = False
        position = 0
        while position < len(text):
            match = TERM.match(text, position)
            if not match or match.end() == position:
                break
            position = match.end()
            negate, phrase, regex, word = match.groups()
            if word == "OR" and not negate:
                if not self.groups or join_next:
                    raise QueryError("OR must go between two terms, as in: ladder OR hole")
                join_next = True
                continue
            pattern = self._compile(phrase, regex, word)
            if pattern is None:  # punctuation alone (e.g. a pasted ":" or "["): nothing to search for
                continue
            if negate:
                if join_next:
                    raise QueryError("an excluded -term can't be part of an OR")
                self.excluded.append(pattern)
            elif join_next:
                self.groups[-1].append(pattern)
                join_next = False
            else:
                self.groups.append([pattern])
        if text.count('"') % 2:
            raise QueryError('a quote is missing its closing "')
        if join_next:
            raise QueryError("OR must go between two terms, as in: ladder OR hole")
        if not self.groups:
            raise QueryError("the query needs at least one word or number to find (-terms only exclude)")

    def _compile(self, phrase: str | None, regex: str | None, word: str | None) -> re.Pattern | None:
        if regex is not None:
            try:
                return re.compile(regex, re.IGNORECASE)
            except re.error as e:
                raise QueryError(f"/{regex}/ isn't a valid regular expression ({e})") from None
        source = phrase if phrase is not None else word
        if phrase is None and word.startswith("~") and WORD.fullmatch(word[1:]):
            return self._fuzzy(word[1:].lower())
        prefix = phrase is None and word.endswith("*")
        words = WORD.findall(source)
        if not words:
            return None
        body = SEPARATOR.join(map(re.escape, words))
        return re.compile(START + body + (r"[A-Za-z0-9]*" if prefix else END), re.IGNORECASE)

    def _fuzzy(self, word: str) -> re.Pattern:
        limit = 0 if len(word) <= 3 else max(1, len(word) // 5)
        similar = sorted(v for v in self._vocabulary() if _edits_within(word, v, limit)) if self._vocabulary else []
        if not similar:
            return NEVER
        return re.compile(START + "(?:" + "|".join(map(re.escape, similar)) + ")" + END, re.IGNORECASE)

    def matches(self, text: str) -> bool:
        return all(any(p.search(text) for p in group) for group in self.groups) and not any(
            p.search(text) for p in self.excluded
        )

    def first(self, text: str) -> re.Match | None:
        """The earliest match of any term to find, for a result's snippet."""
        found = [m for group in self.groups for p in group if (m := p.search(text))]
        return min(found, key=lambda m: m.start()) if found else None


class HistoryTool(Tool):
    name = "history"
    description = (
        "Read-only access to your own past turns in this game: each turn's prompt (game state text and screenshot), "
        "your responses (a summary of your reasoning, and your tool calls), and every tool call with its result. Use "
        "it to recover details your condensed history no longer holds, and to check your notes against what actually "
        "happened. It uses no game turns, so a lookup costs far less than finding something out again in the game; "
        "use it freely. It is raw and large, so keep what matters in your memory notes too. The summaries your "
        "conversation was condensed into are kept apart (action=summary). It is eventually consistent: the newest events, from the turn in progress, may not appear yet, "
        "but they will on a later call.\n"
        "Every item has a short ID like #3f9a. Actions:\n"
        "* window (default): one-line summaries of the items around a turn ('turn', 'before', 'after'), or of the "
        "most recent items ('last').\n"
        "* item: the content of one item ('id'), up to 20,000 characters.\n"
        "* search: text search ('query') over prompts (including the game state, such as location names), "
        "responses, and tool calls. Case, punctuation, and spacing are ignored, and words match whole words. Query "
        "syntax: bare words must all appear, in any order (ladder 1F); \"quoted words\" must appear in that order "
        "(\"leaves this map\"; \"58,0\" finds '(58, 0)'); a OR b matches either; -word excludes; word* matches words "
        "starting with it; ~word allows a typo or two; /regex/ is a regular expression. Filter with 'kind', 'tool', "
        "'from_turn', 'to_turn'; 'sort' is 'recent' (default) or 'oldest'; results come 'limit' (default 20) per "
        "'page'. Consecutive matching prompts are grouped into turn ranges.\n"
        "* tools: how often each tool was called and failed ('from_turn', 'to_turn').\n"
        "* image: see an image again: a prompt's screenshot or an image a tool returned ('id', 'page' for items with "
        "several).\n"
        "* map: one map ('location', default the one you are on) stitched together from the text maps of every turn "
        "you spent there and every cell you stood on, each cell as you last saw it and ? where you never saw it (or it "
        "was too dark to see), with the exits you have used "
        "and where they led, and the W cells you have seen but never used. 'x_range' and 'y_range' crop it; "
        "'to_turn' shows it as of an earlier turn.\n"
        "* summary: the summaries your conversation was condensed into, each numbered and titled, newest first; "
        "'query' (same syntax as search) finds the ones that mention something, and 'number' opens one. They hold "
        "what you concluded at the time, which may since have turned out wrong."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["window", "item", "search", "tools", "image", "map", "summary"]},
            "turn": {"type": "integer", "description": "window: the turn to center on."},
            "before": {"type": "integer", "description": "window: turns to include before 'turn' (default 1)."},
            "after": {"type": "integer", "description": "window: turns to include after 'turn' (default 1)."},
            "last": {"type": "integer", "description": "window: show the most recent N items (default 15)."},
            "id": {"type": "string", "description": "item / image: the item's ID, e.g. 3f9a."},
            "query": {"type": "string", "description": "search / summary: what to find (see the query syntax above)."},
            "sort": {
                "type": "string",
                "enum": ["recent", "oldest"],
                "description": "search: newest results first (default) or oldest first.",
            },
            "kind": {"type": "string", "enum": KINDS, "description": "search: only this kind of item."},
            "tool": {"type": "string", "description": "search: only calls to this tool."},
            "from_turn": {"type": "integer", "description": "search / tools / map: first turn to include."},
            "to_turn": {"type": "integer", "description": "search / tools / map: last turn to include."},
            "number": {"type": "integer", "description": "summary: which summary to open, e.g. 84."},
            "location": {
                "type": "string",
                "description": "map: the map's name as the game state shows it (default: where you are now).",
            },
            "x_range": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "map: the first and last x to show, e.g. [10, 40] (default: everything seen).",
            },
            "y_range": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "map: the first and last y to show (default: everything seen).",
            },
            "limit": {"type": "integer", "description": "search: results per page (default 20, max 50)."},
            "offset": {"type": "integer", "description": "search: results to skip (an alternative to 'page')."},
            "page": {
                "type": "integer",
                "description": "search / summary: which page of results; image: which image of the item. Both start at 1.",
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, emulator):
        super().__init__(emulator)
        # Set by the agent that owns a session log; None means history isn't available.
        self.index: HistoryIndex | None = None

    def execute(self, tool_input: dict) -> str | ToolOutput:
        if self.index is None:
            return ToolError("Error: history isn't available for this agent.")
        self.index.refresh()
        action = tool_input.get("action") or "window"
        handler = getattr(self, f"_{action}", None)
        if action not in ACTIONS or handler is None:
            return ToolError(f"Error: unknown action {action!r}; use {', '.join(ACTIONS[:-1])}, or {ACTIONS[-1]}.")
        return handler(tool_input)

    # ----- actions --------------------------------------------------------------------------------------------------

    def _window(self, tool_input: dict) -> str:
        items = self.index.items
        if not items:
            return "No history yet."
        if tool_input.get("turn") is not None:
            turn = tool_input["turn"]
            low = turn - max(0, tool_input.get("before", 1))
            high = turn + max(0, tool_input.get("after", 1))
            chosen = [item for item in items if item.turn is not None and low <= item.turn <= high]
            header = f"Turns {low}-{high} ({len(chosen)} items):"
        else:
            count = min(max(1, tool_input.get("last") or 15), 100)
            chosen = items[-count:]
            header = f"Most recent {len(chosen)} items:"
        if not chosen:
            return f"{header}\n(no items)"
        return "\n".join([header, *(self.index.line(item) for item in chosen)])

    def _item(self, tool_input: dict) -> str:
        item = self._lookup(tool_input)
        if isinstance(item, str):
            return item
        text = item.text if len(item.text) <= MAX_ITEM_CHARS else item.text[:MAX_ITEM_CHARS] + "\n… (truncated)"
        header = f"#{self.index.short_id(item)} · turn {item.turn} · {item.kind}"
        if item.kind == "tool":
            if item.ok:
                outcome = "succeeded"
            elif item.ok is False:
                outcome = "failed"
            else:
                outcome = "no result yet"
            header += f" · {item.tool} · {outcome}"
        if item.images:
            header += f" · {len(item.images)} image(s); see them with action=image"
        return f"{header}\n\n{text}"

    def _image(self, tool_input: dict) -> str | ToolOutput:
        item = self._lookup(tool_input)
        if isinstance(item, str):
            return item
        if not item.images:
            return ToolError(f"Error: #{self.index.short_id(item)} has no images.")
        page = tool_input.get("page") or 1
        if not 1 <= page <= len(item.images):
            return ToolError(f"Error: page must be between 1 and {len(item.images)}.")
        data = self.index.image_b64(item, page - 1)
        if not data:
            return ToolError("Error: that image is no longer in the history log.")
        caption = (
            f"Image {page} of {len(item.images)} from #{self.index.short_id(item)} (turn {item.turn} {item.kind}"
            f"{', ' + item.summary if item.kind == 'prompt' else ''})."
        )
        image = base64.standard_b64encode(at_model_scale(base64.standard_b64decode(data))).decode()
        return ToolOutput(text=caption, images=[image])

    def _search(self, tool_input: dict) -> str:
        text = (tool_input.get("query") or "").strip()
        if not text:
            return ToolError("Error: search needs a query.")
        items = self._filtered(tool_input)
        try:
            query = Query(text, vocabulary=lambda: {w.lower() for item in items for w in WORD.findall(item.text)})
        except QueryError as e:
            return ToolError(f"Error: {e}.")
        matches = [item for item in items if query.matches(item.text)]
        results = self._group(matches, query)
        sort = tool_input.get("sort") or "recent"
        if sort == "recent":
            results.reverse()
        if not results:
            return (
                f"No matches for {text!r}. Words match whole words: use word* for words starting with it, ~word for "
                "near spellings, or fewer terms."
            )
        limit = min(max(1, tool_input.get("limit") or 20), 50)
        pages = (len(results) + limit - 1) // limit
        if tool_input.get("page"):
            page_number = max(1, tool_input["page"])
            offset = (page_number - 1) * limit
        else:
            offset = max(0, tool_input.get("offset") or 0)
            page_number = offset // limit + 1
        shown = results[offset : offset + limit]
        if not shown:
            return f"No more results: there are {len(results)}, {pages} page{'s' if pages != 1 else ''} of {limit}."
        header = f"{len(matches)} matching items in {len(results)} results for {text!r}, {sort} first:"
        footer = (
            f"Page {page_number} of {pages} (results {offset + 1}-{offset + len(shown)} of {len(results)})."
            + (f" Next: page={page_number + 1}." if offset + limit < len(results) else "")
        )
        return "\n".join([header, *shown, footer])

    def _tools(self, tool_input: dict) -> str:
        calls = Counter()
        failures = Counter()
        first, last = {}, {}
        for item in self._filtered({**tool_input, "kind": "tool"}):
            calls[item.tool] += 1
            failures[item.tool] += item.ok is False
            first.setdefault(item.tool, item.turn)
            last[item.tool] = item.turn
        if not calls:
            return "No tool calls in that range."
        lines = [
            f"{name}: {count} calls, {failures[name]} failed, turns {first[name]}-{last[name]}"
            for name, count in calls.most_common()
        ]
        return "\n".join(lines)

    def _map(self, tool_input: dict) -> str:
        to_turn = tool_input.get("to_turn")
        prompts = [item for item in self.index.items if item.kind == "prompt" and item.location]
        if to_turn is not None:
            prompts = [item for item in prompts if (item.turn or 0) <= to_turn]
        if not prompts:
            return "No history yet."
        wanted = (tool_input.get("location") or prompts[-1].location).strip()
        location = next((item.location for item in prompts if item.location.lower() == wanted.lower()), None)
        if location is None:
            recent = list(dict.fromkeys(item.location for item in reversed(prompts)))
            return ToolError(
                f"Error: you have never been on {wanted!r}. Maps you have been on, most recent first: "
                + ", ".join(recent[:40])
                + "."
            )
        ranges = []
        for key in ("x_range", "y_range"):
            value = tool_input.get(key)
            if value is not None and (len(value) != 2 or value[0] > value[1]):
                return ToolError(f"Error: '{key}' must be two numbers, first and last, such as [10, 40].")
            ranges.append(tuple(value) if value is not None else None)
        seen = stitch(self.index.items, location, to_turn=to_turn, from_turn=tool_input.get("from_turn"))
        if not seen.visits:
            return f"You weren't on {location} in that range of turns."
        text = seen.render(*ranges, as_of=to_turn)
        if len(text) > MAX_ITEM_CHARS:
            xs = sorted({x for x, _ in seen.all_cells()})
            ys = sorted({y for _, y in seen.all_cells()})
            return ToolError(
                f"Error: that map is too large to show at once (x {xs[0]}-{xs[-1]}, y {ys[0]}-{ys[-1]}). Show part of "
                "it with 'x_range' and 'y_range'."
            )
        return text

    def _summary(self, tool_input: dict) -> str:
        summaries = self.index.summaries
        if not summaries:
            return "Your conversation hasn't been condensed yet, so there are no summaries."
        if tool_input.get("number") is not None:
            chosen = next((s for s in summaries if s.number == tool_input["number"]), None)
            if chosen is None:
                return ToolError(
                    f"Error: no summary #{tool_input['number']}; they are numbered {summaries[0].number}-"
                    f"{summaries[-1].number}."
                )
            text = chosen.text if len(chosen.text) <= MAX_ITEM_CHARS else chosen.text[:MAX_ITEM_CHARS] + "\n… (truncated)"
            return (
                f"Summary #{chosen.number}, written at turn {chosen.turn} when your conversation was condensed: your "
                f"conclusions then, which may since have turned out wrong.\n\n{text}"
            )
        listed = list(reversed(summaries))
        text = (tool_input.get("query") or "").strip()
        if text:
            try:
                query = Query(text, vocabulary=lambda: {w.lower() for s in listed for w in WORD.findall(s.text)})
            except QueryError as e:
                return ToolError(f"Error: {e}.")
            listed = [s for s in listed if query.matches(s.text)]
            if not listed:
                return f"No summary mentions {text!r}."
        limit = 20
        page = max(1, tool_input.get("page") or 1)
        pages = (len(listed) + limit - 1) // limit
        shown = listed[(page - 1) * limit : page * limit]
        if not shown:
            return f"No more summaries: there are {pages} page{'s' if pages != 1 else ''}."
        header = (
            f"{len(listed)} summar{'ies' if len(listed) != 1 else 'y'}"
            + (f" mentioning {text!r}" if text else "")
            + ", newest first (open one with 'number'):"
        )
        lines = [f"#{s.number}  turn {s.turn}  {s.title}" for s in shown]
        footer = f"Page {page} of {pages}." + (f" Next: page={page + 1}." if page < pages else "")
        return "\n".join([header, *lines, footer])

    # ----- helpers --------------------------------------------------------------------------------------------------

    def _lookup(self, tool_input: dict) -> "HistoryItem | str":
        if not tool_input.get("id"):
            return ToolError("Error: this action needs an 'id'.")
        item = self.index.find(str(tool_input["id"]))
        return item or f"Error: no single item matches id {tool_input['id']!r}."

    def _filtered(self, tool_input: dict) -> list[HistoryItem]:
        kind, tool = tool_input.get("kind"), tool_input.get("tool")
        low, high = tool_input.get("from_turn"), tool_input.get("to_turn")
        return [
            item
            for item in self.index.items
            if (not kind or item.kind == kind)
            and (not tool or item.tool == tool)
            and (low is None or (item.turn or 0) >= low)
            and (high is None or (item.turn or 0) <= high)
        ]

    def _group(self, matches: list[HistoryItem], query: Query) -> list[str]:
        """One line per match in turn order, except that prompts on consecutive turns are grouped into one line."""
        entries: list[tuple[int, int, str]] = []  # (turn, position, line) so results stay in history order
        run: list[HistoryItem] = []

        def flush():
            if len(run) == 1:
                entries.append((run[0].turn or 0, positions[id(run[0])], self._match_line(run[0], query)))
            elif run:
                places = []
                for item in run:
                    if item.location and (not places or places[-1] != item.location):
                        places.append(item.location)
                route = " → ".join(places[:6]) + (" → …" if len(places) > 6 else "")
                ids = f"#{self.index.short_id(run[0])} … #{self.index.short_id(run[-1])}"
                line = f"{ids}  turns {run[0].turn}-{run[-1].turn}  prompt ×{len(run)}  {route}"
                entries.append((run[0].turn or 0, positions[id(run[0])], line))
            run.clear()

        positions = {id(item): position for position, item in enumerate(self.index.items)}
        for item in matches:
            if item.kind != "prompt":
                entries.append((item.turn or 0, positions[id(item)], self._match_line(item, query)))
                continue
            consecutive = run and item.turn is not None and run[-1].turn is not None and item.turn - run[-1].turn <= 1
            if not consecutive:
                flush()
            run.append(item)
        flush()
        return [line for _, _, line in sorted(entries, key=lambda entry: (entry[0], entry[1]))]

    def _match_line(self, item: HistoryItem, query: Query) -> str:
        text = " ".join(item.text.split())
        match = query.first(text)
        at, length = (match.start(), len(match.group())) if match else (0, 0)
        start = max(0, at - SNIPPET_CHARS)
        end = at + length + SNIPPET_CHARS
        snippet = ("…" if start else "") + text[start:end] + ("…" if end < len(text) else "")
        return f"#{self.index.short_id(item)}  turn {item.turn}  {item.kind:<8} {snippet}"
