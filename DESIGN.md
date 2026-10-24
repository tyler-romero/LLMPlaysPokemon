---
name: LLM Plays Pokémon Observer
world: Pokémon Red field kit
source: pokemon_agent/observer/static/observer.css (all tokens live in :root)
colors:
  paper: "#f8f8f0"
  ink: "#181818"
  ink-2: "#55554d"
  on-plate: "#f8f8f0"
  on-plate-2: "#c4c4b8"
  hp-good: "#58a858"
  hp-warn: "#f8c030"
  hp-low: "#e04838"
typography:
  pixel:
    fontFamily: '"Press Start 2P", ui-monospace, "SF Mono", Menlo, monospace'
    fontSize: 16px
    lineHeight: 24px
    fontWeight: 400
    textTransform: uppercase
  prose:
    fontFamily: 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
    fontSize: 15px
    lineHeight: 1.5
  dialogue:
    fontFamily: prose
    fontSize: 14px
    lineHeight: 1.45
  meta:
    fontFamily: prose
    fontSize: 12px
  data:
    fontFamily: 'ui-monospace, "SF Mono", Menlo, Consolas, monospace'
    fontSize: 12px-13px
    lineHeight: 1.4-1.5
spacing:
  tile: 8px
  px: 2px
  gutter: 16px
rounded:
  none: 0
  notch: "2px chamfer via clip-path polygon (--notch)"
components:
  textbox: "paper fill, 2px ink rule + 2px paper gap + 2px ink inner rule (double rule), notched corners"
  plate: "ink fill, on-plate text, notched corners"
  cursor: "8x8 pixel ▶ SVG symbol #i-cursor"
---

# DESIGN.md — LLM Plays Pokémon observer

The observer is a read-only, desktop-only watch page for a running agent. It speaks Pokémon Red's own interface: the model's turns arrive as dialogue text boxes, the team is the party screen, and the narration box describes each action the way the game's battle text does. No dark overlay, no card grid. Everything is built from the parts listed below.

## Colors

- **Two inks on paper.** `--paper #f8f8f0` and `--ink #181818` make up the whole page. `--ink-2 #55554d` (7.0:1 on paper) is the only secondary text color. On plates, `--on-plate` is paper and `--on-plate-2 #c4c4b8` is for asides and `dt` labels.
- **HP colors are the only hues.** `--hp-good`, `--hp-warn` and `--hp-low` fill the HP bars, switching at >50%, >20% and below that. They are also used, sparingly, as status signals: the tool ✓ is green and the ✗ is red; a failed tool name and a step over 60s are underlined in red (the text itself stays ink); RECONNECTING is yellow on the plate; a refreshed memory file flashes from a 35% yellow wash. No other hue exists. Red text is 3.8:1 on paper, so red never colors text: pair it with ink text as an icon, fill, or underline.
- Gray is never a tinted surface. Hover is `color-mix(ink 8–10%)` and party row dividers are `ink 20%`.
- The light scheme is the only scheme (`color-scheme: light`). Selection inverts to ink on paper.

## Typography: three voices

1. **Pixel** (`--pixel`, Press Start 2P, loaded from Google Fonts at runtime and falling back to monospace): **one size, 16px**, 24px line (3 tiles), uppercase, no smoothing. It is the game's own voice: plate and section titles, narration (sentence case), party names/levels/HP numbers/status, input counts, the location name, the "History condensed" title, run notices, the LATEST and SOUND buttons, and the screen's empty state. Rank never comes from a pixel size change. The 18–20px party line-heights only tighten rows; they don't change the size.
2. **Prose** (`--prose`, system-ui): for reading. Body 15/1.5; dialogue and summary text 14/1.45 capped at 80ch; plate facts 13px (`dd` 600 weight); step rules, types, summary stats and the connection line at 12px; memory activity 11px bold.
3. **Data** (`--data`, system monospace): anything machine-literal, like tool calls (12.5/1.4, name bold, args in ink-2 with an ellipsis), tool output (12/1.5), memory paths (13px) and file contents (12/1.5, pre-wrap).

Numbers are `tabular-nums` site-wide. No font files or licenses are checked into the repo.

## The tile grid

`--tile: 8px` is the unit. Gutters and shell padding are 2 tiles (16px), and stacks use 1 or ½ tile. Rules are `--px: 2px`; hairlines (1px) are used only for step rules, the tool-result spine, party dividers and the Follow button. Corners are never rounded. They are chamfered by `--notch`, a 2px clip-path polygon that applies to every framed or inverted shape.

## The text box

`.textbox` is the game's dialogue frame: paper fill, a 2px ink border, then an inset 2px paper gap and a 2px ink inner rule, so it reads as a double rule. Corners are notched, with padding of 2 tiles + 2px. It frames:
- **Dialogue** (`entry--thought`), with tighter 1 tile × 1.5 tile padding, 14px prose. When `data-reasoning` is set, the box loses the inner rule and its text turns ink-2.
- **Narration** under the game screen, in pixel, sized to the screen width.
- **Where** (location + badges), with compact padding.
- **History condensed** (`entry--summary`): a pixel title, `before → after tokens` stats right-aligned at 12px, and the summary text shown by default.

## Inverted plates and their rank

Rank comes from inversion and position, never from scale.
1. **Status plate:** a full-width black bar across the top that holds the title, run facts (`dt` on-plate-2, `dd` bold), the connection state and SOUND. It shows once per page.
2. **Section labels** (`.label`): a black plate with the section title in pixel on the left and a live aside (`on-plate-2`, original case) on the right: PARTY 6/6, MODEL THINKING, INPUTS count, MEMORY /memories.
3. **Selected item:** inversion marks the one thing that is current, such as the open memory file row, the reader bar above its contents, a pressed SOUND toggle, and the LATEST jump button.
Dark fills appear only on these three ranks and inside the game's bezel.

## The ▶ cursor

`#i-cursor` is an 8×8 pixel ▶ drawn as an SVG with `crispEdges`. It marks the current choice the way the game menu does. It sits next to the open memory file (hidden on other rows) and is the tool-row caret, which rotates 90° when the row is expanded. Its counterpart, `#i-more` ▼, blinks at the bottom right of the newest dialogue box and leads the LATEST button. All icons are pixel SVG symbols in one sprite sheet, never text glyphs. The d-pad and button icons use `--icon-lit` for their lit pixels.

## Components

- **Game screen:** a 160×144 canvas at integer scale (`--scale` 3 when the viewport is ≥1024 wide and ≥1040 tall, otherwise 2) with `image-rendering: pixelated`, inside a 1-tile black bezel. Nothing overlaps or crops it.
- **Party screen:** two lines per Pokémon like the in-game menu. Line 1 is the name, types (12px ink-2) and `:L` level; line 2 is `HP:`, the bar, and `cur/ max`. The list caps at 560px wide and must fit 6 Pokémon at 1440×800 without scrolling. A fainted Pokémon turns ink-2.
- **Model log:** each step opens with a 12px prose rule (`STEP n` bold ink on the left; `0.3s at 00:00:00` on the right) followed by its dialogue boxes and tool rows. A **tool row** is one collapsed line: ▶ caret, bold tool name, args, and a 12px pixel state icon (pending ink-2 blinking, ok green ✓, failed red ✗). Clicking reveals the output under a 1px left spine, capped at 40vh. Run start and end appear as centered pixel notices. The list keeps the newest 600 entries.
- **Inputs:** 32px pixel button icons, newest first. Repeated presses collapse into `x3`. The newest chip carries a 2px ink frame. Non-held presses drop to 60% opacity. The list keeps 40 chips.
- **Memory:** a monospace file list with size and the latest model action (`READ 11s ago`). Below it is the reader: a framed panel whose inverted bar states the open file and the reason, either `MODEL READ`/`MODEL WROTE`/… or `YOU OPENED`, plus a **Follow model** button while the reader is pinned. Memory gets the most vertical space in its column.

## Motion grammar

- **Game-native motion is stepped.** The HP bar drains in `600ms steps(12)`, and its color switches after the drain. The cursor, the advance arrow and the pending tool icon blink at `1s steps(1)`. The newest dialogue box types out at 1400 chars/s; older boxes appear whole.
- **Interface affordances ease out** with `cubic-bezier(0.16, 1, 0.3, 1)`: the caret rotates in 160ms, a new input chip drops 4px in 220ms, and a refreshed memory view washes in 900ms.
- Layout properties never animate. `prefers-reduced-motion` turns off all of the above, including the typewriter.

## Layout

- **≥1024px (primary):** a 100dvh shell with no page scroll. The plate sits above three columns: `auto | 1.6fr | minmax(300px, 1fr)`.
  - Left stage: game, narration and party. The party flexes to fill the remaining height and scrolls only as a last resort.
  - Middle: the model log, which scrolls on its own.
  - Right: where/badges (compact), inputs (max 34% height, scrolls internally) and memory (flex fill, reader unbounded).
  - At max-height 880px the narration box compacts.
- **<1024px (medium, stacked):** plate; game beside where and inputs; the log at 70vh; then party and memory full-width. Mobile is out of scope.

## States

- **Connection:** `CONNECTING` (on-plate-2), hidden when live, `RECONNECTING` in hp-warn when lost.
- **Model status aside:** IDLE / THINKING / ACTING. The narration box mirrors it ("COPILOT is thinking…", "COPILOT pressed UP!").
- **Follow:** the log follows the newest entry until the reader scrolls away. LATEST then appears bottom-right, and clicking it re-pins. Only user input changes follow mode.
- **Empty states**, in plain words inside each section: "Waiting for the emulator…", "No Pokémon yet.", "The model's turns will appear here.", "No buttons pressed yet.", "No memory files yet."
- **Failure:** a failed tool shows a red ✗ and a red underline under its name, and its output stays one click away. An unreadable memory file reads "Couldn't load this file. The run may have ended."

## Do / Don't

- **Do** keep pixel type at 16px and rank with plates, case and the ▶ cursor.
- **Do** put long model text in prose inside a text box. Pixel type is for short game-voice strings.
- **Do** frame new surfaces with `.textbox` or a `.label` plate and the `--notch` chamfer.
- **Don't** add a hue beyond the three HP colors, or tint surfaces.
- **Don't** round corners, add drop shadows, or scale the game screen to a non-integer size.
- **Don't** add write controls. The page is read-only apart from view toggles (sound, tool output, memory file).
