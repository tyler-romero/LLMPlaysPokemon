"use strict";

const BADGES = ["BOULDER", "CASCADE", "THUNDER", "RAINBOW", "SOUL", "MARSH", "VOLCANO", "EARTH"];
const MAX_LOG_ENTRIES = 600;
const SLOW_STEP_SECONDS = 60;
const TYPE_RATE = 1400; // characters per second for the newest dialogue box
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

const $ = (id) => document.getElementById(id);
const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};
const icon = (id, className) => {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("aria-hidden", "true");
  if (className) svg.setAttribute("class", className);
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#${id}`);
  svg.append(use);
  return svg;
};

const ui = {
  shell: $("shell"),
  connection: $("connection"),
  canvas: $("game"),
  screenEmpty: $("screen-empty"),
  narration: $("narration"),
  log: $("log"),
  logEmpty: $("log-empty"),
  logStatus: $("log-status"),
  latest: $("latest"),
  tools: $("tools"),
  toolsTable: $("tools-table"),
  toolsEmpty: $("tools-empty"),
  toolCount: $("tool-count"),
  objectiveBody: $("objective-body"),
  objectiveMedium: $("objective-medium"),
  objectiveMediumTerm: $("objective-medium-term"),
  objectiveShort: $("objective-short"),
  objectiveEmpty: $("objective-empty"),
  objectiveAge: $("objective-age"),
  party: $("party"),
  partyEmpty: $("party-empty"),
  partyCount: $("party-count"),
  badges: $("badges"),
  badgeCount: $("badge-count"),
  memoryFiles: $("memory-files"),
  memoryEmpty: $("memory-empty"),
  memoryReader: $("memory-reader"),
  memoryCurrent: $("memory-current"),
  memoryFollow: $("memory-follow"),
  memoryView: $("memory-view"),
  sound: $("sound"),
};

const run = {
  agent: null,
  startedAt: null,
  endedAt: null,
  steps: null,
  step: 0,
  stepStartedAt: null,
  headingStep: 0,
  stepHeadings: new Map(),
  tools: new Map(),
  latestThought: null,
  follow: true,
  memory: { selected: null, followModel: true, reason: null, activity: new Map(), listing: "", updated: {} },
};

const MEMORY_ACTIONS = {
  view: "READ", create: "WROTE", edit: "EDITED", insert: "ADDED TO", delete: "DELETED", rename: "MOVED",
  save_screenshot: "SAVED",
};

// ---------- formatting ----------

const formatDuration = (seconds) => {
  const s = Math.max(0, Math.floor(seconds));
  const h = String(Math.floor(s / 3600)).padStart(2, "0");
  const m = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  return `${h}:${m}:${String(s % 60).padStart(2, "0")}`;
};
const formatTokens = (n) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n));
const agentName = () => (run.agent || "The model").toUpperCase();
// Events from a helper (the subagent tool) carry actor: "helper"; the main agent's carry none.
const actorName = (event) => (event.actor === "helper" ? "HELPER" : agentName());
const compactArgs = (args) => {
  const text = JSON.stringify(args ?? {});
  return text.length > 160 ? `${text.slice(0, 157)}…` : text;
};

function narrate(name, args, who = agentName()) {
  switch (name) {
    case "press_buttons": return `${who} pressed ${(args.buttons || []).join(", ").toUpperCase()}!`;
    case "on_screen_keyboard_input": return `${who} typed "${args.text}".`;
    case "view": return `${who} read ${args.path}.`;
    case "list": return `${who} listed ${args.path || "/memories"}.`;
    case "save_screenshot": return `${who} saved a screenshot to ${args.path}.`;
    case "history": return `${who} searched its history${args.query ? ` for "${args.query}"` : ""}.`;
    case "create": return `${who} wrote ${args.path}.`;
    case "edit": return `${who} updated ${args.path}.`;
    case "insert": return `${who} added to ${args.path}.`;
    case "delete": return `${who} deleted ${args.path}.`;
    case "rename": return `${who} moved ${args.old_path} to ${args.new_path}.`;
    case "subagent": return `${who} asked a helper for an analysis.`;
    case "objective": return `${who} set its objective: ${args.short_term ?? args.goal}`;
    case "trigger_compaction": return `${who} asked to condense its history.`;
    case "finish": return `${who} wrapped up its task.`;
    default: return `${who} used ${name.toUpperCase()}!`;
  }
}

function setActivity(label, narration) {
  ui.logStatus.textContent = label;
  if (narration) ui.narration.textContent = narration;
}

// ---------- game screen ----------

const screen = ui.canvas.getContext("2d");
let framePending = null;

function drawFrame(base64) {
  if (!base64) return;
  framePending = base64;
  if (drawFrame.busy) return;
  drawFrame.busy = true;
  const image = new Image();
  image.onload = () => {
    screen.drawImage(image, 0, 0);
    ui.screenEmpty.hidden = true;
    drawFrame.busy = false;
    if (framePending !== base64) drawFrame(framePending);
  };
  image.onerror = () => { drawFrame.busy = false; };
  image.src = `data:image/png;base64,${base64}`;
}

function fitScreen() {
  // Desktop only: 3x when a full-height three-column window has room, otherwise 2x.
  const scale = window.innerWidth >= 1024 && window.innerHeight >= 1040 ? 3 : 2;
  document.documentElement.style.setProperty("--scale", scale);
}
window.addEventListener("resize", fitScreen);
fitScreen();

// ---------- audio ----------

const audio = {
  ctx: null,
  enabled: false,
  sampleRate: 24000,
  nextTime: 0,

  toggle() {
    this.enabled = !this.enabled;
    if (this.enabled) {
      this.ctx ??= new AudioContext();
      this.ctx.resume();
      this.nextTime = 0;
    } else {
      this.ctx?.suspend();
    }
    ui.sound.setAttribute("aria-pressed", String(this.enabled));
    ui.sound.querySelector("use").setAttribute("href", this.enabled ? "#i-sound-on" : "#i-sound-off");
    ui.sound.querySelector(".sound__label").textContent = this.enabled ? "SOUND ON" : "SOUND OFF";
  },

  push(base64) {
    if (!this.enabled || !this.ctx || !base64) return;
    const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
    const samples = new Int8Array(bytes.buffer);
    const frames = samples.length / 2;
    if (!frames) return;
    const buffer = this.ctx.createBuffer(2, frames, this.sampleRate);
    const left = buffer.getChannelData(0);
    const right = buffer.getChannelData(1);
    for (let i = 0; i < frames; i++) {
      left[i] = samples[i * 2] / 128;
      right[i] = samples[i * 2 + 1] / 128;
    }
    const now = this.ctx.currentTime;
    // Keep a small cushion against network jitter; resync if we drift far ahead or fall behind.
    if (this.nextTime < now + 0.03 || this.nextTime > now + 0.6) this.nextTime = now + 0.08;
    const source = this.ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(this.ctx.destination);
    source.start(this.nextTime);
    this.nextTime += buffer.duration;
  },
};
ui.sound.addEventListener("click", () => audio.toggle());

// ---------- status plate ----------

function renderRun(info) {
  if (Array.isArray(info.tools)) setToolDefs(info.tools, info.helper_tools);
  if (info.agent) run.agent = info.agent;
  $("run-agent").textContent = info.agent ? info.agent.toUpperCase() : $("run-agent").textContent;
  if (info.model) $("run-model").textContent = info.model;
  $("run-model").title = [info.provider, info.api].filter(Boolean).join(" · ");
}

function renderStepCount() {
  // While a step is in progress, show its number so the plate matches the newest STEP heading.
  const inProgress = run.startedAt && !run.endedAt && (!run.steps || run.step < run.steps);
  const shown = inProgress ? run.step + 1 : run.step;
  $("run-step").textContent = run.steps ? `${shown} / ${run.steps}` : String(shown);
}

setInterval(() => {
  if (!run.startedAt) return;
  const end = run.endedAt ?? Date.now() / 1000;
  $("run-time").textContent = formatDuration(end - run.startedAt);
}, 1000);

// ---------- where / badges ----------

function renderState(state) {
  if (!state) return;
  $("where-place").textContent = state.location || "Unknown place";
  const coords = state.map_coordinates;
  $("where-coords").textContent = Array.isArray(coords) ? `X ${coords[0]}, Y ${coords[1]}` : "—";
  $("where-money").textContent = state.money ?? "—";
  const named = (name) => name && name !== "Not yet set";
  $("where-trainers").textContent = named(state.player) || named(state.rival)
    ? `${named(state.player) ? state.player : "?"} vs ${named(state.rival) ? state.rival : "?"}`
    : "Names not set yet";

  const earned = new Set(state.badges || []);
  ui.badges.replaceChildren(...BADGES.map((name) => {
    const cell = el("li", "badge");
    cell.dataset.earned = String(earned.has(name));
    cell.title = `${name} BADGE${earned.has(name) ? "" : " (not yet)"}`;
    cell.setAttribute("aria-label", cell.title);
    return cell;
  }));
  ui.badgeCount.textContent = `${earned.size}/8 badges`;

  renderParty(state.pokemon_party || []);
}

function renderParty(party) {
  ui.partyEmpty.hidden = party.length > 0;
  ui.partyCount.textContent = party.length ? `${party.length}/6` : "";
  ui.party.replaceChildren(...party.map((mon) => {
    const [current, max] = String(mon.hp).split("/").map(Number);
    const fraction = max ? Math.max(0, Math.min(1, current / max)) : 0;
    const types = String(mon.types).replace(", ", "/");
    const moves = (mon.moves || []).map((m) => `${m.name} (${m.pp} PP)`).join(", ");
    const row = el("li", "mon");
    row.dataset.fainted = String(current === 0);
    row.title = [mon.species_name, types, moves && `Moves: ${moves}`].filter(Boolean).join("\n");

    const head = el("p", "party__head");
    head.append(el("span", "party__name", mon.nickname || mon.species_name), el("span", "party__status", mon.status || ""), el("span", "party__types", types));
    row.append(head, el("p", "party__level", `:L${mon.level}`));

    const hp = el("div", "party__hp");
    hp.append(el("span", "party__hp-tag", "HP:"));
    const bar = el("div", "party__hp-bar");
    bar.setAttribute("role", "meter");
    bar.setAttribute("aria-label", `${mon.nickname || mon.species_name} HP`);
    bar.setAttribute("aria-valuemin", "0");
    bar.setAttribute("aria-valuemax", String(max || 0));
    bar.setAttribute("aria-valuenow", String(current || 0));
    const fill = el("div", "party__hp-fill");
    fill.style.setProperty("--hp", fraction);
    fill.dataset.level = fraction > 0.5 ? "good" : fraction > 0.2 ? "warn" : "low";
    bar.append(fill);
    hp.append(bar, el("span", "party__hp-num", `${String(current).padStart(3, " ")}/${String(max).padStart(3, " ")}`));
    row.append(hp);
    return row;
  }));
}

// ---------- objective ----------

// { long_term, medium_term, short_term, turn }; earlier objectives lack medium_term, and older runs have { goal, plan }.
// The long term isn't shown: the model tends to restate the whole game's goal there.
let objective = null;

function renderObjective(next = objective) {
  objective = next || null;
  ui.objectiveBody.hidden = !objective;
  ui.objectiveEmpty.hidden = Boolean(objective);
  if (!objective) {
    ui.objectiveAge.textContent = "";
    return;
  }
  const medium = objective.medium_term ?? objective.goal;
  ui.objectiveMedium.textContent = medium || "";
  ui.objectiveMedium.hidden = ui.objectiveMediumTerm.hidden = !medium;
  ui.objectiveShort.textContent = objective.short_term ?? objective.plan;
  const ago = Math.max(0, run.step + 1 - objective.turn);
  ui.objectiveAge.textContent = ago === 0 ? "this turn" : `${ago} turn${ago === 1 ? "" : "s"} ago`;
  ui.objectiveAge.title = `Stated on turn ${objective.turn}`;
}

// ---------- tool stats ----------

// Whole-run totals per tool come in the snapshot (the log only replays recent events); live events update them.
const toolStats = { byName: {}, names: new Map() };

function countTool(event) {
  const stats = (toolStats.byName[event.name] ??= { calls: 0, failed: 0, last: 0 });
  stats.calls += 1;
  stats.last = run.step + 1;
  renderToolStats();
}

function countToolResult(event) {
  const name = toolStats.names.get(event.id);
  toolStats.names.delete(event.id);
  if (name && event.ok === false && toolStats.byName[name]) {
    toolStats.byName[name].failed += 1;
    renderToolStats();
  }
}

function renderToolStats() {
  const entries = Object.entries(toolStats.byName).sort((a, b) => b[1].calls - a[1].calls || a[0].localeCompare(b[0]));
  const total = entries.reduce((sum, [, stats]) => sum + stats.calls, 0);
  const top = entries[0]?.[1].calls || 1;
  const turn = run.step + 1;
  ui.toolCount.textContent = total.toLocaleString();
  ui.toolsTable.hidden = !entries.length;
  ui.toolsEmpty.hidden = entries.length > 0;
  ui.tools.replaceChildren(...entries.map(([name, stats]) => {
    const row = el("tr", "tools__row");
    const label = el("th", "tools__name");
    label.scope = "row";
    if (defs.tools.has(name)) {
      const link = el("button", "tools__link", name);
      link.type = "button";
      link.title = `See ${name}'s definition`;
      link.addEventListener("click", () => openToolDefs(name));
      label.append(link);
    } else {
      label.append(el("span", "", name));
    }
    label.title = name;
    const bar = el("span", "tools__bar");
    bar.style.setProperty("--share", stats.calls / top);
    label.append(bar);
    const share = total ? Math.round((100 * stats.calls) / total) : 0;
    const calls = el("td", "num", stats.calls.toLocaleString());
    calls.title = `${share}% of all calls`;
    const rate = stats.calls ? stats.failed / stats.calls : 0;
    const failed = el("td", "num tools__failed", stats.failed ? `${Math.round(rate * 100)}%` : "—");
    failed.title = `${stats.failed} of ${stats.calls} failed`;
    failed.dataset.level = rate >= 0.25 ? "high" : "";
    const ago = Math.max(0, turn - stats.last);
    const last = el("td", "num tools__last", ago === 0 ? "now" : ago.toLocaleString());
    last.title = `${ago} turns since the last call (turn ${stats.last})`;
    row.append(label, calls, failed, last);
    return row;
  }));
}

// ---------- tool definitions: what the model is told about each tool ----------

const defs = { tools: new Map(), helper: new Set(), dialog: $("defs") };

function setToolDefs(tools, helperTools) {
  defs.tools = new Map(tools.map((tool) => tool.function || tool).filter((fn) => fn?.name).map((fn) => [fn.name, fn]));
  defs.helper = new Set(helperTools || []);
  $("tool-defs").hidden = defs.tools.size === 0;
  renderToolStats();
}

function describeType(schema) {
  if (!schema) return "";
  if (schema.enum) return schema.enum.map((value) => JSON.stringify(value)).join(" | ");
  if (schema.type === "array") return `array of ${describeType(schema.items) || "any"}`;
  return Array.isArray(schema.type) ? schema.type.join(" | ") : schema.type || "";
}

function renderToolDef(fn) {
  const item = el("article", "defs__tool");
  item.id = `def-${fn.name}`;
  const head = el("header", "defs__head");
  head.append(el("h3", "defs__name", fn.name));
  if (defs.helper.has(fn.name)) head.append(el("span", "defs__tag", "helper too"));
  item.append(head, el("p", "defs__desc", fn.description || ""));
  const props = Object.entries(fn.parameters?.properties || {});
  if (!props.length) {
    item.append(el("p", "defs__none", "No arguments."));
    return item;
  }
  const required = new Set(fn.parameters?.required || []);
  const table = el("table", "defs__params");
  const head2 = el("tr");
  for (const text of ["Argument", "Type", "Description"]) head2.append(el("th", "", text));
  table.append(el("thead"), el("tbody"));
  table.tHead.append(head2);
  for (const [name, schema] of props) {
    const row = el("tr");
    const cell = el("td", "defs__arg", name);
    if (required.has(name)) cell.append(el("span", "defs__req", "required"));
    row.append(cell, el("td", "defs__type", describeType(schema)), el("td", "defs__argdesc", schema.description || ""));
    table.tBodies[0].append(row);
  }
  item.append(table);
  return item;
}

function openToolDefs(name) {
  const fns = [...defs.tools.values()];
  $("defs-intro").textContent =
    `The ${fns.length} tools the model can call, exactly as they are described to it.` +
    (defs.helper.size ? ` Tools marked "helper too" are also given to the read-only subagent helper.` : "");
  $("defs-nav").replaceChildren(...fns.map((fn) => {
    const link = el("a", "defs__jump", fn.name);
    link.href = `#def-${fn.name}`;
    link.addEventListener("click", (event) => {
      event.preventDefault();
      document.getElementById(`def-${fn.name}`)?.scrollIntoView({ block: "start" });
    });
    return link;
  }));
  $("defs-body").replaceChildren(...fns.map(renderToolDef));
  if (!defs.dialog.open) defs.dialog.showModal();
  const target = name && document.getElementById(`def-${name}`);
  if (target) target.scrollIntoView({ block: "start" });
  else $("defs-body").scrollTop = 0;
}

$("tool-defs").addEventListener("click", () => openToolDefs());
$("defs-close").addEventListener("click", () => defs.dialog.close());
// A click on the backdrop (outside the dialog's box) closes it too.
defs.dialog.addEventListener("click", (event) => {
  const box = defs.dialog.getBoundingClientRect();
  const inside = event.clientX >= box.left && event.clientX <= box.right && event.clientY >= box.top && event.clientY <= box.bottom;
  if (!inside) defs.dialog.close();
});

// ---------- model log (newest first) ----------

function isAtTop() {
  return ui.log.scrollTop < 64;
}

function pinLog() {
  if (run.follow) ui.log.scrollTop = 0;
}

function appendEntry(node) {
  ui.logEmpty.hidden = true;
  ui.log.prepend(node);
  while (ui.log.children.length > MAX_LOG_ENTRIES) ui.log.lastElementChild.remove();
  pinLog();
}

// Only the reader's own scrolling changes follow mode; reflows and programmatic pins never do.
let userScrollUntil = 0;
for (const type of ["wheel", "pointerdown", "keydown", "touchstart"]) {
  ui.log.addEventListener(type, () => { userScrollUntil = performance.now() + 1000; }, { passive: true });
}
ui.log.addEventListener("scroll", () => {
  if (performance.now() < userScrollUntil) run.follow = isAtTop();
  ui.latest.hidden = run.follow;
});
// Content can grow after it is added (fonts loading, text typing out, results arriving); keep following it.
new ResizeObserver(pinLog).observe(ui.log);
new MutationObserver(pinLog).observe(ui.log, { childList: true, subtree: true, characterData: true });
document.fonts?.ready.then(pinLog);
ui.latest.addEventListener("click", () => {
  ui.log.scrollTop = 0;
  run.follow = true;
  ui.latest.hidden = true;
});

function ensureStepHeading(t, actor) {
  const step = run.step + 1;
  if (run.headingStep >= step) return;
  run.headingStep = step;
  const heading = el("li", "entry entry--step");
  heading.append(el("span", "step__name", actor === "helper" ? `STEP ${step} · HELPER` : `STEP ${step}`));
  const time = el("span", "step__time");
  heading.append(time);
  run.stepHeadings.set(step, { node: heading, time, startedAt: run.stepStartedAt ?? t });
  appendEntry(heading);
}

function completeStep(event) {
  const heading = run.stepHeadings.get(event.step);
  const startedAt = heading?.startedAt ?? run.stepStartedAt ?? event.t;
  const duration = event.t - startedAt;
  if (heading) {
    heading.time.replaceChildren(
      el("b", "", `${duration.toFixed(1)}s`),
      document.createTextNode(run.startedAt ? `  at ${formatDuration(event.t - run.startedAt)}` : ""),
    );
    heading.node.dataset.slow = String(duration > SLOW_STEP_SECONDS);
  }
  run.step = event.step;
  renderToolStats();
  renderObjective();
  if (event.total) run.steps = event.total;
  run.stepStartedAt = event.t;
  renderStepCount();
}

// Types segments ([node, text] pairs) out in order; setting box.dataset.done finishes them at once.
function typeOut(box, segments) {
  const finish = () => { for (const [node, text] of segments) node.textContent = text; };
  if (reducedMotion.matches) { finish(); return; }
  const started = performance.now();
  for (const [node] of segments) node.textContent = "";
  const tick = (now) => {
    if (box.dataset.done === "true") { finish(); return; }
    let budget = Math.floor(((now - started) / 1000) * TYPE_RATE);
    for (const [node, text] of segments) {
      node.textContent = text.slice(0, Math.max(0, budget));
      budget -= text.length;
    }
    if (budget < 0) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

// Reasoning summaries come in titled parts ("**Title**" then the body). Events from older runs carry only the joined
// text, where a later part's title trails the previous sentence ("…area. **Title**\n\nbody"), so split on bold text that
// ends a line.
function thoughtParts(event) {
  if (Array.isArray(event.parts) && event.parts.length) return event.parts;
  // split() with a capture group yields [body, title, body, title, body, ...].
  const pieces = event.text.split(/(?:^|[ \t]+)\*\*([^*\n]+)\*\*[ \t]*(?:\n|$)/m);
  const result = [{ title: null, text: pieces[0].trim() }];
  for (let i = 1; i < pieces.length; i += 2) result.push({ title: pieces[i].trim(), text: (pieces[i + 1] || "").trim() });
  return result.filter((part) => part.title || part.text);
}

// The model's own words to viewers are its dialogue: a text box that types out. A reasoning summary (the provider's
// condensed account of its thinking) is secondary: a collapsed row that previews its titles.
function addThought(event, { live = true } = {}) {
  if (event.reasoning) {
    addReasoning(event);
    return;
  }
  if (event.actor === "helper") {
    addHelperAnswer(event);
    return;
  }
  ensureStepHeading(event.t, event.actor);
  if (run.latestThought) {
    run.latestThought.classList.remove("is-latest");
    run.latestThought.dataset.done = "true";
  }
  const box = el("li", "entry entry--thought textbox is-latest");
  if (event.actor) box.dataset.actor = event.actor;
  const segments = [];
  for (const part of thoughtParts(event)) {
    const section = el("div", "thought__part");
    if (part.title) {
      const title = el("p", "thought__title");
      section.append(title);
      segments.push([title, part.title]);
    }
    const body = el("p");
    section.append(body);
    segments.push([body, part.text]);
    box.append(section);
  }
  box.append(icon("i-more", "advance"));
  run.latestThought = box;
  appendEntry(box);
  if (live) typeOut(box, segments);
  else for (const [node, text] of segments) node.textContent = text;
}

// A collapsed one-line row (caret, bold name, summary, state icon) that reveals its output when clicked.
function toolRow(name, summary, actor) {
  const row = el("li", "entry entry--tool");
  row.dataset.state = "pending";
  if (actor) row.dataset.actor = actor;
  const call = el("button", "tool__call");
  call.type = "button";
  call.setAttribute("aria-expanded", "false");
  call.disabled = true;
  call.append(
    icon("i-cursor", "tool__caret"),
    el("span", "tool__name", name),
    el("span", "tool__args", summary),
    icon("i-tool-pending", "tool__state"),
  );
  const result = el("pre", "tool__result");
  result.hidden = true;
  call.addEventListener("click", () => {
    const open = result.hidden;
    result.hidden = !open;
    call.setAttribute("aria-expanded", String(open));
  });
  row.append(call, result);
  return { row, call, result };
}

function addTool(event, { live = true } = {}) {
  ensureStepHeading(event.t, event.actor);
  const tool = toolRow(event.name, compactArgs(event.args), event.actor);
  run.tools.set(event.id, { ...tool, who: actorName(event) });
  appendEntry(tool.row);
  if (live) setActivity("ACTING", narrate(event.name, event.args || {}, actorName(event)));
}

// A quiet collapsed row (reasoning summaries, a turn's input): an ink-2 label and preview that open onto `body`.
function foldRow(label, preview, actor) {
  const row = el("li", "entry entry--fold");
  if (actor) row.dataset.actor = actor;
  const toggle = el("button", "fold__toggle");
  toggle.type = "button";
  toggle.setAttribute("aria-expanded", "false");
  toggle.append(icon("i-cursor", "tool__caret"), el("span", "fold__label", label), el("span", "fold__preview", preview));
  const body = el("div", "fold__body");
  body.hidden = true;
  toggle.addEventListener("click", () => {
    body.hidden = !body.hidden;
    toggle.setAttribute("aria-expanded", String(!body.hidden));
  });
  row.append(toggle, body);
  return { row, body };
}

function addReasoning(event) {
  ensureStepHeading(event.t, event.actor);
  const parts = thoughtParts(event);
  // Every part's title, so the whole summary's outline shows while collapsed; untitled summaries preview their text.
  const titles = parts.map((part) => part.title).filter(Boolean);
  const preview = titles.length ? titles.join(" · ") : (parts[0]?.text || "").split("\n")[0];
  const { row, body } = foldRow("Reasoning summary", preview, event.actor);
  for (const part of parts) {
    const section = el("div", "thought__part");
    if (part.title) section.append(el("p", "thought__title", part.title));
    section.append(el("p", "", part.text));
    body.append(section);
  }
  appendEntry(row);
}

// The text the model received at the start of a turn (its screenshot is attached separately), previewed by where the
// player is, whether text is on screen, and how full the context is.
function addPrompt(event) {
  ensureStepHeading(event.t, event.actor);
  const text = event.text || "";
  const location = text.match(/"location": "([^"]+)"/)?.[1];
  const coords = text.match(/"map_coordinates": \[\s*(\d+),\s*(\d+)\s*\]/);
  const context = text.match(/Model context usage: [\d,]+ of [\d,]+ tokens \((\d+%)\)/)?.[1];
  const preview = [
    location && (coords ? `${location} (${coords[1]}, ${coords[2]})` : location),
    /"dialog": "/.test(text) && "text on screen",
    context && `${context} of context used`,
  ].filter(Boolean).join(" · ") || text.split("\n")[0];
  const { row, body } = foldRow("Turn input", preview, event.actor);
  body.append(el("pre", "fold__pre", `${text}\n\n[+ screenshot]`));
  appendEntry(row);
}

// A helper's reply text is its answer, which the "Helper answers" box shows in full; log it as a collapsed row.
function addHelperAnswer(event) {
  ensureStepHeading(event.t, event.actor);
  // Preview the first line of prose, skipping markdown headings and emphasis.
  const lines = event.text.split("\n").map((line) => line.replace(/[*_`]/g, "").trim()).filter(Boolean);
  const firstLine = lines.find((line) => !line.startsWith("#")) || lines[0] || "";
  const { row, call, result } = toolRow("answer", firstLine.length > 160 ? `${firstLine.slice(0, 157)}…` : firstLine, "helper");
  row.dataset.state = "ok";
  result.textContent = event.text;
  call.disabled = false;
  call.title = "The helper's reply · click to show it";
  call.querySelector(".tool__state use").setAttribute("href", "#i-tool-ok");
  appendEntry(row);
}

function addToolResult(event, { live = true } = {}) {
  const tool = run.tools.get(event.id);
  if (!tool) return;
  const ok = event.ok !== false;
  tool.row.dataset.state = ok ? "ok" : "failed";
  tool.result.textContent = event.result;
  // Images the tool returned (e.g. an old screenshot from history) show under the row, exactly as the model saw them.
  for (const data of event.images || []) {
    const image = el("img", "tool__image");
    image.src = `data:image/png;base64,${data}`;
    image.alt = `Image returned by ${tool.row.querySelector(".tool__name").textContent}`;
    tool.row.append(image);
  }
  tool.call.disabled = false;
  tool.call.title = ok ? "Succeeded · click to show output" : "Failed · click to show output";
  tool.call.querySelector(".tool__state use").setAttribute("href", ok ? "#i-tool-ok" : "#i-tool-failed");
  if (live) setActivity("THINKING", `${tool.who} is thinking…`);
}

// A helper's work: phase "start" (with its task) and phase "end" (with its answer). A read-only helper analyzes within
// the main agent's step and reports tool calls and seconds. Older runs' helpers played the game for a number of turns,
// and their end event carries the whole report the main agent received; show just the helper's own account from it.
const HELPER_ACCOUNT = /The helper's own account[^\n]*\n([\s\S]*?)\s*<\/helper_report>/;

function addSubagent(event, { live = true } = {}) {
  const start = event.phase === "start";
  const played = event.turns !== undefined || event.used !== undefined;
  const box = el("li", "entry entry--summary entry--helper textbox");
  const head = el("p", "summary__head");
  const title = start ? (played ? "Helper takes over" : "Helper asked") : played ? "Helper reports back" : "Helper answers";
  head.append(el("span", "summary__title", title));
  let stats = "";
  if (start) stats = played ? `up to ${event.turns} turns` : "";
  else if (played) stats = [`${event.used} turns used`, event.reason].filter(Boolean).join(" · ");
  else stats = `${event.tool_calls} tool call${event.tool_calls === 1 ? "" : "s"} · ${event.seconds}s`;
  head.append(el("span", "summary__stats", stats));
  const report = event.report || "";
  const text = start ? event.task : (report.match(HELPER_ACCOUNT)?.[1] ?? report.replace(/<\/?helper_report>\n?/g, "")).trim();
  box.append(head, el("p", "summary__text", text));
  appendEntry(box);
  if (live) setActivity("THINKING", start ? "The HELPER is analyzing…" : `${agentName()} is reading the helper's answer…`);
}

function addSummary(event) {
  const box = el("li", "entry entry--summary textbox");
  const head = el("p", "summary__head");
  head.append(el("span", "summary__title", "History condensed"));
  let stats = "";
  if (event.before && event.after) stats = `${formatTokens(event.before)} → ${formatTokens(event.after)} tokens`;
  else if (event.messages) stats = `${event.messages} messages`;
  if (stats) head.append(el("span", "summary__stats", stats));
  box.append(head, el("p", "summary__text", event.text || "The runtime condensed the conversation but didn't report a summary."));
  appendEntry(box);
}

function addNotice(text) {
  appendEntry(el("li", "notice", text));
}

// ---------- memory ----------

const relativeTime = (t) => {
  const seconds = Math.max(0, Date.now() / 1000 - t);
  if (seconds < 10) return "now";
  if (seconds < 60) return `${Math.floor(seconds)}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
};

function renderMemory(listing = run.memory.listing, updated = run.memory.updated) {
  run.memory.listing = listing;
  run.memory.updated = updated || {};
  const rows = String(listing || "").split("\n").map((line) => line.trim()).filter(Boolean).map((line) => {
    const [size, path] = line.split(/\s+/);
    return { size: Number(size), path };
  });
  ui.memoryEmpty.hidden = rows.length > 0;
  ui.memoryFiles.replaceChildren(...rows.map(({ size, path }) => {
    const item = el("li");
    if (path.endsWith("/")) {
      item.append(el("span", "memory__dir", path));
      return item;
    }
    const activity = run.memory.activity.get(path);
    const button = el("button", "memory__file");
    button.type = "button";
    button.dataset.path = path;
    const name = el("span");
    name.append(icon("i-cursor", "cursor"));
    if (path.toLowerCase().endsWith(".png")) name.append(icon("i-photo", "memory__photo"));
    name.append(el("span", "", path));
    const modified = run.memory.updated[path];
    const updatedLabel = el("span", "memory__updated", modified ? `updated ${relativeTime(modified)}` : "");
    if (modified) updatedLabel.title = `Last updated ${new Date(modified * 1000).toLocaleString()}`;
    button.append(
      name,
      el("span", "memory__activity", activity ? `${activity.action} ${relativeTime(activity.t)}` : ""),
      updatedLabel,
      el("span", "memory__size", `${size} B`),
    );
    if (path === run.memory.selected) button.setAttribute("aria-current", "true");
    button.addEventListener("click", () => {
      run.memory.followModel = false;
      openMemory(path, "YOU OPENED");
    });
    item.append(button);
    return item;
  }));
}
setInterval(() => renderMemory(), 10_000);

function noteMemoryActivity(event) {
  const action = MEMORY_ACTIONS[event.name];
  const path = event.name === "rename" ? event.args?.new_path : event.args?.path;
  if (!action || !path || path.endsWith("/") || path === "/memories") return;
  run.memory.activity.set(path, { action, t: event.t });
  renderMemory();
  if (run.memory.followModel && event.name !== "delete") openMemory(path, `MODEL ${action}`);
}

async function openMemory(path, reason) {
  const changed = path !== run.memory.selected;
  run.memory.selected = path;
  run.memory.reason = reason;
  for (const button of ui.memoryFiles.querySelectorAll("button")) {
    button.toggleAttribute("aria-current", button.dataset.path === path);
    if (button.dataset.path === path) button.setAttribute("aria-current", "true");
  }
  ui.memoryReader.hidden = false;
  ui.memoryCurrent.replaceChildren(el("b", "", reason), document.createTextNode(path));
  ui.memoryFollow.hidden = run.memory.followModel;
  try {
    if (path.toLowerCase().endsWith(".png")) {
      // Reload when the file changes (a screenshot can be deleted and saved again under the same path), and when the
      // first request came too early (the model's call is shown before the file is written).
      const version = String(run.memory.updated[path] ?? "");
      const shown = ui.memoryView.querySelector("img");
      if (changed || !shown || shown.dataset.version !== version) {
        const image = el("img", "memory__image");
        image.src = `/memory?path=${encodeURIComponent(path)}&v=${encodeURIComponent(version)}`;
        image.alt = `Screenshot ${path}`;
        image.dataset.version = version;
        ui.memoryView.replaceChildren(image);
      }
      return;
    }
    const response = await fetch(`/memory?path=${encodeURIComponent(path)}`);
    const text = await response.text();
    if (text !== ui.memoryView.textContent || changed || ui.memoryView.querySelector("img")) {
      ui.memoryView.textContent = text;
      if (!reducedMotion.matches) {
        ui.memoryView.classList.remove("is-fresh");
        void ui.memoryView.offsetWidth;
        ui.memoryView.classList.add("is-fresh");
      }
    }
  } catch {
    ui.memoryView.textContent = "Couldn't load this file. The run may have ended.";
  }
}

ui.memoryFollow.addEventListener("click", () => {
  run.memory.followModel = true;
  ui.memoryFollow.hidden = true;
  const latest = [...run.memory.activity.entries()].sort((a, b) => b[1].t - a[1].t)[0];
  if (latest) openMemory(latest[0], `MODEL ${latest[1].action}`);
});

// ---------- events ----------

function resetLog() {
  ui.log.replaceChildren();
  run.tools.clear();
  run.stepHeadings.clear();
  run.headingStep = run.step;
  run.latestThought = null;
}

function applySnapshot(snapshot) {
  run.step = snapshot.step || 0;
  run.steps = null;
  run.startedAt = null;
  run.endedAt = null;
  run.stepStartedAt = null;
  renderRun(snapshot.run || {});
  renderState(snapshot.state);
  run.memory.activity.clear();
  renderMemory(snapshot.memory, snapshot.memoryUpdated);
  audio.sampleRate = snapshot.audioSampleRate || audio.sampleRate;
  drawFrame(snapshot.frame);

  objective = snapshot.objective || null;
  toolStats.byName = structuredClone(snapshot.toolStats || {});
  toolStats.names.clear();

  run.step = 0;
  resetLog();
  for (const event of snapshot.log || []) handle(event, { live: false });
  run.step = snapshot.step || run.step;
  const info = snapshot.run || {};
  if (info.started_at) run.startedAt = info.started_at;
  if (info.steps) run.steps = info.steps;
  run.endedAt = info.ended_at ?? run.endedAt;
  if (info.input_tokens) $("run-tokens").textContent = `${formatTokens(info.input_tokens)} in`;
  renderStepCount();
  renderToolStats();
  renderObjective();
  ui.logEmpty.hidden = ui.log.children.length > 0;
  if (run.latestThought) run.latestThought.dataset.done = "true";
  run.follow = true;
  ui.latest.hidden = true;
  pinLog();
  if (!snapshot.run?.agent) setActivity("IDLE", "Waiting for a run to start.");
  else if (run.endedAt) setActivity("FINISHED", "The run is over.");
  else setActivity("THINKING", `${agentName()} is thinking…`);
}

function handle(event, options = { live: true }) {
  switch (event.kind) {
    case "snapshot": applySnapshot(event); break;
    case "media":
      drawFrame(event.frame);
      audio.push(event.audio);
      break;
    case "run": renderRun(event); break;
    case "run_start":
      run.startedAt ??= event.t;
      run.stepStartedAt = event.t;
      run.endedAt = null;
      run.steps = event.steps;
      renderStepCount();
      addNotice(event.resumed_from
        ? `RESUMED AT STEP ${event.resumed_from}`
        : `${agentName()} STARTS A ${event.steps}-STEP RUN`);
      if (options.live) setActivity("THINKING", `${agentName()} is thinking…`);
      break;
    case "run_end":
      run.endedAt = event.t;
      addNotice(`RUN OVER AFTER ${event.steps} STEPS`);
      setActivity("FINISHED", "The run is over.");
      break;
    case "state": renderState(event.state); break;
    case "memory":
      renderMemory(event.listing, event.updated);
      if (run.memory.selected) openMemory(run.memory.selected, run.memory.reason);
      break;
    case "usage": $("run-tokens").textContent = `${formatTokens(event.input)} in`; break;
    case "thought": addThought(event, options); break;
    case "prompt": addPrompt(event); break;
    case "tool":
      addTool(event, options);
      noteMemoryActivity(event);
      // Replayed log events are already in the snapshot's totals; only remember their names for later results.
      toolStats.names.set(event.id, event.name);
      if (options.live) countTool(event);
      break;
    case "tool_result":
      addToolResult(event, options);
      if (options.live) countToolResult(event);
      else toolStats.names.delete(event.id);
      break;
    case "summary": addSummary(event); break;
    case "subagent": addSubagent(event, options); break;
    case "objective": renderObjective(event); break;
    case "checkpoint": appendEntry(el("li", "entry entry--checkpoint", `Checkpoint saved at step ${event.step}`)); break;
    case "notice": addNotice(event.text); break;
    case "step": completeStep(event); break;
  }
}

function connect() {
  const source = new EventSource("/events");
  source.onopen = () => {
    ui.shell.dataset.connection = "live";
  };
  source.onerror = () => {
    ui.shell.dataset.connection = "lost";
    ui.connection.textContent = "RECONNECTING";
  };
  source.onmessage = (message) => {
    ui.shell.dataset.connection = "live";
    handle(JSON.parse(message.data));
  };
}

connect();
