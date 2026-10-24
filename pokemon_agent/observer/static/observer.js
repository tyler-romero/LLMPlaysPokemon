"use strict";

const BADGES = ["BOULDER", "CASCADE", "THUNDER", "RAINBOW", "SOUL", "MARSH", "VOLCANO", "EARTH"];
const INPUT_ICONS = { a: "i-a", b: "i-b", up: "i-up", down: "i-down", left: "i-left", right: "i-right", start: "i-start", select: "i-select" };
const MAX_CHIPS = 40;
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
  inputs: $("inputs"),
  inputsEmpty: $("inputs-empty"),
  inputCount: $("input-count"),
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
  memory: { selected: null, followModel: true, reason: null, activity: new Map(), listing: "" },
};

const MEMORY_ACTIONS = { view: "READ", create: "WROTE", edit: "EDITED", delete: "DELETED", rename: "MOVED" };

// ---------- formatting ----------

const formatDuration = (seconds) => {
  const s = Math.max(0, Math.floor(seconds));
  const h = String(Math.floor(s / 3600)).padStart(2, "0");
  const m = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  return `${h}:${m}:${String(s % 60).padStart(2, "0")}`;
};
const formatTokens = (n) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n));
const agentName = () => (run.agent || "The model").toUpperCase();
const compactArgs = (args) => {
  const text = JSON.stringify(args ?? {});
  return text.length > 160 ? `${text.slice(0, 157)}…` : text;
};

function narrate(name, args) {
  const who = agentName();
  switch (name) {
    case "press_buttons": return `${who} pressed ${(args.buttons || []).join(", ").toUpperCase()}!`;
    case "navigate_to_screen_coordinates": return `${who} walked toward (${args.x}, ${args.y}).`;
    case "on_screen_keyboard_input": return `${who} typed "${args.text}".`;
    case "view": return `${who} read ${args.path}.`;
    case "create": return `${who} wrote ${args.path}.`;
    case "edit": return `${who} updated ${args.path}.`;
    case "delete": return `${who} deleted ${args.path}.`;
    case "rename": return `${who} moved ${args.old_path} to ${args.new_path}.`;
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

// ---------- inputs ----------

function addInput(event, { animate = true } = {}) {
  ui.inputsEmpty.hidden = true;
  const first = ui.inputs.firstElementChild;
  if (first && first.dataset.button === event.button) {
    const count = Number(first.dataset.count) + 1;
    first.dataset.count = String(count);
    first.querySelector(".inputs__count").textContent = `x${count}`;
    return;
  }
  const chip = el("li", "chip");
  chip.dataset.button = event.button;
  chip.dataset.count = "1";
  chip.dataset.held = String(event.wait !== false);
  chip.title = `${event.button.toUpperCase()} · step ${event.step ?? "?"}`;
  chip.append(icon(INPUT_ICONS[event.button] || "i-a"), el("span", "inputs__count", ""));
  chip.setAttribute("aria-label", event.button);
  ui.inputs.prepend(chip);
  while (ui.inputs.children.length > MAX_CHIPS) ui.inputs.lastElementChild.remove();
  if (animate && !reducedMotion.matches) {
    chip.animate([{ transform: "translateY(-4px)", opacity: 0.4 }, { transform: "none", opacity: 1 }], {
      duration: 220,
      easing: "cubic-bezier(0.16, 1, 0.3, 1)",
    });
  }
}

// ---------- model log ----------

function isAtBottom() {
  const list = ui.log;
  return list.scrollHeight - list.scrollTop - list.clientHeight < 64;
}

function pinLog() {
  if (run.follow) ui.log.scrollTop = ui.log.scrollHeight;
}

function appendEntry(node) {
  ui.logEmpty.hidden = true;
  ui.log.append(node);
  while (ui.log.children.length > MAX_LOG_ENTRIES) ui.log.firstElementChild.remove();
  pinLog();
}

// Only the reader's own scrolling changes follow mode; reflows and programmatic pins never do.
let userScrollUntil = 0;
for (const type of ["wheel", "pointerdown", "keydown", "touchstart"]) {
  ui.log.addEventListener(type, () => { userScrollUntil = performance.now() + 1000; }, { passive: true });
}
ui.log.addEventListener("scroll", () => {
  if (performance.now() < userScrollUntil) run.follow = isAtBottom();
  ui.latest.hidden = run.follow;
});
// Content can grow after it is appended (fonts loading, text typing out, results arriving); keep following it.
new ResizeObserver(pinLog).observe(ui.log);
new MutationObserver(pinLog).observe(ui.log, { childList: true, subtree: true, characterData: true });
document.fonts?.ready.then(pinLog);
ui.latest.addEventListener("click", () => {
  ui.log.scrollTop = ui.log.scrollHeight;
  run.follow = true;
  ui.latest.hidden = true;
});

function ensureStepHeading(t) {
  const step = run.step + 1;
  if (run.headingStep >= step) return;
  run.headingStep = step;
  const heading = el("li", "entry entry--step");
  heading.append(el("span", "step__name", `STEP ${step}`));
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
  if (event.total) run.steps = event.total;
  run.stepStartedAt = event.t;
  renderStepCount();
}

function typeOut(node, text) {
  if (reducedMotion.matches) { node.textContent = text; return; }
  const started = performance.now();
  node.textContent = "";
  const tick = (now) => {
    if (node.dataset.done === "true") { node.textContent = text; return; }
    const shown = Math.min(text.length, Math.floor(((now - started) / 1000) * TYPE_RATE));
    node.textContent = text.slice(0, shown);
    if (shown < text.length) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function addThought(event, { live = true } = {}) {
  ensureStepHeading(event.t);
  if (run.latestThought) {
    run.latestThought.classList.remove("is-latest");
    run.latestThought.querySelector("p").dataset.done = "true";
  }
  const box = el("li", "entry entry--thought textbox is-latest");
  if (event.reasoning) box.dataset.reasoning = "true";
  const text = el("p");
  box.append(text, icon("i-more", "advance"));
  run.latestThought = box;
  appendEntry(box);
  if (live) typeOut(text, event.text);
  else text.textContent = event.text;
}

function addTool(event, { live = true } = {}) {
  ensureStepHeading(event.t);
  const row = el("li", "entry entry--tool");
  row.dataset.state = "pending";
  const call = el("button", "tool__call");
  call.type = "button";
  call.setAttribute("aria-expanded", "false");
  call.disabled = true;
  call.append(
    icon("i-cursor", "tool__caret"),
    el("span", "tool__name", event.name),
    el("span", "tool__args", compactArgs(event.args)),
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
  run.tools.set(event.id, { row, call, result });
  appendEntry(row);
  if (live) setActivity("ACTING", narrate(event.name, event.args || {}));
}

function addToolResult(event, { live = true } = {}) {
  const tool = run.tools.get(event.id);
  if (!tool) return;
  const ok = event.ok !== false;
  tool.row.dataset.state = ok ? "ok" : "failed";
  tool.result.textContent = event.result;
  tool.call.disabled = false;
  tool.call.title = ok ? "Succeeded · click to show output" : "Failed · click to show output";
  tool.call.querySelector(".tool__state use").setAttribute("href", ok ? "#i-tool-ok" : "#i-tool-failed");
  if (live) setActivity("THINKING", `${agentName()} is thinking…`);
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
  return `${Math.floor(seconds / 3600)}h ago`;
};

function renderMemory(listing = run.memory.listing) {
  run.memory.listing = listing;
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
    name.append(icon("i-cursor", "cursor"), el("span", "", path));
    button.append(
      name,
      el("span", "memory__activity", activity ? `${activity.action} ${relativeTime(activity.t)}` : ""),
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
    const response = await fetch(`/memory?path=${encodeURIComponent(path)}`);
    const text = await response.text();
    if (text !== ui.memoryView.textContent || changed) {
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
  renderMemory(snapshot.memory);
  audio.sampleRate = snapshot.audioSampleRate || audio.sampleRate;
  drawFrame(snapshot.frame);

  ui.inputs.replaceChildren();
  ui.inputsEmpty.hidden = !snapshot.inputs?.length;
  for (const input of snapshot.inputs || []) addInput(input, { animate: false });
  ui.inputCount.textContent = String(snapshot.inputCount || 0);

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
  ui.logEmpty.hidden = ui.log.children.length > 0;
  if (run.latestThought) run.latestThought.querySelector("p").dataset.done = "true";
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
      run.startedAt = event.t;
      run.stepStartedAt = event.t;
      run.endedAt = null;
      run.steps = event.steps;
      renderStepCount();
      addNotice(`${agentName()} STARTS A ${event.steps}-STEP RUN`);
      if (options.live) setActivity("THINKING", `${agentName()} is thinking…`);
      break;
    case "run_end":
      run.endedAt = event.t;
      addNotice(`RUN OVER AFTER ${event.steps} STEPS`);
      setActivity("FINISHED", "The run is over.");
      break;
    case "state": renderState(event.state); break;
    case "memory":
      renderMemory(event.listing);
      if (run.memory.selected) openMemory(run.memory.selected, run.memory.reason);
      break;
    case "input":
      addInput(event);
      ui.inputCount.textContent = String(Number(ui.inputCount.textContent) + 1);
      break;
    case "usage": $("run-tokens").textContent = `${formatTokens(event.input)} in`; break;
    case "thought": addThought(event, options); break;
    case "tool":
      addTool(event, options);
      noteMemoryActivity(event);
      break;
    case "tool_result": addToolResult(event, options); break;
    case "summary": addSummary(event); break;
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
