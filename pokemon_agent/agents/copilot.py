"""Agent built on the GitHub Copilot SDK, which runs the model/tool loop and compacts context on its own."""

import asyncio
import inspect
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from copilot import CopilotClient, Tool, ToolInvocation, ToolResult, ToolSet
from copilot.generated.rpc import MetadataContextInfoRequest, SessionHistoryCompactRequest, Trigger
from copilot.tools import ToolBinaryResult
from copilot.session import PermissionHandler
from copilot.session_events import (
    AssistantMessageData,
    AssistantUsageData,
    SessionCompactionCompleteData,
    SessionErrorData,
    SessionEvent,
    SessionWarningData,
    ToolExecutionStartData,
)

from pokemon_agent.agents.base import BaseAgent
from pokemon_agent.agents.copilot_config import CopilotAgentConfig
from pokemon_agent.game.observation import GameObservation
from pokemon_agent.agents.prompts import (
    AFTER_FAILURE_REFUSAL,
    AFTER_GAME_ACTION_REFUSAL,
    GAME_ACTION_NOT_LAST_REFUSAL,
    HELPER_ANSWER_TEMPLATE,
    HELPER_TASK_PREFIX,
    OBJECTIVE_DUE_PREFIX,
    CONDENSED_NOTE,
    NOTICES_HEADER,
    RESUME_STEP_PREFIX,
    START_STEP_PREFIX,
    STEP_PREFIX,
    helper_system_prompt,
    system_prompt,
)
from pokemon_agent.history import HistoryIndex
from pokemon_agent.runs import copy_tree
from pokemon_agent.tools.base import ToolError, ToolOutput
from pokemon_agent.tools.factory import DEFAULT_TOOL_NAMES, HELPER_TOOL_NAMES, ToolFactory

logger = logging.getLogger(__name__)

# Runtime lock files inside a session folder; never copied into or out of checkpoints.
SESSION_RUNTIME_FILES = ("*.lock", "*.hold")

# Our memory tools reuse these Copilot built-in names; the SDK requires an explicit override to register them.
COPILOT_BUILTIN_TOOL_NAMES = {"view", "create", "edit"}

# Tools whose results the model must read before choosing a game action, so it can't list one before its game action.
READ_TOOLS = {"view", "list", "history", "subagent"}

CANCELLED_REFUSAL = "Not run: the response it was part of timed out and was cancelled."
READ_BEFORE_ACTION_REFUSAL = (
    "Not run: you called {reads} earlier in this response, and you need to read those results before choosing your "
    "game action. Read them, then call your game action in a new response."
)
OBJECTIVE_DENIAL = (
    "Not run: this turn you must call 'objective' to state your current goal and plan before your game action. Call "
    "it (earlier in the same response is fine), then your game action."
)

HELPER_BUDGET_DENIAL = "Not run: you have used all your tool calls. Answer now with what you have found."

# A reasoning summary part reads "**Title**\n\nbody".
REASONING_TITLE = re.compile(r"\*\*(.+?)\*\*\s*\n+(.*)", re.DOTALL)


@dataclass
class _Batch:
    """The tool calls of one model response. They run one at a time in the order listed, and the first failure stops
    the rest (see CopilotAgent._sdk_tools)."""

    calls: list[tuple[str, str]] = field(default_factory=list)  # (tool_call_id, tool name), in the order listed
    failed: str | None = None  # the first call that failed, by tool name


def _quiet_unknown_event_errors(loop: asyncio.AbstractEventLoop, context: dict):
    # The bundled runtime can emit event fields newer than the SDK's parser accepts; those events are dropped either
    # way, so log one line instead of a full traceback.
    if isinstance(context.get("exception"), AssertionError) and "handle_notification" in repr(context.get("handle")):
        logger.debug(f"Ignored unparseable Copilot session event: {context.get('handle')!r:.300}")
        return
    loop.default_exception_handler(context)


class CopilotAgent(BaseAgent):
    """Agent that delegates the model/tool loop and context management to a Copilot SDK session.

    Each step sends the current game state and screenshot as one message. The model can call tools freely within the
    step, several per response if it likes: they run one at a time in the order listed, and the first failure stops the
    rest of that response. Tools with `ends_turn` (game actions) end the step, so the model always acts on a fresh
    screenshot; one may only be the last call of a response, and not after a read (see _refusal).

    The `subagent` tool asks a read-only helper for an analysis: a second session with a fresh context and only
    HELPER_TOOL_NAMES (memory and history reads). It runs inside the tool call, within the main agent's step, and its
    final reply is the tool result.
    """

    # The SDK keeps a session log, so this agent also gets the read-only `history` tool over it.
    tool_names = [*DEFAULT_TOOL_NAMES, "history", "subagent", "objective", "trigger_compaction"]

    def __init__(self, rom_path, config: CopilotAgentConfig, **kwargs):
        """Initialize the Copilot SDK agent.

        Args:
            rom_path: Path to the ROM file
            config: Copilot SDK model and session settings
            **kwargs: BaseAgent options (headless, sound, save_every, load_state, observer, run)
        """
        super().__init__(rom_path, **kwargs)
        self.config = config
        self.system_prompt = system_prompt(config.model)
        # In a durable run, the SDK's session lives inside the run folder under a stable ID so it can be resumed.
        self.copilot_home = self.run_store.root / "copilot-sdk" if self.run_store else Path(config.copilot_home)
        self.session_id = self.run_store.run_id if self.run_store else None
        self.resume_session = False

        self.helper_factory = ToolFactory(self.emulator, HELPER_TOOL_NAMES, memory=self.memory)
        for tool in self.helper_factory.tools.values():
            tool.emit = self.emit
        self._history: HistoryIndex | None = None
        # The number of the summary the conversation was just condensed into, until the next turn mentions it.
        self._condensed: int | None = None
        if self.session_id:
            index = self._history = HistoryIndex(self._session_dir() / "events.jsonl", self.run_store.events_path)
            for factory in (self.tool_factory, self.helper_factory):
                if history_tool := factory.tools.get("history"):
                    history_tool.index = index
        self._game_actions = {tool.name for tool in self.tool_factory.tools.values() if tool.ends_turn}
        # The tool calls of the latest response from the main agent and from its helper, from their assistant.message.
        self._batch = _Batch()
        self._helper_batch = _Batch()
        self._cancelled_calls: set[str] = set()  # calls of a response that timed out; any still dispatched are refused
        self._tool_locks: dict[bool, asyncio.Lock] = {}  # each session's tool lock, by whether it's the helper's
        self._helper_session = None  # the running helper's SDK session, so a timed-out turn can abort it too
        self._client: CopilotClient | None = None  # set while running, for helper sessions
        self._observation: GameObservation | None = None  # what started the current step; helpers reuse it
        self._helper_text = ""  # the helper's latest text reply: its answer once it stops calling tools
        self._helper_tool_calls = 0
        self._objective_set = False  # whether 'objective' was called this step
        if objective_tool := self.tool_factory.tools.get("objective"):
            objective_tool.on_set = lambda: setattr(self, "_objective_set", True)
        # Set by 'trigger_compaction' to its custom instructions ("" for none); the conversation is condensed after the
        # turn. None when no compaction is requested.
        self._compaction_request: str | None = None
        if compaction_tool := self.tool_factory.tools.get("trigger_compaction"):
            compaction_tool.on_request = lambda instructions: setattr(self, "_compaction_request", instructions or "")
        if self.resume_checkpoint:
            self.restore_agent_state(self.resume_checkpoint / "agent")
        logger.info(f"Using Copilot SDK with GitHub Copilot model {config.model or 'default'!r}")
        self.emit(
            "run",
            agent="copilot",
            model=config.model or "Copilot default",
            provider="GitHub Copilot",
            tools=self.tool_factory.get_available_tools(),  # the observer shows these definitions on request
            helper_tools=list(self.helper_factory.tools),
        )

    def _sdk_tools(self, factory: ToolFactory, helper: bool = False) -> list[Tool]:
        """The factory's tools for an SDK session. Each session's calls run one at a time, in the order the model
        listed them; once one fails, the rest of that response isn't run (each gets a "Not run" result)."""
        lock = self._tool_locks[helper] = asyncio.Lock()

        def in_order(name: str, handler):
            async def run(invocation: ToolInvocation) -> ToolResult:
                async with lock:
                    batch = self._helper_batch if helper else self._batch
                    if refusal := self._refusal(batch, invocation.tool_call_id, name):
                        batch.failed = batch.failed or name
                        return self._refuse(name, invocation.arguments, refusal)
                    try:
                        result = handler(invocation)
                        if inspect.isawaitable(result):
                            result = await result
                    except Exception as e:  # (run_tool has already reported it to the observer)
                        logger.error(f"[Tool] {name} raised", exc_info=True)
                        error = f"Error: {name} failed unexpectedly ({type(e).__name__}: {e})."
                        result = ToolResult(text_result_for_llm=error, result_type="failure", error=error)
                    if result.result_type == "failure":
                        batch.failed = batch.failed or name
                    return result

            return run

        async def helper_handler(invocation: ToolInvocation) -> ToolResult:
            args = invocation.arguments or {}
            self._tool_calls += 1
            call_id = self._tool_calls
            self.emit("tool", id=call_id, name="subagent", args=args)
            try:
                answer = await self._run_helper((args.get("task") or "").strip())
            except Exception as e:
                logger.error("[Helper] Failed", exc_info=True)
                answer = ToolError(f"Error: the helper failed ({type(e).__name__}: {e}).")
            failed = isinstance(answer, ToolError)
            self.emit("tool_result", id=call_id, result=answer, ok=not failed)
            if failed:
                return ToolResult(text_result_for_llm=answer, result_type="failure", error=answer)
            return ToolResult(text_result_for_llm=answer)

        def make_handler(name: str):
            def handler(invocation: ToolInvocation) -> ToolResult:
                result = self.run_tool(name, invocation.arguments or {}, factory)
                if isinstance(result, ToolOutput):
                    images = [
                        ToolBinaryResult(data=image, mime_type="image/png", type="image") for image in result.images
                    ]
                    return ToolResult(text_result_for_llm=result.text, binary_results_for_llm=images)
                if isinstance(result, ToolError):
                    # A failure doesn't end the turn even for a game action (one fails only when nothing happened in
                    # the game), so the model can read why and retry right away.
                    return ToolResult(text_result_for_llm=result, result_type="failure", error=result)
                return ToolResult(text_result_for_llm=result)

            return handler

        return [
            Tool(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                handler=in_order(tool.name, helper_handler if tool.name == "subagent" else make_handler(tool.name)),
                skip_permission=True,
                is_terminal=tool.ends_turn,
                overrides_built_in_tool=tool.name in COPILOT_BUILTIN_TOOL_NAMES,
            )
            for tool in factory.tools.values()
        ]

    def _session_options(self, system_message: str, factory: ToolFactory, pre_tool_hook=None, helper=False) -> dict:
        options = dict(
            on_permission_request=PermissionHandler.approve_all,
            model=self.config.model,
            system_message={"mode": "replace", "content": system_message},
            available_tools=ToolSet().add_custom("*"),
            tools=self._sdk_tools(factory, helper),
            reasoning_effort=self.config.reasoning_effort,
        )
        if pre_tool_hook:
            options["hooks"] = {"on_pre_tool_use": pre_tool_hook}
        return options

    def _step_message(
        self, prefix_content: str, observation: GameObservation | None = None, with_memory_listing: bool = False
    ) -> tuple[str, list[dict]]:
        """Build the step prompt text and screenshot attachment from a new observation of the game, or a given one.
        The memory file list is included only on request (a helper's one message); the main agent uses 'list'."""
        notices = []
        if observation is None:  # the main agent's turn (a helper reuses the turn's observation)
            observation = self._observation = self.observe()
            if lines := self.notices():
                notices = [NOTICES_HEADER, *(f"- {line}" for line in lines)]
        sections = self.game_state_sections(observation, self.memory.list({}) if with_memory_listing else None)
        prompt = "\n".join([prefix_content, *notices, *sections])
        attachments = [
            {"type": "blob", "data": observation.screenshot_b64, "mimeType": "image/png", "displayName": "screenshot"}
        ]
        return prompt, attachments

    def _refusal(self, batch: _Batch, call_id: str, name: str) -> str | None:
        """Why a call in `batch` must not run, or None. Calls run in the order listed, so everything listed before this
        one has already run."""
        if call_id in self._cancelled_calls:
            return CANCELLED_REFUSAL
        ids = [listed_id for listed_id, _ in batch.calls]
        if call_id in ids:  # (a call missing from the list can't be placed in it, so only the objective rule applies)
            position = ids.index(call_id)
            earlier = [listed for _, listed in batch.calls[:position]]
            if any(listed in self._game_actions for listed in earlier):
                return AFTER_GAME_ACTION_REFUSAL
            if batch.failed:
                return AFTER_FAILURE_REFUSAL.format(failed=batch.failed)
            if name in self._game_actions and position != len(ids) - 1:
                return GAME_ACTION_NOT_LAST_REFUSAL
            if name in self._game_actions and (reads := sorted({n for n in earlier if n in READ_TOOLS})):
                return READ_BEFORE_ACTION_REFUSAL.format(reads=", ".join(f"'{read}'" for read in reads))
        if name in self._game_actions and self._objective_due() and not self._objective_set:
            return OBJECTIVE_DENIAL
        return None

    def _refuse(self, name: str, args, reason: str) -> ToolResult:
        """Report a call that wasn't run, as a failure (so a refused game action doesn't end the turn)."""
        self.report_not_run(name, args, reason)
        return ToolResult(text_result_for_llm=reason, result_type="failure", error=reason)

    def _objective_due(self) -> bool:
        """Whether the current step must call 'objective' before its game action (every objective_every_turns)."""
        every = self.config.objective_every_turns
        return bool(every) and "objective" in self.tool_factory.tools and (self.steps_completed + 1) % every == 0

    def _limit_helper_tools(self, hook_input: dict, _context: dict) -> dict | None:
        """Pre-tool-use hook for a helper's session: deny calls past its tool budget, so it answers."""
        self._helper_tool_calls += 1
        if self._helper_tool_calls <= self.config.helper_max_tool_calls:
            return None
        return self._deny(hook_input, HELPER_BUDGET_DENIAL)

    def _deny(self, hook_input: dict, reason: str) -> dict:
        name = hook_input.get("toolName")
        self._tool_calls += 1
        self.emit("tool", id=self._tool_calls, name=name, args=hook_input.get("toolArgs") or {})
        self.emit("tool_result", id=self._tool_calls, result=reason, ok=False)
        logger.info(f"[Tool] Denied {name}: {reason}")
        return {"permissionDecision": "deny", "permissionDecisionReason": reason}

    def _emit_reasoning(self, message: AssistantMessageData):
        """Show the response's reasoning, split into the titled parts the provider summarized it as."""
        parts = []
        blocks = message.reasoning_blocks.blocks if message.reasoning_blocks else None
        for block in blocks or []:
            if not isinstance(block, dict):
                continue
            for summary in block.get("summary") or []:
                if text := (summary.get("text") or "").strip():
                    match = REASONING_TITLE.match(text)
                    title, body = (match[1], match[2].strip()) if match else (None, text)
                    parts.append({"title": title, "text": body})
        text = "\n\n".join(part["text"] for part in parts) or (message.reasoning_text or "").strip()
        if text:
            self.emit("thought", text=text, parts=parts or None, reasoning=True, step=self.steps_completed + 1)

    def _log_event(self, event: SessionEvent, helper: bool = False):
        """Log and publish a session event; `helper` marks events from a subagent helper's session."""
        match event.data:
            case AssistantMessageData() as message:
                if helper:
                    self._helper_text = message.content or self._helper_text
                if message.tool_requests:
                    calls = [(request.tool_call_id, request.name) for request in message.tool_requests]
                    batch = self._helper_batch if helper else self._batch
                    if calls != batch.calls:
                        setattr(self, "_helper_batch" if helper else "_batch", _Batch(calls))
                if not message.chunk_index:
                    self._emit_reasoning(message)
                if message.content:
                    logger.info(f"[Text] {message.content}")
                    self.emit("thought", text=message.content, step=self.steps_completed + 1)
            case ToolExecutionStartData(tool_name=name, arguments=arguments):
                logger.info(f"[Tool] Using tool {name} with args {arguments}")
            case AssistantUsageData() as usage if not helper:  # the plate shows the main agent's context
                logger.info(f"Agent response usage: input={usage.input_tokens} output={usage.output_tokens}")
                self.emit("usage", input=usage.input_tokens, output=usage.output_tokens)
            case SessionCompactionCompleteData() as compaction:
                logger.info(f"[Agent] Context compacted: success={compaction.success}")
                if compaction.success and not helper:
                    self._condensed = int(getattr(compaction, "checkpoint_number", None) or 0)
                    self.notes_condensed()
                self.emit(
                    "summary",
                    text=compaction.summary_content or "",
                    before=compaction.pre_compaction_tokens,
                    after=compaction.post_compaction_tokens,
                )
            case SessionWarningData(message=message):
                logger.warning(f"[Copilot] {message}")
            case SessionErrorData(message=message):
                logger.error(f"[Copilot] {message}")

    # ----- durable conversation -----------------------------------------------------------------------------------

    def _session_dir(self) -> Path:
        return self.copilot_home / "session-state" / str(self.session_id)

    def write_agent_state(self, directory: Path):
        if self.session_id and self._session_dir().is_dir():
            copy_tree(self._session_dir(), directory / "copilot-session", ignore_patterns=SESSION_RUNTIME_FILES)

    def restore_agent_state(self, directory: Path):
        snapshot = directory / "copilot-session"
        if not snapshot.is_dir():
            logger.warning("Checkpoint has no Copilot session; starting a new conversation")
            return
        copy_tree(snapshot, self._session_dir(), ignore_patterns=SESSION_RUNTIME_FILES)
        self.resume_session = True
        logger.info(f"Restored Copilot session {self.session_id}")

    async def _run_async(self, num_steps: int):
        asyncio.get_running_loop().set_exception_handler(_quiet_unknown_event_errors)

        session_options = self._session_options(self.system_prompt, self.tool_factory)
        async with CopilotClient(mode="empty", base_directory=str(self.copilot_home)) as client:
            self._client = client
            if self.resume_session:
                open_session = client.resume_session(self.session_id, **session_options)
                prefix_content = RESUME_STEP_PREFIX
            else:
                open_session = client.create_session(session_id=self.session_id, **session_options)
                prefix_content = START_STEP_PREFIX
            async with await open_session as session:
                session.on(self._log_event)
                while self.running and self.steps_completed < num_steps:
                    self._objective_set = False
                    if self._objective_due():
                        prefix_content = f"{OBJECTIVE_DUE_PREFIX}\n{prefix_content}"
                    if self._condensed is not None:
                        prefix_content = f"{prefix_content}\n{self._condensed_note()}".strip()
                    await self._send_step(session, prefix_content)
                    if self._compaction_request is not None:
                        await self._compact(session)
                    self.steps_completed += 1
                    self._on_step_complete(self.steps_completed, num_steps)
                    prefix_content = STEP_PREFIX

    # ----- helper (the subagent tool) ---------------------------------------------------------------------------------

    async def _run_helper(self, task: str) -> str:
        """Have a read-only helper analyze `task` in a fresh session; returns its answer as the tool result."""
        if not task:
            return ToolError("Error: subagent needs a 'task'.")
        if self._client is None:
            return ToolError("Error: no helper is available right now.")
        if self.actor == "helper":
            return ToolError("Error: a helper is already working; wait for its answer before asking another.")
        step = self.steps_completed + 1
        # The helper sees the observation that started this step: observing again could advance the game.
        prompt, attachments = self._step_message(
            HELPER_TASK_PREFIX.format(task=task), self._observation, with_memory_listing=True
        )
        options = self._session_options(
            helper_system_prompt(self.config.model, self.config.helper_max_tool_calls),
            self.helper_factory,
            self._limit_helper_tools,
            helper=True,
        )
        session_id = f"{self.session_id}-helper-{step}-{int(time.time())}" if self.session_id else None
        self._helper_text, self._helper_tool_calls = "", 0
        started = time.monotonic()
        note = ""
        self.emit("subagent", phase="start", task=task, step=step)
        logger.info(f"[Helper] Analyzing at step {step}: {task}")
        self.actor = "helper"
        try:
            async with await self._client.create_session(session_id=session_id, **options) as session:
                self._helper_session = session
                session.on(lambda event: self._log_event(event, helper=True))
                try:
                    await session.send_and_wait(
                        prompt, attachments=attachments, timeout=self.config.helper_timeout_seconds
                    )
                except TimeoutError:
                    await self._abort(session)
                    limit = self.config.helper_timeout_seconds
                    note = f"\n\n(The helper ran out of time after {limit:.0f}s; this is its latest reply.)"
        finally:
            self.actor = None
            self._helper_session = None
        answer = (self._helper_text.strip() or "(The helper gave no answer.)") + note
        seconds = round(time.monotonic() - started)
        calls = self._helper_tool_calls
        self.emit("subagent", phase="end", report=answer, tool_calls=calls, seconds=seconds, step=step)
        logger.info(f"[Helper] Answered after {calls} tool calls and {seconds}s")
        return HELPER_ANSWER_TEMPLATE.format(answer=answer)

    async def _send_step(self, session, prefix_content: str):
        """Send one step, retrying with a fresh observation when the model request fails."""
        for attempt in range(self.config.step_retries + 1):
            prompt, attachments = self._step_message(f"{prefix_content}\n{await self._context_note(session)}".strip())
            self.emit("prompt", text=prompt, step=self.steps_completed + 1)  # the observer shows it, collapsed
            try:
                await session.send_and_wait(prompt, attachments=attachments, timeout=self.config.turn_timeout_seconds)
                return
            except Exception as e:
                await self._cancel_turn(session)
                if attempt == self.config.step_retries:
                    raise
                wait = self.config.step_retry_wait_seconds * (attempt + 1)
                logger.warning(f"Step {self.steps_completed + 1} failed ({e}); retrying in {wait:.0f}s")
                step = self.steps_completed + 1
                self.emit("notice", text=f"Model request failed; retrying step {step} in {wait:.0f}s")
                await asyncio.sleep(wait)

    async def _context_note(self, session) -> str:
        """How full the model's context is, and how close it is to being condensed automatically ("" if unknown)."""
        try:
            request = MetadataContextInfoRequest(output_token_limit=0, prompt_token_limit=0)
            info = (await session.rpc.metadata.context_info(request)).context_info
        except Exception as e:
            logger.warning(f"Couldn't read the context size: {e}")
            return ""
        if info is None or not info.limit:
            return ""
        used, threshold = info.total_tokens, info.compaction_threshold
        note = (
            f"Model context usage: {used:,} of {info.limit:,} tokens ({used / info.limit:.0%}). Your conversation is "
            f"condensed automatically after {threshold:,} tokens ({threshold / info.limit:.0%}), "
        )
        if used >= threshold:
            return note + "which it has passed, so it will be condensed soon."
        return note + f"{threshold - used:,} tokens from now."

    def _condensed_note(self, shown: int = 8) -> str:
        """Titles of the summaries before the one the conversation was just condensed into ("" if none yet)."""
        newest, self._condensed = self._condensed, None
        if self._history is None:
            return ""
        self._history.refresh()
        earlier = [s for s in self._history.summaries if not newest or s.number < newest][-shown:]
        if not earlier:
            return ""
        titles = "\n".join(f"- #{s.number} (turn {s.turn}): {s.title}" for s in reversed(earlier))
        return CONDENSED_NOTE.format(titles=titles)

    async def _compact(self, session):
        """Condense the conversation now, as the model asked with 'trigger_compaction'."""
        instructions, self._compaction_request = self._compaction_request or None, None
        logger.info(f"[Agent] Condensing the conversation, as the model asked (instructions: {instructions!r})")
        try:
            request = SessionHistoryCompactRequest(custom_instructions=instructions, trigger=Trigger.MANUAL)
            await session.rpc.history.compact(request, timeout=self.config.turn_timeout_seconds)
        except Exception as e:
            logger.warning(f"Requested compaction failed: {e}")
            self.emit("notice", text=f"Condensing the history failed ({type(e).__name__})")

    async def _cancel_turn(self, session):
        """Stop a turn that timed out or failed before observing the game again: send_and_wait only stops waiting, so
        abort the request (and any helper it started), refuse its calls that are still to run, and wait until no tool
        is running."""
        self._cancelled_calls.update(call_id for call_id, _ in self._batch.calls)
        for target in (self._helper_session, session):
            if target is not None:
                await self._abort(target)
        if lock := self._tool_locks.get(False):
            try:
                async with asyncio.timeout(self.config.helper_timeout_seconds + 60):
                    async with lock:
                        pass
            except TimeoutError:
                logger.warning("A tool from the cancelled turn is still running")

    @staticmethod
    async def _abort(session):
        try:
            await session.abort()
        except Exception as e:
            logger.warning(f"Couldn't abort the session's request: {e}")

    def run(self, num_steps=1):
        """Main agent loop.

        Args:
            num_steps: Number of steps to run for
        """
        logger.info(f"Running until step {num_steps} (starting at {self.steps_completed})")
        self.emit("run_start", steps=num_steps, resumed_from=self.steps_completed or None)
        try:
            asyncio.run(self._run_async(num_steps))
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, stopping")
            self.running = False
        except Exception as e:
            logger.error("Error in agent loop.", exc_info=True)
            self.save_state_after_failure()
            raise e

        self.save_state()
        self.emit("run_end", steps=self.steps_completed)
        return self.steps_completed
