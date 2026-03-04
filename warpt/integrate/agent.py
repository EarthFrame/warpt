"""Core orchestration for the integration agent.

Uses the Claude Code SDK to run an agent session that generates
backend implementations from vendor SDK documentation.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import click

from warpt.integrate.models import (
    QuestionsDocument,
    QuestionStatus,
)
from warpt.integrate.session import (
    increment_pass,
    load_session,
    save_session,
    session_exists,
)
from warpt.integrate.system_prompt import build_system_prompt

# Root of the warpt repository (two levels up from this file)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class _ProgressTracker:
    """Live spinner + status line for agent sessions."""

    def __init__(self) -> None:
        self._turn = 0
        self._status = "Starting agent session..."
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the spinner thread."""
        self._thread = threading.Thread(
            target=self._spin, daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the spinner and clear the line."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        # Clear spinner line
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()

    def update(self, status: str) -> None:
        """Update the status message."""
        with self._lock:
            self._status = status

    def bump_turn(self) -> None:
        """Increment the turn counter."""
        with self._lock:
            self._turn += 1

    def _spin(self) -> None:
        """Spinner loop (runs in background thread)."""
        idx = 0
        while not self._stop.is_set():
            frame = _SPINNER_FRAMES[idx % len(_SPINNER_FRAMES)]
            with self._lock:
                turn = self._turn
                status = self._status
            # Truncate status to fit terminal
            label = f"{frame} [Turn {turn}] {status}"
            if len(label) > 78:
                label = label[:75] + "..."
            sys.stderr.write(f"\r\033[K{label}")
            sys.stderr.flush()
            idx += 1
            time.sleep(0.1)

    @staticmethod
    def describe_tool(name: str, tool_input: dict) -> str:
        """Map a tool call to a short human-readable status."""
        if name in ("Read", "read_file"):
            path = tool_input.get(
                "file_path", tool_input.get("path", "")
            )
            short = Path(path).name if path else "file"
            return f"Reading {short}"
        if name in ("Write", "write_file"):
            path = tool_input.get(
                "file_path", tool_input.get("path", "")
            )
            short = Path(path).name if path else "file"
            return f"Writing {short}"
        if name in ("Edit", "edit_file"):
            path = tool_input.get(
                "file_path", tool_input.get("path", "")
            )
            short = Path(path).name if path else "file"
            return f"Editing {short}"
        if name in ("Bash", "bash"):
            cmd = str(
                tool_input.get("command", "")
            )[:40]
            return f"Running: {cmd}"
        if name in ("Glob", "glob"):
            return "Searching files"
        if name in ("Grep", "grep"):
            pattern = tool_input.get("pattern", "")
            return f"Searching for '{pattern[:30]}'"
        return f"Using {name}"


def _questions_path() -> Path:
    """Path to the questions.yaml file."""
    return _REPO_ROOT / "questions.yaml"


def _load_questions() -> QuestionsDocument:
    """Load the questions document."""
    path = _questions_path()
    if path.exists():
        try:
            return QuestionsDocument.from_yaml(
                path.read_text()
            )
        except Exception:
            click.echo(
                "Warning: questions.yaml has invalid format. "
                "The agent may not have followed the schema. "
                "Review the file manually."
            )
            return QuestionsDocument()
    return QuestionsDocument()


def _init_questions_doc(
    vendor: str,
    sdk_source: str,
    session_id: str,
) -> None:
    """Create the initial questions.yaml file."""
    doc = QuestionsDocument(
        metadata={
            "vendor": vendor,
            "sdk_source": sdk_source,
            "created": datetime.now(UTC).isoformat(),
            "session_id": session_id,
            "pass_number": 1,
        },
        questions=[],
    )
    _questions_path().write_text(doc.to_yaml())


def _git_run(*args: str) -> subprocess.CompletedProcess[str]:
    """Run a git command in the repo root."""
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )


def _create_git_branch(vendor: str) -> str:
    """Create and checkout a new git branch for the integration.

    Parameters
    ----------
    vendor : str
        Vendor name used to form the branch name.

    Returns
    -------
    str
        The parent branch name (the branch we branched from).

    Raises
    ------
    click.ClickException
        If the working tree is dirty or git operations fail.
    """
    branch_name = f"backend/{vendor}"

    # Check for uncommitted changes
    status = _git_run("status", "--porcelain")
    if status.stdout.strip():
        raise click.ClickException(
            "You have uncommitted changes.\n"
            "Please commit or stash them before starting "
            "an integration:\n"
            "  git stash        # stash changes\n"
            "  git commit -am 'wip'  # or commit them"
        )

    # Get current branch name
    current = _git_run("branch", "--show-current")
    current_branch = current.stdout.strip() or "HEAD (detached)"

    # Check if branch already exists
    existing = _git_run("branch", "--list", branch_name)
    if branch_name in existing.stdout:
        click.echo(
            f"Branch {branch_name} already exists, "
            "checking out..."
        )
        result = _git_run("checkout", branch_name)
        if result.returncode != 0:
            raise click.ClickException(
                f"Failed to checkout {branch_name}: "
                f"{result.stderr.strip()}"
            )
    else:
        if not click.confirm(
            f"Create branch '{branch_name}' "
            f"from '{current_branch}'?",
        ):
            raise click.ClickException(
                "Aborted. Switch to the branch you want "
                "as the parent and try again."
            )
        result = _git_run("checkout", "-b", branch_name)
        if result.returncode != 0:
            raise click.ClickException(
                f"Failed to create branch {branch_name}: "
                f"{result.stderr.strip()}"
            )

    return current_branch



def _build_init_prompt(
    vendor: str,
    sdk_docs_text: str,
) -> str:
    """Build the initial user prompt for the agent."""
    cap_vendor = vendor.capitalize()
    return (
        f"Integrate the {vendor} hardware accelerator "
        f"backend into warpt.\n\n"
        "## CRITICAL RULES\n\n"
        "- Your system prompt ALREADY contains: the "
        "AcceleratorBackend ABC, NVIDIA reference "
        "implementation, factory pattern, PowerBackend ABC, "
        "NVIDIA power reference, test patterns, GPUInfo "
        "model, power models, and coding standards. "
        "Do NOT read any of those files.\n"
        "- Do NOT use TodoWrite or any task-planning tools. "
        "Just execute.\n"
        "- Do NOT read files one chunk at a time. Read the "
        "whole file in one call.\n"
        "- Write ALL code files BEFORE running any tests or "
        "linting.\n\n"
        "## EXECUTION PLAN\n\n"
        "Follow these steps in EXACTLY this order:\n\n"
        "**Step 1** — Read `.sdk_docs.txt` (one read, full "
        "file). Identify which SDK functions map to each "
        "AcceleratorBackend method.\n\n"
        "**Step 2** — Write ALL FOUR files in a single "
        "batch (do not run tests between writes):\n\n"
        f"  a) `warpt/backends/{vendor}.py` — "
        f"class `{cap_vendor}Backend(AcceleratorBackend)` "
        "implementing all 12 methods. Follow the NVIDIA "
        "reference in your system prompt exactly.\n\n"
        f"  b) `warpt/backends/power/{vendor}_power.py` — "
        f"class `{cap_vendor}PowerBackend(PowerBackend)` "
        "with `is_available`, `get_source`, `initialize`, "
        "`get_power_readings`, `get_gpu_power_info`, "
        "`get_total_gpu_power`, `cleanup`.\n\n"
        "  c) `warpt/backends/factory.py` — Edit to add a "
        f"try/except block for `{cap_vendor}Backend` "
        "between NVIDIA and Intel.\n\n"
        f"  d) `tests/test_{vendor}_backend.py` — "
        "Mock-based tests covering both backend classes, "
        "factory fallthrough, and edge cases. Use the test "
        "patterns from your system prompt.\n\n"
        "**Step 3** — Run tests and lint:\n"
        f"  - `pytest tests/test_{vendor}_backend.py -v`\n"
        f"  - `ruff check warpt/backends/{vendor}.py "
        f"warpt/backends/power/{vendor}_power.py "
        f"tests/test_{vendor}_backend.py`\n\n"
        "**Step 4** — If anything fails, fix and re-run. "
        "Use `ruff check --fix` for auto-fixable lint.\n\n"
        "**Step 5** — Write questions to `questions.yaml` "
        "following the EXACT schema in your system prompt. "
        "Log unit ambiguities, API assumptions, and design "
        "decisions.\n\n"
        "You are DONE after step 5. Do not do anything "
        "else.\n\n"
        f"# {vendor} SDK Documentation\n\n"
        f"{sdk_docs_text}"
    )


def _build_iterate_prompt(vendor: str) -> str:
    """Build the prompt for an iteration pass."""
    return (
        f"Continue the {vendor} backend integration.\n\n"
        "Read questions.yaml and find all questions with "
        'status: "answered".\n'
        "For each answered question:\n"
        "1. Read the vendor's answer\n"
        "2. Update the code accordingly\n"
        "3. Update the question status to 'implemented'\n"
        f"4. Run: pytest tests/test_{vendor}_backend.py -v\n"
        "5. If tests pass, update status to 'verified'\n"
        "6. If the answer raises new questions, log them\n\n"
        "When done, provide a summary of changes made."
    )


async def _query_with_skip(prompt, options):
    """Wrap query() to skip unknown message types.

    The Claude CLI may emit message types (e.g.,
    rate_limit_event) that older SDK versions don't
    recognize. This wrapper catches those parse errors
    and continues instead of crashing.
    """
    from claude_code_sdk import query
    from claude_code_sdk._errors import MessageParseError

    try:
        from claude_code_sdk._internal.message_parser import (
            parse_message,
        )
        from claude_code_sdk._internal.transport.subprocess_cli import (
            SubprocessCLITransport,
        )

        transport = SubprocessCLITransport(
            prompt=prompt, options=options
        )
        await transport.connect()

        from claude_code_sdk._internal.query import Query

        q = Query(
            transport=transport,
            is_streaming_mode=False,
        )
        await q.start()

        async for data in q.receive_messages():
            try:
                yield parse_message(data)
            except MessageParseError:
                # Skip unrecognized message types
                continue

        await q.close()
        return
    except ImportError:
        pass

    # Fallback: use query() directly if internals changed
    async for message in query(
        prompt=prompt, options=options
    ):
        yield message


async def _run_claude_session_async(
    system_prompt: str,
    user_prompt: str,
    session_id: str | None = None,
) -> tuple[str, str]:
    """Run a Claude Code SDK session (async).

    Parameters
    ----------
    system_prompt : str
        The system prompt for the agent.
    user_prompt : str
        The user message / task prompt.
    session_id : str | None
        Session ID to resume, or None for a new session.

    Returns
    -------
    tuple[str, str]
        (session_id, agent_output_text)
    """
    try:
        from claude_code_sdk import (
            AssistantMessage,
            ClaudeCodeOptions,
            ResultMessage,
            TextBlock,
            ToolUseBlock,
        )
    except ImportError as exc:
        raise click.ClickException(
            "claude-code-sdk is required for the integrate "
            "command.\nInstall with: pip install 'warpt"
            "[integrate]'"
        ) from exc

    options = ClaudeCodeOptions(
        system_prompt=system_prompt,
        permission_mode="bypassPermissions",
        disallowed_tools=[
            "TodoWrite",
            "TodoRead",
            "WebSearch",
            "WebFetch",
        ],
        cwd=str(_REPO_ROOT),
    )

    if session_id:
        options.resume = session_id

    output_parts: list[str] = []
    result_session_id = session_id or ""

    # Write SDK docs to a file inside the repo so the
    # agent can read it with its file tools. Temp files
    # in /tmp or /var may be inaccessible to the sandbox.
    sdk_docs_file = _REPO_ROOT / ".sdk_docs.txt"
    sdk_docs_file.write_text(user_prompt, encoding="utf-8")

    progress = _ProgressTracker()
    progress.start()

    try:
        short_prompt = (
            "Your full task instructions and vendor SDK "
            "documentation are in the file .sdk_docs.txt "
            "in the repo root. Read that file first before "
            "doing anything else."
        )

        async for message in _query_with_skip(
            short_prompt, options
        ):
            if isinstance(message, ResultMessage):
                result_session_id = message.session_id
                if message.result:
                    output_parts.append(message.result)
                # Show session stats
                turns = getattr(
                    message, "num_turns", None
                )
                duration = getattr(
                    message, "duration_ms", None
                )
                cost = getattr(
                    message, "total_cost_usd", None
                )
                parts = []
                if turns:
                    parts.append(f"{turns} turns")
                if duration:
                    mins = duration / 60_000
                    parts.append(f"{mins:.1f}m")
                if cost:
                    parts.append(f"${cost:.2f}")
                if parts:
                    progress.stop()
                    click.echo(
                        f"Done ({', '.join(parts)})"
                    )
            elif isinstance(message, AssistantMessage):
                progress.bump_turn()
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        status = progress.describe_tool(
                            block.name, block.input or {}
                        )
                        progress.update(status)
                    elif isinstance(block, TextBlock):
                        output_parts.append(block.text)
                        # Show first sentence as status
                        first_line = block.text.strip().split(
                            "\n"
                        )[0][:60]
                        if first_line:
                            progress.update(first_line)
    except Exception as exc:
        progress.stop()
        click.echo(f"\nAgent session failed: {exc}")
        raise
    finally:
        progress.stop()
        # Clean up SDK docs file from repo
        sdk_docs_file.unlink(missing_ok=True)

    # Print session summary from ResultMessage
    if result_session_id:
        click.echo(f"Session: {result_session_id[:16]}...")

    return result_session_id, "\n".join(output_parts)


def _run_claude_session(
    system_prompt: str,
    user_prompt: str,
    session_id: str | None = None,
) -> tuple[str, str]:
    """Run a Claude Code SDK session (sync wrapper).

    Parameters
    ----------
    system_prompt : str
        The system prompt for the agent.
    user_prompt : str
        The user message / task prompt.
    session_id : str | None
        Session ID to resume, or None for a new session.

    Returns
    -------
    tuple[str, str]
        (session_id, agent_output_text)
    """
    return asyncio.run(
        _run_claude_session_async(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            session_id=session_id,
        )
    )


def _print_summary(doc: QuestionsDocument) -> None:
    """Print a summary of the questions document."""
    if not doc.questions:
        click.echo("No questions logged.")
        return

    click.echo("\nQuestions Summary:")
    click.echo("-" * 40)

    summary = doc.summary()
    for key, count in sorted(summary.items()):
        tier, status = key.split(":")
        click.echo(f"  {tier:25s} {status:15s} {count}")

    # Count blocking/open items
    blocking_open = len(
        [
            q
            for q in doc.questions
            if q.tier.value == "blocking"
            and q.status == QuestionStatus.OPEN
        ]
    )
    if blocking_open > 0:
        click.echo(
            f"\n  {blocking_open} BLOCKING question(s) "
            "need answers before proceeding."
        )
    else:
        click.echo("\n  No blocking questions.")


def run_init(
    vendor: str,
    sdk_docs_text: str,
    vendor_context: str | None = None,
) -> None:
    """Run the initial integration pass.

    Parameters
    ----------
    vendor : str
        Vendor name.
    sdk_docs_text : str
        Loaded SDK documentation text.
    vendor_context : str | None
        Optional hardware context.
    """
    if session_exists(vendor):
        click.echo(
            f"Session already exists for {vendor}. "
            "Run 'warpt integrate' to continue, "
            "or delete ~/.warpt/integrate/"
            f"{vendor}/ to start fresh."
        )
        return

    # Reject if SDK docs exceed the token limit
    from warpt.models.constants import MAX_SDK_TOKENS

    token_estimate = len(sdk_docs_text) // 4
    if token_estimate > MAX_SDK_TOKENS:
        raise click.ClickException(
            f"SDK docs are ~{token_estimate:,} tokens, "
            f"exceeding the {MAX_SDK_TOKENS:,} token limit. "
            "Point --sdk-docs at a smaller subdirectory."
        )

    click.echo(f"Starting {vendor} backend integration...")

    # Build system prompt (instructions only, no SDK docs)
    system_prompt = build_system_prompt(
        vendor=vendor,
        sdk_docs_text="",
        vendor_context=vendor_context,
    )

    # Create git branch
    parent_branch = _create_git_branch(vendor)

    # Generate a placeholder session ID for the questions doc
    placeholder_sid = f"pending-{vendor}"

    # Initialize questions document
    _init_questions_doc(
        vendor=vendor,
        sdk_source="cli-provided",
        session_id=placeholder_sid,
    )

    # Build the user prompt (includes SDK docs)
    user_prompt = _build_init_prompt(vendor, sdk_docs_text)

    click.echo("Running agent session (this will take a few minutes)...")

    # Run the agent
    session_id, output = _run_claude_session(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    # Save session
    save_session(
        vendor=vendor,
        session_id=session_id,
        metadata={
            "vendor": vendor,
            "pass_count": 1,
            "parent_branch": parent_branch,
        },
    )

    # Update questions doc with real session ID
    doc = _load_questions()
    doc.metadata["session_id"] = session_id
    _questions_path().write_text(doc.to_yaml())

    # Print results
    click.echo("\n" + "=" * 50)
    click.echo(f"Integration pass 1 complete for {vendor}")
    click.echo("=" * 50)

    if output:
        click.echo(f"\nAgent output:\n{output[:2000]}")

    # Print file status
    generated_files = [
        f"warpt/backends/{vendor}.py",
        f"warpt/backends/power/{vendor}_power.py",
        f"tests/test_{vendor}_backend.py",
    ]
    click.echo("\nGenerated files:")
    for f in generated_files:
        fpath = _REPO_ROOT / f
        status = "created" if fpath.exists() else "not created"
        click.echo(f"  {f}: {status}")

    # Print question summary
    doc = _load_questions()
    _print_summary(doc)

    click.echo(
        "\nNext steps:\n"
        "  1. Review generated code\n"
        "  2. Answer questions in questions.yaml "
        "(set status: answered)\n"
        "  3. Run: warpt integrate\n"
    )


def run_iterate(vendor: str) -> None:
    """Run an iteration pass to process answered questions.

    Parameters
    ----------
    vendor : str
        Vendor name.
    """
    if not session_exists(vendor):
        raise click.ClickException(
            f"No session found for {vendor}. "
            "Run 'warpt integrate --vendor "
            f"{vendor} --sdk-docs <path>' first."
        )

    # Load session
    session_id, metadata = load_session(vendor)
    pass_number = increment_pass(vendor)

    click.echo(
        f"Iterating {vendor} integration (pass {pass_number})..."
    )

    # Check for answered questions
    doc = _load_questions()
    answered = doc.get_by_status(QuestionStatus.ANSWERED)
    if not answered:
        click.echo("No answered questions found. Nothing to do.")
        click.echo(
            "Edit questions.yaml and set status to "
            "'answered' for questions you've addressed."
        )
        return

    click.echo(
        f"Found {len(answered)} answered question(s) to process."
    )

    # Build system prompt (re-read source files for current state)
    # For iterate, we don't need the full SDK docs again —
    # the agent session has context from init
    system_prompt = build_system_prompt(
        vendor=vendor,
        sdk_docs_text="(SDK docs provided in previous session)",
        vendor_context=None,
    )

    user_prompt = _build_iterate_prompt(vendor)

    click.echo("Running agent session...")

    # Run agent with resumed session
    new_session_id, output = _run_claude_session(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        session_id=session_id,
    )

    # Update session
    save_session(
        vendor=vendor,
        session_id=new_session_id,
        metadata={
            **metadata,
            "pass_count": pass_number,
        },
    )

    # Print results
    click.echo(f"\nIteration pass {pass_number} complete.")
    if output:
        click.echo(f"\nAgent output:\n{output[:2000]}")

    doc = _load_questions()
    _print_summary(doc)


def run_status(vendor: str | None = None) -> None:
    """Show the status of integration questions.

    Parameters
    ----------
    vendor : str | None
        Filter by vendor, or show all.
    """
    doc = _load_questions()

    if not doc.questions:
        click.echo("No questions document found.")
        click.echo(
            "Start one with: warpt integrate "
            "--vendor <name> --sdk-docs <path>"
        )
        return

    if vendor and doc.metadata.get("vendor") != vendor:
        click.echo(f"No questions found for vendor: {vendor}")
        return

    click.echo(
        f"Integration: {doc.metadata.get('vendor', 'unknown')}"
    )
    click.echo(
        f"Pass: {doc.metadata.get('pass_number', '?')}"
    )
    click.echo(
        f"Session: {doc.metadata.get('session_id', 'none')[:16]}..."
    )
    click.echo()

    for q in doc.questions:
        marker = {
            "open": " ",
            "answered": "A",
            "implemented": "I",
            "verified": "V",
        }.get(q.status.value, "?")
        tier_color = {
            "blocking": "red",
            "defaulted": "yellow",
            "clarification_needed": "cyan",
            "informational": "white",
        }.get(q.tier.value, "white")

        click.echo(
            f"  [{marker}] #{q.id:3d} "
            f"({click.style(q.tier.value, fg=tier_color)}) "
            f"{q.title}"
        )

    _print_summary(doc)


def run_validate(vendor: str) -> None:
    """Run tests and linting on the generated backend.

    Parameters
    ----------
    vendor : str
        Vendor name.
    """
    backend_file = _REPO_ROOT / "warpt" / "backends" / f"{vendor}.py"
    test_file = _REPO_ROOT / "tests" / f"test_{vendor}_backend.py"

    if not backend_file.exists():
        raise click.ClickException(
            f"Backend file not found: {backend_file}"
        )

    results: dict[str, bool] = {}

    # Run ruff
    click.echo(f"Running ruff on {backend_file.name}...")
    ruff_result = subprocess.run(
        ["ruff", "check", str(backend_file)],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    results["ruff"] = ruff_result.returncode == 0
    if ruff_result.returncode != 0:
        click.echo(f"  Ruff issues:\n{ruff_result.stdout}")
    else:
        click.echo("  Ruff: clean")

    # Run power backend ruff if exists
    power_file = (
        _REPO_ROOT
        / "warpt"
        / "backends"
        / "power"
        / f"{vendor}_power.py"
    )
    if power_file.exists():
        click.echo(f"Running ruff on {power_file.name}...")
        ruff_power = subprocess.run(
            ["ruff", "check", str(power_file)],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        results["ruff_power"] = ruff_power.returncode == 0
        if ruff_power.returncode != 0:
            click.echo(f"  Ruff issues:\n{ruff_power.stdout}")
        else:
            click.echo("  Ruff: clean")

    # Run tests
    if test_file.exists():
        click.echo(f"Running pytest on {test_file.name}...")
        pytest_result = subprocess.run(
            ["pytest", str(test_file), "-v"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        results["pytest"] = pytest_result.returncode == 0
        click.echo(pytest_result.stdout)
        if pytest_result.returncode != 0:
            click.echo(pytest_result.stderr)
    else:
        click.echo(f"  Test file not found: {test_file}")
        results["pytest"] = False

    # Summary
    click.echo("\nValidation Summary:")
    all_pass = True
    for check, passed in results.items():
        icon = "PASS" if passed else "FAIL"
        click.echo(f"  {check}: {icon}")
        if not passed:
            all_pass = False

    if all_pass:
        click.echo("\nAll checks passed!")
    else:
        click.echo(
            "\nSome checks failed. Fix issues and re-run."
        )
