"""Query and pretty-print case data from the warpt database."""

from __future__ import annotations

import ast
import json
from datetime import datetime
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from warpt.daemon.casefile import CaseFile

SEPARATOR = "=" * 60


def _fmt_ts(ts: object) -> str:
    """Format a timestamp to second precision."""
    if isinstance(ts, datetime):
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    return str(ts).split(".")[0]


def show_case(cf: CaseFile, case_id: int) -> None:
    """Pretty-print a single case with full context."""
    rows = cf.query(
        "SELECT case_id, status, opened_at, title, "
        "hypothesis, confidence_pct, recommended_action, "
        "observation, reasoning_chain, diagnostician_model "
        "FROM cases WHERE case_id = ?",
        [case_id],
    )
    if not rows:
        click.echo(f"No case found with ID {case_id}.")
        return

    row = rows[0]
    (
        cid, status, opened_at, title,
        hypothesis, confidence_pct, recommended_action,
        observation, reasoning_chain, diagnostician_model,
    ) = row

    # Extract GPU GUID from title (format: "GPU-xxxx: metric_name ...")
    gpu_guid = (
        title.split(":")[0] if title and ":" in title else title or ""
    )

    click.echo(f"\n{SEPARATOR}")
    click.echo(f"  WARPT CASE #{cid}")
    click.echo(SEPARATOR)
    click.echo()
    click.echo(f"{'STATUS:':<13}{status}")
    click.echo(f"{'OPENED:':<13}{_fmt_ts(opened_at)}")
    click.echo(f"{'GPU:':<13}{gpu_guid}")

    if hypothesis:
        click.echo("\n--- HYPOTHESIS ---")
        click.echo(hypothesis)

    if confidence_pct is not None:
        click.echo("\n--- CONFIDENCE ---")
        click.echo(f"{confidence_pct:.2f}%")

    if recommended_action:
        click.echo("\n--- RECOMMENDED ACTION ---")
        click.echo(recommended_action)

    # Parse observation JSON for baseline and LLM interpretation
    if observation:
        _print_observation(observation, diagnostician_model)

    # Parse reasoning_chain for triage reasoning
    if reasoning_chain:
        _print_reasoning(reasoning_chain)

    # Events
    events = cf.query(
        "SELECT severity, ts, summary FROM events "
        "WHERE case_id = ? ORDER BY ts",
        [case_id],
    )
    if events:
        click.echo("\n--- EVENTS ---")
        for severity, ts, summary in events:
            click.echo(
                f"  [{severity.upper()}] {_fmt_ts(ts)}"
                f" — {summary}"
            )

    if not hypothesis:
        click.echo("\n--- DIAGNOSIS PENDING ---")
        click.echo(
            "No diagnosis has been generated for this case yet."
        )

    click.echo(f"\n{SEPARATOR}")


def _print_observation(observation: str, model: str | None) -> None:
    """Print baseline and LLM interpretation from observation JSON."""
    try:
        obs = json.loads(observation)
    except (json.JSONDecodeError, TypeError):
        return

    baseline = obs.get("baseline", {})
    deviation = obs.get("deviation_pct")
    current = obs.get("current_value")
    if baseline or deviation is not None or current is not None:
        click.echo("\n--- BASELINE ---")
        for label, key in [
            ("1h avg", "1h_avg"),
            ("24h avg", "24h_avg"),
            ("7d avg", "7d_avg"),
        ]:
            val = baseline.get(key)
            if val is not None:
                click.echo(f"  {label + ':':<14}{val:.1f}%")
        if deviation is not None:
            click.echo(f"  {'Deviation:':<14}{deviation:.1f}%")
        if current is not None:
            click.echo(f"  {'Current:':<14}{current:.1f}%")

    interpretation = obs.get("interpretation")
    if interpretation:
        model_name = model or "unknown"
        click.echo(
            f"\n--- LLM INTERPRETATION ({model_name}) ---"
        )
        click.echo(interpretation)


def _print_reasoning(reasoning_chain: str) -> None:
    """Print triage reasoning from the reasoning_chain column."""
    try:
        chain = ast.literal_eval(reasoning_chain)
    except (ValueError, SyntaxError):
        return

    if not isinstance(chain, list) or not chain:
        return

    click.echo("\n--- TRIAGE REASONING ---")
    for entry in chain:
        if not isinstance(entry, dict):
            continue
        category = entry.get("category", "Unknown")
        click.echo(f"  [{category}]")
        finding = entry.get("finding")
        if finding:
            click.echo(f"  Finding: {finding}")
        implication = entry.get("implication")
        if implication:
            click.echo(f"  Implication: {implication}")
        click.echo()


def show_latest(cf: CaseFile) -> None:
    """Show the most recent case."""
    rows = cf.query(
        "SELECT case_id FROM cases "
        "ORDER BY opened_at DESC LIMIT 1"
    )
    if not rows:
        click.echo("No cases found.")
        return
    show_case(cf, rows[0][0])


def list_cases(cf: CaseFile) -> None:
    """Print a summary table of all cases."""
    rows = cf.query(
        "SELECT case_id, status, opened_at, title "
        "FROM cases ORDER BY case_id"
    )
    if not rows:
        click.echo("No cases found.")
        return

    click.echo(
        f" {'ID':>3} | {'Status':<8} | {'Opened':<19} | Title"
    )
    click.echo(
        f"-----+----------+---------------------+{'─' * 34}"
    )
    for cid, status, opened_at, title in rows:
        title_trunc = (
            (title[:31] + "...")
            if title and len(title) > 34
            else (title or "")
        )
        click.echo(
            f" {cid:>3} | {status:<8} | {_fmt_ts(opened_at):<19}"
            f" | {title_trunc}"
        )
