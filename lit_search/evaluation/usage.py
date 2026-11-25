"""Usage statistics over saved hunt sessions.

Every hunt leaves a ``session.json`` with its moves (per-source steps,
failures, seconds) and its candidates (the routes that reached each one and
the agent's mark).  This module turns a directory of them into the numbers
that decide where calls and agent attention are best spent: how often each
move and source fails and why, and how many kept papers each move reaches.

Attribution rule: a candidate is credited to the move that reached it first.
That understates moves that mostly confirm earlier finds, so ``reached`` (any
route) is reported beside ``first``.  Marks are the driving agent's, not the
owner's.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KEPT_MARKS = frozenset({"keep", "core"})
_FIND_SOURCE = re.compile(r"(\w+)#\d+")


@dataclass
class MoveStats:
    """Counters for one move kind."""

    moves: int = 0
    seconds: float = 0.0
    steps: int = 0
    failed: Counter[str] = field(default_factory=Counter)
    reached: int = 0
    reached_kept: int = 0
    first: int = 0
    first_kept: int = 0
    exclusive: int = 0
    exclusive_kept: int = 0


@dataclass
class UsageReport:
    """Aggregated usage over a set of hunt sessions."""

    hunts: int = 0
    candidates: int = 0
    kept: int = 0
    marks: Counter[str] = field(default_factory=Counter)
    by_move: dict[str, MoveStats] = field(default_factory=lambda: defaultdict(MoveStats))
    by_source: dict[str, MoveStats] = field(default_factory=lambda: defaultdict(MoveStats))
    related_with_negatives: int = 0
    repeat_share: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready view."""

        def table(rows: dict[str, MoveStats]) -> dict[str, Any]:
            return {
                name: {
                    "moves": s.moves,
                    "seconds": round(s.seconds, 1),
                    "steps": s.steps,
                    "failed": dict(s.failed),
                    "reached": s.reached,
                    "reached_kept": s.reached_kept,
                    "first": s.first,
                    "first_kept": s.first_kept,
                    "exclusive": s.exclusive,
                    "exclusive_kept": s.exclusive_kept,
                }
                for name, s in rows.items()
            }

        return {
            "hunts": self.hunts,
            "candidates": self.candidates,
            "kept": self.kept,
            "marks": dict(self.marks),
            "by_move": table(self.by_move),
            "by_source": table(self.by_source),
            "related_with_negatives": self.related_with_negatives,
            "repeat_share": round(self.repeat_share, 3),
        }


def failure_class(step: dict[str, Any]) -> str:
    """Name the kind of failure a failed step records.

    Parameters
    ----------
    step : dict[str, Any]
        One entry of a move's ``steps``.

    Returns
    -------
    str
        ``"busy"`` (refused locally, nothing sent), ``"429"``, ``"deadline"``,
        ``"404"``, ``"5xx"`` or ``"other"``.
    """
    text = str(step.get("error") or step.get("detail") or "")
    if step.get("error_type") == "RateLimitBusy" or text.startswith("rate-limited"):
        return "busy"
    if "429" in text:
        return "429"
    if "deadline" in text or step.get("error_type") == "Timeout":
        return "deadline"
    if "404" in text:
        return "404"
    if re.search(r"\b5\d\d\b", text):
        return "5xx"
    return "other"


def _step_source(step: dict[str, Any]) -> str:
    source = str(step.get("source") or "?")
    # Snowball and add steps embed a paper title after the step name.
    parts = source.split(":")
    return ":".join(parts[:2]) if len(parts) > 2 else source


def load_sessions(root: Path, since: dt.date | None = None) -> list[dict[str, Any]]:
    """Read every readable ``session.json`` under *root*.

    Parameters
    ----------
    root : Path
        The hunts directory.
    since : dt.date | None
        Keep only hunts started on or after this date.

    Returns
    -------
    list[dict[str, Any]]
        Parsed sessions, oldest first.  Unreadable files are skipped.
    """
    sessions = []
    for path in sorted(root.glob("*/session.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        started = data.get("started")
        if (
            since is not None
            and isinstance(started, (int, float))
            and dt.datetime.fromtimestamp(started).date() < since
        ):
            continue
        sessions.append(data)
    return sessions


def _title_key(candidate: dict[str, Any]) -> str:
    title = (candidate.get("paper") or {}).get("title") or ""
    return re.sub(r"\W+", "", title.lower())[:80]


def summarize(sessions: Iterable[dict[str, Any]]) -> UsageReport:
    """Aggregate move, source and yield statistics.

    Parameters
    ----------
    sessions : Iterable[dict[str, Any]]
        Parsed hunt sessions.

    Returns
    -------
    UsageReport
        Counters over all sessions.
    """
    report = UsageReport()
    appearances = 0
    distinct: set[str] = set()
    for session in sessions:
        report.hunts += 1
        moves = {m.get("number"): m for m in session.get("moves") or []}
        for move in moves.values():
            kind = str(move.get("kind") or "?")
            stats = report.by_move[kind]
            stats.moves += 1
            stats.seconds += float(move.get("seconds") or 0.0)
            if kind == "related" and (move.get("args") or {}).get("not"):
                report.related_with_negatives += 1
            for step in move.get("steps") or []:
                if not isinstance(step, dict):
                    continue
                source = _step_source(step)
                stats.steps += 1
                per_source = report.by_source[source]
                per_source.steps += 1
                if step.get("status") == "failed" or step.get("error"):
                    stats.failed[failure_class(step)] += 1
                    per_source.failed[failure_class(step)] += 1

        seen_here: set[str] = set()
        for candidate in (session.get("candidates") or {}).values():
            report.candidates += 1
            mark = candidate.get("mark")
            report.marks[str(mark)] += 1
            kept = mark in KEPT_MARKS
            report.kept += kept
            key = _title_key(candidate)
            if key:
                seen_here.add(key)
            routes = candidate.get("routes") or []
            kinds: set[str] = set()
            sources: set[str] = set()
            for route in routes:
                move = moves.get(route.get("move")) or {}
                kinds.add(str(move.get("kind") or "?"))
                if move.get("kind") == "find":
                    sources.update(_find_sources(str(route.get("label") or "")))
            for kind in kinds:
                report.by_move[kind].reached += 1
                report.by_move[kind].reached_kept += kept
            for source in sources:
                report.by_source[source].reached += 1
                report.by_source[source].reached_kept += kept
            if len(sources) == 1 and kinds == {"find"}:
                only = next(iter(sources))
                report.by_source[only].exclusive += 1
                report.by_source[only].exclusive_kept += kept
            if routes:
                first = moves.get(routes[0].get("move")) or {}
                kind = str(first.get("kind") or "?")
                report.by_move[kind].first += 1
                report.by_move[kind].first_kept += kept
        appearances += len(seen_here)
        distinct |= seen_here
    if appearances:
        report.repeat_share = (appearances - len(distinct)) / appearances
    return report


def _find_sources(label: str) -> set[str]:
    """Every source named in a find route label such as ``find#1 (arxiv#1,openalex#4)``."""
    inside = label.split("(", 1)[1] if "(" in label else ""
    return set(_FIND_SOURCE.findall(inside))


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "-"


def render(report: UsageReport) -> str:
    """Render a usage report as plain-text tables.

    Parameters
    ----------
    report : UsageReport
        Aggregated statistics.

    Returns
    -------
    str
        Human- and agent-readable text.
    """
    lines = [
        f"{report.hunts} hunts · {report.candidates} candidates · {report.kept} kept or core · "
        f"{100 * report.repeat_share:.0f}% of candidate appearances repeat "
        "across hunts",
        "",
        "by move   (first = kept papers this move reached first; failed = failed source steps)",
        f"{'move':<12}{'moves':>6}{'kept/move':>10}{'first-kept':>11}{'reached':>9}"
        f"{'keep%':>7}{'s/move':>8}{'steps':>7}{'failed':>8}  failure kinds",
    ]
    for kind, s in sorted(report.by_move.items(), key=lambda kv: -kv[1].moves):
        kinds = ", ".join(f"{k} {v}" for k, v in s.failed.most_common())
        lines.append(
            f"{kind:<12}{s.moves:>6}{(s.first_kept / s.moves if s.moves else 0):>10.1f}"
            f"{s.first_kept:>11}{s.reached:>9}{_pct(s.reached_kept, s.reached):>7}"
            f"{(s.seconds / s.moves if s.moves else 0):>8.1f}{s.steps:>7}"
            f"{_pct(sum(s.failed.values()), s.steps):>8}  {kinds}"
        )
    lines += [
        "",
        "by source (find sources: reached = any find route; only-it = no other find source and "
        "no other move reached it)",
        f"{'source':<26}{'steps':>7}{'failed':>8}{'reached':>9}{'keep%':>7}{'only-it':>8}"
        f"{'only-kept':>10}  failure kinds",
    ]
    for source, s in sorted(report.by_source.items(), key=lambda kv: -kv[1].steps):
        if not s.steps:
            continue
        kinds = ", ".join(f"{k} {v}" for k, v in s.failed.most_common())
        lines.append(
            f"{source:<26}{s.steps:>7}{_pct(sum(s.failed.values()), s.steps):>8}"
            f"{s.reached or '':>9}{_pct(s.reached_kept, s.reached) if s.reached else '':>7}"
            f"{s.exclusive or '':>8}{s.exclusive_kept or '':>10}  {kinds}"
        )
    related = report.by_move.get("related")
    if related and related.moves:
        lines += [
            "",
            f"related moves using negatives: {report.related_with_negatives}/{related.moves}",
        ]
    lines.append(
        "Caveats: marks are the driving agent's; first-route credit understates confirming moves."
    )
    return "\n".join(lines)
