"""Replay a recorded hunt's searching moves in a fresh session and grade it.

The replay bench measures the tool, not the agent: the same moves (queries,
seeds, snippet sentences, added papers) run again through the real CLI, so a
change to sources, endpoints, fallbacks, caching or ranking shows up as a
change in target recall, target rank, calls and refusals.  Judging moves
(show, mark, read, rank, report) are not replayed.

Seeds given as candidate ids are translated to the paper's Semantic Scholar
request id from the recorded session, because candidate ids are assigned in
arrival order and differ between runs.  Replays run in a separate store
(``<store root>/eval-store``) so they never appear among real hunts, while
sharing the cross-process rate limiter with every live process.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lit_search.evaluation.grading import TierGrade, grade, session_items
from lit_search.evaluation.usage import failure_class
from lit_search.hunt.session import HuntSession
from lit_search.runners.explore_runner import s2_request_id_for_paper
from lit_search.store.store import STORE_ROOT_ENV_VAR, default_store_root
from lit_search.utils.rate_limit import rate_limit_stats, reset_rate_limit_stats

REPLAYED_KINDS = ("find", "related", "snippets", "snowball", "add")
_CID = re.compile(r"^c\d+$")


@dataclass
class ReplayResult:
    """What one replay measured."""

    source_hunt: str
    replay_hunt: str
    moves: int
    skipped: list[str] = field(default_factory=list)
    seconds: float = 0.0
    requests: dict[str, int] = field(default_factory=dict)
    failures: dict[str, int] = field(default_factory=dict)
    candidates: int = 0
    grades: dict[str, list[TierGrade]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready view."""
        return {
            "source_hunt": self.source_hunt,
            "replay_hunt": self.replay_hunt,
            "moves": self.moves,
            "skipped": self.skipped,
            "seconds": round(self.seconds, 1),
            "requests": self.requests,
            "failures": self.failures,
            "candidates": self.candidates,
            "grades": {
                case_id: [
                    {
                        "label": t.label,
                        "retrieved": t.retrieved,
                        "of": len(t.entries),
                        "entries": [e.__dict__ for e in t.entries],
                    }
                    for t in tiers
                ]
                for case_id, tiers in self.grades.items()
            },
        }


def eval_store_root() -> Path:
    """Where replay sessions live."""
    return default_store_root() / "eval-store"


@contextlib.contextmanager
def _store_root(path: Path) -> Iterator[None]:
    previous = os.environ.get(STORE_ROOT_ENV_VAR)
    os.environ[STORE_ROOT_ENV_VAR] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(STORE_ROOT_ENV_VAR, None)
        else:
            os.environ[STORE_ROOT_ENV_VAR] = previous


def _seed(original: HuntSession, ref: str) -> str | None:
    if not _CID.match(ref):
        return ref
    candidate = original.candidates.get(ref)
    if candidate is None:
        return None
    return s2_request_id_for_paper(candidate.paper)


def _words(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return str(value).split()


def move_argv(move: dict[str, Any], original: HuntSession, hunt_id: str) -> list[str] | None:
    """Rebuild a recorded move as ``lit-search`` arguments, or ``None`` if it cannot be.

    Parameters
    ----------
    move : dict[str, Any]
        A move from the recorded session.
    original : HuntSession
        The recorded session (for translating candidate ids).
    hunt_id : str
        The replay session's id.

    Returns
    -------
    list[str] | None
        Arguments for :func:`lit_search.cli.main.main`.
    """
    kind = move.get("kind")
    if kind not in REPLAYED_KINDS:
        return None
    if (move.get("args") or {}).get("retry_of"):
        return None
    params = dict((move.get("replay") or {}).get("params") or {})
    args = move.get("args") or {}
    argv = ["hunt", str(kind), hunt_id, "--brief", "--quiet"]
    if kind in ("find", "snippets"):
        text = _words(params.get("text") or args.get("text"))
        if not text:
            return None
        argv += text
        if params.get("k"):
            argv += ["-k", str(params["k"])]
        for name in ("since", "until"):
            if params.get(name) or args.get(name):
                argv += [f"--{name}", str(params.get(name) or args.get(name))]
        for database in params.get("databases") or []:
            argv += ["-d", database]
        if kind == "snippets" and (params.get("field") or args.get("field")):
            argv += ["--field", str(params.get("field") or args.get("field"))]
        return argv
    if kind == "add":
        refs = params.get("refs") or [r.strip() for r in str(args.get("refs") or "").split("|")]
        refs = [r for r in refs if r]
        return argv + refs if refs else None
    seeds = [_seed(original, r) for r in _words(params.get("seeds") or args.get("seeds"))]
    if not seeds or any(s is None for s in seeds):
        return None
    argv += [s for s in seeds if s]
    if kind == "related":
        negatives = params.get("negative") or _words(args.get("not"))
        for ref in negatives:
            translated = _seed(original, ref)
            if translated:
                argv += ["--not", translated]
        if params.get("recs"):
            argv += ["--recs", str(params["recs"])]
        return argv
    direction = args.get("direction") or ""
    if params.get("forward") or direction == "forward":
        argv.append("--forward")
    if params.get("backward") or direction == "backward":
        argv.append("--backward")
    for name in ("since", "until"):
        if params.get(name) or args.get(name):
            argv += [f"--{name}", str(params.get(name) or args.get(name))]
    if params.get("max"):
        argv += ["--max", str(params["max"])]
    return argv


def replay_hunt(
    source_hunt: str, cases: list[dict[str, Any]], *, minutes: float = 240
) -> ReplayResult:
    """Replay *source_hunt*'s searching moves and grade the result against *cases*.

    Parameters
    ----------
    source_hunt : str
        Recorded hunt id (in the real hunts store).
    cases : list[dict[str, Any]]
        Confidence-set cases the replay is graded against.
    minutes : float
        Budget of the replay session (large: replays are not timed).

    Returns
    -------
    ReplayResult
        Grades, calls, refusals and time.
    """
    from lit_search.cli.main import main

    original = HuntSession.load(source_hunt)
    question = str(original.data.get("question") or "replay")
    profile = str(original.data.get("profile") or "prior-art")
    root = eval_store_root()
    root.mkdir(parents=True, exist_ok=True)
    reset_rate_limit_stats()
    started = time.monotonic()
    with _store_root(root):
        out = io.StringIO()
        start_argv = ["hunt", "start", question, "--profile", profile, "--minutes", str(minutes)]
        if original.data.get("wide"):
            start_argv.append("--wide")
        with contextlib.redirect_stdout(out):
            main(start_argv)
        match = re.search(r"hunt-\d{8}-\d{4}-[0-9a-f]+", out.getvalue())
        if match is None:
            raise RuntimeError(f"could not start a replay session: {out.getvalue()[:200]}")
        replay_id = match.group(0)
        result = ReplayResult(source_hunt, replay_id, moves=0)
        for move in original.data.get("moves") or []:
            argv = move_argv(move, original, replay_id)
            if argv is None:
                if move.get("kind") in REPLAYED_KINDS:
                    result.skipped.append(f"move {move.get('number')} {move.get('kind')}")
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                main(argv)
            result.moves += 1
        session = HuntSession.load(replay_id)
    result.seconds = time.monotonic() - started
    for stats in rate_limit_stats():
        service = stats.scope_key.split(":")[0]
        result.requests[service] = result.requests.get(service, 0) + stats.requests
    for move in session.data.get("moves") or []:
        for step in move.get("steps") or []:
            if isinstance(step, dict) and (step.get("status") == "failed" or step.get("error")):
                key = f"{str(step.get('source')).split(':')[0]}:{failure_class(step)}"
                result.failures[key] = result.failures.get(key, 0) + 1
    result.candidates = len(session.candidates)
    items = session_items(session)
    for case in cases:
        result.grades[case["id"]], _ = grade(case, items)
    return result


def render_replay(result: ReplayResult) -> str:
    """Render one replay as text."""
    lines = [
        f"replay of {result.source_hunt} as {result.replay_hunt}: {result.moves} moves, "
        f"{result.candidates} candidates, {result.seconds:.0f}s",
        "  requests: " + (", ".join(f"{k} {v}" for k, v in sorted(result.requests.items())) or "-"),
        "  failed steps: "
        + (", ".join(f"{k} {v}" for k, v in sorted(result.failures.items())) or "none"),
    ]
    if result.skipped:
        lines.append(f"  not replayable: {', '.join(result.skipped)}")
    for case_id, tiers in result.grades.items():
        for tier in tiers:
            found = [e for e in tier.entries if e.retrieved]
            ranks = ", ".join(f"{e.name[:40]} @{e.score_rank}" for e in found)
            lines.append(
                f"  {case_id} {tier.label}: {tier.retrieved}/{len(tier.entries)} retrieved"
                + (f" (score ranks: {ranks})" if ranks else "")
            )
    return "\n".join(lines)
