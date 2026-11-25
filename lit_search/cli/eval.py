"""``lit-search eval``: measure how the tool performs in real use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lit_search.cli.output import EXIT_OK, emit_json, emit_text
from lit_search.evaluation.grading import load_case
from lit_search.evaluation.replay import render_replay, replay_hunt
from lit_search.evaluation.requests import analyse, read_log, render_requests
from lit_search.evaluation.usage import load_sessions, render, summarize
from lit_search.hunt.session import hunts_root
from lit_search.utils.request_log import log_path

DEFAULT_CASES = Path(__file__).resolve().parents[2] / "docs" / "confidence" / "cases.json"
DEFAULT_BENCH = Path(__file__).resolve().parents[2] / "docs" / "confidence" / "bench.json"


def add_eval_parser(subparsers: argparse._SubParsersAction, parents: list) -> None:
    """Register ``eval`` and its actions.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparser registry.
    parents : list
        Shared parent parsers.

    Returns
    -------
    None
    """
    from lit_search.cli.parser import iso_date

    parser = subparsers.add_parser(
        "eval",
        help="measure the tool: usage statistics over saved hunts",
        description="Measure how lit-search performs, from the hunts it has already run.",
    )
    actions = parser.add_subparsers(dest="action", metavar="ACTION", required=True)
    usage = actions.add_parser(
        "usage",
        parents=parents,
        help="failure rates, calls and kept-paper yield per move and per source",
    )
    usage.add_argument("--since", type=iso_date, help="only hunts started on or after this date")
    usage.add_argument("--hunts-dir", type=Path, help="hunts directory (default: the store's)")

    requests = actions.add_parser(
        "requests",
        parents=parents,
        help="how each service refused: 429 rate by request gap and by trailing request count",
    )
    requests.add_argument("--hours", type=float, default=None, help="only the last N hours")

    replay = actions.add_parser(
        "replay",
        parents=parents,
        help="re-run a recorded hunt's searching moves in a fresh session and grade it",
    )
    replay.add_argument("case", help="case id in the cases file")
    replay.add_argument("source_hunt", help="recorded hunt whose moves are replayed")
    replay.add_argument("--cases", type=Path, default=DEFAULT_CASES)

    bench = actions.add_parser(
        "bench",
        parents=parents,
        help="replay every (case, hunt) pair of the bench file; the tool-level regression check",
    )
    bench.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    bench.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    bench.add_argument("--only", action="append", default=[], metavar="CASE")
    bench.add_argument("--out", type=Path, help="also write the JSON results here")


def cmd_eval(args: argparse.Namespace) -> int:
    """Run an ``eval`` action.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code.
    """
    if args.action == "requests":
        import time

        since = time.time() - args.hours * 3600 if args.hours else None
        path = log_path()
        emit_text(render_requests(analyse(read_log([path.with_suffix(".jsonl.1"), path], since))))
        return EXIT_OK
    if args.action == "replay":
        result = replay_hunt(args.source_hunt, [load_case(args.cases, args.case)])
        if args.json:
            emit_json(result.to_dict())
        else:
            emit_text(render_replay(result))
        return EXIT_OK
    if args.action == "bench":
        pairs = json.loads(args.bench.read_text(encoding="utf-8"))["pairs"]
        results = []
        for pair in pairs:
            if args.only and not set(args.only) & set(pair["cases"]):
                continue
            cases = [load_case(args.cases, c) for c in pair["cases"]]
            result = replay_hunt(pair["hunt"], cases)
            results.append(result.to_dict())
            if not args.json:
                emit_text(render_replay(result))
        tier1 = [g[0] for r in results for g in r["grades"].values() if g]
        totals = {
            "retrieved": sum(t["retrieved"] for t in tier1),
            "targets": sum(t["of"] for t in tier1),
            "requests": sum(sum(r["requests"].values()) for r in results),
            "failed_steps": sum(sum(r["failures"].values()) for r in results),
            "seconds": round(sum(r["seconds"] for r in results), 1),
        }
        if args.out:
            args.out.write_text(json.dumps({"totals": totals, "results": results}, indent=1))
        if args.json:
            emit_json({"totals": totals, "results": results})
        else:
            emit_text(
                f"bench: tier 1 {totals['retrieved']}/{totals['targets']} retrieved · "
                f"{totals['requests']} requests · {totals['failed_steps']} failed steps · "
                f"{totals['seconds']:.0f}s"
            )
        return EXIT_OK
    report = summarize(load_sessions(args.hunts_dir or hunts_root(), args.since))
    if args.json:
        emit_json(report.to_dict())
    else:
        emit_text(render(report))
    return EXIT_OK
