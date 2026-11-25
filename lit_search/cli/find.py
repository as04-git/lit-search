"""The ``find`` subcommand: instant relevance search with a compact status block."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import textwrap
from typing import Any

from lit_search import credentials
from lit_search.cli.context import resolve_store
from lit_search.cli.emit import write_json_output
from lit_search.cli.output import (
    EXIT_ERROR,
    EXIT_INCOMPLETE,
    EXIT_OK,
    emit_error,
    emit_json,
    emit_text,
)
from lit_search.core.coverage import CoverageStatus, ExecutionStatus, SourceCoverage
from lit_search.core.paper import Paper
from lit_search.runners.find_runner import (
    DEFAULT_DEADLINE_SECONDS,
    DEFAULT_PER_SOURCE,
    FIND_SOURCES,
    FindOutcome,
    FindRequest,
    build_find_connectors,
    run_find,
)
from lit_search.store.operations import from_search_result

_SOURCE_TAGS = {
    "semantic_scholar": "S2",
    "openalex": "OA",
    "arxiv": "AX",
    "openaire": "OE",
    "huggingface": "HF",
}
_SORTS = ("relevance", "recent", "cited")


def add_find_parser(subparsers: argparse._SubParsersAction, parents: list) -> None:
    """Register the ``find`` subcommand.

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
    from lit_search.cli.parser import iso_date, positive_int

    parser = subparsers.add_parser(
        "find",
        parents=parents,
        help="instant relevance search: one ranked request per source, fused",
        description=(
            "Free-text relevance search. Sends one ranked request to each source in "
            "parallel and fuses the lists. Every source's status is shown; a failed "
            "source is reported and the others still answer."
        ),
    )
    parser.add_argument("text", nargs="+", help='free text; "quoted spans" stay phrases')
    parser.add_argument(
        "-d",
        "--database",
        action="append",
        dest="databases",
        choices=FIND_SOURCES,
        metavar="DB",
        help=f"restrict to this source; repeatable (default: {', '.join(FIND_SOURCES)})",
    )
    parser.add_argument(
        "-k",
        "--per-source",
        type=positive_int,
        default=DEFAULT_PER_SOURCE,
        metavar="N",
        help=f"ranked results requested from each source (default: {DEFAULT_PER_SOURCE})",
    )
    parser.add_argument("--since", type=iso_date, metavar="YYYY-MM-DD", help="earliest publication")
    parser.add_argument("--until", type=iso_date, metavar="YYYY-MM-DD", help="latest publication")
    parser.add_argument(
        "--sort",
        choices=_SORTS,
        default="relevance",
        help="order of the fused list (default: relevance)",
    )
    parser.add_argument(
        "--deadline",
        type=float,
        default=DEFAULT_DEADLINE_SECONDS,
        metavar="SECONDS",
        help=f"stop waiting for slow sources after this long (default: {DEFAULT_DEADLINE_SECONDS:g})",
    )
    parser.add_argument("--abstracts", action="store_true", help="show the start of each abstract")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 3 when any source failed or fell short (default: report and carry on)",
    )
    parser.add_argument(
        "--handle", action="store_true", help="store the result as a handle and print its id"
    )
    parser.add_argument("--label", metavar="TEXT", help="human-readable name for the handle")
    parser.set_defaults(limit=20)


def _paper_id(paper: Paper) -> str:
    doi = paper.doi or ""
    if doi.lower().startswith("10.48550/arxiv."):
        return "arXiv:" + doi.split(".", 2)[2]
    if doi:
        return f"doi:{doi}"
    return paper.url or "no identifier"


def _sorted(papers: list[Paper], sort: str) -> list[Paper]:
    if sort == "recent":
        return sorted(papers, key=lambda p: p.publication_date or dt.date.min, reverse=True)
    if sort == "cited":
        return sorted(papers, key=lambda p: p.citations or 0, reverse=True)
    return papers


_URL_QUERY_RE = re.compile(r"(https?://[^\s?]+)\?\S*")
_RETRIEVED_PREFIX_RE = re.compile(r"^retrieved \d+ of \d+ records the source reported; ")


def _short(text: str | None) -> str:
    return _URL_QUERY_RE.sub(r"\1?...", text or "")


def _status_line(coverage: SourceCoverage, seconds: float) -> str:
    name = f"{coverage.source:<17}"
    timing = f"{seconds:5.1f}s"
    if coverage.status is CoverageStatus.FAILED:
        return f"  FAIL {name} {timing}  {coverage.error_type}: {_short(coverage.error)}"
    if coverage.status is CoverageStatus.SKIPPED_BY_POLICY:
        return f"  skip {name} {timing}  {_short(coverage.detail)}"
    if coverage.status is CoverageStatus.SKIPPED_NO_CREDENTIAL:
        return f"  skip {name} {timing}  missing credential {coverage.missing_credential}"
    total = f"{coverage.reported_total:,}" if coverage.reported_total is not None else "?"
    if coverage.status is CoverageStatus.CAPPED:
        of_total = (
            f"of {total} matches"
            if coverage.reported_total is not None
            else "(source gives no total)"
        )
        return f"  ok   {name} {timing}  top {coverage.retrieved} {of_total}"
    if coverage.status is CoverageStatus.COMPLETE:
        return f"  ok   {name} {timing}  all {coverage.retrieved} matches"
    detail = _RETRIEVED_PREFIX_RE.sub("", _short(coverage.detail))
    if coverage.execution_status is ExecutionStatus.COMPLETE:
        return f"  ok   {name} {timing}  {coverage.retrieved} ranked (source gives no total)"
    return f"  part {name} {timing}  {coverage.retrieved} of {total}: {detail}"


def _is_shortfall(coverage: SourceCoverage) -> bool:
    if coverage.status in (CoverageStatus.FAILED, CoverageStatus.SKIPPED_BY_POLICY):
        return True
    return coverage.execution_status is not ExecutionStatus.COMPLETE


def _render(outcome: FindOutcome, papers: list[Paper], args: argparse.Namespace) -> str:
    result = outcome.result
    ledger = list(result.coverage or [])
    answered = [
        c
        for c in ledger
        if c.status not in (CoverageStatus.FAILED, CoverageStatus.SKIPPED_BY_POLICY)
    ]
    lines = [
        f"find: {result.query} -- {len(papers)} papers from {len(answered)}/{len(ledger)} "
        f"sources in {result.runtime_seconds or 0:.1f}s"
    ]
    lines += [_status_line(c, outcome.seconds_per_source.get(c.source, 0.0)) for c in ledger]
    if outcome.title_merges:
        lines.append(
            f"  note {outcome.title_merges} record(s) merged on matching title and year "
            "(no conflicting DOI)"
        )
    lines.append("")
    limit = len(papers) if args.limit is None else args.limit
    for index, paper in enumerate(papers[:limit], start=1):
        ranks = outcome.ranks.get(id(paper), {})
        tags = " ".join(f"{_SOURCE_TAGS.get(s, s)}#{r}" for s, r in sorted(ranks.items()))
        date = paper.publication_date.strftime("%Y-%m") if paper.publication_date else "n.d."
        cited = f" · cited {paper.citations}" if paper.citations is not None else ""
        lines.append(f"{index:3}. {paper.title}")
        lines.append(f"     {date} · {_paper_id(paper)}{cited} · {tags}")
        if args.abstracts and paper.abstract:
            snippet = textwrap.shorten(paper.abstract, width=320, placeholder=" ...")
            lines += textwrap.wrap(
                snippet, width=96, initial_indent="     ", subsequent_indent="     "
            )
    if limit < len(papers):
        lines.append(f"  ... {len(papers) - limit} more (--limit 0 shows all)")
    failed = [c for c in ledger if _is_shortfall(c)]
    if failed and answered:
        lines.append(
            f"! {len(failed)} of {len(ledger)} sources fell short "
            f"({', '.join(c.source for c in failed)}); results come from the rest"
        )
    return "\n".join(lines)


def _payload(outcome: FindOutcome, papers: list[Paper], written: str | None) -> dict[str, Any]:
    ledger = list(outcome.result.coverage or [])
    return {
        "command": "find",
        "query": outcome.result.query,
        "sources": [
            {**c.to_dict(), "seconds": round(outcome.seconds_per_source.get(c.source, 0.0), 2)}
            for c in ledger
        ],
        "shortfalls": [c.source for c in ledger if _is_shortfall(c)],
        "title_merges": outcome.title_merges,
        "paper_count": len(papers),
        "papers": [
            {**paper.to_dict(), "ranks": outcome.ranks.get(id(paper), {})} for paper in papers
        ],
        "errors": outcome.tracebacks,
        "output": written,
    }


def cmd_find(args: argparse.Namespace) -> int:
    """Run an instant relevance search.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        0 when at least one source answered (shortfalls are shown, not fatal);
        1 when no source answered; 3 under ``--strict`` when any source fell
        short.
    """
    sources = tuple(dict.fromkeys(args.databases or FIND_SOURCES))
    connectors = build_find_connectors(
        sources,
        semantic_scholar_api_key=credentials.get("SEMANTIC_SCHOLAR_API_KEY"),
        openalex_api_key=credentials.get("OPENALEX_API_KEY"),
        openaire_access_token=credentials.get("OPENAIRE_ACCESS_TOKEN"),
        openaire_refresh_token=credentials.get("OPENAIRE_REFRESH_TOKEN"),
    )
    request = FindRequest(
        text=" ".join(args.text),
        per_source=args.per_source,
        since=args.since,
        until=args.until,
    )
    outcome = run_find(request, connectors, deadline_seconds=args.deadline)
    papers = _sorted(list(outcome.result.papers), args.sort)
    outcome.result.papers = papers

    for source, trace in outcome.tracebacks.items():
        emit_error(f"bug in the {source} path (the other sources still answered):\n{trace}")

    handle_id = None
    if args.handle:
        handle_id = resolve_store(args).put(from_search_result(outcome.result, label=args.label))
    written = write_json_output(outcome.result, args)
    if not args.quiet:
        if args.json:
            payload = _payload(outcome, papers, written)
            if handle_id:
                payload["handle"] = handle_id
            emit_json(payload)
        else:
            emit_text(_render(outcome, papers, args))
            if handle_id:
                emit_text(f"Handle: {handle_id}")
            if written:
                emit_text(f"Written: {written}")

    ledger = list(outcome.result.coverage or [])
    if all(c.status is CoverageStatus.FAILED for c in ledger):
        if args.quiet or args.json:
            emit_error("no source answered; see the per-source errors")
        return EXIT_ERROR
    if args.strict and any(_is_shortfall(c) for c in ledger):
        return EXIT_INCOMPLETE
    return EXIT_OK


__all__ = ["add_find_parser", "cmd_find"]
