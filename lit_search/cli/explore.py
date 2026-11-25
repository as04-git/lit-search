"""The ``related`` and ``snippets`` subcommands: bounded exploration moves."""

from __future__ import annotations

import argparse
import textwrap
from typing import Any

from lit_search.cli.context import resolve_store
from lit_search.cli.emit import write_json_output
from lit_search.cli.find import _paper_id, _short
from lit_search.cli.output import (
    EXIT_ERROR,
    EXIT_INCOMPLETE,
    EXIT_OK,
    emit_error,
    emit_json,
    emit_text,
)
from lit_search.connectors.semantic_scholar import SemanticScholarConnector
from lit_search.core.coverage import CoverageLedger, CoverageStatus, ExecutionStatus, SourceCoverage
from lit_search.core.search_result import SearchResult
from lit_search.exceptions import InvalidParameterError
from lit_search.runners.explore_runner import (
    ExploreOutcome,
    run_related,
    run_snippets,
    s2_connector,
    s2_request_id,
    s2_request_id_for_paper,
)
from lit_search.store.operations import from_search_result

_MAX_SEEDS = 20


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 3 when any step failed or fell short (default: report and carry on)",
    )
    parser.add_argument(
        "--handle", action="store_true", help="store the result as a handle and print its id"
    )
    parser.add_argument("--label", metavar="TEXT", help="human-readable name for the handle")
    parser.set_defaults(limit=20)


def add_explore_parsers(subparsers: argparse._SubParsersAction, parents: list) -> None:
    """Register ``related`` and ``snippets``.

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

    related = subparsers.add_parser(
        "related",
        parents=parents,
        help="papers related to seeds: recommendations plus one hop of citations",
        description=(
            "Expand seed papers through Semantic Scholar recommendations and the papers "
            "they cite and are cited by. A fixed handful of requests; never a traversal."
        ),
    )
    related.add_argument(
        "seeds", nargs="+", help="DOIs, arXiv ids, Semantic Scholar ids, or handles (h_...)"
    )
    related.add_argument(
        "--not",
        dest="negative",
        action="append",
        default=[],
        metavar="ID",
        help="steer recommendations away from this paper; repeatable",
    )
    related.add_argument(
        "--recs", type=int, default=30, metavar="N", help="recommendations (0 skips; default 30)"
    )
    related.add_argument("--no-refs", action="store_true", help="skip papers the seeds cite")
    related.add_argument("--no-cites", action="store_true", help="skip papers citing the seeds")
    related.add_argument(
        "--neighbours",
        type=positive_int,
        default=60,
        metavar="N",
        help="citation neighbours kept, most-connected first (default 60)",
    )
    _common(related)

    snippets = subparsers.add_parser(
        "snippets",
        parents=parents,
        help="full-text search: papers whose body text matches, with the sentence",
        description=(
            "Search paper full text through Semantic Scholar's snippet index. Finds work "
            "that says the thing in its body even when the title and abstract use other words."
        ),
    )
    snippets.add_argument("text", nargs="+", help="what the passage should say")
    snippets.add_argument(
        "-k", type=positive_int, default=20, metavar="N", help="snippets requested (default 20)"
    )
    snippets.add_argument(
        "--since", type=iso_date, metavar="YYYY-MM-DD", help="earliest publication"
    )
    snippets.add_argument("--until", type=iso_date, metavar="YYYY-MM-DD", help="latest publication")
    snippets.add_argument(
        "--field",
        metavar="NAME",
        help='field-of-study filter, e.g. "Computer Science" (snippet matching is lexical; '
        "phrase the text the way a paper would)",
    )
    _common(snippets)


def _connector() -> SemanticScholarConnector:
    return s2_connector()


def _is_shortfall(step: SourceCoverage) -> bool:
    if step.status is CoverageStatus.FAILED:
        return True
    return step.execution_status is not ExecutionStatus.COMPLETE


def _step_line(step: SourceCoverage) -> str:
    name = f"{step.source:<24}"
    if step.status is CoverageStatus.FAILED:
        return f"  FAIL {name} {step.error_type}: {_short(step.error)}"
    tag = "part" if _is_shortfall(step) else "ok  "
    detail = f"  ({_short(step.detail)})" if step.detail else ""
    return f"  {tag} {name} {step.retrieved}{detail}"


def _emit(command: str, title: str, outcome: ExploreOutcome, args: argparse.Namespace) -> int:
    result = SearchResult(
        query=title,
        papers=outcome.papers,
        runtime_seconds=outcome.seconds,
        databases=["semantic_scholar"],
        enrichment_databases=[],
        coverage=CoverageLedger.from_sources(outcome.steps),
    )
    handle_id = (
        resolve_store(args).put(from_search_result(result, label=args.label))
        if args.handle
        else None
    )
    written = write_json_output(result, args)
    if not args.quiet:
        if args.json:
            payload: dict[str, Any] = {
                "command": command,
                "query": title,
                "steps": [s.to_dict() for s in outcome.steps],
                "shortfalls": [s.source for s in outcome.steps if _is_shortfall(s)],
                "paper_count": len(outcome.papers),
                "papers": [
                    {
                        **p.to_dict(),
                        "routes": outcome.routes.get(id(p), []),
                        "evidence": outcome.evidence.get(id(p), []),
                    }
                    for p in outcome.papers
                ],
                "output": written,
                "handle": handle_id,
            }
            emit_json(payload)
        else:
            lines = [
                f"{command}: {title} -- {len(outcome.papers)} papers in {outcome.seconds:.1f}s"
            ]
            lines += [_step_line(s) for s in outcome.steps]
            lines.append("")
            limit = len(outcome.papers) if args.limit is None else args.limit
            for index, paper in enumerate(outcome.papers[:limit], start=1):
                date = (
                    paper.publication_date.strftime("%Y-%m") if paper.publication_date else "n.d."
                )
                cited = f" · cited {paper.citations}" if paper.citations is not None else ""
                route = ", ".join(outcome.routes.get(id(paper), []))
                lines.append(f"{index:3}. {paper.title}")
                lines.append(f"     {date} · {_paper_id(paper)}{cited} · {route}")
                for sentence in outcome.evidence.get(id(paper), [])[:1]:
                    snippet = textwrap.shorten(sentence, width=300, placeholder=" ...")
                    lines += textwrap.wrap(
                        f'"{snippet}"', width=96, initial_indent="     ", subsequent_indent="      "
                    )
            if limit < len(outcome.papers):
                lines.append(f"  ... {len(outcome.papers) - limit} more (--limit 0 shows all)")
            short = [s.source for s in outcome.steps if _is_shortfall(s)]
            if short and outcome.papers:
                lines.append(f"! fell short: {', '.join(short)}; results come from the rest")
            emit_text("\n".join(lines))
            if handle_id:
                emit_text(f"Handle: {handle_id}")
            if written:
                emit_text(f"Written: {written}")
    if not outcome.papers and any(s.status is CoverageStatus.FAILED for s in outcome.steps):
        if args.quiet or args.json:
            emit_error(f"{command}: no step produced papers; see the step errors")
        return EXIT_ERROR
    if args.strict and any(_is_shortfall(s) for s in outcome.steps):
        return EXIT_INCOMPLETE
    return EXIT_OK


def _resolve_seeds(raw: list[str], args: argparse.Namespace) -> tuple[list[str], list[str]]:
    seeds: list[str] = []
    skipped: list[str] = []
    for value in raw:
        if value.startswith("h_"):
            store = resolve_store(args)
            for paper in store.get(store.resolve(value)).members:
                request_id = s2_request_id_for_paper(paper)
                if request_id is None:
                    skipped.append(paper.title or "untitled")
                else:
                    seeds.append(request_id)
        else:
            seeds.append(s2_request_id(value))
    return list(dict.fromkeys(seeds)), skipped


def cmd_related(args: argparse.Namespace) -> int:
    """Expand seed papers.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code.
    """
    seeds, skipped = _resolve_seeds(args.seeds, args)
    if skipped:
        emit_error(f"{len(skipped)} seed paper(s) carry no id Semantic Scholar accepts; skipped")
    if len(seeds) > _MAX_SEEDS:
        emit_error(f"using the first {_MAX_SEEDS} of {len(seeds)} seeds")
        seeds = seeds[:_MAX_SEEDS]
    if not seeds:
        raise InvalidParameterError("no usable seed papers")
    outcome = run_related(
        _connector(),
        seeds,
        negative=[s2_request_id(n) for n in args.negative],
        recommendations=max(0, args.recs),
        references=not args.no_refs,
        citations=not args.no_cites,
        max_neighbours=args.neighbours,
    )
    return _emit("related", " ".join(args.seeds), outcome, args)


def cmd_snippets(args: argparse.Namespace) -> int:
    """Search paper full text.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code.
    """
    text = " ".join(args.text)
    outcome = run_snippets(
        _connector(),
        text,
        limit=args.k,
        since=args.since,
        until=args.until,
        field_of_study=args.field,
    )
    return _emit("snippets", text, outcome, args)


__all__ = ["add_explore_parsers", "cmd_related", "cmd_snippets"]
