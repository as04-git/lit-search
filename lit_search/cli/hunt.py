"""The ``hunt`` subcommand family: a time-budgeted working set an agent drives.

Typical loop (see ``docs/hunt-playbook.md``)::

    lit-search hunt start "question" --profile prior-art      # prints the hunt id
    lit-search hunt find HUNT "a re-description of the idea"
    lit-search hunt add HUNT 2404.08819 "Title of a paper you already know"
    lit-search hunt snowball HUNT c3 c7 --forward --since 2025-06-01
    lit-search hunt related HUNT c3 c7
    lit-search hunt snippets HUNT "a sentence a paper would contain"
    lit-search hunt search HUNT 'abs[model checking] AND abs[specification]'
    lit-search hunt mark HUNT c3 c7 --keep --note "same mechanism"
    lit-search hunt report HUNT
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import textwrap
import time
from typing import Any

from lit_search import credentials
from lit_search.cli.context import resolve_store
from lit_search.cli.find import _paper_id, _short
from lit_search.cli.output import (
    EXIT_ERROR,
    EXIT_INCOMPLETE,
    EXIT_OK,
    emit_error,
    emit_json,
    emit_text,
)
from lit_search.core.coverage import CoverageLedger, CoverageStatus, ExecutionStatus, SourceCoverage
from lit_search.core.paper import Paper
from lit_search.core.search_result import SearchResult
from lit_search.exceptions import InvalidParameterError
from lit_search.hunt.session import (
    DEFAULT_MINUTES,
    DEFAULT_SHORTLIST,
    PROFILE_GUIDE,
    PROFILES,
    Candidate,
    HuntSession,
    MoveRecord,
    format_seconds,
    hunts_root,
)
from lit_search.runners.explore_runner import (
    _hydrate,
    run_related,
    run_snippets,
    run_snowball,
    s2_connector,
    s2_request_id,
    s2_request_id_for_paper,
)
from lit_search.runners.find_runner import (
    FIND_SOURCES,
    FindRequest,
    build_find_connectors,
    run_find,
)
from lit_search.store.operations import from_search_result
from lit_search.utils.rate_limit import request_deadline
from lit_search.utils.relevance import relevance_scores

#: Wall-clock seconds a Semantic Scholar move may spend, waiting for request slots included.
#: Past it, a step reports itself busy (nothing is sent) instead of stalling the hunt.
MOVE_DEADLINES = {"related": 60.0, "snippets": 45.0, "snowball": 150.0}

#: Databases a hunt's Boolean ``search`` move queries unless told otherwise: broad,
#: keyless, and answering Boolean queries natively.
HUNT_SEARCH_DEFAULT = ("openalex", "semantic_scholar", "arxiv", "dblp")


def _rank_weight(rank: int, scale: float = 1.0) -> float:
    return scale * 30.0 / (29.0 + rank)


# ----------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------


def add_hunt_parser(subparsers: argparse._SubParsersAction, parents: list) -> None:
    """Register ``hunt`` and its actions.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparser registry.
    parents : list
        Shared parent parsers (``--json``, ``-v``, ``--quiet``, ``--store-root``).

    Returns
    -------
    None
    """
    from lit_search.cli.parser import SEARCH_DATABASES, iso_date, positive_int

    parser = subparsers.add_parser(
        "hunt",
        help="time-budgeted, purpose-driven search session for an agent to drive",
        description=(
            "A hunt keeps a durable working set across moves: short candidate ids, every "
            "route and evidence sentence, a wall-clock budget, and refusal to repeat a move. "
            "See docs/hunt-playbook.md."
        ),
    )
    actions = parser.add_subparsers(dest="hunt_action", metavar="ACTION", required=True)

    start = actions.add_parser("start", parents=parents, help="start a hunt and print its id")
    start.add_argument("question", nargs="+", help="what the hunt is for")
    start.add_argument("--profile", choices=PROFILES, required=True)
    start.add_argument(
        "--minutes",
        type=float,
        default=DEFAULT_MINUTES,
        help=f"wall-clock budget including the agent's own time (default {DEFAULT_MINUTES:g})",
    )
    start.add_argument(
        "--wide",
        action="store_true",
        help="moves return several times more candidates (find -k 50, related 50/120, snippets 60)",
    )
    start.add_argument(
        "--strict",
        action="store_true",
        help="enforce written reasoning and inspection between moves (see hunt-playbook-strict)",
    )
    start.add_argument(
        "--min-share",
        type=float,
        default=80,
        metavar="PERCENT",
        help="percent of the budget used before the report opens (default 80; known-item exempt)",
    )
    start.add_argument(
        "--shortlist",
        type=positive_int,
        default=DEFAULT_SHORTLIST,
        help=f"default shortlist size for the report (default {DEFAULT_SHORTLIST})",
    )
    start.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="HANDLE",
        help="leave out papers in this handle (already known, e.g. an earlier hunt's report); "
        "repeatable",
    )

    def move_parser(name: str, help_text: str) -> argparse.ArgumentParser:
        move: argparse.ArgumentParser = actions.add_parser(name, parents=parents, help=help_text)
        move.add_argument("hunt", help="hunt id or unique prefix")
        move.add_argument(
            "--overtime", action="store_true", help="run even though the budget is spent"
        )
        move.add_argument("--brief", action="store_true", help="omit abstracts of new candidates")
        return move

    find = move_parser("find", "relevance search across sources; adds candidates")
    find.add_argument("text", nargs="+")
    find.add_argument(
        "-k", type=positive_int, default=None, help="per source (default 15; 50 in a --wide hunt)"
    )
    find.add_argument("--since", type=iso_date, metavar="YYYY-MM-DD")
    find.add_argument("--until", type=iso_date, metavar="YYYY-MM-DD")
    find.add_argument(
        "-d", "--database", action="append", dest="databases", choices=FIND_SOURCES, metavar="DB"
    )

    related = move_parser("related", "recommendations + one citation hop from seeds")
    related.add_argument("seeds", nargs="+", help="candidate ids (c3) or DOIs / arXiv ids")
    related.add_argument("--not", dest="negative", action="append", default=[], metavar="REF")
    related.add_argument("--recs", type=int, default=None, help="default 20; 50 when --wide")
    related.add_argument("--no-refs", action="store_true")
    related.add_argument("--no-cites", action="store_true")
    related.add_argument(
        "--neighbours", type=positive_int, default=None, help="default 40; 120 when --wide"
    )

    snowball = move_parser(
        "snowball",
        "follow every citation edge of seeds: forward (citing papers) and/or backward (references)",
    )
    snowball.add_argument("seeds", nargs="+", help="candidate ids (c3) or DOIs / arXiv ids")
    snowball.add_argument(
        "--forward", action="store_true", help="only papers citing the seeds (default: both)"
    )
    snowball.add_argument(
        "--backward", action="store_true", help="only papers the seeds cite (default: both)"
    )
    snowball.add_argument("--since", type=iso_date, metavar="YYYY-MM-DD")
    snowball.add_argument("--until", type=iso_date, metavar="YYYY-MM-DD")
    snowball.add_argument(
        "--max",
        type=positive_int,
        default=None,
        help="neighbours kept (default 60; 200 when --wide)",
    )
    snowball.add_argument(
        "--edges", type=positive_int, default=2000, help="edges read per seed and direction"
    )

    add = move_parser(
        "add", "add papers you already know (ids, exact titles, or a handle's members)"
    )
    add.add_argument("refs", nargs="*", help='arXiv ids, DOIs, or "quoted titles"')
    add.add_argument(
        "--from",
        dest="handles",
        action="append",
        default=[],
        metavar="HANDLE",
        help="add every paper of a stored set (a search, snowball, or another hunt's report)",
    )

    search = move_parser(
        "search",
        "Boolean sweep across databases (exact terms, recall-heavy); adds candidates",
    )
    search.add_argument("query", help="Boolean query, e.g. 'abs[model checking] AND abs[LLM]'")
    search.add_argument(
        "-d",
        "--database",
        action="append",
        dest="databases",
        choices=SEARCH_DATABASES,
        metavar="DB",
        help=f"repeatable (default: {', '.join(HUNT_SEARCH_DEFAULT)})",
    )
    search.add_argument(
        "--max", type=positive_int, default=None, help="per database (default 50; 200 when --wide)"
    )
    search.add_argument("--since", type=iso_date, metavar="YYYY-MM-DD")
    search.add_argument("--until", type=iso_date, metavar="YYYY-MM-DD")

    snippets = move_parser("snippets", "full-text sentence search; adds candidates with evidence")
    snippets.add_argument("text", nargs="+")
    snippets.add_argument("-k", type=positive_int, default=None, help="default 20; 60 when --wide")
    snippets.add_argument("--field", metavar="NAME")

    show = actions.add_parser("show", parents=parents, help="full detail of candidates")
    show.add_argument("hunt")
    show.add_argument("cids", nargs="+")

    mark = actions.add_parser("mark", parents=parents, help="record judgments on candidates")
    mark.add_argument("hunt")
    mark.add_argument("cids", nargs="+")
    group = mark.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--core",
        action="store_const",
        const="core",
        dest="mark",
        help="does the central move (a near-duplicate): shortlisted first",
    )
    group.add_argument("--keep", action="store_const", const="keep", dest="mark")
    group.add_argument("--drop", action="store_const", const="drop", dest="mark")
    group.add_argument("--clear", action="store_const", const="clear", dest="mark")
    mark.add_argument("--note", default="", help="why (shown in the report)")

    retry = actions.add_parser(
        "retry",
        parents=parents,
        help="list or re-run moves whose sources failed (rate limits, errors); "
        "safe to run in the background while the hunt continues",
    )
    retry.add_argument("hunt")
    retry.add_argument("moves", nargs="*", type=int, help="move numbers (default: list them)")
    retry.add_argument("--all", action="store_true", help="re-run every failed move")
    retry.add_argument(
        "--wait",
        type=float,
        default=0.0,
        metavar="MINUTES",
        help="keep re-running still-failing moves, a minute apart, for up to this long",
    )

    exclude = actions.add_parser(
        "exclude",
        parents=parents,
        help="leave a handle's papers out of later moves (papers known from elsewhere)",
    )
    exclude.add_argument("hunt")
    exclude.add_argument("handles", nargs="+", metavar="HANDLE")

    rank = actions.add_parser(
        "rank", parents=parents, help="set the shortlist order yourself, best first"
    )
    rank.add_argument("hunt")
    rank.add_argument("cids", nargs="+")

    read = actions.add_parser(
        "read", parents=parents, help="fetch a candidate's PDF and print its intro or related work"
    )
    read.add_argument("hunt")
    read.add_argument("cid")
    read.add_argument(
        "--part",
        choices=("intro", "related", "conclusion", "all"),
        default="intro",
        help="section to print (default intro, which usually lists the contributions)",
    )
    read.add_argument("--chars", type=positive_int, default=6000, help="maximum characters")

    note = actions.add_parser(
        "note", parents=parents, help="record reasoning (plan, per-move, final)"
    )
    note.add_argument("hunt")
    note.add_argument("text", nargs="+")
    kind = note.add_mutually_exclusive_group()
    kind.add_argument("--plan", action="store_const", const="plan", dest="kind")
    kind.add_argument("--final", action="store_const", const="final", dest="kind")

    status = actions.add_parser("status", parents=parents, help="budget, moves and top candidates")
    status.add_argument("hunt")
    status.add_argument("--top", type=positive_int, default=15)
    status.add_argument(
        "--from",
        dest="start_at",
        type=positive_int,
        default=1,
        metavar="N",
        help="start the list at position N (page through a long unjudged queue)",
    )
    status.add_argument(
        "--unjudged",
        action="store_true",
        help="list only top candidates not yet marked, with abstracts (review before the report)",
    )

    extend = actions.add_parser("extend", parents=parents, help="add minutes to the budget")
    extend.add_argument("hunt")
    extend.add_argument("minutes", type=float)

    report = actions.add_parser("report", parents=parents, help="write the final report")
    report.add_argument("hunt")
    report.add_argument("--shortlist", type=positive_int, help="override the shortlist size")

    actions.add_parser("list", parents=parents, help="list hunts")


# ----------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------


def _footer(session: HuntSession) -> str:
    marks = [c.mark for c in session.candidates.values()]
    queue = session.snowball_queue()
    pending = (
        " · not yet snowballed: "
        + "; ".join(f"{d} {' '.join(cids)}" for d, cids in queue.items() if cids)
        if any(queue.values())
        else ""
    )
    left = session.remaining
    budget = (
        f"time left {format_seconds(left)}"
        if left > 0
        else f"BUDGET SPENT (over by {format_seconds(-left)})"
    )
    used = ""
    if left > 0 and session.data["profile"] != "known-item":
        used = f" · {session.budget_used:.0%} of budget used"
        if session.needs_more_time():
            used += f" (report opens at {session.min_share:.0%})"
    return (
        f"{budget}{used} · {len(session.candidates)} candidates ({marks.count('core')} core, "
        f"{marks.count('keep')} kept, {marks.count('drop')} dropped) · "
        f"{len(session.data['moves'])} moves{pending}"
    )


def _candidate_line(candidate: Candidate, *, abstract: bool, width: int = 240) -> list[str]:
    paper = candidate.paper
    date = paper.publication_date.strftime("%Y-%m") if paper.publication_date else "n.d."
    cited = f" · cited {paper.citations}" if paper.citations is not None else ""
    mark = f" [{candidate.mark}]" if candidate.mark else ""
    lines = [
        f"  {candidate.cid:<5}{paper.title}{mark}",
        f"       {date} · {_paper_id(paper)}{cited}",
    ]
    if candidate.evidence:
        sentence = textwrap.shorten(candidate.evidence[-1], width=240, placeholder=" ...")
        lines.append(f'       evidence: "{sentence}"')
    if abstract and paper.abstract:
        snippet = textwrap.shorten(paper.abstract, width=width, placeholder=" ...")
        lines += textwrap.wrap(
            snippet, width=100, initial_indent="       ", subsequent_indent="       "
        )
    return lines


def _step_problem(step: SourceCoverage) -> str | None:
    if step.status is CoverageStatus.FAILED:
        return f"  FAIL {step.source}: {step.error_type}: {_short(step.error)[:160]}"
    if step.execution_status is not ExecutionStatus.COMPLETE:
        return f"  part {step.source}: {_short(step.detail)[:160]}"
    return None


def _emit_move(
    session: HuntSession, record: MoveRecord, steps: list[SourceCoverage], args: argparse.Namespace
) -> int:
    answered = [s for s in steps if s.status is not CoverageStatus.FAILED]
    new, order_by = _order_new(session, record.new)
    with session.locked():
        for logged in session.data["moves"]:
            if logged["number"] == record.number:
                logged["shown"] = new
                logged["order_by"] = order_by
    if args.json:
        emit_json(
            {
                "hunt": session.data["id"],
                "move": record.__dict__,
                "new": [session.candidates[c].to_dict() for c in new],
                "order_by": order_by,
                "time_left_seconds": round(session.remaining),
            }
        )
    elif not args.quiet:
        label = " ".join(f"{k}={v}" for k, v in record.args.items() if v not in (None, [], False))
        again = f" ({' '.join(record.again[:12])}{' ...' if len(record.again) > 12 else ''})"
        lines = [
            f"move {record.number} {record.kind} {label} · {len(answered)}/{len(steps)} steps ok · "
            f"{record.seconds:.1f}s · +{len(record.new)} new, {len(record.again)} again"
            + (again if record.again else "")
            + (f", {record.excluded} excluded as known" if record.excluded else "")
        ]
        lines += [p for p in (_step_problem(s) for s in steps) if p]
        wide = bool(session.data.get("wide"))
        if order_by.startswith("relevance"):
            lines.append(f"  new candidates by relevance to the question ({order_by[10:]}):")
        elif wide:
            lines.append(f"  new candidates by score ({len(new)}); `hunt show` for full abstracts:")
        for cid in new:
            lines += _candidate_line(
                session.candidates[cid], abstract=not args.brief, width=160 if wide else 240
            )
        lines.append(_footer(session))
        emit_text("\n".join(lines))
    if steps and not answered:
        return EXIT_ERROR
    return EXIT_OK


_REPLAY_FIELDS: dict[str, tuple[str, ...]] = {
    "find": ("text", "k", "since", "until", "databases"),
    "related": ("seeds", "negative", "recs", "no_refs", "no_cites", "neighbours"),
    "snippets": ("text", "k", "field"),
    "snowball": ("seeds", "forward", "backward", "since", "until", "max", "edges"),
    "add": ("refs", "handles"),
    "search": ("query", "databases", "max", "since", "until"),
}


def _order_new(session: HuntSession, cids: list[str]) -> tuple[list[str], str]:
    """Order a move's new candidates: by relevance to the question when the local
    service answers (it put targets in an agent's first 10 twice as often, 2026-10-16
    first-exposure test), otherwise by the hunt's own score."""
    by_score = sorted(cids, key=lambda c: -session.score(session.candidates[c]))
    if len(by_score) < 3:
        return by_score, "score"
    docs = [
        f"{session.candidates[c].paper.title or ''}. {session.candidates[c].paper.abstract or ''}"
        for c in by_score
    ]
    scored = relevance_scores(str(session.data.get("question") or ""), docs)
    if scored is None:
        return by_score, "score"
    scores, model = scored
    order = sorted(range(len(by_score)), key=lambda i: -scores[i])
    return [by_score[i] for i in order], f"relevance:{model}"


def _replay(args: argparse.Namespace, kind: str) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name in _REPLAY_FIELDS[kind]:
        value = getattr(args, name, None)
        params[name] = value.isoformat() if isinstance(value, dt.date) else value
    return {"kind": kind, "params": params}


def _retry_suffix(args: argparse.Namespace) -> str:
    origin = getattr(args, "retry_of", None)
    return f"|retry:{origin}:{time.time():.3f}" if origin else ""


def _guard(session: HuntSession, key: str, args: argparse.Namespace) -> int | None:
    if getattr(args, "retry_of", None):
        return None
    previous = session.previous_move(key)
    if previous is not None:
        emit_text(
            f"already ran as move {previous['number']}; nothing done. Try a different "
            f"phrasing or seeds.\n{_footer(session)}"
        )
        return EXIT_OK
    if session.remaining <= 0 and not args.overtime:
        emit_text(
            f"budget spent (over by {format_seconds(-session.remaining)}): write the report "
            "(`hunt report`), or add time (`hunt extend`), or pass --overtime."
        )
        return EXIT_INCOMPLETE
    blockers = session.move_blockers()
    if blockers:
        emit_text("strict: move refused until you\n" + "\n".join(f"  - {b}" for b in blockers))
        return EXIT_INCOMPLETE
    return None


# ----------------------------------------------------------------------
# Actions
# ----------------------------------------------------------------------


def _start(args: argparse.Namespace) -> int:
    session = HuntSession.start(
        " ".join(args.question),
        profile=args.profile,
        minutes=args.minutes,
        shortlist=args.shortlist,
        strict=args.strict,
        wide=args.wide,
        min_share=args.min_share / 100,
    )
    excluded = _exclude_handles(session, args, args.exclude) if args.exclude else ""
    if args.json:
        emit_json({"hunt": session.data["id"], "path": str(session.path.parent)})
        return EXIT_OK
    emit_text(
        f"hunt {session.data['id']} · profile {args.profile} · budget {args.minutes:g} min · "
        f"shortlist {args.shortlist} · report opens at {args.min_share:g}%{' · STRICT' if args.strict else ''}{' · WIDE' if args.wide else ''}\nguide: {PROFILE_GUIDE[args.profile]}"
        + (
            "\nstrict: start with `hunt note ID --plan`; see docs/hunt-playbook-strict.md"
            if args.strict
            else ""
        )
        + (f"\nexcluded as known: {excluded}" if excluded else "")
    )
    return EXIT_OK


def _width(session: HuntSession, value: int | None, normal: int, wide: int) -> int:
    if value is not None:
        return value
    return wide if session.data.get("wide") else normal


def _find(session: HuntSession, args: argparse.Namespace) -> int:
    args.k = _width(session, args.k, 15, 50)
    text = " ".join(args.text)
    sources = tuple(dict.fromkeys(args.databases or FIND_SOURCES))
    key = f"find|{text.casefold().strip()}|{args.k}|{args.since}|{args.until}|{','.join(sources)}"
    key += _retry_suffix(args)
    if (code := _guard(session, key, args)) is not None:
        return code
    started = time.time()
    connectors = build_find_connectors(
        sources,
        semantic_scholar_api_key=credentials.get("SEMANTIC_SCHOLAR_API_KEY"),
        openalex_api_key=credentials.get("OPENALEX_API_KEY"),
        openaire_access_token=credentials.get("OPENAIRE_ACCESS_TOKEN"),
        openaire_refresh_token=credentials.get("OPENAIRE_REFRESH_TOKEN"),
    )
    deadline = max(5.0, min(20.0, session.remaining)) if not args.overtime else 20.0
    outcome = run_find(
        FindRequest(text=text, per_source=args.k, since=args.since, until=args.until),
        connectors,
        deadline_seconds=deadline,
    )
    for source, trace in outcome.tracebacks.items():
        emit_error(f"bug in the {source} path:\n{trace}")
    steps = list(outcome.result.coverage or [])
    papers: list[tuple[Paper, str, float, list[str]]] = []
    for rank, paper in enumerate(outcome.result.papers, start=1):
        tags = ",".join(f"{s}#{r}" for s, r in sorted(outcome.ranks.get(id(paper), {}).items()))
        papers.append((paper, f"find#{rank} ({tags})", _rank_weight(rank), []))
    record = session.absorb(
        kind="find",
        replay=_replay(args, "find"),
        key=key,
        args={"text": text, "since": str(args.since) if args.since else None},
        started=started,
        steps=steps,
        papers=papers,
    )
    return _emit_move(session, record, steps, args)


def _seed_ids(session: HuntSession, refs: list[str]) -> list[str]:
    ids: list[str] = []
    for ref in refs:
        if ref.startswith("c") and ref[1:].isdigit():
            request_id = s2_request_id_for_paper(session.resolve(ref).paper)
            if request_id is None:
                raise InvalidParameterError(f"{ref} carries no id Semantic Scholar accepts")
            ids.append(request_id)
        else:
            ids.append(s2_request_id(ref))
    return ids


def _related(session: HuntSession, args: argparse.Namespace) -> int:
    args.recs = _width(session, args.recs, 20, 50)
    args.neighbours = _width(session, args.neighbours, 40, 120)
    seeds = _seed_ids(session, args.seeds)
    negative = _seed_ids(session, args.negative)
    key = (
        f"related|{','.join(sorted(seeds))}|not:{','.join(sorted(negative))}|{args.recs}|"
        f"{args.no_refs}|{args.no_cites}|{args.neighbours}"
    )
    key += _retry_suffix(args)
    if (code := _guard(session, key, args)) is not None:
        return code
    started = time.time()
    connector = s2_connector()
    with request_deadline(started + MOVE_DEADLINES["related"]):
        outcome = run_related(
            connector,
            seeds,
            negative=negative,
            recommendations=max(0, args.recs),
            references=not args.no_refs,
            citations=not args.no_cites,
            max_neighbours=args.neighbours,
        )
    papers: list[tuple[Paper, str, float, list[str]]] = []
    for paper in outcome.papers:
        routes = outcome.routes.get(id(paper), [])
        weight = 0.0
        for route in routes:
            if route.startswith("rec#"):
                weight += _rank_weight(int(route[4:]), 0.7)
            else:
                weight += 0.5 * int(
                    route.split()[-2] if route.startswith("cited by") else route.split()[1]
                )
        papers.append((paper, "related: " + ", ".join(routes), weight, []))
    record = session.absorb(
        kind="related",
        replay=_replay(args, "related"),
        key=key,
        args={"seeds": " ".join(args.seeds), "not": " ".join(args.negative) or None},
        started=started,
        steps=outcome.steps,
        papers=papers,
    )
    return _emit_move(session, record, outcome.steps, args)


def _snowball(session: HuntSession, args: argparse.Namespace) -> int:
    forward = args.forward or not args.backward
    backward = args.backward or not args.forward
    keep = _width(session, args.max, 60, 200)
    seeds = _seed_ids(session, args.seeds)
    labels = dict(zip(seeds, args.seeds, strict=True))
    key = (
        f"snowball|{','.join(sorted(seeds))}|{forward}|{backward}|{args.since}|{args.until}|"
        f"{keep}|{args.edges}"
    )
    key += _retry_suffix(args)
    if (code := _guard(session, key, args)) is not None:
        return code
    started = time.time()
    kept = [c for c in session.candidates.values() if c.mark in ("core", "keep")]
    focus = " ".join([session.data["question"], *(c.paper.title or "" for c in kept)])
    with request_deadline(started + MOVE_DEADLINES["snowball"]):
        outcome, _ = run_snowball(
            s2_connector(),
            seeds,
            forward=forward,
            backward=backward,
            since=args.since,
            until=args.until,
            keep=keep,
            edge_cap=args.edges,
            focus=focus,
            labels=labels,
        )
    papers: list[tuple[Paper, str, float, list[str]]] = []
    for paper in outcome.papers:
        routes = outcome.routes.get(id(paper), [])
        weight = sum(1.3 if "(" in r else 1.0 for r in routes)
        papers.append(
            (paper, "snowball: " + ", ".join(routes), weight, outcome.evidence.get(id(paper), []))
        )
    record = session.absorb(
        kind="snowball",
        replay=_replay(args, "snowball"),
        key=key,
        args={
            "seeds": " ".join(args.seeds),
            "direction": "both" if forward and backward else "forward" if forward else "backward",
            "since": str(args.since) if args.since else None,
        },
        started=started,
        steps=outcome.steps,
        papers=papers,
    )
    crawled = _crawled_directions(session, args.seeds, outcome.steps, forward, backward)
    with session.locked():
        for ref, directions in crawled.items():
            done = session.data.setdefault("crawled", {}).setdefault(ref, [])
            done.extend(d for d in directions if d not in done)
    return _emit_move(session, record, outcome.steps, args)


def _crawled_directions(
    session: HuntSession,
    refs: list[str],
    steps: list[SourceCoverage],
    forward: bool,
    backward: bool,
) -> dict[str, list[str]]:
    """Directions each seed was actually crawled in.

    A direction counts only when its edge step for that seed answered; a
    refused or failed step leaves the seed in the snowball queue.  Steps are
    named ``snowball:<direction>:<seed title[:50]>``.  When no step's title
    matches a seed (titles can differ between sources), the direction counts
    only if none of that direction's steps failed.
    """

    def norm(text: str) -> str:
        return re.sub(r"\W+", "", text.casefold())

    answered: dict[str, set[str]] = {"forward": set(), "backward": set()}
    any_failed = {"forward": False, "backward": False}
    for step in steps:
        parts = step.source.split(":", 2)
        if len(parts) < 3 or parts[0] != "snowball" or parts[1] not in answered:
            continue
        if step.status is CoverageStatus.FAILED:
            any_failed[parts[1]] = True
        else:
            answered[parts[1]].add(norm(parts[2]))
    result: dict[str, list[str]] = {}
    for ref in refs:
        if ref not in session.candidates:
            continue
        title = norm((session.candidates[ref].paper.title or "")[:50])
        directions = []
        for direction, on in (("forward", forward), ("backward", backward)):
            if not on:
                continue
            labels = answered[direction]
            if (title and title in labels) or (labels and not any_failed[direction]):
                directions.append(direction)
        result[ref] = directions
    return result


_MATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/match"


def _handle_members(args: argparse.Namespace, refs: list[str]) -> list[tuple[str, list[Paper]]]:
    store = resolve_store(args)
    loaded = []
    for ref in refs:
        handle_id = store.resolve(ref)
        loaded.append((handle_id, list(store.get(handle_id).members)))
    return loaded


def _add(session: HuntSession, args: argparse.Namespace) -> int:
    handles = list(getattr(args, "handles", None) or [])
    if not args.refs and not handles:
        raise InvalidParameterError("hunt add needs ids, titles, or --from HANDLE")
    key = "add|" + "|".join(sorted(r.casefold().strip() for r in args.refs + handles))
    key += _retry_suffix(args)
    if (code := _guard(session, key, args)) is not None:
        return code
    started = time.time()
    steps: list[SourceCoverage] = []
    from_sets: list[tuple[Paper, str, float, list[str]]] = []
    for handle_id, members in _handle_members(args, handles):
        steps.append(
            SourceCoverage.complete(f"add:{handle_id}", len(members), reported_total=len(members))
        )
        from_sets += [(p, f"from {handle_id}", 1.0, []) for p in members]
    if not args.refs:
        record = session.absorb(
            kind="add",
            replay=_replay(args, "add"),
            key=key,
            args={"from": " ".join(handles)},
            started=started,
            steps=steps,
            papers=from_sets,
        )
        return _emit_move(session, record, steps, args)
    connector = s2_connector()
    ids: list[str] = []
    titles: list[str] = []
    for ref in args.refs:
        try:
            ids.append(s2_request_id(ref))
        except InvalidParameterError:
            titles.append(ref)
    found: list[Paper] = []
    if ids:
        by_id, step = _hydrate(connector, ids, "add:ids")
        steps.append(step)
        found += [by_id[i] for i in ids if i in by_id]
    from lit_search.connectors.semantic_scholar import _PAPER_FIELDS

    for title in titles:
        try:
            data = connector._get(_MATCH_URL, {"query": title, "fields": _PAPER_FIELDS}).json()
            item = (data.get("data") or [None])[0]
            paper = connector._parse_paper(item) if item else None
            if paper is None:
                steps.append(
                    SourceCoverage.partial(f"add:title:{title[:40]}", 0, detail="no match")
                )
            else:
                found.append(paper)
                steps.append(
                    SourceCoverage.complete(f"add:title:{title[:40]}", 1, reported_total=1)
                )
        except Exception as exc:
            steps.append(
                SourceCoverage.failed(
                    f"add:title:{title[:40]}", error=str(exc), error_type=type(exc).__name__
                )
            )
    record = session.absorb(
        kind="add",
        replay=_replay(args, "add"),
        key=key,
        args={"refs": " | ".join(args.refs), "from": " ".join(handles) or None},
        started=started,
        steps=steps,
        papers=[(p, "added by the agent", 1.0, []) for p in found] + from_sets,
    )
    return _emit_move(session, record, steps, args)


def _search(session: HuntSession, args: argparse.Namespace) -> int:
    from lit_search.engine import Engine

    args.max = _width(session, args.max, 50, 200)
    databases = list(dict.fromkeys(args.databases or HUNT_SEARCH_DEFAULT))
    key = (
        f"search|{args.query.strip()}|{args.max}|{args.since}|{args.until}|{','.join(databases)}"
        + _retry_suffix(args)
    )
    if (code := _guard(session, key, args)) is not None:
        return code
    started = time.time()
    result = Engine().search(
        args.query,
        databases=databases,
        max_papers_per_database=args.max,
        since=args.since,
        until=args.until,
        num_workers=len(databases),
        show_progress=False,
        enrichment_databases=[],
    )
    steps = list(result.coverage or [])
    papers: list[tuple[Paper, str, float, list[str]]] = [
        (paper, f"search#{rank}", _rank_weight(rank, 0.6), [])
        for rank, paper in enumerate(result.papers, start=1)
    ]
    record = session.absorb(
        kind="search",
        replay=_replay(args, "search"),
        key=key,
        args={"query": args.query, "since": str(args.since) if args.since else None},
        started=started,
        steps=steps,
        papers=papers,
    )
    return _emit_move(session, record, steps, args)


def _exclude_handles(session: HuntSession, args: argparse.Namespace, refs: list[str]) -> str:
    parts = []
    for handle_id, members in _handle_members(args, refs):
        added = session.exclude(members, handle_id)
        parts.append(f"{handle_id}: {added} of {len(members)} papers newly excluded")
    return "\n".join(parts)


def _exclude(session: HuntSession, args: argparse.Namespace) -> int:
    emit_text(_exclude_handles(session, args, args.handles) + "\n" + _footer(session))
    return EXIT_OK


def _snippets(session: HuntSession, args: argparse.Namespace) -> int:
    args.k = _width(session, args.k, 20, 60)
    text = " ".join(args.text)
    key = f"snippets|{text.casefold().strip()}|{args.k}|{args.field}"
    key += _retry_suffix(args)
    if (code := _guard(session, key, args)) is not None:
        return code
    started = time.time()
    connector = s2_connector()
    with request_deadline(started + MOVE_DEADLINES["snippets"]):
        outcome = run_snippets(connector, text, limit=args.k, field_of_study=args.field)
    papers = [
        (paper, f"snippet#{rank}", _rank_weight(rank, 0.8), outcome.evidence.get(id(paper), []))
        for rank, paper in enumerate(outcome.papers, start=1)
    ]
    record = session.absorb(
        kind="snippets",
        replay=_replay(args, "snippets"),
        key=key,
        args={"text": text, "field": args.field},
        started=started,
        steps=outcome.steps,
        papers=papers,
    )
    return _emit_move(session, record, outcome.steps, args)


def _show(session: HuntSession, args: argparse.Namespace) -> int:
    chosen = [session.resolve(c) for c in args.cids]
    if args.json:
        emit_json({"candidates": [c.to_dict() for c in chosen]})
        return EXIT_OK
    lines: list[str] = []
    for candidate in chosen:
        paper = candidate.paper
        lines += _candidate_line(candidate, abstract=False)
        if paper.abstract:
            lines += textwrap.wrap(
                paper.abstract, width=100, initial_indent="       ", subsequent_indent="       "
            )
        authors = ", ".join(a.name for a in (paper.authors or [])[:6])
        if authors:
            lines.append(
                f"       authors: {authors}{' et al.' if len(paper.authors or []) > 6 else ''}"
            )
        lines.append(
            f"       score {session.score(candidate):.2f} · routes: "
            + "; ".join(f"move {r['move']} {r['label']}" for r in candidate.routes)
        )
        for sentence in candidate.evidence[:3]:
            lines.append(
                f'       evidence: "{textwrap.shorten(sentence, width=300, placeholder=" ...")}"'
            )
        if candidate.note:
            lines.append(f"       note: {candidate.note}")
    emit_text("\n".join(lines))
    return EXIT_OK


def _mark(session: HuntSession, args: argparse.Namespace) -> int:
    value = None if args.mark == "clear" else args.mark
    if session.strict and value in ("core", "keep") and not args.note.strip():
        emit_text("strict: --core/--keep need --note with the reason, citing the abstract")
        return EXIT_INCOMPLETE
    chosen = session.mark(args.cids, value, args.note)
    emit_text(f"marked {' '.join(c.cid for c in chosen)} {args.mark}\n{_footer(session)}")
    return EXIT_OK


def _failed_line(move: dict[str, Any]) -> str:
    failed = HuntSession._failed_steps(move)
    detail = " ".join(f"{k}={v}" for k, v in move["args"].items() if v)
    sources = ", ".join(f"{s.source} ({_short(s.error or s.detail or '')[:60]})" for s in failed)
    return f"  move {move['number']} {move['kind']} {detail}\n      failed: {sources}"


def _rerun(session: HuntSession, move: dict[str, Any], args: argparse.Namespace) -> bool:
    replay = move["replay"]
    params = dict(replay["params"])
    for name in ("since", "until"):
        if params.get(name):
            params[name] = dt.date.fromisoformat(params[name])
    if replay["kind"] == "find":
        failed = {s.source for s in HuntSession._failed_steps(move)}
        params["databases"] = [d for d in (params.get("databases") or FIND_SOURCES) if d in failed]
    namespace = argparse.Namespace(
        **params,
        hunt=args.hunt,
        json=False,
        quiet=args.quiet,
        brief=True,
        overtime=True,
        retry_of=move["number"],
    )
    handler = {
        "find": _find,
        "related": _related,
        "snippets": _snippets,
        "snowball": _snowball,
        "add": _add,
        "search": _search,
    }[replay["kind"]]
    handler(session, namespace)
    with session.locked():
        latest = session.data["moves"][-1]
        latest["args"]["retry_of"] = move["number"]
        ok = not HuntSession._failed_steps(latest)
        if ok:
            for logged in session.data["moves"]:
                if logged["number"] == move["number"]:
                    logged["resolved_by"] = latest["number"]
    return ok


def _retry(session: HuntSession, args: argparse.Namespace) -> int:
    failed = session.failed_moves()
    if not args.moves and not args.all:
        lines = [f"{len(failed)} move(s) with failed sources in {session.data['id']}:"]
        lines += [_failed_line(m) for m in failed]
        if failed:
            lines.append(
                "re-run with `hunt retry ID N ...` or `--all`; add `--wait 15` to keep trying "
                "while the services cool down (run it in the background and carry on)"
            )
        emit_text("\n".join(lines))
        return EXIT_OK
    wanted = {m["number"] for m in failed} if args.all else set(args.moves)
    pending = [m for m in failed if m["number"] in wanted]
    unknown = wanted - {m["number"] for m in failed}
    if unknown:
        emit_text(f"not failed or not replayable: {' '.join(map(str, sorted(unknown)))}")
    deadline = time.time() + args.wait * 60
    while pending:
        still = [m for m in pending if not _rerun(session, m, args)]
        emit_text(
            f"retry round: {len(pending) - len(still)} fixed, {len(still)} still failing"
            + (f" ({' '.join(str(m['number']) for m in still)})" if still else "")
        )
        if not still or time.time() + 60 > deadline:
            pending = still
            break
        time.sleep(60)
        session = HuntSession.load(session.data["id"])
        pending = [m for m in session.failed_moves() if m["number"] in {p["number"] for p in still}]
    return EXIT_INCOMPLETE if pending else EXIT_OK


def _rank(session: HuntSession, args: argparse.Namespace) -> int:
    chosen = session.set_order(args.cids)
    emit_text(f"order set: {' '.join(c.cid for c in chosen)}\n{_footer(session)}")
    return EXIT_OK


_HEADINGS = {
    "intro": r"introduction",
    "related": r"related\s+work|background(?:\s+and\s+related\s+work)?|prior\s+work",
    "conclusion": r"conclusions?|discussion",
}
_ANY_HEADING = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?|[IVX]+\.)?\s*[A-Z][A-Za-z ,:&-]{2,60}\s*$", re.MULTILINE
)


def section(text: str, part: str) -> str | None:
    """Cut one numbered or titled section out of pdftotext output.

    Parameters
    ----------
    text : str
        Full text of the paper.
    part : str
        ``intro``, ``related``, ``conclusion`` or ``all``.

    Returns
    -------
    str | None
        The section from its heading to the next heading of the same form,
        or ``None`` when no heading matches.
    """
    if part == "all":
        return text
    heading = re.compile(
        rf"^\s*(?:\d+\.?|[IVX]+\.)?\s*(?:{_HEADINGS[part]})\s*$", re.IGNORECASE | re.MULTILINE
    )
    match = heading.search(text)
    if match is None:
        return None
    rest = text[match.end() :]
    numbered = re.match(r"\s*(\d+|[IVX]+)", match.group(0))
    if numbered:
        nxt = re.search(r"^\s*(?:\d+|[IVX]+)\.?\s+[A-Z][^\n]{2,60}$", rest, re.MULTILINE)
    else:
        nxt = _ANY_HEADING.search(rest, 200)
    return (match.group(0).strip() + "\n" + rest[: nxt.start() if nxt else len(rest)]).strip()


def _read(session: HuntSession, args: argparse.Namespace) -> int:
    import subprocess

    from lit_search.fetch import sci_hub
    from lit_search.fetch.fetcher import PdfFetcher
    from lit_search.fetch.targets import target_from_paper

    candidate = session.resolve(args.cid)
    pdfs = session.path.parent / "pdfs"
    pdfs.mkdir(exist_ok=True)
    fetcher = PdfFetcher(
        pdfs,
        scihub=sci_hub.scihub_enabled_from_env(),
        openalex_api_key=credentials.get("OPENALEX_API_KEY") or None,
        timeout=30.0,
    )
    outcome = fetcher.fetch(target_from_paper(candidate.paper))
    if not outcome.has_pdf or not outcome.path:
        emit_text(
            f"no PDF for {args.cid} ({outcome.status.value}"
            f"{': ' + outcome.error if outcome.error else ''}); judge it from `hunt show`."
        )
        return EXIT_INCOMPLETE
    try:
        text = subprocess.run(
            ["pdftotext", outcome.path, "-"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        emit_text(f"could not extract text from {outcome.path}: {exc}")
        return EXIT_ERROR
    body = section(text, args.part)
    note = ""
    if body is None:
        body = text
        note = f"(no '{args.part}' heading found; showing the start of the paper)\n"
    with session.locked():
        read = session.data.setdefault("read", [])
        if candidate.cid not in read:
            read.append(candidate.cid)
    clipped = body[: args.chars]
    more = (
        f"\n... {len(body) - len(clipped)} more characters (--chars)"
        if len(body) > len(clipped)
        else ""
    )
    emit_text(
        f"{candidate.cid} {candidate.paper.title}\n{note}{clipped}{more}\npdf: {outcome.path}\n"
        + _footer(session)
    )
    return EXIT_OK


def _note(session: HuntSession, args: argparse.Namespace) -> int:
    session.add_note(" ".join(args.text), args.kind or "note")
    emit_text(
        f"noted ({args.kind or 'note'}) after move {len(session.data['moves'])}\n{_footer(session)}"
    )
    return EXIT_OK


def _status(session: HuntSession, args: argparse.Namespace) -> int:
    ranked = session.ranked()
    if args.unjudged:
        ranked = [c for c in ranked if c.mark is None]
    first = getattr(args, "start_at", 1)
    page = ranked[first - 1 : first - 1 + args.top]
    if args.json:
        emit_json(
            {
                "hunt": session.data["id"],
                "time_left_seconds": round(session.remaining),
                "moves": session.data["moves"],
                "top": [{**c.to_dict(), "score": session.score(c)} for c in page],
                "from": first,
                "of": len(ranked),
            }
        )
        return EXIT_OK
    lines = [
        f"hunt {session.data['id']} · {session.data['profile']} · {session.data['question']}",
        *(
            f"  move {m['number']} {m['kind']} "
            + " ".join(f"{k}={v}" for k, v in m["args"].items() if v)
            + f" · +{len(m['new'])} new"
            for m in session.data["moves"]
        ),
        *session.shortfalls(),
        f"{'unjudged ' if args.unjudged else ''}candidates {first}-{first + len(page) - 1} of "
        f"{len(ranked)} by mark then score:",
    ]
    for candidate in page:
        lines += _candidate_line(candidate, abstract=args.unjudged)
    if first + len(page) <= len(ranked):
        lines.append(
            f"  next page: hunt status {session.data['id']} "
            f"{'--unjudged ' if args.unjudged else ''}--from {first + len(page)}"
        )
    lines.append(_footer(session))
    emit_text("\n".join(lines))
    return EXIT_OK


def _report(session: HuntSession, args: argparse.Namespace) -> int:
    if session.needs_more_time():
        emit_text(
            f"the report opens at {session.min_share:.0%} of the budget; {session.budget_used:.0%} used, {format_seconds(session.remaining)} "
            "left. Spend it: snowball the papers listed as not yet snowballed, search a "
            "vocabulary you have not tried, or go deeper on your best candidates (`hunt show`, "
            "`hunt read`) and set the final order with `hunt rank`."
        )
        return EXIT_INCOMPLETE
    report_blockers = session.report_blockers()
    if report_blockers:
        emit_text(
            "strict: report refused until you\n" + "\n".join(f"  - {b}" for b in report_blockers)
        )
        return EXIT_INCOMPLETE
    size = args.shortlist or int(session.data["shortlist"])
    ranked = session.ranked()
    kept = [c for c in ranked if c.mark in ("core", "keep")]
    shortlist = (
        kept[:size]
        + [c for c in ranked if c.mark not in ("core", "keep")][: max(0, size - len(kept))]
    )
    overflow = kept[size:]
    moves = session.data["moves"]
    find_sources = {s["source"] for m in moves if m["kind"] == "find" for s in m["steps"]}
    not_searched = [s for s in FIND_SOURCES if s not in find_sources]
    lines = [
        f"# Hunt: {session.data['question']}",
        "",
        f"{session.data['id']} · profile {session.data['profile']} · "
        f"{format_seconds(session.elapsed)} of {session.data['minutes']:g} min · {len(moves)} moves · "
        f"{len(session.candidates)} candidates ({len(kept)} core or kept)",
        "",
        f"## Shortlist ({len(shortlist)})",
        "",
    ]
    for index, candidate in enumerate(shortlist, start=1):
        paper = candidate.paper
        date = paper.publication_date.strftime("%Y") if paper.publication_date else "n.d."
        cited = f", cited {paper.citations}" if paper.citations is not None else ""
        tag = {"core": "CORE", "keep": "kept"}.get(candidate.mark or "", "by score, not judged")
        if candidate.cid in (session.data.get("read") or []):
            tag += ", read"
        lines.append(
            f"{index}. **{paper.title}** ({date}, {_paper_id(paper)}{cited}) [{tag}] `{candidate.cid}`"
        )
        if candidate.note:
            lines.append(f"   - why: {candidate.note}")
        lines.append(
            "   - found by: "
            + "; ".join(f"move {r['move']} {r['label']}" for r in candidate.routes[:4])
            + (" ..." if len(candidate.routes) > 4 else "")
        )
        if candidate.evidence:
            sentence = textwrap.shorten(candidate.evidence[0], width=260, placeholder=" ...")
            lines.append(f'   - evidence: "{sentence}"')
    if overflow:
        lines.append(
            f"\n{len(overflow)} more kept beyond the shortlist: "
            + ", ".join(f"{c.cid} {textwrap.shorten(c.paper.title or '', 60)}" for c in overflow)
        )
    lines += ["", "## What was searched", ""]
    for move in moves:
        detail = " ".join(f"{k}={v}" for k, v in move["args"].items() if v)
        lines.append(f"- move {move['number']} {move['kind']} {detail} → +{len(move['new'])} new")
    log = session.reasoning_log()
    if log:
        lines += ["", "## Reasoning log", "", *log]
    problems = session.shortfalls()
    lines += ["", "## Shortfalls", ""]
    lines += [f"- {p}" for p in problems] or ["- none"]
    lines += ["", "## Not searched", ""]
    gaps = []
    if not_searched:
        gaps.append(f"- relevance sources never queried: {', '.join(not_searched)}")
    if not any(m["kind"] == "snippets" for m in moves):
        gaps.append("- no full-text (snippet) search")
    if not any(m["kind"] == "snowball" for m in moves):
        gaps.append("- no snowballing (citation crawl)")
    queue = session.snowball_queue()
    for direction, cids in queue.items():
        if cids:
            gaps.append(f"- core/kept papers never snowballed {direction}: {' '.join(cids)}")
    unresolved = session.failed_moves()
    if unresolved:
        gaps.append(
            f"- {len(unresolved)} move(s) lost sources to failures or rate limits "
            f"({' '.join(str(m['number']) for m in unresolved)}); `hunt retry` re-runs them"
        )
    gaps.append(
        "- always outside this tool's reach: PubMed, OpenReview, DBLP, publisher-only indexes "
        "(available via `lit-search search -d ...`)"
    )
    lines += gaps
    text = "\n".join(lines) + "\n"
    report_path = session.path.parent / "report.md"
    report_path.write_text(text, encoding="utf-8")
    with session.locked():
        session.data["report"] = {
            "at": time.time(),
            "elapsed_seconds": round(session.elapsed),
            "shortlist": [c.cid for c in shortlist],
        }

    def handle_for(candidates: list[Candidate], label: str) -> str:
        result = SearchResult(
            query=session.data["question"],
            papers=[c.paper for c in candidates],
            databases=sorted(find_sources) or ["semantic_scholar"],
            enrichment_databases=[],
            coverage=CoverageLedger.from_sources(
                [
                    SourceCoverage.from_dict(
                        {**step, "source": f"move{m['number']}:{step['source']}"}
                    )
                    for m in moves
                    for step in m["steps"]
                ]
                or [SourceCoverage.skipped_by_policy("hunt", detail="no moves ran")]
            ),
        )
        return resolve_store(args).put(from_search_result(result, label=label))

    short_handle = handle_for(shortlist, f"{session.data['id']} shortlist")
    all_handle = handle_for(session.ranked(), f"{session.data['id']} all candidates")
    if args.json:
        emit_json(
            {
                "hunt": session.data["id"],
                "report": str(report_path),
                "shortlist_handle": short_handle,
                "all_handle": all_handle,
                "shortlist": [c.to_dict() for c in shortlist],
                "shortfalls": problems,
            }
        )
    else:
        emit_text(
            text
            + f"\nreport: {report_path}\nshortlist handle: {short_handle}\nall candidates handle: {all_handle}"
        )
    return EXIT_OK


def _list(args: argparse.Namespace) -> int:
    root = hunts_root()
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("hunt-*/session.json")) if root.exists() else []:
        session = HuntSession.load(path.parent.name, root)
        rows.append(
            {
                "hunt": session.data["id"],
                "profile": session.data["profile"],
                "question": session.data["question"],
                "moves": len(session.data["moves"]),
                "candidates": len(session.candidates),
            }
        )
    if args.json:
        emit_json({"hunts": rows})
    else:
        emit_text(
            "\n".join(
                f"{r['hunt']}  {r['profile']:<10} {r['moves']:>2} moves {r['candidates']:>4} cand.  {r['question'][:70]}"
                for r in rows
            )
            or f"no hunts under {root}"
        )
    return EXIT_OK


def cmd_hunt(args: argparse.Namespace) -> int:
    """Dispatch a ``hunt`` action.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code: 0 normally, 1 when a move produced nothing because every
        step failed, 3 when a move was refused because the budget is spent.
    """
    if args.hunt_action == "start":
        return _start(args)
    if args.hunt_action == "list":
        return _list(args)
    session = HuntSession.load(args.hunt)
    if args.hunt_action == "extend":
        session.extend(args.minutes)
        emit_text(_footer(session))
        return EXIT_OK
    handlers = {
        "find": _find,
        "related": _related,
        "snowball": _snowball,
        "add": _add,
        "search": _search,
        "exclude": _exclude,
        "rank": _rank,
        "retry": _retry,
        "read": _read,
        "snippets": _snippets,
        "show": _show,
        "mark": _mark,
        "note": _note,
        "status": _status,
        "report": _report,
    }
    return handlers[args.hunt_action](session, args)


__all__ = ["add_hunt_parser", "cmd_hunt"]
