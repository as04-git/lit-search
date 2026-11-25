"""``lit-search fetch``: download PDFs, showing every place that was tried.

Kept in its own module so the parser and dispatcher carry one line each.

Exit codes: ``0`` when the command ran, whatever it found (misses are shown,
not fatal); ``3`` with ``--strict`` when any paper ended without a PDF; ``1``
when it could not run (unknown handle, bad mirror configuration, unwritable
output directory); ``2`` for usage errors, including an unrecognised
identifier.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from lit_search import credentials
from lit_search.cli.context import resolve_store
from lit_search.cli.output import (
    EXIT_ERROR,
    EXIT_INCOMPLETE,
    EXIT_OK,
    EXIT_USAGE,
    emit_error,
    emit_json,
    emit_text,
)
from lit_search.fetch import sci_hub
from lit_search.fetch.fetcher import PdfFetcher
from lit_search.fetch.model import AttemptResult, FetchOutcome, FetchStatus, FetchTarget
from lit_search.fetch.targets import parse_identifier, target_from_paper
from lit_search.store.store import STORE_ROOT_ENV_VAR

#: Default number of handle papers fetched; ``--limit 0`` fetches all.
DEFAULT_HANDLE_LIMIT = 25

_TAGS = {
    FetchStatus.DOWNLOADED: "ok  ",
    FetchStatus.ALREADY_PRESENT: "have",
    FetchStatus.NOT_FOUND: "miss",
    FetchStatus.FAILED: "FAIL",
    FetchStatus.UNRESOLVED: "unkn",
}


def default_pdf_directory() -> Path:
    """Return ``$LIT_SEARCH_HOME/pdfs``, else ``~/.local/share/lit-search/pdfs``.

    Returns
    -------
    Path
        Default output directory (not created by this call).
    """
    home = os.environ.get(STORE_ROOT_ENV_VAR)
    if home:
        return Path(home).expanduser() / "pdfs"
    return Path.home() / ".local" / "share" / "lit-search" / "pdfs"


def _fetch_limit(raw: str) -> int | None:
    """Parse ``--limit`` for fetch, where ``0`` means every paper."""
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected an integer, got {raw!r}") from error
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected 0 or more, got {value}")
    return None if value == 0 else value


def add_fetch_parser(subparsers: argparse._SubParsersAction, parents: list) -> None:
    """Register the ``fetch`` subcommand.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparser registry.
    parents : list
        Shared parent parsers (common options and store options).

    Returns
    -------
    None
    """
    parser = subparsers.add_parser(
        "fetch",
        parents=parents,
        help="download PDFs, recording every place that was tried",
        description=(
            "Download PDFs for DOIs, arXiv ids, or a stored handle's papers. Each "
            "paper walks an ordered chain -- arxiv, the known pdf_url, OpenAlex "
            "open-access locations, the publisher landing page, then Sci-Hub if "
            "enabled -- and stops at the first body whose bytes start with %PDF. "
            "Every attempt is recorded; one line per paper says what happened."
        ),
    )
    parser.add_argument(
        "identifiers",
        nargs="*",
        metavar="ID",
        help="DOI, arXiv id, 10.48550/arXiv.* DOI, or doi.org / arxiv.org URL",
    )
    parser.add_argument(
        "--handle", metavar="H", help="also fetch the papers of this stored handle (or prefix)"
    )
    parser.add_argument(
        "--limit",
        type=_fetch_limit,
        default=DEFAULT_HANDLE_LIMIT,
        metavar="N",
        help=(
            f"fetch at most N of the handle's papers, in stored order; 0 fetches all "
            f"(default: {DEFAULT_HANDLE_LIMIT}). Explicit IDs are never limited"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="DIR",
        help="directory for PDFs (default: $LIT_SEARCH_HOME/pdfs, else ~/.local/share/lit-search/pdfs)",
    )
    parser.add_argument(
        "--scihub",
        action="store_true",
        help=(
            f"enable the Sci-Hub step (also ${sci_hub.SCIHUB_ENV_VAR}=1); mirrors from "
            f"${sci_hub.SCIHUB_MIRRORS_ENV_VAR}, comma-separated"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        metavar="S",
        help="per-request timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 3 if any paper ends without a PDF",
    )


def _human_size(size: int | None) -> str:
    """Format a byte count compactly."""
    if size is None:
        return "?"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"  # pragma: no cover - loop always returns


def _attempt_label(step: str, url: str | None) -> str:
    """Label an attempt by step, adding the host for Sci-Hub mirrors."""
    if step == "scihub" and url:
        return f"scihub[{urlsplit(url).netloc}]"
    return step


def render_outcome(outcome: FetchOutcome) -> str:
    """Render one outcome as a single line.

    Parameters
    ----------
    outcome : FetchOutcome
        Outcome to render.

    Returns
    -------
    str
        E.g. ``ok   10.1/x  via arxiv  -> /p.pdf (1.2 MB)`` or
        ``miss 10.1/y  tried: openalex(no OA location), scihub(disabled)``.
    """
    tag = _TAGS[outcome.status]
    label = outcome.target.label
    if outcome.status is FetchStatus.DOWNLOADED:
        line = f"{tag} {label}  via {outcome.via}  -> {outcome.path} ({_human_size(outcome.size)})"
        problems = [
            f"{_attempt_label(a.step, a.url)} {a.result.value}"
            for a in outcome.attempts
            if a.result in (AttemptResult.ERROR, AttemptResult.BLOCKED)
        ]
        if problems:
            line += f"  (after: {', '.join(problems)})"
        return line
    if outcome.status is FetchStatus.ALREADY_PRESENT:
        return f"{tag} {label}  already present -> {outcome.path} ({_human_size(outcome.size)})"
    # Consecutive attempts with the same label share one parenthesis, so a
    # followed landing page reads "publisher(landing page -> ...; HTTP 403)".
    groups: list[tuple[str, list[str]]] = []
    for attempt in outcome.attempts:
        if attempt.result is AttemptResult.SKIPPED and attempt.reason != "disabled":
            continue
        label_ = _attempt_label(attempt.step, attempt.url)
        if groups and groups[-1][0] == label_:
            groups[-1][1].append(attempt.reason)
        else:
            groups.append((label_, [attempt.reason]))
    parts = [f"{name}({'; '.join(reasons)})" for name, reasons in groups]
    tried = ", ".join(parts) if parts else "nothing to try"
    line = f"{tag} {label}  tried: {tried}"
    if outcome.error:
        line += f"  error: {outcome.error}"
    return line


def _summary(outcomes: list[FetchOutcome], output_dir: Path, note: str | None) -> str:
    """One-line summary of a run."""
    counts = dict.fromkeys(FetchStatus, 0)
    for outcome in outcomes:
        counts[outcome.status] += 1
    line = (
        f"fetch: {len(outcomes)} paper(s): {counts[FetchStatus.DOWNLOADED]} downloaded, "
        f"{counts[FetchStatus.ALREADY_PRESENT]} already present, "
        f"{counts[FetchStatus.NOT_FOUND]} not found, "
        f"{counts[FetchStatus.UNRESOLVED]} unresolved (a source refused), "
        f"{counts[FetchStatus.FAILED]} failed"
        f"  -> {output_dir}"
    )
    if note:
        line += f"  ({note})"
    return line


def _collect_targets(args: argparse.Namespace) -> tuple[list[FetchTarget], str | None]:
    """Build the target list from IDs and ``--handle``; return it with a truncation note.

    Raises
    ------
    ValueError
        If an identifier is unrecognised (a usage error).
    """
    targets = [parse_identifier(raw) for raw in args.identifiers]
    note: str | None = None
    if args.handle:
        store = resolve_store(args)
        handle_id = store.resolve(args.handle)
        members = list(store.get(handle_id).members)
        chosen = members if args.limit is None else members[: args.limit]
        if len(chosen) < len(members):
            note = (
                f"handle {handle_id[:12]}: fetched first {len(chosen)} of {len(members)} "
                f"papers; --limit 0 for all"
            )
        targets.extend(target_from_paper(paper) for paper in chosen)
    seen: set[str] = set()
    unique: list[FetchTarget] = []
    for target in targets:
        key = target.filename()
        if key not in seen:
            seen.add(key)
            unique.append(target)
    return unique, note


def cmd_fetch(args: argparse.Namespace) -> int:
    """Run ``lit-search fetch``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code; see the module docstring.
    """
    if not args.identifiers and not args.handle:
        emit_error("fetch needs at least one ID or --handle")
        return EXIT_USAGE
    try:
        targets, note = _collect_targets(args)
    except ValueError as error:
        emit_error(str(error))
        return EXIT_USAGE
    try:
        mirrors = sci_hub.mirrors_from_env()
    except ValueError as error:
        emit_error(str(error))
        return EXIT_ERROR

    output_dir = Path(args.output).expanduser() if args.output else default_pdf_directory()
    output_dir.mkdir(parents=True, exist_ok=True)
    scihub_enabled = bool(args.scihub) or sci_hub.scihub_enabled_from_env()
    fetcher = PdfFetcher(
        output_dir,
        scihub=scihub_enabled,
        scihub_mirrors=mirrors,
        openalex_api_key=credentials.get("OPENALEX_API_KEY") or None,
        timeout=args.timeout,
    )
    streaming = not args.json and not args.quiet
    try:
        outcomes = fetcher.fetch_all(
            targets,
            on_outcome=(lambda outcome: emit_text(render_outcome(outcome))) if streaming else None,
        )
    finally:
        fetcher.close()

    if args.json:
        payload: dict[str, Any] = {
            "command": "fetch",
            "output_dir": str(output_dir),
            "scihub_enabled": scihub_enabled,
            "scihub_mirrors": list(mirrors) if scihub_enabled else [],
            "summary": {status.value: 0 for status in FetchStatus},
            "note": note,
            "results": [outcome.to_dict() for outcome in outcomes],
        }
        for outcome in outcomes:
            payload["summary"][outcome.status.value] += 1
        emit_json(payload)
    elif not args.quiet:
        emit_text(_summary(outcomes, output_dir, note))

    if args.strict and not all(outcome.has_pdf for outcome in outcomes):
        return EXIT_INCOMPLETE
    return EXIT_OK


__all__ = ["add_fetch_parser", "cmd_fetch", "default_pdf_directory", "render_outcome"]
