"""Shared emission for every command that produces a set of papers.

Extracted so that the handle, set-algebra and index commands emit sets through
exactly the same path as ``search``, ``get`` and ``snowball``.  The rule being
shared is the one that matters: **a set is never printed without its
coverage, and the exit code always reports that coverage.**  Two code paths for
emitting a set is how one of them ends up without the banner.
"""

from __future__ import annotations

import argparse
from typing import Any

from lit_search.cli.output import emit_json, emit_text, exit_code_for, json_envelope
from lit_search.core.rendering import CoverageView, coverage_view, render_result
from lit_search.utils.persistence import Persistable, save_to_json


def write_json_output(result: Persistable, args: argparse.Namespace) -> str | None:
    """Write the full result to ``--output`` when one was given.

    Parameters
    ----------
    result : object
        Result to persist.
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    str | None
        The path written, or ``None`` when no output was requested.
    """
    path = getattr(args, "output", None)
    if not path:
        return None
    save_to_json(result, path)
    return str(path)


def set_payload(
    command: str, result: object, view: CoverageView, written: str | None, **extra: Any
) -> dict[str, Any]:
    """Build the JSON payload for a produced set.

    Parameters
    ----------
    command : str
        Subcommand name.
    result : object
        Result container.
    view : CoverageView
        Coverage of the set.
    written : str | None
        Path the full result was written to, if any.
    **extra : Any
        Command-specific fields.

    Returns
    -------
    dict[str, Any]
        JSON-ready payload.
    """
    papers = list(getattr(result, "papers", []))
    return json_envelope(
        command,
        view,
        paper_count=len(papers),
        papers=[paper.to_dict() for paper in papers],
        output=written,
        **extra,
    )


def emit_set(
    command: str,
    result: Persistable,
    headline: str,
    args: argparse.Namespace,
    *,
    extra_lines: list[str] | None = None,
    **extra: Any,
) -> int:
    """Render a produced set and return the exit code its coverage implies.

    Parameters
    ----------
    command : str
        Subcommand name, used in the JSON envelope.
    result : object
        Result container exposing ``papers`` and optionally ``coverage``.
    headline : str
        One-line description of the operation.
    args : argparse.Namespace
        Parsed arguments.
    extra_lines : list[str] | None
        Extra lines appended to the text rendering, for facts a set alone
        cannot carry -- a new handle identifier, a lane that did not run, a
        deduplication stage that was skipped.
    **extra : Any
        Extra fields for the JSON payload.

    Returns
    -------
    int
        Exit code from :func:`~lit_search.cli.output.exit_code_for`.
    """
    view = coverage_view(result)
    written = write_json_output(result, args)
    if not args.quiet:
        if args.json:
            emit_json(set_payload(command, result, view, written, **extra))
        else:
            emit_text(render_result(result, headline=headline, limit=args.limit))
            for line in extra_lines or []:
                emit_text(line)
            if written:
                emit_text(f"Written: {written}")
    return exit_code_for(view)


__all__ = ["emit_set", "set_payload", "write_json_output"]
