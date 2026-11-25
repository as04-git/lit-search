"""Exit codes and emission for the command line.

Two rules are enforced here rather than left to each command.

**Execution decides the exit code.**  Deliberately bounded source scope and a
caller-supplied cap remain visible in coverage, but do not impersonate an
operational failure.  An unplanned or unknown shortfall exits
:data:`EXIT_INCOMPLETE`.

**Machine-readable output always carries coverage.**  :func:`json_envelope`
puts ``coverage`` in every payload, present even when there is nothing to
report, because an omitted key reads as "fine".
"""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from lit_search.core.rendering import CoverageView

#: The operation finished and its set is completely covered.
EXIT_OK = 0
#: The operation could not run: bad input, missing credential, upstream error.
EXIT_ERROR = 1
#: Reserved for argparse's own usage errors.
EXIT_USAGE = 2
#: The operation ran, and its set is **not** completely covered.
EXIT_INCOMPLETE = 3
#: The subcommand exists but this build does not implement it.
EXIT_NOT_IMPLEMENTED = 4

EXIT_CODE_HELP = """exit codes:
  0  the declared operation completed; bounded scope remains visible
  1  the operation could not run (bad input, missing credential, upstream error)
  2  usage error
  3  the declared operation had an unplanned or unknown shortfall
  4  the subcommand is not implemented in this build

Commands or legacy paths with no execution evidence exit 3. Use the semantic
coverage and scope fields to decide whether a successfully executed but
bounded result is suitable for a particular downstream claim.
"""


def exit_code_for(view: CoverageView) -> int:
    """Return the exit code implied by a coverage view.

    Parameters
    ----------
    view : CoverageView
        Coverage of the set that was produced.

    Returns
    -------
    int
        :data:`EXIT_OK` when complete, :data:`EXIT_INCOMPLETE` otherwise --
        including when no ledger exists, because "we cannot say" is not
        "everything is fine".
    """
    return EXIT_OK if view.execution_is_complete else EXIT_INCOMPLETE


def json_envelope(command: str, view: CoverageView, **fields: Any) -> dict[str, Any]:
    """Build the machine-readable payload for a command.

    ``command``, ``coverage`` and ``complete`` are always present and always
    first, so a consumer that reads only the head of the document still sees
    the verdict.

    Parameters
    ----------
    command : str
        Subcommand name.
    view : CoverageView
        Coverage of the produced set.
    **fields : Any
        Command-specific payload.

    Returns
    -------
    dict[str, Any]
        JSON-ready payload.
    """
    payload: dict[str, Any] = {
        "command": command,
        "complete": view.is_complete,
        "execution_complete": view.execution_is_complete,
        "exit_basis": "execution_complete",
        "coverage": view.to_dict(),
    }
    payload.update(fields)
    return payload


def emit_text(text: str, *, stream: TextIO | None = None) -> None:
    """Write a rendered report to stdout.

    Parameters
    ----------
    text : str
        Rendered report.
    stream : TextIO | None
        Destination.  Defaults to ``sys.stdout``.

    Returns
    -------
    None
    """
    print(text, file=stream if stream is not None else sys.stdout)


def emit_json(payload: dict[str, Any], *, stream: TextIO | None = None) -> None:
    """Write a payload as a single JSON document on stdout.

    Parameters
    ----------
    payload : dict[str, Any]
        Payload, normally from :func:`json_envelope`.
    stream : TextIO | None
        Destination.  Defaults to ``sys.stdout``.

    Returns
    -------
    None
    """
    print(
        json.dumps(payload, ensure_ascii=False, indent=2),
        file=stream if stream is not None else sys.stdout,
    )


def emit_error(message: str, *, stream: TextIO | None = None) -> None:
    """Write an error to stderr, marked so it cannot read as a result.

    Parameters
    ----------
    message : str
        What went wrong.
    stream : TextIO | None
        Destination.  Defaults to ``sys.stderr``.

    Returns
    -------
    None
    """
    print(f"lit-search: error: {message}", file=stream if stream is not None else sys.stderr)
