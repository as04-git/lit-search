"""Entry point for the ``lit-search`` command line.

Dispatch only: parse, configure logging, hand off to a handler in
:mod:`lit_search.cli.commands`, and turn a known failure into a message on
stderr with a non-zero exit.  An *unknown* exception is deliberately not
caught -- a traceback is loud, and this tool's governing rule is that
failures must be loud rather than tidy.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence

from lit_search.cli.commands import HANDLERS
from lit_search.cli.fetch import cmd_fetch
from lit_search.cli.output import EXIT_ERROR, EXIT_OK, EXIT_USAGE, emit_error, emit_text
from lit_search.cli.parser import build_parser
from lit_search.cli.setup import cmd_config, cmd_doctor
from lit_search.exceptions import LitSearchError
from lit_search.utils.logging_config import configure_verbose_logging
from lit_search.utils.version import package_version

#: Conventional shell exit code for an interrupted process.
EXIT_INTERRUPTED = 130


def _configure_logging(verbose: bool) -> None:
    """Set up logging so that coverage warnings reach stderr.

    Parameters
    ----------
    verbose : bool
        Whether the user asked for DEBUG output.

    Returns
    -------
    None
    """
    if verbose:
        configure_verbose_logging()
        return
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return its exit code.

    Parameters
    ----------
    argv : Sequence[str] | None
        Arguments excluding the program name.  ``None`` reads
        ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code; see :data:`lit_search.cli.output.EXIT_CODE_HELP`.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        emit_text(f"lit-search {package_version()}")
        return EXIT_OK
    if args.command is None:
        parser.print_help()
        return EXIT_USAGE

    _configure_logging(args.verbose)
    try:
        if args.command == "fetch":
            return cmd_fetch(args)
        if args.command == "config":
            return cmd_config(args)
        if args.command == "doctor":
            return cmd_doctor(args)
        return HANDLERS[args.command](args)
    except LitSearchError as error:
        emit_error(f"{type(error).__name__}: {error}")
        return EXIT_ERROR
    except OSError as error:
        emit_error(f"{type(error).__name__}: {error}")
        return EXIT_ERROR
    except KeyboardInterrupt:
        emit_error("interrupted; the result is incomplete and was not written")
        return EXIT_INTERRUPTED


if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(main())
