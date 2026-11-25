"""The ``lit-search`` command line.

Five subcommands -- ``search``, ``get``, ``snowball``, ``rerank`` and
``export`` -- around one invariant: **a set of papers is never displayed
without the record of what was and was not consulted.**

The invariant is structural rather than a habit.  Every text rendering goes
through :func:`lit_search.core.rendering.render_result`, which cannot be
called without producing a coverage banner; every machine-readable rendering
goes through :func:`lit_search.cli.output.json_envelope`, whose ``coverage``
key is always present; and the exit code is derived from coverage, so a shell
script cannot treat a partial answer as a finished one by accident.

Search and native snowball traversal carry real ledgers. ``get`` still reports
UNKNOWN because its legacy multi-source lookup does not yet expose per-source
outcomes. A legacy DOI-only snowball path does the same rather than inventing
coverage.
"""

from lit_search.cli.main import main

__all__ = ["main"]
