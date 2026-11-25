"""Argument parser for the ``lit-search`` command line.

The surface is eight subcommands around one idea: every command that produces
a set of papers renders that set's coverage, and reports it in its exit code.

``search`` ``get`` ``snowball``
    Produce sets from outside the store.
``handle``
    ``list`` / ``show`` / ``lineage`` / ``import``.  Handles are content
    addresses, so without discovery the rest of the surface is unusable.
``set``
    ``union`` / ``difference`` / ``intersect`` / ``dedup``.  This is ``O4``
    answered as subcommands: each one writes a new handle whose coverage was
    propagated from its operands, and a subcommand can print that propagation
    where a flag on ``search`` could not.
``index``
    ``add`` / ``status`` / ``optimize`` / ``search`` over the embedded vector
    and full-text index.
``rerank``
    Orders a stored set against a question or a seed paper.
``export``
    Writes a file and prints its path.

Conventions, applied uniformly so a user learns them once:

``-d/--database``
    Repeatable.  Naming a database restricts the run to it; naming none uses
    every database whose credentials are present.  A restriction is itself a
    coverage fact and appears in the ledger.
``-o/--output``
    A path.  Commands write files and print paths; they never print file
    contents (``D9``).
``--limit``
    How many papers to *display*.  ``0`` displays all of them.  Display
    truncation is stated in the output and is never confused with a coverage
    shortfall.
``--json``
    One JSON document on stdout instead of the text rendering.  Its
    ``coverage`` key is always present.
``--store-root`` / ``--index-root``
    Where handles and the vector index live.  Present on every subcommand that
    touches either, so a test or a second project can be isolated without
    environment variables.
``--model`` / ``--deterministic-embedder``
    Which embedding model to use.  There is no automatic fallback between
    them: a run that quietly ranked by a non-semantic stand-in would look
    exactly like one that ranked by meaning.
"""

from __future__ import annotations

import argparse
import datetime as dt

from lit_search.cli.eval import add_eval_parser
from lit_search.cli.explore import add_explore_parsers
from lit_search.cli.fetch import add_fetch_parser
from lit_search.cli.find import add_find_parser
from lit_search.cli.hunt import add_hunt_parser
from lit_search.cli.output import EXIT_CODE_HELP
from lit_search.cli.setup import add_setup_parsers
from lit_search.connectors import SEARCH_REGISTRY
from lit_search.index.mmr import DEFAULT_LAMBDA
from lit_search.store.dedup import DEFAULT_SEMANTIC_THRESHOLD

PROGRAM_NAME = "lit-search"

SEARCH_DATABASES = sorted(str(database) for database in SEARCH_REGISTRY)

_DESCRIPTION = """\
Literature hunts an agent can drive and a person can audit. Every move and
command reports what it did not reach: a source that failed, refused or was
capped never looks like an empty result.

  the workspace     hunt        a time-budgeted session: moves add candidates,
                                marks record judgments, the report writes handles
  quick lookups     find, related, snippets   the same moves without a session
  papers            fetch       PDFs, recording every place tried
  building blocks   search, get, snowball, handle, set, index, rerank, export
                                for programs and audits; they exchange sets as
                                handles, which hunts read (hunt add --from,
                                hunt start --exclude) and write (hunt report)
  setup             config, doctor, eval
"""

_EPILOG = f"""{EXIT_CODE_HELP}
searchable databases:
  {", ".join(SEARCH_DATABASES)}
"""


def iso_date(raw: str) -> dt.date:
    """Parse a ``YYYY-MM-DD`` command-line date.

    Parameters
    ----------
    raw : str
        Text supplied on the command line.

    Returns
    -------
    datetime.date
        Parsed date.

    Raises
    ------
    argparse.ArgumentTypeError
        If *raw* is not an ISO date.
    """
    try:
        return dt.date.fromisoformat(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected a YYYY-MM-DD date, got {raw!r}") from error


def positive_int(raw: str) -> int:
    """Parse a strictly positive integer command-line value.

    Parameters
    ----------
    raw : str
        Text supplied on the command line.

    Returns
    -------
    int
        Parsed value.

    Raises
    ------
    argparse.ArgumentTypeError
        If *raw* is not an integer greater than zero.
    """
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected an integer, got {raw!r}") from error
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a value of 1 or more, got {value}")
    return value


def display_limit(raw: str) -> int | None:
    """Parse ``--limit``, where ``0`` means "display everything".

    Parameters
    ----------
    raw : str
        Text supplied on the command line.

    Returns
    -------
    int | None
        The limit, or ``None`` for no limit.

    Raises
    ------
    argparse.ArgumentTypeError
        If *raw* is not a non-negative integer.
    """
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected an integer, got {raw!r}") from error
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected 0 or more, got {value}")
    return None if value == 0 else value


def _common_options() -> argparse.ArgumentParser:
    """Build the parent parser carrying options shared by every subcommand.

    Returns
    -------
    argparse.ArgumentParser
        Parser to be used as a subparser parent.
    """
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--json",
        action="store_true",
        help="emit one JSON document on stdout instead of the text rendering",
    )
    parent.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="log at DEBUG level",
    )
    parent.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "suppress the rendered report; the exit code still reports coverage "
            "and errors still go to stderr"
        ),
    )
    return parent


def _set_options() -> argparse.ArgumentParser:
    """Build the parent parser for commands that produce a set of papers.

    Returns
    -------
    argparse.ArgumentParser
        Parser to be used as a subparser parent.
    """
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--limit",
        type=display_limit,
        default=10,
        metavar="N",
        help="how many papers to display; 0 displays all of them (default: 10)",
    )
    parent.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="write the full result to this JSON file and print its path",
    )
    return parent


def _store_options() -> argparse.ArgumentParser:
    """Build the parent parser for commands that read or write handles.

    Returns
    -------
    argparse.ArgumentParser
        Parser to be used as a subparser parent.
    """
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--store-root",
        metavar="PATH",
        help="handle store root (default: $LIT_SEARCH_HOME, else the XDG cache directory)",
    )
    return parent


def _model_options() -> argparse.ArgumentParser:
    """Build the parent parser for commands that embed text.

    Returns
    -------
    argparse.ArgumentParser
        Parser to be used as a subparser parent.
    """
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--index-root",
        metavar="PATH",
        help="vector index root (default: <store root>/index)",
    )
    parent.add_argument(
        "--model",
        metavar="ID",
        help="embedding model id (default: $LIT_SEARCH_EMBEDDING_MODEL, else the pinned one)",
    )
    parent.add_argument(
        "--model-revision",
        metavar="REV",
        help="pin the model revision; omitting it is recorded as 'unpinned'",
    )
    parent.add_argument(
        "--allow-symmetric-encoding",
        action="store_true",
        help=(
            "accept a model that ships no 'query' prompt. Without this the load fails, "
            "because encode_query() would silently apply no prefix"
        ),
    )
    parent.add_argument(
        "--deterministic-embedder",
        action="store_true",
        help=(
            "use the reproducible NON-SEMANTIC stand-in instead of a real model; for "
            "exercising the plumbing, never for a real search"
        ),
    )
    return parent


def _add_handle(subparsers: argparse._SubParsersAction, parents: list) -> None:
    """Register the ``handle`` subcommand and its actions.

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
    common, sets, store = parents
    parser = subparsers.add_parser(
        "handle",
        help="list, show, explain and import handles",
        description=(
            "Handles name sets. They are content addresses, so an abbreviated "
            "prefix is accepted and an ambiguous one is an error rather than a "
            "first match."
        ),
    )
    actions = parser.add_subparsers(dest="action", metavar="ACTION", required=True)

    listing = actions.add_parser("list", parents=[common, store], help="list stored handles")
    listing.add_argument(
        "--limit", type=display_limit, default=20, metavar="N", help="rows to show; 0 for all"
    )

    show = actions.add_parser(
        "show", parents=[common, sets, store], help="render one handle and its coverage"
    )
    show.add_argument("handle", help="handle identifier or unambiguous prefix")

    lineage = actions.add_parser(
        "lineage", parents=[common, store], help="print how a handle came to be"
    )
    lineage.add_argument("handle", help="handle identifier or unambiguous prefix")

    importer = actions.add_parser(
        "import",
        parents=[common, sets, store],
        help="pin a saved JSON result as a handle so set algebra can use it",
    )
    importer.add_argument("input", help="path to a JSON result file")
    importer.add_argument("--label", metavar="TEXT", help="human-readable name for the handle")
    importer.add_argument(
        "--kind",
        default="result_file",
        metavar="KIND",
        help="what is being pinned, recorded in lineage (default: result_file)",
    )


def _add_set(subparsers: argparse._SubParsersAction, parents: list) -> None:
    """Register the ``set`` subcommand and its actions.

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
    common, sets, store, model = parents
    parser = subparsers.add_parser(
        "set",
        help="union, difference, intersection and deduplication over handles",
        description=(
            "Set algebra over stored handles. Every operation writes a new "
            "handle and propagates coverage conservatively: the result inherits "
            "the worst status of its operands, and a difference taken against a "
            "subtrahend that was not complete is itself not complete."
        ),
    )
    actions = parser.add_subparsers(dest="action", metavar="ACTION", required=True)
    shared = [common, sets, store]

    union = actions.add_parser("union", parents=shared, help="combine two or more handles")
    union.add_argument("handles", nargs="+", help="two or more handle identifiers")
    union.add_argument("--label", metavar="TEXT", help="human-readable name for the result")

    difference = actions.add_parser(
        "difference",
        parents=shared,
        help="remove the members of one or more handles from a base handle",
    )
    difference.add_argument("base", help="handle to subtract from")
    difference.add_argument("subtract", nargs="+", help="one or more handles to subtract")
    difference.add_argument("--label", metavar="TEXT", help="human-readable name for the result")

    intersect = actions.add_parser(
        "intersect", parents=shared, help="keep only members present in every operand"
    )
    intersect.add_argument("handles", nargs="+", help="two or more handle identifiers")
    intersect.add_argument("--label", metavar="TEXT", help="human-readable name for the result")

    dedup = actions.add_parser(
        "dedup",
        parents=[*shared, model],
        help="collapse duplicates and link versions within one handle",
        description=(
            "Exact matching first, semantic matching only on the residue. "
            "Versions are LINKED, never merged: the preprint and the published "
            "record both survive, with a reading target and a citation target "
            "recorded. Without --semantic the residue is reported as "
            "UNEXAMINED, which is not the same as clean."
        ),
    )
    dedup.add_argument("handle", help="handle to deduplicate")
    dedup.add_argument("--label", metavar="TEXT", help="human-readable name for the result")
    dedup.add_argument(
        "--semantic",
        action="store_true",
        help="run the second stage over the residue with an embedding model",
    )
    dedup.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SEMANTIC_THRESHOLD,
        metavar="F",
        help=f"cosine similarity for a semantic match (default: {DEFAULT_SEMANTIC_THRESHOLD})",
    )


def _add_index(subparsers: argparse._SubParsersAction, parents: list) -> None:
    """Register the ``index`` subcommand and its actions.

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
    common, store, model = parents
    parser = subparsers.add_parser(
        "index",
        help="maintain and query the embedded vector and full-text index",
        description=(
            "The index is a LanceDB table. Keyword retrieval needs a full-text "
            "index, which is created on first ingest; rows added afterwards are "
            "served by a flat scan until the table is compacted, so the "
            "unindexed tail is reported rather than assumed away."
        ),
    )
    actions = parser.add_subparsers(dest="action", metavar="ACTION", required=True)

    adder = actions.add_parser(
        "add", parents=[common, store, model], help="embed a handle's members into the index"
    )
    adder.add_argument("handle", help="handle whose members to index")

    actions.add_parser(
        "status", parents=[common, store, model], help="report size, identity and unindexed tail"
    )
    actions.add_parser(
        "optimize",
        parents=[common, store, model],
        help="build the full-text index if absent and compact the unindexed tail",
    )

    searcher = actions.add_parser(
        "search",
        parents=[common, store, model],
        help="query the whole index, reporting which lane surfaced each record",
    )
    searcher.add_argument("query", help="free text; quote a span to use the literal lane")
    searcher.add_argument(
        "--limit", type=display_limit, default=20, metavar="N", help="results to show"
    )
    searcher.add_argument(
        "--exact",
        action="append",
        metavar="TEXT",
        help="an extra literal substring for the exact lane; repeatable",
    )


def _add_handle_output(parser: argparse.ArgumentParser) -> None:
    """Let a set-producing command store its result as a handle.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Subcommand parser to extend.

    Returns
    -------
    None
    """
    parser.add_argument(
        "--handle",
        action="store_true",
        help="store the result as a handle and print its identifier",
    )
    parser.add_argument("--label", metavar="TEXT", help="human-readable name for the handle")


def _add_search(subparsers: argparse._SubParsersAction, parents: list) -> None:
    """Register the ``search`` subcommand.

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
    parser = subparsers.add_parser(
        "search",
        parents=parents,
        help="run a Boolean query across databases and report per-source coverage",
        description=(
            "Run a Boolean query across databases. The rendered result always "
            "states which sources were consulted and what each of them failed "
            "to return."
        ),
    )
    parser.add_argument("query", help="Boolean query, e.g. 'ti[transformer] AND abs[attention]'")
    parser.add_argument(
        "-d",
        "--database",
        action="append",
        dest="databases",
        choices=SEARCH_DATABASES,
        metavar="DB",
        help="restrict to this database; repeatable (default: usable source profile)",
    )
    parser.add_argument(
        "--max-papers",
        type=positive_int,
        metavar="N",
        help="per-database retrieval cap; a cap that bites is recorded as CAPPED coverage",
    )
    parser.add_argument("--since", type=iso_date, metavar="YYYY-MM-DD", help="earliest publication")
    parser.add_argument("--until", type=iso_date, metavar="YYYY-MM-DD", help="latest publication")
    parser.add_argument(
        "--no-enrich",
        action="store_true",
        help="skip the post-search metadata enrichment pass",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=1,
        metavar="N",
        help="databases queried in parallel (default: 1)",
    )
    _add_handle_output(parser)


def _add_get(subparsers: argparse._SubParsersAction, parents: list) -> None:
    """Register the ``get`` subcommand.

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
    parser = subparsers.add_parser(
        "get",
        parents=parents,
        help="fetch one paper by DOI or landing-page URL",
        description=(
            "Fetch one paper by DOI or landing-page URL. The lookup runner "
            "records no coverage ledger yet, so the result always reports "
            "UNKNOWN coverage and says why."
        ),
    )
    parser.add_argument("identifier", help="a DOI, a doi.org URL, or a paper landing page")
    parser.add_argument(
        "-d",
        "--database",
        action="append",
        dest="databases",
        metavar="DB",
        help="restrict to this source; repeatable (default: all available)",
    )


def _add_snowball(subparsers: argparse._SubParsersAction, parents: list) -> None:
    """Register the ``snowball`` subcommand.

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
    parser = subparsers.add_parser(
        "snowball",
        parents=parents,
        help="expand a seed paper's citation neighbourhood",
        description=(
            "Expand a stored set or one seed paper through provenance-bearing, "
            "source-native citation edges. A handle seeds every member; DOI and "
            "landing-page inputs remain supported for focused traversals."
        ),
    )
    parser.add_argument("seed", help="handle (preferred), DOI, or landing-page URL")
    parser.add_argument(
        "--depth", type=positive_int, default=2, metavar="N", help="BFS levels (default: 2)"
    )
    parser.add_argument(
        "--direction",
        choices=["both", "backward", "forward"],
        default="both",
        help="follow references, citations, or both (default: both)",
    )
    parser.add_argument(
        "--max-papers-per-level",
        type=positive_int,
        metavar="N",
        help="keep only the top N most-cited papers found at each level",
    )
    parser.add_argument(
        "-d",
        "--database",
        action="append",
        dest="databases",
        metavar="DB",
        help="citation source; repeatable (crossref, openalex, semantic_scholar)",
    )
    parser.add_argument(
        "--no-enrich",
        action="store_true",
        help="skip metadata-only enrichment after graph traversal",
    )
    managed = parser.add_mutually_exclusive_group()
    managed.add_argument(
        "--managed-run",
        action="store_true",
        help=(
            "checkpoint native provider requests/pages under the store root and print a "
            "run ID before traversal starts"
        ),
    )
    managed.add_argument(
        "--resume-run",
        metavar="RUN_ID",
        help=(
            "resume a checkpointed run; the seed, traversal options, and credential "
            "fingerprints must match exactly"
        ),
    )
    _add_handle_output(parser)


def _add_rerank(subparsers: argparse._SubParsersAction, parents: list) -> None:
    """Register the ``rerank`` subcommand.

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
    parser = subparsers.add_parser(
        "rerank",
        parents=parents,
        help="order a stored set against a question or a seed paper",
        description=(
            "Order a stored handle. Two distinct operations: --query ranks "
            "against a question and embeds it with the model's query "
            "convention, --seed ranks against a member paper and embeds it "
            "with the document convention. The result is written as a new "
            "ranked handle in the RELEVANCE order; under --scoring hybrid a "
            "diversified order is shown beside it, and that is a display "
            "choice only -- never a source of snowball seeds."
        ),
    )
    parser.add_argument("handle", help="handle to order, or an unambiguous prefix")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", metavar="TEXT", help="rank against a question (query mode)")
    group.add_argument(
        "--seed", metavar="DOI_OR_TITLE", help="rank against a member paper (seed mode)"
    )
    parser.add_argument("--label", metavar="TEXT", help="human-readable name for the result")
    parser.add_argument(
        "--scoring",
        choices=["two-stage", "hybrid"],
        default="two-stage",
        help=(
            "two-stage is the semantic re-rank: a bi-encoder recalls, a "
            "cross-encoder rescores the head. hybrid instead orders by the "
            "index's three retrieval lanes fused, which is a different claim "
            "and is NOT a semantic re-rank (default: two-stage)"
        ),
    )
    parser.add_argument(
        "--top-k",
        type=positive_int,
        metavar="N",
        help="how many of the recalled head the cross-encoder rescores",
    )
    parser.add_argument(
        "--no-cross-encoder",
        action="store_true",
        help=(
            "run stage one only; the result records that stage two did not run "
            "rather than presenting a one-stage order as a two-stage one"
        ),
    )
    parser.add_argument(
        "--mmr-lambda",
        type=float,
        default=DEFAULT_LAMBDA,
        metavar="F",
        help=f"relevance weight for the diversified display order (default: {DEFAULT_LAMBDA})",
    )
    parser.add_argument(
        "--no-diversify",
        action="store_true",
        help="do not compute the diversified display order",
    )


def _add_export(subparsers: argparse._SubParsersAction, parents: list) -> None:
    """Register the ``export`` subcommand.

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
    parser = subparsers.add_parser(
        "export",
        parents=parents,
        help="write a stored result to a file and print its path",
        description=(
            "Write a stored result to a file and print its path. Contents are "
            "never printed. A coverage sidecar is written beside every export, "
            "so a file handed to someone else does not travel without the "
            "record of what it is missing. BibTeX requires --citation-keys: "
            "Better BibTeX is the single authority for citation keys (D7) and "
            "this tool refuses to mint its own."
        ),
    )
    parser.add_argument("input", help="path to a JSON result file produced by this tool")
    parser.add_argument(
        "-f",
        "--format",
        choices=["json", "bibtex", "csv"],
        default="json",
        help="output format (default: json)",
    )
    parser.add_argument("-o", "--output", required=True, metavar="PATH", help="file to write")
    parser.add_argument(
        "--citation-keys",
        metavar="PATH",
        help=(
            "JSON object mapping DOI (or lower-cased title) to its Better BibTeX key; "
            "required for --format bibtex, and a record missing from it is an error"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the whole ``lit-search`` parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with every subcommand registered.
    """
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    common = _common_options()
    sets = _set_options()
    store = _store_options()
    model = _model_options()
    add_hunt_parser(subparsers, [common, store])
    add_find_parser(subparsers, [common, sets, store])
    add_explore_parsers(subparsers, [common, sets, store])
    add_fetch_parser(subparsers, [common, store])
    _add_search(subparsers, [common, sets, store])
    _add_get(subparsers, [common, sets])
    _add_snowball(subparsers, [common, sets, store])
    _add_handle(subparsers, [common, sets, store])
    _add_set(subparsers, [common, sets, store, model])
    _add_index(subparsers, [common, store, model])
    _add_rerank(subparsers, [common, sets, store, model])
    _add_export(subparsers, [common])
    add_setup_parsers(subparsers, [common])
    add_eval_parser(subparsers, [common])
    return parser
