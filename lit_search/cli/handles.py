"""Handle discovery and set algebra at the command line.

This is the surface ``O4`` left open: whether set algebra is subcommands,
flags, or implicit.  It is **subcommands under ``set``**, for one reason that
outweighs the ergonomics: every set operation writes a new handle whose
coverage was propagated from its operands, and a subcommand can print that
propagation.  A flag on ``search`` could not -- it would fold "here is your
result" and "here is why this result may be missing things it should have
removed" into one line, and the second half is the half that gets skipped.

``handle`` exists because ``set`` is unusable without it.  Handle identifiers
are content addresses, so a person who cannot list them keeps them in a scratch
file by hand and mistypes one.  Listing is read-only and needs no mutable alias
namespace, which the store deliberately does not have -- naming is a write, and
writes are what immutability removes.

Coverage propagation, in one paragraph
--------------------------------------

A derived set inherits the **worst** status of its operands, and a difference
against a non-complete subtrahend is itself non-complete.  The second half is
the one with teeth: subtracting a partially-synced Zotero library from a search
result labels already-held papers as new, and nothing in the member list
reveals it.  The algebra records that as a derivation note, which floors the
result's status without falsely claiming Zotero was *searched*.

Order, through the same operations
----------------------------------

``difference`` and ``intersect`` inherit the first operand's order, because
removing members cannot make a claim about the survivors' relative sequence
false.  ``union`` keeps an order only when every operand is ranked under
mutually mergeable, pool-independent provenance, and otherwise invalidates --
which is why a diversified ranking, being pool-dependent, can never be merged.
``dedup`` invalidates a ranked base outright: collapsing two records produces a
record whose text is not the text that was scored.
"""

from __future__ import annotations

import argparse
import datetime
from typing import Any

from lit_search.cli.context import resolve_embedder, resolve_store
from lit_search.cli.emit import emit_set
from lit_search.cli.output import EXIT_ERROR, EXIT_OK, emit_error, emit_json, emit_text
from lit_search.core.paper import Paper
from lit_search.store import operations
from lit_search.store.dedup import deduplicate, record_deduplication
from lit_search.store.lineage import ExternalReference
from lit_search.store.record import HandleRecord
from lit_search.store.store import HandleStore
from lit_search.store.view import handle_view
from lit_search.utils.persistence import load_from_json


def _resolve_all(store: HandleStore, prefixes: list[str]) -> list[str]:
    """Expand abbreviated handle identifiers.

    Parameters
    ----------
    store : HandleStore
        Store to resolve against.
    prefixes : list[str]
        Full identifiers or unambiguous prefixes.

    Returns
    -------
    list[str]
        Full identifiers, in the order given.
    """
    return [store.resolve(prefix) for prefix in prefixes]


def _emit_handle(
    command: str,
    store: HandleStore,
    record: HandleRecord,
    args: argparse.Namespace,
    *,
    extra_lines: list[str] | None = None,
) -> int:
    """Persist a derived handle and render it with its propagated coverage.

    Parameters
    ----------
    command : str
        Subcommand name.
    store : HandleStore
        Store to write to.
    record : HandleRecord
        Handle to persist.
    args : argparse.Namespace
        Parsed arguments.
    extra_lines : list[str] | None
        Extra text lines to print after the rendering.

    Returns
    -------
    int
        Exit code implied by the derived set's coverage.
    """
    handle_id = store.put(record)
    view = handle_view(handle_id, record)
    lines = [f"Handle: {handle_id}", *(extra_lines or [])]
    return emit_set(command, view, view.headline(), args, extra_lines=lines, handle=handle_id)


# ----------------------------------------------------------------------
# handle
# ----------------------------------------------------------------------


def cmd_handle_list(args: argparse.Namespace) -> int:
    """List stored handles, newest first, with what produced each one.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        :data:`~lit_search.cli.output.EXIT_OK`, or
        :data:`~lit_search.cli.output.EXIT_ERROR` when a stored handle could
        not be read -- an unreadable handle is reported rather than skipped,
        because a listing that quietly omits one is a listing you cannot trust.
    """
    store = resolve_store(args)
    rows: list[dict[str, Any]] = []
    unreadable: list[str] = []
    for handle_id in store.list_ids():
        try:
            record = store.get(handle_id)
        except Exception as error:
            unreadable.append(f"{handle_id}: {type(error).__name__}: {error}")
            continue
        rows.append(
            {
                "handle": handle_id,
                "created_at": record.created_at.isoformat(),
                "operation": record.operation.name.value,
                "members": len(record.members),
                "order": record.order.state.value,
                "label": record.label,
            }
        )
    rows.sort(key=lambda row: str(row["created_at"]), reverse=True)
    if args.limit is not None:
        rows = rows[: args.limit]

    if args.json:
        emit_json({"command": "handle list", "handles": rows, "unreadable": unreadable})
    elif not args.quiet:
        emit_text(f"{len(rows)} handle(s) in {store.handles_directory}")
        for row in rows:
            label = f" {row['label']!r}" if row["label"] else ""
            emit_text(
                f"  {row['handle']}  {row['created_at']}  {row['operation']:<12} "
                f"{row['members']:>5} member(s)  order={row['order']}{label}"
            )
    for line in unreadable:
        emit_error(f"unreadable handle -- {line}")
    return EXIT_ERROR if unreadable else EXIT_OK


def cmd_handle_show(args: argparse.Namespace) -> int:
    """Render one handle, its members, and its coverage.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code implied by the handle's coverage.
    """
    store = resolve_store(args)
    handle_id = store.resolve(args.handle)
    view = handle_view(handle_id, store.get(handle_id))
    return emit_set("handle show", view, view.headline(), args, handle=handle_id)


def cmd_handle_lineage(args: argparse.Namespace) -> int:
    """Print how a handle came to be, including any missing ancestor.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        :data:`~lit_search.cli.output.EXIT_OK`, or
        :data:`~lit_search.cli.output.EXIT_ERROR` when an ancestor is missing:
        a lineage with a hole in it is a different fact from a complete one.
    """
    store = resolve_store(args)
    handle_id = store.resolve(args.handle)
    graph = store.lineage(handle_id)
    payload = {
        "command": "handle lineage",
        "root": graph.root,
        "nodes": list(graph.nodes),
        "edges": [
            {
                "child": edge.child,
                "parent": edge.parent,
                "role": edge.role,
                "operation": edge.operation.value,
            }
            for edge in graph.edges
        ],
        "external": [
            {"child": child, "kind": reference.kind, "identity": dict(reference.identity)}
            for child, reference in graph.external
        ],
        "missing": list(graph.missing),
    }
    if args.json:
        emit_json(payload)
    elif not args.quiet:
        emit_text(f"lineage of {graph.root}: {len(graph.nodes)} node(s)")
        for edge in graph.edges:
            emit_text(f"  {edge.child} <- {edge.parent} ({edge.operation.value}/{edge.role})")
        for child, reference in graph.external:
            emit_text(f"  {child} <- external {reference.kind} {dict(reference.identity)}")
    for missing in graph.missing:
        emit_error(
            f"ancestor {missing} is referenced by this lineage and is not in the store, so "
            "how this set came to be cannot be fully reconstructed"
        )
    return EXIT_ERROR if graph.missing else EXIT_OK


def cmd_handle_import(args: argparse.Namespace) -> int:
    """Pin a saved result file as a handle so set algebra can use it.

    Importing rather than reading the file at each operation is what turns a
    mutable outside thing into an operand with a timestamp: a Zotero library or
    a result file can change between runs, and a difference taken against
    "whatever that file said at the time" is not reproducible.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code implied by the imported set's coverage, which is UNKNOWN
        unless the file carried a ledger.
    """
    store = resolve_store(args)
    loaded = load_from_json(args.input)
    papers: list[Paper] = list(loaded) if isinstance(loaded, list) else list(loaded.papers)
    reference = ExternalReference(
        kind=args.kind,
        identity={"path": str(args.input), "item_count": len(papers)},
        captured_at=datetime.datetime.now(datetime.UTC),
    )
    coverage = getattr(loaded, "coverage", None)
    serialize = getattr(coverage, "to_dict", None)
    record = operations.imported(
        papers,
        source=reference,
        parameters={"source_path": str(args.input)},
        label=args.label,
        coverage=serialize() if callable(serialize) else None,
    )
    return _emit_handle("handle import", store, record, args)


# ----------------------------------------------------------------------
# set
# ----------------------------------------------------------------------


def cmd_set_union(args: argparse.Namespace) -> int:
    """Combine two or more handles into one.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code implied by the union's coverage, which carries every
        operand's shortfalls.
    """
    store = resolve_store(args)
    ids = _resolve_all(store, args.handles)
    record = operations.union(store, ids, label=args.label)
    return _emit_handle("set union", store, record, args)


def cmd_set_difference(args: argparse.Namespace) -> int:
    """Remove the members of one or more handles from a base handle.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code implied by the result's coverage.  A subtrahend that was not
        complete floors it, because records it failed to retrieve were not
        subtracted and are present here in error.
    """
    store = resolve_store(args)
    base = store.resolve(args.base)
    subtrahends = _resolve_all(store, args.subtract)
    record = operations.difference(store, base, subtrahends, label=args.label)
    return _emit_handle("set difference", store, record, args)


def cmd_set_intersect(args: argparse.Namespace) -> int:
    """Keep only the members present in every operand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code implied by the intersection's coverage.  Being smaller does
        not make it safer: a record that belongs in both sets is absent
        whenever either operand failed to retrieve it.
    """
    store = resolve_store(args)
    ids = _resolve_all(store, args.handles)
    record = operations.intersection(store, ids, label=args.label)
    return _emit_handle("set intersect", store, record, args)


def cmd_set_dedup(args: argparse.Namespace) -> int:
    """Collapse duplicates, link versions, and say whether stage two ran.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code implied by the result's coverage.
    """
    store = resolve_store(args)
    base_id = store.resolve(args.handle)
    base = store.get(base_id)
    embedder = resolve_embedder(args) if args.semantic else None
    plan = deduplicate(list(base.members), embedder=embedder, semantic_threshold=args.threshold)
    record = record_deduplication(store, base_id, plan, label=args.label)
    lines = [plan.semantic_stage.describe()]
    lines.extend(f"  collapse: {group.describe()}" for group in plan.duplicate_groups)
    lines.extend(f"  link: {link.describe()}" for link in plan.version_links)
    if plan.divergent_links:
        lines.append(
            f"  {len(plan.divergent_links)} version link(s) point a reading list and a "
            "bibliography at different records; that divergence is information, not an "
            "error to resolve"
        )
    return _emit_handle("set dedup", store, record, args, extra_lines=lines)


HANDLE_ACTIONS = {
    "list": cmd_handle_list,
    "show": cmd_handle_show,
    "lineage": cmd_handle_lineage,
    "import": cmd_handle_import,
}

SET_ACTIONS = {
    "union": cmd_set_union,
    "difference": cmd_set_difference,
    "intersect": cmd_set_intersect,
    "dedup": cmd_set_dedup,
}


def cmd_handle(args: argparse.Namespace) -> int:
    """Dispatch a ``handle`` subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code from the action.
    """
    return HANDLE_ACTIONS[args.action](args)


def cmd_set(args: argparse.Namespace) -> int:
    """Dispatch a ``set`` subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code from the action.
    """
    return SET_ACTIONS[args.action](args)


__all__ = ["HANDLE_ACTIONS", "SET_ACTIONS", "cmd_handle", "cmd_set"]
