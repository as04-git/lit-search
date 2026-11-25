"""Operations that build new handles from existing ones.

Every function here returns a *new* :class:`~lit_search.store.record.HandleRecord`
and never edits an existing one.  Each records its operands as lineage, so the
question "how did this set come to be" is answered by walking the graph rather
than by reconstructing it from memory.

Order-transition rules
----------------------

The single principle: **an operation may keep an order only if every member of
its result was scored under the same ranking claim.**  Otherwise the order is
invalidated.

=================  ==========================================================
Operation          Order of the result
=================  ==========================================================
``search``         unordered.
``query_family``   unordered.
``snowball``       unordered.
``import``         unordered.
``rerank``         ranked, with fresh provenance.
``difference``     inherits the base operand.  A ranking survives, because
                   removing members cannot make a pairwise ordering claim
                   false.
``intersection``   inherits the first operand, for the same reason.
``filter``         inherits the base operand, for the same reason.
``union``          ranked only when *every* operand is ranked under mutually
                   mergeable provenance, in which case members are merged by
                   score.  Otherwise invalidated when any operand was ranked,
                   and unordered when none was.
``deduplicate``    invalidated when the base was ranked.  Collapsing two
                   records produces a record whose text is not the text that
                   was scored, so the surviving score describes something that
                   no longer exists.
=================  ==========================================================

The conservative rules cost nothing in the ordinary workflow, which is search,
then deduplicate, then re-rank: deduplication runs on an unordered set, so
there is no ranking for it to invalidate.  A rule that is only expensive in the
unusual order is a rule worth keeping strict.

What these functions deliberately do not do
-------------------------------------------

Union and intersection are **membership** operations.  When two operands hold
the same paper, the record from the earliest operand is kept and the other is
discarded -- neither is merged into the other.  Merging is destructive (it
keeps one DOI and drops the other, and it collapses per-version retraction
status), so it is a separate, explicit operation and not a side effect of set
algebra.

Deduplication strategy and re-ranking models live outside this module.
:func:`deduplicated` and :func:`reranked` take the finished member list from
the component that computed it and are responsible only for recording the
result truthfully as a handle.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from lit_search.core.identity import same_record_identity
from lit_search.core.paper import Paper
from lit_search.core.query_family import EvidenceKind, QueryFamilyRevision
from lit_search.core.query_family_result import QueryCellOutcome, QueryFamilyResult
from lit_search.core.search_result import SearchResult
from lit_search.core.snowball_result import SnowballResult
from lit_search.exceptions import InvalidParameterError
from lit_search.store.lineage import (
    ExternalReference,
    Operand,
    Operation,
    OperationName,
)
from lit_search.store.order import OrderRecord, OrderState, RankingProvenance
from lit_search.store.propagation import (
    BASE_ROLE,
    MEMBER_ROLE,
    SUBTRAHEND_ROLE,
    CoverageOperand,
    CoveragePropagator,
    LedgerCoveragePropagator,
    RefusingCoveragePropagator,
)
from lit_search.store.record import HandleRecord
from lit_search.store.store import HandleStore

__all__ = [
    "CoverageOperand",
    "CoveragePropagator",
    "LedgerCoveragePropagator",
    "RefusingCoveragePropagator",
    "deduplicated",
    "difference",
    "filtered",
    "from_query_cell_outcome",
    "from_query_family_result",
    "from_search_result",
    "from_snowball_result",
    "imported",
    "intersection",
    "reranked",
    "searched",
    "union",
]


def _propagate(
    propagator: CoveragePropagator | None,
    operation: OperationName,
    records: Sequence[HandleRecord],
    roles: Sequence[str],
) -> Any:
    """Combine operand coverage ledgers with the supplied or default propagator.

    The default is :class:`~lit_search.store.propagation.LedgerCoveragePropagator`,
    which applies the coverage algebra rather than refusing.  This is the wiring
    that makes the algebra reachable: it had no callers, so every set operation
    on a handle derived from a real search used to raise, and the propagation
    clause of the contract was prose with nothing behind it.

    *roles* is passed through to the propagator so that ``difference`` can tell
    its minuend from its subtrahends without inferring it from position.

    Parameters
    ----------
    propagator : CoveragePropagator | None
        Propagator to use, or ``None`` for the default.
    operation : OperationName
        Operation being performed.
    records : Sequence[HandleRecord]
        Operand handles, in operand order.
    roles : Sequence[str]
        Each operand's role, in the same order.

    Returns
    -------
    Any
        Coverage payload for the result.
    """
    active = propagator if propagator is not None else LedgerCoveragePropagator()
    operands = [
        CoverageOperand(role=role, coverage=record.coverage, member_count=len(record.members))
        for record, role in zip(records, roles, strict=True)
    ]
    return active.combine(operation, operands)


# ---------------------------------------------------------------------------
# Source handles
# ---------------------------------------------------------------------------


def searched(
    papers: Sequence[Paper],
    *,
    query: str,
    databases: Sequence[str],
    parameters: Mapping[str, Any] | None = None,
    label: str | None = None,
    coverage: Any = None,
) -> HandleRecord:
    """Build a handle for a set produced by a database search.

    Parameters
    ----------
    papers : Sequence[Paper]
        Papers found.  Their sequence is recorded but carries no ranking
        claim.
    query : str
        Boolean query that was run.
    databases : Sequence[str]
        Databases that were asked.
    parameters : Mapping[str, Any] | None
        Any further reproducibility parameters, for example date bounds and
        per-database caps.
    label : str | None
        Optional human-readable name.
    coverage : Any
        Coverage ledger, when one is available.

    Returns
    -------
    HandleRecord
        An unordered handle.
    """
    merged: dict[str, Any] = {"query": query, "databases": list(databases)}
    merged.update(parameters or {})
    return HandleRecord(
        operation=Operation(name=OperationName.SEARCH, parameters=merged),
        order=OrderRecord.unordered(),
        members=tuple(papers),
        label=label,
        coverage=coverage,
    )


def from_search_result(
    result: SearchResult, *, label: str | None = None, coverage: Any = None
) -> HandleRecord:
    """Build a handle from the library's own search result object.

    The search result's ``failed_databases`` list is carried into the
    operation parameters under ``recorded_failed_databases``.  It is **not** a
    coverage ledger and must not be read as one: it records databases that
    raised, and says nothing about a database skipped for a missing
    credential, a query the source could not express, a result cap that
    truncated, or a connector that swallowed its own exception and returned an
    empty list.  It is kept because discarding it would lose information, and
    it is superseded once the ``coverage`` slot is populated.

    Parameters
    ----------
    result : SearchResult
        Result produced by the search runner.
    label : str | None
        Optional human-readable name.
    coverage : Any
        Coverage ledger to attach.  When omitted, the result's own ledger is
        carried across, so building a handle from a search never drops the
        coverage the search recorded.

    Returns
    -------
    HandleRecord
        An unordered handle.
    """
    return searched(
        result.papers,
        query=result.query,
        databases=list(result.databases or []),
        parameters={
            "since": result.since.isoformat() if result.since else None,
            "until": result.until.isoformat() if result.until else None,
            "max_papers_per_database": result.max_papers_per_database,
            "enrichment_databases": result.enrichment_databases,
            "max_cited_by": result.max_cited_by,
            "recorded_failed_databases": list(result.failed_databases),
        },
        label=label,
        coverage=coverage if coverage is not None else _ledger_payload(result),
    )


def from_query_cell_outcome(
    outcome: QueryCellOutcome,
    *,
    plan_id: str,
    label: str | None = None,
) -> HandleRecord:
    """Build the immutable ``SEARCH`` handle for one query-family cell.

    The ordinary search parameters remain present, but the operation also
    pins the semantic revision, execution plan, cell content address, and the
    source-translation decision.  Those fields let the family-level handle
    verify that a supplied cell handle is the exact cell it claims to be,
    rather than merely another search that happened to return the same papers.

    Parameters
    ----------
    outcome : QueryCellOutcome
        Exact cell and one-source search result produced for it.
    plan_id : str
        Content address of the complete query-family execution plan.
    label : str | None
        Optional human-readable handle label.

    Returns
    -------
    HandleRecord
        An unordered search handle carrying the cell's own coverage ledger.

    Raises
    ------
    InvalidParameterError
        If *plan_id* is blank.
    """
    if not plan_id.strip():
        raise InvalidParameterError("query-family cell handle requires a non-blank plan_id")

    cell = outcome.cell
    result = outcome.result
    return searched(
        result.papers,
        query=result.query,
        databases=list(result.databases or []),
        parameters={
            "since": result.since.isoformat() if result.since else None,
            "until": result.until.isoformat() if result.until else None,
            "max_papers_per_database": result.max_papers_per_database,
            "enrichment_databases": result.enrichment_databases,
            "max_cited_by": result.max_cited_by,
            "recorded_failed_databases": list(result.failed_databases),
            "revision_id": cell.revision_id,
            "plan_id": plan_id,
            "cell_id": cell.cell_id,
            "variant_id": cell.variant_id,
            "source": cell.source,
            "query_digest": cell.query_digest,
            "translation_id": cell.translation_id,
            "translation_mode": cell.translation_mode.value,
            "degradation_reason": cell.degradation_reason,
        },
        label=label,
        coverage=_ledger_payload(result),
    )


def from_query_family_result(
    result: QueryFamilyResult,
    *,
    cell_handles: Sequence[tuple[str, HandleRecord]],
    revision: QueryFamilyRevision,
    label: str | None = None,
) -> HandleRecord:
    """Build the family handle above exact cell and evidence handles.

    ``cell_handles`` is deliberately a sequence of ``(stored_id, record)``
    pairs rather than a bare list of identifiers.  This constructor can
    therefore prove count, order, content address, cell metadata, members, and
    coverage before recording an edge.  :class:`HandleStore` separately
    enforces that every resulting operand already exists when the family
    handle is written.

    Handle-member evidence is not just embedded in the revision payload.  Its
    unique source handles are explicit ``query_evidence`` operands, making the
    path from an executed query family back to the papers that introduced its
    vocabulary walkable through the ordinary handle-lineage DAG.

    Parameters
    ----------
    result : QueryFamilyResult
        Executed family result whose members and coverage the handle asserts.
    cell_handles : Sequence[tuple[str, HandleRecord]]
        Stored cell handle identifiers paired with their records, in exact
        ``result.cell_outcomes`` order.
    revision : QueryFamilyRevision
        Immutable semantic revision used to compile *result* and from which
        handle-member evidence dependencies are read.
    label : str | None
        Optional human-readable handle label.

    Returns
    -------
    HandleRecord
        Unordered family result with complete cell/evidence lineage.

    Raises
    ------
    InvalidParameterError
        If the revision, cell count, order, content, or pinned metadata does
        not correspond exactly to *result*.
    """
    if revision.revision_id != result.revision_id:
        raise InvalidParameterError(
            "query-family result revision mismatch: "
            f"result pins {result.revision_id}, supplied revision is {revision.revision_id}"
        )

    expected_count = len(result.cell_outcomes)
    if expected_count > 0 and not cell_handles:
        raise InvalidParameterError(
            "query-family result has applicable cells but no cell handles were supplied"
        )
    if len(cell_handles) != expected_count:
        raise InvalidParameterError(
            "query-family cell handle count mismatch: "
            f"expected {expected_count}, got {len(cell_handles)}"
        )

    stored_cell_ids: list[str] = []
    for index, ((handle_id, record), outcome) in enumerate(
        zip(cell_handles, result.cell_outcomes, strict=True)
    ):
        _require_query_cell_handle(
            handle_id,
            record,
            outcome,
            revision_id=result.revision_id,
            plan_id=result.plan_id,
            index=index,
        )
        stored_cell_ids.append(handle_id)

    evidence_handle_ids: list[str] = []
    for evidence in sorted(revision.evidence, key=lambda item: item.evidence_id):
        if evidence.kind is not EvidenceKind.HANDLE_MEMBER:
            continue
        if not evidence.handle_id:
            raise InvalidParameterError(
                f"handle-member evidence {evidence.evidence_id!r} has no handle_id"
            )
        if evidence.handle_id not in evidence_handle_ids:
            evidence_handle_ids.append(evidence.handle_id)

    operands = (
        *(Operand.of_handle(handle_id, "query_cell") for handle_id in stored_cell_ids),
        *(Operand.of_handle(handle_id, "query_evidence") for handle_id in evidence_handle_ids),
    )
    parameters = {
        "revision_id": result.revision_id,
        "plan_id": result.plan_id,
        "selected_sources": list(result.selected_sources),
        "scope_exclusions": [item.to_dict() for item in result.scope_exclusions],
        "cell_ids": [outcome.cell.cell_id for outcome in result.cell_outcomes],
        "cell_handle_ids": stored_cell_ids,
    }
    return HandleRecord(
        operation=Operation(
            name=OperationName.QUERY_FAMILY,
            operands=operands,
            parameters=parameters,
        ),
        order=OrderRecord.unordered(),
        members=tuple(result.papers),
        label=label,
        coverage=_ledger_payload(result),
    )


def _require_query_cell_handle(
    handle_id: str,
    record: HandleRecord,
    outcome: QueryCellOutcome,
    *,
    revision_id: str,
    plan_id: str,
    index: int,
) -> None:
    """Refuse a cell handle that does not exactly represent its outcome."""
    context = f"query-family cell handle at position {index}"
    if handle_id != record.handle_id:
        raise InvalidParameterError(
            f"{context} pairs identifier {handle_id!r} with record {record.handle_id!r}"
        )
    if record.operation.name is not OperationName.SEARCH:
        raise InvalidParameterError(
            f"{context} has operation {record.operation.name.value!r}, expected 'search'"
        )
    if record.order.state is not OrderState.UNORDERED:
        raise InvalidParameterError(f"{context} must be unordered")

    cell = outcome.cell
    expected_parameters: dict[str, Any] = {
        "query": outcome.result.query,
        "databases": list(outcome.result.databases or []),
        "revision_id": revision_id,
        "plan_id": plan_id,
        "cell_id": cell.cell_id,
        "variant_id": cell.variant_id,
        "source": cell.source,
        "query_digest": cell.query_digest,
        "translation_id": cell.translation_id,
        "translation_mode": cell.translation_mode.value,
        "degradation_reason": cell.degradation_reason,
    }
    for key, expected in expected_parameters.items():
        actual = record.operation.parameters.get(key)
        if actual != expected:
            raise InvalidParameterError(
                f"{context} parameter {key!r} mismatch: expected {expected!r}, got {actual!r}"
            )

    expected_members = [paper.to_dict() for paper in outcome.result.papers]
    actual_members = [paper.to_dict() for paper in record.members]
    if actual_members != expected_members:
        raise InvalidParameterError(f"{context} members do not match its QueryCellOutcome")
    expected_coverage = _ledger_payload(outcome.result)
    if record.coverage != expected_coverage:
        raise InvalidParameterError(f"{context} coverage does not match its QueryCellOutcome")


def _ledger_payload(result: Any) -> Any:
    """Serialize a result object's coverage ledger, if it carries one.

    Read defensively rather than by direct attribute access.  The coverage
    ledger is owned by the coverage contract, not by this module, and this
    function's job is only to make sure a ledger that exists is carried into
    the handle instead of being dropped on the floor.

    Parameters
    ----------
    result : Any
        A result object that may expose a ``coverage`` attribute.

    Returns
    -------
    Any
        The ledger's dictionary form, or ``None`` when there is no ledger.
    """
    ledger = getattr(result, "coverage", None)
    if ledger is None:
        return None
    to_dict = getattr(ledger, "to_dict", None)
    return to_dict() if callable(to_dict) else ledger


def from_snowball_result(
    result: SnowballResult,
    *,
    label: str | None = None,
    coverage: Any = None,
    seed_handle: str | None = None,
) -> HandleRecord:
    """Build a handle from the library's own snowball result object.

    The handle stores the expanded set (seeds plus discoveries), carries the
    result's edge-fetch ledger, and records a seed handle as lineage when one
    was supplied by the CLI.

    Parameters
    ----------
    result : SnowballResult
        Result produced by the snowball runner.
    label : str | None
        Optional human-readable name.
    coverage : Any
        Coverage ledger, when one is available.

    Returns
    -------
    HandleRecord
        An unordered handle.
    """
    seed = result.seed_paper
    parameters: dict[str, Any] = {
        "seed_doi": seed.doi if seed is not None else None,
        "seed_title": seed.title if seed is not None else None,
        "max_depth": result.max_depth,
        "direction": getattr(result.direction, "value", result.direction),
        "since": result.since.isoformat() if result.since is not None else None,
        "until": result.until.isoformat() if result.until is not None else None,
        "databases": list(result.databases or []),
        "enrichment_databases": list(result.enrichment_databases or []),
        "max_cited_by": result.max_cited_by,
        "max_papers_per_level": result.max_papers_per_level,
        "max_expansion_per_level": result.max_expansion_per_level,
        "seed_count": len(result.seed_papers),
    }
    operands = (Operand.of_handle(seed_handle, "seed_set"),) if seed_handle else ()
    effective_coverage = coverage if coverage is not None else _ledger_payload(result)
    return HandleRecord(
        operation=Operation(name=OperationName.SNOWBALL, operands=operands, parameters=parameters),
        order=OrderRecord.unordered(),
        members=tuple(result.all_papers),
        label=label,
        coverage=effective_coverage,
    )


def imported(
    papers: Sequence[Paper],
    *,
    source: ExternalReference,
    parameters: Mapping[str, Any] | None = None,
    label: str | None = None,
    coverage: Any = None,
) -> HandleRecord:
    """Build a handle pinning a snapshot of something outside the store.

    This is how a mutable source becomes usable in set algebra.  A Zotero
    library changes between runs, so subtracting it directly would make every
    result depend on an unrecorded moment in time.  Importing it first turns
    that dependency into an operand with a timestamp and an item count, which
    a later reader can inspect.

    Parameters
    ----------
    papers : Sequence[Paper]
        Papers in the snapshot.
    source : ExternalReference
        Identity of what was snapshotted and when.
    parameters : Mapping[str, Any] | None
        Any further reproducibility parameters.
    label : str | None
        Optional human-readable name.
    coverage : Any
        Coverage ledger, when one is available.  A partially-synced library is
        an incomplete set, and a difference taken against it is unsound
        without this.

    Returns
    -------
    HandleRecord
        An unordered handle whose sole operand is *source*.
    """
    return HandleRecord(
        operation=Operation(
            name=OperationName.IMPORT,
            operands=(Operand.of_external(source),),
            parameters=dict(parameters or {}),
        ),
        order=OrderRecord.unordered(),
        members=tuple(papers),
        label=label,
        coverage=coverage,
    )


# ---------------------------------------------------------------------------
# Derived handles
# ---------------------------------------------------------------------------


def reranked(
    store: HandleStore,
    base_id: str,
    *,
    members: Sequence[Paper],
    scores: Sequence[float],
    ranking: RankingProvenance,
    label: str | None = None,
    propagator: CoveragePropagator | None = None,
) -> HandleRecord:
    """Build a ranked handle by re-ordering an existing set.

    *members* must be a permutation of the base handle's members: a re-rank
    reorders a set, it does not select from one.  A re-ranker that dropped
    everything below a score threshold and returned the remainder would
    otherwise produce a smaller set wearing the label "ordering", and the
    truncation would be invisible.  Take the top *k* by composing
    :func:`reranked` with :func:`filtered`, which puts the truncation in the
    lineage where it can be seen.

    Parameters
    ----------
    store : HandleStore
        Store holding the base handle.
    base_id : str
        Full identifier of the set being re-ordered.
    members : Sequence[Paper]
        The base's members in their new order.
    scores : Sequence[float]
        One score per member, in the same order.
    ranking : RankingProvenance
        Model, mode, prompt convention, and scoring convention behind the
        order.  Recorded on every re-rank because a wrong prompt convention
        degrades retrieval silently and cannot be recovered afterwards.
    label : str | None
        Optional human-readable name.
    propagator : CoveragePropagator | None
        Coverage propagator.  Re-ranking does not change membership, so a
        faithful implementation passes the base's ledger through unchanged.

    Returns
    -------
    HandleRecord
        A ranked handle.

    Raises
    ------
    InvalidParameterError
        If *members* is not a permutation of the base's members, or if the
        score count does not match.
    """
    base = store.get(base_id)
    if len(scores) != len(members):
        raise InvalidParameterError(
            f"rerank received {len(members)} member(s) and {len(scores)} score(s)"
        )
    _require_same_membership(base.members, members, base_id)

    return HandleRecord(
        operation=Operation(
            name=OperationName.RERANK,
            operands=(Operand.of_handle(base_id, "base"),),
            parameters={"ranking": ranking.to_dict()},
        ),
        order=OrderRecord.ranked(ranking, scores),
        members=tuple(members),
        label=label,
        coverage=_propagate(propagator, OperationName.RERANK, [base], [BASE_ROLE]),
    )


def union(
    store: HandleStore,
    handle_ids: Sequence[str],
    *,
    label: str | None = None,
    propagator: CoveragePropagator | None = None,
) -> HandleRecord:
    """Combine two or more sets.

    Membership is first-seen-wins in operand order: when two operands hold the
    same paper, the record from the earlier operand is kept and the later one
    is discarded rather than merged into it.

    The order of the result follows the rule in the module docstring.  It is
    ranked only when every operand is ranked and their provenances are
    mutually mergeable, which additionally requires pool-independent scores --
    a diversified ordering scores a paper partly by what else is in the pool,
    so its scores cannot be compared across pools.

    Parameters
    ----------
    store : HandleStore
        Store holding the operands.
    handle_ids : Sequence[str]
        Full identifiers of at least two operands, in significant order.
    label : str | None
        Optional human-readable name.
    propagator : CoveragePropagator | None
        Coverage propagator.

    Returns
    -------
    HandleRecord
        The combined handle.

    Raises
    ------
    InvalidParameterError
        If fewer than two operands are given.
    """
    records = _load_operands(store, handle_ids, minimum=2, operation="union")

    members: list[Paper] = []
    origin: list[int] = []
    for index, record in enumerate(records):
        for member in record.members:
            if _identity_index(members, member) is not None:
                continue
            members.append(member)
            origin.append(index)

    order = _union_order(records, members, origin)
    merged_scores = order.scores
    if order.state is OrderState.RANKED and merged_scores is not None:
        # Ties keep first-seen order, so the result is deterministic.
        ranked_positions = sorted(
            range(len(members)), key=lambda position: (-merged_scores[position], position)
        )
        members = [members[position] for position in ranked_positions]
        order = OrderRecord.ranked(
            _expect_ranking(order), [merged_scores[position] for position in ranked_positions]
        )

    return HandleRecord(
        operation=Operation(
            name=OperationName.UNION,
            operands=tuple(Operand.of_handle(item, "member") for item in handle_ids),
        ),
        order=order,
        members=tuple(members),
        label=label,
        coverage=_propagate(propagator, OperationName.UNION, records, [MEMBER_ROLE] * len(records)),
    )


def difference(
    store: HandleStore,
    base_id: str,
    subtrahend_ids: Sequence[str],
    *,
    label: str | None = None,
    propagator: CoveragePropagator | None = None,
) -> HandleRecord:
    """Remove the members of one or more sets from a base set.

    Operand position is meaningful and is recorded in the operand roles: the
    base is ``"base"`` and every subtrahend is ``"subtrahend"``.

    The base's order survives.  Removing members cannot make a claim about the
    relative sequence of the survivors false, so a ranked base yields a ranked
    result with the survivors' scores intact.

    Soundness warning: the *result* of a difference is only as trustworthy as
    the *subtrahends*.  Subtracting an incomplete library confidently labels
    already-held papers as new, and nothing in the member list reveals it.
    That is what coverage propagation is for, and why the default propagator
    refuses rather than dropping an operand's ledger.

    Parameters
    ----------
    store : HandleStore
        Store holding the operands.
    base_id : str
        Full identifier of the set to subtract from.
    subtrahend_ids : Sequence[str]
        Full identifiers of at least one set to subtract.
    label : str | None
        Optional human-readable name.
    propagator : CoveragePropagator | None
        Coverage propagator.

    Returns
    -------
    HandleRecord
        The reduced handle.

    Raises
    ------
    InvalidParameterError
        If no subtrahend is given.
    """
    if not subtrahend_ids:
        raise InvalidParameterError("difference needs at least one subtrahend operand")
    records = _load_operands(store, [base_id, *subtrahend_ids], minimum=2, operation="difference")
    base = records[0]

    removed = [member for record in records[1:] for member in record.members]

    kept = [
        index
        for index, member in enumerate(base.members)
        if _identity_index(removed, member) is None
    ]
    operands = (
        Operand.of_handle(base_id, "base"),
        *(Operand.of_handle(item, "subtrahend") for item in subtrahend_ids),
    )
    return HandleRecord(
        operation=Operation(name=OperationName.DIFFERENCE, operands=operands),
        order=base.order.subset(kept),
        members=tuple(base.members[index] for index in kept),
        label=label,
        coverage=_propagate(
            propagator,
            OperationName.DIFFERENCE,
            records,
            [BASE_ROLE, *([SUBTRAHEND_ROLE] * (len(records) - 1))],
        ),
    )


def intersection(
    store: HandleStore,
    handle_ids: Sequence[str],
    *,
    label: str | None = None,
    propagator: CoveragePropagator | None = None,
) -> HandleRecord:
    """Keep only members present in every operand.

    The result is a subset of the first operand and inherits its order and
    records, for the same reason difference does.

    Parameters
    ----------
    store : HandleStore
        Store holding the operands.
    handle_ids : Sequence[str]
        Full identifiers of at least two operands.  The first is the one whose
        records and order are inherited.
    label : str | None
        Optional human-readable name.
    propagator : CoveragePropagator | None
        Coverage propagator.

    Returns
    -------
    HandleRecord
        The intersected handle.

    Raises
    ------
    InvalidParameterError
        If fewer than two operands are given.
    """
    records = _load_operands(store, handle_ids, minimum=2, operation="intersection")
    base = records[0]

    kept = [
        index
        for index, member in enumerate(base.members)
        if all(_identity_index(record.members, member) is not None for record in records[1:])
    ]
    operands = (
        Operand.of_handle(handle_ids[0], "base"),
        *(Operand.of_handle(item, "member") for item in handle_ids[1:]),
    )
    return HandleRecord(
        operation=Operation(name=OperationName.INTERSECTION, operands=operands),
        order=base.order.subset(kept),
        members=tuple(base.members[index] for index in kept),
        label=label,
        coverage=_propagate(
            propagator,
            OperationName.INTERSECTION,
            records,
            [BASE_ROLE, *([MEMBER_ROLE] * (len(records) - 1))],
        ),
    )


def filtered(
    store: HandleStore,
    base_id: str,
    predicate: Callable[[Paper], bool],
    *,
    description: str,
    label: str | None = None,
    propagator: CoveragePropagator | None = None,
) -> HandleRecord:
    """Keep the members of a set that satisfy a predicate.

    A predicate cannot be serialized, so *description* is required and must be
    non-empty: a filter that cannot say what it removed is a hole in the
    lineage, and a hole in the lineage is how a truncated set passes for a
    complete one.

    Parameters
    ----------
    store : HandleStore
        Store holding the base handle.
    base_id : str
        Full identifier of the set to filter.
    predicate : Callable[[Paper], bool]
        Returns ``True`` for members to keep.
    description : str
        Human-readable statement of what the predicate does, for example
        ``"top 100 by rerank score"`` or ``"publication_year >= 2020"``.
    label : str | None
        Optional human-readable name.
    propagator : CoveragePropagator | None
        Coverage propagator.

    Returns
    -------
    HandleRecord
        The reduced handle, inheriting the base's order.

    Raises
    ------
    InvalidParameterError
        If *description* is empty.
    """
    if not description.strip():
        raise InvalidParameterError(
            "filter requires a non-empty description; an unexplained filter leaves a "
            "hole in the lineage"
        )
    base = store.get(base_id)
    kept = [index for index, member in enumerate(base.members) if predicate(member)]
    return HandleRecord(
        operation=Operation(
            name=OperationName.FILTER,
            operands=(Operand.of_handle(base_id, "base"),),
            parameters={"description": description},
        ),
        order=base.order.subset(kept),
        members=tuple(base.members[index] for index in kept),
        label=label,
        coverage=_propagate(propagator, OperationName.FILTER, [base], [BASE_ROLE]),
    )


def deduplicated(
    store: HandleStore,
    base_id: str,
    *,
    members: Sequence[Paper],
    strategy: str,
    label: str | None = None,
    propagator: CoveragePropagator | None = None,
) -> HandleRecord:
    """Record the result of collapsing duplicates within one set.

    The deduplication itself -- exact identifier matching, then a semantic pass
    over the residue -- is computed elsewhere and its finished member list is
    passed in.  This function's job is to record it truthfully.

    A ranked base yields an **invalidated** result.  Collapsing two records
    produces a record whose text is not the text that was scored, so carrying
    the old score forward would attach a measurement to an object that no
    longer exists.  In the ordinary workflow deduplication precedes
    re-ranking, so this costs nothing.

    Parameters
    ----------
    store : HandleStore
        Store holding the base handle.
    base_id : str
        Full identifier of the set that was deduplicated.
    members : Sequence[Paper]
        The surviving members.  Each must preserve a distinct base member by
        explicit identity or exact immutable snapshot.
    strategy : str
        How duplicates were identified, for example
        ``"exact-doi then model2vec cosine >= 0.92"``.
    label : str | None
        Optional human-readable name.
    propagator : CoveragePropagator | None
        Coverage propagator.

    Returns
    -------
    HandleRecord
        The deduplicated handle.

    Raises
    ------
    InvalidParameterError
        If *strategy* is empty, the member count grew, or an output cannot be
        matched one-to-one to the base.
    """
    if not strategy.strip():
        raise InvalidParameterError("deduplicate requires a non-empty strategy description")
    base = store.get(base_id)
    if len(members) > len(base.members):
        raise InvalidParameterError(
            f"deduplicate returned {len(members)} member(s) from a base of "
            f"{len(base.members)}; deduplication cannot add members"
        )
    if not _has_identity_subset(base.members, members):
        raise InvalidParameterError(
            f"deduplicate of {base_id} returned a member that was not preserved from "
            "its base; deduplication may remove or enrich identified records, but it "
            "cannot substitute unrelated papers"
        )
    return HandleRecord(
        operation=Operation(
            name=OperationName.DEDUPLICATE,
            operands=(Operand.of_handle(base_id, "base"),),
            parameters={"strategy": strategy},
        ),
        order=base.order.invalidate(OperationName.DEDUPLICATE.value),
        members=tuple(members),
        label=label,
        coverage=_propagate(propagator, OperationName.DEDUPLICATE, [base], [BASE_ROLE]),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_operands(
    store: HandleStore, handle_ids: Sequence[str], *, minimum: int, operation: str
) -> list[HandleRecord]:
    """Load operand handles, refusing an under-specified operation.

    Parameters
    ----------
    store : HandleStore
        Store holding the operands.
    handle_ids : Sequence[str]
        Full identifiers, in operand order.
    minimum : int
        Fewest operands the operation is defined for.
    operation : str
        Operation name, used in the error message.

    Returns
    -------
    list[HandleRecord]
        Loaded handles, in operand order.

    Raises
    ------
    InvalidParameterError
        If fewer than *minimum* operands are given.
    """
    if len(handle_ids) < minimum:
        raise InvalidParameterError(
            f"{operation} needs at least {minimum} operands, got {len(handle_ids)}"
        )
    return [store.get(handle_id) for handle_id in handle_ids]


def _union_order(
    records: Sequence[HandleRecord],
    members: Sequence[Paper],
    origin: Sequence[int],
) -> OrderRecord:
    """Decide what order, if any, a union may claim.

    Parameters
    ----------
    records : Sequence[HandleRecord]
        Operand handles, in operand order.
    members : Sequence[Paper]
        Union members, in first-seen order.
    origin : Sequence[int]
        Index of the operand each member was first seen in.
    Returns
    -------
    OrderRecord
        Ranked when every operand is ranked under mergeable provenance,
        invalidated when any operand was ranked, unordered otherwise.
    """
    orders = [record.order for record in records]
    if all(order.state is OrderState.UNORDERED for order in orders):
        return OrderRecord.unordered()

    if not all(order.state is OrderState.RANKED for order in orders):
        return _first_ranked(orders).invalidate(OperationName.UNION.value)

    reference = _expect_ranking(orders[0])
    for order in orders[1:]:
        if not reference.mergeable_with(_expect_ranking(order)):
            return orders[0].invalidate(OperationName.UNION.value)

    scores = [0.0] * len(members)
    for index, record in enumerate(records):
        record_scores = record.order.scores
        if record_scores is None:  # pragma: no cover - guarded by the state check above
            continue
        for member_index, member in enumerate(record.members):
            position = _identity_index(members, member)
            if position is None:  # pragma: no cover - members were built from these records
                continue
            # Operand order is significant: the earliest operand holding a
            # member supplies its score, matching first-seen-wins membership.
            if origin[position] == index:
                scores[position] = record_scores[member_index]
    return OrderRecord.ranked(reference, scores)


def _first_ranked(orders: Iterable[OrderRecord]) -> OrderRecord:
    """Return the first order carrying or having carried a ranking claim.

    Parameters
    ----------
    orders : Iterable[OrderRecord]
        Operand orders, in operand order.

    Returns
    -------
    OrderRecord
        The first non-unordered order, or an unordered record when there is
        none.
    """
    for order in orders:
        if order.state is not OrderState.UNORDERED:
            return order
    return OrderRecord.unordered()


def _expect_ranking(order: OrderRecord) -> RankingProvenance:
    """Return a ranked order's provenance, asserting that it has one.

    Parameters
    ----------
    order : OrderRecord
        Order expected to be ranked.

    Returns
    -------
    RankingProvenance
        The provenance.

    Raises
    ------
    OrderSemanticsError
        If the order is not ranked.
    """
    from lit_search.exceptions import OrderSemanticsError

    if order.ranking is None:
        raise OrderSemanticsError("Expected a ranked order to carry provenance")
    return order.ranking


def _require_same_membership(
    base_members: Sequence[Paper], new_members: Sequence[Paper], base_id: str
) -> None:
    """Refuse a re-rank that changed which papers are in the set.

    Parameters
    ----------
    base_members : Sequence[Paper]
        Members of the base handle.
    new_members : Sequence[Paper]
        Members returned by the re-ranker.
    base_id : str
        Base identifier, used in the error message.

    Returns
    -------
    None

    Raises
    ------
    InvalidParameterError
        If the two member collections are not the same set.
    """
    if len(base_members) != len(new_members):
        raise InvalidParameterError(
            f"rerank of {base_id} returned {len(new_members)} member(s) but its base "
            f"holds {len(base_members)}; a re-rank reorders a set, it does not choose a "
            "subset of one.  Compose rerank with filter to take a top-k, so that the "
            "truncation appears in the lineage."
        )
    if not _has_identity_permutation(base_members, new_members):
        raise InvalidParameterError(
            f"rerank of {base_id} returned a different set of papers than the base; a "
            "re-rank must be a permutation of its input"
        )


def _identity_index(members: Sequence[Paper], candidate: Paper) -> int | None:
    """Return the first member with the same explicit record identity.

    Membership algebra must not inherit :class:`Paper`'s object equality or
    hashing semantics.  In particular, title equality is evidence for a
    possible relationship, not proof that two source records are one member.
    A linear scan is intentional: the identity relation can involve multiple
    provider aliases, so no single hash key faithfully represents it.
    """
    return next(
        (index for index, member in enumerate(members) if same_record_identity(member, candidate)),
        None,
    )


def _has_identity_permutation(base_members: Sequence[Paper], new_members: Sequence[Paper]) -> bool:
    """Return whether *new_members* is a one-to-one identity permutation.

    A simple ``all(any(...))`` check is insufficient when an input contains
    repeated or overlapping aliases: it can match two outputs to one base
    member.  This small bipartite matching keeps multiplicity exact while
    allowing a separately deserialized record carrying the same DOI or native
    provider ID, or an exact immutable snapshot loaded independently from the
    same handle, to stand for the same member.
    """
    if len(base_members) != len(new_members):
        return False

    return _has_identity_subset(base_members, new_members)


def _has_identity_subset(base_members: Sequence[Paper], new_members: Sequence[Paper]) -> bool:
    """Return whether outputs consume distinct identity-matching base members.

    This is the membership-preservation rule shared by rerank and the generic
    deduplication recorder.  Rerank separately requires equal lengths;
    deduplication permits fewer members, but neither operation may use one base
    position to justify repeated outputs.
    """
    if len(new_members) > len(base_members):
        return False

    matched_new_by_base: list[int | None] = [None] * len(base_members)

    def assign(new_index: int, visited_base: set[int]) -> bool:
        for base_index, base_member in enumerate(base_members):
            if base_index in visited_base or not _same_preserved_member(
                base_member, new_members[new_index]
            ):
                continue
            visited_base.add(base_index)
            previous_new = matched_new_by_base[base_index]
            if previous_new is None or assign(previous_new, visited_base):
                matched_new_by_base[base_index] = new_index
                return True
        return False

    return all(assign(new_index, set()) for new_index in range(len(new_members)))


def _same_preserved_member(left: Paper, right: Paper) -> bool:
    """Return whether a preserving operation may treat two records as one.

    Exact record identity remains the primary rule.  Rerank and the generic
    deduplication recorder also admit an exact serialized snapshot because
    their contracts are narrower than general set algebra: the caller declares
    that outputs are members preserved from one specific immutable base, and
    caller and operation load those snapshots independently from the handle
    store.  This exception is deliberately not used by union, difference, or
    intersection.
    """
    return same_record_identity(left, right) or left.to_dict() == right.to_dict()
