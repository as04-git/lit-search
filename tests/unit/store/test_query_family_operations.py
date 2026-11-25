"""Immutable handle lineage for executed query-family plans."""

from __future__ import annotations

from dataclasses import replace

import pytest

from lit_search.core.coverage import CoverageLedger, CoverageScope, SourceCoverage
from lit_search.core.paper import Paper
from lit_search.core.query_family import Evidence, EvidenceKind, QueryFamilyRevision
from lit_search.core.query_family_result import (
    QueryCellOutcome,
    QueryFamilyResult,
    union_cell_papers,
)
from lit_search.core.search_result import SearchResult
from lit_search.exceptions import InvalidParameterError
from lit_search.query.family_compiler import plan_query_family
from lit_search.query.family_evidence import field_evidence_digest, member_evidence_digest
from lit_search.store import operations as ops
from lit_search.store.lineage import OperandKind, OperationName
from lit_search.store.order import OrderState
from lit_search.store.record import HandleRecord
from tests.unit.core.test_query_family import _revision


def _paper(title: str, doi: str) -> Paper:
    return Paper(
        title=title,
        abstract="",
        authors=[],
        source=None,
        publication_date=None,
        doi=doi,
    )


def _family_result(
    revision: QueryFamilyRevision,
    sources: tuple[str, ...] = ("arxiv", "openalex"),
) -> QueryFamilyResult:
    plan = plan_query_family(revision, list(sources))
    outcomes: list[QueryCellOutcome] = []
    for index, cell in enumerate(plan.cells):
        paper = _paper(
            f"{cell.source} result",
            f"10.1234/{index}-{cell.source}",
        )
        result = SearchResult(
            query=cell.query_text,
            databases=[cell.source],
            papers=[paper],
            enrichment_databases=[],
            coverage=CoverageLedger(
                entries=(
                    SourceCoverage.complete(
                        cell.source,
                        1,
                        reported_total=1,
                    ),
                ),
                scope=CoverageScope(
                    profile="query-family-cell",
                    selected_sources=(cell.source,),
                    registered_sources=(cell.source,),
                ),
            ),
        )
        outcomes.append(QueryCellOutcome(cell=cell, result=result))

    ordered = tuple(sorted(outcomes, key=lambda outcome: outcome.cell.cell_id))
    family_entries = tuple(
        SourceCoverage.complete(source, 1, reported_total=1) for source in plan.selected_sources
    )
    if not plan.cells:
        family_entries = tuple(
            SourceCoverage.skipped_by_policy(
                source,
                detail="all declared variants exclude this source",
            )
            for source in plan.selected_sources
        )
    return QueryFamilyResult(
        plan=plan,
        cell_outcomes=ordered,
        papers=union_cell_papers(ordered),
        coverage=CoverageLedger(
            entries=family_entries,
            scope=CoverageScope(
                profile="query-family",
                selected_sources=plan.selected_sources,
                registered_sources=plan.selected_sources,
            ),
        ),
    )


def _cell_handles(result: QueryFamilyResult) -> list[tuple[str, HandleRecord]]:
    records = [
        ops.from_query_cell_outcome(outcome, plan_id=result.plan_id)
        for outcome in result.cell_outcomes
    ]
    return [(record.handle_id, record) for record in records]


def test_query_cell_search_handle_pins_plan_translation_and_exact_result() -> None:
    """A cell remains an ordinary search whose family decision is explicit."""
    result = _family_result(_revision(), ("openalex",))
    outcome = result.cell_outcomes[0]

    record = ops.from_query_cell_outcome(outcome, plan_id=result.plan_id)

    assert record.operation.name is OperationName.SEARCH
    assert record.order.state is OrderState.UNORDERED
    assert record.members == tuple(outcome.result.papers)
    assert record.coverage == outcome.result.coverage.to_dict()
    assert record.operation.parameters == {
        "query": outcome.cell.query_text,
        "databases": ["openalex"],
        "since": None,
        "until": None,
        "max_papers_per_database": None,
        "enrichment_databases": [],
        "max_cited_by": None,
        "recorded_failed_databases": [],
        "revision_id": result.revision_id,
        "plan_id": result.plan_id,
        "cell_id": outcome.cell.cell_id,
        "variant_id": outcome.cell.variant_id,
        "source": "openalex",
        "query_digest": outcome.cell.query_digest,
        "translation_id": "st_openalex",
        "translation_mode": "degraded",
        "degradation_reason": "OpenAlex lacks the requested controlled-vocabulary facet.",
    }


def test_family_handle_pins_exact_plan_cells_members_scope_and_coverage() -> None:
    """The aggregate is an auditable plan result, not an opaque union."""
    revision = _revision()
    result = _family_result(revision)
    cell_handles = _cell_handles(result)

    record = ops.from_query_family_result(
        result,
        cell_handles=cell_handles,
        revision=revision,
    )

    assert record.operation.name is OperationName.QUERY_FAMILY
    assert record.order.state is OrderState.UNORDERED
    assert record.members == result.papers
    assert record.coverage == result.coverage.to_dict()
    assert record.operation.parameters == {
        "revision_id": revision.revision_id,
        "plan_id": result.plan_id,
        "selected_sources": list(result.selected_sources),
        "scope_exclusions": [],
        "cell_ids": [outcome.cell.cell_id for outcome in result.cell_outcomes],
        "cell_handle_ids": [handle_id for handle_id, _ in cell_handles],
    }
    assert [operand.role for operand in record.operation.operands] == [
        "query_cell",
        "query_cell",
    ]


def test_family_lineage_reaches_each_unique_handle_member_evidence(store, make_record) -> None:
    """Vocabulary learned from a paper remains walkable from family output."""
    evidence_paper = _paper("Model checking vocabulary", "10.1234/evidence")
    evidence_handle = store.put(make_record([evidence_paper]))
    evidence = (
        Evidence(
            "e_paper_title",
            EvidenceKind.HANDLE_MEMBER,
            "The paper names the neighboring community's terminology.",
            handle_id=evidence_handle,
            member_digest=member_evidence_digest(evidence_paper),
            field="title",
            field_value_digest=field_evidence_digest(evidence_paper.to_dict()["title"]),
        ),
        Evidence(
            "e_same_paper_abstract",
            EvidenceKind.HANDLE_MEMBER,
            "A second field from the same immutable evidence handle.",
            handle_id=evidence_handle,
            member_digest=member_evidence_digest(evidence_paper),
            field="abstract",
            field_value_digest=field_evidence_digest(evidence_paper.to_dict()["abstract"]),
        ),
    )
    base = _revision()
    revision = replace(base, evidence=(*base.evidence, *evidence))
    result = _family_result(revision, ("arxiv",))
    cell_pairs = _cell_handles(result)
    cell_ids = [store.put(record) for _, record in cell_pairs]
    stored_pairs = list(zip(cell_ids, (record for _, record in cell_pairs), strict=True))

    family_id = store.put(
        ops.from_query_family_result(
            result,
            cell_handles=stored_pairs,
            revision=revision,
        )
    )

    graph = store.lineage(family_id)
    roles = [(edge.parent, edge.role) for edge in graph.edges if edge.child == family_id]
    assert roles == [(cell_ids[0], "query_cell"), (evidence_handle, "query_evidence")]
    assert evidence_handle in graph.nodes
    assert sum(parent == evidence_handle for parent, _ in roles) == 1


@pytest.mark.parametrize("mutation", ["swapped", "wrong_id", "members", "coverage"])
def test_family_refuses_mismatched_cell_handles(mutation: str) -> None:
    """An unrelated or reordered search handle cannot masquerade as a cell."""
    revision = _revision()
    result = _family_result(revision)
    cell_handles = _cell_handles(result)

    if mutation == "swapped":
        supplied = list(reversed(cell_handles))
        message = "mismatch"
    elif mutation == "wrong_id":
        supplied = [(cell_handles[1][0], cell_handles[0][1]), cell_handles[1]]
        message = "pairs identifier"
    elif mutation == "members":
        _, original = cell_handles[0]
        tampered = replace(
            original,
            members=(_paper("Unrelated paper", "10.1234/unrelated"),),
        )
        supplied = [(tampered.handle_id, tampered), cell_handles[1]]
        message = "members"
    else:
        _, original = cell_handles[0]
        tampered = replace(original, coverage={"status": "forged"})
        supplied = [(tampered.handle_id, tampered), cell_handles[1]]
        message = "coverage"

    with pytest.raises(InvalidParameterError, match=message):
        ops.from_query_family_result(
            result,
            cell_handles=supplied,
            revision=revision,
        )


def test_family_refuses_missing_or_extra_cell_handles() -> None:
    """Every applicable cell has exactly one stored cell handle."""
    revision = _revision()
    result = _family_result(revision, ("arxiv",))
    cell_handles = _cell_handles(result)

    with pytest.raises(InvalidParameterError, match="no cell handles"):
        ops.from_query_family_result(result, cell_handles=[], revision=revision)
    with pytest.raises(InvalidParameterError, match="count mismatch"):
        ops.from_query_family_result(
            result,
            cell_handles=[*cell_handles, *cell_handles],
            revision=revision,
        )


def test_zero_applicable_cell_plan_allows_zero_cell_operands() -> None:
    """A fully scoped-out plan is explicit and does not invent a search cell."""
    base = _revision()
    revision = replace(
        base,
        variants=(replace(base.variants[0], applies_to_sources=("openalex",)),),
    )
    result = _family_result(revision, ("arxiv",))

    record = ops.from_query_family_result(
        result,
        cell_handles=[],
        revision=revision,
    )

    assert record.operation.name is OperationName.QUERY_FAMILY
    assert record.operation.operands == ()
    assert record.members == ()
    assert record.operation.parameters["cell_ids"] == []


def test_family_refuses_a_different_revision_even_if_shape_matches() -> None:
    """Revision provenance cannot be swapped after execution."""
    revision = _revision()
    result = _family_result(revision, ("arxiv",))
    different = replace(revision, research_question="A different research question")

    with pytest.raises(InvalidParameterError, match="revision mismatch"):
        ops.from_query_family_result(
            result,
            cell_handles=_cell_handles(result),
            revision=different,
        )


def test_query_family_operation_round_trips_through_closed_lineage_schema() -> None:
    """The new operation name is accepted by strict operation serialization."""
    revision = _revision()
    result = _family_result(revision, ("arxiv",))
    record = ops.from_query_family_result(
        result,
        cell_handles=_cell_handles(result),
        revision=revision,
    )

    restored = HandleRecord.from_dict(record.to_dict())

    assert restored.operation.name is OperationName.QUERY_FAMILY
    assert all(operand.kind is OperandKind.HANDLE for operand in restored.operation.operands)
