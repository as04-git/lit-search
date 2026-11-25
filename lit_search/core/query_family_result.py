"""Results from executing one immutable query-family plan.

The family result does not flatten its constituent searches into an opaque
bag.  Every query cell retains the exact :class:`SearchResult` that produced
it, including that result's coverage ledger, while :attr:`papers` is a stable
first-seen union for ordinary downstream consumption.

The representation is write-only at this layer.  Durable replay is owned by
immutable result handles, which can verify the stored plan and member digests
before reconstructing an executable object; this module does not invent a
second, weaker persistence authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lit_search.core.coverage import CoverageLedger, CoverageStatus
from lit_search.core.identity import same_record_identity
from lit_search.core.paper import Paper
from lit_search.core.search_result import SearchResult
from lit_search.query.family_compiler import QueryCell, QueryFamilyPlan, ScopeExclusion

QUERY_FAMILY_RESULT_SCHEMA = "lit-search/query-family-result"
QUERY_FAMILY_RESULT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class QueryCellOutcome:
    """The auditable result of executing one planned query cell."""

    cell: QueryCell
    result: SearchResult

    def __post_init__(self) -> None:
        """Refuse a result that cannot have come from the declared cell."""
        if self.result.query != self.cell.query_text:
            raise ValueError(
                f"cell {self.cell.cell_id} expected query {self.cell.query_text!r}, "
                f"but the result records {self.result.query!r}"
            )
        scope = self.result.coverage.scope
        if scope is None or tuple(scope.selected_sources) != (self.cell.source,):
            actual = None if scope is None else tuple(scope.selected_sources)
            raise ValueError(
                f"cell {self.cell.cell_id} must carry exact source scope "
                f"({self.cell.source!r},), got {actual!r}"
            )
        if tuple(self.result.databases or ()) != (self.cell.source,):
            raise ValueError(
                f"cell {self.cell.cell_id} must record databases "
                f"[{self.cell.source!r}], got {self.result.databases!r}"
            )
        applicable_entries = {
            entry.source
            for entry in self.result.coverage.entries
            if entry.status is not CoverageStatus.SKIPPED_BY_POLICY
        }
        if applicable_entries != {self.cell.source}:
            raise ValueError(
                f"cell {self.cell.cell_id} must carry one applicable coverage source "
                f"{self.cell.source!r}, got {sorted(applicable_entries)!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete, JSON-ready cell outcome."""
        return {
            "cell": self.cell.to_dict(),
            "result": self.result.to_dict(),
        }


def union_cell_papers(outcomes: tuple[QueryCellOutcome, ...]) -> tuple[Paper, ...]:
    """Return a deterministic union using only explicit record identity.

    Identifier-free records are deliberately retained unless they are the
    very same object.  A title resemblance is evidence for later work/version
    reasoning, never permission for this execution layer to erase a record.
    """
    papers: list[Paper] = []
    for outcome in outcomes:
        for paper in outcome.result.papers:
            if not any(same_record_identity(existing, paper) for existing in papers):
                papers.append(paper)
    return tuple(papers)


@dataclass(frozen=True, slots=True)
class QueryFamilyResult:
    """Deterministically ordered result of one query-family plan."""

    plan: QueryFamilyPlan
    cell_outcomes: tuple[QueryCellOutcome, ...]
    papers: tuple[Paper, ...]
    coverage: CoverageLedger

    def __post_init__(self) -> None:
        """Validate ordering and content-address linkages."""
        expected_outcomes = tuple(
            sorted(self.cell_outcomes, key=lambda outcome: outcome.cell.cell_id)
        )
        if self.cell_outcomes != expected_outcomes:
            raise ValueError("cell outcomes must be sorted by deterministic cell_id")
        if tuple(outcome.cell for outcome in self.cell_outcomes) != tuple(
            sorted(self.plan.cells, key=lambda cell: cell.cell_id)
        ):
            raise ValueError("cell outcomes must correspond exactly to the supplied plan")
        if any(outcome.cell.revision_id != self.revision_id for outcome in self.cell_outcomes):
            raise ValueError("a cell outcome belongs to a different query-family revision")
        if self.papers != union_cell_papers(self.cell_outcomes):
            raise ValueError("papers must be the explicit-identity union of cell outcomes")

    @property
    def revision_id(self) -> str:
        """Return the immutable revision content address."""
        return self.plan.revision_id

    @property
    def plan_id(self) -> str:
        """Return the immutable execution-plan content address."""
        return self.plan.plan_id

    @property
    def selected_sources(self) -> tuple[str, ...]:
        """Return the family-level declared source boundary."""
        return self.plan.selected_sources

    @property
    def scope_exclusions(self) -> tuple[ScopeExclusion, ...]:
        """Return variant/source cells excluded by the declared plan."""
        return self.plan.scope_exclusions

    @property
    def execution_is_complete(self) -> bool:
        """Whether every applicable query cell completed operationally."""
        return self.coverage.execution_is_complete

    def to_dict(self) -> dict[str, Any]:
        """Return a strict, versioned, JSON-ready handle payload."""
        return {
            "schema": QUERY_FAMILY_RESULT_SCHEMA,
            "schema_version": QUERY_FAMILY_RESULT_SCHEMA_VERSION,
            "revision_id": self.revision_id,
            "plan_id": self.plan_id,
            "plan": self.plan.to_dict(),
            "selected_sources": list(self.selected_sources),
            "scope_exclusions": [item.to_dict() for item in self.scope_exclusions],
            "cell_outcomes": [outcome.to_dict() for outcome in self.cell_outcomes],
            "papers": [paper.to_dict() for paper in self.papers],
            "coverage": self.coverage.to_dict(),
        }


__all__ = [
    "QUERY_FAMILY_RESULT_SCHEMA",
    "QUERY_FAMILY_RESULT_SCHEMA_VERSION",
    "QueryCellOutcome",
    "QueryFamilyResult",
    "union_cell_papers",
]
