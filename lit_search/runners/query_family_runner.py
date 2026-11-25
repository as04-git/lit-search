"""Execute an immutable query family as explicit source-by-variant cells."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from lit_search.core.coverage import (
    CoverageLedger,
    CoverageScope,
    CoverageStatus,
    DerivationNote,
    ExecutionStatus,
    SourceCoverage,
    union_coverage,
)
from lit_search.core.query_family import QueryFamilyRevision
from lit_search.core.query_family_result import (
    QueryCellOutcome,
    QueryFamilyResult,
    union_cell_papers,
)
from lit_search.core.search_result import SearchResult
from lit_search.query.family_compiler import QueryFamilyPlan, plan_query_family
from lit_search.runners.search_runner import SearchRunner


class QueryFamilyExecutionError(ValueError):
    """A supplied plan or executed cell violates the family contract."""


class _ExecutableSearch(Protocol):
    def run(self, verbose: bool = False, show_progress: bool = True) -> SearchResult:
        """Execute one scoped search."""


SearchRunnerFactory = Callable[..., _ExecutableSearch]
EvidenceVerifier = Callable[[QueryFamilyRevision], bool | None]


class QueryFamilyRunner:
    """Run every applicable cell while retaining per-cell operational truth.

    Search cells are deliberately sequential in this first execution layer.
    A source may impose a shared rate limit across variants, and hiding cell
    concurrency inside this class would make quota behaviour hard to audit.
    Source-level scheduling can be layered above this deterministic contract.
    """

    def __init__(
        self,
        revision: QueryFamilyRevision,
        sources: Sequence[str],
        *,
        plan: QueryFamilyPlan | None = None,
        search_runner_factory: SearchRunnerFactory = SearchRunner,
        search_runner_kwargs: Mapping[str, Any] | None = None,
        evidence_verifier: EvidenceVerifier | None = None,
    ) -> None:
        """Validate evidence, compile the plan, and freeze execution inputs.

        ``evidence_verifier`` is intentionally store-agnostic.  A caller may
        verify handle/member digests there; an exception or ``False`` aborts
        before planning and therefore before any source runner is created.
        """
        if evidence_verifier is not None:
            verified = evidence_verifier(revision)
            if verified is False:
                raise QueryFamilyExecutionError("query-family evidence verification failed")

        expected = plan_query_family(revision, list(sources))
        if plan is not None:
            if plan.revision_id != revision.revision_id:
                raise QueryFamilyExecutionError(
                    "supplied plan belongs to a different query-family revision"
                )
            if plan.plan_id != expected.plan_id:
                raise QueryFamilyExecutionError(
                    "supplied plan does not match a fresh compilation of this revision and scope"
                )
        self._plan = expected if plan is None else plan
        self._search_runner_factory = search_runner_factory
        kwargs = dict(search_runner_kwargs or {})
        forbidden = {"query", "databases"}.intersection(kwargs)
        if forbidden:
            names = ", ".join(sorted(forbidden))
            raise QueryFamilyExecutionError(
                f"query-family execution owns {names}; do not override it in search_runner_kwargs"
            )
        # Per-cell enrichment duplicates network work and obscures provenance.
        # A later family-level operation may enrich the union exactly once.
        kwargs.setdefault("enrichment_databases", [])
        self._search_runner_kwargs = kwargs

    @property
    def plan(self) -> QueryFamilyPlan:
        """Return the immutable, content-addressed execution plan."""
        return self._plan

    def run(
        self,
        *,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> QueryFamilyResult:
        """Execute all applicable cells in deterministic cell-id order."""
        outcomes: list[QueryCellOutcome] = []
        for cell in sorted(self._plan.cells, key=lambda item: item.cell_id):
            runner = self._search_runner_factory(
                query=cell.query_text,
                databases=[cell.source],
                **self._search_runner_kwargs,
            )
            search_result = runner.run(verbose=verbose, show_progress=show_progress)
            try:
                outcomes.append(QueryCellOutcome(cell=cell, result=search_result))
            except ValueError as error:
                raise QueryFamilyExecutionError(str(error)) from error

        frozen_outcomes = tuple(outcomes)
        coverage = self._aggregate_coverage(frozen_outcomes)
        return QueryFamilyResult(
            plan=self._plan,
            cell_outcomes=frozen_outcomes,
            papers=union_cell_papers(frozen_outcomes),
            coverage=coverage,
        )

    def _aggregate_coverage(
        self,
        outcomes: tuple[QueryCellOutcome, ...],
    ) -> CoverageLedger:
        if not outcomes:
            return CoverageLedger(
                entries=tuple(
                    SourceCoverage.skipped_by_policy(
                        source,
                        detail="every query variant excluded this source by declared scope",
                    )
                    for source in self._plan.selected_sources
                ),
                derivations=(
                    DerivationNote(
                        operation="query_family",
                        role="applicable cells",
                        status=CoverageStatus.COMPLETE,
                        detail="the declared plan contained no applicable query cells",
                        execution_status=ExecutionStatus.COMPLETE,
                    ),
                ),
                scope=CoverageScope(
                    profile=f"query-family:{self._plan.plan_id}",
                    selected_sources=self._plan.selected_sources,
                    registered_sources=self._plan.selected_sources,
                ),
            )

        merged = union_coverage(*(outcome.result.coverage for outcome in outcomes))
        registered_sources = (
            merged.scope.registered_sources
            if merged.scope is not None
            else self._plan.selected_sources
        )
        registered_sources = tuple(
            dict.fromkeys((*registered_sources, *self._plan.selected_sources))
        )
        scope = CoverageScope(
            profile=f"query-family:{self._plan.plan_id}",
            selected_sources=self._plan.selected_sources,
            registered_sources=registered_sources,
        )
        entries = list(merged.entries)
        sources_with_cells = {outcome.cell.source for outcome in outcomes}
        for source in self._plan.selected_sources:
            if source not in sources_with_cells:
                entries.append(
                    SourceCoverage.skipped_by_policy(
                        source,
                        detail="every query variant excluded this source by declared scope",
                    )
                )
        return CoverageLedger(
            entries=tuple(entries),
            derivations=merged.derivations,
            scope=scope,
        )


__all__ = [
    "EvidenceVerifier",
    "QueryFamilyExecutionError",
    "QueryFamilyRunner",
    "SearchRunnerFactory",
]
