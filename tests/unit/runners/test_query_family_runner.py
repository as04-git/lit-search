"""Execution-contract tests for the first-class query-family runner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest

from lit_search.core.citation import IdentifierKind, PaperIdentifier
from lit_search.core.coverage import (
    CoverageLedger,
    CoverageScope,
    ExecutionStatus,
    SourceCoverage,
)
from lit_search.core.paper import Paper
from lit_search.core.query_family import AllOf, TermRef
from lit_search.core.query_family_result import QUERY_FAMILY_RESULT_SCHEMA
from lit_search.core.search_result import SearchResult
from lit_search.query.family_compiler import plan_query_family
from lit_search.runners.query_family_runner import (
    QueryFamilyExecutionError,
    QueryFamilyRunner,
)
from tests.unit.core.test_query_family import _revision


def _paper(title: str, doi: str | None = None) -> Paper:
    return Paper(
        title=title,
        abstract="",
        authors=[],
        source=None,
        publication_date=None,
        doi=doi,
    )


def _result(
    query: str,
    source: str,
    papers: list[Paper],
    *,
    coverage: SourceCoverage | None = None,
    selected_sources: tuple[str, ...] | None = None,
) -> SearchResult:
    entry = coverage or SourceCoverage.complete(
        source,
        len(papers),
        reported_total=len(papers),
    )
    selected = selected_sources if selected_sources is not None else (source,)
    registered = tuple(dict.fromkeys((*selected, source)))
    return SearchResult(
        query=query,
        databases=[source],
        papers=papers,
        enrichment_databases=[],
        coverage=CoverageLedger(
            entries=(entry,),
            scope=CoverageScope(
                profile="explicit",
                selected_sources=selected,
                registered_sources=registered,
            ),
        ),
    )


class _FakeSearch:
    def __init__(self, result: SearchResult) -> None:
        self._result = result

    def run(self, verbose: bool = False, show_progress: bool = True) -> SearchResult:
        del verbose, show_progress
        return self._result


class _Factory:
    def __init__(
        self,
        make_result: Callable[[str, str], SearchResult],
    ) -> None:
        self._make_result = make_result
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> _FakeSearch:
        self.calls.append(kwargs)
        query = str(kwargs["query"])
        source = str(kwargs["databases"][0])
        return _FakeSearch(self._make_result(query, source))


def test_executes_exact_scoped_cells_and_unions_only_explicit_identity() -> None:
    """A stable plan produces exact one-source calls and a non-fuzzy union."""
    duplicate_left = _paper("Shared DOI from arXiv", "10.1000/shared")
    duplicate_right = _paper("Shared DOI from OpenAlex", "https://doi.org/10.1000/SHARED")
    same_title_left = _paper("Generic title")
    same_title_right = _paper("Generic title")

    def make_result(query: str, source: str) -> SearchResult:
        papers = (
            [duplicate_left, same_title_left]
            if source == "arxiv"
            else [duplicate_right, same_title_right]
        )
        return _result(query, source, papers)

    factory = _Factory(make_result)
    runner = QueryFamilyRunner(
        _revision(),
        ["openalex", "arxiv"],
        search_runner_factory=factory,
    )
    result = runner.run(show_progress=False)

    assert [outcome.cell.cell_id for outcome in result.cell_outcomes] == sorted(
        outcome.cell.cell_id for outcome in result.cell_outcomes
    )
    assert {call["databases"][0] for call in factory.calls} == {"arxiv", "openalex"}
    assert all(call["enrichment_databases"] == [] for call in factory.calls)
    assert len(result.papers) == 3
    assert sum(paper.title == "Generic title" for paper in result.papers) == 2
    assert result.execution_is_complete


def test_family_union_unversioned_alias_cannot_bridge_conflicting_versions() -> None:
    """Cell order cannot let unknown provider metadata erase DOI versions."""
    shared = PaperIdentifier(IdentifierKind.SEMANTIC_SCHOLAR, "shared")
    papers = [
        _paper("Unversioned"),
        _paper("First", "10.1000/first"),
        _paper("Second", "10.1000/second"),
    ]
    for paper in papers:
        paper.identifiers = [shared]
    calls = 0

    def make_result(query: str, source: str) -> SearchResult:
        nonlocal calls
        paper = papers[calls]
        calls += 1
        return _result(query, source, [paper])

    revision = _revision()
    result = QueryFamilyRunner(
        revision,
        ["semantic_scholar", "openalex", "arxiv"],
        search_runner_factory=_Factory(make_result),
    ).run(show_progress=False)

    assert len(result.papers) == 3


def test_family_union_published_alias_cannot_erase_preprint_version() -> None:
    """A cell's DOI+arXiv metadata cannot bridge two publication versions."""
    published = _paper("Published", "10.1000/published")
    published.identifiers.append(PaperIdentifier(IdentifierKind.ARXIV, "2601.00001"))
    doi_only = _paper("Published DOI", "10.1000/published")
    arxiv_only = _paper("Preprint")
    arxiv_only.identifiers = [PaperIdentifier(IdentifierKind.ARXIV, "2601.00001")]
    papers = [published, doi_only, arxiv_only]
    calls = 0

    def make_result(query: str, source: str) -> SearchResult:
        nonlocal calls
        paper = papers[calls]
        calls += 1
        return _result(query, source, [paper])

    result = QueryFamilyRunner(
        _revision(),
        ["semantic_scholar", "openalex", "arxiv"],
        search_runner_factory=_Factory(make_result),
    ).run(show_progress=False)

    assert len(result.papers) == 2
    assert {paper.doi for paper in result.papers} == {"10.1000/published", None}


def test_scope_exclusion_is_visible_but_not_an_execution_failure() -> None:
    """A variant-level source exclusion is not laundered into a failed cell."""
    revision = _revision()
    scoped_variant = replace(revision.variants[0], applies_to_sources=("openalex",))
    scoped_revision = replace(revision, variants=(scoped_variant,))
    factory = _Factory(lambda query, source: _result(query, source, []))

    result = QueryFamilyRunner(
        scoped_revision,
        ["arxiv", "openalex"],
        search_runner_factory=factory,
    ).run(show_progress=False)

    assert len(factory.calls) == 1
    assert factory.calls[0]["databases"] == ["openalex"]
    assert [(item.variant_id, item.source) for item in result.scope_exclusions] == [
        ("v_main", "arxiv")
    ]
    assert result.coverage.get("arxiv") is not None
    assert result.coverage.get("arxiv").execution_status is ExecutionStatus.NOT_APPLICABLE
    assert result.coverage.execution_status is ExecutionStatus.COMPLETE


def test_fully_excluded_plan_has_explicit_not_applicable_sources_and_no_calls() -> None:
    """A vacuous plan remains auditable and operationally complete."""
    revision = _revision()
    scoped_variant = replace(revision.variants[0], applies_to_sources=("openalex",))
    scoped_revision = replace(revision, variants=(scoped_variant,))
    factory = _Factory(lambda query, source: _result(query, source, []))

    result = QueryFamilyRunner(
        scoped_revision,
        ["arxiv"],
        search_runner_factory=factory,
    ).run(show_progress=False)

    assert factory.calls == []
    assert result.cell_outcomes == ()
    assert result.coverage.get("arxiv") is not None
    assert result.coverage.get("arxiv").execution_status is ExecutionStatus.NOT_APPLICABLE
    assert result.coverage.execution_status is ExecutionStatus.COMPLETE


def test_cell_failure_survives_in_cell_and_family_coverage() -> None:
    """A source failure remains attributable after family-level unioning."""
    failure = SourceCoverage.failed(
        "openalex",
        error="upstream timeout",
        error_type="TimeoutError",
    )
    factory = _Factory(
        lambda query, source: _result(
            query,
            source,
            [],
            coverage=failure if source == "openalex" else None,
        )
    )
    result = QueryFamilyRunner(
        _revision(),
        ["arxiv", "openalex"],
        search_runner_factory=factory,
    ).run(show_progress=False)

    openalex = next(
        outcome for outcome in result.cell_outcomes if outcome.cell.source == "openalex"
    )
    assert openalex.result.coverage.execution_status is ExecutionStatus.INCOMPLETE
    assert result.coverage.execution_status is ExecutionStatus.INCOMPLETE
    assert "openalex (failed)" in result.coverage.execution_shortfalls


def test_supplied_plan_mismatch_fails_before_any_source_runner_is_created() -> None:
    """A plan cannot be replayed against a different declared source set."""
    factory = _Factory(lambda query, source: _result(query, source, []))
    plan = plan_query_family(_revision(), ["arxiv"])

    with pytest.raises(QueryFamilyExecutionError, match="fresh compilation"):
        QueryFamilyRunner(
            _revision(),
            ["openalex"],
            plan=plan,
            search_runner_factory=factory,
        )
    assert factory.calls == []


def test_evidence_verifier_runs_before_even_invalid_revision_planning() -> None:
    """Failed evidence provenance emits zero search calls and precedes planning."""
    revision = _revision()
    invalid_variant = replace(
        revision.variants[0],
        expression=AllOf((TermRef("not_accepted"),)),
    )
    invalid_revision = replace(revision, variants=(invalid_variant,))
    factory = _Factory(lambda query, source: _result(query, source, []))
    verifier_calls: list[str] = []

    def reject(candidate: Any) -> bool:
        verifier_calls.append(candidate.revision_id)
        return False

    with pytest.raises(QueryFamilyExecutionError, match="evidence verification"):
        QueryFamilyRunner(
            invalid_revision,
            ["arxiv"],
            search_runner_factory=factory,
            evidence_verifier=reject,
        )
    assert verifier_calls == [invalid_revision.revision_id]
    assert factory.calls == []


def test_wrong_cell_scope_is_refused_instead_of_silently_widened() -> None:
    """A malformed search result cannot claim it executed an exact cell."""
    factory = _Factory(
        lambda query, source: _result(
            query,
            source,
            [],
            selected_sources=(source, "openalex" if source != "openalex" else "arxiv"),
        )
    )
    runner = QueryFamilyRunner(
        _revision(),
        ["arxiv"],
        search_runner_factory=factory,
    )

    with pytest.raises(QueryFamilyExecutionError, match="exact source scope"):
        runner.run(show_progress=False)


def test_serialization_is_versioned_complete_and_deterministically_ordered() -> None:
    """The family envelope retains its plan, cells, union, and coverage."""
    factory = _Factory(lambda query, source: _result(query, source, []))
    result = QueryFamilyRunner(
        _revision(),
        ["openalex", "arxiv"],
        search_runner_factory=factory,
    ).run(show_progress=False)

    payload = result.to_dict()
    assert payload["schema"] == QUERY_FAMILY_RESULT_SCHEMA
    assert payload["revision_id"] == _revision().revision_id
    assert payload["plan_id"] == result.plan_id
    assert [item["cell"]["cell_id"] for item in payload["cell_outcomes"]] == sorted(
        item["cell"]["cell_id"] for item in payload["cell_outcomes"]
    )
    assert payload["coverage"]["execution"]["complete"] is True
