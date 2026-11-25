"""Falsifiers for the pure snowball planning, routing, and reduction seam."""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from typing import cast

from lit_search.core.citation import (
    CitationEdge,
    CitationTarget,
    IdentifierKind,
    PaperIdentifier,
)
from lit_search.core.paper import Paper
from lit_search.core.snowball_run import TraversalDirection
from lit_search.snowball.flow import (
    AcquisitionIntent,
    AcquisitionPort,
    CurrentCitationCapPolicy,
    LevelReduction,
    PolicyDecision,
    RouteBatch,
    TraversalAssertion,
    TraversalLevelPlan,
    plan_frontier,
    reduce_level,
    route_assertions,
)


def _target(
    *,
    doi: str | None = None,
    arxiv: str | None = None,
    semantic_scholar: str | None = None,
    openalex: str | None = None,
) -> CitationTarget:
    target = CitationTarget.from_values(
        doi=doi,
        arxiv_id=arxiv,
        semantic_scholar_id=semantic_scholar,
        openalex_id=openalex,
    )
    assert target is not None
    return target


def _paper(
    title: str,
    *,
    doi: str | None = None,
    citations: int | None = None,
    semantic_scholar: str | None = None,
    arxiv: str | None = None,
    reference_edges: list[CitationEdge] | None = None,
    cited_by_edges: list[CitationEdge] | None = None,
) -> Paper:
    identifiers = []
    if semantic_scholar is not None:
        identifiers.append(PaperIdentifier(IdentifierKind.SEMANTIC_SCHOLAR, semantic_scholar))
    if arxiv is not None:
        identifiers.append(PaperIdentifier(IdentifierKind.ARXIV, arxiv))
    return Paper(
        title=title,
        abstract="",
        authors=[],
        source=None,
        publication_date=dt.date(2024, 1, 1),
        doi=doi,
        citations=citations,
        identifiers=identifiers,
        reference_edges=reference_edges,
        cited_by_edges=cited_by_edges,
    )


def _assertion(
    target: CitationTarget,
    *,
    source: str,
    parent: str = "doi:10.1000/parent",
    direction: str = "backward",
) -> TraversalAssertion:
    return TraversalAssertion(
        target=target,
        target_key=json.dumps(sorted(target.keys), sort_keys=True, separators=(",", ":")),
        aliases=tuple(sorted(target.keys)),
        level=1,
        direction=cast(TraversalDirection, direction),
        asserting_source=source,
        parent_key=parent,
        parent_aliases=(parent,),
    )


def _plan(*frontier: Paper, canonical: bool = True) -> TraversalLevelPlan:
    return plan_frontier(
        tuple(frontier),
        level=1,
        direction="both",
        canonical=canonical,
    )


def _assertion_plan(*assertions: TraversalAssertion) -> TraversalLevelPlan:
    return TraversalLevelPlan(
        level=1,
        assertions=assertions,
        reservations=tuple(dict.fromkeys(assertion.route_key for assertion in assertions)),
    )


def _policy(
    *,
    result_cap: int | None = None,
    expansion_cap: int | None = None,
) -> CurrentCitationCapPolicy:
    return CurrentCitationCapPolicy(
        max_papers_per_level=result_cap,
        max_expansion_per_level=expansion_cap,
    )


def test_plan_preserves_parent_direction_source_plurality_for_one_target() -> None:
    """Shared targets must not erase the distinct claims that discovered them."""
    target = _target(doi="10.1000/shared")
    first = _paper(
        "First parent",
        doi="10.1000/parent-a",
        reference_edges=[CitationEdge(target, "crossref")],
        cited_by_edges=[CitationEdge(target, "semantic_scholar")],
    )
    second = _paper(
        "Second parent",
        doi="10.1000/parent-b",
        reference_edges=[CitationEdge(target, "crossref")],
        cited_by_edges=[CitationEdge(target, "semantic_scholar")],
    )

    plan = _plan(first, second)

    assert len(plan.assertions) == 4
    assert {
        (assertion.parent_key, assertion.direction, assertion.asserting_source)
        for assertion in plan.assertions
    } == {
        ("doi:10.1000/parent-a", "backward", "crossref"),
        ("doi:10.1000/parent-a", "forward", "semantic_scholar"),
        ("doi:10.1000/parent-b", "backward", "crossref"),
        ("doi:10.1000/parent-b", "forward", "semantic_scholar"),
    }


def test_route_deduplicates_provider_target_without_erasing_assertions() -> None:
    """Request coalescing is not permission to coalesce provenance assertions."""
    target = _target(doi="10.1000/shared")
    assertions = (
        _assertion(target, source="crossref", parent="doi:10.1000/a"),
        _assertion(target, source="crossref", parent="doi:10.1000/b"),
    )

    batches = route_assertions(_assertion_plan(*assertions), canonical=True)

    assert len(batches) == 1
    assert batches[0].requested_source == "crossref"
    assert batches[0].execution_source == "crossref"
    assert batches[0].assertions == assertions
    assert batches[0].targets == (target,)


def test_route_alias_dedup_preserves_versioned_and_unversioned_records() -> None:
    """A provider alias is exact identity only when both claims are unversioned."""
    provider_only = _target(semantic_scholar="shared")
    first_version = _target(semantic_scholar="shared", doi="10.1/first")
    second_version = _target(semantic_scholar="shared", doi="10.1/second")

    compatible = route_assertions(
        _assertion_plan(
            _assertion(provider_only, source="semantic_scholar", parent="doi:10.1/a"),
            _assertion(first_version, source="semantic_scholar", parent="doi:10.1/b"),
        ),
        canonical=True,
    )
    conflicting = route_assertions(
        _assertion_plan(
            _assertion(first_version, source="semantic_scholar", parent="doi:10.1/a"),
            _assertion(second_version, source="semantic_scholar", parent="doi:10.1/b"),
        ),
        canonical=True,
    )

    assert len(compatible[0].assertions) == 2
    assert len(compatible[0].targets) == 2
    assert len(conflicting[0].assertions) == 2
    assert len(conflicting[0].targets) == 2


def test_route_identity_cannot_bridge_publisher_and_preprint_versions() -> None:
    """A provider record's publisher DOI is primary; arXiv remains link evidence."""
    composite = _target(
        semantic_scholar="shared",
        doi="10.1/publisher",
        arxiv="2401.00001",
    )
    publisher = _target(semantic_scholar="shared", doi="10.1/publisher")
    preprint = _target(semantic_scholar="shared", arxiv="2401.00001")
    assertions = (
        _assertion(composite, source="semantic_scholar", parent="doi:10.1/a"),
        _assertion(publisher, source="semantic_scholar", parent="doi:10.1/b"),
        _assertion(preprint, source="semantic_scholar", parent="doi:10.1/c"),
    )

    for ordering in (assertions, tuple(reversed(assertions))):
        batch = route_assertions(
            _assertion_plan(*ordering),
            canonical=True,
        )[0]
        assert len(batch.assertions) == 3
        assert len(batch.targets) == 2


def test_current_router_matches_provider_matrix_and_datacite_arxiv_exception() -> None:
    """Routing stays source-native except for the explicit arXiv DOI exception."""
    cases = (
        (
            _assertion(
                _target(semantic_scholar="S2", doi="10.1/s2"),
                source="semantic_scholar",
            ),
            "semantic_scholar",
        ),
        (
            _assertion(_target(openalex="W123", doi="10.1/oa"), source="openalex"),
            "openalex",
        ),
        (_assertion(_target(arxiv="2401.00001"), source="arxiv"), "arxiv"),
        (_assertion(_target(doi="10.1/crossref"), source="crossref"), "crossref"),
        (_assertion(_target(doi="10.1/legacy"), source="legacy_doi"), "legacy_doi"),
        (
            _assertion(_target(doi="10.48550/arXiv.2401.00002"), source="crossref"),
            "arxiv",
        ),
        (
            _assertion(_target(doi="10.48550/arXiv.2401.00003"), source="legacy_doi"),
            "arxiv",
        ),
    )

    batches = route_assertions(_assertion_plan(*(case[0] for case in cases)), canonical=True)
    routed = {
        assertion.target_key: batch.execution_source
        for batch in batches
        for assertion in batch.assertions
    }

    assert routed == {assertion.target_key: expected for assertion, expected in cases}

    unknown = _assertion(_target(doi="10.1/unknown"), source="unknown_graph")
    unknown_batches = route_assertions(_assertion_plan(unknown), canonical=True)
    assert len(unknown_batches) == 1
    assert unknown_batches[0].requested_source == "unknown_graph"
    assert unknown_batches[0].execution_source == "unknown_graph"
    assert unknown_batches[0].assertions == (unknown,)


def test_acquisition_intent_and_port_expose_projection_contract() -> None:
    """Projection is a first-class intent rather than an implicit bool convention."""

    class RecordingPort:
        def __init__(self) -> None:
            self.intents: list[AcquisitionIntent] = []

        def acquire(
            self,
            batches: tuple[RouteBatch, ...],
            *,
            intent: AcquisitionIntent,
        ) -> Sequence[Paper]:
            self.intents.append(intent)
            return ()

    port = RecordingPort()
    typed_port: AcquisitionPort = port
    typed_port.acquire((), intent=AcquisitionIntent.PROJECTION)
    typed_port.acquire((), intent=AcquisitionIntent.TRAVERSAL_EDGES)

    assert AcquisitionIntent.PROJECTION.value == "projection"
    assert AcquisitionIntent.TRAVERSAL_EDGES.value == "traversal_edges"
    assert port.intents == [
        AcquisitionIntent.PROJECTION,
        AcquisitionIntent.TRAVERSAL_EDGES,
    ]


def test_current_policy_matches_caps_none_ties_and_last_level() -> None:
    """The extracted compatibility policy preserves the runner's stable cap rules."""
    none = _paper("Unknown citations", doi="10.1/none", citations=None)
    tied_first = _paper("Tie first", doi="10.1/tie-a", citations=7)
    high = _paper("High", doi="10.1/high", citations=20)
    tied_second = _paper("Tie second", doi="10.1/tie-b", citations=7)
    zero = _paper("Zero", doi="10.1/zero", citations=0)
    papers = (none, tied_first, high, tied_second, zero)

    uncapped: PolicyDecision = _policy().select(papers, level=1)
    assert uncapped.result_members == papers
    assert uncapped.expansion_members == papers
    assert uncapped.excluded_from_result == ()
    assert uncapped.unexpanded == ()

    capped = _policy(result_cap=3, expansion_cap=2).select(papers, level=1)
    assert capped.result_members == (high, tied_first, tied_second)
    assert capped.expansion_members == (high, tied_first)
    assert capped.excluded_from_result == (none, zero)
    assert capped.unexpanded == (none, tied_second, zero)

    # Current selection is level-invariant. The runner, not the pure policy,
    # skips enrichment and ignores the returned frontier after the last level.
    last = _policy(expansion_cap=2).select(papers, level=2)
    assert last.result_members == papers
    assert last.expansion_members == (high, tied_first)
    assert last.unexpanded == (none, tied_second, zero)


def test_reduce_keeps_conflicting_version_unresolved_despite_provider_alias() -> None:
    """A shared provider ID cannot resolve a differently stated publication version."""
    wanted = _target(doi="10.1/version-a", semantic_scholar="shared-provider-id")
    conflicting = _paper(
        "Conflicting version",
        doi="10.1/version-b",
        semantic_scholar="shared-provider-id",
        citations=4,
    )
    assertion = _assertion(wanted, source="semantic_scholar")

    reduced: LevelReduction = reduce_level(
        _assertion_plan(assertion),
        (conflicting,),
        eligibility=lambda _paper: True,
        policy=_policy(),
    )

    assert reduced.resolved_assertions == ()
    assert reduced.unresolved_assertions == (assertion,)
    # Acquisition and assertion resolution are separate: the acquired record
    # remains policy-visible even though it does not satisfy this assertion.
    assert reduced.policy.result_members == (conflicting,)


def test_primary_version_rule_aligns_planning_and_reduction() -> None:
    """Visited, attempted, and acquired provider aliases cannot wildcard a DOI."""
    versioned = _target(doi="10.1/version", semantic_scholar="shared")
    composite = _target(
        doi="10.1/version",
        arxiv="2401.00001",
        semantic_scholar="shared",
    )
    frontier = _paper(
        "Parent",
        doi="10.1/parent",
        reference_edges=[CitationEdge(versioned, "semantic_scholar")],
    )
    provider_group = frozenset({"semantic_scholar:shared"})

    visited_plan = plan_frontier(
        (frontier,),
        level=1,
        direction="backward",
        visited=set(provider_group),
        visited_alias_groups=(provider_group,),
    )
    attempted_plan = plan_frontier(
        (frontier,),
        level=1,
        direction="backward",
        attempted_routes={"semantic_scholar": (provider_group,)},
    )

    assert len(visited_plan.assertions) == 1
    assert len(attempted_plan.assertions) == 1

    assertion = _assertion(composite, source="semantic_scholar")
    publisher = _paper(
        "Publisher",
        doi="10.1/version",
        semantic_scholar="shared",
    )
    preprint = _paper(
        "Preprint",
        arxiv="2401.00001",
        semantic_scholar="shared",
    )
    provider_only = _paper("Provider only", semantic_scholar="shared")

    publisher_reduction = reduce_level(
        _assertion_plan(assertion),
        (publisher,),
        eligibility=lambda _paper: True,
        policy=_policy(),
    )
    for non_primary in (preprint, provider_only):
        reduction = reduce_level(
            _assertion_plan(assertion),
            (non_primary,),
            eligibility=lambda _paper: True,
            policy=_policy(),
        )
        assert reduction.resolved_assertions == ()
        assert reduction.unresolved_assertions == (assertion,)

    assert publisher_reduction.resolved_assertions == (assertion,)
    assert publisher_reduction.unresolved_assertions == ()


def test_reduce_separates_resolved_and_unresolved_assertions() -> None:
    """One successful target does not launder a different acquisition shortfall."""
    found_target = _target(doi="10.1/found")
    missing_target = _target(doi="10.1/missing")
    found_assertion = _assertion(found_target, source="crossref")
    missing_assertion = _assertion(missing_target, source="crossref")
    found = _paper("Found", doi="10.1/found", citations=1)

    reduced = reduce_level(
        _assertion_plan(found_assertion, missing_assertion),
        (found,),
        eligibility=lambda _paper: True,
        policy=_policy(expansion_cap=1),
    )

    assert reduced.resolved_assertions == (found_assertion,)
    assert reduced.unresolved_assertions == (missing_assertion,)
    assert reduced.policy.result_members == (found,)
    assert reduced.policy.expansion_members == (found,)


def test_acquisition_shortfall_and_policy_residual_remain_separate() -> None:
    """An unresolved request and a deliberate cap omission are different facts."""
    high_target = _target(doi="10.1/high")
    low_target = _target(doi="10.1/low")
    missing_target = _target(doi="10.1/missing")
    high_assertion = _assertion(high_target, source="crossref")
    low_assertion = _assertion(low_target, source="crossref")
    missing_assertion = _assertion(missing_target, source="crossref")
    high = _paper("High", doi="10.1/high", citations=20)
    low = _paper("Low", doi="10.1/low", citations=1)

    reduced = reduce_level(
        _assertion_plan(high_assertion, low_assertion, missing_assertion),
        (low, high),
        eligibility=lambda _paper: True,
        policy=_policy(result_cap=1, expansion_cap=1),
    )

    assert reduced.unresolved_assertions == (missing_assertion,)
    assert reduced.policy.excluded_from_result == (low,)
    assert reduced.policy.unexpanded == (low,)


def test_canonical_plan_and_routes_ignore_frontier_order() -> None:
    """Durable plans have one semantic order regardless of discovery ordering."""
    first_target = _target(doi="10.1/a")
    second_target = _target(doi="10.1/b")
    first = _paper(
        "Parent A",
        doi="10.1/parent-a",
        reference_edges=[CitationEdge(second_target, "crossref")],
    )
    second = _paper(
        "Parent B",
        doi="10.1/parent-b",
        reference_edges=[CitationEdge(first_target, "crossref")],
    )

    forward = _plan(first, second)
    reversed_plan = _plan(second, first)

    assert forward == reversed_plan
    assert route_assertions(forward, canonical=True) == route_assertions(
        reversed_plan, canonical=True
    )
