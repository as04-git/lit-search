"""Tests for typed citation targets and provenance-bearing edges."""

import pytest

from lit_search.core.citation import (
    CitationEdge,
    CitationTarget,
    IdentifierKind,
    NativeBatchResult,
    PaperIdentifier,
)
from lit_search.core.paper import Paper


def test_target_keeps_s2_arxiv_and_doi_aliases() -> None:
    """One target may be routed through any identifier its source supplied."""
    target = CitationTarget.from_values(
        semantic_scholar_id="ABCDEF",
        arxiv_id="2604.11556",
        doi="https://doi.org/10.48550/arXiv.2604.11556",
        title="FM-Agent",
    )

    assert target is not None
    assert target.get(IdentifierKind.SEMANTIC_SCHOLAR) == "ABCDEF"
    assert target.get(IdentifierKind.ARXIV) == "2604.11556"
    assert target.get(IdentifierKind.DOI) == "10.48550/arXiv.2604.11556"
    assert len(target.keys) == 3


def test_paper_round_trip_preserves_doi_less_edge_provenance() -> None:
    """Persistence must not collapse a DOI-less S2 edge back to a string."""
    target = CitationTarget(
        (PaperIdentifier(IdentifierKind.SEMANTIC_SCHOLAR, "s2-target"),),
        title="DOI-less target",
    )
    paper = Paper(
        title="Source",
        abstract="",
        authors=[],
        source=None,
        publication_date=None,
        doi="10.1/source",
        reference_edges=[CitationEdge(target=target, source="semantic_scholar")],
    )

    restored = Paper.from_dict(paper.to_dict())

    assert restored.reference_edges == paper.reference_edges
    assert restored.reference_edges[0].source == "semantic_scholar"
    assert restored.reference_edges[0].target.get(IdentifierKind.DOI) is None


def test_native_batch_refuses_unaccounted_request_position(make_paper) -> None:
    """A batch cannot call two identifiers complete after resolving only one."""
    resolved = make_paper("Resolved", doi="10.1/resolved")

    with pytest.raises(ValueError, match="account for every requested position"):
        NativeBatchResult(
            source="semantic_scholar",
            papers=(resolved,),
            requested=2,
        )


def test_native_batch_alias_duplicates_remain_positional(make_paper) -> None:
    """Two requested aliases require two terminal positions when not deduplicated."""
    resolved = make_paper("Same work", doi="10.1/same")

    outcome = NativeBatchResult(
        source="semantic_scholar",
        papers=(resolved, resolved),
        requested=2,
    )

    assert outcome.complete


def test_native_batch_keeps_edge_page_failures_outside_target_position_accounting(
    make_paper,
) -> None:
    """A resolved focal target may still carry a failed citation-page request."""
    resolved = make_paper("Resolved focal", doi="10.1/focal")
    outcome = NativeBatchResult(
        source="openalex",
        papers=(resolved,),
        requested=1,
        edge_parse_failures=2,
        edge_request_failures=1,
        error_messages=("cited-by page failed",),
    )

    assert not outcome.complete
    assert NativeBatchResult.from_dict(outcome.to_dict()).to_dict() == outcome.to_dict()
