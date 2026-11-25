"""Tests for three-lane hybrid retrieval.

Two claims carry the design and are tested directly:

* Each returned paper records **which lane found it**.  That is the acceptance
  test's mechanism-attribution output, and it is the reason fusion is done here
  rather than by LanceDB's single ``_relevance_score``.
* A lane that could not run is reported as unavailable **with a reason**, never
  as an empty result.  "Matched nothing" and "did not run" mean opposite
  things.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from lit_search.core.paper import Paper
from lit_search.index.hybrid import (
    Lane,
    LaneOutcome,
    hybrid_search,
)
from lit_search.index.paper_index import PaperIndex


class TestLaneOutcome:
    """An unavailable lane must say why."""

    def test_unexplained_unavailability_is_refused(self) -> None:
        """An empty lane with no reason is indistinguishable from a clean miss."""
        with pytest.raises(ValueError, match="gives no reason"):
            LaneOutcome(lane=Lane.BM25, available=False, requested=10, returned=0)

    def test_describe_names_the_reason(self) -> None:
        """The reason is what a reader acts on."""
        outcome = LaneOutcome(
            lane=Lane.BM25,
            available=False,
            requested=10,
            returned=0,
            unavailable_reason="no full-text index",
        )
        assert "DID NOT RUN" in outcome.describe()
        assert "no full-text index" in outcome.describe()


class TestAttribution:
    """Which mechanism surfaced each paper is recorded per paper."""

    def test_every_candidate_names_its_lanes(
        self, index: PaperIndex, corpus: list[Paper], embedder
    ) -> None:
        """The attribution map is the acceptance test's fourth pass condition."""
        index.add(corpus)
        result = hybrid_search(index, embedder, 'specification inference "FM-Agent"', limit=4)
        attribution = result.attribution()
        assert attribution
        for lanes in attribution.values():
            assert lanes
            assert set(lanes) <= {lane.value for lane in Lane}

    def test_a_literal_match_is_attributed_to_the_exact_lane(
        self, index: PaperIndex, corpus: list[Paper], embedder
    ) -> None:
        """The literal lane exists because BM25 stemming destroys exact phrases."""
        index.add(corpus)
        result = hybrid_search(index, embedder, '"FM-Agent"', limit=4)
        target = "10.48550/arxiv.2604.11556"
        assert Lane.EXACT.value in result.attribution()[target]

    def test_the_exact_lane_finds_an_author_name_the_query_lane_would_not(
        self, index: PaperIndex, corpus: list[Paper], embedder
    ) -> None:
        """Author names are what pure vector search handles worst."""
        index.add(corpus)
        rows = index.exact_lane(["Wei Li"], 10)
        assert {row["paper_key"] for row in rows} == {"10.48550/arxiv.2604.11556"}


class TestLaneAvailability:
    """A lane that cannot run is reported, not silently skipped."""

    def test_bm25_is_unavailable_without_a_full_text_index(
        self, index: PaperIndex, embedder
    ) -> None:
        """An index with no rows has no full-text index either."""
        result = hybrid_search(index, embedder, "specification", limit=5)
        unavailable = {outcome.lane for outcome in result.unavailable_lanes}
        assert Lane.BM25 in unavailable
        assert "full-text index" in result.headline() or "bm25" in result.headline()

    def test_the_exact_lane_declines_a_long_unquoted_query(
        self, index: PaperIndex, corpus: list[Paper], embedder
    ) -> None:
        """Matching a prose sentence literally returns nothing and would look empty."""
        index.add(corpus)
        long_query = (
            "decompose a repository into logical components and extract evidence channels "
            "from code documentation tests and call sites"
        )
        result = hybrid_search(index, embedder, long_query, limit=5)
        reasons = {outcome.lane: outcome.unavailable_reason for outcome in result.unavailable_lanes}
        assert Lane.EXACT in reasons
        assert "quote" in (reasons[Lane.EXACT] or "")

    def test_supplied_exact_terms_revive_the_lane(
        self, index: PaperIndex, corpus: list[Paper], embedder
    ) -> None:
        """A caller who knows the literal can always supply it."""
        index.add(corpus)
        result = hybrid_search(
            index,
            embedder,
            "a very long prose query with many words in it indeed",
            limit=5,
            exact_terms=["Hoare-Style"],
        )
        assert Lane.EXACT not in {outcome.lane for outcome in result.unavailable_lanes}


class TestFusion:
    """Reciprocal rank fusion, with per-lane weights and deterministic ties."""

    def test_a_paper_found_by_three_lanes_outranks_one_found_by_one(
        self, index: PaperIndex, corpus: list[Paper], embedder
    ) -> None:
        """That is the whole benefit of fusing rather than picking a lane."""
        index.add(corpus)
        result = hybrid_search(index, embedder, 'specification inference "FM-Agent"', limit=4)
        best = result.candidates[0]
        assert len(best.lanes) >= 2
        assert best.key == "10.48550/arxiv.2604.11556"

    def test_repeating_a_query_gives_the_same_order(
        self, index: PaperIndex, corpus: list[Paper], embedder
    ) -> None:
        """A ranking that shuffles between runs cannot be reasoned about."""
        index.add(corpus)
        first = hybrid_search(index, embedder, "specification inference", limit=4)
        second = hybrid_search(index, embedder, "specification inference", limit=4)
        assert [c.key for c in first.candidates] == [c.key for c in second.candidates]

    def test_degraded_text_travels_into_the_result(
        self, index: PaperIndex, corpus: list[Paper], embedder
    ) -> None:
        """A record scored on a title alone must be visible as such."""
        index.add(corpus)
        result = hybrid_search(index, embedder, "specification synthesis", limit=4)
        degraded = {c.key: c.degraded_text for c in result.candidates if c.degraded_text}
        assert "abstract" in degraded.get("10.0/preguss", "")

    def test_restricting_to_keys_confines_every_lane(
        self, index: PaperIndex, corpus: list[Paper], embedder, make_paper: Callable[..., Paper]
    ) -> None:
        """Re-ranking one stored set must not pull in the rest of the index."""
        index.add(corpus)
        index.add([make_paper("An Unrelated Paper on Marine Biology", "10.0/marine")])
        keys = ["10.48550/arxiv.2604.11556", "10.0/marine"]
        result = hybrid_search(index, embedder, "specification", limit=10, keys=keys)
        assert {c.key for c in result.candidates} <= set(keys)

    def test_a_supplied_query_vector_replaces_the_embedded_one(
        self, index: PaperIndex, corpus: list[Paper], embedder
    ) -> None:
        """Seed mode embeds a document, not a query, and must be able to say so."""
        index.add(corpus)
        seed_vector = embedder.encode_documents(["Agentic model checking with dual sources"])[0]
        result = hybrid_search(index, embedder, "ignored text", limit=3, query_vector=seed_vector)
        assert result.candidates
        assert result.cache_statistics["misses"] == 0
