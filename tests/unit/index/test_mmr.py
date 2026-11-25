"""Tests for diversification as a presentation layer.

The governing constraint is that diversified output must never become snowball
seeds, so the tests are written around the three mechanisms that enforce it
rather than around the arithmetic of the selection.
"""

from __future__ import annotations

import pytest

from lit_search.index.embedding import EmbeddingIdentity
from lit_search.index.hybrid import Candidate
from lit_search.index.mmr import DiversificationError, cosine, diversify


def _candidate(key: str, score: float, *, degraded: str = "") -> Candidate:
    """Build a candidate with a known fused score.

    Parameters
    ----------
    key : str
        Record key.
    score : float
        Fused score.
    degraded : str
        Missing-field label.

    Returns
    -------
    Candidate
        Candidate.
    """
    return Candidate(
        key=key,
        row={"paper_key": key, "title": key.title(), "missing_fields": degraded},
        lane_ranks={"vector": 1},
        fused_score=score,
    )


#: Three near-identical vectors and one far away, so that diversification has
#: something to do and a cluster has something to suppress.
CLUSTER = {
    "a": [1.0, 0.0, 0.0],
    "b": [0.99, 0.14, 0.0],
    "c": [0.98, 0.20, 0.0],
    "far": [0.0, 0.0, 1.0],
}
CANDIDATES = [
    _candidate("a", 0.9),
    _candidate("b", 0.8),
    _candidate("c", 0.7),
    _candidate("far", 0.6),
]


class TestPresentationOnly:
    """Both orders survive, and the seeding accessor returns the relevance one."""

    def test_relevance_order_is_untouched(self) -> None:
        """The set's real order must always be recoverable."""
        view = diversify(CANDIDATES, CLUSTER, limit=3, lambda_value=0.3)
        assert [c.key for c in view.relevance_order] == ["a", "b", "c", "far"]

    def test_display_order_differs_from_relevance_order(self) -> None:
        """Otherwise the test below would prove nothing."""
        view = diversify(CANDIDATES, CLUSTER, limit=3, lambda_value=0.3)
        assert [c.key for c in view.for_display()] != [c.key for c in view.relevance_order][:3]

    def test_seeding_returns_the_relevance_order(self) -> None:
        """A frontier seeded from a diversified list is thinned where it hurts most."""
        view = diversify(CANDIDATES, CLUSTER, limit=3, lambda_value=0.3)
        assert view.for_seeding() == view.relevance_order
        assert view.presentation_only is True

    def test_the_far_paper_is_promoted_by_diversification(self) -> None:
        """That is the behaviour D21 asks for: adjacent sub-areas get shown."""
        view = diversify(CANDIDATES, CLUSTER, limit=2, lambda_value=0.2)
        assert "far" in [c.key for c in view.for_display()]


class TestSuppressionIsRecorded:
    """What diversity removed from the window is listed, not implied."""

    def test_suppressed_papers_are_named_with_what_crowded_them_out(self) -> None:
        """MMR can suppress genuinely important papers; the list makes that checkable."""
        view = diversify(CANDIDATES, CLUSTER, limit=2, lambda_value=0.2)
        assert view.suppressed
        note = view.suppressed[0]
        assert note.relevance_rank <= 2
        assert note.similar_to in CLUSTER
        assert "suppressed" in note.describe()

    def test_a_headline_says_diversification_is_not_a_coverage_fact(self) -> None:
        """A reader must not mistake a display choice for something being missing."""
        view = diversify(CANDIDATES, CLUSTER, limit=2, lambda_value=0.2)
        assert "display choice" in view.headline()

    def test_no_suppressions_when_nothing_is_pushed_out(self) -> None:
        """A window covering everything cannot suppress anything."""
        view = diversify(CANDIDATES, CLUSTER, lambda_value=1.0)
        assert view.suppressed == ()


class TestCandidatesWithoutVectors:
    """A candidate that cannot be compared is kept, never dropped."""

    def test_it_stays_in_the_display_order(self) -> None:
        """Dropping it would be a silent omission."""
        vectors = {key: value for key, value in CLUSTER.items() if key != "far"}
        view = diversify(CANDIDATES, vectors, limit=4, lambda_value=0.5)
        assert "far" in view.undiversifiable
        assert "far" in [c.key for c in view.for_display()]


class TestProvenance:
    """A handle written from a diversified order is marked pool-dependent."""

    def test_pool_independent_is_false(self) -> None:
        """union() then refuses to merge it, which is the enforcement."""
        view = diversify(CANDIDATES, CLUSTER, limit=3, lambda_value=0.5)
        provenance = view.ranking_provenance(
            identity=EmbeddingIdentity(
                model_id="acme/embed", dimension=3, normalization="l2", prompt_convention="q/p"
            ),
            query_text="how are specifications derived",
            text_policy="title+abstract",
        )
        assert provenance.pool_independent is False
        assert "mmr(lambda=0.5)" in provenance.scoring_convention

    def test_degraded_members_reach_the_provenance(self) -> None:
        """A record that sank because its abstract was empty must be explainable."""
        candidates = [*CANDIDATES, _candidate("thin", 0.1, degraded="abstract")]
        vectors = {**CLUSTER, "thin": [0.0, 1.0, 0.0]}
        view = diversify(candidates, vectors, limit=5, lambda_value=0.5)
        provenance = view.ranking_provenance(
            identity=EmbeddingIdentity(
                model_id="acme/embed", dimension=3, normalization="l2", prompt_convention="q/p"
            ),
            query_text="q",
            text_policy="title+abstract",
        )
        assert "thin" in provenance.degraded_text


class TestGuards:
    """Refusals rather than silent clamping."""

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_lambda_out_of_range_is_refused(self, bad: float) -> None:
        """A clamped lambda produces an ordering nobody asked for."""
        with pytest.raises(DiversificationError, match="between 0 and 1"):
            diversify(CANDIDATES, CLUSTER, lambda_value=bad)

    def test_cosine_of_mismatched_vectors_is_zero_not_an_exception(self) -> None:
        """Comparing across widths is meaningless and must not look like similarity."""
        assert cosine([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0
        assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
