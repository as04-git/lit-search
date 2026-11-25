"""Tests for the two operations, the two stages, and what they refuse to do."""

from __future__ import annotations

import pytest

from lit_search.exceptions import RerankInputError
from lit_search.rerank import (
    SCORING_COSINE,
    SCORING_CROSS_ENCODER_LOGIT,
    SEED_INSTRUCTION,
    Reranker,
    order_by_score,
)
from lit_search.store.order import RankMode
from tests.unit.rerank.conftest import blank_title, make_paper


class TestTheTwoOperationsStayDistinct:
    """D14: query-to-document and document-to-document are not one thing."""

    def test_question_mode_records_query_mode_and_the_question(
        self, bi_encoder, cross_encoder, candidates
    ) -> None:
        """The mode is derived from the method, never passed in."""
        ranking = Reranker(bi_encoder, cross_encoder).rank_against_question(
            candidates, "how are specifications inferred from call sites"
        )
        assert ranking.mode is RankMode.QUERY
        assert ranking.recall.provenance.mode is RankMode.QUERY
        assert ranking.recall.provenance.query_text == (
            "how are specifications inferred from call sites"
        )
        assert ranking.recall.provenance.seed_identity is None

    def test_seed_mode_records_seed_mode_and_the_seed(self, bi_encoder, candidates) -> None:
        """A seed ranking names the paper it ranked against."""
        ranking = Reranker(bi_encoder).rank_against_seed_paper(candidates, candidates[0])
        assert ranking.mode is RankMode.SEED
        assert ranking.recall.provenance.seed_identity == "doi:10.1000/fm-agent"
        assert ranking.recall.provenance.query_text is None

    def test_the_two_provenances_never_merge(self, bi_encoder, candidates) -> None:
        """Scores from the two operations must not be pooled."""
        reranker = Reranker(bi_encoder)
        query = reranker.rank_against_question(candidates, "specification inference")
        seed = reranker.rank_against_seed_paper(candidates, candidates[0])
        assert not query.recall.provenance.mergeable_with(seed.recall.provenance)

    def test_the_same_text_scores_differently_in_the_two_modes(
        self, bi_encoder, candidates
    ) -> None:
        """The sharpest form of the D14 check.

        The left-hand text is held identical and only the *operation* changes.
        Query mode prefixes it; seed mode does not.  If the scores came out
        equal, the two operations would be one operation with two names, and
        the query prompt would be doing nothing.
        """
        from lit_search.rerank import build_document_text

        seed = candidates[0]
        seed_text = build_document_text(seed).text
        reranker = Reranker(bi_encoder)
        as_question = reranker.rank_against_question(candidates, seed_text)
        as_seed = reranker.rank_against_seed_paper(candidates, seed)
        assert as_question.recall.scores != as_seed.recall.scores


class TestTwoStages:
    """R6: recall broadly, then rescore the head jointly."""

    def test_both_stages_run_and_are_recorded_separately(
        self, bi_encoder, cross_encoder, candidates
    ) -> None:
        """Each stage carries its own model identity and scoring convention."""
        ranking = Reranker(bi_encoder, cross_encoder).rank_against_question(
            candidates, "specification inference", top_k=2
        )
        assert ranking.stages_run == 2
        assert ranking.recall.provenance.scoring_convention == SCORING_COSINE
        assert ranking.recall.provenance.model_id == "fake/bi-encoder"
        assert ranking.rescored is not None
        assert ranking.rescored.provenance.scoring_convention == SCORING_CROSS_ENCODER_LOGIT
        assert ranking.rescored.provenance.model_id == "fake/cross-encoder"

    def test_stage_one_covers_the_whole_set_and_stage_two_only_the_head(
        self, bi_encoder, cross_encoder, candidates
    ) -> None:
        """A re-rank reorders a set; the truncation is stage two's alone."""
        ranking = Reranker(bi_encoder, cross_encoder).rank_against_question(
            candidates, "specification inference", top_k=2
        )
        assert len(ranking.recall.papers) == len(candidates)
        assert ranking.rescored is not None
        assert len(ranking.rescored.papers) == 2
        assert ranking.top_k == 2

    def test_stage_two_rescores_exactly_the_stage_one_head(
        self, bi_encoder, cross_encoder, candidates
    ) -> None:
        """Stage two must not re-select; it reorders what stage one surfaced."""
        ranking = Reranker(bi_encoder, cross_encoder).rank_against_question(
            candidates, "specification inference", top_k=3
        )
        assert ranking.rescored is not None
        head = {p.doi for p in ranking.recall.papers[:3]}
        assert {p.doi for p in ranking.rescored.papers} == head

    def test_missing_cross_encoder_is_recorded_not_assumed(self, bi_encoder, candidates) -> None:
        """One stage is a legitimate configuration; silence about it is not."""
        ranking = Reranker(bi_encoder).rank_against_question(candidates, "spec inference")
        assert ranking.stages_run == 1
        assert ranking.rescored is None
        assert ranking.top_k is None
        assert "did not run" in ranking.describe()

    def test_reading_order_returns_every_candidate_exactly_once(
        self, bi_encoder, cross_encoder, candidates
    ) -> None:
        """The head then the tail, with nothing lost and nothing duplicated."""
        ranking = Reranker(bi_encoder, cross_encoder).rank_against_question(
            candidates, "specification inference", top_k=2
        )
        order = ranking.reading_order()
        assert len(order) == len(candidates)
        assert {p.doi for p in order} == {p.doi for p in candidates}

    def test_reading_order_puts_the_rescored_head_first(
        self, bi_encoder, cross_encoder, candidates
    ) -> None:
        """Stage two's judgement outranks stage one's for the papers it saw."""
        ranking = Reranker(bi_encoder, cross_encoder).rank_against_question(
            candidates, "specification inference", top_k=2
        )
        assert ranking.rescored is not None
        assert ranking.reading_order()[:2] == ranking.rescored.papers

    def test_no_fused_score_vector_is_offered(self, bi_encoder, cross_encoder, candidates) -> None:
        """Cosines and logits are not on one scale; fusing would fabricate.

        The result deliberately exposes ``reading_order()`` with no scores
        rather than a single score list spanning both stages.
        """
        ranking = Reranker(bi_encoder, cross_encoder).rank_against_question(
            candidates, "specification inference", top_k=2
        )
        assert not hasattr(ranking, "scores")
        assert len(ranking.recall.scores) == len(candidates)
        assert ranking.rescored is not None
        assert len(ranking.rescored.scores) == 2

    def test_same_title_identifier_free_records_keep_their_own_text_and_scores(
        self, bi_encoder, cross_encoder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Neither stage may key document or score membership by title."""
        first = make_paper(
            "Shared title",
            "Alpha-only evidence about verification contracts and caller intent." * 2,
        )
        second = make_paper(
            "Shared title",
            "Beta-only evidence about proof obligations and implementation behaviour." * 2,
        )
        tail = make_paper(
            "Unrelated tail",
            "Gamma-only background that stage two should not rescore but must retain." * 2,
        )
        stage_one_documents = []
        stage_two_documents = []
        reranker = Reranker(bi_encoder, cross_encoder)

        def score_stage_one(question, documents):
            stage_one_documents.extend(documents)
            return [0.9, 0.8, 0.1]

        def score_stage_two(left, documents, *, instruction=None):
            stage_two_documents.extend(documents)
            return [0.2, 0.7]

        monkeypatch.setattr(reranker._recall, "score_against_question", score_stage_one)
        assert reranker._rescore is not None
        monkeypatch.setattr(reranker._rescore, "score", score_stage_two)

        ranking = reranker.rank_against_question(
            [first, second, tail], "verification contracts", top_k=2
        )

        assert len(stage_one_documents) == 3
        assert len({document.identity for document in stage_one_documents}) == 3
        assert "Alpha-only" in stage_one_documents[0].text
        assert "Beta-only" in stage_one_documents[1].text
        assert [document.identity for document in stage_two_documents] == list(
            ranking.recall.identities[:2]
        )
        assert "Alpha-only" in stage_two_documents[0].text
        assert "Beta-only" in stage_two_documents[1].text

        recall_scores = {
            id(paper): score
            for paper, score in zip(ranking.recall.papers, ranking.recall.scores, strict=True)
        }
        assert recall_scores[id(first)] == 0.9
        assert recall_scores[id(second)] == 0.8
        assert ranking.rescored is not None
        rescored_scores = {
            id(paper): score
            for paper, score in zip(ranking.rescored.papers, ranking.rescored.scores, strict=True)
        }
        assert rescored_scores[id(first)] == 0.2
        assert rescored_scores[id(second)] == 0.7

        reading_order = ranking.reading_order()
        assert len(reading_order) == 3
        assert {id(paper) for paper in reading_order} == {id(first), id(second), id(tail)}
        assert len(set(ranking.reading_order_identities())) == 3


class TestSeedModeCrossEncoder:
    """Stage two in seed mode is off by default, and labelled when on."""

    def test_seed_mode_skips_stage_two_by_default(
        self, bi_encoder, cross_encoder, candidates
    ) -> None:
        """The reranker is trained on query-document relevance, not doc-doc."""
        ranking = Reranker(bi_encoder, cross_encoder).rank_against_seed_paper(
            candidates, candidates[0]
        )
        assert ranking.stages_run == 1

    def test_opting_in_substitutes_the_seed_instruction(
        self, bi_encoder, cross_encoder, candidates
    ) -> None:
        """The default instruction says 'web search query'; that would be false."""
        ranking = Reranker(bi_encoder, cross_encoder).rank_against_seed_paper(
            candidates, candidates[0], use_cross_encoder=True, top_k=2
        )
        assert ranking.stages_run == 2
        assert cross_encoder.model.instructions == [SEED_INSTRUCTION]

    def test_the_substituted_instruction_is_recorded_verbatim(
        self, bi_encoder, cross_encoder, candidates
    ) -> None:
        """An unvalidated configuration must be legible as one."""
        ranking = Reranker(bi_encoder, cross_encoder).rank_against_seed_paper(
            candidates, candidates[0], use_cross_encoder=True, top_k=2
        )
        assert ranking.rescored is not None
        convention = ranking.rescored.provenance.prompt_convention
        assert "caller override" in convention
        assert "research paper" in convention

    def test_query_mode_uses_the_model_default_instruction(
        self, bi_encoder, cross_encoder, candidates
    ) -> None:
        """None, not the empty string: an empty prompt does not suppress it."""
        Reranker(bi_encoder, cross_encoder).rank_against_question(
            candidates, "specification inference", top_k=2
        )
        assert cross_encoder.model.instructions == [None]


class TestDeterministicOrdering:
    """bfloat16 ties often, and the fixed order must not depend on input order."""

    def test_ties_break_on_identity_not_input_order(self) -> None:
        """The same set in two input orders must rank identically."""
        forward = order_by_score(["b", "a", "c"], [1.0, 1.0, 1.0])
        assert [["b", "a", "c"][i] for i in forward] == ["a", "b", "c"]
        backward = order_by_score(["c", "a", "b"], [1.0, 1.0, 1.0])
        assert [["c", "a", "b"][i] for i in backward] == ["a", "b", "c"]

    def test_scores_still_dominate_identity(self) -> None:
        """The tie-break must only apply to actual ties."""
        order = order_by_score(["a", "b"], [0.1, 0.9])
        assert order == [1, 0]

    def test_length_mismatch_is_refused(self) -> None:
        """A silently zipped-short ranking would drop members."""
        with pytest.raises(ValueError, match="identity"):
            order_by_score(["a", "b"], [1.0])

    def test_a_whole_ranking_is_order_independent(self, bi_encoder, candidates) -> None:
        """End to end: shuffling the candidate list changes nothing."""
        reranker = Reranker(bi_encoder)
        first = reranker.rank_against_question(candidates, "specification inference")
        second = reranker.rank_against_question(
            list(reversed(candidates)), "specification inference"
        )
        assert [p.doi for p in first.recall.papers] == [p.doi for p in second.recall.papers]


class TestRefusals:
    """Never convert an error into a plausible result."""

    def test_blank_question_is_refused(self, bi_encoder, candidates) -> None:
        """An empty query still embeds, and still produces an order."""
        with pytest.raises(RerankInputError, match="blank question"):
            Reranker(bi_encoder).rank_against_question(candidates, "   ")

    def test_seed_with_no_text_is_refused(self, bi_encoder, candidates) -> None:
        """There is nothing to compare against."""
        with pytest.raises(RerankInputError, match="no title and no abstract"):
            Reranker(bi_encoder).rank_against_seed_paper(
                candidates, blank_title(make_paper("t", ""))
            )

    def test_empty_candidate_set_is_not_an_error(self, bi_encoder, cross_encoder) -> None:
        """An empty input is a legitimate empty output, and stays loud-free."""
        ranking = Reranker(bi_encoder, cross_encoder).rank_against_question([], "anything")
        assert ranking.recall.papers == ()
        assert ranking.reading_order() == ()

    def test_a_degraded_record_is_ranked_not_dropped(self, bi_encoder, candidates) -> None:
        """Dropping it would shrink the set with nothing to show it."""
        ranking = Reranker(bi_encoder).rank_against_question(candidates, "specification")
        assert len(ranking.recall.papers) == len(candidates)
        assert "doi:10.1000/specula" in {p and f"doi:{p.doi}" for p in ranking.recall.papers}


class TestProvenanceCompleteness:
    """Everything needed to interpret or reproduce the order travels with it."""

    def test_every_required_field_is_populated(self, bi_encoder, candidates) -> None:
        """Model, revision, prompts, normalization, text policy, pooling."""
        provenance = (
            Reranker(bi_encoder)
            .rank_against_question(candidates, "specification inference")
            .recall.provenance
        )
        assert provenance.model_id == "fake/bi-encoder"
        assert provenance.model_revision == "feedface"
        assert provenance.normalization == "l2"
        assert provenance.text_policy.endswith("/v1")
        assert provenance.pool_independent is True

    def test_degraded_records_are_named_in_the_provenance(self, bi_encoder, candidates) -> None:
        """R16: which records were scored on incomplete text."""
        provenance = (
            Reranker(bi_encoder)
            .rank_against_question(candidates, "specification inference")
            .recall.provenance
        )
        assert provenance.degraded_text == ("doi:10.1000/specula",)

    def test_degraded_set_does_not_block_merging(self, bi_encoder, candidates) -> None:
        """It describes the inputs, not the scoring configuration."""
        import dataclasses

        provenance = (
            Reranker(bi_encoder)
            .rank_against_question(candidates, "specification inference")
            .recall.provenance
        )
        other = dataclasses.replace(provenance, degraded_text=())
        assert provenance.mergeable_with(other)

    def test_cross_encoder_provenance_declares_no_normalization(
        self, bi_encoder, cross_encoder, candidates
    ) -> None:
        """A pair scorer produces no vectors, so 'l2' would be a false claim."""
        ranking = Reranker(bi_encoder, cross_encoder).rank_against_question(
            candidates, "specification inference", top_k=2
        )
        assert ranking.rescored is not None
        assert ranking.rescored.provenance.normalization == "n/a"

    def test_describe_surfaces_the_degradation(self, bi_encoder, candidates) -> None:
        """The order cannot be shown without what qualifies it."""
        report = (
            Reranker(bi_encoder)
            .rank_against_question(candidates, "specification inference")
            .describe()
        )
        assert "degraded text  : 1 of 4" in report
        assert "doi:10.1000/specula" in report
        assert "embedding dim  : 16 (read from model)" in report

    def test_provenance_round_trips_through_the_handle_schema(self, bi_encoder, candidates) -> None:
        """The store's strict reader must accept what the pipeline writes."""
        from lit_search.store.order import RankingProvenance

        provenance = (
            Reranker(bi_encoder)
            .rank_against_question(candidates, "specification inference")
            .recall.provenance
        )
        restored = RankingProvenance.from_dict(provenance.to_dict(), "order.ranking")
        assert restored == provenance
        assert restored.degraded_text == provenance.degraded_text
