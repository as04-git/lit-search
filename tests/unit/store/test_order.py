"""Tests for ordered-versus-set semantics."""

from __future__ import annotations

import dataclasses

import pytest

from lit_search.exceptions import HandleSchemaError, OrderSemanticsError
from lit_search.store.order import OrderRecord, OrderState, RankingProvenance, RankMode


class TestRankingProvenance:
    """A ranking claim must say what it ranked against and how."""

    def test_query_mode_requires_query_text(self) -> None:
        """A query-mode ranking with no question asserts nothing checkable."""
        with pytest.raises(OrderSemanticsError, match="query text"):
            RankingProvenance(
                mode=RankMode.QUERY,
                model_id="m",
                prompt_convention="none",
                scoring_convention="cosine",
                normalization="none",
                text_policy="title/v1",
                pool_independent=True,
            )

    def test_seed_mode_requires_seed_identity(self) -> None:
        """A seed-mode ranking must name the paper it was ranked against."""
        with pytest.raises(OrderSemanticsError, match="seed identity"):
            RankingProvenance(
                mode=RankMode.SEED,
                model_id="m",
                prompt_convention="none",
                scoring_convention="cosine",
                normalization="none",
                text_policy="title/v1",
                pool_independent=True,
            )

    def test_prompt_convention_is_required(self, query_ranking) -> None:
        """The prompt convention cannot be omitted; "none" must be explicit."""
        with pytest.raises(OrderSemanticsError, match="prompt convention"):
            dataclasses.replace(query_ranking, prompt_convention="")

    def test_model_id_is_required(self, query_ranking) -> None:
        """A ranking that does not name its model cannot be reproduced."""
        with pytest.raises(OrderSemanticsError, match="model"):
            dataclasses.replace(query_ranking, model_id="")

    def test_scoring_convention_is_required(self, query_ranking) -> None:
        """A score with no stated convention cannot be compared to another."""
        with pytest.raises(OrderSemanticsError, match="scoring convention"):
            dataclasses.replace(query_ranking, scoring_convention="")


class TestMergeability:
    """Scores may only be compared when they mean the same thing."""

    def test_identical_pool_independent_rankings_merge(self, query_ranking) -> None:
        """Same query, same model, same convention, pool independent."""
        assert query_ranking.mergeable_with(dataclasses.replace(query_ranking))

    def test_pool_dependent_rankings_never_merge(self, query_ranking) -> None:
        """A diversified order scores a paper by what else is in the pool."""
        mmr = dataclasses.replace(
            query_ranking, scoring_convention="mmr(lambda=0.5)", pool_independent=False
        )
        assert not mmr.mergeable_with(dataclasses.replace(mmr))

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("query_text", "a different question"),
            ("model_id", "BAAI/bge-large-en-v1.5"),
            ("model_revision", "cafebabe"),
            ("prompt_convention", "none"),
            ("scoring_convention", "cross-encoder-logit"),
        ],
    )
    def test_any_difference_blocks_merging(self, query_ranking, field, value) -> None:
        """The comparison is exact; a near-match is a wrong answer."""
        other = dataclasses.replace(query_ranking, **{field: value})
        assert not query_ranking.mergeable_with(other)

    def test_seed_and_query_rankings_do_not_merge(self, query_ranking) -> None:
        """The two re-rank operations are not interchangeable."""
        seed = RankingProvenance(
            mode=RankMode.SEED,
            seed_identity="10.1000/seed",
            model_id=query_ranking.model_id,
            model_revision=query_ranking.model_revision,
            prompt_convention=query_ranking.prompt_convention,
            scoring_convention=query_ranking.scoring_convention,
            normalization=query_ranking.normalization,
            text_policy=query_ranking.text_policy,
            pool_independent=True,
        )
        assert not query_ranking.mergeable_with(seed)


class TestOrderRecordInvariants:
    """State and payload must agree."""

    def test_ranked_requires_scores(self, query_ranking) -> None:
        """A ranking claim with no scores is unbacked."""
        with pytest.raises(OrderSemanticsError, match="provenance and scores"):
            OrderRecord(state=OrderState.RANKED, ranking=query_ranking)

    def test_ranked_requires_provenance(self) -> None:
        """Scores with no provenance do not say what they measure."""
        with pytest.raises(OrderSemanticsError, match="provenance and scores"):
            OrderRecord(state=OrderState.RANKED, scores=(1.0,))

    def test_unordered_must_not_carry_scores(self, query_ranking) -> None:
        """An unordered set has nothing to score."""
        with pytest.raises(OrderSemanticsError, match="unordered"):
            OrderRecord(state=OrderState.UNORDERED, ranking=query_ranking, scores=(1.0,))

    def test_invalidated_requires_previous_ranking(self) -> None:
        """An invalidated order must say what it lost, and to what."""
        with pytest.raises(OrderSemanticsError, match="must record the ranking it lost"):
            OrderRecord(state=OrderState.INVALIDATED)

    def test_invalidated_must_not_retain_scores(self, query_ranking) -> None:
        """Retained scores would let the fabricated order come back."""
        with pytest.raises(OrderSemanticsError, match="must not retain"):
            OrderRecord(
                state=OrderState.INVALIDATED,
                previous_ranking=query_ranking,
                invalidated_by="union",
                scores=(1.0,),
            )

    def test_non_finite_scores_are_refused(self, query_ranking) -> None:
        """A NaN score would silently scramble the sort it takes part in."""
        with pytest.raises(OrderSemanticsError, match="finite"):
            OrderRecord.ranked(query_ranking, [1.0, float("nan")])

    def test_member_count_mismatch_is_refused(self, query_ranking) -> None:
        """One score per member, or the ranking is not a ranking."""
        order = OrderRecord.ranked(query_ranking, [1.0, 0.5])
        with pytest.raises(OrderSemanticsError, match="3 member"):
            order.validate_member_count(3)

    def test_unordered_member_count_is_unconstrained(self) -> None:
        """An unordered handle has no score list to line up."""
        OrderRecord.unordered().validate_member_count(99)


class TestTransitions:
    """Subsetting preserves an order; invalidation destroys it."""

    def test_subset_keeps_provenance_and_reindexes_scores(self, query_ranking) -> None:
        """Removing members cannot falsify the survivors' relative sequence."""
        order = OrderRecord.ranked(query_ranking, [0.9, 0.7, 0.5, 0.3])
        reduced = order.subset([0, 2])
        assert reduced.state is OrderState.RANKED
        assert reduced.ranking == query_ranking
        assert reduced.scores == (0.9, 0.5)

    def test_subset_of_unordered_is_unchanged(self) -> None:
        """There is no order to carry through."""
        order = OrderRecord.unordered()
        assert order.subset([0, 1]) is order

    def test_invalidate_drops_scores_and_keeps_provenance(self, query_ranking) -> None:
        """The claim is recorded as lost, and its evidence is not carried on."""
        invalidated = OrderRecord.ranked(query_ranking, [1.0]).invalidate("union")
        assert invalidated.state is OrderState.INVALIDATED
        assert invalidated.previous_ranking == query_ranking
        assert invalidated.invalidated_by == "union"
        assert invalidated.scores is None
        assert invalidated.ranking is None

    def test_invalidate_on_unordered_is_a_no_op(self) -> None:
        """Manufacturing an invalidation asserts a ranking that never existed."""
        order = OrderRecord.unordered()
        assert order.invalidate("union") is order

    def test_invalidate_is_idempotent(self, query_ranking) -> None:
        """The first operation to break the claim keeps the attribution."""
        once = OrderRecord.ranked(query_ranking, [1.0]).invalidate("union")
        assert once.invalidate("deduplicate") is once
        assert once.invalidated_by == "union"


class TestOrderSerialization:
    """Order records round-trip and refuse unrecognised payloads."""

    @pytest.mark.parametrize("state", ["unordered", "ranked", "invalidated"])
    def test_round_trip(self, query_ranking, state) -> None:
        """Every state survives a to_dict/from_dict cycle."""
        orders = {
            "unordered": OrderRecord.unordered(),
            "ranked": OrderRecord.ranked(query_ranking, [0.9, 0.1]),
            "invalidated": OrderRecord.ranked(query_ranking, [0.9]).invalidate("union"),
        }
        original = orders[state]
        assert OrderRecord.from_dict(original.to_dict(), "order") == original

    def test_unknown_state_is_refused(self, query_ranking) -> None:
        """The state is a closed enum."""
        payload = OrderRecord.ranked(query_ranking, [1.0]).to_dict()
        payload["state"] = "probably-ranked"
        with pytest.raises(HandleSchemaError, match="not one of"):
            OrderRecord.from_dict(payload, "order")

    def test_unknown_order_key_is_refused(self) -> None:
        """An unrecognised key means the writer knew something this reader does not."""
        payload = OrderRecord.unordered().to_dict()
        payload["tie_break"] = "citations"
        with pytest.raises(HandleSchemaError, match="unrecognised key"):
            OrderRecord.from_dict(payload, "order")

    def test_missing_order_key_is_refused(self) -> None:
        """Absent keys are refused, never defaulted."""
        payload = OrderRecord.unordered().to_dict()
        del payload["scores"]
        with pytest.raises(HandleSchemaError, match="scores: required key is missing"):
            OrderRecord.from_dict(payload, "order")

    def test_non_finite_score_in_payload_is_refused(self, query_ranking) -> None:
        """A NaN that reached a file is caught on the way back in."""
        payload = OrderRecord.ranked(query_ranking, [1.0]).to_dict()
        payload["scores"] = [float("inf")]
        with pytest.raises(HandleSchemaError, match="finite"):
            OrderRecord.from_dict(payload, "order")

    def test_unknown_ranking_key_is_refused(self, query_ranking) -> None:
        """Refusal reaches into the provenance payload too."""
        payload = OrderRecord.ranked(query_ranking, [1.0]).to_dict()
        payload["ranking"]["temperature"] = 0.7
        with pytest.raises(HandleSchemaError, match=r"order\.ranking: unrecognised key"):
            OrderRecord.from_dict(payload, "order")

    def test_ranked_must_not_carry_invalidation_fields(self, query_ranking) -> None:
        """A live ranking and a record of its destruction are contradictory."""
        with pytest.raises(OrderSemanticsError, match="must not carry invalidation"):
            OrderRecord(
                state=OrderState.RANKED,
                ranking=query_ranking,
                scores=(1.0,),
                previous_ranking=query_ranking,
                invalidated_by="union",
            )

    def test_unordered_must_not_carry_invalidation_fields(self, query_ranking) -> None:
        """An unordered set never had a claim to lose."""
        with pytest.raises(OrderSemanticsError, match="unordered order must not carry"):
            OrderRecord(
                state=OrderState.UNORDERED,
                previous_ranking=query_ranking,
                invalidated_by="union",
            )
