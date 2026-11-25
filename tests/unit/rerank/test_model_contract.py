"""Tests for the load-time checks on dimension, normalization, and pinning."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from lit_search.exceptions import ModelContractError
from lit_search.rerank import load_bi_encoder, load_cross_encoder
from tests.unit.rerank.conftest import FakePairScorer, FakeSentenceEncoder, Sigmoid

pytest.importorskip("sentence_transformers", reason="needs the rerank extra")


class TestQueryModeIsMeasuredNotDeclared:
    """A declared query mode is not a query mode.

    ``sentence_transformers`` types prompt values as ``str | None``, so a model
    can carry a ``query`` key whose value is null or empty.  That satisfies
    ``"query" in prompts``, applies no prefix, and leaves ``encode_query``
    doing exactly what ``encode`` does -- the same silent no-op the key check
    was written to catch, arriving through the one door the key check cannot
    close.  The loader therefore encodes a probe text both ways and compares.
    """

    def test_a_null_query_prompt_is_refused(self) -> None:
        """The declaration passes; the measurement must not."""
        model = FakeSentenceEncoder(prompts={"query": None, "document": ""})
        with (
            patch("sentence_transformers.SentenceTransformer", return_value=model),
            pytest.raises(ModelContractError, match="nominal, not real"),
        ):
            load_bi_encoder("fake/null-prompt", revision=None)

    def test_an_empty_query_prompt_is_refused(self) -> None:
        """An empty prefix is no prefix, however it is spelled."""
        model = FakeSentenceEncoder(prompts={"query": "", "document": ""})
        with (
            patch("sentence_transformers.SentenceTransformer", return_value=model),
            pytest.raises(ModelContractError, match="nominal, not real"),
        ):
            load_bi_encoder("fake/empty-prompt", revision=None)

    def test_the_refusal_reports_the_measured_cosine(self) -> None:
        """The number is the evidence, so it belongs in the message."""
        model = FakeSentenceEncoder(prompts={"query": "", "document": ""})
        with (
            patch("sentence_transformers.SentenceTransformer", return_value=model),
            pytest.raises(ModelContractError) as caught,
        ):
            load_bi_encoder("fake/empty-prompt", revision=None)
        assert "cosine 1.000000" in str(caught.value)

    def test_a_genuinely_asymmetric_model_passes(self) -> None:
        """The positive control, so the refusal is not vacuous."""
        with patch("sentence_transformers.SentenceTransformer", return_value=FakeSentenceEncoder()):
            assert load_bi_encoder("fake/asymmetric", revision=None).query_prompt


class TestDimensionIsReadNotAssumed:
    """D21's negative lesson: do not hardcode the embedding dimension."""

    def test_dimension_comes_from_the_model(self) -> None:
        """A 32-wide model must report 32, whatever the default happens to be."""
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=FakeSentenceEncoder(dimension=32),
        ):
            assert load_bi_encoder("fake/wide", revision=None).dimension == 32

    def test_a_declared_dimension_that_disagrees_with_reality_is_refused(self) -> None:
        """An index and a model that disagree would show nothing."""
        model = FakeSentenceEncoder(dimension=16)
        model.get_embedding_dimension = lambda: 1024  # type: ignore[method-assign]
        with (
            patch("sentence_transformers.SentenceTransformer", return_value=model),
            pytest.raises(ModelContractError, match="declares dimension 1024"),
        ):
            load_bi_encoder("fake/liar", revision=None)

    def test_a_model_declaring_no_dimension_is_refused(self) -> None:
        """Guessing here is how an index silently stops matching its model."""
        model = FakeSentenceEncoder()
        model.get_embedding_dimension = lambda: 0  # type: ignore[method-assign]
        with (
            patch("sentence_transformers.SentenceTransformer", return_value=model),
            pytest.raises(ModelContractError, match="does not report an embedding"),
        ):
            load_bi_encoder("fake/silent", revision=None)


class TestNormalizationIsMeasured:
    """Whether cosine equals dot product is measured, not read off a config."""

    def test_normalized_model_reports_l2(self) -> None:
        """The chosen bi-encoder ships a Normalize module."""
        with patch("sentence_transformers.SentenceTransformer", return_value=FakeSentenceEncoder()):
            assert load_bi_encoder("fake/normalized", revision=None).normalization == "l2"

    def test_unnormalized_model_reports_none(self) -> None:
        """Reporting 'l2' here would make every later cosine wrong."""
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=FakeSentenceEncoder(normalize=False),
        ):
            assert load_bi_encoder("fake/raw", revision=None).normalization == "none"


class TestRevisionPinning:
    """Embeddings from two revisions must never be silently compared."""

    def test_a_pinned_revision_is_recorded(self) -> None:
        """The revision travels into every provenance built from this model."""
        with patch("sentence_transformers.SentenceTransformer", return_value=FakeSentenceEncoder()):
            assert load_bi_encoder("fake/m", revision="abc123").revision == "abc123"

    def test_running_unpinned_warns(self, caplog) -> None:
        """Unpinned is allowed and must not be quiet."""
        with (
            patch("sentence_transformers.SentenceTransformer", return_value=FakeSentenceEncoder()),
            caplog.at_level("WARNING"),
        ):
            load_bi_encoder("fake/m", revision=None)
        assert "without a pinned revision" in caplog.text

    def test_the_revision_reaches_the_loader(self) -> None:
        """A recorded revision that was not requested would be a lie."""
        with patch("sentence_transformers.SentenceTransformer") as constructor:
            constructor.return_value = FakeSentenceEncoder()
            load_bi_encoder("fake/m", revision="deadbeef")
        assert constructor.call_args.kwargs["revision"] == "deadbeef"


class TestCrossEncoderLoad:
    """Stage two's recorded facts, and the sigmoid warning."""

    def test_default_instruction_and_activation_are_recorded(self) -> None:
        """Both bear on whether the scores mean what they appear to."""
        with patch("sentence_transformers.CrossEncoder", return_value=FakePairScorer()):
            scorer = load_cross_encoder("fake/ce", revision="c0ffee")
        assert scorer.activation == "Identity"
        assert scorer.default_prompt == "Given a web search query"
        assert scorer.revision == "c0ffee"

    def test_a_sigmoid_activation_warns_about_half_precision(self, caplog) -> None:
        """A sigmoid in bfloat16 saturates and randomizes the top candidates.

        Upstream measured NanoBEIR mean NDCG@10 collapsing from 0.6795 to
        0.1849 on exactly this. The chosen model uses Identity, so this path
        is a guard for a future model swap rather than for today's default.
        """
        model = FakePairScorer()
        model.activation_fn = Sigmoid()
        with (
            patch("sentence_transformers.CrossEncoder", return_value=model),
            caplog.at_level("WARNING"),
        ):
            load_cross_encoder("fake/sigmoid", revision=None)
        assert "randomizes their order" in caplog.text
