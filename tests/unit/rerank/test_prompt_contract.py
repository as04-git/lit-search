"""Tests that fail when the prompt convention is wrong.

This is the file the specification's silent-failure watchlist asks for.  The
standard demanded is *"a test that fails on a wrong prefix"*, not a test that
passes with the right one -- so every test here is written so that removing or
corrupting the prefix handling turns it red.

The defect being guarded against, quoted from the installed
``sentence_transformers`` 6.0.1 source:

.. code-block:: python

    if prompt_name is None and prompt is None and "query" in self.prompts:
        prompt_name = "query"

If the model ships no ``query`` prompt, ``encode_query()`` applies no prefix
and performs a plain symmetric encode.  No exception, no warning, and a
confidently ordered list that is wrong.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from lit_search.exceptions import ModelContractError
from lit_search.rerank import BiEncoderStage, Reranker, build_documents, load_bi_encoder
from lit_search.rerank.models import DEFAULT_BI_ENCODER
from tests.unit.rerank.conftest import QWEN_QUERY_PROMPT, FakeSentenceEncoder

pytest.importorskip("sentence_transformers", reason="needs the rerank extra")


class TestQueryPromptIsMandatory:
    """A model with no query mode must be refused at load, not used."""

    def test_model_without_query_prompt_is_refused(self) -> None:
        """The assertion the whole design hangs on."""
        model = FakeSentenceEncoder(prompts={})
        with (
            patch("sentence_transformers.SentenceTransformer", return_value=model),
            pytest.raises(ModelContractError, match="no 'query' prompt"),
        ):
            load_bi_encoder("fake/symmetric-model", revision=None)

    def test_refusal_names_the_silent_no_op(self) -> None:
        """The message has to explain why, or it will be worked around."""
        model = FakeSentenceEncoder(prompts={"document": ""})
        with (
            patch("sentence_transformers.SentenceTransformer", return_value=model),
            pytest.raises(ModelContractError) as caught,
        ):
            load_bi_encoder("fake/symmetric-model", revision=None)
        message = str(caught.value)
        assert "symmetric encode" in message
        assert "adhoc_query" in message
        assert DEFAULT_BI_ENCODER in message

    def test_document_only_prompt_is_not_enough(self) -> None:
        """A document prompt does not supply a query mode."""
        model = FakeSentenceEncoder(prompts={"document": "passage: "})
        with (
            patch("sentence_transformers.SentenceTransformer", return_value=model),
            pytest.raises(ModelContractError),
        ):
            load_bi_encoder("fake/doc-only", revision=None)

    def test_model_with_query_prompt_loads(self) -> None:
        """The positive control, so the refusal is not vacuous."""
        model = FakeSentenceEncoder()
        with patch("sentence_transformers.SentenceTransformer", return_value=model):
            encoder = load_bi_encoder("fake/asymmetric", revision="abc123")
        assert encoder.query_prompt == QWEN_QUERY_PROMPT
        assert encoder.document_prompt == ""


class TestPrefixReachesTheModel:
    """The prefix must be applied to the query and not to the documents."""

    def test_query_text_reaches_the_model_with_the_query_prefix(
        self, bi_encoder, candidates
    ) -> None:
        """Fails if the stage stops routing the question through encode_query."""
        stage = BiEncoderStage(bi_encoder)
        documents = build_documents(candidates)
        stage.score_against_question("how are specifications inferred", documents)

        seen = bi_encoder.model.seen
        prefixed = [text for text in seen if text.startswith(QWEN_QUERY_PROMPT)]
        assert len(prefixed) == 1, "exactly the question should carry the query prefix"
        assert "how are specifications inferred" in prefixed[0]

    def test_documents_never_receive_the_query_prefix(self, bi_encoder, candidates) -> None:
        """Fails if documents are routed through encode_query."""
        stage = BiEncoderStage(bi_encoder)
        documents = build_documents(candidates)
        stage.score_against_question("how are specifications inferred", documents)

        document_texts = [
            text for text in bi_encoder.model.seen if not text.startswith(QWEN_QUERY_PROMPT)
        ]
        assert len(document_texts) == len(candidates)
        for text in document_texts:
            assert QWEN_QUERY_PROMPT not in text

    def test_seed_mode_applies_the_query_prefix_to_neither_side(
        self, bi_encoder, candidates
    ) -> None:
        """Document-to-document similarity must not use a query-space vector.

        This is the specific defect that made the stack's previous re-rank
        script unusable, in mirror image: it embedded a query symmetrically
        with a document-to-document model.  Here the risk is embedding a
        document with the query prefix.
        """
        stage = BiEncoderStage(bi_encoder)
        documents = build_documents(candidates)
        seed = build_documents([candidates[0]])[0]
        stage.score_against_document(seed, documents)

        assert bi_encoder.model.seen, "the stage must actually encode something"
        for text in bi_encoder.model.seen:
            assert not text.startswith(QWEN_QUERY_PROMPT)

    def test_wrong_prefix_changes_the_scores(self, bi_encoder, candidates) -> None:
        """The prefix is load bearing, so corrupting it must move the numbers.

        If this passes with the prefix mangled, the prefix is decorative and
        every other test in this file is worthless.
        """
        documents = build_documents(candidates)
        correct = BiEncoderStage(bi_encoder).score_against_question("infer a spec", documents)

        bi_encoder.model.prompts["query"] = "Represent this sentence: "
        corrupted = BiEncoderStage(bi_encoder).score_against_question("infer a spec", documents)

        assert correct != corrupted


class TestProvenanceRecordsTheLiteralPrompt:
    """A label cannot be checked afterwards; the literal text can."""

    def test_query_mode_records_both_literal_prompts(
        self, bi_encoder, cross_encoder, candidates
    ) -> None:
        """The stored provenance must carry the actual prefix strings."""
        ranking = Reranker(bi_encoder, cross_encoder).rank_against_question(
            candidates, "infer specifications from call sites"
        )
        convention = ranking.recall.provenance.prompt_convention
        assert "encode_query" in convention
        assert "Instruct: Given a web search query" in convention
        assert "\\n" in convention, "newlines are escaped so the record stays one line"

    def test_seed_mode_records_a_symmetric_convention(self, bi_encoder, candidates) -> None:
        """Seed mode must not claim a query prefix it did not apply."""
        ranking = Reranker(bi_encoder).rank_against_seed_paper(candidates, candidates[0])
        convention = ranking.recall.provenance.prompt_convention
        assert "both sides encode_document" in convention
        assert "Instruct:" not in convention

    def test_corrupted_prefix_is_visible_in_the_provenance(self, bi_encoder, candidates) -> None:
        """A wrong prefix has to be diagnosable from a stored handle alone."""
        bi_encoder = type(bi_encoder)(
            **{**bi_encoder.__dict__, "query_prompt": "Represent this sentence: "}
        )
        ranking = Reranker(bi_encoder).rank_against_question(candidates, "infer a spec")
        assert "Represent this sentence: " in ranking.recall.provenance.prompt_convention
