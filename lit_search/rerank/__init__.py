"""Two-stage semantic re-ranking: a bi-encoder recalls, a cross-encoder rescores.

Why this exists
---------------

A search that returns 400 papers in arbitrary order is a different tool from
one that returns them ordered by relevance to the actual question.  More
sharply: the literature failure that motivated this project was
**terminological** -- the same idea had been published in a neighbouring
research community's vocabulary, and keyword search organised around one
community's terms could not reach it however thorough it was.  Semantic
scoring compares meaning rather than strings, so it is one of the two
mechanisms that can cross that boundary at all.

That makes ranking quality load bearing rather than cosmetic, and it makes the
failure mode dangerous: a badly ranked list looks exactly like a well ranked
one.  Almost everything in this package exists to make one particular silent
degradation loud.

The four things that would otherwise fail silently
--------------------------------------------------

1. **A bi-encoder with no real query mode.**  ``encode_query()`` applies the
   query prefix only if the model ships a ``query`` prompt, and otherwise
   falls through to a symmetric encode with no error.
   :func:`~lit_search.rerank.models.load_bi_encoder` asserts the prompt exists
   and refuses the model when it does not.
2. **Confusing the two operations.**  Ranking against a question and ranking
   against a seed paper need different prompting.  They are two methods taking
   different types, and the mode is never a parameter.  See
   :class:`~lit_search.rerank.pipeline.Reranker`.
3. **Scoring a record on a fragment.**  Records routinely arrive with no
   abstract; one scored on a bare title sinks for a reason unrelated to its
   relevance.  The text policy is explicit and every degraded record is named.
   See :mod:`lit_search.rerank.text`.
4. **An order that depends on nothing recorded.**  Model identity, pinned
   revision, the literal prompt strings, normalization, text policy, and the
   degraded set all travel with the ranking in a
   :class:`~lit_search.store.order.RankingProvenance`.

Quick use
---------

.. code-block:: python

    from lit_search.rerank import Reranker

    reranker = Reranker.load()
    ranking = reranker.rank_against_question(papers, "how are specifications inferred "
                                                     "from call sites")
    print(ranking.describe())
    for paper in ranking.reading_order()[:20]:
        print(paper.title)
"""

from lit_search.rerank.models import (
    DEFAULT_BI_ENCODER,
    DEFAULT_BI_ENCODER_REVISION,
    DEFAULT_CROSS_ENCODER,
    DEFAULT_CROSS_ENCODER_REVISION,
    BiEncoder,
    CrossEncoderModel,
    load_bi_encoder,
    load_cross_encoder,
)
from lit_search.rerank.pipeline import (
    DEFAULT_TOP_K,
    RankedSet,
    Reranker,
    TwoStageRanking,
)
from lit_search.rerank.stages import (
    SCORING_COSINE,
    SCORING_CROSS_ENCODER_LOGIT,
    SEED_INSTRUCTION,
    BiEncoderStage,
    CrossEncoderStage,
    order_by_score,
)
from lit_search.rerank.text import (
    MIN_USABLE_ABSTRACT_CHARS,
    TEXT_POLICY_ID,
    DocumentText,
    TextQuality,
    build_document_text,
    build_documents,
    degraded_identities,
    paper_identity,
    quality_counts,
)

__all__ = [
    "DEFAULT_BI_ENCODER",
    "DEFAULT_BI_ENCODER_REVISION",
    "DEFAULT_CROSS_ENCODER",
    "DEFAULT_CROSS_ENCODER_REVISION",
    "DEFAULT_TOP_K",
    "MIN_USABLE_ABSTRACT_CHARS",
    "SCORING_COSINE",
    "SCORING_CROSS_ENCODER_LOGIT",
    "SEED_INSTRUCTION",
    "TEXT_POLICY_ID",
    "BiEncoder",
    "BiEncoderStage",
    "CrossEncoderModel",
    "CrossEncoderStage",
    "DocumentText",
    "RankedSet",
    "Reranker",
    "TextQuality",
    "TwoStageRanking",
    "build_document_text",
    "build_documents",
    "degraded_identities",
    "load_bi_encoder",
    "load_cross_encoder",
    "order_by_score",
    "paper_identity",
    "quality_counts",
]
