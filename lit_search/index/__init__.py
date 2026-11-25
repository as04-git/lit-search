"""The embedded vector and full-text index over papers.

Four pieces, in the order a query passes through them:

:mod:`lit_search.index.embedding`
    What a vector means -- the model, its revision, its dimension and its
    prompt convention -- and the refusals that keep two vector spaces from
    being mixed.
:mod:`lit_search.index.text`
    What string gets embedded, and which records were built from less text
    than the policy asks for.
:mod:`lit_search.index.paper_index`
    The LanceDB table, the full-text index that keyword retrieval requires,
    and the maintenance that keeps the unindexed tail at zero.
:mod:`lit_search.index.hybrid`
    Three retrieval lanes -- vector, BM25, literal -- fused so that each
    paper keeps the record of which lane found it.
:mod:`lit_search.index.mmr`
    Diversity in what is displayed, with the relevance order preserved
    underneath it and never used as a source of seeds.
"""

from lit_search.index.embedding import (
    DeterministicEmbedder,
    EmbeddingError,
    EmbeddingIdentity,
    EmbeddingModel,
    QueryEmbeddingCache,
    SentenceTransformerEmbedder,
)
from lit_search.index.hybrid import (
    Candidate,
    HybridResult,
    Lane,
    LaneOutcome,
    hybrid_search,
)
from lit_search.index.mmr import (
    DEFAULT_LAMBDA,
    DiversificationError,
    DiversifiedView,
    Suppression,
    diversify,
)
from lit_search.index.paper_index import (
    IndexHealth,
    IngestReport,
    PaperIndex,
    PaperIndexError,
)
from lit_search.index.text import (
    DEFAULT_TEXT_POLICY,
    PaperText,
    TextPolicy,
    build_paper_text,
)

__all__ = [
    "DEFAULT_LAMBDA",
    "DEFAULT_TEXT_POLICY",
    "Candidate",
    "DeterministicEmbedder",
    "DiversificationError",
    "DiversifiedView",
    "EmbeddingError",
    "EmbeddingIdentity",
    "EmbeddingModel",
    "HybridResult",
    "IndexHealth",
    "IngestReport",
    "Lane",
    "LaneOutcome",
    "PaperIndex",
    "PaperIndexError",
    "PaperText",
    "QueryEmbeddingCache",
    "SentenceTransformerEmbedder",
    "Suppression",
    "TextPolicy",
    "build_paper_text",
    "diversify",
    "hybrid_search",
]
