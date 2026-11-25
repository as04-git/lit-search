"""The two stages: a bi-encoder that recalls and orders, then a rescorer.

Why two stages and not one
--------------------------

A bi-encoder embeds the question and each document *independently* and
compares the vectors.  That is what makes it cheap enough to run over a whole
candidate set, and it is also its ceiling: the document's embedding was
computed without ever seeing the question, so a term that matters only in the
context of this question cannot influence it.

A cross-encoder scores the pair *jointly* -- the question and the document go
through the model together, so attention runs across both.  It cannot be run
over thousands of candidates, and it does not need to be.  So the shape is
recall broadly and cheaply, then rescore the head expensively.

The stages are kept separate rather than fused, and each records its own
provenance, because they make different claims and either can be swapped.

Ordering is deterministic, on purpose
-------------------------------------

Scores computed in bfloat16 tie often.  Measured during design: over 32
near-duplicate documents the reranker produced only 8 distinct scores.  That
is expected behaviour on near-identical text rather than a defect -- but
deduplication residue is made of exactly such near-duplicates, so ties are
the common case in the place it matters most.

Leaving tied records to sort stability would make the output depend on input
order, and a handle is a *set*: the same set assembled in a different order
would rank differently for no stated reason.  :func:`order_by_score`
therefore breaks ties on the record's stable identity, so the same set always
produces the same ranking.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from lit_search.rerank.text import DocumentText

logger = logging.getLogger(__name__)

#: Scoring convention recorded for stage one.
SCORING_COSINE = "cosine"
#: Scoring convention recorded for stage two.  A logit, not a probability:
#: the chosen model's output activation is Identity.
SCORING_CROSS_ENCODER_LOGIT = "cross-encoder-logit"

#: Default encode batch size.  Must stay small enough that a batch of
#: longest-tail documents fits in VRAM: at 32, a batch of ~2.2k-token abstracts
#: overflowed an 8 GiB card and the Windows/WSL driver silently spilled to
#: shared system memory instead of raising -- 61s instead of 2.4s for the same
#: 33 documents, at a reported 100% GPU utilization (measured 2026-08-31 on
#: a 33-document benchmark corpus).  The earlier "400 abstracts in 3.0s at 32" measurement
#: used short uniform texts and missed the length tail.
DEFAULT_ENCODE_BATCH = 8
#: Default pair-scoring batch size.  Smaller because a pair carries both texts
#: and the cross-encoder's sequence budget is far larger than the embedder's.
DEFAULT_PAIR_BATCH = 16

#: Instruction used when a *seed paper* is rescored against candidates.  The
#: chosen cross-encoder's own default instruction says "given a web search
#: query", which is a false description of a document-to-document comparison,
#: so seed-mode supplies its own and records it.
SEED_INSTRUCTION = (
    "Given a research paper, retrieve other research papers that address the same "
    "problem or make a closely related technical contribution"
)


def order_by_score(identities: Sequence[str], scores: Sequence[float]) -> list[int]:
    """Return indices ordered by descending score, ties broken deterministically.

    Parameters
    ----------
    identities : Sequence[str]
        Stable identity per item, used as the tie-break key.
    scores : Sequence[float]
        Score per item, parallel to *identities*.

    Returns
    -------
    list[int]
        Indices into the inputs, best first.

    Raises
    ------
    ValueError
        If the two sequences differ in length.
    """
    if len(identities) != len(scores):
        raise ValueError(
            f"order_by_score received {len(identities)} identity(ies) and {len(scores)} score(s)"
        )
    return sorted(range(len(scores)), key=lambda index: (-scores[index], identities[index]))


def _cosine(left: np.ndarray, right: np.ndarray, *, normalized: bool) -> list[float]:
    """Compute cosine similarity of one vector against a matrix of vectors.

    Parameters
    ----------
    left : np.ndarray
        Single vector, shape ``(dim,)``.
    right : np.ndarray
        Matrix, shape ``(n, dim)``.
    normalized : bool
        Whether both sides are already L2-normalized, in which case the dot
        product *is* the cosine and dividing again would be wrong-ish noise.

    Returns
    -------
    list[float]
        One similarity per row of *right*.
    """
    query = np.asarray(left, dtype=np.float64)
    documents = np.asarray(right, dtype=np.float64)
    if not normalized:
        query_norm = np.linalg.norm(query) or 1.0
        document_norms = np.linalg.norm(documents, axis=1)
        document_norms[document_norms == 0] = 1.0
        return [float(value) for value in (documents @ query) / (document_norms * query_norm)]
    return [float(value) for value in documents @ query]


class BiEncoderStage:
    """Stage one: embed and order the whole candidate set.

    Two methods, deliberately not one.  Which method is called decides whether
    the query side is encoded as a *query* or as a *document*, and that is the
    difference between query-to-document retrieval and document-to-document
    similarity.  A ``mode`` parameter on a single method would put that choice
    one keyword away from being wrong.
    """

    def __init__(self, encoder: object, *, batch_size: int = DEFAULT_ENCODE_BATCH) -> None:
        """Wrap a loaded bi-encoder.

        Parameters
        ----------
        encoder : object
            A :class:`~lit_search.rerank.models.BiEncoder`.
        batch_size : int
            Encode batch size.
        """
        from lit_search.rerank.models import BiEncoder

        if not isinstance(encoder, BiEncoder):
            raise TypeError(f"BiEncoderStage needs a BiEncoder, got {type(encoder).__name__}")
        self.encoder: BiEncoder = encoder
        self.batch_size = batch_size

    @property
    def _normalized(self) -> bool:
        """Report whether the model returns L2-normalized vectors.

        Returns
        -------
        bool
            ``True`` when normalization was measured as ``"l2"`` at load.
        """
        return self.encoder.normalization == "l2"

    def _documents(self, documents: Sequence[DocumentText]) -> np.ndarray:
        """Embed candidate text on the document side.

        Parameters
        ----------
        documents : Sequence[DocumentText]
            Built candidate text.

        Returns
        -------
        np.ndarray
            Matrix of document embeddings.
        """
        return np.asarray(
            self.encoder.model.encode_document(
                [document.text for document in documents],
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        )

    def score_against_question(
        self, question: str, documents: Sequence[DocumentText]
    ) -> list[float]:
        """Score candidates against a natural-language question.

        The question goes through ``encode_query`` and the candidates through
        ``encode_document``.  This is the asymmetric path, and it is the only
        method that uses the query prompt.

        Parameters
        ----------
        question : str
            The research question.
        documents : Sequence[DocumentText]
            Built candidate text.

        Returns
        -------
        list[float]
            Cosine similarity per candidate, in input order.
        """
        query_vector = np.asarray(
            self.encoder.model.encode_query([question], show_progress_bar=False)
        )[0]
        return _cosine(query_vector, self._documents(documents), normalized=self._normalized)

    def score_against_document(
        self, seed: DocumentText, documents: Sequence[DocumentText]
    ) -> list[float]:
        """Score candidates against a seed *document*.

        Both sides go through ``encode_document``.  Using the query prompt for
        one side here would compare a query-space vector against
        document-space vectors, which is the mirror image of the defect that
        made the stack's previous re-rank script unusable.

        Parameters
        ----------
        seed : DocumentText
            Built text of the seed paper.
        documents : Sequence[DocumentText]
            Built candidate text.

        Returns
        -------
        list[float]
            Cosine similarity per candidate, in input order.
        """
        seed_vector = np.asarray(
            self.encoder.model.encode_document([seed.text], show_progress_bar=False)
        )[0]
        return _cosine(seed_vector, self._documents(documents), normalized=self._normalized)


class CrossEncoderStage:
    """Stage two: rescore a shortlist by scoring pairs jointly."""

    def __init__(self, scorer: object, *, batch_size: int = DEFAULT_PAIR_BATCH) -> None:
        """Wrap a loaded cross-encoder.

        Parameters
        ----------
        scorer : object
            A :class:`~lit_search.rerank.models.CrossEncoderModel`.
        batch_size : int
            Pair-scoring batch size.
        """
        from lit_search.rerank.models import CrossEncoderModel

        if not isinstance(scorer, CrossEncoderModel):
            raise TypeError(
                f"CrossEncoderStage needs a CrossEncoderModel, got {type(scorer).__name__}"
            )
        self.scorer: CrossEncoderModel = scorer
        self.batch_size = batch_size

    def score(
        self,
        left: str,
        documents: Sequence[DocumentText],
        *,
        instruction: str | None = None,
    ) -> list[float]:
        """Score one text against many candidates, jointly.

        Parameters
        ----------
        left : str
            Question text, or the seed paper's built text.
        documents : Sequence[DocumentText]
            Built candidate text.
        instruction : str | None
            Instruction to apply.  ``None`` uses the model's own default,
            which is a query-retrieval instruction; seed-mode passes
            :data:`SEED_INSTRUCTION` instead.  Note that passing the empty
            string does **not** suppress the default -- verified 2026-08-31,
            an empty instruction produced scores identical to the default --
            so ``None`` and ``""`` are not interchangeable and only ``None``
            means "use the default".

        Returns
        -------
        list[float]
            One raw logit per candidate, in input order.
        """
        if not documents:
            return []
        pairs = [(left, document.text) for document in documents]
        kwargs: dict[str, object] = {
            "batch_size": self.batch_size,
            "show_progress_bar": False,
        }
        if instruction is not None:
            kwargs["prompt"] = instruction
        scores = self.scorer.model.predict(pairs, **kwargs)
        return [float(value) for value in scores]
