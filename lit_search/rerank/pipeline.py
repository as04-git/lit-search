"""The two re-rank operations, and the result they produce.

Two operations, not one with a switch
-------------------------------------

Ranking a set against a **question** is query-to-document retrieval.  Ranking
a set against a **seed paper** is document-to-document similarity.  They need
different prompting on the query side, they mean different things, and a badly
ranked list is indistinguishable from a well ranked one -- so conflating them
is a silent failure, not a cosmetic one.

They are therefore two methods with different parameter *types*:
:meth:`Reranker.rank_against_question` takes a ``str`` and
:meth:`Reranker.rank_against_seed_paper` takes a
:class:`~lit_search.core.paper.Paper`.  The mode is decided by which method
was called and is never accepted as an argument, so there is no keyword to get
wrong.

Why the result refuses to produce one score list
------------------------------------------------

Stage one scores every candidate by cosine against an embedding.  Stage two
scores the head by a cross-encoder logit.  Those numbers are not on a common
scale and no monotone patch makes them comparable -- a fused list would be a
fabricated ordering claim over the tail.

:class:`TwoStageRanking` therefore holds **two** ranked sets and offers
:meth:`TwoStageRanking.reading_order` for the human-facing sequence, without
ever offering a single fused score vector.  That maps exactly onto how the
handle store wants this written down: ``reranked`` requires a permutation of
the base set, and a top-k is taken by composing ``reranked`` with
``filtered`` so the truncation lands in the lineage rather than hiding inside
an operation that calls itself an ordering.

Deliberately absent
-------------------

**No embedding cache.**  The reference two-stage implementation this shape was
mined from caches query embeddings by hash.  Here that would save seconds:
five rounds over overlapping sets of roughly 400 papers is about 2,000
embeddings, and 400 abstracts embed in 3.0 seconds.  The persistent-embedding
question belongs to the vector-index stage, and building a second cache
alongside it would be two answers to one question.

**No store writes.**  Nothing here touches the handle store.  Set operations
on a handle derived from a real search currently raise without a coverage
propagator, and membership-reducing operations do not yet floor coverage --
both are open items owned by another stage.  Wiring re-rank into the store
before they are settled would either not run or would launder a truncated
set.  :class:`RankedSet` carries exactly the three things ``reranked`` needs,
so that wiring is mechanical once those land.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from lit_search.core.paper import Paper
from lit_search.exceptions import RerankInputError
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
from lit_search.rerank.stages import (
    SCORING_COSINE,
    SCORING_CROSS_ENCODER_LOGIT,
    SEED_INSTRUCTION,
    BiEncoderStage,
    CrossEncoderStage,
    order_by_score,
)
from lit_search.rerank.text import (
    TEXT_POLICY_ID,
    DocumentText,
    build_document_text,
    build_documents,
    degraded_identities,
    quality_counts,
)
from lit_search.store.order import RankingProvenance, RankMode

logger = logging.getLogger(__name__)

#: How many candidates stage two rescores by default.  Chosen because the
#: cross-encoder handles this many in about a second on the target card, and
#: because a reading list longer than this is not read.
DEFAULT_TOP_K = 100


@dataclass(frozen=True)
class RankedSet:
    """One ranked layer: a permutation of some set, its scores, its claim.

    Carries precisely the three arguments
    :func:`lit_search.store.operations.reranked` requires, so that a ranking
    can be committed to a handle without restating anything.

    Attributes
    ----------
    papers : tuple[Paper, ...]
        The members in ranked order, best first.
    scores : tuple[float, ...]
        One score per member, parallel to *papers*.
    identities : tuple[str, ...]
        Unique operation-local occurrence identities, parallel to *papers*.
        These are carried through both stages rather than reconstructed from
        titles after the order changes.
    provenance : RankingProvenance
        What the order claims and how it was produced.
    """

    papers: tuple[Paper, ...]
    scores: tuple[float, ...]
    provenance: RankingProvenance
    identities: tuple[str, ...]
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        """Reject a layer whose scores do not line up with its members.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the counts differ.
        """
        if len(self.papers) != len(self.scores):
            raise ValueError(
                f"RankedSet has {len(self.papers)} member(s) and {len(self.scores)} score(s)"
            )
        if len(self.papers) != len(self.identities):
            raise ValueError(
                f"RankedSet has {len(self.papers)} member(s) and "
                f"{len(self.identities)} occurrence identity value(s)"
            )
        if len(set(self.identities)) != len(self.identities):
            raise ValueError(
                "RankedSet occurrence identities must be unique; duplicate identities "
                "would let a score or document overwrite another member"
            )


@dataclass(frozen=True)
class TwoStageRanking:
    """The output of a two-stage re-rank.

    Attributes
    ----------
    mode : RankMode
        Which operation produced this.
    recall : RankedSet
        Stage one over the whole candidate set.
    rescored : RankedSet | None
        Stage two over the head, or ``None`` when stage two did not run.  The
        ``None`` is informative: it says the ranking rests on independent
        embeddings alone.
    top_k : int | None
        Size of the shortlist stage two was given, or ``None``.
    text_policy : str
        Identifier of the field policy the scored text was built under.
    degraded_text : tuple[str, ...]
        Identities of candidates whose text was incomplete.
    quality : dict[str, int]
        Candidate counts by text-quality label.
    dimension : int
        Embedding width, read from the model.
    """

    mode: RankMode
    recall: RankedSet
    rescored: RankedSet | None
    top_k: int | None
    text_policy: str
    degraded_text: tuple[str, ...]
    quality: dict[str, int] = field(default_factory=dict)
    dimension: int = 0

    @property
    def stages_run(self) -> int:
        """Report how many stages actually contributed to the order.

        Returns
        -------
        int
            ``1`` or ``2``.
        """
        return 1 if self.rescored is None else 2

    def reading_order(self) -> tuple[Paper, ...]:
        """Return the sequence a human should read, best first.

        The rescored head comes first in cross-encoder order, then the
        remainder in bi-encoder order.  No score accompanies this sequence,
        because the two halves were scored on different scales and presenting
        them as one number would be an ordering claim nothing supports.

        Returns
        -------
        tuple[Paper, ...]
            Every candidate, exactly once.
        """
        if self.rescored is None:
            return self.recall.papers
        head_identities = set(self.rescored.identities)
        tail = tuple(
            paper
            for paper, identity in zip(self.recall.papers, self.recall.identities, strict=True)
            if identity not in head_identities
        )
        return self.rescored.papers + tail

    def reading_order_identities(self) -> tuple[str, ...]:
        """Return occurrence identities parallel to :meth:`reading_order`.

        Returns
        -------
        tuple[str, ...]
            Every stage-one identity exactly once, in human reading order.
        """
        if self.rescored is None:
            return self.recall.identities
        head_identities = set(self.rescored.identities)
        tail = tuple(
            identity for identity in self.recall.identities if identity not in head_identities
        )
        return self.rescored.identities + tail

    def describe(self) -> str:
        """Render a short report of what this ranking is and is not.

        Every degradation the pipeline knows about appears here, so that a
        caller printing the result cannot show the order without showing what
        qualifies it.

        Returns
        -------
        str
            Multi-line human-readable summary.
        """
        final = self.rescored if self.rescored is not None else self.recall
        lines = [
            f"mode           : {self.mode.value}",
            f"stages         : {self.stages_run}",
            f"candidates     : {len(self.recall.papers)}",
            f"stage 1        : {self.recall.provenance.model_id} "
            f"({self.recall.provenance.scoring_convention}, "
            f"{self.recall.elapsed_seconds:.2f}s)",
        ]
        if self.rescored is None:
            lines.append("stage 2        : did not run; order rests on independent embeddings")
        else:
            lines.append(
                f"stage 2        : {self.rescored.provenance.model_id} "
                f"({self.rescored.provenance.scoring_convention}, top {self.top_k}, "
                f"{self.rescored.elapsed_seconds:.2f}s)"
            )
        lines.append(f"prompting      : {final.provenance.prompt_convention}")
        lines.append(f"text policy    : {self.text_policy}")
        lines.append(f"embedding dim  : {self.dimension} (read from model)")
        lines.append(f"normalization  : {self.recall.provenance.normalization}")
        degraded = len(self.degraded_text)
        if degraded:
            shown = ", ".join(self.degraded_text[:5])
            suffix = ", ..." if degraded > 5 else ""
            lines.append(
                f"degraded text  : {degraded} of {len(self.recall.papers)} scored on "
                f"incomplete text ({shown}{suffix})"
            )
        else:
            lines.append("degraded text  : none")
        return "\n".join(lines)


class Reranker:
    """Holds the loaded models and exposes the two re-rank operations.

    Models are held rather than loaded per call because loading is seconds and
    scoring is milliseconds, and because both models fit alongside each other
    on the target card: measured 2.25 GiB peak of 7.996 GiB with both resident
    in bfloat16 under sentence-transformers 6.0.1 and torch 2.13.0+cu130 on
    2026-08-31.
    """

    def __init__(
        self,
        bi_encoder: BiEncoder,
        cross_encoder: CrossEncoderModel | None = None,
    ) -> None:
        """Wrap loaded models.

        Parameters
        ----------
        bi_encoder : BiEncoder
            Stage one model, already validated by
            :func:`~lit_search.rerank.models.load_bi_encoder`.
        cross_encoder : CrossEncoderModel | None
            Stage two model.  ``None`` runs stage one only, which is recorded
            in the result rather than assumed by the caller.
        """
        self.bi_encoder = bi_encoder
        self.cross_encoder = cross_encoder
        self._recall = BiEncoderStage(bi_encoder)
        self._rescore = CrossEncoderStage(cross_encoder) if cross_encoder is not None else None

    @classmethod
    def load(
        cls,
        *,
        bi_encoder_id: str = DEFAULT_BI_ENCODER,
        bi_encoder_revision: str | None = DEFAULT_BI_ENCODER_REVISION,
        cross_encoder_id: str = DEFAULT_CROSS_ENCODER,
        cross_encoder_revision: str | None = DEFAULT_CROSS_ENCODER_REVISION,
        device: str | None = None,
        dtype: str | None = "bfloat16",
        with_cross_encoder: bool = True,
    ) -> Reranker:
        """Load both models with the project's pinned defaults.

        ``dtype`` defaults to bfloat16 because it was measured safe for these
        two models specifically: the cross-encoder's output activation is
        ``Identity``, so the half-precision sigmoid collapse does not apply,
        and a design-time comparison found the top 10 identical to float32 at
        Spearman 0.9959 for half the memory.  That safety is a property of
        *these* models; a model with a sigmoid head needs float32, and
        :func:`~lit_search.rerank.models.load_cross_encoder` warns when it
        sees one.

        Parameters
        ----------
        bi_encoder_id : str
            Bi-encoder repository identifier.
        bi_encoder_revision : str | None
            Revision to pin.  ``None`` runs unpinned and is logged as a
            warning, because embeddings from two revisions must never be
            compared and nothing downstream could tell them apart.
        cross_encoder_id : str
            Cross-encoder repository identifier.
        cross_encoder_revision : str | None
            Revision to pin, or ``None``.
        device : str | None
            Torch device string.  ``None`` lets sentence-transformers choose.
        dtype : str | None
            Torch dtype name.  ``None`` uses each model's own default.
        with_cross_encoder : bool
            Load stage two.  ``False`` gives a one-stage reranker, and the
            result records that stage two did not run.

        Returns
        -------
        Reranker
            Ready to rank.
        """
        bi = load_bi_encoder(
            bi_encoder_id, revision=bi_encoder_revision, device=device, dtype=dtype
        )
        cross: CrossEncoderModel | None = None
        if with_cross_encoder:
            cross = load_cross_encoder(
                cross_encoder_id, revision=cross_encoder_revision, device=device, dtype=dtype
            )
        return cls(bi, cross)

    def _provenance(
        self,
        *,
        mode: RankMode,
        model_id: str,
        revision: str | None,
        prompt_convention: str,
        scoring_convention: str,
        normalization: str,
        degraded: tuple[str, ...],
        query_text: str | None = None,
        seed_identity: str | None = None,
    ) -> RankingProvenance:
        """Assemble a ranking provenance with this project's required fields.

        Parameters
        ----------
        mode : RankMode
            Query or seed.
        model_id : str
            Model that produced the scores.
        revision : str | None
            Pinned revision, or ``None``.
        prompt_convention : str
            Literal prompting in force.
        scoring_convention : str
            How a score was computed.
        normalization : str
            What was done to vectors before comparison.
        degraded : tuple[str, ...]
            Identities scored on incomplete text.
        query_text : str | None
            The question, for query mode.
        seed_identity : str | None
            The seed's identity, for seed mode.

        Returns
        -------
        RankingProvenance
            Fully populated provenance.
        """
        return RankingProvenance(
            mode=mode,
            model_id=model_id,
            model_revision=revision,
            prompt_convention=prompt_convention,
            scoring_convention=scoring_convention,
            normalization=normalization,
            text_policy=TEXT_POLICY_ID,
            degraded_text=degraded,
            # Both a cosine against a fixed vector and a cross-encoder logit
            # depend only on the pair, never on which other candidates were
            # in the batch, so scores from two such runs are comparable.
            pool_independent=True,
            query_text=query_text,
            seed_identity=seed_identity,
        )

    def _recall_layer(
        self,
        papers: Sequence[Paper],
        documents: Sequence[DocumentText],
        scores: Sequence[float],
        provenance: RankingProvenance,
        elapsed: float,
    ) -> RankedSet:
        """Order a candidate set by stage-one scores.

        Parameters
        ----------
        papers : Sequence[Paper]
            Candidates in input order.
        documents : Sequence[DocumentText]
            Built text, parallel to *papers*, supplying tie-break identities.
        scores : Sequence[float]
            Stage-one scores, parallel to *papers*.
        provenance : RankingProvenance
            Stage-one claim.
        elapsed : float
            Wall-clock seconds spent scoring.

        Returns
        -------
        RankedSet
            Ranked stage-one layer.
        """
        identities = [document.identity for document in documents]
        order = order_by_score(identities, scores)
        return RankedSet(
            papers=tuple(papers[index] for index in order),
            scores=tuple(float(scores[index]) for index in order),
            provenance=provenance,
            identities=tuple(identities[index] for index in order),
            elapsed_seconds=elapsed,
        )

    def _rescore_layer(
        self,
        recall: RankedSet,
        documents_by_identity: dict[str, DocumentText],
        *,
        left: str,
        instruction: str | None,
        top_k: int,
        provenance_factory: object,
    ) -> RankedSet | None:
        """Rescore the head of a stage-one ranking, if stage two is available.

        Parameters
        ----------
        recall : RankedSet
            Stage-one ranking.
        documents_by_identity : dict[str, DocumentText]
            Built text, looked up by identity.
        left : str
            Question text or seed text.
        instruction : str | None
            Cross-encoder instruction; ``None`` means the model default.
        top_k : int
            Shortlist size.
        provenance_factory : object
            Callable taking the prompt convention and returning the stage-two
            provenance.

        Returns
        -------
        RankedSet | None
            Stage-two layer, or ``None`` when stage two is not loaded.
        """
        if self._rescore is None or self.cross_encoder is None or not recall.papers:
            return None
        head_papers = recall.papers[:top_k]
        head_identities = recall.identities[:top_k]
        head_documents = [documents_by_identity[identity] for identity in head_identities]
        started = time.perf_counter()
        scores = self._rescore.score(left, head_documents, instruction=instruction)
        elapsed = time.perf_counter() - started
        identities = [document.identity for document in head_documents]
        order = order_by_score(identities, scores)
        convention = self.cross_encoder.prompt_convention(instruction)
        provenance = provenance_factory(convention)  # type: ignore[operator]
        return RankedSet(
            papers=tuple(head_papers[index] for index in order),
            scores=tuple(float(scores[index]) for index in order),
            provenance=provenance,
            identities=tuple(identities[index] for index in order),
            elapsed_seconds=elapsed,
        )

    def rank_against_question(
        self,
        papers: Sequence[Paper],
        question: str,
        *,
        top_k: int = DEFAULT_TOP_K,
    ) -> TwoStageRanking:
        """Rank a candidate set against a research question.

        This is **query-to-document retrieval**.  The question is encoded
        through ``encode_query`` and therefore carries the model's query
        prefix; the candidates are encoded through ``encode_document``.

        Parameters
        ----------
        papers : Sequence[Paper]
            Candidates to order.  Every one is scored and every one is
            returned; a re-rank reorders a set and does not select from it.
        question : str
            The research question.  Usually narrower than the Boolean query
            that produced the set.
        top_k : int
            How many of the top candidates stage two rescores.

        Returns
        -------
        TwoStageRanking
            Both ranked layers, with provenance and degradation report.

        Raises
        ------
        RerankInputError
            If *question* is blank.  Ranking against nothing produces an order
            that looks like a ranking and means nothing.
        """
        text = question.strip()
        if not text:
            raise RerankInputError(
                "rank_against_question was given a blank question. An empty query still "
                "produces an embedding and therefore still produces a plausible-looking "
                "order, which is the failure this tool exists to prevent."
            )

        documents = build_documents(papers)
        degraded = degraded_identities(documents)

        started = time.perf_counter()
        scores = self._recall.score_against_question(text, documents) if documents else []
        elapsed = time.perf_counter() - started

        recall = self._recall_layer(
            papers,
            documents,
            scores,
            self._provenance(
                mode=RankMode.QUERY,
                model_id=self.bi_encoder.model_id,
                revision=self.bi_encoder.revision,
                prompt_convention=self.bi_encoder.prompt_convention(symmetric=False),
                scoring_convention=SCORING_COSINE,
                normalization=self.bi_encoder.normalization,
                degraded=degraded,
                query_text=text,
            ),
            elapsed,
        )

        rescored = self._rescore_layer(
            recall,
            {document.identity: document for document in documents},
            left=text,
            instruction=None,
            top_k=top_k,
            provenance_factory=lambda convention: self._provenance(
                mode=RankMode.QUERY,
                model_id=self.cross_encoder.model_id,  # type: ignore[union-attr]
                revision=self.cross_encoder.revision,  # type: ignore[union-attr]
                prompt_convention=convention,
                scoring_convention=SCORING_CROSS_ENCODER_LOGIT,
                normalization="n/a",
                degraded=degraded,
                query_text=text,
            ),
        )

        return self._result(RankMode.QUERY, recall, rescored, top_k, documents, degraded)

    def rank_against_seed_paper(
        self,
        papers: Sequence[Paper],
        seed: Paper,
        *,
        top_k: int = DEFAULT_TOP_K,
        use_cross_encoder: bool = False,
    ) -> TwoStageRanking:
        """Rank a candidate set by similarity to a seed *paper*.

        This is **document-to-document similarity**.  Both the seed and the
        candidates go through ``encode_document``; the query prefix is not
        applied to either, because neither side is a query.

        Stage two is **opt-in here and on by default in query mode**, and the
        asymmetry is deliberate.  The chosen cross-encoder is trained on
        query-document relevance and its own instruction says "given a web
        search query"; feeding it a full abstract as the left-hand side is off
        distribution and no measurement supports it.  When it is switched on,
        :data:`~lit_search.rerank.stages.SEED_INSTRUCTION` replaces that
        instruction and the substitution is recorded verbatim in the
        provenance, so the result can never be mistaken for a validated
        configuration.

        Parameters
        ----------
        papers : Sequence[Paper]
            Candidates to order.  If the seed is among them it will rank
            first against itself; drop it downstream rather than here, so the
            removal is visible.
        seed : Paper
            The paper to find neighbours of.
        top_k : int
            Shortlist size for stage two, when enabled.
        use_cross_encoder : bool
            Enable stage two.  Off by default for the reason above.

        Returns
        -------
        TwoStageRanking
            Both ranked layers, with provenance and degradation report.

        Raises
        ------
        RerankInputError
            If the seed has neither a title nor an abstract, leaving nothing
            to compare against.
        """
        seed_document = build_document_text(seed)
        if seed_document.quality.value == "empty":
            raise RerankInputError(
                "rank_against_seed_paper was given a seed with no title and no abstract. "
                "There is nothing to compare candidates against, and scoring anyway would "
                "order the set by an artifact of the model."
            )

        documents = build_documents(papers)
        degraded = degraded_identities(documents)

        started = time.perf_counter()
        scores = self._recall.score_against_document(seed_document, documents) if documents else []
        elapsed = time.perf_counter() - started

        recall = self._recall_layer(
            papers,
            documents,
            scores,
            self._provenance(
                mode=RankMode.SEED,
                model_id=self.bi_encoder.model_id,
                revision=self.bi_encoder.revision,
                prompt_convention=self.bi_encoder.prompt_convention(symmetric=True),
                scoring_convention=SCORING_COSINE,
                normalization=self.bi_encoder.normalization,
                degraded=degraded,
                seed_identity=seed_document.identity,
            ),
            elapsed,
        )

        rescored = None
        if use_cross_encoder:
            rescored = self._rescore_layer(
                recall,
                {document.identity: document for document in documents},
                left=seed_document.text,
                instruction=SEED_INSTRUCTION,
                top_k=top_k,
                provenance_factory=lambda convention: self._provenance(
                    mode=RankMode.SEED,
                    model_id=self.cross_encoder.model_id,  # type: ignore[union-attr]
                    revision=self.cross_encoder.revision,  # type: ignore[union-attr]
                    prompt_convention=convention,
                    scoring_convention=SCORING_CROSS_ENCODER_LOGIT,
                    normalization="n/a",
                    degraded=degraded,
                    seed_identity=seed_document.identity,
                ),
            )

        return self._result(
            RankMode.SEED,
            recall,
            rescored,
            top_k if use_cross_encoder else None,
            documents,
            degraded,
        )

    def _result(
        self,
        mode: RankMode,
        recall: RankedSet,
        rescored: RankedSet | None,
        top_k: int | None,
        documents: Sequence[DocumentText],
        degraded: tuple[str, ...],
    ) -> TwoStageRanking:
        """Assemble the result object and log the degradation summary.

        Parameters
        ----------
        mode : RankMode
            Which operation ran.
        recall : RankedSet
            Stage-one layer.
        rescored : RankedSet | None
            Stage-two layer.
        top_k : int | None
            Shortlist size, or ``None`` when stage two did not run.
        documents : Sequence[DocumentText]
            Built candidate text.
        degraded : tuple[str, ...]
            Degraded identities.

        Returns
        -------
        TwoStageRanking
            The result.
        """
        ranking = TwoStageRanking(
            mode=mode,
            recall=recall,
            rescored=rescored,
            top_k=top_k if rescored is not None else None,
            text_policy=TEXT_POLICY_ID,
            degraded_text=degraded,
            quality=quality_counts(documents),
            dimension=self.bi_encoder.dimension,
        )
        if degraded:
            logger.warning(
                "[rerank] %d of %d candidate(s) were scored on incomplete text under policy "
                "%s. A record with no abstract sinks for a reason unrelated to its "
                "relevance; the identities are recorded in the ranking provenance.",
                len(degraded),
                len(documents),
                TEXT_POLICY_ID,
            )
        return ranking
