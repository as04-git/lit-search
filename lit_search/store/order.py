"""Ordered-versus-set semantics for handles.

A handle holds a *set* of papers.  Some handles additionally carry an *order*,
and an order is not a property of the papers -- it is a **claim** about them:

    these members are sequenced by relevance to question Q, as judged by
    model M, under scoring convention C.

That claim is only true of members that were actually scored under exactly
that (Q, M, C).  Every set operation therefore has to answer the question
"does the result still satisfy the claim?", and the honest answer is often no.

Three states, and the reason there are three
--------------------------------------------

``UNORDERED``
    Membership only.  The member sequence is a stable but arbitrary artifact
    of how the set was built and must never be presented as a ranking.

``RANKED``
    The member sequence is a ranking, backed by a
    :class:`RankingProvenance` and one score per member.

``INVALIDATED``
    The handle *was* ranked and an operation destroyed the guarantee.  The
    members and their sequence survive; the ranking claim does not.

Collapsing ``INVALIDATED`` into ``UNORDERED`` would be the same class of error
as recording a database that was never queried as one that found nothing: two
different facts written to the same value.  "This was never ranked" and "this
was ranked and then a union broke it" lead a researcher to different next
actions -- the second one says "re-rank this".  So the distinction is kept,
and the provenance of the destroyed ranking is kept with it.

Scores are dropped on invalidation
----------------------------------

An invalidated handle keeps ``previous_ranking`` but **not** the per-member
scores.  Retaining them would leave a live mechanism for the fabricated order
to come back -- any downstream "sort by score" would resurrect it.  Nothing is
lost by dropping them, because handles are immutable and the parent handle
still holds the ranking in full.

Pool independence
-----------------

Two rankings can only be merged by score if their scores mean the same thing
for a document considered on its own.  Cosine similarity against a fixed query
embedding and a cross-encoder logit are both *pool independent*: the score of
a document does not depend on which other documents were in the batch.
Rank-percentiles and maximal-marginal-relevance orderings are **not** -- MMR
scores a document partly by how unlike the already-selected documents it is,
so the same paper scores differently in a different pool.

Since diversified top-N is part of the intended retrieval design, the
pool-dependent case is the common one, and ``pool_independent`` is a required
field rather than an inferred one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from lit_search.exceptions import HandleSchemaError, OrderSemanticsError
from lit_search.store.strict import (
    reject_unknown_keys,
    require,
    require_enum,
    require_float_list,
    require_mapping,
    require_optional,
    require_str_list,
)


class OrderState(str, Enum):
    """Whether a handle's member sequence carries a ranking claim."""

    #: Membership only; the sequence is arbitrary.
    UNORDERED = "unordered"
    #: The sequence is a ranking backed by provenance and scores.
    RANKED = "ranked"
    #: The handle was ranked and an operation destroyed the claim.
    INVALIDATED = "invalidated"


class RankMode(str, Enum):
    """Which of the two re-ranking operations produced an order.

    The distinction is load bearing: ranking against a question is
    query-to-document retrieval and needs a model with a genuine asymmetric
    query mode, while ranking against a seed paper is document-to-document
    similarity.  Conflating them produces a badly ordered list that looks
    exactly like a well ordered one, so the mode is recorded rather than
    inferred.
    """

    #: Ranked against a natural-language question.
    QUERY = "query"
    #: Ranked against a seed paper.
    SEED = "seed"


_RANKING_KEYS = (
    "mode",
    "query_text",
    "seed_identity",
    "model_id",
    "model_revision",
    "prompt_convention",
    "scoring_convention",
    "normalization",
    "text_policy",
    "degraded_text",
    "pool_independent",
)


@dataclass(frozen=True)
class RankingProvenance:
    """Everything needed to know what a ranking claim actually asserts.

    Attributes
    ----------
    mode : RankMode
        Query-to-document or document-to-document.
    query_text : str | None
        The question the set was ranked against.  Required when *mode* is
        :attr:`RankMode.QUERY`.
    seed_identity : str | None
        Identifier (DOI, or title when no DOI exists) of the seed paper.
        Required when *mode* is :attr:`RankMode.SEED`.
    model_id : str
        Identifier of the model that produced the scores, for example
        ``"intfloat/e5-large-v2"``.
    model_revision : str | None
        Pinned model revision when one is known.  ``None`` records that the
        run did not pin one, which is itself worth knowing.
    prompt_convention : str
        How query and document text were prefixed, for example
        ``"e5:query:/passage:"``, ``"specter2:adhoc_query"``, or ``"none"``.
        Recorded because getting it wrong degrades retrieval silently and
        cannot be recovered from the model object afterwards.
    scoring_convention : str
        How a score was computed, for example ``"cosine"``,
        ``"cross-encoder-logit"``, or ``"mmr(lambda=0.5)"``.
    normalization : str
        What was done to the vectors before they were compared, for example
        ``"l2"`` or ``"none"``.  Recorded because whether cosine and dot
        product agree depends on it, and because embeddings produced under
        different normalizations must never be compared.  Use ``"n/a"`` for a
        scorer that produces no vectors, such as a cross-encoder.
    text_policy : str
        Which record fields were turned into the text that was scored, for
        example ``"title+abstract+keywords/labelled-lines/v1"``.  A ranking
        computed over titles alone is a different claim from one computed over
        full abstracts, and the difference is invisible in the scores.
    degraded_text : tuple[str, ...]
        Identities of the members whose text was incomplete -- typically a
        missing abstract.  Such a record sinks in the ranking for a reason
        that has nothing to do with its relevance, so recording which ones
        they were is the difference between an explainable low rank and a
        silently lost paper.  Excluded from equality: it describes the inputs,
        not the scoring configuration, and two rankings under the same
        configuration remain directly comparable.
    pool_independent : bool
        Whether a member's score is independent of which other members were
        in the pool.  Only pool-independent scores may be merged across sets.
    """

    mode: RankMode
    model_id: str
    prompt_convention: str
    scoring_convention: str
    normalization: str
    text_policy: str
    pool_independent: bool
    query_text: str | None = None
    seed_identity: str | None = None
    model_revision: str | None = None
    degraded_text: tuple[str, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        """Reject a provenance that does not identify what it ranked against.

        Returns
        -------
        None

        Raises
        ------
        OrderSemanticsError
            If the target for the declared mode is missing, or if a required
            descriptive field is blank.
        """
        if self.mode is RankMode.QUERY and not self.query_text:
            raise OrderSemanticsError("A query-mode ranking must record its query text")
        if self.mode is RankMode.SEED and not self.seed_identity:
            raise OrderSemanticsError("A seed-mode ranking must record its seed identity")
        if not self.model_id:
            raise OrderSemanticsError("A ranking must record the model that produced it")
        if not self.prompt_convention:
            raise OrderSemanticsError(
                "A ranking must record its prompt convention; use 'none' when the "
                "model takes unprefixed text"
            )
        if not self.scoring_convention:
            raise OrderSemanticsError("A ranking must record its scoring convention")
        if not self.normalization:
            raise OrderSemanticsError(
                "A ranking must record its normalization; use 'none' when vectors were "
                "compared unnormalized and 'n/a' when the scorer produces no vectors"
            )
        if not self.text_policy:
            raise OrderSemanticsError(
                "A ranking must record which record fields were turned into the text it "
                "scored; a ranking over titles alone is a different claim from one over "
                "abstracts and the scores do not show the difference"
            )

    def mergeable_with(self, other: RankingProvenance) -> bool:
        """Report whether scores from two rankings may be compared directly.

        Requires exact equality of every field *and* pool independence.  The
        check is deliberately strict: a near-match is a wrong answer that
        looks right, and the cost of being wrong here is a fabricated order,
        while the cost of being too strict is one re-rank.

        Parameters
        ----------
        other : RankingProvenance
            Provenance to compare against.

        Returns
        -------
        bool
            ``True`` when the two score sets are directly comparable.
        """
        return self.pool_independent and other.pool_independent and self == other

    def to_dict(self) -> dict[str, Any]:
        """Serialize this provenance to a plain dictionary.

        Returns
        -------
        dict[str, Any]
            JSON-ready representation.
        """
        return {
            "mode": self.mode.value,
            "query_text": self.query_text,
            "seed_identity": self.seed_identity,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "prompt_convention": self.prompt_convention,
            "scoring_convention": self.scoring_convention,
            "normalization": self.normalization,
            "text_policy": self.text_policy,
            "degraded_text": list(self.degraded_text),
            "pool_independent": self.pool_independent,
        }

    @classmethod
    def from_dict(cls, payload: Any, context: str) -> RankingProvenance:
        """Reconstruct a provenance, refusing anything unrecognised.

        Parameters
        ----------
        payload : Any
            Mapping produced by :meth:`to_dict`.
        context : str
            Dotted path of the payload, used in error messages.

        Returns
        -------
        RankingProvenance
            Reconstructed provenance.

        Raises
        ------
        HandleSchemaError
            If a key is missing, mistyped, or unrecognised.
        """
        data = require_mapping(payload, context)
        reject_unknown_keys(data, _RANKING_KEYS, context)
        return cls(
            mode=require_enum(data, "mode", RankMode, context),
            query_text=require_optional(data, "query_text", str, context),
            seed_identity=require_optional(data, "seed_identity", str, context),
            model_id=require(data, "model_id", str, context),
            model_revision=require_optional(data, "model_revision", str, context),
            prompt_convention=require(data, "prompt_convention", str, context),
            scoring_convention=require(data, "scoring_convention", str, context),
            normalization=require(data, "normalization", str, context),
            text_policy=require(data, "text_policy", str, context),
            degraded_text=tuple(require_str_list(data, "degraded_text", context)),
            pool_independent=require(data, "pool_independent", bool, context),
        )


_ORDER_KEYS = ("state", "ranking", "scores", "previous_ranking", "invalidated_by")


@dataclass(frozen=True)
class OrderRecord:
    """The order half of a handle: a state, and whatever backs it.

    Attributes
    ----------
    state : OrderState
        Whether the member sequence is a ranking.
    ranking : RankingProvenance | None
        What the ranking claims.  Present exactly when *state* is
        :attr:`OrderState.RANKED`.
    scores : tuple[float, ...] | None
        One score per member, parallel to the member sequence.  Present
        exactly when *state* is :attr:`OrderState.RANKED`.
    previous_ranking : RankingProvenance | None
        The claim that was destroyed.  Present exactly when *state* is
        :attr:`OrderState.INVALIDATED`.
    invalidated_by : str | None
        Name of the operation that destroyed the claim.  Present exactly when
        *state* is :attr:`OrderState.INVALIDATED`.
    """

    state: OrderState
    ranking: RankingProvenance | None = None
    scores: tuple[float, ...] | None = None
    previous_ranking: RankingProvenance | None = None
    invalidated_by: str | None = None

    def __post_init__(self) -> None:
        """Reject an order record whose fields contradict its state.

        Returns
        -------
        None

        Raises
        ------
        OrderSemanticsError
            If the fields present do not match the declared state.
        """
        if self.state is OrderState.RANKED:
            if self.ranking is None or self.scores is None:
                raise OrderSemanticsError("A ranked order must carry provenance and scores")
            if self.previous_ranking is not None or self.invalidated_by is not None:
                raise OrderSemanticsError("A ranked order must not carry invalidation fields")
        elif self.state is OrderState.INVALIDATED:
            if self.previous_ranking is None or not self.invalidated_by:
                raise OrderSemanticsError(
                    "An invalidated order must record the ranking it lost and the "
                    "operation that destroyed it"
                )
            if self.ranking is not None or self.scores is not None:
                raise OrderSemanticsError(
                    "An invalidated order must not retain provenance or scores; the "
                    "parent handle still holds them"
                )
        elif self.ranking is not None or self.scores is not None:
            raise OrderSemanticsError("An unordered order must not carry provenance or scores")
        elif self.previous_ranking is not None or self.invalidated_by is not None:
            raise OrderSemanticsError("An unordered order must not carry invalidation fields")

    @classmethod
    def unordered(cls) -> OrderRecord:
        """Return the order record for a set with no ranking claim.

        Returns
        -------
        OrderRecord
            An :attr:`OrderState.UNORDERED` record.
        """
        return cls(state=OrderState.UNORDERED)

    @classmethod
    def ranked(cls, ranking: RankingProvenance, scores: Sequence[float]) -> OrderRecord:
        """Return the order record for a freshly ranked set.

        Parameters
        ----------
        ranking : RankingProvenance
            What the ranking claims.
        scores : Sequence[float]
            One score per member, in member order.

        Returns
        -------
        OrderRecord
            A :attr:`OrderState.RANKED` record.
        """
        values = tuple(float(score) for score in scores)
        for index, value in enumerate(values):
            if not math.isfinite(value):
                # A NaN score compares false against everything and quietly
                # scrambles the sort it takes part in.
                raise OrderSemanticsError(f"Score at position {index} is not a finite number")
        return cls(state=OrderState.RANKED, ranking=ranking, scores=values)

    def validate_member_count(self, member_count: int) -> None:
        """Check that the score list lines up with the member list.

        Parameters
        ----------
        member_count : int
            Number of members in the handle.

        Returns
        -------
        None

        Raises
        ------
        OrderSemanticsError
            If a ranked record's score count differs from *member_count*.
        """
        if (
            self.state is OrderState.RANKED
            and self.scores is not None
            and len(self.scores) != member_count
        ):
            raise OrderSemanticsError(
                f"Ranked handle has {member_count} member(s) but {len(self.scores)} score(s)"
            )

    def subset(self, kept_indices: Sequence[int]) -> OrderRecord:
        """Carry this order through an operation that only removes members.

        Removing members cannot make a pairwise ordering claim false: every
        surviving member was scored under the same provenance, and their
        relative sequence is unchanged.  Difference, intersection, and
        predicate filters therefore preserve a ranking.

        Parameters
        ----------
        kept_indices : Sequence[int]
            Indices of surviving members, in the order they will appear.

        Returns
        -------
        OrderRecord
            The order for the reduced set.
        """
        if self.state is not OrderState.RANKED or self.scores is None:
            return self
        return replace(self, scores=tuple(self.scores[index] for index in kept_indices))

    def invalidate(self, operation_name: str) -> OrderRecord:
        """Downgrade this order because an operation broke its guarantee.

        An unordered record is returned unchanged: there was no claim to
        destroy, and manufacturing an ``INVALIDATED`` state for it would
        assert a ranking that never existed.

        Parameters
        ----------
        operation_name : str
            Operation responsible, recorded so the downgrade is attributable.

        Returns
        -------
        OrderRecord
            An :attr:`OrderState.INVALIDATED` record, or ``self`` when there
            was nothing to invalidate.
        """
        if self.state is OrderState.UNORDERED:
            return self
        if self.state is OrderState.INVALIDATED:
            return self
        return OrderRecord(
            state=OrderState.INVALIDATED,
            previous_ranking=self.ranking,
            invalidated_by=operation_name,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this order record to a plain dictionary.

        Returns
        -------
        dict[str, Any]
            JSON-ready representation.
        """
        return {
            "state": self.state.value,
            "ranking": self.ranking.to_dict() if self.ranking is not None else None,
            "scores": list(self.scores) if self.scores is not None else None,
            "previous_ranking": (
                self.previous_ranking.to_dict() if self.previous_ranking is not None else None
            ),
            "invalidated_by": self.invalidated_by,
        }

    @classmethod
    def from_dict(cls, payload: Any, context: str) -> OrderRecord:
        """Reconstruct an order record, refusing anything unrecognised.

        Parameters
        ----------
        payload : Any
            Mapping produced by :meth:`to_dict`.
        context : str
            Dotted path of the payload, used in error messages.

        Returns
        -------
        OrderRecord
            Reconstructed record.

        Raises
        ------
        HandleSchemaError
            If a key is missing, mistyped, or unrecognised.
        OrderSemanticsError
            If the reconstructed fields contradict the declared state.
        """
        data = require_mapping(payload, context)
        reject_unknown_keys(data, _ORDER_KEYS, context)
        state = require_enum(data, "state", OrderState, context)

        ranking: RankingProvenance | None = None
        if _present(data, "ranking", context):
            ranking = RankingProvenance.from_dict(data["ranking"], f"{context}.ranking")

        previous: RankingProvenance | None = None
        if _present(data, "previous_ranking", context):
            previous = RankingProvenance.from_dict(
                data["previous_ranking"], f"{context}.previous_ranking"
            )

        scores: tuple[float, ...] | None = None
        if _present(data, "scores", context):
            scores = tuple(require_float_list(data, "scores", context))

        return cls(
            state=state,
            ranking=ranking,
            scores=scores,
            previous_ranking=previous,
            invalidated_by=require_optional(data, "invalidated_by", str, context),
        )


def _present(payload: Mapping[str, Any], key: str, context: str) -> bool:
    """Report whether a required-but-nullable key holds a value.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Payload to inspect.
    key : str
        Key that must be present.
    context : str
        Dotted path of the payload, used in the error message.

    Returns
    -------
    bool
        ``True`` when the key is present and not ``None``.

    Raises
    ------
    HandleSchemaError
        If the key is absent entirely.
    """
    if key not in payload:
        raise HandleSchemaError(f"{context}.{key}: required key is missing")
    return payload[key] is not None
