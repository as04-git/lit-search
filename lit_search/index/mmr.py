"""Maximal marginal relevance, as a presentation layer over a fixed ranking.

``D21`` wants diversity in the top-N because adjacent sub-areas are where the
missed papers live.  The specification is equally clear about what MMR is
**not**: it gives embedding-space novelty, not sub-area coverage, and it can
suppress several genuinely important papers that happen to sit in one dense
cluster.  So the rule is that diversification changes **what is shown** and
never **what the set is**.

Three mechanisms enforce that here, rather than a comment asking for it.

**Both orders are kept.**  :class:`DiversifiedView` holds the untouched
relevance order alongside the diversified display order.  Nothing has to
reconstruct the original, and no code path can lose it.

**The seeding accessor returns the relevance order.**
:meth:`DiversifiedView.for_seeding` exists precisely so that "give me the
papers to snowball from" cannot accidentally return the diversified list.
Diversified output must never become snowball seeds: MMR deliberately drops
close neighbours of what it has already picked, and a close neighbour of a
central paper is exactly the paper whose citation neighbourhood most needs
traversing.  A frontier built from a diversified list is systematically thinned
in the one region that matters most, and nothing downstream can tell.

**A handle written from the display order is marked pool-dependent.**
:meth:`DiversifiedView.ranking_provenance` returns provenance with
``pool_independent=False``, which
:func:`lit_search.store.operations.union` already refuses to merge across
sets -- an MMR score depends on what else was in the pool, so comparing two
of them is meaningless.  The existing order machinery does the enforcing.

**What MMR pushed out is recorded.**  :class:`Suppression` names each paper
that would have been in the top-N under pure relevance and is not under
diversification, together with what it was judged too similar to.  The idea is
taken from ``TKasperczyk/claude-memory``'s ``mmrExclusions`` diagnostic, and it
is the part of that design most worth having: it turns "MMR can suppress
important papers" from a caveat in a specification into a list a reader can
check.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lit_search.index.embedding import EmbeddingIdentity
from lit_search.index.hybrid import Candidate
from lit_search.store.order import RankingProvenance, RankMode

#: Default trade-off.  0.7 leans towards relevance: the tool's failure mode is
#: missing a paper, and an aggressively diversified top-20 misses papers too,
#: just from a different cause.
DEFAULT_LAMBDA = 0.7


class DiversificationError(ValueError):
    """Raised when a diversification request cannot be honoured as stated.

    Currently only an out-of-range lambda.  Raised rather than clamped: a
    silently clamped lambda produces an ordering the caller did not ask for and
    cannot detect.
    """


@dataclass(frozen=True, slots=True)
class Suppression:
    """One paper that pure relevance would have shown and diversity did not.

    Parameters
    ----------
    key : str
        Suppressed record's key.
    title : str
        Its title.
    relevance_rank : int
        Its one-based rank in the undiversified order.
    similar_to : str
        Key of the already-selected paper it was judged closest to.
    similarity : float
        Cosine similarity to that paper.
    """

    key: str
    title: str
    relevance_rank: int
    similar_to: str
    similarity: float

    def describe(self) -> str:
        """Return a one-line human-readable summary.

        Returns
        -------
        str
            Summary line.
        """
        return (
            f"rank {self.relevance_rank} {self.title!r} suppressed as "
            f"{self.similarity:.3f}-similar to {self.similar_to}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, Any]
            Serialized suppression.
        """
        return {
            "key": self.key,
            "title": self.title,
            "relevance_rank": self.relevance_rank,
            "similar_to": self.similar_to,
            "similarity": self.similarity,
        }


@dataclass(frozen=True, slots=True)
class DiversifiedView:
    """A diversified display order beside the relevance order it came from.

    Parameters
    ----------
    relevance_order : tuple[Candidate, ...]
        The ranking as retrieved, untouched.  This is the set's real order.
    display_order : tuple[Candidate, ...]
        The diversified order, for showing to a person.
    lambda_value : float
        Relevance-versus-novelty trade-off actually used.
    suppressed : tuple[Suppression, ...]
        Papers pure relevance would have shown in the same-sized window.
    undiversifiable : tuple[str, ...]
        Keys of candidates with no stored vector, which therefore could not be
        compared to anything.  They are appended to the display order in
        relevance order rather than dropped -- dropping them would be a silent
        omission, which is the failure this project exists to prevent.
    """

    relevance_order: tuple[Candidate, ...]
    display_order: tuple[Candidate, ...]
    lambda_value: float
    suppressed: tuple[Suppression, ...] = ()
    undiversifiable: tuple[str, ...] = ()

    #: Diversification is a view, never a set.  Read by callers that need to
    #: assert it rather than remember it.
    presentation_only: bool = True

    def for_display(self) -> tuple[Candidate, ...]:
        """Return the order to show a person.

        Returns
        -------
        tuple[Candidate, ...]
            Diversified order.
        """
        return self.display_order

    def for_seeding(self) -> tuple[Candidate, ...]:
        """Return the order to expand, traverse, or derive a further set from.

        Deliberately the **relevance** order.  MMR drops close neighbours of
        already-selected papers, and a close neighbour of a central paper is
        the paper whose citation neighbourhood most needs traversing -- so a
        frontier seeded from the diversified list is thinned exactly where
        thinning costs most, invisibly.

        Returns
        -------
        tuple[Candidate, ...]
            Relevance order.
        """
        return self.relevance_order

    def degraded_members(self) -> tuple[str, ...]:
        """Return the keys of displayed records built from incomplete text.

        Parameters
        ----------
        None

        Returns
        -------
        tuple[str, ...]
            Keys whose indexed text was missing a policy field.  A record that
            sinks in a ranking because its abstract was empty has sunk for a
            reason unrelated to its relevance, which is the difference between
            an explainable low rank and a silently lost paper.
        """
        return tuple(candidate.key for candidate in self.relevance_order if candidate.degraded_text)

    def ranking_provenance(
        self, *, identity: EmbeddingIdentity, query_text: str, text_policy: str
    ) -> RankingProvenance:
        """Return provenance for a handle written from the display order.

        ``pool_independent`` is ``False``, which is not a formality: an MMR
        score is computed partly from what else was in the pool, so two such
        scores from different pools are not comparable and
        :func:`lit_search.store.operations.union` will refuse to merge them.

        Parameters
        ----------
        identity : EmbeddingIdentity
            Model the vectors came from.
        query_text : str
            Query the set was ranked against.
        text_policy : str
            Which paper fields were embedded, from
            :meth:`lit_search.index.text.TextPolicy.describe`.

        Returns
        -------
        RankingProvenance
            Provenance recording the model, the convention and the lambda.
        """
        return RankingProvenance(
            mode=RankMode.QUERY,
            query_text=query_text,
            model_id=identity.model_id,
            model_revision=identity.model_revision,
            prompt_convention=identity.prompt_convention,
            scoring_convention=f"rrf-hybrid then mmr(lambda={self.lambda_value})",
            normalization=identity.normalization,
            text_policy=text_policy,
            degraded_text=self.degraded_members(),
            pool_independent=False,
        )

    def headline(self) -> str:
        """Return the one-line verdict, naming what diversification cost.

        Returns
        -------
        str
            Summary line.
        """
        base = (
            f"diversified display order, lambda={self.lambda_value} "
            f"({len(self.display_order)} shown of {len(self.relevance_order)})"
        )
        if not self.suppressed:
            return base
        return (
            f"{base}; {len(self.suppressed)} paper(s) that pure relevance would have "
            "shown are not shown -- this is a display choice, not a coverage fact"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, Any]
            Serialized view, with both orders present.
        """
        return {
            "lambda": self.lambda_value,
            "presentation_only": self.presentation_only,
            "display_order": [candidate.to_dict() for candidate in self.display_order],
            "relevance_order": [candidate.to_dict() for candidate in self.relevance_order],
            "suppressed": [item.to_dict() for item in self.suppressed],
            "undiversifiable": list(self.undiversifiable),
        }


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine similarity of two vectors.

    Parameters
    ----------
    left : Sequence[float]
        First vector.
    right : Sequence[float]
        Second vector.

    Returns
    -------
    float
        Similarity in ``[-1, 1]``, or ``0.0`` when either vector is zero or the
        lengths differ.
    """
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _select(
    ordered: Sequence[Candidate],
    vectors: Mapping[str, Sequence[float]],
    lambda_value: float,
    window: int,
) -> tuple[list[Candidate], list[Candidate]]:
    """Run the greedy MMR selection.

    Parameters
    ----------
    ordered : Sequence[Candidate]
        Candidates with vectors, in relevance order.
    vectors : Mapping[str, Sequence[float]]
        Key to vector.
    lambda_value : float
        Relevance weight.
    window : int
        How many to select.

    Returns
    -------
    tuple[list[Candidate], list[Candidate]]
        Selected candidates in display order, and the unselected remainder in
        relevance order.
    """
    remaining = list(ordered)
    selected: list[Candidate] = []
    if not remaining:
        return selected, remaining
    selected.append(remaining.pop(0))

    while remaining and len(selected) < window:
        best_index, best_score = 0, -math.inf
        for index, candidate in enumerate(remaining):
            novelty = max(
                (cosine(vectors[chosen.key], vectors[candidate.key]) for chosen in selected),
                default=0.0,
            )
            score = lambda_value * candidate.fused_score - (1.0 - lambda_value) * novelty
            if score > best_score:
                best_index, best_score = index, score
        selected.append(remaining.pop(best_index))
    return selected, remaining


def _suppressions(
    ordered: Sequence[Candidate],
    selected: Sequence[Candidate],
    vectors: Mapping[str, Sequence[float]],
    window: int,
) -> list[Suppression]:
    """List the papers pure relevance would have shown and diversity did not.

    Parameters
    ----------
    ordered : Sequence[Candidate]
        Candidates with vectors, in relevance order.
    selected : Sequence[Candidate]
        Candidates chosen by MMR.
    vectors : Mapping[str, Sequence[float]]
        Key to vector.
    window : int
        Size of the display window.

    Returns
    -------
    list[Suppression]
        One entry per suppressed paper, in relevance order.
    """
    chosen_keys = {candidate.key for candidate in selected}
    notes: list[Suppression] = []
    for rank, candidate in enumerate(ordered[:window], start=1):
        if candidate.key in chosen_keys:
            continue
        nearest_key, nearest = "", -math.inf
        for other in selected:
            similarity = cosine(vectors[other.key], vectors[candidate.key])
            if similarity > nearest:
                nearest_key, nearest = other.key, similarity
        notes.append(
            Suppression(
                key=candidate.key,
                title=candidate.title,
                relevance_rank=rank,
                similar_to=nearest_key,
                similarity=0.0 if nearest == -math.inf else nearest,
            )
        )
    return notes


def diversify(
    candidates: Sequence[Candidate],
    vectors: Mapping[str, Sequence[float]],
    *,
    limit: int | None = None,
    lambda_value: float = DEFAULT_LAMBDA,
) -> DiversifiedView:
    """Build a diversified display order without disturbing the relevance order.

    Parameters
    ----------
    candidates : Sequence[Candidate]
        Candidates in relevance order.
    vectors : Mapping[str, Sequence[float]]
        Key to stored vector.  Candidates absent from this mapping cannot be
        compared and are appended in relevance order rather than dropped.
    limit : int | None
        Size of the display window.  ``None`` diversifies the whole list.
    lambda_value : float
        Relevance weight in ``[0, 1]``.  ``1.0`` is pure relevance.

    Returns
    -------
    DiversifiedView
        Both orders, the suppressions, and the undiversifiable keys.

    Raises
    ------
    DiversificationError
        If *lambda_value* is outside ``[0, 1]``.
    """
    if not 0.0 <= lambda_value <= 1.0:
        raise DiversificationError(
            f"lambda must be between 0 and 1, got {lambda_value}; refusing to clamp it, "
            "because a clamped lambda produces an ordering nobody asked for"
        )
    relevance = tuple(candidates)
    with_vectors = [candidate for candidate in relevance if candidate.key in vectors]
    without = tuple(candidate.key for candidate in relevance if candidate.key not in vectors)
    window = len(relevance) if limit is None else max(0, limit)

    selected, remainder = _select(with_vectors, vectors, lambda_value, window)
    notes = _suppressions(with_vectors, selected, vectors, window)
    tail = [candidate for candidate in relevance if candidate.key in without]

    return DiversifiedView(
        relevance_order=relevance,
        display_order=tuple([*selected, *tail, *remainder][:window] if window else []),
        lambda_value=lambda_value,
        suppressed=tuple(notes),
        undiversifiable=without,
    )


__all__ = [
    "DEFAULT_LAMBDA",
    "DiversificationError",
    "DiversifiedView",
    "Suppression",
    "cosine",
    "diversify",
]
