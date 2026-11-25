"""Three-lane hybrid retrieval, fused so that each lane's contribution is visible.

``D21`` asks for vector similarity combined with keyword matching.  This module
runs **three** lanes rather than two, and the third one is not padding:

``vector``
    Cosine similarity over the embedding column.  Reaches across vocabulary,
    which is the mechanism the motivating failure needed and did not have.
``bm25``
    LanceDB's native full-text index.  Stems, folds case and drops stop words,
    which makes it good at topical keyword recall.
``exact``
    A ``LIKE`` predicate pushed down over a raw-text column.  Necessary
    *because* BM25 stems and drops stop words: those transformations destroy
    exactly the queries that must match literally -- an author's name, a system
    name like ``FM-Agent``, an arXiv identifier, a quoted phrase.  An FTS index
    is not a substitute for it.

Two independent projects converged on this shape, which is the reason to trust
it over a two-lane design.  ``TKasperczyk/claude-memory`` (design reference,
no license file, TypeScript, 12 stars) combines cosine similarity with a
DataFusion ``LIKE`` over a dedicated ``exact_text`` column.
``ziwenhahaha/daily-paper-reader`` (MIT, 887 stars, pushed 2026-08-28) arrived
independently at BM25 plus exact match plus vector recall.  Their agreement is
the evidence; neither project's code is used.

Why fusion is done here rather than by LanceDB's hybrid query
-------------------------------------------------------------

LanceDB can fuse a vector lane and an FTS lane itself, with an ``RRFReranker``,
and return a single ``_relevance_score``.  That is one row per paper and no
statement of which lane found it.  The acceptance test's fourth pass condition
is **mechanism attribution**: record which mechanism surfaced each paper,
because that is what tells us which mechanism is doing the work.  A single
fused score cannot answer it, so the lanes are run separately and fused with
reciprocal rank fusion here, where each paper keeps the list of lanes that
found it and its rank within each.

The cost is three queries instead of one.  At a corpus of a few thousand
papers that is not a consideration; if it ever becomes one, the native hybrid
query is the fallback and it costs the attribution.

A lane that could not run says so
---------------------------------

:class:`LaneOutcome` distinguishes "this lane ran and matched nothing" from
"this lane could not run".  The second happens for real -- the BM25 lane is
unavailable until the full-text index exists -- and a lane that returned an
empty list without saying which of the two it was is the failure this whole
project exists to prevent, in miniature.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from lit_search.index.embedding import EmbeddingModel, QueryEmbeddingCache
from lit_search.index.paper_index import IndexHealth, PaperIndex
from lit_search.index.schema import KEY_COLUMN

#: Reciprocal-rank-fusion constant.  60 is the value from the original RRF
#: paper and the default LanceDB's own ``RRFReranker`` uses; it damps the
#: influence of a single lane's top rank enough that one lane cannot dominate
#: the fused order on its own.
RRF_K = 60

#: How many rows each lane is asked for, as a multiple of the requested limit.
#: A lane asked for exactly *k* rows contributes at most *k* candidates to the
#: fusion, so the fused top-*k* would be little more than one lane's top-*k*.
LANE_OVERSAMPLE = 4


class Lane(StrEnum):
    """Which retrieval mechanism surfaced a record.

    Attributes
    ----------
    VECTOR
        Cosine similarity over embeddings.
    BM25
        Full-text index over normalised text.
    EXACT
        Literal substring match over raw text.
    """

    VECTOR = "vector"
    BM25 = "bm25"
    EXACT = "exact"


#: Relative weight of each lane in the fused score.  The vector lane leads
#: because reaching across vocabulary is the mechanism the motivating failure
#: needed; the exact lane is weighted equal to it because a literal match on a
#: system name is a strong signal, not a weak one.
DEFAULT_LANE_WEIGHTS: Mapping[Lane, float] = {
    Lane.VECTOR: 1.0,
    Lane.EXACT: 1.0,
    Lane.BM25: 0.8,
}


@dataclass(frozen=True, slots=True)
class LaneOutcome:
    """What one lane did, including being unable to run.

    Parameters
    ----------
    lane : Lane
        Which lane this is.
    available : bool
        Whether the lane could run at all.
    requested : int
        How many rows it was asked for.
    returned : int
        How many rows it returned.
    unavailable_reason : str | None
        Why it could not run.  Required when *available* is ``False``.

    Raises
    ------
    ValueError
        If a lane is marked unavailable without a reason.
    """

    lane: Lane
    available: bool
    requested: int
    returned: int
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        """Reject an unexplained unavailable lane.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the lane is unavailable and carries no reason.
        """
        if not self.available and not self.unavailable_reason:
            raise ValueError(
                f"Lane {self.lane.value!r} is unavailable and gives no reason; an "
                "unexplained empty lane is indistinguishable from a lane that matched "
                "nothing, and the two mean opposite things"
            )

    def describe(self) -> str:
        """Return a one-line human-readable summary.

        Returns
        -------
        str
            Summary line.
        """
        if not self.available:
            return f"{self.lane.value}: DID NOT RUN -- {self.unavailable_reason}"
        return f"{self.lane.value}: {self.returned} of {self.requested} requested"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, Any]
            Serialized outcome.
        """
        return {
            "lane": self.lane.value,
            "available": self.available,
            "requested": self.requested,
            "returned": self.returned,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True, slots=True)
class Candidate:
    """One retrieved record, with the lanes that surfaced it.

    Parameters
    ----------
    key : str
        Record key.
    row : Mapping[str, Any]
        The projected index row.
    lane_ranks : Mapping[str, int]
        One-based rank within each lane that returned this record.
    fused_score : float
        Reciprocal-rank-fusion score.
    """

    key: str
    row: Mapping[str, Any]
    lane_ranks: Mapping[str, int]
    fused_score: float

    @property
    def lanes(self) -> tuple[str, ...]:
        """Return the lanes that surfaced this record, worst rank last.

        Returns
        -------
        tuple[str, ...]
            Lane names, ordered by the rank they gave this record.
        """
        return tuple(sorted(self.lane_ranks, key=lambda lane: self.lane_ranks[lane]))

    @property
    def title(self) -> str:
        """Return the record title.

        Returns
        -------
        str
            Title.
        """
        return str(self.row.get("title", ""))

    @property
    def degraded_text(self) -> str:
        """Return the fields missing from this record's indexed text.

        Returns
        -------
        str
            Comma-separated field names, empty when nothing was missing.
        """
        return str(self.row.get("missing_fields", ""))

    def describe(self) -> str:
        """Return a one-line summary naming the lanes that found it.

        Returns
        -------
        str
            Summary line, with a degraded-text marker when it applies.
        """
        lanes = "+".join(self.lanes)
        degraded = f" [scored without {self.degraded_text}]" if self.degraded_text else ""
        return f"{self.fused_score:.4f} [{lanes}] {self.title}{degraded}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, Any]
            Serialized candidate.
        """
        return {
            "key": self.key,
            "title": self.title,
            "doi": self.row.get("doi", ""),
            "preprint_doi": self.row.get("preprint_doi", ""),
            "year": self.row.get("year", 0),
            "lanes": list(self.lanes),
            "lane_ranks": dict(self.lane_ranks),
            "fused_score": self.fused_score,
            "missing_fields": self.degraded_text,
            "model_id": self.row.get("model_id", ""),
            "model_revision": self.row.get("model_revision", ""),
        }


@dataclass(frozen=True, slots=True)
class HybridResult:
    """A fused retrieval, with per-lane accounting.

    Parameters
    ----------
    query_text : str
        The query as given.
    candidates : tuple[Candidate, ...]
        Fused results, best first.
    lanes : tuple[LaneOutcome, ...]
        What each lane did.
    health : IndexHealth
        State of the index at query time, so a report can say that some rows
        were served by a flat scan.
    cache_statistics : Mapping[str, int]
        Query-embedding cache hits and misses.
    """

    query_text: str
    candidates: tuple[Candidate, ...] = ()
    lanes: tuple[LaneOutcome, ...] = ()
    health: IndexHealth | None = None
    cache_statistics: Mapping[str, int] = field(default_factory=dict)

    @property
    def unavailable_lanes(self) -> tuple[LaneOutcome, ...]:
        """Return the lanes that could not run.

        Returns
        -------
        tuple[LaneOutcome, ...]
            Unavailable lanes.
        """
        return tuple(outcome for outcome in self.lanes if not outcome.available)

    def attribution(self) -> dict[str, tuple[str, ...]]:
        """Return which lanes surfaced each record.

        This is the acceptance test's mechanism-attribution output: it is what
        tells us whether the vector lane, the keyword lane or the literal lane
        is doing the work of reaching another research community's vocabulary.

        Returns
        -------
        dict[str, tuple[str, ...]]
            Record key to lane names.
        """
        return {candidate.key: candidate.lanes for candidate in self.candidates}

    def headline(self) -> str:
        """Return the one-line verdict, naming any lane that did not run.

        Returns
        -------
        str
            Summary line.
        """
        missing = self.unavailable_lanes
        base = f"hybrid retrieval for {self.query_text!r} -- {len(self.candidates)} candidate(s)"
        if not missing:
            return base
        names = ", ".join(outcome.lane.value for outcome in missing)
        return f"{base}; LANES THAT DID NOT RUN: {names}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, Any]
            Serialized result.
        """
        return {
            "query_text": self.query_text,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "lanes": [outcome.to_dict() for outcome in self.lanes],
            "health": None if self.health is None else self.health.to_dict(),
            "cache_statistics": dict(self.cache_statistics),
        }


def _fuse(
    lane_rows: Mapping[Lane, Sequence[Mapping[str, Any]]],
    weights: Mapping[Lane, float],
) -> list[Candidate]:
    """Fuse per-lane ranked lists with weighted reciprocal rank fusion.

    Ties break on first-seen lane order, so the output is deterministic for a
    given set of lane results.

    Parameters
    ----------
    lane_rows : Mapping[Lane, Sequence[Mapping[str, Any]]]
        Rows returned by each lane, in that lane's order.
    weights : Mapping[Lane, float]
        Per-lane weight.

    Returns
    -------
    list[Candidate]
        Fused candidates, best first.
    """
    rows_by_key: dict[str, Mapping[str, Any]] = {}
    ranks: dict[str, dict[str, int]] = {}
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}

    for lane, rows in lane_rows.items():
        weight = weights.get(lane, 1.0)
        for position, row in enumerate(rows, start=1):
            key = str(row[KEY_COLUMN])
            rows_by_key.setdefault(key, row)
            first_seen.setdefault(key, len(first_seen))
            lane_ranks = ranks.setdefault(key, {})
            # A lane cannot rank the same record twice; keep the better rank.
            if lane.value not in lane_ranks or position < lane_ranks[lane.value]:
                lane_ranks[lane.value] = position
            scores[key] = scores.get(key, 0.0) + weight / (RRF_K + position)

    ordered = sorted(scores, key=lambda key: (-scores[key], first_seen[key]))
    return [
        Candidate(
            key=key,
            row=rows_by_key[key],
            lane_ranks=dict(ranks[key]),
            fused_score=scores[key],
        )
        for key in ordered
    ]


def _exact_needles(query_text: str, extra: Sequence[str] | None) -> list[str]:
    """Choose the literal substrings the exact lane looks for.

    Quoted spans in the query are taken verbatim, because a user who typed
    quotation marks asked for a literal match.  Anything else supplied by the
    caller is added as-is.  When the query contains no quoted span and the
    caller supplied nothing, the exact lane is given the whole query only when
    it is short enough to plausibly be a name or a title -- a long prose query
    matched literally returns nothing and would look like an empty lane.

    Parameters
    ----------
    query_text : str
        The query as given.
    extra : Sequence[str] | None
        Literals supplied by the caller, for example an author name.

    Returns
    -------
    list[str]
        Literal substrings, deduplicated, order preserved.
    """
    import re

    needles: list[str] = [span for span in re.findall(r'"([^"]+)"', query_text) if span.strip()]
    needles.extend(item for item in (extra or []) if item.strip())
    if not needles:
        stripped = query_text.strip()
        if stripped and len(stripped.split()) <= 6:
            needles.append(stripped)
    seen: set[str] = set()
    unique: list[str] = []
    for needle in needles:
        lowered = needle.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        unique.append(needle)
    return unique


def hybrid_search(
    index: PaperIndex,
    embedder: EmbeddingModel,
    query_text: str,
    *,
    limit: int = 20,
    exact_terms: Sequence[str] | None = None,
    weights: Mapping[Lane, float] | None = None,
    cache: QueryEmbeddingCache | None = None,
    keys: Sequence[str] | None = None,
    query_vector: Sequence[float] | None = None,
) -> HybridResult:
    """Run all three lanes and fuse them, reporting what each lane did.

    Parameters
    ----------
    index : PaperIndex
        Index to query.
    embedder : EmbeddingModel
        Model used to embed the query.  Must be the model the index was built
        with; :class:`~lit_search.index.paper_index.PaperIndex` enforces that
        at open time.
    query_text : str
        The query.
    limit : int
        How many fused candidates to return.
    exact_terms : Sequence[str] | None
        Extra literal substrings for the exact lane, for example author names.
    weights : Mapping[Lane, float] | None
        Per-lane weights.  Defaults to :data:`DEFAULT_LANE_WEIGHTS`.
    cache : QueryEmbeddingCache | None
        Query-embedding cache.  A fresh one is used when none is given.
    keys : Sequence[str] | None
        Restrict every lane to these record keys.  Used when re-ranking one
        stored set rather than searching the whole index.
    query_vector : Sequence[float] | None
        Vector to use for the vector lane instead of embedding *query_text*.
        Supplied in seed mode, where the "query" is a paper and must be
        embedded with the **document** convention rather than the query one --
        ranking against a seed paper is document-to-document similarity, and
        applying a query prefix to a document is the ``D14`` defect.

    Returns
    -------
    HybridResult
        Fused candidates and per-lane accounting.
    """
    active_weights = dict(weights or DEFAULT_LANE_WEIGHTS)
    active_cache = cache if cache is not None else QueryEmbeddingCache()
    health = index.health()
    pool = max(limit * LANE_OVERSAMPLE, limit)

    lane_rows: dict[Lane, Sequence[Mapping[str, Any]]] = {}
    outcomes: list[LaneOutcome] = []

    vector = (
        list(query_vector)
        if query_vector is not None
        else active_cache.embed_query(embedder, query_text)
    )
    vector_rows = index.vector_lane(vector, pool, keys=keys)
    lane_rows[Lane.VECTOR] = vector_rows
    outcomes.append(
        LaneOutcome(lane=Lane.VECTOR, available=True, requested=pool, returned=len(vector_rows))
    )

    if health.has_text_index:
        text_rows = index.text_lane(query_text, pool, keys=keys)
        lane_rows[Lane.BM25] = text_rows
        outcomes.append(
            LaneOutcome(lane=Lane.BM25, available=True, requested=pool, returned=len(text_rows))
        )
    else:
        outcomes.append(
            LaneOutcome(
                lane=Lane.BM25,
                available=False,
                requested=pool,
                returned=0,
                unavailable_reason=(
                    "no full-text index exists on this table, so keyword retrieval could "
                    "not run; build it with 'lit-search index optimize' or by adding papers"
                ),
            )
        )

    needles = _exact_needles(query_text, exact_terms)
    if needles:
        exact_rows = index.exact_lane(needles, pool, keys=keys)
        lane_rows[Lane.EXACT] = exact_rows
        outcomes.append(
            LaneOutcome(lane=Lane.EXACT, available=True, requested=pool, returned=len(exact_rows))
        )
    else:
        outcomes.append(
            LaneOutcome(
                lane=Lane.EXACT,
                available=False,
                requested=pool,
                returned=0,
                unavailable_reason=(
                    "no literal term was available: the query has no quoted span, no "
                    "exact terms were supplied, and the query is too long to match "
                    "literally. Quote a phrase to use this lane."
                ),
            )
        )

    fused = _fuse(lane_rows, active_weights)
    return HybridResult(
        query_text=query_text,
        candidates=tuple(fused[:limit]),
        lanes=tuple(outcomes),
        health=health,
        cache_statistics=active_cache.statistics,
    )


__all__ = [
    "DEFAULT_LANE_WEIGHTS",
    "LANE_OVERSAMPLE",
    "RRF_K",
    "Candidate",
    "HybridResult",
    "Lane",
    "LaneOutcome",
    "hybrid_search",
]
