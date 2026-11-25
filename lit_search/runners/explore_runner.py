"""Bounded exploration moves: full-text snippet search and related-paper expansion.

Both moves are fixed-cost by construction -- a handful of Semantic Scholar
requests -- so they can be used interactively and inside a time-budgeted hunt,
unlike the open-ended citation traversal in ``snowball``.

Each move returns a :class:`ExploreOutcome`: papers in rank order, one coverage
entry per step (named ``<move>:<step>``), the route by which each paper was
reached, and, for snippets, the matching sentence as evidence.
"""

from __future__ import annotations

import datetime
import re
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import requests

from lit_search import credentials
from lit_search.connectors.semantic_scholar import _PAPER_FIELDS, SemanticScholarConnector
from lit_search.core.citation import IdentifierKind
from lit_search.core.coverage import ExecutionStatus, SourceCoverage
from lit_search.core.paper import Paper
from lit_search.exceptions import ConnectorError, InvalidParameterError, RateLimitBusyError

_SNIPPET_URL = "https://api.semanticscholar.org/graph/v1/snippet/search"
_RECOMMEND_URL = "https://api.semanticscholar.org/recommendations/v1/papers"
_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
_BATCH_LIMIT = 500
_EDGE_FIELDS = (
    "paperId,title,citationCount,referenceCount,"
    "references.paperId,references.title,citations.paperId,citations.title"
)
_ARXIV_RE = re.compile(r"^(?:arxiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)$", re.IGNORECASE)
_S2_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass
class ExploreOutcome:
    """Result of one exploration move.

    Parameters
    ----------
    papers : list[Paper]
        Papers in rank order.
    steps : list[SourceCoverage]
        One coverage entry per request step, named ``<move>:<step>``.
    routes : dict[int, list[str]]
        ``id(paper) -> route labels`` (for example ``"rec#3"``, ``"cites 2 seeds"``).
    evidence : dict[int, list[str]]
        ``id(paper) -> matching sentences`` (snippet search only).
    seconds : float
        Wall time.
    """

    papers: list[Paper] = field(default_factory=list)
    steps: list[SourceCoverage] = field(default_factory=list)
    routes: dict[int, list[str]] = field(default_factory=dict)
    evidence: dict[int, list[str]] = field(default_factory=dict)
    seconds: float = 0.0


def s2_connector() -> SemanticScholarConnector:
    """Semantic Scholar connector tuned for its observed throttling.

    Keyed requests are throttled intermittently with no ``Retry-After``
    (about half refused at a 3 s pace on 2026-10-11); three retries starting at
    2 s ride that out without the default 5 s base making each move crawl.

    Returns
    -------
    SemanticScholarConnector
        Connector using ``SEMANTIC_SCHOLAR_API_KEY``.
    """
    connector = SemanticScholarConnector(api_key=credentials.get("SEMANTIC_SCHOLAR_API_KEY"))
    connector._max_rate_limit_retries = 3
    connector._rate_limit_base_delay = 2.0
    return connector


def s2_request_id(raw: str) -> str:
    """Normalise a user-supplied identifier into Semantic Scholar's syntax.

    Parameters
    ----------
    raw : str
        DOI, arXiv id, ``DOI:``/``ARXIV:``/``CorpusId:`` prefixed id, or a
        Semantic Scholar paper id.

    Returns
    -------
    str
        Identifier accepted by the Semantic Scholar Graph API.

    Raises
    ------
    InvalidParameterError
        When the identifier has no recognisable form.
    """
    value = raw.strip()
    lowered = value.lower()
    if lowered.startswith(("doi:", "arxiv:", "corpusid:")) and not _ARXIV_RE.match(value):
        prefix, rest = value.split(":", 1)
        return (
            f"{ {'doi': 'DOI', 'arxiv': 'ARXIV', 'corpusid': 'CorpusId'}[prefix.lower()] }:{rest}"
        )
    if match := _ARXIV_RE.match(value):
        return f"ARXIV:{match.group(1)}"
    if lowered.startswith("10.48550/arxiv."):
        return f"ARXIV:{value.split('.', 2)[2]}"
    if lowered.startswith("10."):
        return f"DOI:{value}"
    if _S2_SHA_RE.match(lowered):
        return lowered
    raise InvalidParameterError(f"cannot read {raw!r} as a DOI, arXiv id or Semantic Scholar id")


def s2_request_id_for_paper(paper: Paper) -> str | None:
    """Choose the Semantic Scholar identifier for a stored paper.

    Parameters
    ----------
    paper : Paper
        Paper from a handle or a previous result.

    Returns
    -------
    str | None
        Identifier, or ``None`` when the paper carries none Semantic Scholar
        accepts.
    """
    by_kind = {identifier.kind: identifier.value for identifier in paper.identifiers}
    if value := by_kind.get(IdentifierKind.SEMANTIC_SCHOLAR):
        return value
    if value := by_kind.get(IdentifierKind.ARXIV):
        return f"ARXIV:{value}"
    if paper.doi:
        return s2_request_id(paper.doi)
    return None


def _failed(step: str, exc: Exception) -> SourceCoverage:
    return SourceCoverage.failed(step, error=str(exc) or repr(exc), error_type=type(exc).__name__)


def _ranked(step: str, *, retrieved: int, seen: int, requested: int) -> SourceCoverage:
    """Coverage for a ranked list whose source publishes no match total."""
    unparsed = seen - retrieved
    if unparsed > 0:
        return SourceCoverage.partial(
            step, retrieved, detail=f"{unparsed} of {seen} records could not be parsed"
        )
    if seen >= requested:
        return SourceCoverage.capped(step, retrieved, cap=requested, detail="top by relevance")
    return SourceCoverage.partial(
        step,
        retrieved,
        detail="all results the source ranked; it reports no match total",
        execution_status=ExecutionStatus.COMPLETE,
    )


def _hydrate(
    connector: SemanticScholarConnector, ids: Sequence[str], step: str
) -> tuple[dict[str, Paper], SourceCoverage]:
    """Fetch full metadata for Semantic Scholar ids, keeping positions.

    Parameters
    ----------
    connector : SemanticScholarConnector
        Connector providing auth and rate limiting.
    ids : Sequence[str]
        Request ids.
    step : str
        Step name for the coverage entry.

    Returns
    -------
    tuple[dict[str, Paper], SourceCoverage]
        Papers by request id, and the step's coverage.
    """
    found: dict[str, Paper] = {}
    missing = 0
    unparsed = 0
    try:
        for start in range(0, len(ids), _BATCH_LIMIT):
            chunk = list(ids[start : start + _BATCH_LIMIT])
            payload = connector._post(
                _BATCH_URL, params={"fields": _PAPER_FIELDS}, json_body={"ids": chunk}
            ).json()
            if not isinstance(payload, list):
                raise ConnectorError("paper batch response was not a list")
            for request_id, item in zip(chunk, payload, strict=False):
                if item is None:
                    missing += 1
                    continue
                paper = connector._parse_paper(item)
                if paper is None:
                    unparsed += 1
                else:
                    found[request_id] = paper
    except (requests.RequestException, ConnectorError, ValueError, RateLimitBusyError) as exc:
        coverage = _failed(step, exc)
        if found:
            coverage = SourceCoverage.partial(
                step,
                len(found),
                detail=f"metadata request failed after {len(found)} papers: {exc}",
                error=str(exc),
                error_type=type(exc).__name__,
            )
        return found, coverage
    notes = []
    if missing:
        notes.append(f"{missing} not found")
    if unparsed:
        notes.append(f"{unparsed} could not be parsed")
    if notes:
        return found, SourceCoverage.partial(
            step, len(found), detail=f"{len(found)} of {len(ids)} resolved; " + ", ".join(notes)
        )
    return found, SourceCoverage.complete(step, len(found), reported_total=len(ids))


def run_snippets(
    connector: SemanticScholarConnector,
    text: str,
    *,
    limit: int = 20,
    since: datetime.date | None = None,
    until: datetime.date | None = None,
    field_of_study: str | None = None,
) -> ExploreOutcome:
    """Search paper full text and return the papers with their matching sentences.

    Parameters
    ----------
    connector : SemanticScholarConnector
        Connector providing auth and rate limiting.
    text : str
        Free-text query matched against paper bodies.
    limit : int
        Snippets requested (several may come from one paper).
    since : datetime.date | None
        Earliest publication date.
    until : datetime.date | None
        Latest publication date.
    field_of_study : str | None
        Semantic Scholar field-of-study filter, e.g. ``"Computer Science"``.

    Returns
    -------
    ExploreOutcome
        Papers in order of their best snippet, with evidence sentences.
    """
    if not text.strip():
        raise InvalidParameterError("snippets needs a non-empty query")
    start = time.monotonic()
    outcome = ExploreOutcome()
    params: dict[str, Any] = {"query": text, "limit": limit}
    date_param = SemanticScholarConnector._build_ss_date_param(since, until)
    if date_param is not None:
        params["publicationDateOrYear"] = date_param
    if field_of_study:
        params["fieldsOfStudy"] = field_of_study
    try:
        data = connector._get(_SNIPPET_URL, params).json()
    except (requests.RequestException, ConnectorError, ValueError, RateLimitBusyError) as exc:
        outcome.steps.append(_failed("snippets:search", exc))
        outcome.seconds = time.monotonic() - start
        return outcome

    hits = data.get("data") or []
    order: list[str] = []
    sentences: dict[str, list[str]] = {}
    for hit in hits:
        corpus_id = (hit.get("paper") or {}).get("corpusId")
        sentence = ((hit.get("snippet") or {}).get("text") or "").strip()
        if corpus_id is None:
            continue
        request_id = f"CorpusId:{corpus_id}"
        if request_id not in sentences:
            order.append(request_id)
            sentences[request_id] = []
        if sentence:
            sentences[request_id].append(sentence)
    outcome.steps.append(
        _ranked("snippets:search", retrieved=len(order), seen=len(order), requested=limit)
        if len(hits) < limit
        else SourceCoverage.capped(
            "snippets:search", len(order), cap=limit, detail=f"top {limit} snippets"
        )
    )

    papers_by_id, hydrate = _hydrate(connector, order, "snippets:metadata") if order else ({}, None)
    if hydrate is not None:
        outcome.steps.append(hydrate)
    for rank, request_id in enumerate(order, start=1):
        paper = papers_by_id.get(request_id)
        if paper is None:
            continue
        outcome.papers.append(paper)
        outcome.routes[id(paper)] = [f"snippet#{rank}"]
        outcome.evidence[id(paper)] = sentences[request_id]
    outcome.seconds = time.monotonic() - start
    return outcome


def run_related(
    connector: SemanticScholarConnector,
    seeds: Sequence[str],
    *,
    negative: Sequence[str] = (),
    recommendations: int = 30,
    references: bool = True,
    citations: bool = True,
    max_neighbours: int = 60,
) -> ExploreOutcome:
    """Expand seeds through recommendations and one hop of citation edges.

    Parameters
    ----------
    connector : SemanticScholarConnector
        Connector providing auth and rate limiting.
    seeds : Sequence[str]
        Semantic Scholar request ids of the seed papers.
    negative : Sequence[str]
        Request ids of papers the result should move away from.
    recommendations : int
        Recommended papers requested; ``0`` skips the step.
    references : bool
        Include papers the seeds cite.
    citations : bool
        Include papers citing the seeds.
    max_neighbours : int
        Citation neighbours kept, most-connected first, before metadata lookup.

    Returns
    -------
    ExploreOutcome
        Papers ranked by how many routes reach them, then by recommendation
        rank, with the routes recorded.
    """
    if not seeds:
        raise InvalidParameterError("related needs at least one seed paper")
    start = time.monotonic()
    outcome = ExploreOutcome()
    seed_set = set(seeds)
    rec_rank: dict[str, int] = {}
    rec_papers: dict[str, Paper] = {}

    if recommendations > 0:
        try:
            payload = connector._post(
                _RECOMMEND_URL,
                params={"limit": recommendations, "fields": _PAPER_FIELDS},
                json_body={"positivePaperIds": list(seeds), "negativePaperIds": list(negative)},
            ).json()
            items = payload.get("recommendedPapers") or []
            for rank, item in enumerate(items, start=1):
                paper = connector._parse_paper(item)
                paper_id = item.get("paperId")
                if paper is not None and paper_id:
                    rec_rank[paper_id] = rank
                    rec_papers[paper_id] = paper
            outcome.steps.append(
                _ranked(
                    "related:recommendations",
                    retrieved=len(rec_papers),
                    seen=len(items),
                    requested=recommendations,
                )
            )
        except (requests.RequestException, ConnectorError, ValueError, RateLimitBusyError) as exc:
            outcome.steps.append(_failed("related:recommendations", exc))

    cites_seeds: Counter[str] = Counter()
    cited_by_seeds: Counter[str] = Counter()
    if references or citations:
        try:
            payload = connector._post(
                _BATCH_URL, params={"fields": _EDGE_FIELDS}, json_body={"ids": list(seeds)}
            ).json()
            if not isinstance(payload, list):
                raise ConnectorError("paper batch response was not a list")
            unresolved = [s for s, item in zip(seeds, payload, strict=False) if item is None]
            truncated: list[str] = []
            for item in payload:
                if item is None:
                    continue
                if references:
                    refs = [r["paperId"] for r in item.get("references") or [] if r.get("paperId")]
                    cited_by_seeds.update(set(refs))
                    if (item.get("referenceCount") or 0) > len(item.get("references") or []):
                        truncated.append(f"{(item.get('title') or '?')[:40]} (references)")
                if citations:
                    cits = [c["paperId"] for c in item.get("citations") or [] if c.get("paperId")]
                    cites_seeds.update(set(cits))
                    if (item.get("citationCount") or 0) > len(item.get("citations") or []):
                        truncated.append(f"{(item.get('title') or '?')[:40]} (citations)")
            edges = len(cites_seeds) + len(cited_by_seeds)
            notes = []
            if unresolved:
                notes.append(f"seeds not in Semantic Scholar: {', '.join(unresolved)}")
            if truncated:
                notes.append(
                    "Semantic Scholar lists fewer edges than its counts for: "
                    + "; ".join(truncated)
                )
            if notes:
                outcome.steps.append(
                    SourceCoverage.partial(
                        "related:edges",
                        edges,
                        detail=" | ".join(notes),
                        execution_status=ExecutionStatus.INCOMPLETE,
                    )
                )
            else:
                outcome.steps.append(
                    SourceCoverage.complete("related:edges", edges, reported_total=edges)
                )
        except (requests.RequestException, ConnectorError, ValueError, RateLimitBusyError) as exc:
            outcome.steps.append(_failed("related:edges", exc))

    neighbours = (set(cites_seeds) | set(cited_by_seeds)) - seed_set

    def links(paper_id: str) -> int:
        return cites_seeds[paper_id] + cited_by_seeds[paper_id]

    ordered = sorted(
        (n for n in neighbours if n not in rec_papers),
        key=lambda n: (-links(n), n),
    )
    to_fetch = ordered[:max_neighbours]
    if len(ordered) > len(to_fetch):
        dropped = ordered[len(to_fetch) :]
        outcome.steps.append(
            SourceCoverage.capped(
                "related:neighbours",
                len(to_fetch),
                cap=max_neighbours,
                reported_total=len(ordered),
                detail=(
                    f"kept the {len(to_fetch)} most-connected of {len(ordered)} neighbours; "
                    f"{len(dropped)} dropped (most linked to {max(links(n) for n in dropped)} "
                    "seed(s)); raise --neighbours to keep more"
                ),
            )
        )
    fetched: dict[str, Paper] = {}
    if to_fetch:
        fetched, hydrate = _hydrate(connector, to_fetch, "related:metadata")
        outcome.steps.append(hydrate)

    candidates = {**fetched, **rec_papers}
    for paper_id in seed_set:
        candidates.pop(paper_id, None)

    def score(paper_id: str) -> tuple[int, int]:
        routes = links(paper_id) + (1 if paper_id in rec_rank else 0)
        return (-routes, rec_rank.get(paper_id, 10**6))

    for paper_id in sorted(candidates, key=score):
        paper = candidates[paper_id]
        route: list[str] = []
        if paper_id in rec_rank:
            route.append(f"rec#{rec_rank[paper_id]}")
        if cites_seeds[paper_id]:
            route.append(f"cites {cites_seeds[paper_id]} seed(s)")
        if cited_by_seeds[paper_id]:
            route.append(f"cited by {cited_by_seeds[paper_id]} seed(s)")
        outcome.papers.append(paper)
        outcome.routes[id(paper)] = route
    outcome.seconds = time.monotonic() - start
    return outcome


_EDGE_URL = "https://api.semanticscholar.org/graph/v1/paper/{paper_id}/{direction}"
_EDGE_PAGE = 1000
_EDGE_ITEM_FIELDS = "contexts,intents,isInfluential,paperId,title,year,publicationDate"
_SEED_FIELDS = "paperId,title,year,citationCount,referenceCount"
_WORD_RE = re.compile(r"[a-z][a-z0-9]+")
_STOPWORDS = frozenset(
    [
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "into",
        "using",
        "use",
        "via",
        "are",
        "our",
        "its",
        "their",
        "which",
        "have",
        "has",
        "been",
        "can",
        "not",
        "but",
        "than",
        "also",
        "more",
        "most",
        "such",
        "these",
        "those",
        "other",
        "based",
        "towards",
        "toward",
        "new",
        "how",
        "what",
        "when",
        "where",
        "why",
        "model",
        "models",
        "method",
        "methods",
        "approach",
        "paper",
        "work",
        "results",
        "show",
    ]
)


def _words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.casefold()) if len(w) > 2 and w not in _STOPWORDS}


@dataclass
class SnowballSeed:
    """A resolved seed and what one snowball move did with it.

    Parameters
    ----------
    request_id : str
        Identifier as given.
    paper_id : str
        Semantic Scholar paper id.
    title : str
        Seed title.
    citations : int
        Citation count Semantic Scholar reports.
    references : int
        Reference count Semantic Scholar reports.
    """

    request_id: str
    paper_id: str
    title: str
    citations: int
    references: int


def _edge_date_ok(
    item: dict[str, Any], since: datetime.date | None, until: datetime.date | None
) -> bool:
    if since is None and until is None:
        return True
    raw = item.get("publicationDate")
    year = item.get("year")
    try:
        day = datetime.date.fromisoformat(raw) if raw else None
    except ValueError:
        day = None
    if day is None:
        if year is None:
            return since is None
        if since is not None and year < since.year:
            return False
        return not (until is not None and year > until.year)
    if since is not None and day < since:
        return False
    return not (until is not None and day > until)


def run_snowball(
    connector: SemanticScholarConnector,
    seeds: Sequence[str],
    *,
    forward: bool = True,
    backward: bool = True,
    since: datetime.date | None = None,
    until: datetime.date | None = None,
    keep: int = 60,
    edge_cap: int = 2000,
    focus: str = "",
    labels: dict[str, str] | None = None,
) -> tuple[ExploreOutcome, list[SnowballSeed]]:
    """Follow every citation edge of the seeds, in one or both directions.

    Unlike :func:`run_related`, which samples one hop for recommendations,
    this reads each seed's full list of citing papers (forward) and cited
    papers (backward), with the sentences in which the citation is made.
    Edges are filtered by date before anything is ranked, so a date window
    applied to the citers of an older paper is the way to find recent work
    that builds on it.

    Neighbours are ranked by how many seeds they connect to, then by word
    overlap of their title and citation sentences with ``focus`` (the hunt's
    question and its kept papers), then by whether Semantic Scholar marks the
    citation as influential or methodological, then by recency.  Only the top
    ``keep`` are fetched in full.

    Parameters
    ----------
    connector : SemanticScholarConnector
        Connector providing auth and rate limiting.
    seeds : Sequence[str]
        Semantic Scholar request ids of the seed papers.
    forward : bool
        Follow papers citing the seeds.
    backward : bool
        Follow papers the seeds cite.
    since : datetime.date | None
        Earliest publication date of a neighbour.
    until : datetime.date | None
        Latest publication date of a neighbour.
    keep : int
        Neighbours fetched in full and returned.
    edge_cap : int
        Edges read per seed and direction (Semantic Scholar pages of 1000).
    focus : str
        Text whose words make a neighbour rank higher among equally linked ones.
    labels : dict[str, str] | None
        Display name per request id (for example a hunt's ``c3``) used in routes.

    Returns
    -------
    tuple[ExploreOutcome, list[SnowballSeed]]
        Neighbours in rank order with routes and citation sentences as
        evidence, and the resolved seeds.
    """
    if not seeds:
        raise InvalidParameterError("snowball needs at least one seed paper")
    if not (forward or backward):
        raise InvalidParameterError("snowball needs a direction")
    start = time.monotonic()
    outcome = ExploreOutcome()
    resolved: list[SnowballSeed] = []
    try:
        payload = connector._post(
            _BATCH_URL, params={"fields": _SEED_FIELDS}, json_body={"ids": list(seeds)}
        ).json()
        if not isinstance(payload, list):
            raise ConnectorError("paper batch response was not a list")
    except (requests.RequestException, ConnectorError, ValueError, RateLimitBusyError) as exc:
        outcome.steps.append(_failed("snowball:seeds", exc))
        outcome.seconds = time.monotonic() - start
        return outcome, resolved
    missing = []
    for request_id, item in zip(seeds, payload, strict=False):
        if not item or not item.get("paperId"):
            missing.append(request_id)
            continue
        resolved.append(
            SnowballSeed(
                request_id=request_id,
                paper_id=item["paperId"],
                title=item.get("title") or request_id,
                citations=int(item.get("citationCount") or 0),
                references=int(item.get("referenceCount") or 0),
            )
        )
    if missing:
        outcome.steps.append(
            SourceCoverage.partial(
                "snowball:seeds",
                len(resolved),
                detail=f"not in Semantic Scholar: {', '.join(missing)}",
            )
        )
    seed_ids = {s.paper_id for s in resolved}
    focus_words = _words(focus)

    # paper id -> {"item": edge item, "links": [(seed title, direction, contexts, intents, influential)]}
    neighbours: dict[str, dict[str, Any]] = {}
    for seed in resolved:
        directions = [d for d, on in (("citations", forward), ("references", backward)) if on]
        for direction in directions:
            step = f"snowball:{'forward' if direction == 'citations' else 'backward'}"
            label = seed.title[:50]
            reported = seed.citations if direction == "citations" else seed.references
            read = 0
            in_window = 0
            try:
                offset = 0
                while offset < edge_cap:
                    limit = min(_EDGE_PAGE, edge_cap - offset)
                    data = connector._get(
                        _EDGE_URL.format(paper_id=seed.paper_id, direction=direction),
                        {"fields": _EDGE_ITEM_FIELDS, "limit": limit, "offset": offset},
                    ).json()
                    rows = data.get("data") or []
                    for row in rows:
                        item = row.get("citingPaper" if direction == "citations" else "citedPaper")
                        if not item or not item.get("paperId") or item["paperId"] in seed_ids:
                            continue
                        read += 1
                        if not _edge_date_ok(item, since, until):
                            continue
                        in_window += 1
                        entry = neighbours.setdefault(item["paperId"], {"item": item, "links": []})
                        entry["links"].append(
                            (
                                seed,
                                direction,
                                [c for c in row.get("contexts") or [] if c],
                                row.get("intents") or [],
                                bool(row.get("isInfluential")),
                            )
                        )
                    if data.get("next") is None or not rows:
                        break
                    offset = int(data["next"])
            except (
                requests.RequestException,
                ConnectorError,
                ValueError,
                RateLimitBusyError,
            ) as exc:
                outcome.steps.append(
                    SourceCoverage.partial(
                        f"{step}:{label}",
                        in_window,
                        detail=f"edge listing failed after {read}: {exc}",
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )
                    if read
                    else _failed(f"{step}:{label}", exc)
                )
                continue
            window = f", {in_window} in the date window" if (since or until) else ""
            if direction == "references" and read == 0:
                outcome.steps.append(
                    SourceCoverage.partial(
                        f"{step}:{label}",
                        0,
                        detail=(
                            "Semantic Scholar has no reference list for this paper; read its "
                            "related-work section instead (`lit-search fetch <id>`)"
                        ),
                    )
                )
            elif reported > read and read >= edge_cap - len(seed_ids):
                outcome.steps.append(
                    SourceCoverage.capped(
                        f"{step}:{label}",
                        in_window,
                        cap=edge_cap,
                        reported_total=reported,
                        detail=f"read {read} of {reported} edges{window}; narrow with --since",
                        execution_status=ExecutionStatus.INCOMPLETE,
                    )
                )
            else:
                outcome.steps.append(
                    SourceCoverage.complete(f"{step}:{label}", in_window, reported_total=in_window)
                    if not (since or until)
                    else SourceCoverage.partial(
                        f"{step}:{label}",
                        in_window,
                        detail=f"all {read} edges read{window}",
                        execution_status=ExecutionStatus.COMPLETE,
                    )
                )

    def rank_key(paper_id: str) -> tuple[float, ...]:
        entry = neighbours[paper_id]
        links = entry["links"]
        text = (
            (entry["item"].get("title") or "")
            + " "
            + " ".join(c for link in links for c in link[2])
        )
        overlap = len(focus_words & _words(text)) if focus_words else 0
        strong = any(link[4] or "methodology" in link[3] for link in links)
        year = entry["item"].get("year") or 0
        return (-len({link[0].paper_id for link in links}), -overlap, -int(strong), -year)

    ordered = sorted(neighbours, key=rank_key)
    chosen = ordered[:keep]
    if len(ordered) > len(chosen):
        outcome.steps.append(
            SourceCoverage.capped(
                "snowball:kept",
                len(chosen),
                cap=keep,
                reported_total=len(ordered),
                detail=(
                    f"kept the top {len(chosen)} of {len(ordered)} neighbours (seeds linked, then "
                    "word overlap with the question and kept papers); narrow with --since or "
                    "raise --max"
                ),
            )
        )
    fetched: dict[str, Paper] = {}
    if chosen:
        fetched, hydrate = _hydrate(connector, chosen, "snowball:metadata")
        outcome.steps.append(hydrate)
    for paper_id in chosen:
        paper = fetched.get(paper_id)
        if paper is None:
            continue
        links = neighbours[paper_id]["links"]
        routes: list[str] = []
        sentences: list[str] = []
        for seed, direction, contexts, intents, influential in links:
            tags = [t for t in intents if t != "background"] + (
                ["influential"] if influential else []
            )
            tag = f" ({', '.join(tags)})" if tags else ""
            short = (labels or {}).get(seed.request_id, seed.request_id)
            if direction == "citations":
                routes.append(f"cites {short}{tag}")
                sentences += [f"[cites {short}] {c}" for c in contexts[:2]]
            else:
                routes.append(f"cited by {short}{tag}")
                sentences += [f"[{short} cites it] {c}" for c in contexts[:2]]
        outcome.papers.append(paper)
        outcome.routes[id(paper)] = routes
        outcome.evidence[id(paper)] = sentences
    outcome.seconds = time.monotonic() - start
    return outcome, resolved


__all__ = [
    "ExploreOutcome",
    "SnowballSeed",
    "run_related",
    "run_snippets",
    "run_snowball",
    "s2_request_id",
    "s2_request_id_for_paper",
]
