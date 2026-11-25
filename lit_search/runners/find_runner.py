"""Instant relevance search: one ranked request per source, fused into one list.

``search`` answers "give me every record matching this Boolean query" and
pages through whole result sets sorted by date.  ``find`` answers the everyday
question instead: "what are the most relevant papers for this text, now".  It
sends one relevance-ranked request to each source in parallel, merges records
only on explicit identity, and orders the union by reciprocal-rank fusion.

Every source still reports a coverage status.  A ranked top-*k* request is a
planned cap, so ``CAPPED`` (with the source's own total) is the normal outcome
and is not an execution failure.  A source that errors or misses the deadline
is ``FAILED`` with its reason; the remaining sources still produce a result.
"""

from __future__ import annotations

import dataclasses
import datetime
import itertools
import json
import logging
import os
import re
import threading
import time
import traceback
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from lit_search.connectors.arxiv import _BASE_URL as _ARXIV_URL
from lit_search.connectors.arxiv import _NS as _ARXIV_NS
from lit_search.connectors.arxiv import ArxivConnector
from lit_search.connectors.openaire import _SEARCH_URL as _OPENAIRE_URL
from lit_search.connectors.openaire import OpenAIREConnector
from lit_search.connectors.openalex import _BASE_URL as _OPENALEX_URL
from lit_search.connectors.openalex import OpenAlexConnector
from lit_search.connectors.semantic_scholar import _PAPER_FIELDS as _S2_FIELDS
from lit_search.connectors.semantic_scholar import SemanticScholarConnector
from lit_search.core.author import Author
from lit_search.core.citation import IdentifierKind, PaperIdentifier
from lit_search.core.coverage import (
    CoverageLedger,
    CoverageStatus,
    CoveredPapers,
    ExecutionStatus,
    SourceCoverage,
)
from lit_search.core.identity import (
    has_version_identity_conflict,
    record_identity_keys,
    same_record_identity,
)
from lit_search.core.paper import Database, Paper
from lit_search.core.search_result import SearchResult
from lit_search.exceptions import ConnectorError, InvalidParameterError, RateLimitBusyError
from lit_search.store.store import default_store_root
from lit_search.utils.rate_limit import request_deadline

logger = logging.getLogger(__name__)

FIND_SOURCES: tuple[str, ...] = (
    "semantic_scholar",
    "openalex",
    "arxiv",
    "openaire",
    "huggingface",
)
DEFAULT_PER_SOURCE = 25
DEFAULT_DEADLINE_SECONDS = 20.0
# arXiv's limits behave like penalties (429/503 for minutes). Semantic Scholar's 429s
# are brief load-shedding: a cooldown there skipped it for most of a hunt (2026-10-10),
# so it gets one quick retry instead and no cooldown.
COOLDOWN_SECONDS = {"arxiv": 120.0, "semantic_scholar": 0.0}
DEFAULT_COOLDOWN_SECONDS = 20.0
_RRF_K = 30
# A meaning-based source vouches for papers that share no words with the query.
_SEMANTIC_SOURCE = "huggingface"
_UNREVIEWED_DOI = "10.5281/zenodo."
_UNREVIEWED_FACTOR = 0.4
_S2_RELEVANCE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_S2_MAX_LIMIT = 100
_OPENALEX_MAX_PER_PAGE = 200
_ARXIV_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "via",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "without",
        "do",
        "does",
        "can",
    ]
)
_PHRASE_RE = re.compile(r'"([^"]+)"')
_ARXIV_RELAX_MIN_TERMS = 3
_ARXIV_RELAX_MIN_COVERAGE = 0.5


@dataclass(frozen=True)
class FindRequest:
    """One instant search.

    Parameters
    ----------
    text : str
        Free-text query.  Double-quoted spans are kept as phrases where the
        source supports it.
    per_source : int
        Ranked results requested from each source.
    since : datetime.date | None
        Earliest publication date.
    until : datetime.date | None
        Latest publication date.
    """

    text: str
    per_source: int = DEFAULT_PER_SOURCE
    since: datetime.date | None = None
    until: datetime.date | None = None


@dataclass
class FindOutcome:
    """A finished instant search.

    Parameters
    ----------
    result : SearchResult
        The fused papers, in fused-rank order, with the coverage ledger.
    ranks : dict[int, dict[str, int]]
        ``id(paper) -> {source: 1-based rank}`` for every paper in *result*.
    seconds_per_source : dict[str, float]
        Wall time each source took, including sources that failed.
    tracebacks : dict[str, str]
        Tracebacks of unexpected (non-network) errors, by source.  These are
        bugs, kept so they can be reported rather than hidden.
    title_merges : int
        Records merged on title and year rather than a shared identifier.
    """

    result: SearchResult
    ranks: dict[int, dict[str, int]] = field(default_factory=dict)
    seconds_per_source: dict[str, float] = field(default_factory=dict)
    tracebacks: dict[str, str] = field(default_factory=dict)
    title_merges: int = 0


def _ranked_coverage(
    source: str, *, retrieved: int, seen: int, requested: int, total: int | None
) -> SourceCoverage:
    """Coverage for one ranked top-*k* request.

    Parameters
    ----------
    source : str
        Source identifier.
    retrieved : int
        Records parsed into papers.
    seen : int
        Records the source returned, parsed or not.
    requested : int
        The *k* asked for.
    total : int | None
        Matching records the source reported.

    Returns
    -------
    SourceCoverage
        ``COMPLETE`` when the whole match set came back, ``CAPPED`` when the
        source has more than *k*, ``PARTIAL`` when records were lost.
    """
    unparsed = seen - retrieved
    if unparsed > 0:
        return SourceCoverage.partial(
            source,
            retrieved,
            detail=f"{unparsed} of {seen} records could not be parsed",
            reported_total=total,
        )
    if total is not None and retrieved >= total:
        return SourceCoverage.from_pagination(source, retrieved=retrieved, reported_total=total)
    if seen >= requested:
        return SourceCoverage.capped(
            source,
            retrieved,
            cap=requested,
            reported_total=total,
            detail=f"top {requested} by relevance",
        )
    return SourceCoverage.from_pagination(source, retrieved=retrieved, reported_total=total)


def _arxiv_query(text: str) -> str:
    """Translate free text into an arXiv relevance query.

    Quoted spans stay phrases.  Terms are AND-ed: arXiv's relevance ranking
    over OR-ed terms surfaces papers that merely share one common word.

    Parameters
    ----------
    text : str
        Free-text query.

    Returns
    -------
    str
        arXiv ``search_query`` value.
    """
    phrases = [p.strip() for p in _PHRASE_RE.findall(text) if p.strip()]
    rest = _PHRASE_RE.sub(" ", text)
    tokens = [
        t
        for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", rest)
        if t.lower() not in _ARXIV_STOPWORDS
    ]
    if not phrases and not tokens:
        raise InvalidParameterError(f"query {text!r} has no searchable terms for arXiv")
    phrase_part = " AND ".join(f'all:"{p}"' for p in phrases)
    token_part = " AND ".join(f"all:{t}" for t in tokens)
    if phrase_part and token_part:
        return f"{phrase_part} AND ({token_part})"
    return phrase_part or token_part


def _arxiv_relaxed_query(text: str) -> str | None:
    """A looser arXiv query for when the strict AND of every term matches nothing.

    Adjacent unquoted terms are AND-ed in pairs and the pairs OR-ed, so a record
    must share at least one neighbouring pair of query words; quoted phrases stay
    required.  ``None`` when the query has too few terms for relaxing to differ
    from the strict form.

    Parameters
    ----------
    text : str
        Free-text query.

    Returns
    -------
    str | None
        arXiv ``search_query`` value, or ``None``.
    """
    phrases = [p.strip() for p in _PHRASE_RE.findall(text) if p.strip()]
    tokens = [
        t
        for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", _PHRASE_RE.sub(" ", text))
        if t.lower() not in _ARXIV_STOPWORDS
    ]
    if len(tokens) < _ARXIV_RELAX_MIN_TERMS:
        return None
    pairs = " OR ".join(f"(all:{a} AND all:{b})" for a, b in itertools.pairwise(tokens))
    if phrases:
        phrase_part = " AND ".join(f'all:"{p}"' for p in phrases)
        return f"{phrase_part} AND ({pairs})"
    return pairs


def _search_semantic_scholar(
    connector: SemanticScholarConnector, request: FindRequest
) -> CoveredPapers:
    limit = min(request.per_source, _S2_MAX_LIMIT)
    params: dict[str, Any] = {"query": request.text, "limit": limit, "fields": _S2_FIELDS}
    date_param = SemanticScholarConnector._build_ss_date_param(request.since, request.until)
    if date_param is not None:
        params["publicationDateOrYear"] = date_param
    data = connector._get(_S2_RELEVANCE_URL, params).json()
    items = data.get("data") or []
    papers = [p for p in (connector._parse_paper(item) for item in items) if p is not None]
    total = data.get("total") if isinstance(data.get("total"), int) else None
    return CoveredPapers(
        papers=tuple(papers),
        coverage=_ranked_coverage(
            connector.name, retrieved=len(papers), seen=len(items), requested=limit, total=total
        ),
    )


def _search_openalex(connector: OpenAlexConnector, request: FindRequest) -> CoveredPapers:
    limit = min(request.per_source, _OPENALEX_MAX_PER_PAGE)
    max_pub_date = (datetime.date.today() + datetime.timedelta(days=365)).isoformat()
    params = OpenAlexConnector._build_oa_page_params(
        {"search": request.text}, limit, "*", max_pub_date, request.since, request.until
    )
    params.pop("cursor", None)
    params["sort"] = "relevance_score:desc"
    data = connector._get(_OPENALEX_URL, params).json()
    items = data.get("results") or []
    papers = [p for p in (connector._parse_paper(item) for item in items) if p is not None]
    count = (data.get("meta") or {}).get("count")
    return CoveredPapers(
        papers=tuple(papers),
        coverage=_ranked_coverage(
            connector.name,
            retrieved=len(papers),
            seen=len(items),
            requested=limit,
            total=count if isinstance(count, int) else None,
        ),
    )


def _arxiv_request(
    connector: ArxivConnector, query: str, request: FindRequest
) -> tuple[list[Paper], int, int | None]:
    if request.since or request.until:
        from_date = request.since.strftime("%Y%m%d") + "0000" if request.since else "199101010000"
        to_date = request.until.strftime("%Y%m%d") + "2359" if request.until else "999912312359"
        query = f"({query}) AND submittedDate:[{from_date} TO {to_date}]"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": request.per_source,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    response = connector._get(_ARXIV_URL, params)
    try:
        tree = ET.fromstring(response.text)
    except ET.ParseError as exc:
        raise ConnectorError(f"arXiv returned an unparseable response: {exc}") from exc
    total_el = tree.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
    total = int(total_el.text.strip()) if total_el is not None and total_el.text else None
    entries = tree.findall("atom:entry", _ARXIV_NS)
    papers = [p for p in (connector._parse_paper(e) for e in entries) if p is not None]
    return papers, len(entries), total


def _search_arxiv(connector: ArxivConnector, request: FindRequest) -> CoveredPapers:
    papers, seen, total = _arxiv_request(connector, _arxiv_query(request.text), request)
    coverage = _ranked_coverage(
        connector.name, retrieved=len(papers), seen=seen, requested=request.per_source, total=total
    )
    if seen == 0 and total == 0:
        relaxed = _arxiv_relaxed_query(request.text)
        if relaxed is None:
            coverage = SourceCoverage.complete(connector.name, 0, reported_total=0)
            return CoveredPapers(papers=(), coverage=coverage)
        # arXiv search is Boolean, unlike the other find sources: a long query
        # AND-ed word by word usually matches nothing, and reporting that as a
        # complete zero reads as "arXiv has no such papers".
        wider = dataclasses.replace(request, per_source=min(2 * request.per_source, 100))
        candidates, _, _ = _arxiv_request(connector, relaxed, wider)
        terms = query_terms(request.text)
        papers = [p for p in candidates if term_coverage(p, terms) >= _ARXIV_RELAX_MIN_COVERAGE]
        coverage = SourceCoverage.partial(
            connector.name,
            len(papers),
            detail=(
                f"no record matched every term; {len(papers)} relaxed matches "
                f"(adjacent term pairs, at least half the terms)"
            ),
        )
    return CoveredPapers(papers=tuple(papers), coverage=coverage)


def _search_openaire(connector: OpenAIREConnector, request: FindRequest) -> CoveredPapers:
    params: dict[str, Any] = {
        "search": request.text,
        "pageSize": request.per_source,
        "sortBy": "relevance DESC",
    }
    if request.since is not None:
        params["fromPublicationDate"] = request.since.isoformat()
    if request.until is not None:
        params["toPublicationDate"] = request.until.isoformat()
    body = connector._read_body(connector._get(_OPENAIRE_URL, params))
    items = body.get("results") or []
    papers = [p for p in (connector._parse_item(item) for item in items) if p is not None]
    num_found = (body.get("header") or {}).get("numFound")
    return CoveredPapers(
        papers=tuple(papers),
        coverage=_ranked_coverage(
            connector.name,
            retrieved=len(papers),
            seen=len(items),
            requested=request.per_source,
            total=num_found if isinstance(num_found, int) else None,
        ),
    )


class HuggingFacePapers:
    """Minimal client for Hugging Face Papers search (hybrid semantic + keyword).

    Covers ML papers someone submitted to Hugging Face, keyed by arXiv id.
    ``HF_TOKEN`` is sent when set; the endpoint also answers anonymously.
    """

    name = "huggingface"
    url = "https://huggingface.co/api/papers/search"
    max_limit = 100

    def __init__(self, token: str | None = None, timeout: float = 15.0) -> None:
        self._token = token
        self._timeout = timeout
        self._session = requests.Session()

    def search(self, text: str, limit: int) -> list[dict[str, Any]]:
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        response = self._session.get(
            self.url,
            params={"q": text[:250], "limit": str(min(limit, self.max_limit))},
            headers=headers,
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            raise ConnectorError("Hugging Face papers search did not return a list")
        return data


def _parse_hf_item(item: dict[str, Any]) -> Paper | None:
    nested = item.get("paper")
    paper: dict[str, Any] = nested if isinstance(nested, dict) else item
    arxiv_id = str(paper.get("id") or "").strip()
    title = " ".join(str(paper.get("title") or item.get("title") or "").split())
    if not arxiv_id or not title:
        return None
    published = str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10]
    try:
        date = datetime.date.fromisoformat(published) if published else None
    except ValueError:
        date = None
    authors = []
    for author in paper.get("authors") or []:
        name = (author or {}).get("name") if isinstance(author, dict) else None
        if name and str(name).strip():
            authors.append(Author(name=str(name)))
    return Paper(
        title=title,
        abstract=" ".join(str(paper.get("summary") or item.get("summary") or "").split()),
        authors=authors,
        source=None,
        publication_date=date,
        url=f"https://arxiv.org/abs/{arxiv_id}",
        doi=f"10.48550/arXiv.{arxiv_id}",
        found_in={Database.HUGGINGFACE},
        identifiers=[PaperIdentifier(IdentifierKind.ARXIV, arxiv_id)],
    )


def _search_huggingface(connector: HuggingFacePapers, request: FindRequest) -> CoveredPapers:
    if request.since or request.until:
        return CoveredPapers.none(
            SourceCoverage.query_unsupported(
                connector.name, detail="Hugging Face Papers search has no date filter"
            )
        )
    limit = min(request.per_source, connector.max_limit)
    items = connector.search(request.text, limit)
    papers = [p for p in (_parse_hf_item(item) for item in items) if p is not None]
    unparsed = len(items) - len(papers)
    if unparsed:
        coverage = SourceCoverage.partial(
            connector.name,
            len(papers),
            detail=f"{unparsed} of {len(items)} records could not be parsed",
        )
    elif len(items) >= limit:
        coverage = SourceCoverage.capped(
            connector.name, len(papers), cap=limit, detail="top by hybrid relevance"
        )
    else:
        coverage = SourceCoverage.partial(
            connector.name,
            len(papers),
            detail="all results the source ranked; it reports no match total",
            execution_status=ExecutionStatus.COMPLETE,
        )
    return CoveredPapers(papers=tuple(papers), coverage=coverage)


_SEARCHERS: dict[str, Callable[[Any, FindRequest], CoveredPapers]] = {
    "semantic_scholar": _search_semantic_scholar,
    "openalex": _search_openalex,
    "arxiv": _search_arxiv,
    "openaire": _search_openaire,
    "huggingface": _search_huggingface,
}


def build_find_connectors(
    sources: tuple[str, ...],
    *,
    semantic_scholar_api_key: str | None = None,
    openalex_api_key: str | None = None,
    email: str | None = None,
    openaire_access_token: str | None = None,
    openaire_refresh_token: str | None = None,
) -> dict[str, Any]:
    """Instantiate the connectors ``find`` will query.

    Parameters
    ----------
    sources : tuple[str, ...]
        Source identifiers, each one of :data:`FIND_SOURCES`.
    semantic_scholar_api_key : str | None
        Semantic Scholar key.
    openalex_api_key : str | None
        OpenAlex key.
    email : str | None
        Contact address for OpenAlex.
    openaire_access_token : str | None
        OpenAIRE access token.
    openaire_refresh_token : str | None
        OpenAIRE refresh token.

    Returns
    -------
    dict[str, Any]
        Connector by source identifier.

    Raises
    ------
    InvalidParameterError
        For a source ``find`` does not support.
    """
    unknown = [s for s in sources if s not in _SEARCHERS]
    if unknown:
        raise InvalidParameterError(
            f"find does not support {', '.join(unknown)}; choose from {', '.join(FIND_SOURCES)}"
        )
    factories: dict[str, Callable[[], Any]] = {
        "semantic_scholar": lambda: SemanticScholarConnector(api_key=semantic_scholar_api_key),
        "openalex": lambda: OpenAlexConnector(api_key=openalex_api_key, email=email),
        "arxiv": ArxivConnector,
        "openaire": lambda: OpenAIREConnector(
            access_token=openaire_access_token, refresh_token=openaire_refresh_token
        ),
        "huggingface": lambda: HuggingFacePapers(token=os.environ.get("HF_TOKEN") or None),
    }
    connectors = {source: factories[source]() for source in sources}
    for connector in connectors.values():
        if isinstance(connector, HuggingFacePapers):
            continue
        # Interactive search fails fast and says why; the connectors' default
        # back-off (several retries, honouring Retry-After) can stall a single
        # request for most of a minute against a source that has said no.
        connector._max_retries = 1
        connector._max_rate_limit_retries = 0
        connector._timeout = 15.0
    if "semantic_scholar" in connectors:
        # Measured 2026-10-11: keyed requests paced at 3 s still got 429 about half
        # the time, with no Retry-After. Retrying is what works; 1.5/3/6 s fits
        # inside the find deadline.
        connectors["semantic_scholar"]._max_rate_limit_retries = 3
        connectors["semantic_scholar"]._rate_limit_base_delay = 1.5
    return connectors


def _cooldown_path() -> Path:
    return default_store_root() / "find-cooldown.json"


def _read_cooldowns() -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(_cooldown_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _note_refusal(source: str, coverage: SourceCoverage) -> None:
    """Start a cooldown when a source refused or stalled."""
    text = f"{coverage.error_type or ''} {coverage.error or ''}"
    if not re.search(r"\b(429|503)\b|Timeout|timed out", text):
        return
    seconds = COOLDOWN_SECONDS.get(source, DEFAULT_COOLDOWN_SECONDS)
    if seconds <= 0:
        return
    cooldowns = _read_cooldowns()
    cooldowns[source] = {
        "until": time.time() + seconds,
        "reason": re.sub(r"\?\S*", "", re.sub(r"\s+", " ", text)).strip()[:120],
    }
    path = _cooldown_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cooldowns), encoding="utf-8")
    except OSError:
        logger.warning("could not record a cooldown for %s", source, exc_info=True)


def _cooling(source: str) -> SourceCoverage | None:
    entry = _read_cooldowns().get(source)
    if not entry or float(entry.get("until", 0)) <= time.time():
        return None
    until = datetime.datetime.fromtimestamp(float(entry["until"])).strftime("%H:%M:%S")
    return SourceCoverage.skipped_by_policy(
        source,
        detail=f"cooling down until {until} after it refused or stalled ({entry.get('reason')})",
    )


def _run_one(source: str, connector: Any, request: FindRequest) -> tuple[CoveredPapers, str | None]:
    """Run one source, converting every failure into an explicit status.

    Network and connector errors are ordinary failures.  Anything else is a
    bug; it is still converted, so one broken source cannot take the others
    down, but its traceback is returned so it is reported rather than hidden.

    Parameters
    ----------
    source : str
        Source identifier.
    connector : Any
        Connector instance.
    request : FindRequest
        The search.

    Returns
    -------
    tuple[CoveredPapers, str | None]
        The result and, for an unexpected error, its traceback.
    """
    try:
        return _SEARCHERS[source](connector, request), None
    except RateLimitBusyError as exc:
        return (
            CoveredPapers.none(
                SourceCoverage.failed(source, error=str(exc), error_type="RateLimitBusy")
            ),
            None,
        )
    except (requests.RequestException, ConnectorError) as exc:
        return (
            CoveredPapers.none(
                SourceCoverage.failed(
                    source, error=str(exc) or repr(exc), error_type=type(exc).__name__
                )
            ),
            None,
        )
    except Exception as exc:
        logger.error("find: unexpected error from %s", source, exc_info=True)
        return (
            CoveredPapers.none(
                SourceCoverage.failed(
                    source,
                    error=f"unexpected error (bug): {type(exc).__name__}: {exc}",
                    error_type=type(exc).__name__,
                )
            ),
            traceback.format_exc(),
        )


def _title_key(paper: Paper) -> tuple[str, int] | None:
    title = re.sub(r"\W+", "", (paper.title or "").casefold())
    if len(title) < 12 or paper.publication_date is None:
        return None
    return title, paper.publication_date.year


def query_terms(text: str) -> list[str]:
    """Content words of a query, crudely stemmed, for the term-coverage check.

    Parameters
    ----------
    text : str
        Free-text query.

    Returns
    -------
    list[str]
        Lower-cased word stems (first six letters), stopwords removed.
    """
    words = re.findall(r"[a-z0-9]+", text.casefold())
    return list(dict.fromkeys(w[:6] for w in words if w not in _ARXIV_STOPWORDS and len(w) > 2))


def term_coverage(paper: Paper, terms: list[str]) -> float:
    """Fraction of query terms that occur in the paper's title or abstract.

    Parameters
    ----------
    paper : Paper
        Candidate.
    terms : list[str]
        Output of :func:`query_terms`.

    Returns
    -------
    float
        In ``[0, 1]``; ``1.0`` when there are no terms.
    """
    if not terms:
        return 1.0
    words = {
        w[:6] for w in re.findall(r"[a-z0-9]+", f"{paper.title} {paper.abstract or ''}".casefold())
    }
    return sum(term in words for term in terms) / len(terms)


def _fuse(
    per_source: dict[str, tuple[Paper, ...]],
    terms: list[str] | None = None,
) -> tuple[list[Paper], dict[int, dict[str, int]], int]:
    """Merge records and order them by reciprocal-rank fusion.

    Ordering is reciprocal-rank fusion of the sources' own rankings, scaled
    by how many query terms occur in the paper's title or abstract (when
    *terms* is given): a source's top hit that shares no words with the query
    sinks instead of leading the list.  A record only one source found whose
    DOI is a self-deposit (Zenodo) is scaled down too: keyword-stuffed
    self-deposits top OpenAIRE's ranking for broad queries.

    Records merge on explicit identity first.  A record that shares no
    identifier with another but has the same normalised title and year, and
    no conflicting DOI, is folded in as well: sources such as Semantic Scholar
    often index a new preprint before linking its arXiv identifier, and
    showing it twice would split its fused score.  The title match is only
    used when both years are known and equal, exactly one earlier record
    matches, and the two state no conflicting version identity, so a preprint
    and its published version stay distinct.

    Parameters
    ----------
    per_source : dict[str, tuple[Paper, ...]]
        Each source's papers, in that source's relevance order.

    Returns
    -------
    tuple[list[Paper], dict[int, dict[str, int]], int]
        Fused papers (best first), each paper's per-source ranks, and how many
        records were merged on title rather than identifier.
    """
    merged: list[Paper] = []
    by_key: dict[str, list[Paper]] = {}
    by_title: dict[tuple[str, int], list[Paper]] = {}
    ranks: dict[int, dict[str, int]] = {}
    title_merges = 0
    for source, papers in per_source.items():
        for rank, paper in enumerate(papers, start=1):
            candidates = {
                id(c): c for key in record_identity_keys(paper) for c in by_key.get(key, ())
            }
            representative = next(
                (c for c in candidates.values() if same_record_identity(c, paper)), None
            )
            title_key = _title_key(paper)
            if representative is None and title_key is not None:
                same_title = by_title.get(title_key, [])
                if len(same_title) == 1 and not has_version_identity_conflict(same_title[0], paper):
                    representative = same_title[0]
                    title_merges += 1
            if representative is None:
                representative = paper
                merged.append(representative)
            else:
                representative.merge(paper)
            source_ranks = ranks.setdefault(id(representative), {})
            source_ranks[source] = min(rank, source_ranks.get(source, rank))
            for key in record_identity_keys(representative):
                bucket = by_key.setdefault(key, [])
                if all(existing is not representative for existing in bucket):
                    bucket.append(representative)
            if title_key is not None:
                title_bucket = by_title.setdefault(title_key, [])
                if all(existing is not representative for existing in title_bucket):
                    title_bucket.append(representative)

    def score(paper: Paper) -> float:
        fused = sum(1.0 / (_RRF_K + r) for r in ranks[id(paper)].values())
        if terms and _SEMANTIC_SOURCE not in ranks[id(paper)]:
            fused *= 0.25 + 0.75 * term_coverage(paper, terms)
        if len(ranks[id(paper)]) == 1 and (paper.doi or "").lower().startswith(_UNREVIEWED_DOI):
            fused *= _UNREVIEWED_FACTOR
        return fused

    merged.sort(key=score, reverse=True)
    return merged, ranks, title_merges


def run_find(
    request: FindRequest,
    connectors: dict[str, Any],
    *,
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
) -> FindOutcome:
    """Run an instant search across *connectors* in parallel.

    Parameters
    ----------
    request : FindRequest
        The search.
    connectors : dict[str, Any]
        Connector by source identifier, from :func:`build_find_connectors`.
    deadline_seconds : float
        Wall-clock budget.  A source still running at the deadline is reported
        ``FAILED`` (timed out) and its late answer is discarded.

    Returns
    -------
    FindOutcome
        Fused result, ranks, timings and any unexpected-error tracebacks.
    """
    if not request.text.strip():
        raise InvalidParameterError("find needs a non-empty query")
    if request.per_source < 1:
        raise InvalidParameterError("per-source result count must be at least 1")

    start = time.monotonic()
    lock = threading.Lock()
    closed = False
    answers: dict[str, tuple[CoveredPapers, str | None, float]] = {}
    all_done = threading.Event()

    # Wall-clock deadline for slot reservations: a source that cannot get a
    # request slot in time reports itself busy instead of queueing a request
    # nobody will read.  One second is left for the request itself.
    slot_deadline = time.time() + max(0.0, deadline_seconds - 1.0)

    def worker(source: str) -> None:
        with request_deadline(slot_deadline):
            answer = _run_one(source, connectors[source], request)
        with lock:
            if not closed:
                answers[source] = (*answer, time.monotonic() - start)
                if len(answers) == len(connectors) - len(skipped):
                    all_done.set()

    # Daemon threads: a source still retrying after the deadline must not keep
    # the process alive, and its late answer is dropped by the closed gate.
    skipped: dict[str, SourceCoverage] = {}
    for source in connectors:
        cooling = _cooling(source)
        if cooling is not None:
            skipped[source] = cooling
            continue
        threading.Thread(target=worker, args=(source,), name=f"find-{source}", daemon=True).start()
    if len(connectors) > len(skipped):
        all_done.wait(timeout=deadline_seconds)
    with lock:
        closed = True
        settled = dict(answers)

    per_source: dict[str, tuple[Paper, ...]] = {}
    coverages: list[SourceCoverage] = []
    tracebacks: dict[str, str] = {}
    seconds: dict[str, float] = {}
    for source in connectors:
        if source in skipped:
            coverages.append(skipped[source])
            seconds[source] = 0.0
            continue
        if source not in settled:
            _note_refusal(
                source, SourceCoverage.failed(source, error="deadline", error_type="Timeout")
            )
            coverages.append(
                SourceCoverage.failed(
                    source,
                    error=f"no answer within the {deadline_seconds:g}s deadline",
                    error_type="Timeout",
                )
            )
            seconds[source] = deadline_seconds
            continue
        covered, trace, elapsed = settled[source]
        covered = covered.with_source(source)
        if covered.coverage.status is CoverageStatus.FAILED:
            _note_refusal(source, covered.coverage)
        per_source[source] = covered.papers
        coverages.append(covered.coverage)
        seconds[source] = elapsed
        if trace:
            tracebacks[source] = trace

    papers, ranks, title_merges = _fuse(per_source, query_terms(request.text))
    result = SearchResult(
        query=request.text,
        since=request.since,
        until=request.until,
        max_papers_per_database=request.per_source,
        databases=list(connectors),
        papers=papers,
        runtime_seconds=time.monotonic() - start,
        runtime_seconds_per_database=seconds,
        failed_databases=[c.source for c in coverages if c.status is CoverageStatus.FAILED],
        enrichment_databases=[],
        coverage=CoverageLedger.from_sources(coverages),
    )
    return FindOutcome(
        result=result,
        ranks=ranks,
        seconds_per_source=seconds,
        tracebacks=tracebacks,
        title_merges=title_merges,
    )


__all__ = [
    "DEFAULT_DEADLINE_SECONDS",
    "DEFAULT_PER_SOURCE",
    "FIND_SOURCES",
    "FindOutcome",
    "FindRequest",
    "build_find_connectors",
    "run_find",
]
