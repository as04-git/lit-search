"""Semantic Scholar searcher implementation."""

from __future__ import annotations

import contextlib
import datetime
import logging
import re
from collections.abc import Callable
from typing import Any

import requests

from lit_search.connectors.doi_lookup_base import DOILookupConnectorBase
from lit_search.connectors.search_base import SearchConnectorBase
from lit_search.connectors.url_lookup_base import URLLookupConnectorBase
from lit_search.core.author import Author
from lit_search.core.citation import (
    CitationEdge,
    CitationTarget,
    IdentifierKind,
    NativeBatchResult,
    PaperIdentifier,
)
from lit_search.core.coverage import CoveredPapers, SourceCoverage
from lit_search.core.paper import Database, Paper, PaperType
from lit_search.core.query import Query
from lit_search.core.source import Source, SourceType
from lit_search.query.builder import QueryBuilder
from lit_search.query.builders.semantic_scholar import SemanticScholarQueryBuilder
from lit_search.utils.pagination_logging import (
    log_pagination_empty_before_total,
    log_pagination_request_failure,
)

logger = logging.getLogger(__name__)

_BULK_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
_PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper"
_PAPER_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
_AUTHOR_BATCH_URL = "https://api.semanticscholar.org/graph/v1/author/batch"
# Recommendations API lives on a separate host from the graph API used above.
_RECOMMENDATIONS_URL = "https://api.semanticscholar.org/recommendations/v1/papers/forpaper"
_PAGE_SIZE = 100  # Semantic Scholar max per request
_CITATION_PAGE_SIZE = 1000  # Max per page for citations/references endpoints
_AUTHOR_BATCH_SIZE = 1000  # Max authors per batch request
_PAPER_BATCH_SIZE = 500
# The endpoint accepts 500 IDs but caps each response at 10 MB. A 183-paper
# request that combined full metadata with nested edges produced a live HTTP
# 504 on 2026-08-31. Traversal therefore uses a deliberately small field set
# and a conservative 250-ID width; terminal metadata-only batches retain 500.
_PAPER_EDGE_BATCH_SIZE = 250

# Rate limits: 1000 req/s without key (shared among all unauthenticated users),
# 1 req/s with key (introductory; can be increased upon request)
_MIN_REQUEST_INTERVAL_DEFAULT = 1.1  # conservative for shared pool
_MIN_REQUEST_INTERVAL_WITH_KEY = 1.1  # respects 1 RPS introductory limit

# Fields to retrieve in each paper record
_PAPER_FIELDS = (
    "paperId,externalIds,title,abstract,authors,year,publicationDate,"
    "journal,venue,citationCount,openAccessPdf,url,fieldsOfStudy,"
    "s2FieldsOfStudy,publicationTypes,publicationVenue,isOpenAccess,referenceCount"
)

_PAPER_EDGE_BASE_FIELDS = (
    "paperId,externalIds,title,year,publicationDate,citationCount,referenceCount"
)
_PAPER_FIELDS_WITH_EDGES = (
    f"{_PAPER_FIELDS},references.paperId,references.externalIds,"
    "citations.paperId,citations.externalIds"
)

# Mapping from Semantic Scholar publicationVenue.type to SourceType.
_SS_VENUE_TYPE_MAP: dict[str, SourceType] = {
    "journal": SourceType.JOURNAL,
    "conference": SourceType.CONFERENCE,
    "book": SourceType.BOOK,
    "repository": SourceType.REPOSITORY,
}

# Regex that matches Semantic Scholar paper landing-page URLs and captures the
# 40-character hexadecimal paper ID.
# Handles:
#   https://www.semanticscholar.org/paper/Attention-is-All-you-Need/204e3073870fae3d05bcbc2f6a8e263d9b72e776
#   https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776
_SS_URL_RE = re.compile(
    r"semanticscholar\.org/paper/(?:[^/]+/)?([0-9a-f]{40})",
    re.IGNORECASE,
)

# Mapping from publicationTypes list entries to SourceType (fallback).
_SS_PUB_TYPE_MAP: dict[str, SourceType] = {
    "JournalArticle": SourceType.JOURNAL,
    "Review": SourceType.JOURNAL,
    "Conference": SourceType.CONFERENCE,
    "Book": SourceType.BOOK,
    "BookSection": SourceType.BOOK,
}

# Mapping from Semantic Scholar publicationTypes entries to PaperType.
_SS_PAPER_TYPE_MAP: dict[str, PaperType] = {
    "JournalArticle": PaperType.ARTICLE,
    "Review": PaperType.ARTICLE,
    "CaseReport": PaperType.ARTICLE,
    "ClinicalTrial": PaperType.ARTICLE,
    "Editorial": PaperType.ARTICLE,
    "LettersAndComments": PaperType.ARTICLE,
    "MetaAnalysis": PaperType.ARTICLE,
    "Study": PaperType.ARTICLE,
    "Conference": PaperType.INPROCEEDINGS,
    "Book": PaperType.BOOK,
    "BookSection": PaperType.INBOOK,
    "Dataset": PaperType.MISC,
    "News": PaperType.MISC,
}


class SemanticScholarConnector(
    SearchConnectorBase,
    DOILookupConnectorBase,
    URLLookupConnectorBase,
):
    """Connector for the Semantic Scholar research corpus.

    Uses the Bulk Search endpoint:
    https://api.semanticscholar.org/api-docs/#tag/Paper-Data/operation/bulk_paper_search

    Rate limits:
    - Without API key: up to 1000 req/s (shared among all unauthenticated users)
    - With API key: 1 req/s (introductory; can be increased upon request)
    """

    # Measured 2026-10-16 on an idle key, one sequential client: 45-65% of
    # requests refused (429) at one per 1.1 s and at one per 3 s alike, on
    # lookup, batch, bulk and relevance search, from the first request.
    # Refusals that do not depend on our rate are not cured by slowing every
    # process down; they are cured by retrying soon.  So: short jittered
    # pauses, mild and brief widening, and more quick retries (each move's
    # deadline still caps them).
    _rate_limit_base_delay: float = 1.0
    _max_rate_limit_retries: int = 6
    _refusal_max_factor: float = 1.5
    _refusal_slow_seconds: float = 60.0
    _retry_max_delay: float = 4.0

    def __init__(
        self,
        query_builder: SemanticScholarQueryBuilder | None = None,
        api_key: str | None = None,
    ) -> None:
        """Create a Semantic Scholar searcher.

        Parameters
        ----------
        query_builder : SemanticScholarQueryBuilder | None
            Builder used to validate and convert queries.  When ``None`` a
            default :class:`SemanticScholarQueryBuilder` is created automatically.
        api_key : str | None
            Semantic Scholar API key (optional; provides a dedicated 1 RPS quota,
            decoupled from the shared unauthenticated pool, and can be increased
            upon request).
        """
        super().__init__()
        self._query_builder: SemanticScholarQueryBuilder = (
            query_builder or SemanticScholarQueryBuilder()
        )
        self._api_key = api_key
        self._request_interval = (
            _MIN_REQUEST_INTERVAL_WITH_KEY if api_key else _MIN_REQUEST_INTERVAL_DEFAULT
        )

        if not api_key:
            logger.warning(
                "No API key provided for Semantic Scholar. "
                "Without a key, requests share the unauthenticated pool "
                "(up to 1000 req/s shared among all anonymous users). "
                "Request a key at https://www.semanticscholar.org/product/api "
                "for a dedicated quota."
            )

    @property
    def name(self) -> str:
        """Return the database identifier.

        Returns
        -------
        str
            Database name.
        """
        return Database.SEMANTIC_SCHOLAR.value

    @property
    def query_builder(self) -> QueryBuilder:
        """Return the Semantic Scholar query builder.

        Returns
        -------
        QueryBuilder
            The underlying builder instance.
        """
        return self._query_builder

    @property
    def min_request_interval(self) -> float:
        """Return the minimum seconds between HTTP requests.

        Returns
        -------
        float
            Interval in seconds (varies with API key).
        """
        return self._request_interval

    def _prepare_headers(self, headers: dict) -> dict:
        """Inject the Semantic Scholar API key header when configured.

        Parameters
        ----------
        headers : dict
            Raw HTTP headers.

        Returns
        -------
        dict
            Headers with ``x-api-key`` added when a key is set.
        """
        updated = super()._prepare_headers(headers)
        if self._api_key:
            updated["x-api-key"] = self._api_key
        return updated

    # ------------------------------------------------------------------
    # URL lookup
    # ------------------------------------------------------------------

    @property
    def url_pattern(self) -> re.Pattern[str]:
        """Return the regex matching Semantic Scholar paper landing-page URLs.

        Returns
        -------
        re.Pattern[str]
            Compiled regex whose first capture group is the 40-char paper ID.
        """
        return _SS_URL_RE

    def fetch_paper_by_id(self, paper_id: str) -> Paper | None:
        """Fetch a single paper by its Semantic Scholar paper ID.

        Parameters
        ----------
        paper_id : str
            40-character hexadecimal Semantic Scholar paper ID.

        Returns
        -------
        Paper | None
            A populated :class:`~lit_search.core.paper.Paper`, or ``None``
            when the paper is not found or the response cannot be parsed.
        """
        url = f"{_PAPER_URL}/{paper_id}"
        params = {"fields": _PAPER_FIELDS_WITH_EDGES}
        try:
            response = self._get(url, params=params)
            data = response.json()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                logger.debug("Semantic Scholar: paper ID %s not found (404).", paper_id)
                return None
            logger.debug("Semantic Scholar: HTTP error fetching paper ID %s: %s", paper_id, exc)
            return None
        except (requests.RequestException, ValueError):
            logger.debug("Semantic Scholar: failed to fetch paper ID %s.", paper_id)
            return None

        return self._parse_paper(data)

    # ------------------------------------------------------------------
    # DOI lookup
    # ------------------------------------------------------------------

    def fetch_paper_by_doi(self, doi: str) -> Paper | None:
        """Fetch a single paper by its DOI from Semantic Scholar.

        Queries ``GET /graph/v1/paper/DOI:{doi}?fields=...`` and converts
        the response into a :class:`~lit_search.core.paper.Paper`.

        Parameters
        ----------
        doi : str
            Bare DOI identifier (e.g. ``"10.1038/nature12373"``).

        Returns
        -------
        Paper | None
            A populated :class:`~lit_search.core.paper.Paper`, or ``None``
            when the DOI is not found or the response cannot be parsed.
        """
        url = f"{_PAPER_URL}/DOI:{doi}"
        params = {"fields": _PAPER_FIELDS_WITH_EDGES}
        try:
            response = self._get(url, params=params)
            data = response.json()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                logger.debug("Semantic Scholar: DOI %s not found (404).", doi)
                return None
            logger.debug("Semantic Scholar: HTTP error fetching DOI %s: %s", doi, exc)
            return None
        except (requests.RequestException, ValueError):
            logger.debug("Semantic Scholar: failed to fetch DOI %s.", doi)
            return None

        return self._parse_paper(data)

    def fetch_papers_batch(
        self,
        identifiers: list[str],
        *,
        include_edges: bool = True,
        include_references: bool = True,
        include_citations: bool = True,
    ) -> NativeBatchResult:
        """Fetch paper identifiers, splitting them into source-native requests.

        The API returns one array position per requested identifier, using
        ``null`` for an identifier it cannot resolve.  Request and parse
        failures remain explicit in the returned outcome; they are not folded
        into an ordinary not-found count.
        """
        papers: list[Paper] = []
        not_found = 0
        parse_failures = 0
        truncated_edge_records = 0
        request_failures = 0
        errors: list[str] = []
        batch_size = self.max_paper_batch_size(include_edges=include_edges)
        for start in range(0, len(identifiers), batch_size):
            batch = identifiers[start : start + batch_size]
            outcome = self.fetch_papers_batch_request(
                batch,
                include_edges=include_edges,
                include_references=include_references,
                include_citations=include_citations,
            )
            papers.extend(outcome.papers)
            not_found += outcome.not_found
            parse_failures += outcome.parse_failures
            truncated_edge_records += outcome.truncated_edge_records
            request_failures += outcome.request_failures
            errors.extend(f"batch at offset {start}: {error}" for error in outcome.error_messages)

        return NativeBatchResult(
            source=self.name,
            papers=tuple(papers),
            requested=len(identifiers),
            not_found=not_found,
            parse_failures=parse_failures,
            truncated_edge_records=truncated_edge_records,
            request_failures=request_failures,
            error_messages=tuple(errors),
        )

    @staticmethod
    def max_paper_batch_size(*, include_edges: bool = True) -> int:
        """Return the canonical maximum width of one paper-batch request.

        Edge-bearing responses use a lower limit because the endpoint's
        response-size cap can reject otherwise valid 500-position batches.
        """
        return _PAPER_EDGE_BATCH_SIZE if include_edges else _PAPER_BATCH_SIZE

    def fetch_papers_batch_request(
        self,
        identifiers: list[str],
        *,
        include_edges: bool = True,
        include_references: bool = True,
        include_citations: bool = True,
    ) -> NativeBatchResult:
        """Execute at most one Semantic Scholar paper-batch HTTP request.

        Inputs wider than :meth:`max_paper_batch_size` are rejected rather
        than silently chunked.  This gives callers an exact request boundary
        that can be durably checkpointed and replayed.
        """
        limit = self.max_paper_batch_size(include_edges=include_edges)
        if len(identifiers) > limit:
            raise ValueError(
                f"Semantic Scholar paper batch accepts at most {limit} identifiers "
                f"for include_edges={include_edges}; got {len(identifiers)}"
            )
        if not identifiers:
            return NativeBatchResult(source=self.name, papers=(), requested=0)

        if include_edges:
            nested_fields: list[str] = []
            if include_references:
                nested_fields.extend(["references.paperId", "references.externalIds"])
            if include_citations:
                nested_fields.extend(["citations.paperId", "citations.externalIds"])
            fields = _PAPER_EDGE_BASE_FIELDS
            if nested_fields:
                fields = f"{fields},{','.join(nested_fields)}"
        else:
            fields = _PAPER_FIELDS

        try:
            response = self._post(
                _PAPER_BATCH_URL,
                params={"fields": fields},
                json_body={"ids": identifiers},
            )
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            return NativeBatchResult(
                source=self.name,
                papers=(),
                requested=len(identifiers),
                request_failures=len(identifiers),
                error_messages=(f"{type(exc).__name__}: {exc}",),
            )

        if not isinstance(payload, list):
            return NativeBatchResult(
                source=self.name,
                papers=(),
                requested=len(identifiers),
                request_failures=len(identifiers),
                error_messages=("response was not a list",),
            )

        papers: list[Paper] = []
        not_found = 0
        parse_failures = 0
        truncated_edge_records = 0
        errors: list[str] = []
        # The endpoint's response is positional.  Ignore impossible surplus
        # positions rather than violating NativeBatchResult's accounting.
        for item in payload[: len(identifiers)]:
            if item is None:
                not_found += 1
                continue
            if not isinstance(item, dict):
                parse_failures += 1
                continue
            try:
                paper = self._parse_paper(item)
            except (KeyError, TypeError, ValueError):
                paper = None
            if paper is None:
                parse_failures += 1
                continue
            papers.append(paper)
            if include_edges:
                returned_refs = len(item.get("references") or [])
                returned_citations = len(item.get("citations") or [])
                expected_refs = item.get("referenceCount")
                expected_citations = item.get("citationCount")
                if (
                    include_references
                    and isinstance(expected_refs, int)
                    and returned_refs < expected_refs
                ):
                    truncated_edge_records += expected_refs - returned_refs
                if (
                    include_citations
                    and isinstance(expected_citations, int)
                    and returned_citations < expected_citations
                ):
                    truncated_edge_records += expected_citations - returned_citations

        missing_positions = max(0, len(identifiers) - len(payload))
        if missing_positions:
            errors.append(f"returned {len(payload)} positions for {len(identifiers)} identifiers")
        if len(payload) > len(identifiers):
            errors.append(
                f"returned {len(payload)} positions for {len(identifiers)} identifiers; "
                "ignored surplus positions"
            )

        return NativeBatchResult(
            source=self.name,
            papers=tuple(papers),
            requested=len(identifiers),
            not_found=not_found,
            parse_failures=parse_failures,
            truncated_edge_records=truncated_edge_records,
            request_failures=missing_positions,
            error_messages=tuple(errors),
        )

    # ------------------------------------------------------------------
    # Forward citation lookup
    # ------------------------------------------------------------------

    def _fetch_citation_page(
        self,
        doi: str,
        endpoint: str,
        offset: int,
    ) -> tuple[list[Paper], int]:
        """Fetch one page of references or citations for a paper.

        Parameters
        ----------
        doi : str
            DOI of the paper.
        endpoint : str
            ``"references"`` or ``"citations"``.
        offset : int
            Pagination offset.

        Returns
        -------
        tuple[list[Paper], int]
            Parsed papers from this page and the total ``next`` offset
            (``-1`` when there are no more pages).
        """
        url = f"{_PAPER_URL}/DOI:{doi}/{endpoint}"
        params: dict[str, Any] = {
            "fields": _PAPER_FIELDS,
            "limit": _CITATION_PAGE_SIZE,
            "offset": offset,
        }
        try:
            response = self._get(url, params)
        except requests.RequestException:
            logger.debug(
                "Semantic Scholar: failed to fetch %s for DOI %s (offset=%d).",
                endpoint,
                doi,
                offset,
            )
            return [], -1

        data = response.json()
        items = data.get("data") or []
        papers: list[Paper] = []
        for item in items:
            # The citations/references endpoints wrap each paper in
            # {"citedPaper": {...}} or {"citingPaper": {...}}.
            nested_key = "citedPaper" if endpoint == "references" else "citingPaper"
            paper_data = item.get(nested_key) or item
            paper = self._parse_paper(paper_data)
            if paper is not None:
                papers.append(paper)

        next_offset = data.get("next")
        if next_offset is None or len(items) < _CITATION_PAGE_SIZE:
            next_offset = -1

        return papers, next_offset

    def _fetch_all_citation_pages(
        self,
        doi: str,
        endpoint: str,
        *,
        progress_callback: Callable[[int], None] | None = None,
        max_papers: int | None = None,
    ) -> list[Paper]:
        """Paginate through all references or citations for a paper.

        Parameters
        ----------
        doi : str
            DOI of the paper.
        endpoint : str
            ``"references"`` or ``"citations"``.
        progress_callback : Callable[[int], None] | None
            Optional callback invoked after each page with the number
            of new papers fetched in that page.
        max_papers : int | None
            Maximum number of papers to return.  ``None`` (default) means no
            limit.  Pagination stops early once this many papers are collected,
            preventing runaway fetches for highly-cited works.

        Returns
        -------
        list[Paper]
            All papers from the paginated endpoint (up to *max_papers*).
        """
        all_papers: list[Paper] = []
        offset = 0
        page_num = 0

        while offset >= 0:
            page_num += 1
            page_papers, next_offset = self._fetch_citation_page(doi, endpoint, offset)
            all_papers.extend(page_papers)
            if progress_callback is not None:
                progress_callback(len(page_papers))
            offset = next_offset
            if max_papers is not None and len(all_papers) >= max_papers:
                all_papers = all_papers[:max_papers]
                break

        logger.info(
            "Semantic Scholar: DOI %s %s: %d pages, %d papers total.",
            doi,
            endpoint,
            page_num,
            len(all_papers),
        )

        return all_papers

    def fetch_cited_by(
        self,
        paper: Paper,
        progress_callback: Callable[[int], None] | None = None,
        max_papers: int | None = None,
    ) -> list[Paper]:
        """Return papers that cite the given paper (forward snowballing).

        Uses the Semantic Scholar ``/paper/{id}/citations`` endpoint.
        Pagination stops once *max_papers* results have been collected,
        preventing runaway fetches for highly-cited works.

        Parameters
        ----------
        paper : Paper
            The paper whose citing papers should be fetched.  Must have a DOI.
        progress_callback : Callable[[int], None] | None
            Optional callback for per-page progress reporting.
        max_papers : int | None
            Maximum number of citing papers to return.  ``None`` (default)
            means no limit.

        Returns
        -------
        list[Paper]
            Papers that cite *paper*, or empty list on failure.
        """
        if not paper.doi:
            return []

        logger.debug("Semantic Scholar: fetching citations for DOI %s.", paper.doi)
        return self._fetch_all_citation_pages(
            paper.doi,
            "citations",
            progress_callback=progress_callback,
            max_papers=max_papers,
        )

    # ------------------------------------------------------------------
    # Content-similarity lookup (used by Engine.similar())
    # ------------------------------------------------------------------

    def fetch_related(
        self,
        paper: Paper,
        max_papers: int | None = None,
    ) -> list[Paper]:
        """Return content-similar papers via the SPECTER-based Recommendations API.

        Uses ``GET /recommendations/v1/papers/forpaper/{paper_id}``, always
        passing ``from=all-cs`` explicitly.  Despite its name, ``all-cs`` is
        *not* restricted to Computer Science: it is the widest candidate pool
        the API offers (the whole corpus, across every field, with no date
        restriction).  The alternative pool, ``from=recent`` (the API's own
        default when ``from`` is omitted), only covers papers published in
        the last 60 days, which silently returns an empty list for anything
        older.  This was confirmed by live testing against both a 2017 CS
        paper and a 2012 biology paper: ``all-cs`` returned relevant results
        for both, spanning Medicine, Biology, and Computer Science alike.

        Parameters
        ----------
        paper : Paper
            Seed paper.  Must have a DOI: the endpoint is DOI-anchored, like
            every other lookup in this connector.
        max_papers : int | None
            Maps to the API's ``limit`` query parameter.  ``None`` (default)
            omits the parameter, so the API's own default (100 results) is
            used.  The API accepts values above 100 (up to roughly 500).

        Returns
        -------
        list[Paper]
            Related papers, or an empty list when *paper* has no DOI, the
            request fails, or no recommendations are returned.
        """
        if not paper.doi:
            return []

        url = f"{_RECOMMENDATIONS_URL}/DOI:{paper.doi}"
        params: dict[str, Any] = {"from": "all-cs", "fields": _PAPER_FIELDS}
        if max_papers is not None:
            params["limit"] = max_papers

        try:
            response = self._get(url, params)
            data = response.json()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                logger.debug("Semantic Scholar: recommendations: DOI %s not found.", paper.doi)
                return []
            logger.debug(
                "Semantic Scholar: HTTP error fetching recommendations for DOI %s: %s",
                paper.doi,
                exc,
            )
            return []
        except (requests.RequestException, ValueError):
            logger.debug("Semantic Scholar: recommendations failed for DOI %s.", paper.doi)
            return []

        items = data.get("recommendedPapers") or []
        papers: list[Paper] = []
        for item in items:
            related = self._parse_paper(item)
            if related is not None:
                papers.append(related)

        if max_papers is not None:
            papers = papers[:max_papers]
        return papers

    @staticmethod
    def _parse_ss_pub_date(item: dict[str, Any]) -> datetime.date | None:
        """Extract publication date from a Semantic Scholar item.

        Prefers ``publicationDate``; falls back to ``year`` when absent.

        Parameters
        ----------
        item : dict
            Semantic Scholar paper metadata dict.

        Returns
        -------
        datetime.date | None
            Parsed date or ``None``.
        """
        date_str = (item.get("publicationDate") or "").strip()
        if date_str:
            with contextlib.suppress(ValueError):
                return datetime.date.fromisoformat(date_str[:10])
        if item.get("year"):
            with contextlib.suppress(ValueError, TypeError):
                return datetime.date(int(item["year"]), 1, 1)
        return None

    @staticmethod
    def _parse_ss_doi(item: dict[str, Any]) -> str | None:
        """Extract DOI from Semantic Scholar external IDs.

        Prefers the explicit DOI field; falls back to deriving a canonical
        arXiv DOI (``10.48550/arXiv.<id>``) when only the ArXivId is available.

        Parameters
        ----------
        item : dict
            Semantic Scholar paper metadata dict.

        Returns
        -------
        str | None
            DOI string or ``None``.
        """
        external_ids = item.get("externalIds") or {}
        doi = (external_ids.get("DOI") or "").strip() or None
        if doi is None:
            arxiv_id = (external_ids.get("ArXivId") or "").strip()
            if arxiv_id:
                doi = f"10.48550/arXiv.{arxiv_id}"
        return doi

    @staticmethod
    def _parse_ss_identifiers(item: dict[str, Any]) -> list[PaperIdentifier]:
        """Extract every routable identifier without requiring a DOI."""
        identifiers: list[PaperIdentifier] = []
        paper_id = (item.get("paperId") or "").strip()
        if paper_id:
            identifiers.append(PaperIdentifier(IdentifierKind.SEMANTIC_SCHOLAR, paper_id))
        external_ids = item.get("externalIds") or {}
        if isinstance(external_ids, dict):
            arxiv_id = (external_ids.get("ArXivId") or "").strip()
            doi = (external_ids.get("DOI") or "").strip()
            if arxiv_id:
                identifiers.append(PaperIdentifier(IdentifierKind.ARXIV, arxiv_id))
            if doi:
                identifiers.append(PaperIdentifier(IdentifierKind.DOI, doi))
        return identifiers

    @classmethod
    def _parse_ss_target(cls, item: dict[str, Any]) -> CitationTarget | None:
        """Build a typed target from a nested citation/reference record."""
        identifiers = cls._parse_ss_identifiers(item)
        if not identifiers:
            return None
        title = (item.get("title") or "").strip() or None
        return CitationTarget(tuple(identifiers), title=title)

    @classmethod
    def _parse_ss_edges(cls, item: dict[str, Any], field: str) -> list[CitationEdge]:
        """Parse nested ``references`` or ``citations`` into S2 edges."""
        edges: list[CitationEdge] = []
        for raw_target in item.get(field) or []:
            if not isinstance(raw_target, dict):
                continue
            target = cls._parse_ss_target(raw_target)
            if target is not None:
                edges.append(CitationEdge(target=target, source=Database.SEMANTIC_SCHOLAR.value))
        return edges

    @staticmethod
    def _parse_ss_fos_subjects(
        item: dict[str, Any],
    ) -> tuple[set[str], set[str]]:
        """Extract fields of study and subjects from a Semantic Scholar item.

        ``fieldsOfStudy`` -> broad fields; ``s2FieldsOfStudy`` -> subjects
        (categories not already in fields).

        Parameters
        ----------
        item : dict
            Semantic Scholar paper metadata dict.

        Returns
        -------
        tuple[set[str], set[str]]
            ``(fields_of_study, subjects)`` sets.
        """
        fields_of_study: set[str] = set()
        for field in item.get("fieldsOfStudy") or []:
            if isinstance(field, str) and field.strip():
                fields_of_study.add(field.strip())
        subjects: set[str] = set()
        for entry in item.get("s2FieldsOfStudy") or []:
            if isinstance(entry, dict):
                cat = (entry.get("category") or "").strip()
                if cat and cat not in fields_of_study:
                    subjects.add(cat)
        return fields_of_study, subjects

    @staticmethod
    def _parse_ss_source(
        item: dict[str, Any],
    ) -> tuple[Source | None, str | None]:
        """Build a source and page range from a Semantic Scholar item.

        Parameters
        ----------
        item : dict
            Semantic Scholar paper metadata dict.

        Returns
        -------
        tuple[Source | None, str | None]
            ``(source, pages)`` pair.
        """
        journal = item.get("journal") or {}
        venue = (item.get("venue") or "").strip()
        pub_title = (journal.get("name") or venue or "").strip()

        source_type: SourceType | None = None
        # The Recommendations API (unlike the Graph API's search/citation
        # endpoints) can return "publicationVenue" as a bare venue-ID string
        # instead of the nested {"type": ..., ...} object; guard against
        # that shape here rather than assuming a dict.
        pub_venue = item.get("publicationVenue")
        pub_venue = pub_venue if isinstance(pub_venue, dict) else {}
        venue_type = (pub_venue.get("type") or "").strip().lower()
        if venue_type:
            source_type = _SS_VENUE_TYPE_MAP.get(venue_type)
        if source_type is None:
            for pt in item.get("publicationTypes") or []:
                if isinstance(pt, str) and pt in _SS_PUB_TYPE_MAP:
                    source_type = _SS_PUB_TYPE_MAP[pt]
                    break

        source: Source | None = (
            Source(title=pub_title, source_type=source_type) if pub_title else None
        )
        pages: str | None = (journal.get("pages") or "").strip() or None
        return source, pages

    @staticmethod
    def _parse_ss_paper_type(item: dict[str, Any]) -> PaperType | None:
        """Infer paper type from Semantic Scholar publication types list.

        Parameters
        ----------
        item : dict
            Semantic Scholar paper metadata dict.

        Returns
        -------
        PaperType | None
            Inferred paper type or ``None``.
        """
        for pt in item.get("publicationTypes") or []:
            if isinstance(pt, str) and pt in _SS_PAPER_TYPE_MAP:
                return _SS_PAPER_TYPE_MAP[pt]
        return None

    def _parse_paper(self, item: dict[str, Any]) -> Paper | None:
        """Parse a single Semantic Scholar paper record.

        Parameters
        ----------
        item : dict
            Paper metadata dictionary from Semantic Scholar API.

        Returns
        -------
        Paper | None
            Parsed paper or ``None`` when required fields are missing.
        """
        title = (item.get("title") or "").strip()
        if not title:
            return None

        abstract = (item.get("abstract") or "").strip()

        # Authors: affiliations are fetched in a batch request after the search.
        authors: list[Author] = []
        for author_entry in item.get("authors", []):
            name = (author_entry.get("name") or "").strip()
            if name:
                authors.append(Author(name=name))

        pub_date = self._parse_ss_pub_date(item)
        doi = self._parse_ss_doi(item)
        identifiers = self._parse_ss_identifiers(item)
        reference_edges = self._parse_ss_edges(item, "references")
        cited_by_edges = self._parse_ss_edges(item, "citations")
        citation_edge_totals = {
            self.name: {
                "references": int(item.get("referenceCount") or 0),
                "citations": int(item.get("citationCount") or 0),
            }
        }

        # URL
        url: str | None = (item.get("url") or "").strip() or None

        # PDF URL
        pdf_url: str | None = None
        open_access_pdf = item.get("openAccessPdf")
        if isinstance(open_access_pdf, dict):
            pdf_url = (open_access_pdf.get("url") or "").strip() or None

        # Citations
        citations: int | None = item.get("citationCount")

        fields_of_study, subjects = self._parse_ss_fos_subjects(item)
        source, pages = self._parse_ss_source(item)
        paper_type = self._parse_ss_paper_type(item)

        # Open access flag from Semantic Scholar.
        raw_is_oa = item.get("isOpenAccess")
        is_open_access: bool | None = bool(raw_is_oa) if raw_is_oa is not None else None

        try:
            paper = Paper(
                title=title,
                abstract=abstract,
                authors=authors,
                source=source,
                publication_date=pub_date,
                url=url,
                pdf_url=pdf_url,
                doi=doi,
                citations=citations,
                keywords=None,
                page_range=pages,
                found_in={self.name},
                paper_type=paper_type,
                fields_of_study=fields_of_study if fields_of_study else None,
                subjects=subjects if subjects else None,
                is_open_access=is_open_access,
                identifiers=identifiers,
                reference_edges=reference_edges,
                cited_by_edges=cited_by_edges,
                references=[
                    doi_value
                    for edge in reference_edges
                    if (doi_value := edge.target.get(IdentifierKind.DOI)) is not None
                ],
                cited_by=[
                    doi_value
                    for edge in cited_by_edges
                    if (doi_value := edge.target.get(IdentifierKind.DOI)) is not None
                ],
                citation_edge_totals=citation_edge_totals,
            )
        except ValueError:  # pragma: no cover: title already validated above
            return None

        return paper

    def _enrich_author_affiliations(
        self,
        author_id_to_authors: dict[str, list[Author]],
    ) -> None:
        """Batch-fetch affiliations from the Semantic Scholar Author API.

        Uses ``POST /author/batch?fields=affiliations`` to retrieve
        affiliation data for up to 1,000 authors per request, then
        updates the corresponding :class:`Author` objects in-place.

        Parameters
        ----------
        author_id_to_authors : dict[str, list[Author]]
            Mapping from Semantic Scholar author ID to :class:`Author`
            instances that share that ID across the retrieved papers.
        """
        all_ids = list(author_id_to_authors.keys())
        logger.info(
            "Fetching affiliations for %d unique authors from Semantic Scholar.",
            len(all_ids),
        )

        for start in range(0, len(all_ids), _AUTHOR_BATCH_SIZE):
            batch_ids = all_ids[start : start + _AUTHOR_BATCH_SIZE]
            try:
                response = self._post(
                    _AUTHOR_BATCH_URL,
                    json_body={"ids": batch_ids},
                    params={"fields": "affiliations"},
                )
            except requests.RequestException:
                logger.debug(
                    "Failed to fetch author affiliations batch (offset=%d).",
                    start,
                )
                continue

            results = response.json()
            for author_data in results:
                if not isinstance(author_data, dict):
                    continue
                author_id = author_data.get("authorId")
                affiliations = author_data.get("affiliations") or []
                if not author_id or not affiliations:
                    continue
                affiliation_str = "; ".join(affiliations)
                for author in author_id_to_authors.get(author_id, []):
                    if not author.affiliation:
                        author.affiliation = affiliation_str

    def _fetch_papers(
        self,
        query: Query,
        max_papers: int | None,
        progress_callback: Callable[[int, int | None], None] | None,
        since: datetime.date | None = None,
        until: datetime.date | None = None,
    ) -> CoveredPapers:
        """Fetch papers from Semantic Scholar bulk search with pagination.

        Bulk search reports ``total`` on every page, so coverage is decided by
        :meth:`~lit_search.core.coverage.SourceCoverage.from_pagination`.  A
        request that fails mid-walk, or records that could not be parsed, are
        reported as a shortfall rather than as a short but clean result.

        Parameters
        ----------
        query : Query
            Validated query object.
        max_papers : int | None
            Maximum papers to retrieve.
        progress_callback : Callable[[int, int | None], None] | None
            Progress callback.
        since : datetime.date | None
            Only return papers published on or after this date.
        until : datetime.date | None
            Only return papers published on or before this date.

        Returns
        -------
        CoveredPapers
            Retrieved papers and their coverage.
        """
        ss_params = self._query_builder.convert_query(query)
        date_param = SemanticScholarConnector._build_ss_date_param(since, until)
        if date_param is not None:
            ss_params["publicationDateOrYear"] = date_param

        papers: list[Paper] = []
        author_id_to_authors: dict[str, list[Author]] = {}
        token: str | None = None
        total: int | None = None
        processed = 0
        error_detail: str | None = None
        error_type: str | None = None
        truncated_early = False

        while True:
            remaining = (max_papers - len(papers)) if max_papers is not None else _PAGE_SIZE
            page_size = min(_PAGE_SIZE, remaining)

            params: dict = {
                **ss_params,
                "fields": _PAPER_FIELDS,
                "limit": page_size,
                "sort": "publicationDate:desc",
            }
            if token:
                params["token"] = token

            try:
                response = self._get(_BULK_SEARCH_URL, params)
            except requests.RequestException as exc:
                log_pagination_request_failure(
                    logger,
                    connector_label="Semantic Scholar",
                    processed=processed,
                    total=total,
                    position_label="token",
                    position=token,
                    exc=exc,
                )
                error_detail = f"request failed after {processed} records: {exc}"
                error_type = type(exc).__name__
                truncated_early = True
                break

            data = response.json()
            if total is None and isinstance(data.get("total"), int):
                total = data["total"]

            items = data.get("data") or []
            if not items:
                if total is not None and processed < total:
                    log_pagination_empty_before_total(
                        logger,
                        connector_label="Semantic Scholar",
                        processed=processed,
                        total=total,
                        position_label="token",
                        position=token,
                    )
                    truncated_early = True
                    error_detail = f"empty page after {processed} of {total} reported records"
                break

            for item in items:
                processed += 1
                paper = self._parse_paper(item)
                if paper is not None:
                    papers.append(paper)
                    SemanticScholarConnector._collect_author_ids(item, paper, author_id_to_authors)

            if progress_callback is not None:
                progress_callback(processed, total)

            if max_papers is not None and len(papers) >= max_papers:
                break

            token = data.get("token")
            if not token or len(items) < page_size:
                break

        if progress_callback is not None:
            progress_callback(processed, total)

        parsed = len(papers)
        result = papers[:max_papers] if max_papers is not None else papers

        if author_id_to_authors:
            self._enrich_author_affiliations(author_id_to_authors)

        return CoveredPapers(
            papers=tuple(result),
            coverage=self._build_coverage(
                retrieved=len(result),
                parsed=parsed,
                processed=processed,
                total=total,
                max_papers=max_papers,
                truncated_early=truncated_early,
                error_detail=error_detail,
                error_type=error_type,
            ),
        )

    def _build_coverage(
        self,
        *,
        retrieved: int,
        parsed: int,
        processed: int,
        total: int | None,
        max_papers: int | None,
        truncated_early: bool,
        error_detail: str | None,
        error_type: str | None,
    ) -> SourceCoverage:
        """Turn the pagination loop's counters into a coverage status.

        Parameters
        ----------
        retrieved : int
            Papers returned to the caller (post-cap).
        parsed : int
            Records parsed into papers (pre-cap).
        processed : int
            Records seen, parsed or not.
        total : int | None
            Total Semantic Scholar reported.
        max_papers : int | None
            Caller-imposed cap.
        truncated_early : bool
            Whether the walk stopped before exhausting the result set for a
            reason other than the cap.
        error_detail : str | None
            Description of what truncated the walk.
        error_type : str | None
            Exception class name for a request failure.

        Returns
        -------
        SourceCoverage
            The status describing this run.
        """
        notes: list[str] = []
        if error_detail:
            notes.append(error_detail)
        unparsed = processed - parsed
        if unparsed > 0:
            notes.append(f"{unparsed} of {processed} records could not be parsed")
        shortfall = "; ".join(notes) or None

        if retrieved == 0 and error_type is not None and error_detail is not None:
            return SourceCoverage.failed(self.name, error=error_detail, error_type=error_type)

        if truncated_early:
            return SourceCoverage.partial(
                self.name,
                retrieved,
                detail=shortfall or "pagination stopped before the result set was exhausted",
                reported_total=total,
                error=error_detail if error_type else None,
                error_type=error_type,
            )

        return SourceCoverage.from_pagination(
            self.name,
            retrieved=retrieved,
            reported_total=total,
            cap=max_papers,
            shortfall_detail=shortfall,
        )

    @staticmethod
    def _build_ss_date_param(
        since: datetime.date | None, until: datetime.date | None
    ) -> str | None:
        """Build the Semantic Scholar ``publicationDateOrYear`` filter value.

        Parameters
        ----------
        since : datetime.date | None
            Start date.
        until : datetime.date | None
            End date.

        Returns
        -------
        str | None
            ``"FROM:TO"`` ISO-date string or ``None`` when no filter is needed.
        """
        if not since and not until:
            return None
        from_part = since.isoformat() if since else ""
        to_part = until.isoformat() if until else ""
        return f"{from_part}:{to_part}"

    @staticmethod
    def _collect_author_ids(
        item: dict,
        paper: Paper,
        author_id_to_authors: dict[str, list[Author]],
    ) -> None:
        """Accumulate the author-ID -> Author mapping from a parsed paper.

        Parameters
        ----------
        item : dict
            Raw Semantic Scholar paper item.
        paper : Paper
            Parsed paper object (authors are indexed positionally).
        author_id_to_authors : dict[str, list[Author]]
            Mapping to accumulate into (mutated in place).

        Returns
        -------
        None
        """
        raw_authors = item.get("authors", [])
        for idx, author_entry in enumerate(raw_authors):
            author_id = author_entry.get("authorId")
            if author_id and idx < len(paper.authors):
                author_id_to_authors.setdefault(author_id, []).append(paper.authors[idx])
