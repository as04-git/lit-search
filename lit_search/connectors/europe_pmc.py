"""Europe PMC searcher implementation.

Ported from ``paper_search_mcp/academic_platforms/europepmc.py`` at pinned
commit ``234678ab231074a7977320978ee0496dcdaddd1f``.  The port is **not** a
mechanical translation: the donor's error contract was rewritten, because in
that implementation ``search()`` wraps the whole request-and-parse block in
``except requests.RequestException`` and ``except Exception`` that both fall
through to ``return papers``.  Every upstream failure -- a dead endpoint, an
expired TLS certificate, a schema change -- therefore reaches the caller as a
successful search that found nothing.  See ``PORTING-GUIDE.md`` for the
before-and-after.

Three behaviours of the live API drive the design here.  All were observed
directly on 2026-08-30 against service version 6.9; none of them is a network
error, and all three would read as an empty result set under the donor
contract:

- **Errors arrive as HTTP 200.**  ``pageSize=1001`` returns
  ``{"errCode": 404, "errMsg": "Invalid page size provided..."}`` with a 200
  status and no ``resultList``.  ``raise_for_status()`` does not fire and
  ``data.get("resultList", {}).get("result", [])`` yields ``[]``.
- **A malformed cursor returns an empty envelope.**  ``cursorMark=GARBAGE!!``
  returns ``{"version": "6.9"}`` -- no ``hitCount``, no ``resultList``, HTTP
  200.  Absence of ``hitCount`` is therefore treated as a malformed response,
  not as zero results.
- **Cursor exhaustion has two forms.**  A result set that fits in one page
  omits ``nextCursorMark`` entirely; a multi-page set ends with an empty page
  whose ``nextCursorMark`` repeats the cursor that was sent.  A short page is
  *not* a reliable end-of-set signal -- an observed 9-hit query returned pages
  of 2, 2, 2, 2, 1, 0.

Rate limit: Europe PMC publishes none.  See :data:`_MIN_REQUEST_INTERVAL`.
"""

from __future__ import annotations

import datetime
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import requests

if TYPE_CHECKING:
    from lit_search.query.builder import QueryBuilder

from lit_search.connectors.search_base import SearchConnectorBase
from lit_search.core.author import Author
from lit_search.core.coverage import CoveredPapers, SourceCoverage
from lit_search.core.paper import Database, Paper, PaperType
from lit_search.core.query import Query
from lit_search.core.source import Source, SourceType
from lit_search.exceptions import ConnectorError, UnsupportedQueryError
from lit_search.query.builders.europe_pmc import EuropePMCQueryBuilder
from lit_search.utils.normalization import normalize_doi, normalize_language, parse_date
from lit_search.utils.pagination_logging import log_pagination_request_failure

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"
_SEARCH_URL = f"{_BASE_URL}/search"

# Verified live 2026-08-30: pageSize=1001 is refused with
# ``errMsg: "Invalid page size provided. Valid size is between 1 and 1000"``.
_MAX_PAGE_SIZE = 1000
_PAGE_SIZE = 100

_MIN_REQUEST_INTERVAL = 1.0
"""Seconds between requests.

**This value is inferred, not documented.**  Europe PMC publishes no rate
limit: neither ``europepmc.org/developers`` nor ``europepmc.org/RestfulWebService``
states a request frequency, and a live response carries no ``X-RateLimit-*``
or ``Retry-After`` headers (checked 2026-08-30).  The only stated restriction
is a prohibition on bulk download: *"It is not permissible to use any kind of
automated process to bulk download other content from Europe PMC."*

One second is the strictest rate any comparable service in this project's
source list publishes for an authenticated client (Semantic Scholar), so it is
a defensible floor for a service that publishes nothing.  The asymmetry is
deliberate: being too slow costs minutes over a few hundred papers, while
being too fast costs intermittent 429s that then have to be disambiguated from
genuine emptiness -- which is the failure this fork exists to prevent.

Revisit if Europe PMC ever publishes a number, or if the developers' forum
does.
"""

# Sentinel Europe PMC accepts for an open end of a date range.  Verified
# equivalent to an explicitly wide range on 2026-08-30: the same query with
# ``[1500-01-01 TO 3000-12-31]`` and with no date filter both returned 26,769
# hits, and ``[2020-01-01 TO *]`` matched ``[2020-01-01 TO 3000-12-31]``.
_OPEN_DATE_BOUND = "*"

# Hard ceiling on pagination rounds.  Only reachable if the API keeps handing
# out fresh cursors indefinitely; it exists so that such a loop ends as an
# honest PARTIAL rather than running forever.
_MAX_PAGES = 10_000

# MEDLINE publication type marking a record as retracted.  The retraction
# *notice* carries "Retraction of Publication" instead and must not be mapped,
# hence the exact-string match rather than a substring test.
_RETRACTED_PUB_TYPE = "Retracted Publication"

# Europe PMC ``source`` codes that denote a preprint server rather than a
# journal.  ``PPR`` is the preprint source in the live API.
_PREPRINT_SOURCES = frozenset({"PPR"})

# Fragments that identify an upstream ``errMsg`` as a complaint about the
# query itself rather than about the connector's request shape.  Only these
# become QUERY_UNSUPPORTED; everything else is a FAILED, because a connector
# bug must not be filed under "the source could not express this query".
_QUERY_ERROR_MARKERS = ("search criteria", "query")


@dataclass(slots=True)
class _Retrieval:
    """Mutable bookkeeping for one paginated Europe PMC run.

    Exists so that the reasons a run can end stay separable all the way to
    :meth:`EuropePMCConnector._build_coverage`.  Collapsing them into "we have
    some papers" is what turns a truncated run into a confident answer.

    Attributes
    ----------
    papers : list[Paper]
        Records parsed so far, in retrieval order.
    processed : int
        Records seen, whether or not they parsed.
    parsed : int
        Records that became a :class:`~lit_search.core.paper.Paper`.
    pages : int
        Pages consumed.
    total : int | None
        ``hitCount`` from the first page.
    total_drift : str | None
        Set when a later page reported a different ``hitCount``.
    truncated_early : bool
        Whether the run ended for a reason other than the cap or exhaustion.
    error_detail : str | None
        Description of the truncation.
    error_type : str | None
        Exception class name, when the truncation came from an exception.
    """

    papers: list[Paper] = field(default_factory=list)
    processed: int = 0
    parsed: int = 0
    pages: int = 0
    total: int | None = None
    total_drift: str | None = None
    truncated_early: bool = False
    error_detail: str | None = None
    error_type: str | None = None

    def note_total(self, hit_count: object) -> None:
        """Record the reported total, or the fact that it moved.

        Parameters
        ----------
        hit_count : object
            Raw ``hitCount`` from a page.  Non-integers are ignored.
        """
        if not isinstance(hit_count, int) or isinstance(hit_count, bool):
            return
        if self.total is None:
            self.total = hit_count
        elif hit_count != self.total and self.total_drift is None:
            self.total_drift = (
                f"Europe PMC reported {self.total} matching records on the first page and "
                f"{hit_count} on a later one, so the index changed mid-retrieval and neither "
                f"total can be checked against what was retrieved"
            )

    def truncate(self, detail: str, error_type: str | None = None) -> None:
        """Mark the run as having stopped short, keeping the first reason.

        Parameters
        ----------
        detail : str
            Why the run stopped.
        error_type : str | None
            Exception class name, when an exception caused it.
        """
        self.truncated_early = True
        if self.error_detail is None:
            self.error_detail = detail
            self.error_type = error_type


def _page_size_for(run: _Retrieval, max_papers: int | None) -> int:
    """Return the page size for the next request.

    Never returns zero (which the API rejects) and never exceeds
    :data:`_MAX_PAGE_SIZE` (which the API rejects with an HTTP 200 error body).

    Parameters
    ----------
    run : _Retrieval
        Bookkeeping for the current run.
    max_papers : int | None
        Caller-imposed cap.

    Returns
    -------
    int
        Page size between 1 and :data:`_MAX_PAGE_SIZE`.
    """
    remaining = (max_papers - len(run.papers)) if max_papers is not None else _PAGE_SIZE
    return max(1, min(_PAGE_SIZE, _MAX_PAGE_SIZE, remaining))


class EuropePMCBodyError(ConnectorError):
    """An HTTP 200 response whose JSON body reports an upstream error.

    Europe PMC signals several errors in the body with a 200 status, so this
    is raised where ``raise_for_status`` would be silent.

    Parameters
    ----------
    message : str
        Message assembled from the body's ``errCode`` / ``errMsg``.
    is_query_error : bool
        Whether the body blames the search criteria rather than the request
        shape.  Decides whether the caller reports QUERY_UNSUPPORTED or FAILED.
    """

    def __init__(self, message: str, *, is_query_error: bool = False) -> None:
        """Store the message and the query-versus-request classification."""
        super().__init__(message)
        self.is_query_error = is_query_error


class EuropePMCConnector(SearchConnectorBase):
    """Connector for Europe PMC.

    Uses the Articles RESTful API:
    https://europepmc.org/RestfulWebService

    Europe PMC aggregates MEDLINE/PubMed, PubMed Central, Agricola, preprints
    and patents.  It requires no credential.
    """

    _credential_attrs: ClassVar[tuple[str, ...]] = ()
    """Europe PMC issues no API key, so the quota is enforced against this
    host's source IP and every Europe PMC process on the machine shares one
    clock.  Declaring the empty tuple rather than accepting the ``("_api_key",)``
    default is deliberate: it states that the absence of a credential was
    checked, not overlooked.
    """

    def __init__(self, query_builder: EuropePMCQueryBuilder | None = None) -> None:
        """Create a Europe PMC searcher.

        Parameters
        ----------
        query_builder : EuropePMCQueryBuilder | None
            Builder used to validate and convert queries.  When ``None`` a
            default :class:`EuropePMCQueryBuilder` is created.
        """
        super().__init__()
        self._query_builder: EuropePMCQueryBuilder = query_builder or EuropePMCQueryBuilder()

    @property
    def name(self) -> str:
        """Return the database identifier.

        Returns
        -------
        str
            Database name.
        """
        return Database.EUROPE_PMC.value

    @property
    def query_builder(self) -> QueryBuilder:
        """Return the Europe PMC query builder.

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
            Interval in seconds.  See :data:`_MIN_REQUEST_INTERVAL` for why
            this number is inferred rather than documented.
        """
        return _MIN_REQUEST_INTERVAL

    @property
    def rate_limit_service(self) -> str:
        """Return the quota authority these requests are paced against.

        Stated explicitly rather than inherited from :attr:`name` so that any
        future connector against another Europe PMC endpoint -- annotations,
        full-text XML, grist -- returns the same string and shares this clock.
        They are one EBI-hosted service behind one unpublished allowance;
        giving each its own scope would multiply the request rate by the number
        of endpoints.

        Returns
        -------
        str
            Quota-authority identifier.
        """
        return "europepmc"

    # ------------------------------------------------------------------
    # Response handling
    # ------------------------------------------------------------------

    @staticmethod
    def _read_body(response: requests.Response) -> dict[str, Any]:
        """Decode and validate one search response body.

        Parameters
        ----------
        response : requests.Response
            Response returned by the search endpoint.

        Returns
        -------
        dict[str, Any]
            Decoded payload, guaranteed to carry ``hitCount``.

        Raises
        ------
        EuropePMCBodyError
            When the body is not JSON, reports an ``errCode``, or lacks the
            ``hitCount`` field that every real search response carries.  All
            three arrive with HTTP 200 and all three would otherwise be read
            as "this database holds no matching paper".
        """
        try:
            payload = response.json()
        except ValueError as exc:
            raise EuropePMCBodyError(
                f"Europe PMC returned a non-JSON body with status {response.status_code}: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise EuropePMCBodyError(
                f"Europe PMC returned a JSON {type(payload).__name__} where an object was expected."
            )

        if "errCode" in payload or "errMsg" in payload:
            message = str(payload.get("errMsg") or "no message supplied")
            lowered = message.lower()
            raise EuropePMCBodyError(
                f"Europe PMC reported errCode {payload.get('errCode')}: {message}",
                is_query_error=any(marker in lowered for marker in _QUERY_ERROR_MARKERS),
            )

        if "hitCount" not in payload:
            # Observed with a malformed cursorMark: HTTP 200, body {"version": "6.9"}.
            raise EuropePMCBodyError(
                "Europe PMC response carried no 'hitCount', so it is not a search "
                "result. Treating it as an empty result set would report a failure "
                "as zero papers."
            )
        return payload

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_authors(item: dict[str, Any]) -> list[Author]:
        """Extract authors from a Europe PMC ``core`` record.

        Parameters
        ----------
        item : dict[str, Any]
            One ``resultList.result`` entry.

        Returns
        -------
        list[Author]
            Parsed authors, with affiliations when the record carries them.
        """
        raw = item.get("authorList", {})
        entries = raw.get("author", []) if isinstance(raw, dict) else []
        if not isinstance(entries, list):
            entries = [entries]

        authors: list[Author] = []
        for entry in entries:
            if isinstance(entry, str):
                if entry.strip():
                    authors.append(Author(name=entry.strip()))
                continue
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("fullName") or "").strip()
            if not name:
                continue
            details = entry.get("authorAffiliationDetailsList", {})
            affiliations = details.get("authorAffiliation", []) if isinstance(details, dict) else []
            if isinstance(affiliations, dict):
                affiliations = [affiliations]
            parts = [
                str(a.get("affiliation") or "").strip()
                for a in affiliations
                if isinstance(a, dict) and str(a.get("affiliation") or "").strip()
            ]
            authors.append(
                Author(name=name, affiliation="; ".join(parts) if parts else None),
            )
        return authors

    @staticmethod
    def _parse_urls(item: dict[str, Any]) -> tuple[str | None, str | None]:
        """Extract the landing-page URL and PDF URL from a record.

        Parameters
        ----------
        item : dict[str, Any]
            One ``resultList.result`` entry.

        Returns
        -------
        tuple[str | None, str | None]
            ``(url, pdf_url)``.  Falls back to a DOI or Europe PMC article URL
            when the record supplies no HTML link.
        """
        raw = item.get("fullTextUrlList", {})
        entries = raw.get("fullTextUrl", []) if isinstance(raw, dict) else []
        if isinstance(entries, dict):
            entries = [entries]

        url: str | None = None
        pdf_url: str | None = None
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            style = str(entry.get("documentStyle") or "").lower()
            href = str(entry.get("url") or "").strip()
            if not href:
                continue
            if style == "html" and url is None:
                url = href
            elif style == "pdf" and pdf_url is None:
                pdf_url = href

        if url is None:
            doi = str(item.get("doi") or "").strip()
            pmcid = str(item.get("pmcid") or "").strip()
            pmid = str(item.get("pmid") or "").strip()
            if doi:
                url = f"https://doi.org/{doi}"
            elif pmcid:
                url = f"https://europepmc.org/article/PMC/{pmcid}"
            elif pmid:
                url = f"https://europepmc.org/article/MED/{pmid}"
        return url, pdf_url

    @staticmethod
    def _parse_source(item: dict[str, Any]) -> Source | None:
        """Build a :class:`Source` from a record's journal information.

        Parameters
        ----------
        item : dict[str, Any]
            One ``resultList.result`` entry.

        Returns
        -------
        Source | None
            Journal source, a preprint repository source for ``PPR`` records,
            or ``None`` when the record names no venue at all.
        """
        info = item.get("journalInfo", {})
        journal = info.get("journal", {}) if isinstance(info, dict) else {}
        title = str(journal.get("title") or "").strip() if isinstance(journal, dict) else ""
        code = str(item.get("source") or "").strip().upper()

        if not title:
            if code in _PREPRINT_SOURCES:
                return Source(title="Preprint", source_type=SourceType.REPOSITORY)
            return None

        issn = None
        if isinstance(journal, dict):
            issn = str(journal.get("issn") or journal.get("essn") or "").strip() or None
        source_type = SourceType.REPOSITORY if code in _PREPRINT_SOURCES else SourceType.JOURNAL
        return Source(title=title, issn=issn, source_type=source_type)

    @staticmethod
    def _parse_keywords(item: dict[str, Any]) -> set[str]:
        """Extract author keywords and MeSH descriptors.

        Parameters
        ----------
        item : dict[str, Any]
            One ``resultList.result`` entry.

        Returns
        -------
        set[str]
            Keyword strings; empty when the record carries none.
        """
        keywords: set[str] = set()
        raw = item.get("keywordList")
        entries = raw.get("keyword", []) if isinstance(raw, dict) else []
        if isinstance(entries, str):
            entries = [entries]
        keywords.update(e.strip() for e in entries if isinstance(e, str) and e.strip())

        mesh = item.get("meshHeadingList")
        headings = mesh.get("meshHeading", []) if isinstance(mesh, dict) else []
        for heading in headings if isinstance(headings, list) else []:
            if isinstance(heading, dict):
                name = str(heading.get("descriptorName") or "").strip()
                if name:
                    keywords.add(name)
        return keywords

    @staticmethod
    def _parse_publication_date(item: dict[str, Any]) -> datetime.date | None:
        """Resolve a publication date from the record's date fields.

        Prefers ``firstPublicationDate`` (a full ``YYYY-MM-DD``) and falls back
        to ``pubYear``, which is treated as 1 January of that year.

        Parameters
        ----------
        item : dict[str, Any]
            One ``resultList.result`` entry.

        Returns
        -------
        datetime.date | None
            Parsed date, or ``None`` when neither field is usable.
        """
        parsed = parse_date(str(item.get("firstPublicationDate") or "").strip() or None)
        if parsed is not None:
            return parsed
        year = str(item.get("pubYear") or "").strip()
        if year.isdigit():
            return parse_date(f"{year}-01-01")
        return None

    def _parse_item(self, item: dict[str, Any]) -> Paper | None:
        """Parse one Europe PMC ``core`` record into a :class:`Paper`.

        Parameters
        ----------
        item : dict[str, Any]
            One ``resultList.result`` entry.

        Returns
        -------
        Paper | None
            Parsed paper, or ``None`` when a required field is missing or the
            record fails model validation.  Callers **must** count the
            ``None`` returns: an unparsed record is a coverage loss, not an
            absent paper.
        """
        if not isinstance(item, dict):
            return None
        title = str(item.get("title") or "").strip()
        if not title:
            return None

        pub_types = item.get("pubTypeList", {})
        type_list = pub_types.get("pubType", []) if isinstance(pub_types, dict) else []
        if isinstance(type_list, str):
            type_list = [type_list]
        types = {str(t).strip() for t in type_list if isinstance(t, str)}

        source = self._parse_source(item)
        paper_type = (
            PaperType.UNPUBLISHED
            if source is not None and source.source_type is SourceType.REPOSITORY
            else PaperType.ARTICLE
        )

        citations = item.get("citedByCount")
        keywords = self._parse_keywords(item)
        url, pdf_url = self._parse_urls(item)

        try:
            return Paper(
                title=title,
                abstract=str(item.get("abstractText") or "").strip(),
                authors=self._parse_authors(item),
                source=source,
                publication_date=self._parse_publication_date(item),
                url=url,
                pdf_url=pdf_url,
                doi=normalize_doi(str(item.get("doi") or "")),
                citations=citations if isinstance(citations, int) else None,
                keywords=keywords or None,
                found_in={self.name},
                paper_type=paper_type,
                language=normalize_language(str(item.get("language") or "") or None),
                is_open_access=str(item.get("isOpenAccess") or "").upper() == "Y",
                # Only ever set True. Absence of the marker is not evidence the
                # paper was not retracted, so it stays None.
                is_retracted=True if _RETRACTED_PUB_TYPE in types else None,
            )
        except ValueError:
            logger.debug("Europe PMC: record %s failed model validation.", item.get("id"))
            return None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _build_query_string(
        self,
        query: Query,
        since: datetime.date | None,
        until: datetime.date | None,
    ) -> str:
        """Convert the query and append a date filter when bounds are given.

        Parameters
        ----------
        query : Query
            Validated query object.
        since : datetime.date | None
            Lower publication-date bound.
        until : datetime.date | None
            Upper publication-date bound.

        Returns
        -------
        str
            Europe PMC query expression.
        """
        expression = self._query_builder.convert_query(query)
        if since is None and until is None:
            return expression
        low = since.isoformat() if since else _OPEN_DATE_BOUND
        high = until.isoformat() if until else _OPEN_DATE_BOUND
        date_filter = f"FIRST_PDATE:[{low} TO {high}]"
        return f"({expression}) AND {date_filter}" if expression else date_filter

    def _fetch_papers(
        self,
        query: Query,
        max_papers: int | None,
        progress_callback: Callable[[int, int | None], None] | None,
        since: datetime.date | None = None,
        until: datetime.date | None = None,
    ) -> CoveredPapers:
        """Fetch papers from Europe PMC with cursor pagination and coverage.

        Every way this loop can end is tracked separately, because they are
        different facts about the result:

        - the caller's ``max_papers`` was reached -> ``CAPPED``;
        - a page request failed, or its body reported an error, with nothing
          retrieved -> raised, so the base class reports ``FAILED`` (or
          ``QUERY_UNSUPPORTED`` when Europe PMC blamed the criteria);
        - the same failure *after* records were retrieved -> ``PARTIAL``
          carrying the error, because the records that did arrive are real;
        - the cursor was exhausted before ``hitCount`` was reached ->
          ``PARTIAL``;
        - ``hitCount`` changed between pages (the index moved under the
          cursor) -> ``PARTIAL``, because neither total can be checked
          against what was retrieved;
        - the cursor ran out naturally -> ``from_pagination`` decides, and it
          returns ``COMPLETE`` only when as many records came back as Europe
          PMC said existed.

        Parameters
        ----------
        query : Query
            Validated query object.
        max_papers : int | None
            Maximum papers to retrieve.  ``None`` means unlimited.
        progress_callback : Callable[[int, int | None], None] | None
            Progress callback called as ``callback(processed, total_or_none)``.
        since : datetime.date | None
            Only return papers first published on or after this date.
        until : datetime.date | None
            Only return papers first published on or before this date.

        Returns
        -------
        CoveredPapers
            Retrieved papers and the coverage status describing them.

        Raises
        ------
        UnsupportedQueryError
            When Europe PMC rejected the search criteria and nothing had been
            retrieved.
        ConnectorError
            When the first request failed or returned an unusable body.
        requests.RequestException
            When the first request failed at the transport level.
        """
        expression = self._build_query_string(query, since, until)
        run = _Retrieval()
        cursor = "*"

        while True:
            params = {
                "query": expression,
                "format": "json",
                "resultType": "core",
                "pageSize": _page_size_for(run, max_papers),
                "cursorMark": cursor,
            }

            try:
                payload = self._read_body(self._get(_SEARCH_URL, params))
            except (requests.RequestException, ConnectorError) as exc:
                self._record_page_failure(run, exc, cursor)
                break

            items = self._absorb_page(run, payload)
            if progress_callback is not None:
                progress_callback(run.processed, run.total)

            advanced = self._advance(run, payload, items, cursor, max_papers)
            if advanced is None:
                break
            cursor = advanced

        if progress_callback is not None:
            progress_callback(run.processed, run.total)

        kept = run.papers[:max_papers] if max_papers is not None else run.papers
        return CoveredPapers(
            papers=tuple(kept),
            coverage=self._build_coverage(
                retrieved=len(kept),
                parsed=run.parsed,
                processed=run.processed,
                total=run.total,
                max_papers=max_papers,
                truncated_early=run.truncated_early,
                total_drift=run.total_drift,
                error_detail=run.error_detail,
                error_type=run.error_type,
            ),
        )

    @staticmethod
    def _record_page_failure(run: _Retrieval, exc: Exception, cursor: str) -> None:
        """Record a page failure, or re-raise when nothing was retrieved.

        The asymmetry is the point of the port.  With no records in hand there
        is no partial result to report and the failure is the whole story, so
        it is re-raised for :meth:`SearchConnectorBase.search` to name; with
        records in hand the run is a real shortfall and the records are kept.
        Neither path can leave ``_fetch_papers`` as an empty success.

        Parameters
        ----------
        run : _Retrieval
            Bookkeeping for the current run.
        exc : Exception
            The failure raised while fetching or reading the page.
        cursor : str
            Cursor the failed request carried, for the message.

        Raises
        ------
        UnsupportedQueryError
            When Europe PMC blamed the search criteria and nothing had been
            retrieved.  Retrying such a query only spends rate-limit budget.
        Exception
            The original failure, when nothing had been retrieved.
        """
        if not run.papers:
            if isinstance(exc, EuropePMCBodyError) and exc.is_query_error:
                raise UnsupportedQueryError(str(exc)) from exc
            raise exc
        log_pagination_request_failure(
            logger,
            connector_label="Europe PMC",
            processed=run.processed,
            total=run.total,
            position_label="cursorMark",
            position=cursor,
            exc=exc,
        )
        run.truncate(f"page request at cursorMark={cursor} failed: {exc}", type(exc).__name__)

    def _absorb_page(self, run: _Retrieval, payload: dict[str, Any]) -> list[Any]:
        """Fold one page into *run* and return the raw items it carried.

        Parameters
        ----------
        run : _Retrieval
            Bookkeeping for the current run, updated in place.
        payload : dict[str, Any]
            Validated response body.

        Returns
        -------
        list[Any]
            Raw ``resultList.result`` entries, including any that failed to
            parse -- the caller needs the count, because a record seen but not
            parsed is a coverage loss rather than an absent paper.
        """
        run.note_total(payload.get("hitCount"))
        result_list = payload.get("resultList") or {}
        raw = result_list.get("result", []) if isinstance(result_list, dict) else []
        items: list[Any] = raw if isinstance(raw, list) else []
        for item in items:
            paper = self._parse_item(item)
            if paper is not None:
                run.papers.append(paper)
                run.parsed += 1
        run.processed += len(items)
        run.pages += 1
        return items

    @staticmethod
    def _advance(
        run: _Retrieval,
        payload: dict[str, Any],
        items: list[Any],
        cursor: str,
        max_papers: int | None,
    ) -> str | None:
        """Return the next cursor, or ``None`` when the run should stop.

        Every stop other than the cap and a natural exhaustion is recorded as
        a truncation, so that it reaches the coverage status instead of the
        log.

        Parameters
        ----------
        run : _Retrieval
            Bookkeeping for the current run, updated in place.
        payload : dict[str, Any]
            Validated response body for the page just absorbed.
        items : list[Any]
            Raw entries that page carried.
        cursor : str
            Cursor that produced this page.
        max_papers : int | None
            Caller-imposed cap.

        Returns
        -------
        str | None
            Cursor for the next request, or ``None`` to stop.
        """
        raw_cursor = payload.get("nextCursorMark")
        next_cursor = raw_cursor if isinstance(raw_cursor, str) else None

        if max_papers is not None and len(run.papers) >= max_papers:
            return None

        # Two exhaustion signals, both verified live: a single-page result
        # omits nextCursorMark, a multi-page result ends with an empty page
        # whose nextCursorMark repeats what was sent.  A short page is not a
        # signal -- an observed 9-hit query paged 2, 2, 2, 2, 1, 0.
        if next_cursor is None or next_cursor == cursor:
            if run.total is not None and run.processed < run.total:
                run.truncate(
                    f"Europe PMC stopped returning results after {run.processed} of "
                    f"{run.total} reported records (cursorMark exhausted at page {run.pages})"
                )
            return None

        if not items:
            # The cursor advanced but the page was empty.  Continuing would
            # spin; stopping without saying so would under-report.
            run.truncate(
                f"Europe PMC returned an empty page at page {run.pages} while still "
                f"offering a new cursor, after {run.processed} of "
                f"{run.total if run.total is not None else 'an unknown number of'} records"
            )
            return None

        if run.pages >= _MAX_PAGES:
            run.truncate(
                f"pagination stopped at the connector's {_MAX_PAGES}-page safety limit "
                f"after {run.processed} records"
            )
            return None

        return next_cursor

    def _build_coverage(
        self,
        *,
        retrieved: int,
        parsed: int,
        processed: int,
        total: int | None,
        max_papers: int | None,
        truncated_early: bool,
        total_drift: str | None,
        error_detail: str | None,
        error_type: str | None,
    ) -> SourceCoverage:
        """Turn the pagination loop's own counters into a coverage status.

        Parameters
        ----------
        retrieved : int
            Papers actually returned to the caller, after the cap is applied.
        parsed : int
            Records successfully parsed into papers, before the cap.
        processed : int
            Records seen, parsed or not.
        total : int | None
            ``hitCount`` from the first page.
        max_papers : int | None
            Caller-imposed cap.
        truncated_early : bool
            Whether the loop ended before the result set was exhausted, for a
            reason other than the cap.
        total_drift : str | None
            Description of a ``hitCount`` change observed mid-retrieval.
        error_detail : str | None
            Description of the failure that truncated the run.
        error_type : str | None
            Exception class name for that failure.

        Returns
        -------
        SourceCoverage
            The status describing this run.
        """
        notes: list[str] = []
        if error_detail:
            notes.append(error_detail)
        if total_drift:
            notes.append(total_drift)
        unparsed = processed - parsed
        if unparsed > 0:
            notes.append(f"{unparsed} of {processed} records could not be parsed into papers")
        shortfall = "; ".join(notes) or None

        if truncated_early or total_drift:
            return SourceCoverage.partial(
                self.name,
                retrieved,
                detail=shortfall or "retrieval stopped before the result set was exhausted",
                reported_total=total,
                error=error_detail,
                error_type=error_type,
            )

        return SourceCoverage.from_pagination(
            self.name,
            retrieved=retrieved,
            reported_total=total,
            cap=max_papers,
            shortfall_detail=shortfall,
        )
