"""DOAJ (Directory of Open Access Journals) searcher implementation.

Ported from ``paper_search_mcp/academic_platforms/doaj.py`` at pinned commit
``234678ab231074a7977320978ee0496dcdaddd1f``.  The port is **not** a mechanical
translation: the donor's ``search()`` wraps everything in
``except requests.exceptions.RequestException`` / ``except ValueError`` /
``except Exception``, all three falling through to ``return papers[:max_results]``,
and additionally does ``if 'error' in data: return papers``.  Under that contract
a dead endpoint, a malformed body and a genuinely unmatched query are the same
value.  See ``PORTING-GUIDE.md`` for the before-and-after.

Four behaviours of the live API drive the design here.  All were observed
directly on 2026-08-31 against ``https://doaj.org/api/search/articles`` (which
serves the v4 schema); none is a network error, and each one is a different way
a naive port reports something other than what happened:

- **The result set is hard-capped at 1,000 records, and the cap arrives as an
  HTTP 400.**  ``page=11&pageSize=100`` returns
  ``{"status": "bad_request", "error": "You cannot access results beyond 1000
  records via this API. ..."}``.  ``raise_for_status`` turns that into an
  ``HTTPError``, which the donor catches and converts into whatever it had
  accumulated -- or, on the first page, into zero.  Here the ceiling is
  computed before the request and reported as a shortfall.
- **``pageSize`` is silently clamped.**  ``pageSize=101`` returns HTTP 200 with
  ``pageSize: 100`` in the body, and ``pageSize=0`` returns ``pageSize: 10``.
  Asking for more than the API will give is not an error, so a connector that
  asks for 500 and receives 100 has no signal that anything happened.
- **An unrecognised field name returns HTTP 200 with ``total: 0``.**  That is
  handled in :mod:`lit_search.query.builders.doaj`, which emits only field
  paths checked against the live index.
- **Wildcards are refused with HTTP 400** (``"Query contains disallowed Lucene
  features"``), so they are rejected in ``validate_query`` before a request is
  spent.

**The page size must not change during a run.**  DOAJ pages by
``(page - 1) * pageSize``, so shrinking ``pageSize`` to fit a remaining cap
would silently re-read records already seen instead of continuing past them.
The size is fixed once, in :meth:`DOAJConnector._page_size_for_run`.

Rate limit: DOAJ publishes one.  See :data:`_MIN_REQUEST_INTERVAL`.
"""

from __future__ import annotations

import datetime
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import quote

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
from lit_search.query.builders.doaj import YEAR_FIELD, DOAJQueryBuilder
from lit_search.utils.normalization import normalize_doi, normalize_language, parse_date
from lit_search.utils.pagination_logging import log_pagination_request_failure

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://doaj.org/api/search/articles"

# Verified live 2026-08-31: pageSize=101 comes back clamped to 100, and the
# published schema states "The page size limit is 100".
_PAGE_SIZE = 100

# Verified live 2026-08-31: page*pageSize beyond this returns HTTP 400,
# "You cannot access results beyond 1000 records via this API."  DOAJ points
# callers at its public data dump and its OAI-PMH endpoints instead, so this is
# a real ceiling on what the search API can ever return, not a tuning knob.
_MAX_RECORDS = 1000

# Hard ceiling on pagination rounds.  Only reachable if the API keeps handing
# back non-empty pages past the record ceiling; it exists so that such a loop
# ends as an honest PARTIAL rather than running forever.
_MAX_PAGES = 1000

_MIN_REQUEST_INTERVAL = 0.5
"""Seconds between requests.

**This value is documented, not inferred.**  DOAJ's API FAQ states: *"There is
a rate limit of two requests per second on all API routes. 'Bursts' are
permitted, which means up to five requests per user are queued by the system
and are fulfilled in turn so long as they average out to two requests per
second overall."*  Read live from ``https://doaj.org/api/docs`` on 2026-08-31.

Two requests per second is 0.5s between requests, and the burst allowance is
deliberately not exploited: it is a queue, not extra capacity, so spending it
buys nothing over a paginated run and costs the margin that absorbs a slow
response.

Note that the donor connector's warning string claims *"public access with
rate limits (100 requests/hour)"*.  No such figure appears in DOAJ's current
documentation, and the published limit is 60x that.
"""

# Fragments of a DOAJ ``error`` body that identify the query itself, rather
# than the connector's request shape, as the thing DOAJ refused.  Only these
# become QUERY_UNSUPPORTED: a bug in this connector must not be filed under
# "the source could not express this query".
_QUERY_ERROR_MARKERS = (
    "disallowed lucene features",
    "error executing your query",
)

# DOAJ ``bibjson.link`` entries whose ``type`` marks the article's full text.
_FULLTEXT_LINK_TYPE = "fulltext"


@dataclass(slots=True)
class _Retrieval:
    """Mutable bookkeeping for one paginated DOAJ run.

    Exists so that the reasons a run can end stay separable all the way to
    :meth:`DOAJConnector._build_coverage`.  Collapsing them into "we have some
    papers" is what turns a truncated run into a confident answer.

    Deliberately a near-twin of ``europe_pmc._Retrieval``.  Hoisting a shared
    bookkeeper is a real option and is recorded as open item **O17** in the
    project specification; resolving it here, in a port, would be resolving an
    open item by guessing.

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
        ``total`` from the first page.
    total_drift : str | None
        Set when a later page reported a different ``total``.
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

    def note_total(self, total: object) -> None:
        """Record the reported total, or the fact that it moved.

        Parameters
        ----------
        total : object
            Raw ``total`` from a page.  Non-integers are ignored.
        """
        if not isinstance(total, int) or isinstance(total, bool):
            return
        if self.total is None:
            self.total = total
        elif total != self.total and self.total_drift is None:
            self.total_drift = (
                f"DOAJ reported {self.total} matching records on the first page and "
                f"{total} on a later one, so the index changed mid-retrieval and neither "
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


class DOAJBodyError(ConnectorError):
    """A DOAJ response that reports an upstream error rather than results.

    Covers both shapes: the HTTP 400 bodies DOAJ returns for a refused query or
    a request past the record ceiling, and any HTTP 200 body that carries an
    ``error`` key instead of a result set.

    Parameters
    ----------
    message : str
        Message assembled from the body's ``status`` / ``error``.
    is_query_error : bool
        Whether the body blames the query rather than the request shape.
        Decides whether the caller reports QUERY_UNSUPPORTED or FAILED.
    """

    def __init__(self, message: str, *, is_query_error: bool = False) -> None:
        """Store the message and the query-versus-request classification."""
        super().__init__(message)
        self.is_query_error = is_query_error


def _classify_body(payload: dict[str, Any]) -> DOAJBodyError:
    """Build the error for a DOAJ body that reports a problem.

    Parameters
    ----------
    payload : dict[str, Any]
        Decoded body carrying ``status`` and/or ``error``.

    Returns
    -------
    DOAJBodyError
        Error carrying the query-versus-request classification.
    """
    message = str(payload.get("error") or "no message supplied")
    status = str(payload.get("status") or "error")
    lowered = message.lower()
    return DOAJBodyError(
        f"DOAJ reported {status}: {message}",
        is_query_error=any(marker in lowered for marker in _QUERY_ERROR_MARKERS),
    )


class DOAJConnector(SearchConnectorBase):
    """Connector for the DOAJ article search API.

    Uses the public article search endpoint:
    https://doaj.org/api/docs

    DOAJ indexes articles from vetted open-access journals.  It requires no
    credential for search.
    """

    _credential_attrs: ClassVar[tuple[str, ...]] = ()
    """DOAJ issues API keys, but **not for search**.

    Verified against the published OpenAPI document on 2026-08-31: the
    ``/api/search/articles/{search_query}`` operation declares no ``security``
    requirement, and the only keyed operations are the article and application
    CRUD endpoints that publishers use to deposit content.  Sending a key on a
    search would neither raise the limit nor change the results.

    The quota is therefore enforced against this host's source IP and every
    DOAJ process on the machine shares one clock.  Declaring the empty tuple
    rather than accepting the ``("_api_key",)`` default states that the absence
    of a credential was checked, not overlooked.
    """

    def __init__(self, query_builder: DOAJQueryBuilder | None = None) -> None:
        """Create a DOAJ searcher.

        Parameters
        ----------
        query_builder : DOAJQueryBuilder | None
            Builder used to validate and convert queries.  When ``None`` a
            default :class:`DOAJQueryBuilder` is created.
        """
        super().__init__()
        self._query_builder: DOAJQueryBuilder = query_builder or DOAJQueryBuilder()

    @property
    def name(self) -> str:
        """Return the database identifier.

        Returns
        -------
        str
            Database name.
        """
        return Database.DOAJ.value

    @property
    def query_builder(self) -> QueryBuilder:
        """Return the DOAJ query builder.

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
            Interval in seconds.  See :data:`_MIN_REQUEST_INTERVAL`.
        """
        return _MIN_REQUEST_INTERVAL

    @property
    def rate_limit_service(self) -> str:
        """Return the quota authority these requests are paced against.

        Stated explicitly rather than inherited from :attr:`name` because
        DOAJ's published limit is *"two requests per second on all API
        routes"*.  A future DOAJ journals or bulk connector must return the
        same string and share this clock rather than doubling the rate.

        Returns
        -------
        str
            Quota-authority identifier.
        """
        return "doaj"

    # ------------------------------------------------------------------
    # Response handling
    # ------------------------------------------------------------------

    def _request_page(self, expression: str, params: dict[str, Any]) -> dict[str, Any]:
        """Fetch and validate one page of search results.

        DOAJ signals a refused query and the record ceiling with HTTP 400 and a
        JSON body naming the reason.  ``raise_for_status`` reduces both to an
        ``HTTPError`` whose message says only "400 Client Error", so the body is
        recovered here before that distinction is lost.

        Parameters
        ----------
        expression : str
            DOAJ query expression, not yet URL-encoded.
        params : dict[str, Any]
            Query-string parameters (``page`` and ``pageSize``).

        Returns
        -------
        dict[str, Any]
            Decoded payload, guaranteed to carry ``total`` and ``results``.

        Raises
        ------
        DOAJBodyError
            When DOAJ named a reason, at any status code.
        requests.RequestException
            When the request failed with no usable body.
        """
        # The query travels in the path, so every reserved character must be
        # percent-encoded: an unencoded '/' inside a DOI term would otherwise
        # become a path segment and 404.
        url = f"{_SEARCH_URL}/{quote(expression, safe='')}"
        try:
            response = self._get(url, params)
        except requests.HTTPError as exc:
            raise _body_error_from_http(exc) from exc
        return self._read_body(response)

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
            Decoded payload, guaranteed to carry ``total``.

        Raises
        ------
        DOAJBodyError
            When the body is not a JSON object, reports an error, or lacks the
            ``total`` field that every real search response carries.  All three
            would otherwise be read as "this database holds no matching paper".
        """
        try:
            payload = response.json()
        except ValueError as exc:
            raise DOAJBodyError(
                f"DOAJ returned a non-JSON body with status {response.status_code}: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise DOAJBodyError(
                f"DOAJ returned a JSON {type(payload).__name__} where an object was expected."
            )
        # ``error`` is the definitive marker: DOAJ's own failure bodies always
        # carry it.  ``status`` deliberately is not tested here -- a successful
        # response gaining that key one day must not make every search fail --
        # and the ``total`` check below catches any other unusable shape.
        if "error" in payload:
            raise _classify_body(payload)
        if "total" not in payload:
            raise DOAJBodyError(
                "DOAJ response carried no 'total', so it is not a search result. "
                "Treating it as an empty result set would report a failure as zero papers."
            )
        return payload

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_authors(bibjson: dict[str, Any]) -> list[Author]:
        """Extract authors from a DOAJ ``bibjson`` block.

        Parameters
        ----------
        bibjson : dict[str, Any]
            The record's ``bibjson`` object.

        Returns
        -------
        list[Author]
            Parsed authors, with affiliations when the record carries them.
        """
        entries = bibjson.get("author", [])
        if not isinstance(entries, list):
            return []
        authors: list[Author] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            if not name:
                continue
            affiliation = str(entry.get("affiliation") or "").strip() or None
            authors.append(Author(name=name, affiliation=affiliation))
        return authors

    @staticmethod
    def _parse_doi(bibjson: dict[str, Any]) -> str | None:
        """Extract the DOI from the record's identifier list.

        Parameters
        ----------
        bibjson : dict[str, Any]
            The record's ``bibjson`` object.

        Returns
        -------
        str | None
            Normalised DOI, or ``None``.
        """
        identifiers = bibjson.get("identifier", [])
        if not isinstance(identifiers, list):
            return None
        for identifier in identifiers:
            if isinstance(identifier, dict) and str(identifier.get("type", "")).lower() == "doi":
                return normalize_doi(str(identifier.get("id") or ""))
        return None

    @staticmethod
    def _parse_urls(
        item: dict[str, Any], bibjson: dict[str, Any], doi: str | None
    ) -> tuple[
        str | None,
        str | None,
    ]:
        """Extract the landing-page URL and PDF URL from a record.

        Parameters
        ----------
        item : dict[str, Any]
            One ``results`` entry.
        bibjson : dict[str, Any]
            The record's ``bibjson`` object.
        doi : str | None
            Already-parsed DOI, used as a fallback landing page.

        Returns
        -------
        tuple[str | None, str | None]
            ``(url, pdf_url)``.
        """
        url: str | None = None
        pdf_url: str | None = None
        links = bibjson.get("link", [])
        for link in links if isinstance(links, list) else []:
            if not isinstance(link, dict):
                continue
            if str(link.get("type") or "").lower() != _FULLTEXT_LINK_TYPE:
                continue
            href = str(link.get("url") or "").strip()
            if not href:
                continue
            is_pdf = str(link.get("content_type") or "").upper() == "PDF" or href.lower().endswith(
                ".pdf"
            )
            if is_pdf and pdf_url is None:
                pdf_url = href
            elif not is_pdf and url is None:
                url = href

        if url is None:
            article_id = str(item.get("id") or "").strip()
            if doi:
                url = f"https://doi.org/{doi}"
            elif article_id:
                url = f"https://doaj.org/article/{article_id}"
        return url, pdf_url

    @staticmethod
    def _parse_source(bibjson: dict[str, Any]) -> Source | None:
        """Build a :class:`Source` from a record's journal block.

        Parameters
        ----------
        bibjson : dict[str, Any]
            The record's ``bibjson`` object.

        Returns
        -------
        Source | None
            Journal source, or ``None`` when the record names no venue.
        """
        journal = bibjson.get("journal")
        if not isinstance(journal, dict):
            return None
        title = str(journal.get("title") or "").strip()
        if not title:
            return None
        issns = journal.get("issns")
        issn = None
        if isinstance(issns, list) and issns:
            issn = str(issns[0]).strip() or None
        elif isinstance(issns, str):
            issn = issns.strip() or None
        publisher = str(journal.get("publisher") or "").strip() or None
        return Source(
            title=title,
            issn=issn,
            publisher=publisher,
            source_type=SourceType.JOURNAL,
        )

    @staticmethod
    def _parse_keywords(bibjson: dict[str, Any]) -> set[str]:
        """Extract keywords and subject terms.

        Parameters
        ----------
        bibjson : dict[str, Any]
            The record's ``bibjson`` object.

        Returns
        -------
        set[str]
            Keyword strings; empty when the record carries none.
        """
        keywords: set[str] = set()
        raw = bibjson.get("keywords")
        if isinstance(raw, list):
            keywords.update(k.strip() for k in raw if isinstance(k, str) and k.strip())
        subjects = bibjson.get("subject")
        for subject in subjects if isinstance(subjects, list) else []:
            if isinstance(subject, dict):
                term = str(subject.get("term") or "").strip()
                if term:
                    keywords.add(term)
        return keywords

    @staticmethod
    def _parse_publication_date(bibjson: dict[str, Any]) -> datetime.date | None:
        """Resolve a publication date from ``year`` and ``month``.

        DOAJ stores ``year`` and ``month`` as separate strings and ``month`` may
        be a name (``"February"``) or a number.  There is no day, so a record
        resolves at best to the first of its month.

        Parameters
        ----------
        bibjson : dict[str, Any]
            The record's ``bibjson`` object.

        Returns
        -------
        datetime.date | None
            Parsed date, or ``None`` when the record carries no usable year.
        """
        year = str(bibjson.get("year") or "").strip()
        if not year.isdigit():
            return None
        month = str(bibjson.get("month") or "").strip()
        if month:
            parsed = parse_date(f"{year}-{month.zfill(2)}-01" if month.isdigit() else None)
            if parsed is not None:
                return parsed
            named = parse_date(f"{month} {year}")
            if named is not None:
                return named
        return parse_date(f"{year}-01-01")

    def _parse_item(self, item: dict[str, Any]) -> Paper | None:
        """Parse one DOAJ ``results`` entry into a :class:`Paper`.

        Parameters
        ----------
        item : dict[str, Any]
            One ``results`` entry.

        Returns
        -------
        Paper | None
            Parsed paper, or ``None`` when a required field is missing or the
            record fails model validation.  Callers **must** count the ``None``
            returns: an unparsed record is a coverage loss, not an absent paper.
        """
        if not isinstance(item, dict):
            return None
        bibjson = item.get("bibjson")
        if not isinstance(bibjson, dict):
            return None
        title = str(bibjson.get("title") or "").strip()
        if not title:
            return None

        doi = self._parse_doi(bibjson)
        url, pdf_url = self._parse_urls(item, bibjson, doi)
        journal = bibjson.get("journal") if isinstance(bibjson.get("journal"), dict) else {}
        languages = journal.get("language") if isinstance(journal, dict) else None
        language = languages[0] if isinstance(languages, list) and languages else None

        try:
            return Paper(
                title=title,
                abstract=str(bibjson.get("abstract") or "").strip(),
                authors=self._parse_authors(bibjson),
                source=self._parse_source(bibjson),
                publication_date=self._parse_publication_date(bibjson),
                url=url,
                pdf_url=pdf_url,
                doi=doi,
                keywords=self._parse_keywords(bibjson) or None,
                found_in={self.name},
                paper_type=PaperType.ARTICLE,
                language=normalize_language(str(language) if language else None),
                # Every article in DOAJ is in a vetted open-access journal;
                # that is the directory's entry criterion, not an inference.
                is_open_access=True,
            )
        except ValueError:
            logger.debug("DOAJ: record %s failed model validation.", item.get("id"))
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
        """Convert the query and append a year filter when bounds are given.

        A one-sided bound is emitted as ``>=`` or ``<=`` rather than as a range
        with an open end.  **DOAJ rejects ``*`` inside a range**: verified live
        2026-08-31, ``bibjson.year:[2020 TO *]`` returns HTTP 400 ``"Query
        contains disallowed Lucene features"``, which the connector would
        report as an outage.  Substituting a wide sentinel range instead is not
        equivalent either -- ``bibjson.title:"protein"`` matched 96,725 records
        while the same query with ``bibjson.year:[1000 TO 3000]`` matched
        96,527, so 198 records carry a year the range cannot hold and a
        sentinel would drop them without saying so.

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
            DOAJ query expression.
        """
        expression = self._query_builder.convert_query(query)
        date_filter = _year_filter(since, until)
        if date_filter is None:
            return expression
        return f"({expression}) AND {date_filter}" if expression else date_filter

    @staticmethod
    def _date_precision_note(
        since: datetime.date | None,
        until: datetime.date | None,
    ) -> str | None:
        """Return a shortfall note when the date bounds cannot be expressed.

        DOAJ indexes ``bibjson.year`` and ``bibjson.month`` but no full
        publication date, so a bound that falls inside a year is widened to the
        whole year.  Widening cannot lose a matching record, but it does mean
        the query DOAJ answered is not the query that was asked, and therefore
        that DOAJ's own ``total`` counts a different set.  Saying so is the
        point: a completeness claim checked against the wrong denominator is
        the failure this fork exists to prevent.

        A year-aligned window (1 January to 31 December) needs no note, because
        the widened query and the requested one are the same set.

        Parameters
        ----------
        since : datetime.date | None
            Lower publication-date bound.
        until : datetime.date | None
            Upper publication-date bound.

        Returns
        -------
        str | None
            Note describing the widening, or ``None`` when none happened.
        """
        aligned_low = since is None or (since.month, since.day) == (1, 1)
        aligned_high = until is None or (until.month, until.day) == (12, 31)
        if aligned_low and aligned_high:
            return None
        return (
            f"DOAJ indexes publication year only, so the requested window "
            f"[{since or '*'} to {until or '*'}] was widened to whole years and DOAJ's "
            f"record count describes that wider query, not the one asked"
        )

    @staticmethod
    def _page_size_for_run(max_papers: int | None) -> int:
        """Return the page size to use for every request in one run.

        Fixed for the whole run on purpose.  DOAJ addresses pages by
        ``(page - 1) * pageSize``, so shrinking the size partway through -- the
        obvious way to stop overshooting a cap -- moves the window backwards and
        re-reads records already seen instead of continuing past them.

        Parameters
        ----------
        max_papers : int | None
            Caller-imposed cap.

        Returns
        -------
        int
            Page size between 1 and :data:`_PAGE_SIZE`.
        """
        if max_papers is None:
            return _PAGE_SIZE
        return max(1, min(_PAGE_SIZE, max_papers))

    def _fetch_papers(
        self,
        query: Query,
        max_papers: int | None,
        progress_callback: Callable[[int, int | None], None] | None,
        since: datetime.date | None = None,
        until: datetime.date | None = None,
    ) -> CoveredPapers:
        """Fetch papers from DOAJ with page-number pagination and coverage.

        Every way this loop can end is tracked separately, because they are
        different facts about the result:

        - the caller's ``max_papers`` was reached -> ``CAPPED``;
        - a page request failed, or its body reported an error, with nothing
          retrieved -> raised, so the base class reports ``FAILED`` (or
          ``QUERY_UNSUPPORTED`` when DOAJ blamed the query);
        - the same failure *after* records were retrieved -> ``PARTIAL``
          carrying the error, because the records that did arrive are real;
        - DOAJ's 1,000-record API ceiling was reached with more records
          reported -> ``PARTIAL``, naming the ceiling;
        - ``total`` changed between pages -> ``PARTIAL``;
        - a sub-year date bound was widened -> ``PARTIAL``;
        - the pages ran out naturally -> ``from_pagination`` decides, and it
          returns ``COMPLETE`` only when as many records came back as DOAJ said
          existed.

        Parameters
        ----------
        query : Query
            Validated query object.
        max_papers : int | None
            Maximum papers to retrieve.  ``None`` means unlimited.
        progress_callback : Callable[[int, int | None], None] | None
            Progress callback called as ``callback(processed, total_or_none)``.
        since : datetime.date | None
            Only return papers published on or after this date.
        until : datetime.date | None
            Only return papers published on or before this date.

        Returns
        -------
        CoveredPapers
            Retrieved papers and the coverage status describing them.

        Raises
        ------
        UnsupportedQueryError
            When DOAJ rejected the query and nothing had been retrieved.
        ConnectorError
            When the first request failed or returned an unusable body.
        requests.RequestException
            When the first request failed at the transport level.
        """
        expression = self._build_query_string(query, since, until)
        page_size = self._page_size_for_run(max_papers)
        run = _Retrieval()
        page = 1

        while True:
            try:
                payload = self._request_page(expression, {"page": page, "pageSize": page_size})
            except (requests.RequestException, ConnectorError) as exc:
                self._record_page_failure(run, exc, page)
                break

            items = self._absorb_page(run, payload)
            if progress_callback is not None:
                progress_callback(run.processed, run.total)

            advanced = self._advance(run, items, page, page_size, max_papers)
            if advanced is None:
                break
            page = advanced

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
                date_note=self._date_precision_note(since, until),
            ),
        )

    @staticmethod
    def _record_page_failure(run: _Retrieval, exc: Exception, page: int) -> None:
        """Record a page failure, or re-raise when nothing was retrieved.

        The asymmetry is the point of the port.  With no records in hand there
        is no partial result to report and the failure is the whole story, so it
        is re-raised for :meth:`SearchConnectorBase.search` to name; with records
        in hand the run is a real shortfall and the records are kept.  Neither
        path can leave ``_fetch_papers`` as an empty success.

        Parameters
        ----------
        run : _Retrieval
            Bookkeeping for the current run.
        exc : Exception
            The failure raised while fetching or reading the page.
        page : int
            Page number the failed request carried, for the message.

        Raises
        ------
        UnsupportedQueryError
            When DOAJ blamed the query and nothing had been retrieved.
            Retrying such a query only spends rate-limit budget.
        Exception
            The original failure, when nothing had been retrieved.
        """
        if not run.papers:
            if isinstance(exc, DOAJBodyError) and exc.is_query_error:
                raise UnsupportedQueryError(str(exc)) from exc
            raise exc
        log_pagination_request_failure(
            logger,
            connector_label="DOAJ",
            processed=run.processed,
            total=run.total,
            position_label="page",
            position=page,
            exc=exc,
        )
        run.truncate(f"page request at page={page} failed: {exc}", type(exc).__name__)

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
            Raw ``results`` entries, including any that failed to parse -- the
            caller needs the count, because a record seen but not parsed is a
            coverage loss rather than an absent paper.
        """
        run.note_total(payload.get("total"))
        raw = payload.get("results", [])
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
        items: list[Any],
        page: int,
        page_size: int,
        max_papers: int | None,
    ) -> int | None:
        """Return the next page number, or ``None`` when the run should stop.

        Every stop other than the cap and a natural exhaustion is recorded as a
        truncation, so that it reaches the coverage status instead of the log.

        Parameters
        ----------
        run : _Retrieval
            Bookkeeping for the current run, updated in place.
        items : list[Any]
            Raw entries the page just absorbed carried.
        page : int
            Page number that produced this page.
        page_size : int
            Fixed page size for this run.
        max_papers : int | None
            Caller-imposed cap.

        Returns
        -------
        int | None
            Page number for the next request, or ``None`` to stop.
        """
        if max_papers is not None and len(run.papers) >= max_papers:
            return None
        if run.total is not None and run.processed >= run.total:
            return None

        if not items:
            # DOAJ still claims more records than arrived. Continuing would
            # spin; stopping without saying so would under-report.
            run.truncate(
                f"DOAJ returned an empty page at page {page} after "
                f"{run.processed} of "
                f"{run.total if run.total is not None else 'an unknown number of'} records"
            )
            return None

        if (page + 1) * page_size > _MAX_RECORDS:
            run.truncate(
                f"DOAJ's search API refuses to return results beyond record {_MAX_RECORDS}; "
                f"retrieval stopped at {run.processed} of "
                f"{run.total if run.total is not None else 'an unknown number of'} reported "
                f"records. DOAJ directs callers to its public data dump or its OAI-PMH "
                f"endpoint for the remainder"
            )
            return None

        if run.pages >= _MAX_PAGES:
            run.truncate(
                f"pagination stopped at the connector's {_MAX_PAGES}-page safety limit "
                f"after {run.processed} records"
            )
            return None

        return page + 1

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
        date_note: str | None,
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
            ``total`` from the first page.
        max_papers : int | None
            Caller-imposed cap.
        truncated_early : bool
            Whether the loop ended before the result set was exhausted, for a
            reason other than the cap.
        total_drift : str | None
            Description of a ``total`` change observed mid-retrieval.
        error_detail : str | None
            Description of the failure that truncated the run.
        error_type : str | None
            Exception class name for that failure.
        date_note : str | None
            Description of a date bound DOAJ could not express exactly.

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
        if date_note:
            notes.append(date_note)
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


def _year_filter(
    since: datetime.date | None,
    until: datetime.date | None,
) -> str | None:
    """Render the publication-date bounds as a DOAJ year constraint.

    Parameters
    ----------
    since : datetime.date | None
        Lower publication-date bound.
    until : datetime.date | None
        Upper publication-date bound.

    Returns
    -------
    str | None
        Lucene clause, or ``None`` when no bound was given.  A range is used
        only when both ends are known; see
        :meth:`DOAJConnector._build_query_string` for why an open end is not
        written as ``*`` or as a sentinel year.
    """
    if since is not None and until is not None:
        return f"{YEAR_FIELD}:[{since.year} TO {until.year}]"
    if since is not None:
        return f"{YEAR_FIELD}:>={since.year}"
    if until is not None:
        return f"{YEAR_FIELD}:<={until.year}"
    return None


def _body_error_from_http(exc: requests.HTTPError) -> Exception:
    """Recover DOAJ's own explanation from a failed HTTP response.

    DOAJ answers a refused query and a request past the record ceiling with
    HTTP 400 and a JSON body naming the reason.  ``raise_for_status`` discards
    that body, leaving a message that says only "400 Client Error", so a caller
    could not tell "this query cannot be run" from "the service is down".

    Parameters
    ----------
    exc : requests.HTTPError
        Error raised by ``raise_for_status``.

    Returns
    -------
    Exception
        A :class:`DOAJBodyError` when the response carried DOAJ's own
        explanation, otherwise *exc* unchanged so the transport failure is
        reported as itself.
    """
    response = exc.response
    if response is None:
        return exc
    try:
        payload = response.json()
    except ValueError:
        return exc
    if not isinstance(payload, dict) or "error" not in payload:
        return exc
    return _classify_body(payload)
