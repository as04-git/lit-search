"""HAL searcher implementation.

Ported from ``paper_search_mcp/academic_platforms/hal.py`` at pinned commit
``234678ab231074a7977320978ee0496dcdaddd1f``.  The port is **not** a mechanical
translation.  The donor's whole error contract is three lines::

    except Exception as exc:
        logger.error("HAL search failed: %s", exc)
        return []

which is broader than any of the donor connectors the porting guide quotes: a
bare ``except Exception`` around the request *and* the JSON decode, falling
through to an empty list.  A dead endpoint, an expired certificate, a changed
response shape and a genuinely empty search all reach the caller as the same
value, and the only trace is a log line.

The donor also issues exactly one request with ``rows=max_results`` and no
pagination, and never reads ``numFound``, so it cannot tell a complete result
from a truncated one either.

Live API behaviours this module is built around, all observed 2026-08-31:

- **Errors arrive as HTTP 200 with no ``response`` object.**  Both
  ``q=notafield_zz:catalysis`` (unknown field) and ``q=title_t:(catalysis``
  (unbalanced parenthesis) return status 200 with
  ``{"error":{"msg":"Error. See help : /docs"}}``.  ``raise_for_status()`` is
  silent and the donor's ``data.get("response", {}).get("docs", [])`` yields
  ``[]``.  :meth:`HALConnector._read_body` raises instead.
- **That error body carries no discriminating information.**  It is the same
  seven-word string for a malformed query, an unknown field and a missing
  ``sort`` parameter.  So this connector reports ``FAILED`` for it and never
  ``QUERY_UNSUPPORTED``: guessing which one it was would be inventing a fact.
  The query shapes that *can* be recognised locally are rejected before any
  request by :class:`~lit_search.query.builders.hal.HALQueryBuilder`.
- **``cursorMark`` requires ``sort``.**  Sending a cursor without one produces
  the same opaque error body, so :data:`_CURSOR_SORT` is always sent.
- **Cursor exhaustion repeats the cursor.**  A one-record query paged
  ``1, 0, 0`` with ``nextCursorMark`` unchanged from the second page onward.
  A short page is not a stop signal.
- **``numFoundExact`` exists and can be false.**  HAL runs Solr 8+, which
  reports ``numFound`` as a lower bound when it short-circuits the count.  A
  ``COMPLETE`` claimed against a lower bound is exactly the unfalsifiable
  completeness R10 describes, so a false ``numFoundExact`` forces ``PARTIAL``.
  Every query sampled on 2026-08-31 reported ``true``; the branch is defensive.

**Policy note, and it is not settled here.**  ``api.archives-ouvertes.fr/robots.txt``
publishes, for ``User-Agent: *``, ``Disallow: /search/`` -- the very path this
connector uses -- alongside ``Disallow: /oai/`` and ``Disallow: /sword/``.  HAL
documents ``/search/`` as a public programmatic API at
``api.archives-ouvertes.fr/docs/search`` and issues no credential for it, so the
directive most plausibly governs crawlers rather than API clients.  That
reading was not accepted for IACR elsewhere in this project, so the two should
be decided together rather than one by default.  This connector is registered
because the build plan lists HAL as a source to port; the judgement belongs to
the owner's policy gate, not to this module.

Rate limit: HAL publishes none.  See :data:`_MIN_REQUEST_INTERVAL`.
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
from lit_search.exceptions import ConnectorError
from lit_search.query.builders.hal import HALQueryBuilder
from lit_search.utils.normalization import normalize_doi, normalize_language, parse_date
from lit_search.utils.pagination_logging import log_pagination_request_failure

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://api.archives-ouvertes.fr/search/"

#: Solr requires a deterministic total ordering for cursor paging, and a
#: cursor sent without one returns HAL's opaque HTTP 200 error body.
_CURSOR_SORT = "docid asc"

#: Rows requested per page.  HAL's own guidance recommends staying at or below
#: 500-1000 rows per response; 100 keeps each request small enough that a
#: mid-run failure costs little.
_ROWS = 100

#: Safety limit on pagination rounds, so a cursor that never settles ends as an
#: honest ``PARTIAL`` rather than never ending.
_MAX_PAGES = 10_000

#: Solr document fields requested.  Each name was verified live on 2026-08-31:
#: an unknown name in ``fl`` is **not** an error, it simply yields documents
#: with no fields (``"docs":[{ }]``), which would present as every record
#: failing to parse.
_FIELDS = ",".join(
    [
        "halId_s",
        "title_s",
        "subTitle_s",
        "abstract_s",
        "authFullName_s",
        "doiId_s",
        "publicationDate_s",
        "producedDate_s",
        "submittedDate_s",
        "journalTitle_s",
        "keyword_s",
        "docType_s",
        "language_s",
        "openAccess_bool",
        "uri_s",
        "fileMain_s",
    ]
)

#: HAL ``docType_s`` codes that denote something other than a journal article.
#: ``UNDEFINED`` and any unlisted code fall back to :data:`SourceType.OTHER`.
_DOC_TYPE_SOURCES: dict[str, SourceType] = {
    "ART": SourceType.JOURNAL,
    "COMM": SourceType.CONFERENCE,
    "POSTER": SourceType.CONFERENCE,
    "PRESCONF": SourceType.CONFERENCE,
    "OUV": SourceType.BOOK,
    "COUV": SourceType.BOOK,
    "DOUV": SourceType.BOOK,
    "THESE": SourceType.OTHER,
    "HDR": SourceType.OTHER,
    "REPORT": SourceType.OTHER,
    "UNDEFINED": SourceType.REPOSITORY,
}

#: ``docType_s`` codes that are unpublished deposits rather than published
#: works, used only to set :attr:`Paper.paper_type`.
_UNPUBLISHED_DOC_TYPES = frozenset({"UNDEFINED", "OTHER", "PREPRINT"})

_MIN_REQUEST_INTERVAL = 1.0
"""Seconds between requests.

**This value is inferred, not documented.**  HAL publishes no rate limit:
``api.archives-ouvertes.fr/docs/search`` states none, and a live response on
2026-08-31 carried no ``X-RateLimit-*``, ``RateLimit-*`` or ``Retry-After``
header.  The only quantitative guidance anywhere in HAL's documentation is
about response size (stay at or below 500-1000 rows per request and use
``cursorMark`` beyond that), not about frequency.

One second matches the value this project already inferred for Europe PMC on
the same reasoning, and is the strictest rate any comparable service in the
source list publishes for an authenticated client.  Being too slow costs
minutes over a few hundred papers; being too fast costs intermittent 429s that
then have to be disambiguated from genuine emptiness, which is the failure this
fork exists to prevent.

Revisit if HAL ever publishes a number.
"""


class HALBodyError(ConnectorError):
    """An HTTP 200 response whose JSON body is not a HAL search result.

    Raised where ``raise_for_status`` is silent: HAL signals a malformed query,
    an unknown field and a missing ``sort`` all with status 200 and
    ``{"error": {...}}``, and each of those would otherwise be read as an empty
    result set.
    """


@dataclass(slots=True)
class _Retrieval:
    """Mutable bookkeeping for one paginated HAL run.

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
        ``numFound`` from the first page.
    total_is_exact : bool
        Whether every page reported ``numFoundExact`` as true.  Solr may report
        ``numFound`` as a lower bound, which cannot justify ``COMPLETE``.
    total_drift : str | None
        Set when a later page reported a different ``numFound``.
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
    total_is_exact: bool = True
    total_drift: str | None = None
    truncated_early: bool = False
    error_detail: str | None = None
    error_type: str | None = None

    def note_total(self, num_found: object, num_found_exact: object) -> None:
        """Record the reported total, whether it is exact, and any drift.

        Parameters
        ----------
        num_found : object
            Raw ``numFound`` from a page.  Non-integers are ignored.
        num_found_exact : object
            Raw ``numFoundExact``.  Only an explicit ``False`` downgrades the
            total; a missing key means the Solr version does not report it,
            which is not evidence either way.
        """
        if num_found_exact is False:
            self.total_is_exact = False
        if not isinstance(num_found, int) or isinstance(num_found, bool):
            return
        if self.total is None:
            self.total = num_found
        elif num_found != self.total and self.total_drift is None:
            self.total_drift = (
                f"HAL reported {self.total} matching records on the first page and "
                f"{num_found} on a later one, so the index changed mid-retrieval and "
                f"neither total can be checked against what was retrieved"
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


def _rows_for(run: _Retrieval, max_papers: int | None) -> int:
    """Return the page size for the next request.

    Parameters
    ----------
    run : _Retrieval
        Bookkeeping for the current run.
    max_papers : int | None
        Caller-imposed cap.

    Returns
    -------
    int
        Row count between 1 and :data:`_ROWS`.
    """
    remaining = (max_papers - len(run.papers)) if max_papers is not None else _ROWS
    return max(1, min(_ROWS, remaining))


def _first(value: Any) -> str:
    """Return the first entry of a Solr multi-valued field as a string.

    HAL returns most text fields as lists even when they hold one value.

    Parameters
    ----------
    value : Any
        Raw field value.

    Returns
    -------
    str
        Stripped first entry, or an empty string.
    """
    if isinstance(value, list):
        value = value[0] if value else ""
    if value is None:
        return ""
    return str(value).strip()


class HALConnector(SearchConnectorBase):
    """Connector for HAL, France's national open archive.

    Uses the public Solr search API:
    https://api.archives-ouvertes.fr/docs/search

    HAL aggregates deposits across every discipline and issues no credential.
    """

    _credential_attrs: ClassVar[tuple[str, ...]] = ()
    """HAL issues no API key for the search endpoint, so the quota is enforced
    against this host's source IP and every HAL process on the machine shares
    one clock.  Declaring the empty tuple rather than accepting the
    ``("_api_key",)`` default states that the absence of a credential was
    checked, not overlooked.
    """

    def __init__(self, query_builder: HALQueryBuilder | None = None) -> None:
        """Create a HAL searcher.

        Parameters
        ----------
        query_builder : HALQueryBuilder | None
            Builder used to validate and convert queries.  When ``None`` a
            default :class:`HALQueryBuilder` is created.
        """
        super().__init__()
        self._query_builder: HALQueryBuilder = query_builder or HALQueryBuilder()

    @property
    def name(self) -> str:
        """Return the database identifier.

        Returns
        -------
        str
            Database name.
        """
        return Database.HAL.value

    @property
    def query_builder(self) -> QueryBuilder:
        """Return the HAL query builder.

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

        Stated explicitly rather than inherited from :attr:`name` so that a
        future connector against another ``api.archives-ouvertes.fr`` endpoint
        -- the reference API, OAI-PMH, SWORD -- returns the same string and
        shares this clock instead of doubling the rate against one host.

        Returns
        -------
        str
            Quota-authority identifier.
        """
        return "archives-ouvertes"

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
            Decoded payload, guaranteed to carry a ``response`` object with a
            ``numFound``.

        Raises
        ------
        HALBodyError
            When the body is not JSON, reports an ``error``, or lacks the
            ``response.numFound`` that every real search result carries.  All
            three arrive with HTTP 200 and all three would otherwise be read as
            "this database holds no matching paper".
        """
        try:
            payload = response.json()
        except ValueError as exc:
            raise HALBodyError(
                f"HAL returned a non-JSON body with status {response.status_code}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise HALBodyError(
                f"HAL returned a JSON {type(payload).__name__} where an object was expected."
            )
        if "error" in payload:
            raw = payload.get("error")
            message = _first(raw.get("msg")) if isinstance(raw, dict) else str(raw)
            raise HALBodyError(
                f"HAL reported an error with HTTP 200: {message or 'no message supplied'}. "
                "HAL uses one message for a malformed query, an unknown field and a "
                "missing sort parameter, so which of those it was cannot be told from "
                "the response."
            )
        block = payload.get("response")
        if not isinstance(block, dict) or "numFound" not in block:
            raise HALBodyError(
                "HAL response carried no 'response.numFound', so it is not a search "
                "result. Treating it as an empty result set would report a failure as "
                "zero papers."
            )
        return payload

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_authors(doc: dict[str, Any]) -> list[Author]:
        """Extract authors from a HAL Solr document.

        Parameters
        ----------
        doc : dict[str, Any]
            One ``response.docs`` entry.

        Returns
        -------
        list[Author]
            Parsed authors; empty when the record names none.
        """
        raw = doc.get("authFullName_s") or []
        entries = raw if isinstance(raw, list) else [raw]
        return [
            Author(name=str(entry).strip())
            for entry in entries
            if isinstance(entry, str) and entry.strip()
        ]

    @staticmethod
    def _parse_source(doc: dict[str, Any]) -> Source | None:
        """Build a :class:`Source` from a HAL document's venue information.

        Parameters
        ----------
        doc : dict[str, Any]
            One ``response.docs`` entry.

        Returns
        -------
        Source | None
            Journal or venue source, a repository source for a deposit with no
            named venue, or ``None`` when the document type is unknown too.
        """
        doc_type = _first(doc.get("docType_s")).upper()
        title = _first(doc.get("journalTitle_s"))
        if title:
            return Source(
                title=title,
                source_type=_DOC_TYPE_SOURCES.get(doc_type, SourceType.OTHER),
            )
        if not doc_type:
            return None
        return Source(title="HAL", source_type=SourceType.REPOSITORY)

    @staticmethod
    def _parse_publication_date(doc: dict[str, Any]) -> datetime.date | None:
        """Resolve a publication date from the document's date fields.

        Prefers ``publicationDate_s`` and falls back to ``producedDate_s`` and
        then ``submittedDate_s``.  A bare year is read as 1 January.

        Parameters
        ----------
        doc : dict[str, Any]
            One ``response.docs`` entry.

        Returns
        -------
        datetime.date | None
            Parsed date, or ``None`` when no field is usable.
        """
        for key in ("publicationDate_s", "producedDate_s", "submittedDate_s"):
            raw = _first(doc.get(key))
            if not raw:
                continue
            candidate = f"{raw}-01-01" if raw.isdigit() and len(raw) == 4 else raw[:10]
            parsed = parse_date(candidate)
            if parsed is not None:
                return parsed
        return None

    def _parse_doc(self, doc: Any) -> Paper | None:
        """Parse one HAL Solr document into a :class:`Paper`.

        Parameters
        ----------
        doc : Any
            One ``response.docs`` entry.

        Returns
        -------
        Paper | None
            Parsed paper, or ``None`` when a required field is missing or the
            record fails model validation.  Callers **must** count the ``None``
            returns: an unparsed record is a coverage loss, not an absent
            paper.
        """
        if not isinstance(doc, dict):
            return None
        title = _first(doc.get("title_s"))
        if not title:
            return None
        subtitle = _first(doc.get("subTitle_s"))
        if subtitle:
            title = f"{title}: {subtitle}"

        hal_id = _first(doc.get("halId_s"))
        keywords = {
            str(entry).strip()
            for entry in (doc.get("keyword_s") or [])
            if isinstance(entry, str) and entry.strip()
        }
        doc_type = _first(doc.get("docType_s")).upper()
        open_access = doc.get("openAccess_bool")

        try:
            return Paper(
                title=title,
                abstract=_first(doc.get("abstract_s")),
                authors=self._parse_authors(doc),
                source=self._parse_source(doc),
                publication_date=self._parse_publication_date(doc),
                url=_first(doc.get("uri_s"))
                or (f"https://hal.science/{hal_id}" if hal_id else None),
                pdf_url=_first(doc.get("fileMain_s")) or None,
                doi=normalize_doi(_first(doc.get("doiId_s"))),
                keywords=keywords or None,
                found_in={self.name},
                paper_type=(
                    PaperType.UNPUBLISHED
                    if doc_type in _UNPUBLISHED_DOC_TYPES
                    else PaperType.ARTICLE
                ),
                language=normalize_language(_first(doc.get("language_s")) or None),
                is_open_access=open_access if isinstance(open_access, bool) else None,
            )
        except ValueError:
            logger.debug("HAL: record %s failed model validation.", hal_id)
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

        Uses ``publicationDate_tdate``, HAL's Solr date field, verified live on
        2026-08-31 to accept an ISO instant range.  A range on the string field
        ``publicationDate_s`` would compare lexically, which silently mis-sorts
        records recorded as a bare year.

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
            HAL Solr query expression.
        """
        expression = self._query_builder.convert_query(query)
        if since is None and until is None:
            return expression
        low = f"{since.isoformat()}T00:00:00Z" if since else "*"
        high = f"{until.isoformat()}T23:59:59Z" if until else "*"
        date_filter = f"publicationDate_tdate:[{low} TO {high}]"
        return f"({expression}) AND {date_filter}" if expression else date_filter

    def _fetch_papers(
        self,
        query: Query,
        max_papers: int | None,
        progress_callback: Callable[[int, int | None], None] | None,
        since: datetime.date | None = None,
        until: datetime.date | None = None,
    ) -> CoveredPapers:
        """Fetch papers from HAL with cursor pagination and coverage.

        Every way this loop can end is tracked separately, because they are
        different facts about the result: the cap, a page failure with nothing
        retrieved (raised, so the base class reports ``FAILED``), the same
        failure after records were retrieved (``PARTIAL``, keeping them), a
        cursor exhausted before ``numFound`` was reached (``PARTIAL``), a
        ``numFound`` that moved between pages (``PARTIAL``), a ``numFound`` that
        Solr flagged as inexact (``PARTIAL``), and natural exhaustion, which is
        the only route to ``COMPLETE``.

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
        HALBodyError
            When the first request returned an unusable body.
        requests.RequestException
            When the first request failed at the transport level.
        """
        expression = self._build_query_string(query, since, until)
        run = _Retrieval()
        cursor = "*"

        while True:
            params = {
                "q": expression,
                "fl": _FIELDS,
                "wt": "json",
                "sort": _CURSOR_SORT,
                "rows": _rows_for(run, max_papers),
                "cursorMark": cursor,
            }

            try:
                payload = self._read_body(self._get(_SEARCH_URL, params))
            except (requests.RequestException, ConnectorError) as exc:
                self._record_page_failure(run, exc, cursor)
                break

            docs = self._absorb_page(run, payload)
            if progress_callback is not None:
                progress_callback(run.processed, run.total)

            advanced = self._advance(run, payload, docs, cursor, max_papers)
            if advanced is None:
                break
            cursor = advanced

        if progress_callback is not None:
            progress_callback(run.processed, run.total)

        kept = run.papers[:max_papers] if max_papers is not None else run.papers
        return CoveredPapers(
            papers=tuple(kept),
            coverage=self._build_coverage(retrieved=len(kept), run=run, max_papers=max_papers),
        )

    @staticmethod
    def _record_page_failure(run: _Retrieval, exc: Exception, cursor: str) -> None:
        """Record a page failure, or re-raise when nothing was retrieved.

        With no records in hand the failure is the whole story and is re-raised
        for :meth:`SearchConnectorBase.search` to name; with records in hand the
        run is a real shortfall and the records are kept.  Neither path can
        leave ``_fetch_papers`` as an empty success, which is the single change
        that distinguishes this connector from the donor.

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
        Exception
            The original failure, when nothing had been retrieved.
        """
        if not run.papers:
            raise exc
        log_pagination_request_failure(
            logger,
            connector_label="HAL",
            processed=run.processed,
            total=run.total,
            position_label="cursorMark",
            position=cursor,
            exc=exc,
        )
        run.truncate(f"page request at cursorMark={cursor} failed: {exc}", type(exc).__name__)

    def _absorb_page(self, run: _Retrieval, payload: dict[str, Any]) -> list[Any]:
        """Fold one page into *run* and return the raw documents it carried.

        Parameters
        ----------
        run : _Retrieval
            Bookkeeping for the current run, updated in place.
        payload : dict[str, Any]
            Validated response body.

        Returns
        -------
        list[Any]
            Raw ``response.docs`` entries, including any that failed to parse
            -- the caller needs the count, because a record seen but not parsed
            is a coverage loss rather than an absent paper.
        """
        block = payload.get("response") or {}
        run.note_total(block.get("numFound"), block.get("numFoundExact"))
        raw = block.get("docs", [])
        docs: list[Any] = raw if isinstance(raw, list) else []
        for doc in docs:
            paper = self._parse_doc(doc)
            if paper is not None:
                run.papers.append(paper)
                run.parsed += 1
        run.processed += len(docs)
        run.pages += 1
        return docs

    @staticmethod
    def _advance(
        run: _Retrieval,
        payload: dict[str, Any],
        docs: list[Any],
        cursor: str,
        max_papers: int | None,
    ) -> str | None:
        """Return the next cursor, or ``None`` when the run should stop.

        Every stop other than the cap and a natural exhaustion is recorded as a
        truncation, so that it reaches the coverage status instead of the log.

        Parameters
        ----------
        run : _Retrieval
            Bookkeeping for the current run, updated in place.
        payload : dict[str, Any]
            Validated response body for the page just absorbed.
        docs : list[Any]
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
        if max_papers is not None and len(run.papers) >= max_papers:
            return None

        raw_cursor = payload.get("nextCursorMark")
        next_cursor = raw_cursor if isinstance(raw_cursor, str) and raw_cursor else None

        # Solr's documented exhaustion signal, verified live: the cursor comes
        # back unchanged.  A short page is deliberately not treated as the end.
        if next_cursor is None or next_cursor == cursor:
            if run.total is not None and run.processed < run.total:
                run.truncate(
                    f"HAL stopped returning results after {run.processed} of "
                    f"{run.total} reported records (cursorMark exhausted at page {run.pages})"
                )
            return None

        if not docs:
            # The cursor advanced but the page was empty.  Continuing would
            # spin; stopping without saying so would under-report.
            run.truncate(
                f"HAL returned an empty page at page {run.pages} while still offering a "
                f"new cursor, after {run.processed} of "
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
        run: _Retrieval,
        max_papers: int | None,
    ) -> SourceCoverage:
        """Turn the pagination loop's own counters into a coverage status.

        Parameters
        ----------
        retrieved : int
            Papers actually returned to the caller, after the cap is applied.
        run : _Retrieval
            Bookkeeping for the finished run.
        max_papers : int | None
            Caller-imposed cap.

        Returns
        -------
        SourceCoverage
            The status describing this run.
        """
        notes: list[str] = []
        if run.error_detail:
            notes.append(run.error_detail)
        if run.total_drift:
            notes.append(run.total_drift)
        if not run.total_is_exact:
            notes.append(
                "HAL reported numFoundExact=false, so its numFound is a lower bound "
                "rather than a count and cannot establish that everything was retrieved"
            )
        unparsed = run.processed - run.parsed
        if unparsed > 0:
            notes.append(f"{unparsed} of {run.processed} records could not be parsed into papers")
        shortfall = "; ".join(notes) or None

        if run.truncated_early or run.total_drift or not run.total_is_exact:
            return SourceCoverage.partial(
                self.name,
                retrieved,
                detail=shortfall or "retrieval stopped before the result set was exhausted",
                reported_total=run.total,
                error=run.error_detail,
                error_type=run.error_type,
            )

        return SourceCoverage.from_pagination(
            self.name,
            retrieved=retrieved,
            reported_total=run.total,
            cap=max_papers,
            shortfall_detail=shortfall,
        )
