"""Zotero connector, driven by the desktop client's local API.

**This is new code, not a port.**  ``D8`` and ``D19`` make Zotero three things
at once: a search source like any other, the set a search is subtracted from
("what do I already have"), and the export sink.  This module implements the
first two.  The third routes through Zotero so that Better BibTeX stays the
single authority for citation keys (``D7``), and is not part of this file.

The failure this module is written against
------------------------------------------

An empty Zotero result does not mean "your library holds nothing matching".  It
means the desktop app is closed, or the local API is switched off, or the
request went to the wrong library prefix.  And because Zotero is the
**subtrahend** of the exclude-my-library operation, an empty result is worse
than a missing source: every paper already in the library is then reported as
new, confidently, with no source having reported a failure.  So there is no
code path here that turns a failure into zero records.

Three outcomes, deliberately distinguished:

===============================================  ==============================
Situation                                        Status
===============================================  ==============================
"Allow other applications..." is off (HTTP 403)  ``SKIPPED_NO_CREDENTIAL``
Zotero is not running (connection refused)       ``FAILED``
Any other transport or body failure              ``FAILED``
===============================================  ==============================

The split follows the enum's own definitions rather than convenience.  The
preference being off is determinate and fixed by configuration, which is what
``SKIPPED_NO_CREDENTIAL`` means -- the source was never contacted, and it is
modelled as a credential because it is a permission the user grants.  The app
being closed may be transient, which is what ``FAILED`` means, and ``FAILED``
also ranks worse, so the more uncertain outcome is the louder one.

What was verified, and what was not
-----------------------------------

Read from ``zotero.org/support/dev/web_api/v3/local_api`` and
``.../v3/basics`` on 2026-08-31:

- The local API is served at ``http://localhost:23119/api/`` and "Requests will
  return ``403 Forbidden`` if it is not enabled" (Settings -> Advanced ->
  "Allow other applications on this computer to communicate with Zotero").
- Reads need no authentication.  The docs add: "Do not forward the port or
  otherwise expose it externally."
- Only the logged-in user's data is reachable; pass ``0`` or the real numeric
  id, and "other IDs give 400".
- "Results are not paginated by default", and the local API "does not impose a
  default or maximum ``limit``".  ``limit`` and ``start`` still work, and this
  connector uses them so that a large library arrives in bounded pages and so
  that progress is reportable.
- Multi-object reads "include a custom HTTP header, ``Total-Results``, that
  provides the total number of results matched by the request".

**Not verified live.**  Zotero desktop was not running and not reachable from
this machine when the connector was written: ``127.0.0.1:23119`` and the WSL
gateway address both fail to connect.  So the request shape here follows the
published specification rather than an observed response.  Two specific
consequences are handled defensively rather than assumed away:
``Total-Results`` being absent is treated as "the source published no total",
which makes the result ``PARTIAL`` instead of letting arithmetic that happens
to balance stand in for completeness; and ``itemType=-attachment`` is the only
item-type filter sent, because it is the exact form the specification gives as
an example, while a multiple-negation form would be a guess.

Why a search is never ``COMPLETE``
----------------------------------

Zotero's ``q`` parameter is a substring quicksearch with no boolean operators
and no negation, so it cannot be asked lit-search's query -- see
:mod:`lit_search.query.builders.zotero` for the full argument.  The library is
enumerated and matched locally, and :mod:`lit_search.query.local_match` states
what that costs.  :meth:`ZoteroConnector.fetch_library`, which applies no
query, is the one path that *can* report ``COMPLETE``, and it is the path the
exclude-my-library operation needs.
"""

from __future__ import annotations

import datetime
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urlparse

import requests

if TYPE_CHECKING:
    from lit_search.query.builder import QueryBuilder

from lit_search.connectors.search_base import SearchConnectorBase
from lit_search.core.author import Author
from lit_search.core.coverage import CoveredPapers, SourceCoverage
from lit_search.core.paper import Database, Paper, PaperType
from lit_search.core.query import Query
from lit_search.core.retrieval_audit import auditing, verify_coverage_claim
from lit_search.core.source import Source, SourceType
from lit_search.exceptions import ConnectorError, MissingApiKeyError
from lit_search.query.builders.zotero import ZoteroQueryBuilder
from lit_search.query.local_match import Predicate, Record
from lit_search.utils.normalization import normalize_doi, normalize_language, parse_date

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:23119/api"
_DEFAULT_LIBRARY = "users/0"

_LOCAL_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})
"""Hostnames the connector will talk to.

A guard rather than a formality.  Pointed at ``api.zotero.org`` this connector
would send an unauthenticated request to the public Web API, receive 403, and
-- without this check -- report it under the same code path as a local
permission problem.  The local API is by definition local; anything else is a
misconfiguration that must be named as one.
"""

_PAGE_SIZE = 100
"""Items per request.

The local API imposes no maximum, so this number is not a limit being obeyed.
It exists so that a large library arrives in bounded pages, so that progress is
reportable while it does, and so that ``Total-Results`` is compared against
something on every page rather than only once.
"""

_MAX_PAGES = 2_000
"""Hard ceiling on pagination rounds, so a library that never stops paging ends
as an honest ``PARTIAL`` rather than running forever.  At 100 items per page
this bounds one run at 200,000 items."""

_MIN_REQUEST_INTERVAL = 0.05
"""Seconds between requests.

**This is not a rate limit, and saying so is the point.**  The local API runs
inside the user's own desktop app over a loopback socket; there is no remote
quota and the documentation states the local API has no rate limits.

The value is nonetheless positive rather than ``0.0`` because ``0.0`` is not
"no limit" in this project -- it is the value that makes
:func:`~lit_search.utils.rate_limit.reserve_slot` return before it announces
anything, so the limiter is silently disabled and its declaration never appears
in the log.  A source publishing no limit is exactly the case where a porter
writes ``0.0``, which is why the specification's silent-failure watchlist calls
it out.  A positive interval keeps the limiter engaged and the declaration
visible, and 0.05s over a handful of paged requests costs nothing measurable.

The single-threaded HTTP server this talks to shares a process with the user's
Zotero window, which is a second, weaker reason not to hammer it.
"""

# Zotero item types that are not bibliographic records.  They are counted and
# reported rather than dropped: an item seen and not returned is a coverage
# fact, and the alternative -- silently shrinking the set -- is the shape of
# failure this project exists to remove.
_NON_BIBLIOGRAPHIC_TYPES = frozenset({"note", "attachment", "annotation"})

# Zotero item type to BibTeX-aligned paper type.  Anything unlisted becomes
# MISC rather than being guessed into a neighbouring type.
_PAPER_TYPES: dict[str, PaperType] = {
    "journalArticle": PaperType.ARTICLE,
    "magazineArticle": PaperType.ARTICLE,
    "newspaperArticle": PaperType.ARTICLE,
    "conferencePaper": PaperType.INPROCEEDINGS,
    "preprint": PaperType.UNPUBLISHED,
    "manuscript": PaperType.UNPUBLISHED,
    "book": PaperType.BOOK,
    "bookSection": PaperType.INCOLLECTION,
    "report": PaperType.TECHREPORT,
}

# Zotero spells the venue differently for each item type.  Read in order.
_VENUE_FIELDS = (
    "publicationTitle",
    "proceedingsTitle",
    "conferenceName",
    "bookTitle",
    "repository",
    "institution",
    "publisher",
)

_SOURCE_TYPES: dict[str, SourceType] = {
    "journalArticle": SourceType.JOURNAL,
    "conferencePaper": SourceType.CONFERENCE,
    "preprint": SourceType.REPOSITORY,
    "book": SourceType.BOOK,
    "bookSection": SourceType.BOOK,
}


class ZoteroLocalApiError(ConnectorError):
    """A Zotero local-API response that is not a usable item list.

    Raised where a default would otherwise be silently substituted: a body that
    is not a JSON array, a library prefix the API rejects, or a transport
    failure that means the desktop app is not there.
    """


@dataclass(slots=True)
class _Retrieval:
    """Mutable bookkeeping for one paginated library read.

    Attributes
    ----------
    papers : list[Paper]
        Records that parsed, and that matched the query when one was applied.
    processed : int
        Items seen, whether or not they parsed or matched.
    parsed : int
        Items that became a :class:`~lit_search.core.paper.Paper`.
    matched : int
        Parsed items that satisfied the query predicate.
    skipped_non_bibliographic : int
        Items whose type is a note, attachment or annotation.
    pages : int
        Requests consumed.
    total : int | None
        ``Total-Results`` from the first page.
    total_drift : str | None
        Set when a later page reported a different ``Total-Results``.
    truncated_early : bool
        Whether the read ended for a reason other than the cap or exhaustion.
    error_detail : str | None
        Description of the truncation.
    error_type : str | None
        Exception class name, when the truncation came from an exception.
    """

    papers: list[Paper] = field(default_factory=list)
    processed: int = 0
    parsed: int = 0
    matched: int = 0
    skipped_non_bibliographic: int = 0
    pages: int = 0
    total: int | None = None
    total_drift: str | None = None
    truncated_early: bool = False
    error_detail: str | None = None
    error_type: str | None = None

    def note_total(self, total: int | None) -> None:
        """Record the reported total, or the fact that it moved.

        Parameters
        ----------
        total : int | None
            ``Total-Results`` for the page just read.
        """
        if total is None:
            return
        if self.total is None:
            self.total = total
        elif total != self.total and self.total_drift is None:
            self.total_drift = (
                f"Zotero reported {self.total} matching items on the first page and "
                f"{total} on a later one, so the library changed mid-read and neither "
                f"total can be checked against what was retrieved"
            )

    def truncate(self, detail: str, error_type: str | None = None) -> None:
        """Mark the read as having stopped short, keeping the first reason.

        Parameters
        ----------
        detail : str
            Why the read stopped.
        error_type : str | None
            Exception class name, when an exception caused it.
        """
        self.truncated_early = True
        if self.error_detail is None:
            self.error_detail = detail
            self.error_type = error_type


class ZoteroConnector(SearchConnectorBase):
    """Connector for a local Zotero library.

    Enumerates the library through the desktop client's local API and applies
    the query locally.  See the module docstring for the three failure
    outcomes, and :meth:`fetch_library` for the unfiltered read the
    exclude-my-library operation needs.
    """

    _credential_attrs: ClassVar[tuple[str, ...]] = ()
    """Reads against the local API are unauthenticated -- there is no key to
    scope a bucket by.  Declaring the empty tuple rather than accepting the
    ``("_api_key",)`` default states that the absence was checked."""

    def __init__(
        self,
        query_builder: ZoteroQueryBuilder | None = None,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        library: str = _DEFAULT_LIBRARY,
    ) -> None:
        """Create a Zotero searcher.

        Parameters
        ----------
        query_builder : ZoteroQueryBuilder | None
            Builder used to validate queries and compile the local predicate.
        base_url : str
            Root of the local API.  Must name a loopback host; see
            :data:`_LOCAL_HOSTNAMES`.
        library : str
            Library prefix, ``"users/0"`` for the logged-in user's own library
            or ``"groups/<id>"`` for a group library.

        Raises
        ------
        ValueError
            When *base_url* does not name a loopback host.
        """
        super().__init__()
        self._query_builder: ZoteroQueryBuilder = query_builder or ZoteroQueryBuilder()
        self._base_url = base_url.rstrip("/")
        self._library = library.strip("/")
        _require_local_url(self._base_url)

    @property
    def name(self) -> str:
        """Return the database identifier.

        Returns
        -------
        str
            Database name.
        """
        return Database.ZOTERO.value

    @property
    def query_builder(self) -> QueryBuilder:
        """Return the Zotero query builder.

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
            this is positive even though there is no remote quota.
        """
        return _MIN_REQUEST_INTERVAL

    @property
    def rate_limit_service(self) -> str:
        """Return the quota authority these requests are paced against.

        Stated explicitly so that a future connector against another local-API
        route -- the full-text endpoint, or the write path an export sink would
        use -- shares this clock instead of getting its own.

        Returns
        -------
        str
            Quota-authority identifier.
        """
        return "zotero_local"

    def _prepare_headers(self, headers: dict) -> dict:
        """Add the API version header Zotero expects.

        Parameters
        ----------
        headers : dict
            Raw headers.

        Returns
        -------
        dict
            Headers with ``Zotero-API-Version`` set.
        """
        prepared = super()._prepare_headers(headers)
        prepared.setdefault("Zotero-API-Version", "3")
        return prepared

    # ------------------------------------------------------------------
    # Requests
    # ------------------------------------------------------------------

    @property
    def _items_url(self) -> str:
        """Return the endpoint listing top-level items.

        ``/items/top`` rather than ``/items`` because the latter also returns
        child notes and attachments, which are not records and would arrive as
        a stream of items that cannot be parsed into papers.

        Returns
        -------
        str
            Full URL.
        """
        return f"{self._base_url}/{self._library}/items/top"

    def _request_page(self, start: int, page_size: int) -> tuple[list[Any], int | None]:
        """Fetch one page of items and its reported total.

        Parameters
        ----------
        start : int
            Zero-based index of the first item to return.
        page_size : int
            Number of items to request.

        Returns
        -------
        tuple[list[Any], int | None]
            The page's raw items and ``Total-Results``, which is ``None`` when
            the header was absent or unreadable.

        Raises
        ------
        MissingApiKeyError
            When Zotero answered 403, which means the local API is switched
            off in the desktop client's preferences.
        ZoteroLocalApiError
            When Zotero is not reachable, or answered with something that is
            not a JSON array of items.
        """
        params = {
            "format": "json",
            "include": "data",
            "itemType": "-attachment",
            "limit": page_size,
            "start": start,
        }
        try:
            response = self._get(self._items_url, params)
        except requests.HTTPError as exc:
            _raise_for_http_error(exc)
            raise  # pragma: no cover - _raise_for_http_error always raises
        except requests.ConnectionError as exc:
            raise ZoteroLocalApiError(
                f"could not connect to the Zotero local API at {self._base_url}. Zotero "
                f"desktop must be running for its library to be read; reporting this as a "
                f"failure rather than as an empty library is deliberate, because an empty "
                f"library would silently mark every already-held paper as new. "
                f"({type(exc).__name__}: {exc})"
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise ZoteroLocalApiError(
                f"the Zotero local API returned a non-JSON body with status "
                f"{response.status_code}: {exc}"
            ) from exc
        if not isinstance(payload, list):
            raise ZoteroLocalApiError(
                f"the Zotero local API returned a JSON {type(payload).__name__} where an "
                f"array of items was expected. Treating it as an empty library would "
                f"report a misconfigured request as a library holding nothing."
            )
        return payload, _read_total(response)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _read_library(
        self,
        matches: Predicate | None,
        max_papers: int | None,
        progress_callback: Callable[[int, int | None], None] | None,
        since: datetime.date | None,
        until: datetime.date | None,
    ) -> _Retrieval:
        """Page through the library, parsing and optionally matching each item.

        Parameters
        ----------
        matches : Predicate | None
            Compiled query predicate, or ``None`` to keep every record.
        max_papers : int | None
            Caller-imposed cap.
        progress_callback : Callable[[int, int | None], None] | None
            Progress callback called as ``callback(items_seen, total_or_none)``.
        since : datetime.date | None
            Only keep papers published on or after this date.
        until : datetime.date | None
            Only keep papers published on or before this date.

        Returns
        -------
        _Retrieval
            Bookkeeping for the finished read.

        Raises
        ------
        MissingApiKeyError
            When the local API is switched off and nothing was retrieved.
        ZoteroLocalApiError
            When the first request failed.
        """
        run = _Retrieval()
        start = 0
        while True:
            try:
                items, total = self._request_page(start, _PAGE_SIZE)
            except (requests.RequestException, ConnectorError, MissingApiKeyError) as exc:
                self._record_page_failure(run, exc, start)
                break

            run.note_total(total)
            self._absorb_page(run, items, matches, max_papers, since, until)
            if progress_callback is not None:
                progress_callback(run.processed, run.total)

            if not self._should_continue(run, len(items), max_papers):
                break
            start += len(items)

        if progress_callback is not None:
            progress_callback(run.processed, run.total)
        return run

    @staticmethod
    def _record_page_failure(run: _Retrieval, exc: Exception, start: int) -> None:
        """Record a page failure, or re-raise when nothing was retrieved.

        Parameters
        ----------
        run : _Retrieval
            Bookkeeping for the current read.
        exc : Exception
            The failure raised while fetching or reading the page.
        start : int
            Offset the failed request carried.

        Raises
        ------
        Exception
            The original failure, when nothing had been seen yet.  With no
            items in hand there is no partial result to report and the failure
            is the whole story.
        """
        if run.processed == 0:
            raise exc
        logger.warning(
            "Zotero: the page at start=%d failed after %d item(s): %s", start, run.processed, exc
        )
        run.truncate(f"the page at start={start} failed: {exc}", type(exc).__name__)

    def _absorb_page(
        self,
        run: _Retrieval,
        items: Sequence[Any],
        matches: Predicate | None,
        max_papers: int | None,
        since: datetime.date | None,
        until: datetime.date | None,
    ) -> None:
        """Fold one page into *run*, keeping the records that match.

        Parameters
        ----------
        run : _Retrieval
            Bookkeeping for the current read, updated in place.
        items : Sequence[Any]
            Raw items from one page.
        matches : Predicate | None
            Compiled query predicate, or ``None`` to keep every record.
        max_papers : int | None
            Caller-imposed cap.
        since : datetime.date | None
            Lower publication-date bound.
        until : datetime.date | None
            Upper publication-date bound.
        """
        for item in items:
            run.processed += 1
            data = item.get("data") if isinstance(item, dict) else None
            if not isinstance(data, dict):
                continue
            if str(data.get("itemType") or "") in _NON_BIBLIOGRAPHIC_TYPES:
                run.skipped_non_bibliographic += 1
                continue
            record = _item_record(data)
            paper = self._parse_item(data, record)
            if paper is None:
                continue
            run.parsed += 1
            if not _within_dates(paper.publication_date, since, until):
                continue
            if matches is not None and not matches(record):
                continue
            run.matched += 1
            if max_papers is None or len(run.papers) < max_papers:
                run.papers.append(paper)
        run.pages += 1

    @staticmethod
    def _should_continue(run: _Retrieval, page_length: int, max_papers: int | None) -> bool:
        """Decide whether another page should be requested.

        Termination is decided from the reported total and from an empty page,
        never from a short one.  A short page is not an end-of-set signal in an
        API that also lets the caller choose the page size, and treating it as
        one is how a walk stops early and calls it done.

        Parameters
        ----------
        run : _Retrieval
            Bookkeeping for the current read.
        page_length : int
            Number of items the last page carried.
        max_papers : int | None
            Caller-imposed cap.

        Returns
        -------
        bool
            ``True`` when another request should be made.
        """
        if page_length == 0:
            if run.total is not None and run.processed < run.total:
                run.truncate(
                    f"Zotero stopped returning items after {run.processed} of the "
                    f"{run.total} it reported"
                )
            return False
        if max_papers is not None and len(run.papers) >= max_papers:
            return False
        if run.total is not None and run.processed >= run.total:
            return False
        if run.pages >= _MAX_PAGES:
            run.truncate(
                f"the read stopped at the connector's {_MAX_PAGES}-page safety limit after "
                f"{run.processed} items"
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def _fetch_papers(
        self,
        query: Query,
        max_papers: int | None,
        progress_callback: Callable[[int, int | None], None] | None,
        since: datetime.date | None = None,
        until: datetime.date | None = None,
    ) -> CoveredPapers:
        """Enumerate the library and match the query locally.

        Parameters
        ----------
        query : Query
            Validated query object.
        max_papers : int | None
            Maximum papers to retrieve.  ``None`` means unlimited.
        progress_callback : Callable[[int, int | None], None] | None
            Progress callback called as ``callback(items_seen, total_or_none)``.
        since : datetime.date | None
            Only return papers published on or after this date.
        until : datetime.date | None
            Only return papers published on or before this date.

        Returns
        -------
        CoveredPapers
            Matching papers and the coverage status describing them.

        Raises
        ------
        MissingApiKeyError
            When the local API is switched off in Zotero's preferences.
        ZoteroLocalApiError
            When Zotero could not be reached or answered unusably.
        """
        matches = self._query_builder.compile_predicate(query)
        run = self._read_library(matches, max_papers, progress_callback, since, until)
        return CoveredPapers(
            papers=tuple(run.papers),
            coverage=self._build_search_coverage(run=run, max_papers=max_papers),
        )

    def fetch_library(self, max_papers: int | None = None) -> CoveredPapers:
        """Read the whole library, applying no query.

        This is the read the exclude-my-library operation needs (``D8``), and
        it is the one path in this connector that can honestly report
        ``COMPLETE``: no query is being answered, so the number of items Zotero
        reports *is* the number that should have arrived, and
        :meth:`~lit_search.core.coverage.SourceCoverage.from_pagination` can
        compare the two.

        The claim is checked rather than trusted.  ``search`` gets that for
        free from
        :meth:`~lit_search.connectors.search_base.SearchConnectorBase.search`;
        this method is outside that path, so it opens its own audit scope and
        runs the same verification -- otherwise a ``COMPLETE`` here would be
        the one unaudited completeness claim in the tree, on precisely the set
        whose incompleteness silently corrupts every difference taken against
        it.

        Parameters
        ----------
        max_papers : int | None
            Maximum records to return.  ``None``, the default, means the whole
            library, which is what a subtrahend needs.

        Returns
        -------
        CoveredPapers
            The library's records and the coverage status describing them.

        Raises
        ------
        MissingApiKeyError
            When the local API is switched off in Zotero's preferences.
        ZoteroLocalApiError
            When Zotero could not be reached or answered unusably.
        """
        with auditing(self.name) as audit:
            run = self._read_library(None, max_papers, audit.note_progress, None, None)
            result = CoveredPapers(
                papers=tuple(run.papers),
                coverage=self._build_library_coverage(run=run, max_papers=max_papers),
            )
            return verify_coverage_claim(
                result.with_source(self.name), max_papers=max_papers, audit=audit
            )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_item(self, data: dict[str, Any], record: Record) -> Paper | None:
        """Build a :class:`Paper` from one Zotero item's ``data`` object.

        Parameters
        ----------
        data : dict[str, Any]
            The item's ``data`` object.
        record : Record
            The same item's text, already normalised for the predicate.

        Returns
        -------
        Paper | None
            Parsed paper, or ``None`` when the item has no title or fails model
            validation.  Callers **must** count the ``None`` returns: an
            unparsed item is a coverage loss, not an absent paper.
        """
        title = str(data.get("title") or "").strip()
        if not title:
            return None
        item_type = str(data.get("itemType") or "")
        try:
            return Paper(
                title=title,
                abstract=str(data.get("abstractNote") or "").strip(),
                authors=[Author(name=name) for name in record.get("authors", ()) if name],
                source=_item_source(data, item_type),
                publication_date=parse_date(str(data.get("date") or "").strip() or None),
                url=str(data.get("url") or "").strip() or None,
                doi=normalize_doi(str(data.get("DOI") or "")),
                keywords=set(record.get("keywords", ())) or None,
                found_in={self.name},
                paper_type=_PAPER_TYPES.get(item_type, PaperType.MISC),
                language=normalize_language(str(data.get("language") or "") or None),
            )
        except ValueError:
            logger.debug("Zotero: item %s failed model validation.", data.get("key"))
            return None

    # ------------------------------------------------------------------
    # Coverage
    # ------------------------------------------------------------------

    def _shared_notes(self, run: _Retrieval) -> list[str]:
        """Return the coverage notes both entry points share.

        Parameters
        ----------
        run : _Retrieval
            Bookkeeping for the finished read.

        Returns
        -------
        list[str]
            Notes describing losses that are not about the query.
        """
        notes: list[str] = []
        if run.error_detail:
            notes.append(run.error_detail)
        if run.total_drift:
            notes.append(run.total_drift)
        if run.skipped_non_bibliographic:
            notes.append(
                f"{run.skipped_non_bibliographic} of {run.processed} items were standalone "
                f"notes or annotations rather than bibliographic records"
            )
        unparsed = run.processed - run.parsed - run.skipped_non_bibliographic
        if unparsed > 0:
            notes.append(
                f"{unparsed} of {run.processed} items carried no title or failed model validation"
            )
        return notes

    def _build_search_coverage(self, *, run: _Retrieval, max_papers: int | None) -> SourceCoverage:
        """Describe what a query-answering read achieved.

        Always ``PARTIAL``, and structurally so: Zotero's quicksearch cannot
        express lit-search's query, so the query was answered here, and the
        library's item count is not a count of records matching it.  Reporting
        ``CAPPED`` when the caller's limit bound would be worse than useless --
        ``CAPPED`` ranks milder than ``PARTIAL`` and means a known amount is
        missing on purpose, so a truncated read would read as healthier than an
        untruncated one.  The cap is named in the detail instead.

        Parameters
        ----------
        run : _Retrieval
            Bookkeeping for the finished read.
        max_papers : int | None
            Caller-imposed cap.

        Returns
        -------
        SourceCoverage
            A ``PARTIAL`` record.
        """
        notes = [
            "Zotero's local API offers a substring quicksearch with no boolean operators "
            "and no negation, so this enumerated the library and matched the query locally",
            f"saw {run.processed} top-level items"
            + (f" of the {run.total} Zotero reported" if run.total is not None else "")
            + f" and matched {run.matched}",
        ]
        if max_papers is not None and run.matched > len(run.papers):
            notes.append(
                f"the caller's limit of {max_papers} stopped collection, so {run.matched} "
                f"records had already matched when it bound"
            )
        notes.extend(self._shared_notes(run))
        return SourceCoverage.partial(
            self.name,
            len(run.papers),
            detail="; ".join(notes),
            error=run.error_detail,
            error_type=run.error_type,
        )

    def _build_library_coverage(self, *, run: _Retrieval, max_papers: int | None) -> SourceCoverage:
        """Describe what an unfiltered library read achieved.

        Parameters
        ----------
        run : _Retrieval
            Bookkeeping for the finished read.
        max_papers : int | None
            Caller-imposed cap.

        Returns
        -------
        SourceCoverage
            ``COMPLETE`` only when as many records came back as Zotero said the
            library held; otherwise whatever
            :meth:`~lit_search.core.coverage.SourceCoverage.from_pagination`
            decides from the counters.
        """
        notes = self._shared_notes(run)
        if run.truncated_early:
            return SourceCoverage.partial(
                self.name,
                len(run.papers),
                detail="; ".join(notes) or "the library read stopped before it was exhausted",
                reported_total=run.total,
                error=run.error_detail,
                error_type=run.error_type,
            )
        return SourceCoverage.from_pagination(
            self.name,
            retrieved=len(run.papers),
            reported_total=run.total,
            cap=max_papers,
            shortfall_detail="; ".join(notes) or None,
        )


def _require_local_url(base_url: str) -> None:
    """Refuse a base URL that is not a loopback address.

    Parameters
    ----------
    base_url : str
        Configured base URL.

    Raises
    ------
    ValueError
        When the host is not one of :data:`_LOCAL_HOSTNAMES`.  Pointed at the
        public Web API this connector would send an unauthenticated request,
        receive 403, and report a misconfiguration under the same code path as
        a local permission problem.
    """
    host = (urlparse(base_url).hostname or "").lower()
    if host not in _LOCAL_HOSTNAMES:
        raise ValueError(
            f"ZoteroConnector talks to the desktop client's local API and refuses "
            f"'{base_url}': host '{host or base_url}' is not a loopback address. The "
            f"public Web API needs a key and would answer an unauthenticated request "
            f"with 403, which must not be read as a local permission problem."
        )


def _raise_for_http_error(exc: requests.HTTPError) -> None:
    """Translate an HTTP error into the status it actually means.

    Parameters
    ----------
    exc : requests.HTTPError
        Error raised by the base class after retries were exhausted.

    Raises
    ------
    MissingApiKeyError
        On 403, which Zotero returns when "Allow other applications on this
        computer to communicate with Zotero" is switched off.  It is modelled
        as a missing credential because that is what the enum's
        ``SKIPPED_NO_CREDENTIAL`` describes: never contacted, determinate, and
        fixable by configuration.
    ZoteroLocalApiError
        For every other status, including the 404 the local API returns for a
        library prefix it does not recognise and the 400 it returns for a user
        id that is not the logged-in one.
    """
    status = exc.response.status_code if exc.response is not None else None
    if status == 403:
        raise MissingApiKeyError(
            "zotero local API access (enable Settings -> Advanced -> 'Allow other "
            "applications on this computer to communicate with Zotero')"
        ) from exc
    raise ZoteroLocalApiError(
        f"the Zotero local API answered with HTTP {status}: {exc}. Treating this as an "
        f"empty library would report a configuration problem as a library holding nothing."
    ) from exc


def _read_total(response: requests.Response) -> int | None:
    """Read ``Total-Results`` from a response, or report that it was absent.

    Parameters
    ----------
    response : requests.Response
        Response to read.

    Returns
    -------
    int | None
        The total, or ``None`` when the header is missing or unreadable.  A
        missing total is not an error, but it does mean completeness cannot be
        verified, and ``from_pagination`` turns that into ``PARTIAL``.
    """
    raw = response.headers.get("Total-Results")
    if raw is None:
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        logger.debug("Zotero: unreadable Total-Results header %r.", raw)
        return None


def _creator_names(data: dict[str, Any]) -> list[str]:
    """Extract creator names from a Zotero item.

    Zotero stores a creator either as ``firstName``/``lastName`` or as a single
    ``name`` field, and it records editors and translators alongside authors.
    Authors are preferred; when an item lists none -- an edited volume, for
    example -- every creator is used rather than returning nobody.

    Parameters
    ----------
    data : dict[str, Any]
        The item's ``data`` object.

    Returns
    -------
    list[str]
        Creator names.
    """
    creators = data.get("creators")
    if not isinstance(creators, list):
        return []
    authors: list[str] = []
    others: list[str] = []
    for creator in creators:
        if not isinstance(creator, dict):
            continue
        name = str(creator.get("name") or "").strip()
        if not name:
            parts = [
                str(creator.get("firstName") or "").strip(),
                str(creator.get("lastName") or "").strip(),
            ]
            name = " ".join(part for part in parts if part)
        if not name:
            continue
        if str(creator.get("creatorType") or "") == "author":
            authors.append(name)
        else:
            others.append(name)
    return authors or others


def _item_record(data: dict[str, Any]) -> Record:
    """Normalise one item into the field bag the local predicate reads.

    Parameters
    ----------
    data : dict[str, Any]
        The item's ``data`` object.

    Returns
    -------
    Record
        Text keyed by the field names in
        :data:`lit_search.query.builders.zotero.FIELD_MAP`.
    """
    tags = data.get("tags")
    keywords = [
        str(tag.get("tag") or "").strip()
        for tag in (tags if isinstance(tags, list) else [])
        if isinstance(tag, dict) and str(tag.get("tag") or "").strip()
    ]
    venues = [
        str(data.get(name) or "").strip() for name in _VENUE_FIELDS if str(data.get(name) or "")
    ]
    return {
        "title": [str(data.get("title") or "").strip()],
        "abstract": [str(data.get("abstractNote") or "").strip()],
        "keywords": keywords,
        "authors": _creator_names(data),
        "source": venues,
    }


def _item_source(data: dict[str, Any], item_type: str) -> Source | None:
    """Build a :class:`Source` from an item's venue fields.

    Parameters
    ----------
    data : dict[str, Any]
        The item's ``data`` object.
    item_type : str
        Zotero item type, which decides how the venue is classified.

    Returns
    -------
    Source | None
        The venue, or ``None`` when the item names none.
    """
    title = next(
        (
            str(data.get(name)).strip()
            for name in _VENUE_FIELDS
            if str(data.get(name) or "").strip()
        ),
        "",
    )
    if not title:
        return None
    return Source(
        title=title,
        issn=str(data.get("ISSN") or "").strip() or None,
        isbn=str(data.get("ISBN") or "").strip() or None,
        source_type=_SOURCE_TYPES.get(item_type, SourceType.OTHER),
    )


def _within_dates(
    value: datetime.date | None,
    since: datetime.date | None,
    until: datetime.date | None,
) -> bool:
    """Whether a publication date falls inside the requested bounds.

    A record with no date is kept.  Zotero's ``date`` field is free text and is
    frequently blank or unparseable, and dropping those would silently remove
    real records from a set whose whole job is to say what the library already
    holds.

    Parameters
    ----------
    value : datetime.date | None
        The record's publication date.
    since : datetime.date | None
        Lower bound.
    until : datetime.date | None
        Upper bound.

    Returns
    -------
    bool
        ``True`` when the record should be kept.
    """
    if value is None:
        return True
    if since is not None and value < since:
        return False
    return not (until is not None and value > until)
