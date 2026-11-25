"""chemRxiv searcher, implemented as a CrossRef filter profile.

chemRxiv is **not an independent source** and this module is deliberately not a
translation of ``paper_search_mcp/academic_platforms/chemrxiv.py``.  The donor
class is ``ChemRxivSearcher(CrossRefSearcher)``, which adds one filter to a
CrossRef query::

    base_filter = f"type:posted-content,from-publisher:{self.preprint_server}"

**That filter does not exist.**  Verified live on 2026-08-31, unchanged from the
2026-08-30 reading in the specification: ``api.crossref.org`` answers

.. code-block:: text

    HTTP 400
    {"status":"failed","message-type":"validation-failure",
     "message":[{"type":"filter-not-available","value":"from-publisher", ...}]}

``raise_for_status()`` raises ``HTTPError``, ``HTTPError`` is a
``requests.RequestException``, and donor ``crossref.py`` catches
``requests.RequestException`` and returns ``[]``.  So the donor chemRxiv
connector **returns zero results for every query ever asked of it, and reports
that as "chemRxiv holds no matching paper."**  It is a live instance of the
failure this fork exists to prevent, sitting in the tree we were asked to port
from, and porting it mechanically would have carried it across intact.

The working replacement, verified live on 2026-08-31: chemRxiv content is
registered under DOI prefix ``10.26434`` (the prefix record names the owner as
"American Chemical Society (ACS)", but every DOI under it is a
``10.26434/chemrxiv-...`` preprint).  ``filter=type:posted-content,prefix:10.26434``
returns 55,288 records with no query and 1,007 for ``query=catalysis``.
Filtering by ``member:316`` instead would sweep in the whole of ACS's journal
output, so the prefix is the correct handle.

Two consequences follow from being a CrossRef profile rather than a source:

- The requests are paced against **CrossRef's** bucket, not chemrxiv.org's, so
  :attr:`ChemRxivConnector.rate_limit_service` returns ``"crossref"``.  This
  also moots the unverifiable chemrxiv.org rate limit the specification flags.
- The query semantics are CrossRef's, which are far weaker than a Boolean
  search API's.  See :mod:`lit_search.query.builders.crossref`; the short
  version is that CrossRef has no ``AND``, no ``NOT`` and no abstract index,
  and this connector reports the resulting shortfall as ``PARTIAL`` instead of
  letting it pass as a complete answer.

Live API behaviours this module is built around, all observed 2026-08-31:

- **An invalid cursor is not an error -- it silently restarts pagination.**
  ``cursor=GARBAGE!!`` returned HTTP 200 and the *first* page of results, with
  a ``next-cursor`` pointing just past the first record.  A connector that
  trusted the cursor would loop forever re-reading page one and would report a
  large, plausible, duplicate-filled result.  :meth:`ChemRxivConnector._advance`
  therefore refuses a cursor it has already sent.
- **Cursor exhaustion is an empty page with no ``next-cursor``.**  Walking a
  17-record set at ``rows=10`` gave pages of 10, 7 and then
  ``{"total-results":17,"items":[],"items-per-page":17}`` with the
  ``next-cursor`` key *absent*.  Feeding that absent value back restarted the
  walk from the beginning, per the previous point.
- **Bad request parameters fail loudly, with a body worth keeping.**
  ``rows=1001``, ``offset=10001``, ``query.notafield=x`` and ``filter=type:notatype``
  all return HTTP 400 with a JSON ``validation-failure`` body naming what was
  wrong.  :meth:`ChemRxivConnector._get_page` lifts that message out of the
  response so the ledger says *why* rather than only "400 Client Error".
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

from lit_search.connectors.crossref import CrossRefConnector
from lit_search.connectors.search_base import SearchConnectorBase
from lit_search.core.coverage import CoveredPapers, SourceCoverage
from lit_search.core.paper import Database, Paper, PaperType
from lit_search.core.query import Query
from lit_search.exceptions import ConnectorError
from lit_search.query.builders.crossref import CrossRefQueryBuilder
from lit_search.utils.pagination_logging import log_pagination_request_failure

logger = logging.getLogger(__name__)

_WORKS_URL = "https://api.crossref.org/works"

#: DOI prefix every chemRxiv preprint is registered under.  Verified against
#: ``api.crossref.org/prefixes/10.26434`` on 2026-08-31.
CHEMRXIV_DOI_PREFIX = "10.26434"

#: The CrossRef filter that *is* the chemRxiv profile.  ``posted-content`` is
#: CrossRef's type for a preprint; the prefix narrows it to chemRxiv.
_PROFILE_FILTER = f"type:posted-content,prefix:{CHEMRXIV_DOI_PREFIX}"

#: Verified live 2026-08-31: ``rows=1001`` is refused with HTTP 400
#: ``"must be a positive integer less than or equal to 1000"``.
_MAX_ROWS = 1000
_ROWS = 100

#: Safety limit on pagination rounds.  Only reachable if CrossRef keeps
#: handing out fresh cursors; it exists so such a run ends as an honest
#: ``PARTIAL`` rather than never ending.
_MAX_PAGES = 10_000

_ANONYMOUS_MIN_REQUEST_INTERVAL = 1.0
"""Seconds between requests when no polite-pool mailto is configured.

**Measured, not inferred.**  A live request to ``api.crossref.org`` on
2026-08-31 with no ``mailto`` in the ``User-Agent`` returned
``x-rate-limit-limit: 1`` and ``x-rate-limit-interval: 1s``.

Note that this **contradicts the value the tree already ships**:
``lit_search/connectors/crossref.py`` declares ``_MIN_REQUEST_INTERVAL = 0.1``
with the comment "well within limits", which is ten times CrossRef's own
anonymous allowance.  Correcting that connector is separate work and is deliberately not done
here, but the two
connectors share one rate-limit scope, so while it stands the shared bucket is
only as well behaved as its fastest declarant.  This is recorded at the value
rather than left for someone to rediscover from a 429 storm.
"""

_POLITE_MIN_REQUEST_INTERVAL = 0.34
"""Seconds between requests when a polite-pool mailto is configured.

**Measured, not inferred.**  The same live request on 2026-08-31 carrying
``User-Agent: ... (mailto:...)`` returned ``x-rate-limit-limit: 3`` and
``x-rate-limit-interval: 1s``.  0.34s is 3 requests/second with a small margin.
"""


class ChemRxivBodyError(ConnectorError):
    """A CrossRef response that cannot be read as a work list.

    Covers both a non-2xx body carrying CrossRef's ``validation-failure``
    explanation and an HTTP 200 body of an unexpected shape.  Raised rather
    than defaulted away, because ``payload.get("message", {}).get("items", [])``
    returns ``[]`` for every one of these.
    """


@dataclass(slots=True)
class _Retrieval:
    """Mutable bookkeeping for one paginated chemRxiv run.

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
        ``total-results`` from the first page.
    total_drift : str | None
        Set when a later page reported a different ``total-results``.
    seen_cursors : set[str]
        Cursors already sent.  CrossRef restarts pagination for a cursor it
        cannot parse, so a repeat is the only available loop detector.
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
    seen_cursors: set[str] = field(default_factory=set)
    truncated_early: bool = False
    error_detail: str | None = None
    error_type: str | None = None

    def note_total(self, total_results: object) -> None:
        """Record the reported total, or the fact that it moved.

        Parameters
        ----------
        total_results : object
            Raw ``total-results`` from a page.  Non-integers are ignored.
        """
        if not isinstance(total_results, int) or isinstance(total_results, bool):
            return
        if self.total is None:
            self.total = total_results
        elif total_results != self.total and self.total_drift is None:
            self.total_drift = (
                f"CrossRef reported {self.total} matching records on the first page and "
                f"{total_results} on a later one, so the index changed mid-retrieval and "
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
        Row count between 1 and :data:`_MAX_ROWS`.
    """
    remaining = (max_papers - len(run.papers)) if max_papers is not None else _ROWS
    return max(1, min(_ROWS, _MAX_ROWS, remaining))


def _validation_message(response: requests.Response | None) -> str | None:
    """Extract CrossRef's explanation from a rejected request's body.

    Parameters
    ----------
    response : requests.Response | None
        Response attached to a :class:`requests.HTTPError`.

    Returns
    -------
    str | None
        Joined ``message`` entries, or ``None`` when the body carries none.
    """
    if response is None:
        return None
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    entries = payload.get("message")
    if not isinstance(entries, list):
        return None
    texts = [
        str(entry.get("message")).strip()
        for entry in entries
        if isinstance(entry, dict) and entry.get("message")
    ]
    return "; ".join(texts) or None


class ChemRxivConnector(SearchConnectorBase):
    """Connector for chemRxiv, driven as a CrossRef filter profile.

    Parameters
    ----------
    email : str | None
        Contact address for CrossRef's polite pool.  Supplying it triples the
        allowed request rate (3/s against 1/s, both read from live
        ``x-rate-limit-*`` headers on 2026-08-31) and puts this connector in a
        different rate-limit bucket from anonymous CrossRef traffic.  The
        search runner does not pass it today, so the default is the anonymous
        pool -- which is the safe direction to be wrong in.
    query_builder : CrossRefQueryBuilder | None
        Builder used to validate and convert queries.
    """

    _credential_attrs: ClassVar[tuple[str, ...]] = ("_email",)
    """CrossRef's quota is keyed by the polite-pool mailto, not by an API key.

    Declared explicitly so that this connector and
    :class:`~lit_search.connectors.crossref.CrossRefConnector` fingerprint the
    same way for the same address and therefore share one clock, which they
    must: they draw on one allowance at one host.
    """

    def __init__(
        self,
        email: str | None = None,
        query_builder: CrossRefQueryBuilder | None = None,
    ) -> None:
        """Create a chemRxiv searcher.

        Parameters
        ----------
        email : str | None
            Contact address for CrossRef's polite pool.
        query_builder : CrossRefQueryBuilder | None
            Builder used to validate and convert queries.  When ``None`` a
            default :class:`CrossRefQueryBuilder` is created.
        """
        super().__init__()
        self._email = email
        self._query_builder: CrossRefQueryBuilder = query_builder or CrossRefQueryBuilder()

    @property
    def name(self) -> str:
        """Return the database identifier.

        Returns
        -------
        str
            Database name.
        """
        return Database.CHEMRXIV.value

    @property
    def query_builder(self) -> QueryBuilder:
        """Return the CrossRef query builder.

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
            1.0s anonymously, 0.34s with a polite-pool mailto.  Both values
            come from live ``x-rate-limit-*`` headers; see
            :data:`_ANONYMOUS_MIN_REQUEST_INTERVAL`.
        """
        if self._email:
            return _POLITE_MIN_REQUEST_INTERVAL
        return _ANONYMOUS_MIN_REQUEST_INTERVAL

    @property
    def rate_limit_service(self) -> str:
        """Return the quota authority these requests are paced against.

        ``"crossref"``, not ``"chemrxiv"``.  Every request this connector makes
        goes to ``api.crossref.org``, so giving it a bucket of its own would
        let it and
        :class:`~lit_search.connectors.crossref.CrossRefConnector` each spend a
        full allowance and together double the rate against one host.

        Returns
        -------
        str
            Quota-authority identifier.
        """
        return Database.CROSSREF.value

    def _prepare_headers(self, headers: dict) -> dict:
        """Add the polite-pool ``User-Agent`` CrossRef reads the mailto from.

        Parameters
        ----------
        headers : dict
            Headers supplied by the caller.

        Returns
        -------
        dict
            Headers with a ``User-Agent`` added when one is not already set.
        """
        prepared = super()._prepare_headers(headers)
        if "User-Agent" not in prepared:
            suffix = f" (mailto:{self._email})" if self._email else ""
            prepared["User-Agent"] = f"lit-search/{Database.CHEMRXIV.value}{suffix}"
        return prepared

    # ------------------------------------------------------------------
    # Requests and response handling
    # ------------------------------------------------------------------

    def _build_filter(self, since: datetime.date | None, until: datetime.date | None) -> str:
        """Return the CrossRef ``filter`` value for this run.

        Parameters
        ----------
        since : datetime.date | None
            Lower posted-date bound.
        until : datetime.date | None
            Upper posted-date bound.

        Returns
        -------
        str
            Comma-joined filter expression.
        """
        parts = [_PROFILE_FILTER]
        if since is not None:
            parts.append(f"from-posted-date:{since.isoformat()}")
        if until is not None:
            parts.append(f"until-posted-date:{until.isoformat()}")
        return ",".join(parts)

    def _get_page(self, params: dict[str, Any]) -> dict[str, Any]:
        """Fetch and validate one page of CrossRef results.

        Parameters
        ----------
        params : dict[str, Any]
            Request parameters.

        Returns
        -------
        dict[str, Any]
            The response's ``message`` object, guaranteed to carry
            ``total-results`` and a list of ``items``.

        Raises
        ------
        ChemRxivBodyError
            When the request was rejected with a body worth reporting, or when
            an HTTP 200 body is not a work list.
        """
        try:
            response = self._get(_WORKS_URL, params)
        except requests.HTTPError as exc:
            explanation = _validation_message(exc.response)
            if explanation is None:
                raise
            raise ChemRxivBodyError(
                f"CrossRef rejected the request with HTTP "
                f"{exc.response.status_code if exc.response is not None else '?'}: {explanation}"
            ) from exc
        return self._read_body(response)

    @staticmethod
    def _read_body(response: requests.Response) -> dict[str, Any]:
        """Decode and validate one search response body.

        Parameters
        ----------
        response : requests.Response
            Response returned by the works endpoint.

        Returns
        -------
        dict[str, Any]
            The ``message`` object.

        Raises
        ------
        ChemRxivBodyError
            When the body is not JSON, is not a CrossRef envelope, does not
            report ``status: ok``, or carries no ``total-results``.  Every one
            of these would otherwise be read as "chemRxiv holds no matching
            paper".
        """
        try:
            payload = response.json()
        except ValueError as exc:
            raise ChemRxivBodyError(
                f"CrossRef returned a non-JSON body with status {response.status_code}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ChemRxivBodyError(
                f"CrossRef returned a JSON {type(payload).__name__} where an object was expected."
            )
        status = payload.get("status")
        if status != "ok":
            raise ChemRxivBodyError(
                f"CrossRef reported status {status!r} rather than 'ok': "
                f"{_validation_message(response) or 'no explanation supplied'}"
            )
        message = payload.get("message")
        if not isinstance(message, dict) or "total-results" not in message:
            raise ChemRxivBodyError(
                "CrossRef response carried no 'message.total-results', so it is not a "
                "work list. Treating it as an empty result set would report a failure "
                "as zero papers."
            )
        return message

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _posted_date(work: dict[str, Any]) -> datetime.date | None:
        """Return the record's ``posted`` date, if it carries one.

        A preprint's date is the date it was posted, and it is the field this
        connector's ``from-posted-date`` / ``until-posted-date`` filters are
        applied to.  The inherited CrossRef date parser does not look at
        ``posted`` at all -- it reads ``published-print``, ``published-online``,
        ``published``, ``issued`` and ``created`` -- so a record carrying only
        ``posted`` would come back dateless while still matching a date bound.

        Sampled chemRxiv records on 2026-08-31 carried ``posted``, ``issued``,
        ``published`` and ``created`` with identical values, so this rarely
        changes the answer.  It is here so that the field the query filters on
        is the field the record reports.

        Parameters
        ----------
        work : dict[str, Any]
            CrossRef work record.

        Returns
        -------
        datetime.date | None
            Posted date, or ``None`` when the field is absent or malformed.
        """
        posted = work.get("posted")
        parts = posted.get("date-parts") if isinstance(posted, dict) else None
        if not isinstance(parts, list) or not parts or not isinstance(parts[0], list):
            return None
        nums = parts[0]
        if not nums:
            return None
        try:
            year = int(nums[0])
            month = int(nums[1]) if len(nums) > 1 else 1
            day = int(nums[2]) if len(nums) > 2 else 1
            return datetime.date(year, month, day)
        except (ValueError, TypeError):
            return None

    def _parse_item(self, work: Any) -> Paper | None:
        """Parse one CrossRef work record into a :class:`Paper`.

        Field mapping is delegated to
        :class:`~lit_search.connectors.crossref.CrossRefConnector`, which is
        the same parser reading the same API.  Three things are re-stamped
        afterwards: the provenance (the record was reached as chemRxiv), the
        paper type (every record in this profile is a preprint by
        construction, since the filter is ``type:posted-content``), and the
        date (see :meth:`_posted_date`).

        Parameters
        ----------
        work : Any
            One ``message.items`` entry.

        Returns
        -------
        Paper | None
            Parsed paper, or ``None`` when a required field is missing.
            Callers **must** count the ``None`` returns: an unparsed record is
            a coverage loss, not an absent paper.
        """
        if not isinstance(work, dict):
            return None
        try:
            paper = CrossRefConnector._build_paper(work)
        except ValueError:
            logger.debug("chemRxiv: record %s failed model validation.", work.get("DOI"))
            return None
        if paper is None:
            return None
        paper.found_in = {self.name}
        paper.paper_type = PaperType.UNPUBLISHED
        posted = self._posted_date(work)
        if posted is not None:
            paper.publication_date = posted
        return paper

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _fetch_papers(
        self,
        query: Query,
        max_papers: int | None,
        progress_callback: Callable[[int, int | None], None] | None,
        since: datetime.date | None = None,
        until: datetime.date | None = None,
    ) -> CoveredPapers:
        """Fetch chemRxiv preprints from CrossRef, with coverage.

        Parameters
        ----------
        query : Query
            Validated query object.
        max_papers : int | None
            Maximum papers to retrieve.  ``None`` means unlimited.
        progress_callback : Callable[[int, int | None], None] | None
            Progress callback called as ``callback(processed, total_or_none)``.
        since : datetime.date | None
            Only return preprints posted on or after this date.
        until : datetime.date | None
            Only return preprints posted on or before this date.

        Returns
        -------
        CoveredPapers
            Retrieved papers and the coverage status describing them.

        Raises
        ------
        ChemRxivBodyError
            When the first request was rejected or returned an unusable body.
        requests.RequestException
            When the first request failed at the transport level.
        """
        base_params: dict[str, Any] = dict(self._query_builder.convert_query(query))
        base_params["filter"] = self._build_filter(since, until)
        degradation = self._query_builder.scope_degradation(query)

        run = _Retrieval()
        cursor = "*"

        while True:
            run.seen_cursors.add(cursor)
            params = dict(base_params)
            params["rows"] = _rows_for(run, max_papers)
            params["cursor"] = cursor

            try:
                message = self._get_page(params)
            except (requests.RequestException, ConnectorError) as exc:
                self._record_page_failure(run, exc, cursor)
                break

            items = self._absorb_page(run, message)
            if progress_callback is not None:
                progress_callback(run.processed, run.total)

            advanced = self._advance(run, message, items, max_papers)
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
                run=run,
                max_papers=max_papers,
                degradation=degradation,
            ),
        )

    @staticmethod
    def _record_page_failure(run: _Retrieval, exc: Exception, cursor: str) -> None:
        """Record a page failure, or re-raise when nothing was retrieved.

        With no records in hand the failure is the whole story and must not be
        softened into a status the connector invented, so it is re-raised for
        :meth:`SearchConnectorBase.search` to name.  With records in hand the
        run is a real shortfall and the records are kept.

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
            connector_label="chemRxiv",
            processed=run.processed,
            total=run.total,
            position_label="cursor",
            position=cursor,
            exc=exc,
        )
        run.truncate(f"page request at cursor={cursor} failed: {exc}", type(exc).__name__)

    def _absorb_page(self, run: _Retrieval, message: dict[str, Any]) -> list[Any]:
        """Fold one page into *run* and return the raw items it carried.

        Parameters
        ----------
        run : _Retrieval
            Bookkeeping for the current run, updated in place.
        message : dict[str, Any]
            Validated ``message`` object.

        Returns
        -------
        list[Any]
            Raw ``items`` entries, including any that failed to parse -- the
            caller needs the count, because a record seen but not parsed is a
            coverage loss rather than an absent paper.
        """
        run.note_total(message.get("total-results"))
        raw = message.get("items", [])
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
        message: dict[str, Any],
        items: list[Any],
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
        message : dict[str, Any]
            Validated ``message`` object for the page just absorbed.
        items : list[Any]
            Raw entries that page carried.
        max_papers : int | None
            Caller-imposed cap.

        Returns
        -------
        str | None
            Cursor for the next request, or ``None`` to stop.
        """
        if max_papers is not None and len(run.papers) >= max_papers:
            return None

        raw_cursor = message.get("next-cursor")
        next_cursor = raw_cursor if isinstance(raw_cursor, str) and raw_cursor else None

        # Exhaustion, verified live: the page past the end carries an empty
        # ``items`` list and no ``next-cursor`` at all.
        if not items or next_cursor is None:
            if run.total is not None and run.processed < run.total:
                run.truncate(
                    f"CrossRef stopped returning results after {run.processed} of "
                    f"{run.total} reported records (cursor exhausted at page {run.pages})"
                )
            return None

        if next_cursor in run.seen_cursors:
            # CrossRef silently restarts pagination for a cursor it cannot
            # parse, so a repeat means the walk is going backwards rather than
            # forwards.  Continuing would re-read page one indefinitely.
            run.truncate(
                f"CrossRef returned a cursor already used at page {run.pages}, which is "
                f"how it responds to a cursor it cannot parse (an invalid cursor returns "
                f"HTTP 200 and the first page again). Stopped after {run.processed} of "
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
        degradation: str | None,
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
        degradation : str | None
            Note from
            :meth:`~lit_search.query.builders.crossref.CrossRefQueryBuilder.scope_degradation`
            describing scopes CrossRef narrowed.

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
        unparsed = run.processed - run.parsed
        if unparsed > 0:
            notes.append(f"{unparsed} of {run.processed} records could not be parsed into papers")
        if degradation:
            notes.append(degradation)
        shortfall = "; ".join(notes) or None

        # A narrowed scope is not a pagination shortfall and the counters
        # cannot see it: CrossRef's own total counts only the fields it
        # searched, so ``retrieved == total`` would read as COMPLETE while
        # every abstract-only match was unreachable.
        if run.truncated_early or run.total_drift or degradation:
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
