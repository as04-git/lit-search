"""bioRxiv and medRxiv connectors, driven by date window rather than keyword.

**This is new code, not a port, and the donor is the reason.**
``paper_search_mcp/academic_platforms/biorxiv.py`` exposes ``search(query,
max_results, days=30)``, sets ``category = query.lower().replace(' ', '_')``,
and consults only the last thirty days.  Two silent failures compound there:

1. A boolean query becomes a category string that matches nothing, so an
   ordinary search returns an empty list.
2. Even a correct category is asked only of the last thirty days, and nothing
   in the return value says so.

The result is a confident empty set from a server that holds thousands of
matching preprints, which is the motivating failure of this whole project with
a different hostname on it.  ``medrxiv.py`` in the donor is the same file with
the server name changed.

**What is actually there, verified live 2026-08-31.**

===============================================================  =============================
Request                                                          Response
===============================================================  =============================
``/details/biorxiv/2026-08-01/2026-08-02/0``                     HTTP 200, 30 records, ``total``
``/details/biorxiv/2026-08-01/2026-08-01/30``                    HTTP 200, next 30
``/details/biorxiv/2026-08-01/2026-08-01/900``                   HTTP 200, ``"no posts found"``
``/details/biorxiv/NOTADATE/2026-08-02/0``                       HTTP 200, ``"Both dates must be in yyyy-mm-dd format"``
``/details/notaserver/2026-08-01/2026-08-01/0``                  HTTP 200, ``"Server not recognized"``
===============================================================  =============================

**Every failure arrives as HTTP 200 with an empty ``collection``.**
``raise_for_status`` is silent and ``payload.get("collection", [])`` returns
``[]``, so a naive port reports each of those four rows as "bioRxiv holds no
matching paper".  :meth:`_XRxivConnector._read_body` reads the ``messages``
status instead and raises for everything that is not ``ok`` or the one
end-of-pagination marker.

So the API *is* real, it just is not a search: it serves a date window,
paginated 30 records at a time (its own documentation says so), and the query
has to be applied here.  That gives a working connector and a permanent
coverage consequence:

**A result from this connector is never ``COMPLETE``.**  ``total`` is the
number of postings in the *window*, not the number matching the query, and the
server is never told the query.  Passing ``total`` as a ``reported_total``
would let arithmetic that happens to balance stand in for a claim nothing
supports.  :meth:`_XRxivConnector._build_coverage` tops out at ``PARTIAL``, and
the detail always names the window that was consulted -- so the thirty-day
horizon the donor hid is now the first thing a reader of the ledger sees.

**Two useful things the donor discards and this connector keeps.**

- ``published`` carries the DOI of the version of record once a preprint is
  published, or the literal string ``"NA"``.  Observed on 2026-08-31:
  ``10.1101/2023.09.16.558066`` -> ``10.1098/rstb.2022.0467``.  That is
  ``D24``'s preprint-to-published link, handed over by the source, so both
  identities are kept: the published DOI in
  :attr:`~lit_search.core.paper.Paper.doi` and the preprint DOI in
  :attr:`~lit_search.core.paper.Paper.preprint_doi`.
- ``version`` says which revision the record describes, and a date window
  returns a posting *event*, so the same paper legitimately appears twice
  across two windows at two versions.

**A note on DOI prefixes, because it broke something else.**  bioRxiv and
medRxiv issued ``10.1101/...`` historically and issue ``10.64898/...`` for
records posted in 2026 (both observed live on 2026-08-31).  The project's
preprint-DOI classifier knew only the first, so every recent preprint from
these servers was being classified as a version of record.  Fixed in
:data:`~lit_search.utils.normalization.PREPRINT_DOI_PREFIXES`.
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
from lit_search.core.coverage import CoveredPapers, ExecutionStatus, SourceCoverage
from lit_search.core.paper import Database, Paper, PaperType
from lit_search.core.query import Query
from lit_search.core.source import Source, SourceType
from lit_search.exceptions import ConnectorError
from lit_search.query.builders.biorxiv import BioRxivQueryBuilder
from lit_search.utils.normalization import parse_date
from lit_search.utils.pagination_logging import log_pagination_request_failure

logger = logging.getLogger(__name__)

#: Records served per response.  Fixed by the API and stated in its own
#: documentation: "results are paginated with 30 papers served in a call"
#: (read from ``https://api.biorxiv.org/`` on 2026-08-31).
_PAGE_SIZE = 30

#: The ``messages[].status`` value meaning the request succeeded.
_OK_STATUS = "ok"

#: The status meaning "this window holds no more postings".  It is returned
#: both for a cursor past the end and for a genuinely empty window, so it is
#: the pagination terminator and **not** a failure.
_EMPTY_STATUS = "no posts found"

#: Default window when the caller supplies no lower bound, in days.
#:
#: Thirty is the donor's own number, kept deliberately.  The donor's defect was
#: never the size of the window -- it was that the window existed and nothing
#: said so.  Keeping the number and reporting it makes the two behaviours
#: directly comparable, and a caller who wants more passes ``since``.
_DEFAULT_WINDOW_DAYS = 30

#: Ceiling on pages per search, so a window that keeps serving records ends as
#: an honest ``PARTIAL`` rather than running for hours.  At 30 records a page
#: this is 30,000 postings, roughly six months of bioRxiv.
_MAX_PAGES = 1_000

#: Literal the API uses in ``published`` when a preprint has no version of
#: record yet.  It is the string "NA", not null, so a truthiness test on the
#: field would treat every unpublished preprint as published.
_NOT_PUBLISHED = "na"

_MIN_REQUEST_INTERVAL = 1.0
"""Seconds between requests.

**This value is inferred, not documented, and the nearest documented number is
deliberately not the one used.**  Checked 2026-08-31:

- ``api.biorxiv.org`` publishes no ``robots.txt`` at all (HTTP 404) and its
  responses carry no ``X-RateLimit-*`` or ``Retry-After`` header.
- The API's own documentation page describes pagination and says nothing about
  a request rate.
- ``www.biorxiv.org/robots.txt`` **does** publish ``Crawl-delay: 7`` for
  ``User-agent: *``.  That governs the website, not this API host, and applying
  a website crawl-delay to a documented bulk API would make a thirty-day
  harvest take twenty minutes instead of three.  It is recorded here so that a
  future connector against ``www.biorxiv.org`` uses **7.0** rather than
  inheriting this number.

One second is the conservative choice among the intervals this project already
uses for unrated JSON APIs.  Revisit it if a rate is ever published or a 429 is
ever observed.
"""


class XRxivBodyError(ConnectorError):
    """A response body that is not a usable bioRxiv/medRxiv result set.

    Raised rather than defaulted away.  Every one of this API's error
    conditions arrives as HTTP 200 with ``"collection": []``, so returning the
    empty collection would report a rejected request as a server holding no
    matching paper.

    Parameters
    ----------
    message : str
        Assembled message.
    status : str | None
        The ``messages[].status`` value the server sent, when it sent one.
    """

    def __init__(self, message: str, *, status: str | None = None) -> None:
        """Store the message and the server's own status string."""
        super().__init__(message)
        self.status = status


@dataclass(slots=True)
class _Harvest:
    """Mutable bookkeeping for one date-window harvest.

    Attributes
    ----------
    papers : list[Paper]
        Records that parsed **and** matched the query, in harvest order.
    processed : int
        Postings seen, whether or not they parsed or matched.
    parsed : int
        Postings that became a :class:`~lit_search.core.paper.Paper`.
    matched : int
        Parsed postings that satisfied the query predicate.
    pages : int
        Responses consumed.
    window_total : int | None
        ``total`` from the first response: how many postings the server says
        the *window* holds.  Never the number matching the query.
    exhausted : bool
        Whether pagination reached the end of the window.
    truncated_early : bool
        Whether the harvest ended for a reason other than the cap or
        exhaustion.
    error_detail : str | None
        Description of the truncation.
    error_type : str | None
        Exception class name, when the truncation came from an exception.
    """

    papers: list[Paper] = field(default_factory=list)
    processed: int = 0
    parsed: int = 0
    matched: int = 0
    pages: int = 0
    window_total: int | None = None
    exhausted: bool = False
    truncated_early: bool = False
    error_detail: str | None = None
    error_type: str | None = None

    def truncate(self, detail: str, error_type: str | None = None) -> None:
        """Mark the harvest as having stopped short, keeping the first reason.

        Parameters
        ----------
        detail : str
            Why the harvest stopped.
        error_type : str | None
            Exception class name, when an exception caused it.
        """
        self.truncated_early = True
        if self.error_detail is None:
            self.error_detail = detail
            self.error_type = error_type


@dataclass(frozen=True, slots=True)
class _Window:
    """The date window one harvest consults.

    Attributes
    ----------
    since : datetime.date
        Lower bound, inclusive.
    until : datetime.date
        Upper bound, inclusive.  The API requires both bounds, so an
        open-ended request is closed at today rather than left unbounded.
    caller_supplied_since : bool
        Whether *since* came from the caller or was defaulted.
    """

    since: datetime.date
    until: datetime.date
    caller_supplied_since: bool

    def describe(self) -> str:
        """Return a human-readable description of the window.

        Returns
        -------
        str
            Description naming the bounds and how *since* was chosen.
        """
        span = f"postings dated {self.since.isoformat()} to {self.until.isoformat()}"
        if self.caller_supplied_since:
            return span
        return (
            f"{span} -- the connector's default {_DEFAULT_WINDOW_DAYS}-day window, because "
            f"the search supplied no start date and this API has no way to be asked about "
            f"the whole archive at once"
        )


def match_record(record: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    """Return one raw API record as fields the local predicate matches against.

    Parameters
    ----------
    record : dict[str, Any]
        One ``collection`` entry.

    Returns
    -------
    dict[str, tuple[str, ...]]
        Field name to values, matching
        :data:`~lit_search.query.builders.biorxiv.FIELD_MAP`.
    """
    return {
        name: (str(record.get(name) or "").strip(),)
        for name in ("title", "abstract", "authors", "category", "type")
    }


class _XRxivConnector(SearchConnectorBase):
    """Shared implementation for the two Cold Spring Harbor preprint servers.

    bioRxiv and medRxiv run the same software behind the same request grammar;
    only the hostname and the server token differ.  Subclasses supply those
    through :attr:`SERVER` and :attr:`API_HOST` and change nothing else.

    Parameters
    ----------
    query_builder : BioRxivQueryBuilder | None
        Builder used to validate queries and compile the local predicate.
    """

    #: Server token in the request path, and the value of ``server`` in
    #: responses.  Override.
    SERVER: ClassVar[str] = ""

    #: API hostname.  Override.
    API_HOST: ClassVar[str] = ""

    #: Human-readable name used in log lines.  Override.
    LABEL: ClassVar[str] = ""

    _credential_attrs: ClassVar[tuple[str, ...]] = ()
    """Neither server issues an API key and neither route accepts one.
    Declared explicitly so the empty tuple reads as checked.
    """

    def __init__(self, query_builder: BioRxivQueryBuilder | None = None) -> None:
        """Create a preprint-server searcher.

        Parameters
        ----------
        query_builder : BioRxivQueryBuilder | None
            Builder used to validate queries and compile the local predicate.
        """
        super().__init__()
        self._query_builder: BioRxivQueryBuilder = query_builder or BioRxivQueryBuilder()

    @property
    def query_builder(self) -> QueryBuilder:
        """Return the query builder.

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
            Interval in seconds.  See :data:`_MIN_REQUEST_INTERVAL`, which
            records that the value is inferred and why the website's published
            ``Crawl-delay: 7`` was not used here.
        """
        return _MIN_REQUEST_INTERVAL

    @property
    def rate_limit_service(self) -> str:
        """Return the quota authority these requests are paced against.

        Both servers return ``"biorxiv"``, so bioRxiv and medRxiv share one
        clock.  They are one operator running one API stack, published on two
        hostnames, and neither states a rate; two independent buckets would
        double the request rate against whatever the real shared limit is.
        Sharing costs a slower harvest when both are searched together, which
        is the safe direction of error.

        Returns
        -------
        str
            Quota-authority identifier.
        """
        return Database.BIORXIV.value

    # ------------------------------------------------------------------
    # Response handling
    # ------------------------------------------------------------------

    def _read_body(self, response: requests.Response) -> dict[str, Any]:
        """Decode one response and reject every HTTP-200 error form.

        Parameters
        ----------
        response : requests.Response
            Response from the details endpoint.

        Returns
        -------
        dict[str, Any]
            Decoded envelope, guaranteed to carry a ``collection`` list and a
            status of either :data:`_OK_STATUS` or :data:`_EMPTY_STATUS`.

        Raises
        ------
        XRxivBodyError
            When the body is not JSON, is not an object, carries no
            ``collection`` list, or reports a status other than the two
            recognised ones.  Each of those otherwise reads as "this server
            holds no matching paper".
        """
        try:
            payload = response.json()
        except ValueError as exc:
            raise XRxivBodyError(
                f"{self.LABEL} returned a non-JSON body with status {response.status_code}: {exc}"
            ) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("collection"), list):
            raise XRxivBodyError(
                f"{self.LABEL} returned a body carrying no 'collection' list, so it is not "
                f"a result set. Treating it as an empty result would report a rejected "
                f"request as a server holding no matching paper."
            )
        status = _status_of(payload)
        if status not in (_OK_STATUS, _EMPTY_STATUS):
            raise XRxivBodyError(
                f"{self.LABEL} rejected the request with HTTP 200 and status "
                f"'{status or 'no status supplied'}'. The response carried an empty "
                f"collection, which is indistinguishable from an empty result set unless "
                f"the status is read.",
                status=status,
            )
        return payload

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_record(self, record: Any) -> Paper | None:
        """Parse one posting into a :class:`Paper`, keeping both DOIs.

        Parameters
        ----------
        record : Any
            One ``collection`` entry.

        Returns
        -------
        Paper | None
            Parsed paper, or ``None`` when a required field is missing.
            Callers **must** count the ``None`` returns: an unparsed record is
            a coverage loss, not an absent paper.
        """
        if not isinstance(record, dict):
            return None
        title = str(record.get("title") or "").strip()
        preprint_doi = str(record.get("doi") or "").strip() or None
        if not title:
            return None

        published = str(record.get("published") or "").strip()
        publisher_doi = published if published and published.lower() != _NOT_PUBLISHED else None
        version = str(record.get("version") or "").strip()

        try:
            return Paper(
                title=title,
                abstract=str(record.get("abstract") or "").strip(),
                authors=_parse_authors(record.get("authors")),
                source=Source(title=self.LABEL, source_type=SourceType.REPOSITORY),
                publication_date=parse_date(str(record.get("date") or "").strip() or None),
                url=f"https://doi.org/{preprint_doi}" if preprint_doi else None,
                # D24: the published record is the citable one; the preprint
                # identity is kept beside it rather than instead of it.
                doi=publisher_doi or preprint_doi,
                preprint_doi=preprint_doi,
                keywords={str(record.get("category")).strip()} if record.get("category") else None,
                comments=f"{self.LABEL} version {version}" if version else None,
                found_in={self.name},
                paper_type=(PaperType.ARTICLE if publisher_doi else PaperType.UNPUBLISHED),
                is_open_access=True,
            )
        except ValueError:
            logger.debug("%s: record %s failed model validation.", self.LABEL, preprint_doi)
            return None

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
        """Harvest a date window and apply the query locally.

        Parameters
        ----------
        query : Query
            Validated query object.
        max_papers : int | None
            Maximum papers to retrieve.  ``None`` means unlimited.
        progress_callback : Callable[[int, int | None], None] | None
            Progress callback called as ``callback(processed, window_total)``.
        since : datetime.date | None
            Lower bound of the harvest window.  Defaulted when absent, and the
            default is reported.
        until : datetime.date | None
            Upper bound.  Closed at today when absent, because the API
            requires both bounds.

        Returns
        -------
        CoveredPapers
            Matching papers and the coverage status describing them.  Never
            ``COMPLETE``; see the module docstring.

        Raises
        ------
        ConnectorError
            When the first request returned an unusable body.
        requests.RequestException
            When the first request failed at the transport level.
        """
        window = _resolve_window(since, until)
        matches = self._query_builder.compile_predicate(query)
        run = _Harvest()
        cursor = 0

        while True:
            url = (
                f"https://{self.API_HOST}/details/{self.SERVER}/"
                f"{window.since.isoformat()}/{window.until.isoformat()}/{cursor}"
            )
            try:
                payload = self._read_body(self._get(url))
            except (requests.RequestException, ConnectorError) as exc:
                self._record_page_failure(run, exc, cursor)
                break

            records = payload["collection"]
            if not records:
                # The API reports both "cursor past the end" and "empty window"
                # with the same marker, and both mean the same thing here.
                run.exhausted = True
                run.pages += 1
                break

            self._absorb_page(run, payload, records, matches, max_papers)
            if progress_callback is not None:
                progress_callback(run.processed, run.window_total)

            if not self._should_continue(run, max_papers):
                break
            cursor += len(records)

        if progress_callback is not None:
            progress_callback(run.processed, run.window_total)

        kept = run.papers[:max_papers] if max_papers is not None else run.papers
        return CoveredPapers(
            papers=tuple(kept),
            coverage=self._build_coverage(
                run=run, retrieved=len(kept), window=window, max_papers=max_papers
            ),
        )

    def _record_page_failure(self, run: _Harvest, exc: Exception, cursor: int) -> None:
        """Record a page failure, or re-raise when nothing was retrieved.

        With no records in hand there is no partial result to report and the
        failure is the whole story, so it is re-raised for
        :meth:`~lit_search.connectors.search_base.SearchConnectorBase.search`
        to turn into ``FAILED``.

        Parameters
        ----------
        run : _Harvest
            Bookkeeping for the current harvest.
        exc : Exception
            The failure raised while fetching or reading the page.
        cursor : int
            Cursor the failed request used.

        Raises
        ------
        Exception
            The original failure, when nothing had been retrieved.
        """
        if not run.papers:
            raise exc
        log_pagination_request_failure(
            logger,
            connector_label=self.LABEL,
            processed=run.processed,
            total=run.window_total,
            position_label="cursor",
            position=str(cursor),
            exc=exc,
        )
        run.truncate(f"page request at cursor={cursor} failed: {exc}", type(exc).__name__)

    def _absorb_page(
        self,
        run: _Harvest,
        payload: dict[str, Any],
        records: list[Any],
        matches: Callable[[dict[str, tuple[str, ...]]], bool],
        max_papers: int | None,
    ) -> None:
        """Fold one response into *run*, keeping the records that match.

        Parameters
        ----------
        run : _Harvest
            Bookkeeping, updated in place.
        payload : dict[str, Any]
            Validated response envelope.
        records : list[Any]
            Its ``collection`` entries.
        matches : Callable[[dict[str, tuple[str, ...]]], bool]
            Compiled query predicate.
        max_papers : int | None
            Caller-imposed cap.
        """
        if run.window_total is None:
            run.window_total = _window_total(payload)
        for record in records:
            run.processed += 1
            paper = self._parse_record(record)
            if paper is None:
                continue
            run.parsed += 1
            if not isinstance(record, dict) or not matches(match_record(record)):
                continue
            run.matched += 1
            if max_papers is None or len(run.papers) < max_papers:
                run.papers.append(paper)
        run.pages += 1

    @staticmethod
    def _should_continue(run: _Harvest, max_papers: int | None) -> bool:
        """Whether another page should be requested.

        Parameters
        ----------
        run : _Harvest
            Bookkeeping, updated in place.
        max_papers : int | None
            Caller-imposed cap.

        Returns
        -------
        bool
            ``True`` to request another page.
        """
        if max_papers is not None and len(run.papers) >= max_papers:
            return False
        if run.window_total is not None and run.processed >= run.window_total:
            run.exhausted = True
            return False
        if run.pages >= _MAX_PAGES:
            run.truncate(
                f"harvest stopped at the connector's {_MAX_PAGES}-page safety limit after "
                f"{run.processed} postings, before the window was exhausted"
            )
            return False
        return True

    def _build_coverage(
        self,
        *,
        run: _Harvest,
        retrieved: int,
        window: _Window,
        max_papers: int | None,
    ) -> SourceCoverage:
        """Describe what the harvest achieved, never claiming completeness.

        ``from_pagination`` is deliberately not used.  It would return
        ``COMPLETE`` whenever ``retrieved == reported_total``, and the only
        total this API offers counts postings in a date window rather than
        matches for a query -- the server was never told the query.  The window
        total is still carried, so the reader loses no information, only the
        unjustified claim.

        The status is not ``CAPPED`` when the caller's limit bound either.
        ``CAPPED`` ranks as milder than ``PARTIAL`` because it means a known
        amount is missing on purpose; here an unknown amount is missing
        whatever the cap did, because everything outside the window was never
        consulted.

        Parameters
        ----------
        run : _Harvest
            Bookkeeping for the finished harvest.
        retrieved : int
            Papers actually returned, after the cap.
        window : _Window
            The window that was consulted.
        max_papers : int | None
            Caller-imposed cap.

        Returns
        -------
        SourceCoverage
            A ``PARTIAL`` record describing the window and the local match.
        """
        notes: list[str] = [
            f"{self.LABEL} has no keyword search, so this consulted {window.describe()} and "
            f"applied the query locally; a matching preprint posted outside that window was "
            f"never examined",
            f"harvested {run.processed} postings"
            + (
                f" of the {run.window_total} the server reports for that window"
                if run.window_total is not None
                else ""
            )
            + f" and matched {run.matched}",
        ]
        if max_papers is not None and run.matched > retrieved:
            notes.append(
                f"the caller's limit of {max_papers} stopped the harvest, so {run.matched} "
                f"postings had already matched within the window when it ended"
            )
        if run.error_detail:
            notes.append(run.error_detail)
        if not run.exhausted and not run.truncated_early:
            notes.append("pagination did not reach the end of the window")
        unparsed = run.processed - run.parsed
        if unparsed > 0:
            notes.append(f"{unparsed} of {run.processed} postings could not be parsed into papers")

        return SourceCoverage.partial(
            self.name,
            retrieved,
            detail="; ".join(notes),
            error=run.error_detail,
            error_type=run.error_type,
            execution_status=(
                ExecutionStatus.COMPLETE
                if (
                    run.error_detail is None
                    and not run.truncated_early
                    and unparsed == 0
                    and (run.exhausted or (max_papers is not None and retrieved >= max_papers))
                )
                else ExecutionStatus.INCOMPLETE
            ),
        )


class BioRxivConnector(_XRxivConnector):
    """Connector for bioRxiv."""

    SERVER = "biorxiv"
    API_HOST = "api.biorxiv.org"
    LABEL = "bioRxiv"

    @property
    def name(self) -> str:
        """Return the database identifier.

        Returns
        -------
        str
            Database name.
        """
        return Database.BIORXIV.value


class MedRxivConnector(_XRxivConnector):
    """Connector for medRxiv."""

    SERVER = "medrxiv"
    API_HOST = "api.medrxiv.org"
    LABEL = "medRxiv"

    @property
    def name(self) -> str:
        """Return the database identifier.

        Returns
        -------
        str
            Database name.
        """
        return Database.MEDRXIV.value


def _status_of(payload: dict[str, Any]) -> str | None:
    """Return the ``messages[0].status`` string, if there is one.

    Parameters
    ----------
    payload : dict[str, Any]
        Decoded response body.

    Returns
    -------
    str | None
        Lower-cased status, or ``None`` when the body carried no message.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    first = messages[0]
    if not isinstance(first, dict):
        return None
    status = first.get("status")
    return str(status).strip().lower() if status is not None else None


def _window_total(payload: dict[str, Any]) -> int | None:
    """Return the window's posting count from a response envelope.

    The field arrives as an ``int`` on some responses and as a decimal
    ``str`` on others -- both observed on 2026-08-31, from the same endpoint,
    minutes apart -- so both are accepted and anything else is ignored rather
    than coerced.

    Parameters
    ----------
    payload : dict[str, Any]
        Decoded response body.

    Returns
    -------
    int | None
        Window total, or ``None`` when absent or unreadable.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages or not isinstance(messages[0], dict):
        return None
    raw = messages[0].get("total")
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.strip().isdigit():
        return int(raw.strip())
    return None


def _parse_authors(raw: Any) -> list[Author]:
    """Split the semicolon-separated author string into authors.

    Parameters
    ----------
    raw : Any
        The record's ``authors`` value, formatted ``"Last, F.; Last, F."``.

    Returns
    -------
    list[Author]
        Parsed authors; empty when the record names none.
    """
    text = str(raw or "").strip()
    if not text:
        return []
    return [Author(name=part.strip()) for part in text.split(";") if part.strip()]


def _resolve_window(
    since: datetime.date | None,
    until: datetime.date | None,
) -> _Window:
    """Choose the harvest window, defaulting whichever bound is absent.

    Parameters
    ----------
    since : datetime.date | None
        Caller's lower bound.
    until : datetime.date | None
        Caller's upper bound.

    Returns
    -------
    _Window
        The window to harvest, remembering whether *since* was defaulted.
    """
    high = until or datetime.date.today()
    if since is not None:
        return _Window(since=since, until=high, caller_supplied_since=True)
    return _Window(
        since=high - datetime.timedelta(days=_DEFAULT_WINDOW_DAYS),
        until=high,
        caller_supplied_since=False,
    )
