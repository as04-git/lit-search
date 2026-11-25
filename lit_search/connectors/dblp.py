"""dblp searcher implementation.

Ported from ``paper_search_mcp/academic_platforms/dblp.py`` at pinned commit
``234678ab231074a7977320978ee0496dcdaddd1f``.  The donor's control flow is not
reused, because it converts every failure into an empty or short list:

- ``except requests.RequestException`` / ``ET.ParseError`` / ``Exception`` each
  fall through to ``_search_html_fallback`` (``dblp.py:115-125``), whose own
  body ends ``except Exception: logger.error(...)`` then ``return papers``
  (``dblp.py:205-208``).  Every upstream outage therefore reaches the caller as
  a successful search that found nothing.
- ``if papers: return papers`` followed by the fallback (``dblp.py:109-113``)
  means a **genuine zero-result query** triggers the HTML scraper, so the one
  case that should be cheapest is the one that costs an extra request against a
  path ``dblp.org/robots.txt`` disallows.

  **Open policy question, raised 2026-08-31 and not resolved here.** That
  robots.txt directive is ``Disallow: /search`` under ``User-agent: *``, with
  no narrower ``Allow``.  Read as a prefix rule -- which is how robots.txt is
  read -- it covers ``/search/publ/api``, the JSON endpoint **this connector
  uses**, not only the HTML page the donor scraped.  dblp separately documents
  that endpoint for programmatic use and publishes a ``Crawl-delay`` that only
  makes sense for automated clients, so the intent plainly permits it; but the
  file as written does not say so, and the same reading is what gated IACR.
  Flagged for the owner to rule on rather than decided by this module.
- A three-attempt ``time.sleep`` retry loop lives inside the connector
  (``dblp.py:78-85``).  Pacing belongs to the shared cross-process limiter; a
  private sleep is invisible to it.
- Parse failures ``continue`` without being counted (``dblp.py:103-105``), so a
  schema change degrades recall with no trace.

The HTML fallback is dropped entirely rather than ported.

**Live API behaviour, verified 2026-08-31.**  Each of these was checked with a
known-good control request immediately before or after, because dblp's
overload responses are easy to mistake for content:

- **A genuine empty result is HTTP 200 with ``hits.@total == 0``** and no
  ``hit`` array.  It is cleanly distinguishable from a failure, which is what
  makes the two different statuses here.
- **Overload arrives as HTTP 429, 500 *and* 503, interchangeably, with HTML
  bodies.**  A burst of five unpaced requests produced ``200, 200, 200, 429``
  and then a dropped connection.  The 429 carried **no ``Retry-After``
  header** -- only ``content-length``, ``cache-control`` and ``content-type``
  -- which contradicts the project's rate-limit table for this row.  Nothing
  here may depend on ``Retry-After`` being present.
- **``h`` is documented as "capped at 1000" and behaves as capped at 100.**
  ``h=1000`` returns ``@computed: 100, @sent: 100``; ``h=200`` and ``h=1001``
  do the same.  ``@computed`` is pinned at ``f + 100`` for every ``h``.  The
  donor's ``'h': min(max_results, 1000)`` with the comment "dblp supports up to
  1000 results" is therefore wrong, and a port that trusts it and reads a short
  page as end-of-set stops at 100 of 268,367 without saying so.
- **High ``f`` is a hard ceiling.**  ``f=9900`` and ``f=10000`` both return
  HTTP 500 while an ``f=0`` control in the same batch returns 200.  Retrieval
  cannot pass that boundary, so reaching it is reported, never rounded off.

Rate limit: see :data:`_MIN_REQUEST_INTERVAL`.
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
from lit_search.query.builders.dblp import DBLPQueryBuilder
from lit_search.utils.normalization import normalize_doi, parse_date
from lit_search.utils.pagination_logging import log_pagination_request_failure

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://dblp.org/search/publ/api"

# Records requested per page.  dblp documents ``h`` as capped at 1000 but
# returns at most 100 whatever is asked (``@computed`` is pinned at ``f + 100``,
# verified 2026-08-31), so 100 is the real page size and asking for more only
# misleads the reader of this code.
_PAGE_SIZE = 100

# Hard ceiling on pagination rounds.  At 100 records per page this bounds a run
# at 100,000 records, well past the ``f`` ceiling dblp enforces itself; it
# exists so a pathological loop ends as an honest PARTIAL rather than running
# forever.
_MAX_PAGES = 1_000

_MIN_REQUEST_INTERVAL = 4.0
"""Seconds between requests.

**This is dblp's own published number, and it is not the one in the project's
rate-limit table.**  ``https://dblp.org/robots.txt`` (read 2026-08-31) carries
``Crawl-delay: 4`` for ``User-agent: *``.  The build plan's row for dblp says
``2.0s``, justified as "no number; docs say 'one or two seconds'; 429 carries
``Retry-After``" -- and the second half of that justification was falsified
live: the 429 observed on 2026-08-31 carried no ``Retry-After`` header at all.
A row whose stated evidence does not hold is not a safe source for the value,
so the published ``Crawl-delay`` is used instead.

Measured behaviour supports the stricter figure.  Five unpaced requests
produced three ``200``s, then ``429``, then a dropped connection; requests
spaced 6-8s apart still drew ``500``s while a penalty was active; 12-14s
spacing was uniformly clean.  dblp signals overload with 429, 500 **and** 503
and publishes no quota headers, so there is no feedback channel to pace
against -- which makes erring slow the only option that fails safely.

The asymmetry is the usual one: too slow costs minutes over a few hundred
papers, while too fast costs intermittent errors that then have to be
disambiguated from genuine emptiness.

Revisit if dblp publishes an API-specific rate, or changes ``Crawl-delay``.
"""


class DBLPBodyError(ConnectorError):
    """A response that is not a usable dblp search result.

    dblp answers an overloaded or malformed request with an HTML error page
    rather than JSON, and answers a query it could not parse with an HTTP 200
    whose body is structurally valid but carries no ``hits``.  Both would read
    as "dblp holds no matching paper" if the body were merely ``.get``-ed
    through, so both raise instead.
    """


@dataclass(slots=True)
class _Retrieval:
    """Mutable bookkeeping for one paginated dblp run.

    Keeping the stop reasons separable all the way to
    :meth:`DBLPConnector._build_coverage` is what stops a truncated run from
    reading as a confident answer.

    Attributes
    ----------
    papers : list[Paper]
        Records parsed so far, in retrieval order.
    processed : int
        Hit entries seen, whether or not they parsed.
    parsed : int
        Entries that became a :class:`~lit_search.core.paper.Paper`.
    pages : int
        Pages consumed.
    total : int | None
        ``hits.@total`` from the first page.
    total_drift : str | None
        Set when a later page reported a different ``@total``.
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

    def note_total(self, total: int | None) -> None:
        """Record the reported total, or the fact that it moved.

        Parameters
        ----------
        total : int | None
            ``hits.@total`` from a page.  ``None`` is ignored.
        """
        if total is None:
            return
        if self.total is None:
            self.total = total
        elif total != self.total and self.total_drift is None:
            self.total_drift = (
                f"dblp reported {self.total} matching records on the first page and "
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


def _as_int(value: object) -> int | None:
    """Coerce a dblp numeric attribute to ``int``.

    dblp returns every numeric attribute as a JSON **string**
    (``"@total": "268367"``), so the coercion is the normal path rather than a
    defensive one.

    Parameters
    ----------
    value : object
        Raw attribute value.

    Returns
    -------
    int | None
        Parsed integer, or ``None`` when the value is absent or unparsable.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def _as_list(value: object) -> list[Any]:
    """Return *value* as a list.

    dblp collapses a single-element array to the bare object, so ``hit`` and
    ``author`` are lists for multi-element results and dicts for one.

    Parameters
    ----------
    value : object
        Raw value from the payload.

    Returns
    -------
    list[Any]
        The value as a list; empty when it is absent.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


class DBLPConnector(SearchConnectorBase):
    """Connector for the dblp computer science bibliography.

    Uses the publication search API documented at
    https://dblp.org/faq/How+to+use+the+dblp+search+API.html

    dblp is a bibliographic index: it holds titles, authors, venues, years and
    electronic-edition links, and **no abstracts**.  Queries scoped to an
    abstract are answered against the metadata index and reported ``PARTIAL``;
    see :meth:`DBLPQueryBuilder.degradation_note`.
    """

    _credential_attrs: ClassVar[tuple[str, ...]] = ()
    """dblp issues no API key, so the quota is enforced against this host's
    source IP and every dblp process on the machine shares one clock.  The
    empty tuple is declared rather than inheriting the ``("_api_key",)``
    default, to state that the absence of a credential was checked rather than
    overlooked.
    """

    _retryable_status_codes: frozenset[int] = frozenset({429, 500, 502, 503, 504})
    """Retryable HTTP statuses, widened from the base class by adding **500**.

    Normally a 500 is a deterministic server fault and retrying it only wastes
    budget.  dblp is the exception, and it was measured: on 2026-08-31 a burst
    of unpaced requests drew ``429``, and requests issued while a throttling
    penalty was active drew ``500`` and ``503`` for queries that returned 200
    with content moments later.  Treating dblp's 500 as permanent would report
    its throttle as a database outage.

    The widening cannot manufacture a false success: the retry budget is finite
    and a genuinely broken request still ends in ``HTTPError`` -> ``FAILED``.
    """

    def __init__(self, query_builder: DBLPQueryBuilder | None = None) -> None:
        """Create a dblp searcher.

        Parameters
        ----------
        query_builder : DBLPQueryBuilder | None
            Builder used to validate and convert queries.  When ``None`` a
            default :class:`DBLPQueryBuilder` is created.
        """
        super().__init__()
        self._query_builder: DBLPQueryBuilder = query_builder or DBLPQueryBuilder()

    @property
    def name(self) -> str:
        """Return the database identifier.

        Returns
        -------
        str
            Database name.
        """
        return Database.DBLP.value

    @property
    def query_builder(self) -> QueryBuilder:
        """Return the dblp query builder.

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
            this diverges from the project's rate-limit table.
        """
        return _MIN_REQUEST_INTERVAL

    @property
    def rate_limit_service(self) -> str:
        """Return the quota authority these requests are paced against.

        Stated explicitly rather than inherited from :attr:`name` so that a
        future connector against another dblp endpoint -- the author or venue
        search APIs, or the ``/rec`` record endpoints -- returns the same
        string and shares this clock.  They are one host behind one
        ``Crawl-delay``; giving each its own scope would multiply the request
        rate by the number of endpoints.

        Returns
        -------
        str
            Quota-authority identifier.
        """
        return "dblp"

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
            The ``result`` object, guaranteed to carry a ``hits`` mapping.

        Raises
        ------
        DBLPBodyError
            When the body is not JSON (dblp serves HTML error pages), carries
            no ``result``, reports a non-200 ``status``, or carries no
            ``hits``.  Each of these would otherwise yield an empty hit list
            and be reported as "dblp holds no matching paper".
        """
        try:
            payload = response.json()
        except ValueError as exc:
            # dblp serves an HTML error page for overload and for out-of-range
            # parameters; both arrive here.
            raise DBLPBodyError(
                f"dblp returned a non-JSON body with status {response.status_code}. "
                f"It serves HTML error pages for overload (429/500/503) and for "
                f"out-of-range parameters, and neither is an empty result set: {exc}"
            ) from exc

        if not isinstance(payload, dict) or not isinstance(payload.get("result"), dict):
            raise DBLPBodyError(
                "dblp response carried no 'result' object, so it is not a search "
                "result. Treating it as an empty result set would report a failure "
                "as zero papers."
            )

        result: dict[str, Any] = payload["result"]
        status = result.get("status")
        if isinstance(status, dict):
            code = _as_int(status.get("@code"))
            if code is not None and code != 200:
                raise DBLPBodyError(
                    f"dblp reported status {code}: {status.get('text') or 'no message supplied'}"
                )

        if not isinstance(result.get("hits"), dict):
            raise DBLPBodyError(
                "dblp response carried no 'hits' object, so the number of matching "
                "records is unknown. An empty result reports 'hits' with '@total': "
                "'0', so absence of the field is a malformed response, not zero hits."
            )
        return result

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_authors(info: dict[str, Any]) -> list[Author]:
        """Extract authors from a dblp ``info`` object.

        Parameters
        ----------
        info : dict[str, Any]
            One ``hits.hit[].info`` entry.

        Returns
        -------
        list[Author]
            Parsed authors.  dblp carries no affiliations.
        """
        raw = info.get("authors")
        entries = _as_list(raw.get("author") if isinstance(raw, dict) else None)

        authors: list[Author] = []
        for entry in entries:
            if isinstance(entry, str):
                name = entry.strip()
            elif isinstance(entry, dict):
                name = str(entry.get("text") or "").strip()
            else:
                continue
            if name:
                authors.append(Author(name=name))
        return authors

    @staticmethod
    def _parse_doi(info: dict[str, Any]) -> str | None:
        """Extract a DOI from a dblp ``info`` object.

        dblp records the electronic edition in ``ee``, which is usually a
        ``doi.org`` URL but may be a publisher link with no DOI in it.

        Parameters
        ----------
        info : dict[str, Any]
            One ``hits.hit[].info`` entry.

        Returns
        -------
        str | None
            Normalised DOI, or ``None`` when the record carries none.
        """
        candidates = [str(value or "").strip() for value in _as_list(info.get("ee"))]
        candidates.append(str(info.get("doi") or "").strip())
        for candidate in candidates:
            if not candidate:
                continue
            if "doi.org/" in candidate:
                doi = normalize_doi(candidate.split("doi.org/", 1)[-1])
            elif candidate.startswith("10."):
                doi = normalize_doi(candidate)
            else:
                continue
            if doi:
                return doi
        return None

    @staticmethod
    def _parse_source(info: dict[str, Any]) -> Source | None:
        """Build a :class:`Source` from a dblp ``info`` object.

        Parameters
        ----------
        info : dict[str, Any]
            One ``hits.hit[].info`` entry.

        Returns
        -------
        Source | None
            The venue, or ``None`` when the record names none.
        """
        venue = ", ".join(
            part for part in (str(v or "").strip() for v in _as_list(info.get("venue"))) if part
        )
        if not venue:
            return None
        return Source(title=venue, source_type=_SOURCE_TYPES.get(str(info.get("type") or "")))

    def _parse_hit(self, hit: object) -> Paper | None:
        """Parse one dblp hit into a :class:`Paper`.

        Parameters
        ----------
        hit : object
            One ``hits.hit[]`` entry.

        Returns
        -------
        Paper | None
            Parsed paper, or ``None`` when a required field is missing or the
            record fails model validation.  Callers **must** count the ``None``
            returns: an unparsed record is a coverage loss, not an absent
            paper.
        """
        if not isinstance(hit, dict) or not isinstance(hit.get("info"), dict):
            return None
        info: dict[str, Any] = hit["info"]
        title = str(info.get("title") or "").strip()
        if not title:
            return None

        year = str(info.get("year") or "").strip()
        url = str(info.get("url") or "").strip() or None
        try:
            return Paper(
                title=title,
                # dblp is a bibliographic index and carries no abstracts at
                # all.  The empty string is the truth about this source, not a
                # parsing shortfall, and the query builder reports the
                # consequence as coverage.
                abstract="",
                authors=self._parse_authors(info),
                source=self._parse_source(info),
                publication_date=parse_date(f"{year}-01-01") if year.isdigit() else None,
                url=url,
                doi=self._parse_doi(info),
                found_in={self.name},
                paper_type=_PAPER_TYPES.get(str(info.get("type") or ""), PaperType.MISC),
            )
        except ValueError:
            logger.debug("dblp: record %s failed model validation.", info.get("key"))
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
        """Fetch papers from dblp with offset pagination and coverage.

        Every way this loop can end is tracked separately:

        - the caller's ``max_papers`` was reached -> ``CAPPED``;
        - a page request failed with nothing retrieved -> raised, so the base
          class reports ``FAILED``;
        - the same failure *after* records were retrieved -> ``PARTIAL``
          carrying the error, because the records that did arrive are real.
          dblp's ``f`` ceiling reaches this path, and is named in the detail;
        - a page came back empty before ``@total`` was reached -> ``PARTIAL``;
        - ``@total`` changed between pages -> ``PARTIAL``;
        - the query named a field dblp does not index -> ``PARTIAL``, however
          the pagination went;
        - the result set was exhausted -> ``from_pagination`` decides, and
          returns ``COMPLETE`` only when as many records came back as dblp said
          existed.

        ``since`` / ``until`` are applied **after** retrieval.  dblp's
        ``year:`` filter is a single year, not a range, so a range cannot be
        pushed into the query; filtering locally keeps the reported total
        honest, because ``@total`` describes the unfiltered set.

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
        ConnectorError
            When the first request returned an unusable body.
        requests.RequestException
            When the first request failed at the transport level.
        """
        expression = self._query_builder.convert_query(query)
        run = _Retrieval()
        first = 0

        while True:
            params = {
                "q": expression,
                "format": "json",
                "h": _PAGE_SIZE,
                "f": first,
            }

            try:
                result = self._read_body(self._get(_SEARCH_URL, params))
            except (requests.RequestException, ConnectorError) as exc:
                self._record_page_failure(run, exc, first)
                break

            sent = self._absorb_page(run, result)
            if progress_callback is not None:
                progress_callback(run.processed, run.total)

            if not self._should_continue(run, sent, first, max_papers):
                break
            first += _PAGE_SIZE

        if progress_callback is not None:
            progress_callback(run.processed, run.total)

        kept, date_filtered = _apply_date_filter(run.papers, since, until)
        if max_papers is not None:
            kept = kept[:max_papers]
        return CoveredPapers(
            papers=tuple(kept),
            coverage=self._build_coverage(
                run=run,
                retrieved=len(kept),
                date_filtered=date_filtered,
                max_papers=max_papers,
                degradation=self._query_builder.degradation_note(query),
            ),
        )

    @staticmethod
    def _record_page_failure(run: _Retrieval, exc: Exception, first: int) -> None:
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
        first : int
            Offset the failed request carried, for the message.

        Raises
        ------
        Exception
            The original failure, when nothing had been retrieved.
        """
        if not run.papers:
            raise exc
        log_pagination_request_failure(
            logger,
            connector_label="dblp",
            processed=run.processed,
            total=run.total,
            position_label="f",
            position=str(first),
            exc=exc,
        )
        run.truncate(
            f"page request at f={first} failed: {exc}. dblp refuses offsets beyond a "
            f"result-window ceiling with an HTTP 500 (f=9900 and f=10000 both observed "
            f"failing against an f=0 control on 2026-08-31), so a run that stops here "
            f"has reached a boundary rather than the end of the result set",
            type(exc).__name__,
        )

    def _absorb_page(self, run: _Retrieval, result: dict[str, Any]) -> int:
        """Fold one page into *run* and return how many entries it carried.

        Parameters
        ----------
        run : _Retrieval
            Bookkeeping for the current run, updated in place.
        result : dict[str, Any]
            Validated ``result`` object.

        Returns
        -------
        int
            Number of hit entries on this page, including any that failed to
            parse -- the caller needs the count, because a record seen but not
            parsed is a coverage loss rather than an absent paper.
        """
        hits: dict[str, Any] = result["hits"]
        run.note_total(_as_int(hits.get("@total")))
        entries = _as_list(hits.get("hit"))
        for entry in entries:
            paper = self._parse_hit(entry)
            if paper is not None:
                run.papers.append(paper)
                run.parsed += 1
        run.processed += len(entries)
        run.pages += 1
        return len(entries)

    @staticmethod
    def _should_continue(
        run: _Retrieval,
        sent: int,
        first: int,
        max_papers: int | None,
    ) -> bool:
        """Decide whether to request another page, recording any truncation.

        Note what is deliberately **not** here: a "short page means the end"
        test.  dblp caps ``@sent`` at 100 however large ``h`` is, so every full
        page of a large result set looks short against the requested size.

        Parameters
        ----------
        run : _Retrieval
            Bookkeeping for the current run, updated in place.
        sent : int
            Entries on the page just absorbed.
        first : int
            Offset that produced this page.
        max_papers : int | None
            Caller-imposed cap.

        Returns
        -------
        bool
            ``True`` to request another page.
        """
        if max_papers is not None and len(run.papers) >= max_papers:
            return False

        if sent == 0:
            # dblp stopped supplying records.  If it also said more existed,
            # that is a shortfall and must not be read as exhaustion.
            if run.total is not None and run.processed < run.total:
                run.truncate(
                    f"dblp returned an empty page at f={first} after {run.processed} of "
                    f"{run.total} reported records, so retrieval stopped before the "
                    f"result set was exhausted"
                )
            return False

        if run.total is not None and run.processed >= run.total:
            return False

        if run.pages >= _MAX_PAGES:
            run.truncate(
                f"pagination stopped at the connector's {_MAX_PAGES}-page safety limit "
                f"after {run.processed} records"
            )
            return False
        return True

    def _build_coverage(
        self,
        *,
        run: _Retrieval,
        retrieved: int,
        date_filtered: int,
        max_papers: int | None,
        degradation: str | None,
    ) -> SourceCoverage:
        """Turn the pagination loop's own counters into a coverage status.

        Parameters
        ----------
        run : _Retrieval
            Bookkeeping for the finished run.
        retrieved : int
            Papers actually returned, after date filtering and the cap.
        date_filtered : int
            Papers dropped by the local ``since`` / ``until`` filter.
        max_papers : int | None
            Caller-imposed cap.
        degradation : str | None
            How far short of the query dblp's index falls, if at all.

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
        if degradation:
            notes.append(degradation)
        unparsed = run.processed - run.parsed
        if unparsed > 0:
            notes.append(f"{unparsed} of {run.processed} records could not be parsed into papers")
        shortfall = "; ".join(notes) or None

        # A locally applied date filter means ``retrieved`` no longer relates
        # to ``@total``, so completeness is not arithmetically checkable and
        # from_pagination must not be handed the mismatch to interpret.
        if run.truncated_early or run.total_drift or degradation or date_filtered:
            detail = shortfall or "retrieval stopped before the result set was exhausted"
            if date_filtered:
                detail = (
                    f"{date_filtered} of {run.parsed} retrieved papers were removed by the "
                    f"requested date range after retrieval, because dblp's 'year:' filter "
                    f"takes a single year and cannot express a range"
                    + (f"; {detail}" if shortfall else "")
                )
            return SourceCoverage.partial(
                self.name,
                retrieved,
                detail=detail,
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


# dblp record types, mapped onto the BibTeX-aligned classification.  Unlisted
# types fall back to MISC rather than being guessed into a category.
_PAPER_TYPES: dict[str, PaperType] = {
    "Journal Articles": PaperType.ARTICLE,
    "Conference and Workshop Papers": PaperType.INPROCEEDINGS,
    "Parts in Books or Collections": PaperType.INCOLLECTION,
    "Books and Theses": PaperType.BOOK,
    "Editorship": PaperType.BOOK,
    "Informal and Other Publications": PaperType.UNPUBLISHED,
    "Reference Works": PaperType.MISC,
    "Data and Artifacts": PaperType.MISC,
}

# Venue classification for the same record types.
_SOURCE_TYPES: dict[str, SourceType] = {
    "Journal Articles": SourceType.JOURNAL,
    "Conference and Workshop Papers": SourceType.CONFERENCE,
    "Parts in Books or Collections": SourceType.BOOK,
    "Books and Theses": SourceType.BOOK,
    "Editorship": SourceType.BOOK,
    "Informal and Other Publications": SourceType.REPOSITORY,
}


def _apply_date_filter(
    papers: list[Paper],
    since: datetime.date | None,
    until: datetime.date | None,
) -> tuple[list[Paper], int]:
    """Apply the caller's date range locally, counting what it removed.

    dblp records carry a year and no finer date, so a paper is kept whenever
    its year could fall inside the range.  A paper with **no** year is kept:
    absence of a date is not evidence that the paper falls outside the range,
    and dropping it would silently narrow recall.

    Parameters
    ----------
    papers : list[Paper]
        Papers retrieved, in order.
    since : datetime.date | None
        Lower bound.
    until : datetime.date | None
        Upper bound.

    Returns
    -------
    tuple[list[Paper], int]
        The kept papers and the number removed.
    """
    if since is None and until is None:
        return papers, 0
    kept = [
        paper
        for paper in papers
        if paper.publication_date is None
        or (
            (since is None or paper.publication_date.year >= since.year)
            and (until is None or paper.publication_date.year <= until.year)
        )
    ]
    return kept, len(papers) - len(kept)
