"""SSRN searcher implementation.

Ported from ``paper_search_mcp/academic_platforms/ssrn.py`` at pinned commit
``234678ab231074a7977320978ee0496dcdaddd1f``.  Almost nothing of the donor
survives, and the measurements below are why.

**What the donor does.**  It scrapes SSRN's HTML search results and loops::

    html, err = self._fetch_page(query, page)
    if err or not html:
        logger.warning("SSRN search page %d fetch failed: %s", page, err)
        break
    page_papers = self._parse_results(html)
    if not page_papers:
        break
    ...
    return papers[:max_results]

Three different outcomes leave that loop by the same door and arrive as the
same value: a fetch failure, a markup change that matches no selector, and a
genuinely empty search.  ``_fetch_page`` even recognises the Cloudflare
challenge by name (``"just a moment" in body``) and then hands the recognition
back as a string that ``search`` discards.  So the one connector in the donor
tree that *knows* it was blocked still reports the block as "SSRN holds no
matching paper".

**What is actually there now, measured 2026-08-31.**

===============================================================  =========================================
Endpoint                                                          Result
===============================================================  =========================================
``www.ssrn.com/index.cfm/en/rps-stage1-results/`` (donor primary)  403 Cloudflare challenge; 404 once past it
``papers.ssrn.com/sol3/results.cfm`` (donor fallback)              403 Cloudflare challenge
``api.ssrn.com/content/v1/search`` (free-text search)              401 ``{"message":"Unauthorized"}``
``api.ssrn.com/content/v1/bindings/{id}/papers`` (browse)          200 ``{"total": N, "papers": [...]}``
===============================================================  =========================================

The donor's primary URL is simply gone.  The fallback is behind Cloudflare for
an ordinary client *and* for a browser-impersonating one.  The only reachable
JSON route is a browse route, not a search route: adding ``&keyword=catalysis``
to it returned the same ``"total": 59666`` as the unfiltered request, so it
accepts the parameter and ignores it.

Two consequences shape this connector, and both are deliberate:

- **It reports ``SKIPPED_NO_CREDENTIAL`` by default.**  SSRN's search route
  needs a credential nobody here has.  The specification already uses this
  status for an access precondition that is not an API key -- BASE until its IP
  is registered, IACR until it is whitelisted -- and this is the same shape.
  What matters is that a default search now says "ssrn: skipped, supply
  ssrn_api_key" instead of omitting SSRN entirely, which is the registry-level
  version of the same silence.
- **It can never report ``COMPLETE``.**  The specification classifies SSRN as
  degraded and requires an explicit degraded label, and the reason is concrete
  here: the response shape this module parses was observed on the *browse*
  route, never on the search route, so nothing establishes that the search
  route's ``total`` is a count of matches rather than a capped figure.  A
  ``COMPLETE`` against an unobserved total is exactly the unfalsifiable
  completeness claim the reviews found four ways to fake.
  :meth:`SSRNConnector._build_coverage` therefore tops out at ``PARTIAL``,
  carrying the cap and the total when they are known.

**Two things in here are unverified, and they are named rather than hidden.**

1. The authentication header (:data:`_API_KEY_HEADER`).  With no key to test
   with, the name is a guess.  It fails *loudly*: a wrong header produces the
   same 401 the anonymous request produces, which reaches the ledger as
   ``FAILED`` with SSRN's own ``Unauthorized`` message in it.
2. Whether the search route returns the same envelope as the browse route.
   :meth:`SSRNConnector._read_body` refuses anything else rather than
   defaulting it to an empty list, so a different shape is a loud failure too.

**Policy note, unresolved and not resolvable here.**  ``ssrn.com/robots.txt``
publishes ``Crawl-Delay: 5`` for ``User-Agent: *`` (which is where
:data:`_MIN_REQUEST_INTERVAL` comes from) and ``Disallow: /`` for ``GPTBot``,
``ChatGPT-User`` and ``Google-Extended``.  Separately, this repository already
contains a ``curl_cffi`` Chrome-impersonation path in
:mod:`lit_search.connectors.web_scraping`, and that path was measured to get
past SSRN's Cloudflare challenge.  This connector deliberately does **not** use
it: routing around an anti-bot control on a site that disallows AI agents by
name is a judgement about somebody else's terms of service, and the build plan
reserves it for an owner gate.  The measurement is recorded so the decision can
be made on facts rather than on a guess about feasibility.
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
from lit_search.core.coverage import CoverageStatus, CoveredPapers, SourceCoverage
from lit_search.core.paper import Database, Paper, PaperType
from lit_search.core.query import Query
from lit_search.core.source import Source, SourceType
from lit_search.exceptions import ConnectorError, MissingApiKeyError
from lit_search.query.builders.ssrn import SSRNQueryBuilder
from lit_search.utils.normalization import parse_date
from lit_search.utils.pagination_logging import log_pagination_request_failure

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://api.ssrn.com/content/v1/search"

#: Name of the credential a caller must supply, phrased as the search runner
#: reports it.
CREDENTIAL_NAME = "ssrn_api_key"

_API_KEY_HEADER = "X-API-Key"
"""Header the credential is sent in.

**Unverified.**  ``api.ssrn.com/content/v1/search`` answers HTTP 401 to every
unauthenticated request, and no SSRN credential was available on 2026-08-31, so
the header name could not be confirmed.  A wrong name fails loudly -- the
response is the same 401, which becomes ``FAILED`` carrying SSRN's own
``Unauthorized`` message -- rather than silently returning an unfiltered or
empty set.  Confirm and correct this the first time a key exists.
"""

#: Records requested per page.  Both parameter names were observed on the
#: reachable browse route on 2026-08-31.
_COUNT = 50

#: Safety limit on pagination rounds, so a route that never reports exhaustion
#: ends as an honest ``PARTIAL`` rather than never ending.
_MAX_PAGES = 1_000

#: Substrings that identify Cloudflare's interstitial.  SSRN serves it with
#: HTTP 403 to non-browser clients; the donor recognised it and then dropped
#: the recognition.
_CHALLENGE_MARKERS = ("just a moment", "cf-challenge", "challenges.cloudflare.com")

_MIN_REQUEST_INTERVAL = 5.0
"""Seconds between requests.

**Documented by the source, not inferred.**  ``https://www.ssrn.com/robots.txt``
and ``https://papers.ssrn.com/robots.txt`` both publish ``Crawl-Delay: 5`` for
``User-Agent: *`` (fetched 2026-08-31).  That is SSRN stating its own
acceptable rate, so it is used verbatim rather than adjusted.

The same files publish ``Disallow: /`` for ``GPTBot``, ``ChatGPT-User`` and
``Google-Extended``.  That does not name this client, but it is the clearest
signal SSRN gives about automated access and it belongs beside the number.
"""

_DEGRADED_LABEL = (
    "SSRN is a degraded source in this tool: its free-text search route is "
    "credentialed and was never observed answering, so the response envelope "
    "parsed here comes from SSRN's public browse route and nothing establishes "
    "that the reported total is a count of matches rather than a capped figure. "
    "Completeness is therefore never claimed for SSRN"
)


class SSRNAccessBlockedError(ConnectorError):
    """SSRN served an anti-bot challenge instead of a response.

    Separated from a plain HTTP error because the remedy is different: a
    challenge is not transient and will not clear on a retry, so a reader of
    the ledger needs to see it named rather than see "403 Client Error".
    """


class SSRNBodyError(ConnectorError):
    """A response body that cannot be read as an SSRN result set.

    Raised rather than defaulted away, because
    ``payload.get("papers", [])`` returns ``[]`` for every unrecognised shape
    and that is indistinguishable from a search that matched nothing.
    """


@dataclass(slots=True)
class _Retrieval:
    """Mutable bookkeeping for one paginated SSRN run.

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
    truncated_early: bool = False
    error_detail: str | None = None
    error_type: str | None = None

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


def _looks_like_a_challenge(response: requests.Response | None) -> bool:
    """Whether *response* carries an anti-bot interstitial.

    Parameters
    ----------
    response : requests.Response | None
        Response attached to an HTTP error, if any.

    Returns
    -------
    bool
        ``True`` when the body matches a known challenge marker.
    """
    if response is None:
        return False
    try:
        body = (response.text or "").lower()
    except Exception:  # pragma: no cover - defensive: .text can raise on odd encodings
        return False
    return any(marker in body for marker in _CHALLENGE_MARKERS)


class SSRNConnector(SearchConnectorBase):
    """Connector for SSRN, driven through its JSON content API.

    Parameters
    ----------
    api_key : str | None
        SSRN API credential.  Without one the connector contacts nothing and
        reports ``SKIPPED_NO_CREDENTIAL``, because SSRN's search route answers
        HTTP 401 to an unauthenticated request.
    query_builder : SSRNQueryBuilder | None
        Builder used to validate and convert queries.
    """

    _credential_attrs: ClassVar[tuple[str, ...]] = ("_api_key",)
    """SSRN's quota is keyed by the credential, matching the default.  Stated
    explicitly so the declaration reads as checked rather than inherited.
    """

    def __init__(
        self,
        api_key: str | None = None,
        query_builder: SSRNQueryBuilder | None = None,
    ) -> None:
        """Create an SSRN searcher.

        The missing credential is deliberately **not** raised here.  Raising in
        the constructor would leave the search runner with no connector object
        and only a class-level note that something was skipped; raising inside
        :meth:`_fetch_papers` lets the base class turn it into a coverage entry
        that names SSRN and names the credential.

        Parameters
        ----------
        api_key : str | None
            SSRN API credential.
        query_builder : SSRNQueryBuilder | None
            Builder used to validate and convert queries.  When ``None`` a
            default :class:`SSRNQueryBuilder` is created.
        """
        super().__init__()
        self._api_key = api_key
        self._query_builder: SSRNQueryBuilder = query_builder or SSRNQueryBuilder()

    @property
    def name(self) -> str:
        """Return the database identifier.

        Returns
        -------
        str
            Database name.
        """
        return Database.SSRN.value

    @property
    def query_builder(self) -> QueryBuilder:
        """Return the SSRN query builder.

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
            Five seconds, taken verbatim from SSRN's published
            ``Crawl-Delay``.  See :data:`_MIN_REQUEST_INTERVAL`.
        """
        return _MIN_REQUEST_INTERVAL

    @property
    def rate_limit_service(self) -> str:
        """Return the quota authority these requests are paced against.

        ``"ssrn"`` covers ``api.ssrn.com``, ``www.ssrn.com`` and
        ``papers.ssrn.com`` alike: the published ``Crawl-Delay`` is identical
        on the two robots files and they are one operator's estate, so a future
        SSRN connector against another of those hosts must share this clock.

        Returns
        -------
        str
            Quota-authority identifier.
        """
        return Database.SSRN.value

    def _prepare_headers(self, headers: dict) -> dict:
        """Add the JSON accept header and the credential.

        The ``Accept`` header is load-bearing rather than cosmetic: the same
        SSRN endpoint returns an XML ``PaperResultSet`` by default and the JSON
        envelope this module parses only when JSON is asked for, observed live
        on 2026-08-31.

        Parameters
        ----------
        headers : dict
            Headers supplied by the caller.

        Returns
        -------
        dict
            Headers with ``Accept`` and, when present, the credential.
        """
        prepared = super()._prepare_headers(headers)
        prepared.setdefault("Accept", "application/json")
        if self._api_key:
            prepared.setdefault(_API_KEY_HEADER, self._api_key)
        return prepared

    # ------------------------------------------------------------------
    # Response handling
    # ------------------------------------------------------------------

    def _get_page(self, params: dict[str, Any]) -> dict[str, Any]:
        """Fetch and validate one page of SSRN results.

        Parameters
        ----------
        params : dict[str, Any]
            Request parameters.

        Returns
        -------
        dict[str, Any]
            Decoded envelope, guaranteed to carry ``papers``.

        Raises
        ------
        SSRNAccessBlockedError
            When SSRN served an anti-bot challenge.
        SSRNBodyError
            When the body is not the envelope this module knows how to read.
        """
        try:
            response = self._get(_SEARCH_URL, params)
        except requests.HTTPError as exc:
            if _looks_like_a_challenge(exc.response):
                status = exc.response.status_code if exc.response is not None else "?"
                raise SSRNAccessBlockedError(
                    f"SSRN served an anti-bot challenge (HTTP {status}) instead of search "
                    f"results. This is not transient and will not clear on a retry: SSRN "
                    f"sits behind Cloudflare and refuses non-browser clients. Reporting it "
                    f"as zero papers would be wrong -- nothing was searched."
                ) from exc
            raise
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
            Decoded envelope.

        Raises
        ------
        SSRNAccessBlockedError
            When an HTTP 200 body is an anti-bot interstitial.
        SSRNBodyError
            When the body is not JSON, is not an object, or carries no
            ``papers`` list.  Each of these would otherwise be read as "SSRN
            holds no matching paper".

        Notes
        -----
        The challenge check runs **only after JSON decoding fails**, and the
        ordering is load-bearing.  Cloudflare's interstitial is an HTML page,
        so a body that decodes as JSON is not one.  Testing the raw text first
        matched :data:`_CHALLENGE_MARKERS` against the *record content* as
        well: a genuine result set containing a paper titled "Just a Moment:
        Timing in Securities Regulation" was rejected as an access block --
        reported as ``FAILED`` with "this is not transient and will not clear
        on a retry", which is a false diagnosis that would stop the user
        retrying a search that in fact worked.
        """
        try:
            payload = response.json()
        except ValueError as exc:
            if _looks_like_a_challenge(response):
                raise SSRNAccessBlockedError(
                    "SSRN returned an anti-bot challenge page with an HTTP "
                    f"{response.status_code} status, so nothing was searched."
                ) from exc
            raise SSRNBodyError(
                f"SSRN returned a non-JSON body with status {response.status_code}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise SSRNBodyError(
                f"SSRN returned a JSON {type(payload).__name__} where an object was expected."
            )
        if not isinstance(payload.get("papers"), list):
            raise SSRNBodyError(
                "SSRN response carried no 'papers' list, so it is not a result set. "
                "Treating it as an empty result set would report a failure as zero papers."
            )
        return payload

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_authors(record: dict[str, Any]) -> list[Author]:
        """Extract authors from an SSRN record.

        Parameters
        ----------
        record : dict[str, Any]
            One ``papers`` entry.

        Returns
        -------
        list[Author]
            Parsed authors; empty when the record names none.
        """
        raw = record.get("authors")
        entries = raw if isinstance(raw, list) else []
        authors: list[Author] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            parts = [
                str(entry.get(key) or "").strip()
                for key in ("first_name", "middle_name", "last_name")
            ]
            name = " ".join(part for part in parts if part)
            if name:
                authors.append(Author(name=name))
        return authors

    def _parse_record(self, record: Any) -> Paper | None:
        """Parse one SSRN record into a :class:`Paper`.

        The field names are those observed on SSRN's reachable browse route on
        2026-08-31.  Note what is **absent** from them: SSRN's list records
        carry no abstract and no DOI, so every paper from this source reaches
        re-ranking on its title alone.  That is a field-level degradation, not
        a coverage one, and it is recorded here because nothing downstream
        currently has a place to record it.

        Parameters
        ----------
        record : Any
            One ``papers`` entry.

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
        if not title:
            return None
        paper_id = record.get("id")
        url = str(record.get("url") or "").strip() or (
            f"https://papers.ssrn.com/sol3/papers.cfm?abstract_id={paper_id}" if paper_id else None
        )
        page_count = record.get("page_count")
        series = str(record.get("abstract_type") or "").strip()

        try:
            return Paper(
                title=title,
                abstract="",
                authors=self._parse_authors(record),
                source=Source(
                    title=series or "SSRN",
                    source_type=SourceType.REPOSITORY,
                ),
                publication_date=parse_date(str(record.get("approved_date") or "").strip() or None),
                url=url,
                doi=None,
                page_count=page_count if isinstance(page_count, int) else None,
                found_in={self.name},
                paper_type=PaperType.UNPUBLISHED,
            )
        except ValueError:
            logger.debug("SSRN: record %s failed model validation.", paper_id)
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
        """Fetch papers from SSRN, with coverage.

        Parameters
        ----------
        query : Query
            Validated query object.
        max_papers : int | None
            Maximum papers to retrieve.  ``None`` means unlimited.
        progress_callback : Callable[[int, int | None], None] | None
            Progress callback called as ``callback(processed, total_or_none)``.
        since : datetime.date | None
            Not sent upstream: SSRN's search route exposes no date filter that
            could be observed, so a bound is enforced client-side by
            ``SearchRunner``'s post-fetch filter (verified in
            ``search_runner.py``, which drops out-of-range papers after the
            connectors return).  Passing one is recorded in the coverage
            detail, because the interaction with ``max_papers`` is a real and
            otherwise invisible recall loss -- capping at N over the *whole*
            corpus and only then filtering to a date range can leave almost
            nothing while thousands of in-range records existed.
        until : datetime.date | None
            Not sent upstream, for the same reason.

        Returns
        -------
        CoveredPapers
            Retrieved papers and the coverage status describing them.  Never
            ``COMPLETE``; see the module docstring.

        Raises
        ------
        MissingApiKeyError
            When no credential was supplied, which is the default.
        SSRNAccessBlockedError
            When SSRN served an anti-bot challenge and nothing was retrieved.
        requests.RequestException
            When the first request failed at the transport level.
        """
        if not self._api_key:
            logger.warning(
                "Skipping SSRN: %s answers HTTP 401 without a credential, and the HTML "
                "search pages this connector's donor scraped now return 404 and a "
                "Cloudflare challenge. Supply %s to search SSRN.",
                _SEARCH_URL,
                CREDENTIAL_NAME,
            )
            raise MissingApiKeyError(CREDENTIAL_NAME)

        term = self._query_builder.convert_query(query)
        run = _Retrieval()

        while True:
            params = {
                "query": term,
                "index": run.processed,
                "count": _count_for(run, max_papers),
            }

            try:
                payload = self._get_page(params)
            except (requests.RequestException, ConnectorError) as exc:
                self._record_page_failure(run, exc)
                break

            records = self._absorb_page(run, payload)
            if progress_callback is not None:
                progress_callback(run.processed, run.total)

            if not self._should_continue(run, records, max_papers):
                break

        if progress_callback is not None:
            progress_callback(run.processed, run.total)

        kept = run.papers[:max_papers] if max_papers is not None else run.papers
        return CoveredPapers(
            papers=tuple(kept),
            coverage=self._build_coverage(
                retrieved=len(kept),
                run=run,
                max_papers=max_papers,
                date_bounded=since is not None or until is not None,
            ),
        )

    @staticmethod
    def _record_page_failure(run: _Retrieval, exc: Exception) -> None:
        """Record a page failure, or re-raise when nothing was retrieved.

        Parameters
        ----------
        run : _Retrieval
            Bookkeeping for the current run.
        exc : Exception
            The failure raised while fetching or reading the page.

        Raises
        ------
        Exception
            The original failure, when nothing had been retrieved.
        """
        if not run.papers:
            raise exc
        log_pagination_request_failure(
            logger,
            connector_label="SSRN",
            processed=run.processed,
            total=run.total,
            position_label="index",
            position=str(run.processed),
            exc=exc,
        )
        run.truncate(f"page request at index={run.processed} failed: {exc}", type(exc).__name__)

    def _absorb_page(self, run: _Retrieval, payload: dict[str, Any]) -> list[Any]:
        """Fold one page into *run* and return the raw records it carried.

        Parameters
        ----------
        run : _Retrieval
            Bookkeeping for the current run, updated in place.
        payload : dict[str, Any]
            Validated response envelope.

        Returns
        -------
        list[Any]
            Raw ``papers`` entries, including any that failed to parse.
        """
        total = payload.get("total")
        if isinstance(total, int) and not isinstance(total, bool) and run.total is None:
            run.total = total
        records: list[Any] = payload["papers"]
        for record in records:
            paper = self._parse_record(record)
            if paper is not None:
                run.papers.append(paper)
                run.parsed += 1
        run.processed += len(records)
        run.pages += 1
        return records

    @staticmethod
    def _should_continue(run: _Retrieval, records: list[Any], max_papers: int | None) -> bool:
        """Whether another page should be requested.

        Parameters
        ----------
        run : _Retrieval
            Bookkeeping for the current run, updated in place.
        records : list[Any]
            Raw entries the last page carried.
        max_papers : int | None
            Caller-imposed cap.

        Returns
        -------
        bool
            ``True`` to request another page.
        """
        if max_papers is not None and len(run.papers) >= max_papers:
            return False
        if not records:
            # An empty page is the only exhaustion signal this route offers.
            if run.total is not None and run.processed < run.total:
                run.truncate(
                    f"SSRN stopped returning records after {run.processed} of "
                    f"{run.total} it reported (empty page at page {run.pages})"
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
        retrieved: int,
        run: _Retrieval,
        max_papers: int | None,
        date_bounded: bool = False,
    ) -> SourceCoverage:
        """Turn the run's counters into a coverage status, capped at ``PARTIAL``.

        ``from_pagination`` is deliberately not used.  It would return
        ``COMPLETE`` whenever ``retrieved == total``, and this connector has no
        standing to make that claim: the envelope it reads was observed on
        SSRN's browse route, never on the search route, so nothing establishes
        that ``total`` counts matches rather than a capped figure.  The cap and
        the total are still carried, so the reader loses no information -- only
        the unjustified claim.

        Parameters
        ----------
        retrieved : int
            Papers actually returned to the caller, after the cap is applied.
        run : _Retrieval
            Bookkeeping for the finished run.
        max_papers : int | None
            Caller-imposed cap.
        date_bounded : bool
            Whether the caller asked for a date range.  SSRN cannot apply one
            upstream, so the range is enforced after retrieval and the two
            restrictions compound; that is worth saying out loud.

        Returns
        -------
        SourceCoverage
            A ``PARTIAL`` record carrying every note that applies.
        """
        notes: list[str] = []
        if run.error_detail:
            notes.append(run.error_detail)
        unparsed = run.processed - run.parsed
        if unparsed > 0:
            notes.append(f"{unparsed} of {run.processed} records could not be parsed into papers")
        capped = max_papers is not None and retrieved >= max_papers
        if capped:
            notes.append(f"retrieval stopped at the caller's cap of {max_papers}")
        if date_bounded:
            notes.append(
                "the requested date range was not sent to SSRN, which exposes no date "
                "filter this connector could verify; it is applied after retrieval"
                + (
                    ", so the cap above truncated the undated corpus before the range "
                    "was applied and in-range records beyond it were never seen"
                    if capped
                    else ""
                )
            )
        notes.append(_DEGRADED_LABEL)

        return SourceCoverage(
            source=self.name,
            status=CoverageStatus.PARTIAL,
            retrieved=retrieved,
            detail="; ".join(notes),
            cap=max_papers if max_papers is not None and retrieved >= max_papers else None,
            reported_total=run.total,
            error=run.error_detail,
            error_type=run.error_type,
        )


def _count_for(run: _Retrieval, max_papers: int | None) -> int:
    """Return the record count to request for the next page.

    Parameters
    ----------
    run : _Retrieval
        Bookkeeping for the current run.
    max_papers : int | None
        Caller-imposed cap.

    Returns
    -------
    int
        Count between 1 and :data:`_COUNT`.
    """
    remaining = (max_papers - len(run.papers)) if max_papers is not None else _COUNT
    return max(1, min(_COUNT, remaining))
