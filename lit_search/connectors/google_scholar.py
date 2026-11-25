"""Google Scholar: shipped disabled, and unable to report a block as zero.

Scholar is in the specification as a **degraded** source for one reason,
quoted: on a CAPTCHA or a markup change the donor
``paper_search_mcp/academic_platforms/google_scholar.py`` *returns whatever it
accumulated instead of raising*.  So a blocked search and an exhausted search
leave by the same door, and a search blocked on page one returns an empty list
that the runner records as a successful zero-result search.  That is the
motivating failure of this project, from a source that is blocked routinely.

This module keeps a scraper and removes that property.  Three separate
mechanisms, because the block has three shapes and only one of them is an HTTP
error.

=========================================  =========================================
Shape                                       What this connector does
=========================================  =========================================
HTTP 429 / 403, or a body matching a        Nothing retrieved -> raise
CAPTCHA marker                              :class:`ScholarBlockedError` -> ``FAILED``.
                                            Records already retrieved -> keep them and
                                            report ``PARTIAL``, naming the block.
A page served, parsed, and yielding no      **Raise.**  A results page that matches no
results, with no "no results" marker        selector is a markup change, not an empty
present                                     index.  Reporting zero here is the exact
                                            donor defect.
A page served with Scholar's own            A genuine empty result: zero papers,
"did not match any articles" marker         ``PARTIAL`` like every other Scholar result.
=========================================  =========================================

**It never reports ``COMPLETE``.**  Scholar publishes no result total that can
be relied on -- the figure on the page is an estimate, it moves between pages,
and access is capped at roughly a thousand results per query whatever the
estimate says.  There is nothing to justify a completeness claim with, and
:meth:`~lit_search.core.coverage.SourceCoverage.complete` requires a
source-reported total precisely so that this case cannot be waved through.

**It is disabled by default, and the reason is policy rather than caution.**
``https://scholar.google.com/robots.txt``, fetched 2026-08-31, carries under
``User-agent: *``::

    Disallow: /search
    Disallow: /scholar
    Disallow: /citations?

``/scholar`` is the path the donor requests.  Google also states plainly that
it does not provide bulk access.  So the scrape is disallowed for every agent
by the site's own file, and switching it on is a decision about somebody else's
terms of service that this code must not make on the operator's behalf.  Until
:data:`_OPT_IN_ENV_VAR` is set, every search reports
``SKIPPED_NO_CREDENTIAL`` -- which keeps Scholar visible in the ledger as a
source deliberately not consulted, rather than absent from it.

**The selectors are unverified, and that is stated rather than hidden.**  No
request was made to Scholar while writing this module, because making one is
the thing the paragraph above says not to do.  The class names in
:data:`_RESULT_XPATH` and its neighbours are the long-standing public ones; if
they are wrong, the failure is **loud** by construction -- a served page that
yields no results and carries no "no results" marker raises rather than
returning zero.  That is the one property worth having here, and it does not
depend on the selectors being right.
"""

from __future__ import annotations

import datetime
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import requests
from lxml import html

if TYPE_CHECKING:
    from lxml.html import HtmlElement

    from lit_search.query.builder import QueryBuilder

from lit_search.connectors.search_base import SearchConnectorBase
from lit_search.core.author import Author
from lit_search.core.coverage import CoveredPapers, SourceCoverage
from lit_search.core.paper import Database, Paper
from lit_search.core.query import Query
from lit_search.core.source import Source, SourceType
from lit_search.exceptions import ConnectorError, MissingApiKeyError
from lit_search.query.builders.google_scholar import GoogleScholarQueryBuilder
from lit_search.utils.pagination_logging import log_pagination_request_failure

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://scholar.google.com/scholar"

_OPT_IN_ENV_VAR = "LIT_SEARCH_GOOGLE_SCHOLAR_ENABLED"
"""Environment variable recording that the operator accepts the policy position.

``scholar.google.com/robots.txt`` disallows ``/scholar`` for every agent.  That
is a fact about Google's terms, not about feasibility, so the code refuses to
decide it and refuses to contact Scholar until the operator has.  Modelling the
acknowledgement as a credential is deliberate: ``SKIPPED_NO_CREDENTIAL``
already means "never contacted, determinate, fixable by configuration", which
is exactly this.
"""

_TRUTHY = frozenset({"1", "true", "yes", "on", "enabled"})

#: Results per page.  Scholar's own maximum for the ``num`` parameter.
_PAGE_SIZE = 20

#: Hard ceiling on pages, matching Scholar's roughly 1,000-result access cap.
_MAX_PAGES = 50

#: Result containers.  See the module docstring on why these are unverified and
#: why that is safe.
_RESULT_XPATH = "//div[contains(@class, 'gs_r') and contains(@class, 'gs_scl')]"
_TITLE_XPATH = ".//h3[contains(@class, 'gs_rt')]"
_BYLINE_XPATH = ".//div[contains(@class, 'gs_a')]"
_SNIPPET_XPATH = ".//div[contains(@class, 'gs_rs')]"

#: Lower-cased substrings that mean the request was refused rather than
#: answered.  A body carrying any of these is a block however it was served,
#: and Scholar serves them with HTTP 200 as often as not.
_BLOCK_MARKERS = (
    "our systems have detected unusual traffic",
    "why did this happen",
    "/sorry/index",
    'id="captcha"',
    "g-recaptcha",
    "unusual traffic from your computer network",
)

#: Lower-cased substrings that mean Scholar answered and matched nothing.  This
#: is the **only** evidence that justifies reporting zero results from a page
#: that was served: without one of these, an unmatched page is a markup change.
_EMPTY_MARKERS = (
    "did not match any articles",
    "did not match any user profiles",
    "your search - ",
)

_MIN_REQUEST_INTERVAL = 30.0
"""Seconds between requests.

**No defensible rate exists, and this number does not pretend to be one.**
Google publishes no Scholar API and no crawl rate for it; its ``robots.txt``
disallows the path outright rather than pacing it.  There is therefore nothing
to verify against, and any interval is a judgement.

Thirty seconds is chosen to be slow enough that the connector cannot contribute
to the traffic pattern Scholar blocks on, on the reasoning that a source which
asks not to be automated at all should at minimum not be automated briskly.  It
makes a multi-page Scholar search take minutes, which is the intended
discouragement rather than an oversight.
"""


class ScholarBlockedError(ConnectorError):
    """Scholar refused the request instead of answering it.

    Separated from a plain HTTP error because the remedy differs: a block is
    not transient in the way a 503 is, and a reader of the ledger needs to see
    it named.
    """


class ScholarMarkupError(ConnectorError):
    """A served page yielded no results and no "no results" marker.

    This is the failure the donor cannot express.  A results page whose markup
    has changed parses to zero results, and zero results is indistinguishable
    from an empty index unless something refuses to make the claim.  This
    exception is that refusal.
    """


@dataclass(slots=True)
class _Run:
    """Bookkeeping for one Scholar search.

    Attributes
    ----------
    papers : list[Paper]
        Records parsed so far, in result order.
    processed : int
        Result blocks seen, whether or not they parsed.
    parsed : int
        Blocks that became a :class:`~lit_search.core.paper.Paper`.
    pages : int
        Pages consumed.
    exhausted : bool
        Whether Scholar ran out of results.
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
    exhausted: bool = False
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


class GoogleScholarConnector(SearchConnectorBase):
    """Search Google Scholar, refusing to report a block as an empty result.

    Parameters
    ----------
    query_builder : GoogleScholarQueryBuilder | None
        Builder used to validate and convert queries.
    """

    _credential_attrs: ClassVar[tuple[str, ...]] = ()
    """Scholar issues no key.  The opt-in flag is an acknowledgement, not a
    secret, and is deliberately excluded: including it would fingerprint the
    rate-limit bucket differently for an opted-in process than an opted-out
    one, against the same host.
    """

    def __init__(self, query_builder: GoogleScholarQueryBuilder | None = None) -> None:
        """Create a Scholar searcher.

        Parameters
        ----------
        query_builder : GoogleScholarQueryBuilder | None
            Builder used to validate and convert queries.
        """
        super().__init__()
        self._query_builder: GoogleScholarQueryBuilder = (
            query_builder or GoogleScholarQueryBuilder()
        )

    @property
    def name(self) -> str:
        """Return the database identifier.

        Returns
        -------
        str
            Database name.
        """
        return Database.GOOGLE_SCHOLAR.value

    @property
    def query_builder(self) -> QueryBuilder:
        """Return the Scholar query builder.

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
            Thirty seconds.  See :data:`_MIN_REQUEST_INTERVAL` for why no
            verified value exists and why this one is deliberately slow.
        """
        return _MIN_REQUEST_INTERVAL

    @property
    def rate_limit_service(self) -> str:
        """Return the quota authority these requests are paced against.

        Returns
        -------
        str
            Quota-authority identifier.
        """
        return Database.GOOGLE_SCHOLAR.value

    @staticmethod
    def _require_opt_in() -> None:
        """Refuse to contact Scholar until the operator has opted in.

        Raises
        ------
        MissingApiKeyError
            When :data:`_OPT_IN_ENV_VAR` is unset or not truthy.
        """
        if os.environ.get(_OPT_IN_ENV_VAR, "").strip().lower() not in _TRUTHY:
            logger.warning(
                "Skipping Google Scholar: scholar.google.com/robots.txt disallows /scholar "
                "for every agent, so this connector does not contact it unless %s is set. "
                "No request was made, and no papers are being reported as absent.",
                _OPT_IN_ENV_VAR,
            )
            raise MissingApiKeyError(_OPT_IN_ENV_VAR)

    # ------------------------------------------------------------------
    # Response handling
    # ------------------------------------------------------------------

    def _get_page(self, params: dict[str, Any]) -> HtmlElement:
        """Fetch one results page and reject every shape that is not one.

        Parameters
        ----------
        params : dict[str, Any]
            Request parameters.

        Returns
        -------
        HtmlElement
            Parsed document root.

        Raises
        ------
        ScholarBlockedError
            When Scholar refused the request, whether by HTTP status or by
            serving an interstitial with HTTP 200.
        """
        try:
            response = self._get(_SEARCH_URL, params)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in (403, 429) or _looks_blocked(_body_of(exc.response)):
                raise ScholarBlockedError(
                    f"Google Scholar refused the request (HTTP {status}). Nothing was "
                    f"searched, so reporting zero papers would say Scholar holds no "
                    f"matching article."
                ) from exc
            raise
        body = response.text or ""
        if _looks_blocked(body.lower()):
            raise ScholarBlockedError(
                "Google Scholar served a block or CAPTCHA interstitial with HTTP "
                f"{response.status_code}. Nothing was searched."
            )
        return html.fromstring(body)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_result(self, block: HtmlElement) -> Paper | None:
        """Parse one result block into a :class:`Paper`.

        Scholar's list view carries a truncated snippet rather than an
        abstract, and no DOI at all.  Both absences are field-level
        degradations that follow every Scholar record downstream into
        re-ranking, and they are named in the coverage detail because nothing
        else records them.

        Parameters
        ----------
        block : HtmlElement
            One result container.

        Returns
        -------
        Paper | None
            Parsed paper, or ``None`` when it carries no title.  Callers
            **must** count the ``None`` returns.
        """
        title_nodes = block.xpath(_TITLE_XPATH)
        if not title_nodes:
            return None
        title = " ".join(str(title_nodes[0].text_content()).split())
        # Scholar prefixes some titles with a bracketed format marker.
        for marker in ("[PDF]", "[HTML]", "[BOOK]", "[CITATION]", "[B]", "[C]"):
            if title.startswith(marker):
                title = title[len(marker) :].strip()
        if not title:
            return None

        links = title_nodes[0].xpath(".//a/@href")
        byline_nodes = block.xpath(_BYLINE_XPATH)
        byline = " ".join(str(byline_nodes[0].text_content()).split()) if byline_nodes else ""
        snippet_nodes = block.xpath(_SNIPPET_XPATH)
        snippet = " ".join(str(snippet_nodes[0].text_content()).split()) if snippet_nodes else ""

        try:
            return Paper(
                title=title,
                abstract=snippet,
                authors=[Author(name=name) for name in _byline_authors(byline)],
                source=Source(
                    title=_byline_venue(byline) or "Google Scholar",
                    source_type=SourceType.OTHER,
                ),
                publication_date=_byline_date(byline),
                url=str(links[0]) if links else None,
                doi=None,
                found_in={self.name},
            )
        except ValueError:
            logger.debug("Google Scholar: a result block failed model validation.")
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
        """Search Scholar, keeping partial results and naming the block.

        Parameters
        ----------
        query : Query
            Validated query object.
        max_papers : int | None
            Maximum papers to retrieve.  ``None`` means unlimited, bounded by
            :data:`_MAX_PAGES`.
        progress_callback : Callable[[int, int | None], None] | None
            Progress callback called as ``callback(processed, None)``.  The
            total is always ``None``: Scholar's result count is an estimate
            that moves between pages, and passing an estimate as a
            source-reported total would be corroboration of nothing.
        since : datetime.date | None
            Lower bound, sent as Scholar's ``as_ylo`` year filter.
        until : datetime.date | None
            Upper bound, sent as ``as_yhi``.

        Returns
        -------
        CoveredPapers
            Retrieved papers and the coverage status describing them.  Never
            ``COMPLETE``.

        Raises
        ------
        MissingApiKeyError
            When the operator has not opted in.
        ScholarBlockedError
            When Scholar refused the first request.
        ScholarMarkupError
            When the first page was served and parsed to nothing without
            Scholar saying it matched nothing.
        requests.RequestException
            When the first request failed at the transport level.
        """
        self._require_opt_in()
        term = self._query_builder.convert_query(query)
        run = _Run()

        while True:
            params: dict[str, Any] = {
                "q": term,
                "start": run.processed,
                "num": _PAGE_SIZE,
                "hl": "en",
            }
            if since is not None:
                params["as_ylo"] = since.year
            if until is not None:
                params["as_yhi"] = until.year

            try:
                document = self._get_page(params)
            except (requests.RequestException, ConnectorError) as exc:
                self._record_page_failure(run, exc)
                break

            blocks = document.xpath(_RESULT_XPATH)
            if not blocks:
                self._handle_empty_page(run, document)
                break

            self._absorb_page(run, blocks, max_papers)
            if progress_callback is not None:
                progress_callback(run.processed, None)

            if not self._should_continue(run, max_papers):
                break

        if progress_callback is not None:
            progress_callback(run.processed, None)

        kept = run.papers[:max_papers] if max_papers is not None else run.papers
        return CoveredPapers(
            papers=tuple(kept),
            coverage=self._build_coverage(run=run, retrieved=len(kept), max_papers=max_papers),
        )

    @staticmethod
    def _handle_empty_page(run: _Run, document: HtmlElement) -> None:
        """Decide whether a resultless page is an empty index or a markup change.

        This is the method the donor does not have, and it is the whole reason
        this connector exists in this form.

        Parameters
        ----------
        run : _Run
            Bookkeeping for the current run, updated in place.
        document : HtmlElement
            The parsed page.

        Raises
        ------
        ScholarMarkupError
            When nothing had been retrieved and the page carries no evidence
            that Scholar itself matched nothing.
        """
        text = str(document.text_content() or "").lower()
        if any(marker in text for marker in _EMPTY_MARKERS):
            run.exhausted = True
            run.pages += 1
            return
        if run.papers:
            run.truncate(
                f"page {run.pages + 1} was served but matched no result markup and carried "
                f"no 'did not match any articles' notice, so pagination stopped without "
                f"reaching the end of the result set"
            )
            return
        raise ScholarMarkupError(
            "Google Scholar served a page that matched no result markup and carried no "
            "'did not match any articles' notice. That is a markup change or an "
            "unrecognised interstitial, not an empty index, and reporting it as zero "
            "papers would say Scholar holds no matching article."
        )

    @staticmethod
    def _record_page_failure(run: _Run, exc: Exception) -> None:
        """Record a page failure, or re-raise when nothing was retrieved.

        Parameters
        ----------
        run : _Run
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
            connector_label="Google Scholar",
            processed=run.processed,
            total=None,
            position_label="start",
            position=str(run.processed),
            exc=exc,
        )
        run.truncate(f"page request at start={run.processed} failed: {exc}", type(exc).__name__)

    def _absorb_page(self, run: _Run, blocks: list[Any], max_papers: int | None) -> None:
        """Fold one page of result blocks into *run*.

        Parameters
        ----------
        run : _Run
            Bookkeeping, updated in place.
        blocks : list[Any]
            Result containers found on the page.
        max_papers : int | None
            Caller-imposed cap.
        """
        for block in blocks:
            run.processed += 1
            paper = self._parse_result(block)
            if paper is None:
                continue
            run.parsed += 1
            if max_papers is None or len(run.papers) < max_papers:
                run.papers.append(paper)
        run.pages += 1

    @staticmethod
    def _should_continue(run: _Run, max_papers: int | None) -> bool:
        """Whether another page should be requested.

        Parameters
        ----------
        run : _Run
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
        if run.pages >= _MAX_PAGES:
            run.truncate(
                f"stopped at the connector's {_MAX_PAGES}-page limit, which matches "
                f"Scholar's own access cap of roughly 1,000 results per query"
            )
            return False
        return True

    def _build_coverage(
        self,
        *,
        run: _Run,
        retrieved: int,
        max_papers: int | None,
    ) -> SourceCoverage:
        """Describe what the search achieved, never claiming completeness.

        Parameters
        ----------
        run : _Run
            Bookkeeping for the finished run.
        retrieved : int
            Papers actually returned, after the cap.
        max_papers : int | None
            Caller-imposed cap.

        Returns
        -------
        SourceCoverage
            A ``PARTIAL`` record carrying every note that applies.
        """
        notes: list[str] = [
            "Google Scholar publishes no reliable result total and caps access at roughly "
            "1,000 results per query, so completeness is never established for this source"
        ]
        if run.error_detail:
            notes.append(run.error_detail)
        unparsed = run.processed - run.parsed
        if unparsed > 0:
            notes.append(
                f"{unparsed} of {run.processed} result blocks could not be parsed into papers"
            )
        if max_papers is not None and retrieved >= max_papers:
            notes.append(f"retrieval stopped at the caller's cap of {max_papers}")
        if retrieved:
            notes.append(
                "Scholar's list view carries a truncated snippet rather than an abstract "
                "and no DOI, so these records reach deduplication without an identifier "
                "and re-ranking without full abstract text"
            )

        return SourceCoverage.partial(
            self.name,
            retrieved,
            detail="; ".join(notes),
            error=run.error_detail,
            error_type=run.error_type,
        )


def _body_of(response: requests.Response | None) -> str:
    """Return a response body lower-cased, tolerating odd encodings.

    Parameters
    ----------
    response : requests.Response | None
        Response attached to an HTTP error, if any.

    Returns
    -------
    str
        Lower-cased body, empty when unavailable.
    """
    if response is None:
        return ""
    try:
        return (response.text or "").lower()
    except Exception:  # pragma: no cover - defensive: .text can raise
        return ""


def _looks_blocked(lowered_body: str) -> bool:
    """Whether a lower-cased body carries a Scholar block marker.

    Parameters
    ----------
    lowered_body : str
        Response body, already lower-cased.

    Returns
    -------
    bool
        ``True`` when any block marker is present.
    """
    return any(marker in lowered_body for marker in _BLOCK_MARKERS)


def _byline_authors(byline: str) -> list[str]:
    """Extract author names from Scholar's byline line.

    The byline is ``"A Author, B Author - Venue, Year - publisher.com"``.  Only
    the part before the first dash is authors, and Scholar abbreviates it with
    an ellipsis when there are many, so the list is routinely incomplete --
    which is a field-level degradation, not a coverage one.

    Parameters
    ----------
    byline : str
        The byline text.

    Returns
    -------
    list[str]
        Author names.
    """
    head = byline.split(" - ", 1)[0]
    return [part.strip() for part in head.split(",") if part.strip() and part.strip() != "..."]


def _byline_venue(byline: str) -> str | None:
    """Extract the venue from Scholar's byline line.

    Parameters
    ----------
    byline : str
        The byline text.

    Returns
    -------
    str | None
        Venue text, or ``None`` when the byline has no venue segment.
    """
    parts = byline.split(" - ")
    if len(parts) < 2:
        return None
    venue = parts[1].strip()
    # Trim a trailing ", 2019" year, which is carried separately.
    if "," in venue:
        head, _, tail = venue.rpartition(",")
        if tail.strip().isdigit() and head.strip():
            venue = head.strip()
    return venue or None


def _byline_date(byline: str) -> datetime.date | None:
    """Extract a publication year from Scholar's byline line.

    Only a year is available, so the date is reported as 1 January of that
    year.  Scholar gives nothing finer.

    Parameters
    ----------
    byline : str
        The byline text.

    Returns
    -------
    datetime.date | None
        1 January of the year, or ``None`` when no plausible year is present.
    """
    for token in reversed(byline.replace(",", " ").split()):
        if len(token) == 4 and token.isdigit() and 1600 <= int(token) <= 2100:
            return datetime.date(int(token), 1, 1)
    return None
