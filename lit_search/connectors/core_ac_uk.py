"""CORE searcher implementation.

Ported from ``paper_search_mcp/academic_platforms/core.py`` at pinned commit
``234678ab231074a7977320978ee0496dcdaddd1f``.  The module is named
``core_ac_uk`` rather than ``core`` so that it does not sit one directory away
from the :mod:`lit_search.core` package under the same name; the connector, the
:class:`~lit_search.core.paper.Database` member and the ledger key are all
plain ``core``.

The port is **not** a mechanical translation.  The donor's ``search``
(``core.py:42-159``) has three separate routes to a confident empty list, and
one of them is worse than the shape ``PORTING-GUIDE.md`` describes:

- ``except requests.RequestException`` and ``except Exception`` both fall
  through to ``return papers`` (``core.py:148-159``);
- **``if response is None: return papers``** (``core.py:128-129``) fires after
  the retry loop exhausts three attempts, so three consecutive timeouts are
  reported as a database that holds no matching paper, with no exception raised
  anywhere;
- ``for attempt in range(3): ... time.sleep(wait_seconds)``
  (``core.py:85-127``) is an in-connector sleep, invisible to the shared
  cross-process limiter and therefore unable to hold back a peer process.  It
  is removed; pacing belongs to :meth:`ConnectorBase._request_with_retry`.

One donor behaviour is dropped on separate grounds.  On HTTP 401 or 403 the
donor **retries the same request with the API key stripped**
(``core.py:100-114``).  That is a credential-identity bug tangled into the
error contract: a rejected key silently becomes an anonymous request, which
draws on a different quota, returns a different result set, and reports success
under the key's name.  A rejected key is an error here.

Live observations from 2026-08-31.  **The first two correct claims this module
previously made in the opposite direction**; both were re-checked against the
live API during adversarial review and are recorded here rather than quietly
edited, because a design decision still standing below rests on the retracted
version.

- **RETRACTED: "CORE refuses keyless search."**  The earlier note recorded
  ``GET /v3/search/works?q=catalysis&limit=2`` with no ``Authorization`` header
  returning HTTP 429, and concluded that a key is a hard requirement.  Re-checked
  2026-08-31: ``GET /v3/search/works/?q=catalysis&limit=2`` **with no key**
  returns **HTTP 200** with ``totalHits: 176809``, two parsed results and
  ``x-ratelimit-limit: 10``.  The 429 was a bucket the probe itself had emptied,
  helped by the missing trailing slash (see :data:`_SEARCH_URL`) charging two
  tokens per probe.

  **The consequence is not yet fixed and is the largest open item on this
  connector.**  :meth:`CoreConnector._fetch_papers` still raises
  :class:`~lit_search.exceptions.MissingApiKeyError` before issuing a request,
  so with no key configured -- which is every run today, because
  ``SearchRunner`` has no way to pass one -- CORE reports
  ``SKIPPED_NO_CREDENTIAL`` for a source that would have answered.  A source
  silently not consulted is the failure this fork exists to prevent, and it is
  now on the wrong side of it.  Whether to search anonymously at the
  100-per-day tier or to keep insisting on a key is a policy call, not a bug
  fix, so it is stated here rather than decided in passing.
- **RETRACTED: "CORE's own two descriptions of its response disagree and
  neither could be checked."**  They do disagree -- the OpenAPI document at
  ``api.core.ac.uk/swagger/v3.json`` says ``total_hits`` / ``published_date`` /
  ``year_published`` / ``download_url`` / ``field_of_study``, the documentation
  page says ``totalHits`` / ``publishedDate`` / ``yearPublished`` /
  ``downloadUrl`` / ``fieldOfStudy`` -- but a live response settles it, and the
  documentation page is right.  A keyless search returned an envelope of
  ``limit`` / ``offset`` / ``results`` / ``searchId`` / ``totalHits`` and work
  keys including ``publishedDate``, ``yearPublished``, ``downloadUrl``,
  ``fieldOfStudy``, ``sourceFulltextUrls``, ``citationCount``, ``documentType``
  and ``journals`` -- **camelCase throughout, no snake_case key present**.
  :func:`_first` reads both spellings anyway; that is now redundancy rather
  than a hedge, and it is kept because it costs nothing and the OpenAPI
  document may yet describe a variant.
- **A rejected key is loud.**  ``Authorization: Bearer NOT_A_REAL_KEY`` returned
  HTTP 401 ``{"message": "The API key you provided is not valid."}`` on both
  the GET and POST forms.  Re-confirmed 2026-08-31.

When no total can be read under either spelling the run reports ``PARTIAL``,
never ``COMPLETE``.  A guessed total is exactly how an unfalsifiable
completeness claim gets made.

Rate limit: token-bucket, tier-dependent.  See :data:`_MIN_REQUEST_INTERVAL`.
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
from lit_search.query.builders.core_ac_uk import CoreQueryBuilder
from lit_search.utils.normalization import normalize_doi, normalize_language, parse_date
from lit_search.utils.pagination_logging import log_pagination_request_failure
from lit_search.utils.rate_limit import RateLimitQuota

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.core.ac.uk/v3"

_SEARCH_URL = f"{_BASE_URL}/search/works/"
"""Search endpoint, **with the trailing slash**, which is load-bearing.

Verified live 2026-08-31.  ``GET /v3/search/works?q=catalysis&limit=1`` (no
slash) returns **HTTP 301** with ``location:
https://api.core.ac.uk/v3/search/works/?q=catalysis&limit=1``; the slashed form
returns HTTP 200 directly.  ``requests`` follows redirects by default, so the
unslashed form still worked -- but at two upstream round trips per page inside
a single :meth:`ConnectorBase._rate_limit` reservation, and CORE's
``x-ratelimit-remaining`` was observed decrementing across the redirect.  That
is a hidden doubling of the request rate against a 10-per-minute bucket, and
the kind of over-send that returns 429s the ledger records as ``FAILED``
against CORE when the cause is our own pacing.
"""

# The donor's own ceiling, described in its docstring as "CORE API max limit is
# 100".  CORE's OpenAPI document states a default of 10 and no maximum, and the
# real ceiling could not be tested without a key, so this is the conservative
# of the two available numbers.
_PAGE_SIZE = 100

_MAX_OFFSET = 1_000_000
"""Offset ceiling used to end a run as an honest ``PARTIAL``.

**Corrected 2026-08-31 after a live measurement contradicted the earlier
guess.**  This was ``10_000``, reasoned from CORE documenting a separate
token-costlier "scroll search" -- the usual shape of a bounded offset API --
and from both sibling sources in this port bounding theirs at exactly 10,000.
The reasoning was good and the conclusion was wrong: a keyless
``GET /v3/search/works/?q=catalysis&limit=10&offset=10000`` returned **HTTP 200
with ten real records**, and so did the same request at ``offset=100000``
(``totalHits`` 176,809 in both).  CORE does not refuse deep offsets at 10,000.

The old value therefore capped CORE at 10,000 records for any broad query and
reported a shortfall where none had occurred.  That was the safe direction of
error, and it was still a real recall loss on a tool whose purpose is recall.

The replacement is **not** a measured CORE limit either -- offsets past 100,000
were not probed, and CORE's response time grows with depth.  It is a safety
stop that keeps a pathological run finite, set far enough out that it does not
bind on any query a literature pass would issue.  A run that reaches it still
stops as a reported ``PARTIAL`` naming the bound as the connector's own.
"""

_MIN_REQUEST_INTERVAL = 6.0
"""Seconds between requests.

**Documented, and the strictest tier is used because the tier is not
discoverable from the key.**  CORE's rate-limit table, read from
``api.core.ac.uk/docs/v3`` on 2026-08-31, has four tiers:

===================================================  ===================
Tier                                                 Allowance
===================================================  ===================
Unauthenticated                                      100/day, max 10/min
Registered personal                                  1,000/day, max 25/min
Registered academic, unaffiliated                    5,000/day, max 10/min
Supporting / sustaining institutions                 ~200,000/day
===================================================  ===================

Six seconds is 10 per minute, which is the strictest of the four.  The project
build plan's 2.0s is **30 per minute** and exceeds three of the four published
tiers, so it is not used.  Nothing in an API key says which tier it belongs to,
which is why the strictest is assumed rather than the likeliest.

**The unauthenticated tier is reachable**, contrary to what this docstring said
before 2026-08-31: a keyless search returns HTTP 200 (see the module
docstring's first retraction).  Its 10-per-minute ceiling is the same as the
strictest keyed tier, so this interval is correct for it too; only
:attr:`request_quota`'s daily figure differs, and it is stated for the keyed
tier.

Two further cautions.  The daily allowance is counted in *tokens*, not
requests: CORE states *"a simple query will cost you 1 token while more complex
queries will cost you between 3 to 5 tokens"*, and whether a paginated works
search counts as simple is not stated.  :attr:`request_quota` therefore counts
requests against the smallest keyed daily allowance, which under-counts if a
search costs more than one token.  Separately, CORE returns
``x-ratelimit-limit`` / ``x-ratelimit-remaining`` /
``x-ratelimit-retry-after`` headers (observed live) that would let the real
tier be read at runtime; the shared limiter takes a statically declared
interval, so that is recorded here as available rather than used.
"""

_CREDENTIAL_NAME = "core_api_key"

_DOCUMENT_TYPE_PAPER_TYPES: dict[str, PaperType] = {
    "book": PaperType.BOOK,
    "conference": PaperType.INPROCEEDINGS,
    "presentation": PaperType.MISC,
    "research": PaperType.ARTICLE,
    "slides": PaperType.MISC,
    "thesis": PaperType.PHDTHESIS,
}


def _first(item: dict[str, Any], *names: str) -> Any:
    """Return the first present, non-empty value among *names*.

    CORE's OpenAPI document and its human documentation disagree on whether
    response fields are ``camelCase`` or ``snake_case``, and no live response
    was available to settle it (see the module docstring).  Every field access
    therefore names both spellings rather than betting on one.

    Parameters
    ----------
    item : dict[str, Any]
        Object to read from.
    *names : str
        Candidate keys, in preference order.

    Returns
    -------
    Any
        The first value that is present and not ``None``/empty, else ``None``.
    """
    for name in names:
        value = item.get(name)
        if value not in (None, "", [], {}):
            return value
    return None


class CoreBodyError(ConnectorError):
    """A CORE response whose body is not a usable search result.

    Raised where ``raise_for_status`` is silent: a 200 response that is not
    JSON, is not an object, or carries no ``results`` list.  Treating any of
    those as an empty result would report a failure as zero papers.
    """


@dataclass(slots=True)
class _Retrieval:
    """Mutable bookkeeping for one paginated CORE run.

    Attributes
    ----------
    papers : list[Paper]
        Records parsed so far, in retrieval order.
    processed : int
        Records seen, whether or not they parsed.
    parsed : int
        Records that became a :class:`~lit_search.core.paper.Paper`.
    offset : int
        Offset the next request will carry.
    pages : int
        Pages consumed.
    total : int | None
        Reported result total, if CORE supplied one under either spelling.
    total_drift : str | None
        Set when a later page reported a different total.
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
    offset: int = 0
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
            Raw total.  Non-integers are ignored, which leaves :attr:`total`
            at ``None`` and makes ``COMPLETE`` unreachable -- the correct
            outcome when the envelope did not carry a count this connector
            recognised.
        """
        if not isinstance(total, int) or isinstance(total, bool):
            return
        if self.total is None:
            self.total = total
        elif total != self.total and self.total_drift is None:
            self.total_drift = (
                f"CORE reported {self.total} matching records on the first page and "
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


class CoreConnector(SearchConnectorBase):
    """Connector for CORE, the aggregator of open-access repositories.

    When no key is configured the connector raises
    :class:`~lit_search.exceptions.MissingApiKeyError` before issuing a
    request, which :meth:`SearchConnectorBase.search` records as
    ``SKIPPED_NO_CREDENTIAL``.  That is deliberately different from failing: a
    source nobody asked and a source that answered with nothing are different
    facts, and only one of them is fixed by configuration.

    **That behaviour now rests on a retracted premise.**  It was chosen because
    a keyless search was believed to return HTTP 429; re-checked live
    2026-08-31, a keyless search returns HTTP 200 with results at
    ``x-ratelimit-limit: 10`` (see the module docstring).  Until the premise is
    revisited, every run with no key reports ``SKIPPED_NO_CREDENTIAL`` for a
    source that would have answered -- and ``SearchRunner`` currently has no
    way to supply a CORE key, so that is every run.
    """

    _credential_attrs: ClassVar[tuple[str, ...]] = ("_api_key",)
    """CORE's key is stored under the default name, stated rather than assumed.

    The quota belongs to the key, so two processes holding one key must share
    one clock; leaving this to the default without checking is how a connector
    ends up in the anonymous bucket by accident.
    """

    def __init__(
        self,
        api_key: str | None = None,
        query_builder: CoreQueryBuilder | None = None,
    ) -> None:
        """Create a CORE searcher.

        Unlike :class:`~lit_search.connectors.ieee.IEEEConnector`, a missing
        key is **not** raised here.  Raising at construction removes the
        connector from the runner's searcher list, whereas raising at fetch
        time routes the same fact through the coverage contract, where it is
        one ``SKIPPED_NO_CREDENTIAL`` entry beside every other source's status.
        Both reach the ledger; this one keeps the connector constructible, so
        the behaviour can be tested without a key.

        Parameters
        ----------
        api_key : str | None
            CORE API key, obtainable at https://core.ac.uk/services/api.
        query_builder : CoreQueryBuilder | None
            Builder used to validate and convert queries.  When ``None`` a
            default :class:`CoreQueryBuilder` is created.
        """
        super().__init__()
        self._api_key: str | None = (api_key or "").strip() or None
        self._query_builder: CoreQueryBuilder = query_builder or CoreQueryBuilder()

    @property
    def name(self) -> str:
        """Return the database identifier.

        Returns
        -------
        str
            Database name.
        """
        return Database.CORE.value

    @property
    def query_builder(self) -> QueryBuilder:
        """Return the CORE query builder.

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
            Interval in seconds.  See :data:`_MIN_REQUEST_INTERVAL` for the
            published tier table and for why the strictest tier is used.
        """
        return _MIN_REQUEST_INTERVAL

    @property
    def rate_limit_service(self) -> str:
        """Return the quota authority these requests are paced against.

        Stated explicitly rather than inherited from :attr:`name` so that a
        future connector against another CORE endpoint -- works by identifier,
        data providers, the recommender -- returns the same string and shares
        this clock.  CORE's token bucket belongs to the key and is drawn on by
        every endpoint, and the more expensive endpoints draw on it harder.

        Returns
        -------
        str
            Quota-authority identifier.
        """
        return "core"

    @property
    def request_quota(self) -> RateLimitQuota:
        """Return CORE's daily allowance.

        Returns
        -------
        RateLimitQuota
            One thousand requests per day, the smallest keyed daily tier.
            CORE counts *tokens* rather than requests and a complex query may
            cost three to five, so this under-counts for expensive calls; it is
            the closest a request-counting quota can come to a token budget
            without a live measurement.
        """
        limit = 1000 if self._api_key else 100
        return RateLimitQuota(limit=limit, window_seconds=86400.0)

    def _prepare_headers(self, headers: dict) -> dict:
        """Add the bearer key.

        Parameters
        ----------
        headers : dict
            Raw HTTP headers.

        Returns
        -------
        dict
            Headers with ``Authorization`` and ``Accept`` set.  There is no
            keyless fallback: see the module docstring on the donor's
            401/403 retry-without-the-key path.
        """
        prepared = super()._prepare_headers(headers)
        prepared.setdefault("Accept", "application/json")
        if self._api_key:
            prepared["Authorization"] = f"Bearer {self._api_key}"
        return prepared

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
            Decoded payload, guaranteed to carry a ``results`` list.

        Raises
        ------
        CoreBodyError
            When the body is not JSON, is not an object, reports a ``message``
            in place of results, or carries no ``results`` list.
        """
        try:
            payload = response.json()
        except ValueError as exc:
            raise CoreBodyError(
                f"CORE returned a non-JSON body with status {response.status_code}: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise CoreBodyError(
                f"CORE returned a JSON {type(payload).__name__} where an object was expected."
            )

        if "results" not in payload:
            if "message" in payload:
                raise CoreBodyError(f"CORE reported an error: {payload['message']}")
            raise CoreBodyError(
                "CORE response carried no 'results' list, so it is not a search result. "
                "Treating it as an empty result set would report a failure as zero papers."
            )
        if not isinstance(payload["results"], list):
            raise CoreBodyError(
                f"CORE returned 'results' as {type(payload['results']).__name__}, not a list."
            )
        return payload

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_authors(item: dict[str, Any]) -> list[Author]:
        """Extract authors from a CORE work.

        Parameters
        ----------
        item : dict[str, Any]
            One ``results`` entry.

        Returns
        -------
        list[Author]
            Parsed authors in the order the record gives them.
        """
        raw = item.get("authors")
        entries = raw if isinstance(raw, list) else []
        authors: list[Author] = []
        for entry in entries:
            if isinstance(entry, str):
                if entry.strip():
                    authors.append(Author(name=entry.strip()))
                continue
            if isinstance(entry, dict):
                name = str(entry.get("name") or "").strip()
                if name:
                    authors.append(Author(name=name))
        return authors

    @staticmethod
    def _parse_publication_date(item: dict[str, Any]) -> datetime.date | None:
        """Resolve a publication date from the record's date fields.

        Parameters
        ----------
        item : dict[str, Any]
            One ``results`` entry.

        Returns
        -------
        datetime.date | None
            Parsed date, or ``None`` when no field is usable.  Falls back to
            the publication year, treated as 1 January.
        """
        raw = _first(item, "publishedDate", "published_date")
        parsed = parse_date(str(raw or "").strip()[:10] or None)
        if parsed is not None:
            return parsed
        year = str(_first(item, "yearPublished", "year_published") or "").strip()
        if year.isdigit():
            return parse_date(f"{year}-01-01")
        return None

    @staticmethod
    def _parse_urls(item: dict[str, Any], doi: str | None) -> tuple[str | None, str | None]:
        """Extract landing-page and PDF URLs from a CORE work.

        Parameters
        ----------
        item : dict[str, Any]
            One ``results`` entry.
        doi : str | None
            Normalised DOI, used as the landing-page fallback.

        Returns
        -------
        tuple[str | None, str | None]
            ``(url, pdf_url)``.
        """
        pdf_url: str | None = None
        download = _first(item, "downloadUrl", "download_url")
        if isinstance(download, str) and download.lower().endswith(".pdf"):
            pdf_url = download
        if pdf_url is None:
            full_text = _first(item, "sourceFulltextUrls", "source_fulltext_urls")
            for candidate in full_text if isinstance(full_text, list) else []:
                if isinstance(candidate, str) and candidate.lower().endswith(".pdf"):
                    pdf_url = candidate
                    break

        url: str | None = None
        links = item.get("links")
        for link in links if isinstance(links, list) else []:
            if isinstance(link, dict) and str(link.get("type") or "").lower() == "display":
                url = str(link.get("url") or "").strip() or None
                break
        if url is None and doi:
            url = f"https://doi.org/{doi}"
        if url is None and isinstance(download, str) and download.strip():
            url = download.strip()
        return url, pdf_url

    @staticmethod
    def _parse_source(item: dict[str, Any]) -> Source | None:
        """Build a :class:`Source` from a CORE work's journal or publisher.

        Parameters
        ----------
        item : dict[str, Any]
            One ``results`` entry.

        Returns
        -------
        Source | None
            Journal source when the record names one, else a repository source
            built from the publisher, else ``None``.
        """
        for journal in item.get("journals") or []:
            if not isinstance(journal, dict):
                continue
            title = str(journal.get("title") or "").strip()
            if title:
                identifiers = journal.get("identifiers")
                issn = next(
                    (
                        str(i).strip()
                        for i in (identifiers if isinstance(identifiers, list) else [])
                        if isinstance(i, str) and str(i).strip()
                    ),
                    None,
                )
                return Source(title=title, issn=issn, source_type=SourceType.JOURNAL)
        publisher = str(item.get("publisher") or "").strip()
        if publisher:
            return Source(title=publisher, source_type=SourceType.REPOSITORY)
        return None

    @staticmethod
    def _parse_keywords(item: dict[str, Any]) -> set[str]:
        """Extract subject and field-of-study terms from a CORE work.

        Parameters
        ----------
        item : dict[str, Any]
            One ``results`` entry.

        Returns
        -------
        set[str]
            Keyword strings; empty when the record carries none.
        """
        keywords: set[str] = set()
        for entry in item.get("subjects") or []:
            if isinstance(entry, str) and entry.strip():
                keywords.add(entry.strip())
        field_of_study = _first(item, "fieldOfStudy", "field_of_study")
        if isinstance(field_of_study, str) and field_of_study.strip():
            keywords.add(field_of_study.strip())
        return keywords

    def _parse_item(self, item: dict[str, Any]) -> Paper | None:
        """Parse one CORE work into a :class:`Paper`.

        Parameters
        ----------
        item : dict[str, Any]
            One ``results`` entry.

        Returns
        -------
        Paper | None
            Parsed paper, or ``None`` when a required field is missing or the
            record fails model validation.  Callers **must** count the ``None``
            returns: an unparsed record is a coverage loss, not an absent
            paper.
        """
        if not isinstance(item, dict):
            return None
        title = str(item.get("title") or "").strip()
        if not title:
            return None

        doi = normalize_doi(str(item.get("doi") or ""))
        url, pdf_url = self._parse_urls(item, doi)
        document_type = str(_first(item, "documentType", "document_type") or "").strip().lower()
        citations = _first(item, "citationCount", "citation_count")

        try:
            return Paper(
                title=title,
                abstract=str(item.get("abstract") or "").strip(),
                authors=self._parse_authors(item),
                source=self._parse_source(item),
                publication_date=self._parse_publication_date(item),
                url=url,
                pdf_url=pdf_url,
                doi=doi,
                citations=citations if isinstance(citations, int) else None,
                keywords=self._parse_keywords(item) or None,
                found_in={self.name},
                paper_type=_DOCUMENT_TYPE_PAPER_TYPES.get(document_type),
                language=self._parse_language(item),
                # CORE indexes open-access repositories, but the record itself
                # makes no access claim, so none is asserted here.
            )
        except ValueError:
            logger.debug("CORE: record %s failed model validation.", item.get("id"))
            return None

    @staticmethod
    def _parse_language(item: dict[str, Any]) -> str | None:
        """Extract the record's language code.

        Parameters
        ----------
        item : dict[str, Any]
            One ``results`` entry.

        Returns
        -------
        str | None
            Normalised language, or ``None``.
        """
        language = item.get("language")
        if isinstance(language, dict):
            return normalize_language(
                str(language.get("code") or language.get("name") or "") or None
            )
        return normalize_language(str(language or "") or None)

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

        CORE's documented searchable date field for ``works`` is
        ``yearPublished``, so a day-precision bound is widened to the calendar
        year that contains it.  That widening is safe in the direction that
        matters -- it can only admit extra records, never hide matching ones --
        and the caller's own ``since``/``until`` are applied again downstream.

        **The comparison form is required; the Lucene range form is an error.**
        This method emitted ``yearPublished:[2020 TO 2021]`` until 2026-08-31,
        when a live probe returned **HTTP 500**: *"Azure search failed with
        status code: 400 ... Invalid expression: A binary operator with
        incompatible types was detected. Found operand types 'Edm.Int32' and
        'Edm.String' for operator kind 'Equal'."*  CORE's search is Azure
        Cognitive Search and ``yearPublished`` is an integer field, so bracket
        ranges do not apply to it.  The consequence was total: **every search
        carrying a ``since`` or an ``until`` ended as ``FAILED`` for CORE**,
        for every query, and the open-ended ``*`` bound failed the same way.
        ``yearPublished>=2020`` returned HTTP 200 with 36,737,162 records in
        the same probe, so the comparison form is what is emitted now.

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
            CORE ``q`` expression.
        """
        expression = self._query_builder.convert_query(query)
        bounds: list[str] = []
        if since is not None:
            bounds.append(f"yearPublished>={since.year}")
        if until is not None:
            bounds.append(f"yearPublished<={until.year}")
        if not bounds:
            return expression
        date_filter = " AND ".join(bounds)
        return f"({expression}) AND {date_filter}" if expression else date_filter

    def _fetch_papers(
        self,
        query: Query,
        max_papers: int | None,
        progress_callback: Callable[[int, int | None], None] | None,
        since: datetime.date | None = None,
        until: datetime.date | None = None,
    ) -> CoveredPapers:
        """Fetch papers from CORE with offset pagination and coverage.

        Every way this loop can end is tracked separately:

        - no API key -> raised, so the base class reports
          ``SKIPPED_NO_CREDENTIAL`` **without issuing a request**;
        - the caller's ``max_papers`` was reached -> ``CAPPED``;
        - a page request failed, or its body was unusable, with nothing
          retrieved -> raised, so the base class reports ``FAILED``;
        - the same failure *after* records were retrieved -> ``PARTIAL``
          carrying the error;
        - the offset ceiling was reached -> ``PARTIAL``;
        - the reported total changed between pages -> ``PARTIAL``;
        - CORE returned an empty page -> ``from_pagination`` decides, and it
          returns ``COMPLETE`` only when as many records came back as CORE said
          existed.  When no total could be read, it returns ``PARTIAL``.

        Parameters
        ----------
        query : Query
            Validated query object.
        max_papers : int | None
            Maximum papers to retrieve.  ``None`` means unlimited.
        progress_callback : Callable[[int, int | None], None] | None
            Progress callback called as ``callback(processed, total_or_none)``.
        since : datetime.date | None
            Only return records published on or after this date.
        until : datetime.date | None
            Only return records published on or before this date.

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
        expression = self._build_query_string(query, since, until)
        run = _Retrieval()

        while True:
            remaining = (max_papers - len(run.papers)) if max_papers is not None else _PAGE_SIZE
            limit = max(1, min(_PAGE_SIZE, remaining))
            params = {"q": expression, "limit": limit, "offset": run.offset}

            try:
                payload = self._read_body(self._get(_SEARCH_URL, params))
            except (requests.RequestException, ConnectorError) as exc:
                self._record_page_failure(run, exc)
                break

            seen = self._absorb_page(run, payload)
            if progress_callback is not None:
                progress_callback(run.processed, run.total)

            if not self._should_continue(run, seen, limit, max_papers):
                break
            run.offset += seen

        if progress_callback is not None:
            progress_callback(run.processed, run.total)

        kept = run.papers[:max_papers] if max_papers is not None else run.papers
        return CoveredPapers(
            papers=tuple(kept),
            coverage=self._build_coverage(run, retrieved=len(kept), max_papers=max_papers),
        )

    @staticmethod
    def _record_page_failure(run: _Retrieval, exc: Exception) -> None:
        """Record a page failure, or re-raise when nothing was retrieved.

        This replaces the donor's ``if response is None: return papers``, which
        turned three exhausted retries into a successful empty search with no
        exception raised anywhere.

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
            connector_label="CORE",
            processed=run.processed,
            total=run.total,
            position_label="offset",
            position=run.offset,
            exc=exc,
        )
        run.truncate(f"page request at offset={run.offset} failed: {exc}", type(exc).__name__)

    def _absorb_page(self, run: _Retrieval, payload: dict[str, Any]) -> int:
        """Fold one page into *run* and return how many records it carried.

        Parameters
        ----------
        run : _Retrieval
            Bookkeeping for the current run, updated in place.
        payload : dict[str, Any]
            Validated response body.

        Returns
        -------
        int
            Number of raw entries on the page, including any that failed to
            parse -- the caller needs the count, because a record seen but not
            parsed is a coverage loss rather than an absent paper.
        """
        run.note_total(_first(payload, "totalHits", "total_hits"))
        results = payload["results"]
        for item in results:
            paper = self._parse_item(item)
            if paper is not None:
                run.papers.append(paper)
                run.parsed += 1
        run.processed += len(results)
        run.pages += 1
        return len(results)

    @staticmethod
    def _should_continue(
        run: _Retrieval,
        seen: int,
        limit: int,
        max_papers: int | None,
    ) -> bool:
        """Decide whether to request another page.

        Parameters
        ----------
        run : _Retrieval
            Bookkeeping for the current run, updated in place.
        seen : int
            Number of records the page just absorbed carried.
        limit : int
            ``limit`` that produced this page.
        max_papers : int | None
            Caller-imposed cap.

        Returns
        -------
        bool
            ``True`` to request another page.
        """
        if max_papers is not None and len(run.papers) >= max_papers:
            return False

        if seen == 0:
            # An empty page is CORE's only available end-of-set signal: the
            # envelope carries no next-page link and no cursor.  A short page
            # is deliberately not treated as the end, because the reference
            # port proved that heuristic wrong on a sibling API and CORE
            # documents no guarantee that a page is full.
            if run.total is not None and run.processed < run.total:
                run.truncate(
                    f"CORE returned an empty page after {run.processed} of {run.total} "
                    f"reported records (offset {run.offset})"
                )
            return False

        if run.total is not None and run.processed >= run.total:
            return False

        if run.offset + seen + limit > _MAX_OFFSET:
            run.truncate(
                f"retrieval stopped at the {_MAX_OFFSET}-record offset ceiling after "
                f"{run.processed} of "
                f"{run.total if run.total is not None else 'an unknown number of'} "
                f"matching records; CORE's deep-pagination limit is undocumented and "
                f"this bound is the connector's own"
            )
            return False
        return True

    def _build_coverage(
        self,
        run: _Retrieval,
        *,
        retrieved: int,
        max_papers: int | None,
    ) -> SourceCoverage:
        """Turn the pagination loop's own counters into a coverage status.

        Parameters
        ----------
        run : _Retrieval
            Bookkeeping for the completed run.
        retrieved : int
            Papers actually returned to the caller, after the cap is applied.
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
        unparsed = run.processed - run.parsed
        if unparsed > 0:
            notes.append(f"{unparsed} of {run.processed} records could not be parsed into papers")
        shortfall = "; ".join(notes) or None

        if run.truncated_early or run.total_drift:
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
