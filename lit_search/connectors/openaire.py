"""OpenAIRE searcher implementation.

Rewritten from ``paper_search_mcp/academic_platforms/openaire.py`` at pinned
commit ``234678ab231074a7977320978ee0496dcdaddd1f``.  This is the port on which
least of the donor survives, and the reasons are worth stating because two of
them are decisions rather than translations.

**The error contract.**  The donor's ``OpenAiresearcher.search`` has six
fall-through paths to a partial or empty list.  ``_search_with_retry``
(``openaire.py:45-125``) walks three request profiles, each up to three times,
inside ``except Exception as exc: last_error = exc``; ``search``
(``openaire.py:290-321``) then catches whatever that raises with
``logger.warning(...)`` and drops to a legacy JSON path whose own
``except Exception`` ends at ``return papers``.  An outage, a schema change, a
malformed query and an empty repository are one value at the caller.
``_get`` (``openaire.py:151-159``) additionally retries with
``verify=False`` on any ``SSLError``, so a man-in-the-middle presenting a bad
certificate is silently accepted; that behaviour is not carried over.

**The endpoint.**  The donor drives ``api.openaire.eu/search/researchProducts``,
OpenAIRE's pre-Graph XML API.  That endpoint is still live (HTTP 200,
``<total>278812</total>`` for ``keywords=catalysis``, verified 2026-08-31), so
this is not a repair of a dead path -- it is a deliberate move to
``api.openaire.eu/graph/v3/research-products``, which OpenAIRE's own
documentation describes as the current API.  The Graph API earns the switch on
contract grounds rather than taste: it reports ``header.numFound`` and offers
``cursor`` / ``nextCursor`` deep pagination, and a source-reported total is the
only thing that makes a ``COMPLETE`` claim justifiable at all.  ``v1``, ``v2``,
``v3`` and ``v4`` were all live on 2026-08-31; ``v3`` is pinned here because it
is the version the documentation calls current and the one whose 60 field
parameters the query builder is written against.

Three behaviours of the live API drive the design.  All were observed directly
on 2026-08-31:

- **The offset window is 10,000 and it is a hard error.**
  ``pageSize=10&page=1001`` returned HTTP 400 ``{"message": "Page * pageSize
  must be less than or equal to 10000", ...}``.  Cursor pagination is used
  instead, so that the window is not a ceiling on recall.
- **An unknown parameter is a loud HTTP 400**, naming the valid parameters --
  unusually helpful, and the reason a malformed request here cannot become a
  silent zero.
- **An empty query is the whole graph.**  ``search=`` returned
  ``numFound == 386,695,835``.  Handled in the query builder.

Rate limit: see :data:`_MIN_REQUEST_INTERVAL_ANONYMOUS`, which records a
conflict between OpenAIRE's documentation and its own response headers.
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

from lit_search.connectors.openaire_auth import OpenAIRETokenManager
from lit_search.connectors.search_base import SearchConnectorBase
from lit_search.core.author import Author
from lit_search.core.coverage import CoveredPapers, SourceCoverage
from lit_search.core.paper import Database, Paper, PaperType
from lit_search.core.query import Query
from lit_search.core.source import Source, SourceType
from lit_search.exceptions import ConnectorError
from lit_search.query.builders.openaire import OpenAIREQueryBuilder
from lit_search.utils.normalization import normalize_doi, normalize_language, parse_date
from lit_search.utils.pagination_logging import log_pagination_request_failure
from lit_search.utils.rate_limit import RateLimitQuota

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.openaire.eu/graph/v3"
_SEARCH_URL = f"{_BASE_URL}/research-products"

_PAGE_SIZE = 100

_MIN_REQUEST_INTERVAL_ANONYMOUS = 60.0
"""Seconds between requests without a token.

**Documented, and contradicted by the service's own headers.**
``graph.openaire.eu/docs/apis/terms`` states: *"The rate limit for the former
type of requests is up to 7200 requests per hour, while the latter is up to 60
requests per hour"* -- 7,200/hour authenticated, 60/hour anonymous.  Sixty per
hour is one per sixty seconds.

Live on 2026-08-31, an **unauthenticated** request to
``/graph/v3/research-products`` returned ``x-ratelimit-used: 6`` and
``x-ratelimit-limit: 7199``, and the legacy ``/search/researchProducts``
endpoint returned the same pair.  The header advertises the authenticated
allowance to an anonymous caller, so the documentation and the header disagree
by a factor of 120 and there is no way to tell from outside which one the
gateway actually enforces.

The documented number is used, because the two errors are not symmetric.  Too
slow costs wall-clock time on a search that still returns everything.  Too fast
exhausts an hourly budget in seconds and then returns errors for the rest of
the hour, and those errors land in the ledger as ``FAILED`` against *OpenAIRE*
-- reading as an upstream outage when the cause was our own pacing.  Revisit if
OpenAIRE publishes a corrected figure, or if a token-holding run shows the
header tracking the real allowance.
"""

_MIN_REQUEST_INTERVAL_AUTHENTICATED = 0.5
"""Seconds between requests with a token.

7,200 requests per hour is one every half second.  Documented on
``graph.openaire.eu/docs/apis/terms``; see
:data:`_MIN_REQUEST_INTERVAL_ANONYMOUS` for why the documented figure is
preferred over the live header.
"""

# Hard ceiling on pagination rounds.  Only reachable if the API keeps handing
# out fresh cursors indefinitely; it exists so that such a loop ends as an
# honest PARTIAL rather than running forever.
_MAX_PAGES = 10_000

_INSTANCE_PAPER_TYPES: dict[str, PaperType] = {
    "article": PaperType.ARTICLE,
    "book": PaperType.BOOK,
    "conference object": PaperType.INPROCEEDINGS,
    "conference paper": PaperType.INPROCEEDINGS,
    "doctoral thesis": PaperType.PHDTHESIS,
    "master thesis": PaperType.MASTERSTHESIS,
    "part of book or chapter of book": PaperType.INBOOK,
    "preprint": PaperType.UNPUBLISHED,
    "report": PaperType.TECHREPORT,
}


class OpenAIREBodyError(ConnectorError):
    """An OpenAIRE response whose body is not a usable search result.

    Raised where ``raise_for_status`` is silent: a 200 response that is not
    JSON, is not an object, or carries no ``header``/``results`` envelope.
    Treating any of those as an empty result would report a failure as zero
    records.
    """


@dataclass(slots=True)
class _Retrieval:
    """Mutable bookkeeping for one paginated OpenAIRE run.

    Exists so that the reasons a run can end stay separable all the way to
    :meth:`OpenAIREConnector._build_coverage`.

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
        ``header.numFound`` from the first page.
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
    total_drift: str | None = None
    truncated_early: bool = False
    error_detail: str | None = None
    error_type: str | None = None

    def note_total(self, num_found: object) -> None:
        """Record the reported total, or the fact that it moved.

        Parameters
        ----------
        num_found : object
            Raw ``header.numFound``.  Non-integers are ignored, which leaves
            :attr:`total` at ``None`` and makes ``COMPLETE`` unreachable.
        """
        if not isinstance(num_found, int) or isinstance(num_found, bool):
            return
        if self.total is None:
            self.total = num_found
        elif num_found != self.total and self.total_drift is None:
            self.total_drift = (
                f"OpenAIRE reported {self.total} matching records on the first page and "
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


class OpenAIREConnector(SearchConnectorBase):
    """Connector for the OpenAIRE Graph API.

    OpenAIRE aggregates research products from thousands of repositories,
    publishers and funders across Europe and beyond.  Search works without a
    credential, at a documented 60 requests per hour; an access token raises
    that to 7,200 per hour, which is a 120-fold difference and effectively
    makes the token mandatory for any real pass.  Absence of a token is
    therefore **not** ``SKIPPED_NO_CREDENTIAL``: the source answers, and
    reporting it as unreachable would be its own kind of dishonesty.
    """

    _credential_attrs: ClassVar[tuple[str, ...]] = ("_quota_scope_credential",)
    """Stable credential material identifying the OpenAIRE quota account.

    Access tokens are short lived and may differ between processes that share
    one month-lived refresh token.  Hashing both would split one account's
    hourly allowance across multiple rate-limit buckets.  The constructor
    therefore freezes the originally supplied refresh token as the quota
    identity when present, falling back to the static access token otherwise.
    The raw value remains in memory and is only exposed to the base class's
    one-way fingerprinting path.
    """

    def __init__(
        self,
        access_token: str | None = None,
        refresh_token: str | None = None,
        access_token_expires_in: float | None = None,
        query_builder: OpenAIREQueryBuilder | None = None,
    ) -> None:
        """Create an OpenAIRE searcher.

        Parameters
        ----------
        access_token : str | None
            Optional one-hour OpenAIRE personal access token.
        refresh_token : str | None
            Optional month-lived OpenAIRE personal refresh token.  When
            supplied, the connector obtains and refreshes bearer tokens
            automatically.  A failed refresh is reported as an authentication
            failure and never falls back to anonymous access.
        access_token_expires_in : float | None
            Remaining lifetime of ``access_token`` in seconds.  Used only with
            ``refresh_token``.  If omitted, an accompanying access token is
            refreshed on first use rather than trusting an unknown age.
        query_builder : OpenAIREQueryBuilder | None
            Builder used to validate and convert queries.  When ``None`` a
            default :class:`OpenAIREQueryBuilder` is created.
        """
        super().__init__()
        self._access_token: str | None = (access_token or "").strip() or None
        self._refresh_token: str | None = (refresh_token or "").strip() or None
        self._quota_scope_credential: str | None = self._refresh_token or self._access_token
        self._token_manager: OpenAIRETokenManager | None = None
        if self._access_token is not None or self._refresh_token is not None:
            self._token_manager = OpenAIRETokenManager(
                access_token=self._access_token,
                refresh_token=self._refresh_token,
                access_token_expires_in=access_token_expires_in,
            )
        self._query_builder: OpenAIREQueryBuilder = query_builder or OpenAIREQueryBuilder()

    @property
    def name(self) -> str:
        """Return the database identifier.

        Returns
        -------
        str
            Database name.
        """
        return Database.OPENAIRE.value

    @property
    def query_builder(self) -> QueryBuilder:
        """Return the OpenAIRE query builder.

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
            0.5 with a token, 60.0 without.  See
            :data:`_MIN_REQUEST_INTERVAL_ANONYMOUS` for the documented figures
            and for the conflict with the live headers.
        """
        if self._token_manager is not None:
            return _MIN_REQUEST_INTERVAL_AUTHENTICATED
        return _MIN_REQUEST_INTERVAL_ANONYMOUS

    @property
    def rate_limit_service(self) -> str:
        """Return the quota authority these requests are paced against.

        Stated explicitly rather than inherited from :attr:`name` so that a
        future connector against another OpenAIRE endpoint -- projects,
        organisations, the legacy XML search -- returns the same string and
        shares this clock.  The hourly allowance belongs to the account, not to
        the endpoint.

        Returns
        -------
        str
            Quota-authority identifier.
        """
        return "openaire"

    @property
    def request_quota(self) -> RateLimitQuota:
        """Return OpenAIRE's hourly allowance.

        Returns
        -------
        RateLimitQuota
            7,200 per hour with a token, 60 per hour without.  This is the
            case the quota mechanism exists for: pacing at an interval alone
            still lets a burst that arrives after an idle hour spend the whole
            budget at once, after which OpenAIRE returns errors for the rest of
            the window and the ledger reads them as an outage.
        """
        if self._token_manager is not None:
            return RateLimitQuota(limit=7200, window_seconds=3600.0)
        return RateLimitQuota(limit=60, window_seconds=3600.0)

    def _prepare_headers(self, headers: dict) -> dict:
        """Add the bearer token when one was supplied.

        Parameters
        ----------
        headers : dict
            Raw HTTP headers.

        Returns
        -------
        dict
            Headers, with ``Authorization`` added when a token is configured.
        """
        prepared = super()._prepare_headers(headers)
        if self._token_manager is not None:
            token = self._token_manager.get_access_token(self._get_session())
            # Keep the connector's declared credential material synchronized
            # for rate-limit scoping and future credential-store integration.
            self._access_token = token
            self._refresh_token = self._token_manager.refresh_token
            prepared["Authorization"] = f"Bearer {token}"
        return prepared

    def _get(
        self,
        url: str,
        params: dict | None = None,
        headers: dict | None = None,
    ) -> requests.Response:
        """GET once, refreshing and retrying once after bearer rejection.

        Only HTTP 401/403 from an authenticated request with a configured
        refresh token activates this path.  The second request calls the base
        implementation directly, so another rejection propagates rather than
        recursing.  Static access-token-only and anonymous requests are never
        retried here.
        """
        try:
            return super()._get(url, params, headers)
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            manager = self._token_manager
            if status not in {401, 403} or manager is None or not manager.can_refresh:
                raise

            token = manager.force_refresh(self._get_session())
            self._access_token = token
            self._refresh_token = manager.refresh_token
            return super()._get(url, params, headers)

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
            Decoded payload, guaranteed to carry ``header`` and ``results``.

        Raises
        ------
        OpenAIREBodyError
            When the body is not JSON, is not an object, or lacks the envelope
            every real search response carries.  Each would otherwise be read
            as "OpenAIRE holds no matching record".
        """
        try:
            payload = response.json()
        except ValueError as exc:
            raise OpenAIREBodyError(
                f"OpenAIRE returned a non-JSON body with status {response.status_code}: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise OpenAIREBodyError(
                f"OpenAIRE returned a JSON {type(payload).__name__} where an object was expected."
            )

        if "message" in payload and "results" not in payload:
            raise OpenAIREBodyError(f"OpenAIRE reported an error: {payload['message']}")

        if not isinstance(payload.get("header"), dict) or "results" not in payload:
            raise OpenAIREBodyError(
                "OpenAIRE response carried no 'header'/'results' envelope, so it is not a "
                "search result. Treating it as an empty result set would report a failure "
                "as zero papers."
            )
        return payload

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_authors(item: dict[str, Any]) -> list[Author]:
        """Extract authors from a Graph API research product.

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
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("fullName") or "").strip()
            if not name:
                given = str(entry.get("name") or "").strip()
                family = str(entry.get("surname") or "").strip()
                name = f"{given} {family}".strip()
            if name:
                authors.append(Author(name=name))
        return authors

    @staticmethod
    def _parse_doi(item: dict[str, Any]) -> str | None:
        """Extract a DOI from the record's persistent identifiers.

        Parameters
        ----------
        item : dict[str, Any]
            One ``results`` entry.

        Returns
        -------
        str | None
            Normalised DOI, or ``None`` when the record carries none.
        """
        for pid in item.get("pids") or []:
            if not isinstance(pid, dict):
                continue
            if str(pid.get("scheme") or "").strip().lower() != "doi":
                continue
            normalised = normalize_doi(str(pid.get("value") or ""))
            if normalised:
                return normalised
        for instance in item.get("instances") or []:
            if not isinstance(instance, dict):
                continue
            for pid in instance.get("pids") or []:
                if isinstance(pid, dict) and str(pid.get("scheme") or "").lower() == "doi":
                    normalised = normalize_doi(str(pid.get("value") or ""))
                    if normalised:
                        return normalised
        return None

    @staticmethod
    def _parse_urls(item: dict[str, Any], doi: str | None) -> tuple[str | None, str | None]:
        """Extract landing-page and PDF URLs from a record's instances.

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
        url: str | None = None
        pdf_url: str | None = None
        for instance in item.get("instances") or []:
            if not isinstance(instance, dict):
                continue
            for candidate in instance.get("urls") or []:
                if not isinstance(candidate, str) or not candidate.startswith("http"):
                    continue
                if url is None:
                    url = candidate
                lowered = candidate.lower()
                if pdf_url is None and (lowered.endswith(".pdf") or "/pdf" in lowered):
                    pdf_url = candidate
        if url is None and doi:
            url = f"https://doi.org/{doi}"
        return url, pdf_url

    @staticmethod
    def _parse_keywords(item: dict[str, Any]) -> set[str]:
        """Extract subject terms from a record.

        Parameters
        ----------
        item : dict[str, Any]
            One ``results`` entry.

        Returns
        -------
        set[str]
            Subject strings; empty when the record carries none.
        """
        keywords: set[str] = set()
        for entry in item.get("subjects") or []:
            if not isinstance(entry, dict):
                continue
            subject = entry.get("subject")
            if isinstance(subject, dict):
                value = str(subject.get("value") or "").strip()
            else:
                value = str(subject or "").strip()
            if value:
                keywords.add(value)
        return keywords

    @staticmethod
    def _parse_source(item: dict[str, Any], paper_type: PaperType | None) -> Source | None:
        """Build a :class:`Source` from a record's container or publisher.

        Parameters
        ----------
        item : dict[str, Any]
            One ``results`` entry.
        paper_type : PaperType | None
            Inferred paper type, used to classify an unnamed venue.

        Returns
        -------
        Source | None
            Journal or repository source, or ``None`` when the record names no
            venue at all.
        """
        container = item.get("container")
        if isinstance(container, dict):
            title = str(container.get("name") or "").strip()
            if title:
                issn = str(
                    container.get("issnPrinted") or container.get("issnOnline") or ""
                ).strip()
                return Source(title=title, issn=issn or None, source_type=SourceType.JOURNAL)

        publisher = str(item.get("publisher") or "").strip()
        if publisher:
            source_type = (
                SourceType.REPOSITORY if paper_type is PaperType.UNPUBLISHED else SourceType.JOURNAL
            )
            return Source(title=publisher, source_type=source_type)
        return None

    @staticmethod
    def _parse_paper_type(item: dict[str, Any]) -> PaperType | None:
        """Infer a :class:`PaperType` from the record's instance types.

        Parameters
        ----------
        item : dict[str, Any]
            One ``results`` entry.

        Returns
        -------
        PaperType | None
            Mapped type, or ``None`` when no instance names a known one.
        """
        for instance in item.get("instances") or []:
            if not isinstance(instance, dict):
                continue
            mapped = _INSTANCE_PAPER_TYPES.get(str(instance.get("type") or "").strip().lower())
            if mapped is not None:
                return mapped
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
            return normalize_language(str(language.get("code") or "") or None)
        return normalize_language(str(language or "") or None)

    def _parse_item(self, item: dict[str, Any]) -> Paper | None:
        """Parse one Graph API research product into a :class:`Paper`.

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
        title = str(item.get("mainTitle") or "").strip()
        if not title:
            return None

        descriptions = item.get("descriptions")
        abstract = ""
        if isinstance(descriptions, list):
            abstract = next(
                (str(d).strip() for d in descriptions if isinstance(d, str) and d.strip()), ""
            )
        elif isinstance(descriptions, str):
            abstract = descriptions.strip()

        doi = self._parse_doi(item)
        url, pdf_url = self._parse_urls(item, doi)
        paper_type = self._parse_paper_type(item)
        access = str(item.get("bestAccessRight") or "")
        if isinstance(item.get("bestAccessRight"), dict):
            access = str(item["bestAccessRight"].get("label") or "")

        try:
            return Paper(
                title=title,
                abstract=abstract,
                authors=self._parse_authors(item),
                source=self._parse_source(item, paper_type),
                publication_date=parse_date(str(item.get("publicationDate") or "").strip() or None),
                url=url,
                pdf_url=pdf_url,
                doi=doi,
                keywords=self._parse_keywords(item) or None,
                found_in={self.name},
                paper_type=paper_type,
                language=self._parse_language(item),
                # Only the positive case is asserted: a non-open access label
                # is a statement about OpenAIRE's best known instance, not
                # evidence that no open version exists.
                is_open_access=True if "open" in access.lower() else None,
            )
        except ValueError:
            logger.debug("OpenAIRE: record %s failed model validation.", item.get("id"))
            return None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _build_params(
        self,
        query: Query,
        since: datetime.date | None,
        until: datetime.date | None,
    ) -> dict[str, Any]:
        """Build the field and date parameters for the search request.

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
        dict[str, Any]
            Query parameters, excluding pagination.
        """
        params: dict[str, Any] = dict(self._query_builder.convert_query(query))
        if since is not None:
            params["fromPublicationDate"] = since.isoformat()
        if until is not None:
            params["toPublicationDate"] = until.isoformat()
        return params

    def _fetch_papers(
        self,
        query: Query,
        max_papers: int | None,
        progress_callback: Callable[[int, int | None], None] | None,
        since: datetime.date | None = None,
        until: datetime.date | None = None,
    ) -> CoveredPapers:
        """Fetch papers from OpenAIRE with cursor pagination and coverage.

        Cursor pagination is used rather than ``page``/``pageSize`` because the
        offset form is refused past 10,000 records with HTTP 400 (verified
        2026-08-31), which would make the tenth thousand a silent ceiling on
        recall for any broad query.

        Every way this loop can end is tracked separately:

        - the caller's ``max_papers`` was reached -> ``CAPPED``;
        - a page request failed, or its body was unusable, with nothing
          retrieved -> raised, so the base class reports ``FAILED``;
        - the same failure *after* records were retrieved -> ``PARTIAL``
          carrying the error;
        - the cursor stopped advancing before ``numFound`` was reached ->
          ``PARTIAL``;
        - ``numFound`` changed between pages -> ``PARTIAL``;
        - the cursor ran out naturally -> ``from_pagination`` decides.

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
        base_params = self._build_params(query, since, until)
        run = _Retrieval()
        cursor = "*"

        while True:
            remaining = (max_papers - len(run.papers)) if max_papers is not None else _PAGE_SIZE
            params = {
                **base_params,
                "pageSize": max(1, min(_PAGE_SIZE, remaining)),
                "cursor": cursor,
            }

            try:
                payload = self._read_body(self._get(_SEARCH_URL, params))
            except (requests.RequestException, ConnectorError) as exc:
                self._record_page_failure(run, exc, cursor)
                break

            results = self._absorb_page(run, payload)
            if progress_callback is not None:
                progress_callback(run.processed, run.total)

            advanced = self._advance(run, payload, results, cursor, max_papers)
            if advanced is None:
                break
            cursor = advanced

        if progress_callback is not None:
            progress_callback(run.processed, run.total)

        kept = run.papers[:max_papers] if max_papers is not None else run.papers
        return CoveredPapers(
            papers=tuple(kept),
            coverage=self._build_coverage(run, retrieved=len(kept), max_papers=max_papers),
        )

    @staticmethod
    def _record_page_failure(run: _Retrieval, exc: Exception, cursor: str) -> None:
        """Record a page failure, or re-raise when nothing was retrieved.

        The asymmetry is the point of the port.  With no records in hand the
        failure is the whole story and is re-raised for
        :meth:`SearchConnectorBase.search` to name; with records in hand the
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
            connector_label="OpenAIRE",
            processed=run.processed,
            total=run.total,
            position_label="cursor",
            position=cursor,
            exc=exc,
        )
        run.truncate(f"page request at cursor={cursor} failed: {exc}", type(exc).__name__)

    def _absorb_page(self, run: _Retrieval, payload: dict[str, Any]) -> list[Any]:
        """Fold one page into *run* and return the raw records it carried.

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
        run.note_total(payload["header"].get("numFound"))
        raw = payload.get("results")
        results: list[Any] = raw if isinstance(raw, list) else []
        for item in results:
            paper = self._parse_item(item)
            if paper is not None:
                run.papers.append(paper)
                run.parsed += 1
        run.processed += len(results)
        run.pages += 1
        return results

    @staticmethod
    def _advance(
        run: _Retrieval,
        payload: dict[str, Any],
        results: list[Any],
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
        results : list[Any]
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
        raw_cursor = payload["header"].get("nextCursor")
        next_cursor = raw_cursor if isinstance(raw_cursor, str) and raw_cursor else None

        if max_papers is not None and len(run.papers) >= max_papers:
            return None

        if next_cursor is None or next_cursor == cursor:
            # OpenAIRE omits nextCursor once the result set is exhausted.  A
            # repeated cursor is treated as exhaustion too, because following
            # it would re-read the same page forever.
            if run.total is not None and run.processed < run.total:
                run.truncate(
                    f"OpenAIRE stopped issuing cursors after {run.processed} of "
                    f"{run.total} reported records (page {run.pages})"
                )
            return None

        if not results:
            # The cursor advanced but the page was empty.  Continuing would
            # spin; stopping without saying so would under-report.
            run.truncate(
                f"OpenAIRE returned an empty page at page {run.pages} while still "
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
