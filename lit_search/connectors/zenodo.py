"""Zenodo searcher implementation.

Ported from ``paper_search_mcp/academic_platforms/zenodo.py`` at pinned commit
``234678ab231074a7977320978ee0496dcdaddd1f``.  The port is **not** a mechanical
translation.  The donor's ``search`` wraps the whole request-and-decode block
in ``except Exception as exc: logger.error(...); return []``
(``zenodo.py:95-103``), so a dead endpoint, an expired certificate, a changed
response shape and a genuinely empty repository all reach the caller as the
same empty list.  See ``PORTING-GUIDE.md`` for the before-and-after.

The donor also **does not paginate at all**.  It sends one request with
``size = max(1, min(max_results, 200))`` and returns whatever comes back, so a
query matching 3,328 records silently becomes at most 200 -- and, as it turns
out, zero.

Five behaviours of the live API drive the design here.  All were observed
directly on 2026-08-31; the first is a live defect in the donor.

- **The donor's page size is refused outright.**  ``size=1001`` -- and
  anything above 25 without a token -- returns HTTP 400
  ``{"errors": [{"field": "size", "messages": ["Page size cannot be greater
  than 25. Please use authenticated requests to increase the limit to 100."]}]}``.
  The donor's ``min(max_results, 200)`` therefore fails for any
  ``max_results > 25`` on an anonymous client, and its ``except Exception:
  return []`` reports that failure as "Zenodo holds no matching record".  The
  project specification's own note that the range is "1-200" is wrong for the
  same reason.
- **An empty query is a match-all.**  ``q=`` returned ``hits.total ==
  7,218,253``.  Handled in the query builder, not here.
- **A malformed query is not an error.**  ``q=title:(((`` returned HTTP 200
  with 47,131 records against 772 for ``q=title:catalysis``.  Also handled in
  the query builder.
- **An unknown record type is a silent zero.**  ``type=BOGUSTYPE`` returned
  ``hits.total == 0`` with HTTP 200, so the ``type`` filter is a fixed
  constant here and never carries caller-supplied text.
- **Pagination ends by the absence of ``links.next``, and the offset window is
  bounded.**  ``page=1001&size=10`` returned HTTP 400 ``"Invalid querystring
  parameters."``, so ``page * size`` may not exceed 10,000.  That ceiling is a
  real coverage boundary and is reported as a shortfall rather than treated as
  exhaustion.

Rate limit: 30 requests per minute on search endpoints.  See
:data:`_MIN_REQUEST_INTERVAL`.
"""

from __future__ import annotations

import datetime
import logging
import re
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
from lit_search.query.builders.zenodo import ZenodoQueryBuilder
from lit_search.utils.normalization import normalize_doi, parse_date
from lit_search.utils.pagination_logging import log_pagination_request_failure
from lit_search.utils.rate_limit import RateLimitQuota

logger = logging.getLogger(__name__)

_BASE_URL = "https://zenodo.org/api"
_RECORDS_URL = f"{_BASE_URL}/records"

# Verified live 2026-08-31: size=26 without a token is refused with
# "Page size cannot be greater than 25. Please use authenticated requests to
# increase the limit to 100."  The donor's 200 is rejected by the API.
_MAX_PAGE_SIZE_ANONYMOUS = 25
_MAX_PAGE_SIZE_AUTHENTICATED = 100

# Verified live 2026-08-31: page=1001 with size=10 returns HTTP 400
# "Invalid querystring parameters."  Zenodo's own documentation does not state
# this ceiling; it was found by probing, and it is treated as a coverage
# boundary rather than as the end of the result set.
_MAX_RESULT_WINDOW = 10_000

# Zenodo record type restricting results to publications.  Verified live
# 2026-08-31: for q=catalysis the unfiltered count was 4,393, type=publication
# gave 3,328 and type=dataset gave 690, so the filter is real.  It is a
# constant and never interpolates caller text, because type=BOGUSTYPE returns
# zero records with HTTP 200 rather than an error.
_PUBLICATION_TYPE = "publication"

# Relevance ordering.  The donor hardcodes "mostrecent", which for a capped
# search returns the newest N records rather than the best-matching N -- a
# materially different set for a recall tool.  Verified accepted live
# 2026-08-31; an invalid value is refused with HTTP 400, not ignored.
_SORT = "bestmatch"

_MIN_REQUEST_INTERVAL = 2.0
"""Seconds between requests.

**Documented, not inferred.**  ``developers.zenodo.org`` states 60 requests per
minute and 2,000 per hour for guests and 100 per minute and 5,000 per hour for
authenticated users, but separately caps *search* REST endpoints at **30
requests per minute** -- and this connector only ever calls a search endpoint,
so the 30/min figure is the one that binds.  Two seconds is 30 per minute
exactly.

Confirmed live on 2026-08-31 by the response headers of an ordinary successful
request: ``x-ratelimit-limit: 30``, ``x-ratelimit-remaining: 25``,
``retry-after: 33``.  The per-minute allowance is also declared as a
:class:`~lit_search.utils.rate_limit.RateLimitQuota`, because interval pacing
alone does not stop a burst that arrives after an idle period from exceeding a
windowed budget.
"""

# Zenodo descriptions are HTML.  Tags are stripped rather than rendered.
_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")

# Publication subtypes that mark a record as not-yet-formally-published.
_PREPRINT_SUBTYPES = frozenset({"preprint", "workingpaper"})

_SUBTYPE_PAPER_TYPES: dict[str, PaperType] = {
    "article": PaperType.ARTICLE,
    "book": PaperType.BOOK,
    "conferencepaper": PaperType.INPROCEEDINGS,
    "preprint": PaperType.UNPUBLISHED,
    "report": PaperType.TECHREPORT,
    "section": PaperType.INBOOK,
    "thesis": PaperType.PHDTHESIS,
    "workingpaper": PaperType.UNPUBLISHED,
}


class ZenodoBodyError(ConnectorError):
    """A Zenodo response whose body is not a usable search result.

    Raised where ``raise_for_status`` is silent: a 200 response that is not
    JSON, is not an object, or carries no ``hits`` envelope.  Treating any of
    those as an empty result would report a failure as zero records.
    """


@dataclass(slots=True)
class _Retrieval:
    """Mutable bookkeeping for one paginated Zenodo run.

    Exists so that the reasons a run can end stay separable all the way to
    :meth:`ZenodoConnector._build_coverage`.  Collapsing them into "we have
    some records" is what turns a truncated run into a confident answer.

    Attributes
    ----------
    papers : list[Paper]
        Records parsed so far, in retrieval order.
    processed : int
        Records seen, whether or not they parsed.
    parsed : int
        Records that became a :class:`~lit_search.core.paper.Paper`.
    page : int
        One-based number of the page most recently requested.
    total : int | None
        ``hits.total`` from the first page.
    total_drift : str | None
        Set when a later page reported a different ``hits.total``.
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
    page: int = 0
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
            Raw ``hits.total``.  Non-integers are ignored, which leaves
            :attr:`total` at ``None`` and makes ``COMPLETE`` unreachable.
        """
        if not isinstance(total, int) or isinstance(total, bool):
            return
        if self.total is None:
            self.total = total
        elif total != self.total and self.total_drift is None:
            self.total_drift = (
                f"Zenodo reported {self.total} matching records on the first page and "
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


class ZenodoConnector(SearchConnectorBase):
    """Connector for the Zenodo record search API.

    Zenodo is a CERN-operated general-purpose repository.  Search needs no
    credential; an access token raises the page-size ceiling from 25 to 100 and
    exposes restricted records the token's owner may read.  Results are
    restricted to ``type=publication`` so that datasets, software and images do
    not enter a paper set -- a scope choice, not a shortfall, and one the
    coverage total is measured against because ``hits.total`` is reported for
    the filtered query.
    """

    _credential_attrs: ClassVar[tuple[str, ...]] = ("_access_token",)
    """Zenodo's token is stored under its own name rather than ``_api_key``.

    Declaring it is what keeps two processes holding the same token on one
    shared clock.  Leaving the default would have bucketed every tokened
    process as anonymous, which over-throttles rather than over-sends, but also
    hides the token's higher allowance.
    """

    def __init__(
        self,
        access_token: str | None = None,
        query_builder: ZenodoQueryBuilder | None = None,
    ) -> None:
        """Create a Zenodo searcher.

        Parameters
        ----------
        access_token : str | None
            Optional Zenodo personal access token.  Absence is not an error:
            search works anonymously, at a lower page-size ceiling.
        query_builder : ZenodoQueryBuilder | None
            Builder used to validate and convert queries.  When ``None`` a
            default :class:`ZenodoQueryBuilder` is created.
        """
        super().__init__()
        self._access_token: str | None = (access_token or "").strip() or None
        self._query_builder: ZenodoQueryBuilder = query_builder or ZenodoQueryBuilder()

    @property
    def name(self) -> str:
        """Return the database identifier.

        Returns
        -------
        str
            Database name.
        """
        return Database.ZENODO.value

    @property
    def query_builder(self) -> QueryBuilder:
        """Return the Zenodo query builder.

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

        Stated explicitly rather than inherited from :attr:`name` so that a
        future connector against another Zenodo endpoint -- deposits, files,
        OAI-PMH -- returns the same string and shares this clock.  Zenodo caps
        by account and by source IP across the whole API, so giving each
        endpoint its own scope would multiply the request rate against one
        allowance.

        Returns
        -------
        str
            Quota-authority identifier.
        """
        return "zenodo"

    @property
    def request_quota(self) -> RateLimitQuota:
        """Return Zenodo's windowed search allowance.

        Returns
        -------
        RateLimitQuota
            Thirty requests per sixty seconds, the documented cap for search
            endpoints and the value the live ``x-ratelimit-limit`` header
            reports.  The interval alone paces a steady stream correctly but
            does not stop a burst arriving after an idle period, which is what
            a windowed budget is for.
        """
        return RateLimitQuota(limit=30, window_seconds=60.0)

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
        if self._access_token:
            prepared["Authorization"] = f"Bearer {self._access_token}"
        return prepared

    @property
    def _max_page_size(self) -> int:
        """Return the largest page size this client is permitted to request.

        Returns
        -------
        int
            25 anonymously, 100 with a token.  Both verified live 2026-08-31;
            exceeding the applicable value is refused with HTTP 400 rather than
            silently clamped.
        """
        return _MAX_PAGE_SIZE_AUTHENTICATED if self._access_token else _MAX_PAGE_SIZE_ANONYMOUS

    # ------------------------------------------------------------------
    # Response handling
    # ------------------------------------------------------------------

    @staticmethod
    def _read_body(response: requests.Response) -> dict[str, Any]:
        """Decode and validate one search response body.

        Parameters
        ----------
        response : requests.Response
            Response returned by the records endpoint.

        Returns
        -------
        dict[str, Any]
            Decoded payload, guaranteed to carry a ``hits`` object.

        Raises
        ------
        ZenodoBodyError
            When the body is not JSON, is not an object, or carries no
            ``hits`` envelope.  All three would otherwise be read as "this
            repository holds no matching record".
        """
        try:
            payload = response.json()
        except ValueError as exc:
            raise ZenodoBodyError(
                f"Zenodo returned a non-JSON body with status {response.status_code}: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise ZenodoBodyError(
                f"Zenodo returned a JSON {type(payload).__name__} where an object was expected."
            )

        if not isinstance(payload.get("hits"), dict):
            raise ZenodoBodyError(
                "Zenodo response carried no 'hits' object, so it is not a search result. "
                "Treating it as an empty result set would report a failure as zero records."
            )
        return payload

    @staticmethod
    def _read_total(payload: dict[str, Any]) -> object:
        """Extract the reported result total from a response body.

        Zenodo's current serialization reports ``hits.total`` as a plain
        integer, but the underlying InvenioRDM search layer also has a
        ``{"value": n, "relation": "eq"}`` form.  Both are read; anything else
        is returned unchanged so that
        :meth:`_Retrieval.note_total` discards it and completeness stays
        unclaimable.

        Parameters
        ----------
        payload : dict[str, Any]
            Validated response body.

        Returns
        -------
        object
            The total, if one could be read, otherwise the raw value.
        """
        total = payload["hits"].get("total")
        if isinstance(total, dict):
            return total.get("value")
        return total

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_description(raw: object) -> str:
        """Strip HTML markup from a Zenodo description.

        Parameters
        ----------
        raw : object
            Raw ``metadata.description``.

        Returns
        -------
        str
            Plain-text abstract, possibly empty.
        """
        text = _HTML_TAG.sub(" ", str(raw or ""))
        return _WHITESPACE.sub(" ", text).strip()

    @staticmethod
    def _parse_authors(metadata: dict[str, Any]) -> list[Author]:
        """Extract creators from a Zenodo record's metadata.

        Parameters
        ----------
        metadata : dict[str, Any]
            One record's ``metadata`` object.

        Returns
        -------
        list[Author]
            Parsed creators, with affiliations when present.
        """
        creators = metadata.get("creators")
        entries = creators if isinstance(creators, list) else []
        authors: list[Author] = []
        for entry in entries:
            if isinstance(entry, str):
                if entry.strip():
                    authors.append(Author(name=entry.strip()))
                continue
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            if not name:
                given = str(entry.get("given_name") or "").strip()
                family = str(entry.get("family_name") or "").strip()
                name = f"{given} {family}".strip()
            if not name:
                continue
            affiliation = str(entry.get("affiliation") or "").strip() or None
            authors.append(Author(name=name, affiliation=affiliation))
        return authors

    @staticmethod
    def _parse_resource_type(metadata: dict[str, Any]) -> tuple[str, str]:
        """Return the record's ``(type, subtype)`` as lowercase strings.

        Parameters
        ----------
        metadata : dict[str, Any]
            One record's ``metadata`` object.

        Returns
        -------
        tuple[str, str]
            Type and subtype, either of which may be empty.
        """
        resource = metadata.get("resource_type")
        if not isinstance(resource, dict):
            return "", ""
        return (
            str(resource.get("type") or "").strip().lower(),
            str(resource.get("subtype") or "").strip().lower(),
        )

    @staticmethod
    def _parse_pdf_url(record: dict[str, Any]) -> str | None:
        """Return the first PDF file link on a record.

        Parameters
        ----------
        record : dict[str, Any]
            One ``hits.hits`` entry.

        Returns
        -------
        str | None
            PDF URL, or ``None`` when the record exposes no PDF.
        """
        files = record.get("files")
        for entry in files if isinstance(files, list) else []:
            if not isinstance(entry, dict):
                continue
            if not str(entry.get("key") or "").lower().endswith(".pdf"):
                continue
            links = entry.get("links")
            if isinstance(links, dict):
                href = str(links.get("self") or links.get("download") or "").strip()
                if href:
                    return href
        return None

    def _parse_record(self, record: dict[str, Any]) -> Paper | None:
        """Parse one Zenodo record into a :class:`Paper`.

        Parameters
        ----------
        record : dict[str, Any]
            One ``hits.hits`` entry.

        Returns
        -------
        Paper | None
            Parsed paper, or ``None`` when a required field is missing or the
            record fails model validation.  Callers **must** count the ``None``
            returns: an unparsed record is a coverage loss, not an absent
            paper.
        """
        if not isinstance(record, dict):
            return None
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            return None
        title = str(metadata.get("title") or record.get("title") or "").strip()
        if not title:
            return None

        _, subtype = self._parse_resource_type(metadata)
        try:
            return Paper(
                title=title,
                abstract=self._clean_description(metadata.get("description")),
                authors=self._parse_authors(metadata),
                source=Source(title="Zenodo", source_type=SourceType.REPOSITORY),
                publication_date=parse_date(
                    str(metadata.get("publication_date") or "").strip() or None
                ),
                url=self._parse_url(record),
                pdf_url=self._parse_pdf_url(record),
                doi=normalize_doi(str(record.get("doi") or metadata.get("doi") or "")),
                keywords=self._parse_keywords(metadata) or None,
                found_in={self.name},
                paper_type=self._paper_type_for(subtype),
                is_open_access=self._open_access_for(metadata),
            )
        except ValueError:
            logger.debug("Zenodo: record %s failed model validation.", record.get("id"))
            return None

    @staticmethod
    def _parse_url(record: dict[str, Any]) -> str | None:
        """Return the record's landing-page URL.

        Parameters
        ----------
        record : dict[str, Any]
            One ``hits.hits`` entry.

        Returns
        -------
        str | None
            The record's own HTML link, a link rebuilt from its identifier, or
            ``None`` when the record carries neither.
        """
        links = record.get("links")
        if isinstance(links, dict):
            url = str(links.get("self_html") or links.get("html") or "").strip()
            if url:
                return url
        record_id = str(record.get("id") or record.get("recid") or "").strip()
        return f"https://zenodo.org/records/{record_id}" if record_id else None

    @staticmethod
    def _parse_keywords(metadata: dict[str, Any]) -> set[str]:
        """Extract keyword terms from a record's metadata.

        Parameters
        ----------
        metadata : dict[str, Any]
            One record's ``metadata`` object.

        Returns
        -------
        set[str]
            Keyword strings; empty when the record carries none.
        """
        keywords = metadata.get("keywords")
        if not isinstance(keywords, list):
            return set()
        return {k.strip() for k in keywords if isinstance(k, str) and k.strip()}

    @staticmethod
    def _paper_type_for(subtype: str) -> PaperType:
        """Map a Zenodo publication subtype to a :class:`PaperType`.

        Parameters
        ----------
        subtype : str
            Lowercase ``resource_type.subtype``.

        Returns
        -------
        PaperType
            The mapped type.  An unrecognised subtype falls back to
            :data:`~lit_search.core.paper.PaperType.MISC` rather than to
            ``ARTICLE``, because Zenodo holds a great deal that is not an
            article and guessing otherwise would mislabel it.
        """
        fallback = PaperType.UNPUBLISHED if subtype in _PREPRINT_SUBTYPES else PaperType.MISC
        return _SUBTYPE_PAPER_TYPES.get(subtype, fallback)

    @staticmethod
    def _open_access_for(metadata: dict[str, Any]) -> bool | None:
        """Return the record's open-access status, asserting only the positive.

        Zenodo publishes ``access_right`` per record, but anything other than
        ``open`` describes *this deposit*, not the paper: a restricted Zenodo
        record can be an open-access article deposited elsewhere.  Only the
        positive case is claimed.

        Parameters
        ----------
        metadata : dict[str, Any]
            One record's ``metadata`` object.

        Returns
        -------
        bool | None
            ``True`` for an open record, otherwise ``None``.
        """
        if str(metadata.get("access_right") or "").strip().lower() == "open":
            return True
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
            Zenodo ``q`` expression.
        """
        expression = self._query_builder.convert_query(query)
        if since is None and until is None:
            return expression
        low = since.isoformat() if since else "*"
        high = until.isoformat() if until else "*"
        date_filter = f"publication_date:[{low} TO {high}]"
        return f"({expression}) AND {date_filter}" if expression else date_filter

    def _page_size_for(self, max_papers: int | None) -> int:
        """Return the page size to use for **every** request in one run.

        Never returns zero and never exceeds the ceiling this client is
        permitted, both of which the API refuses with HTTP 400.

        The value deliberately does not depend on how much of the cap is left.
        Zenodo's window is ``records[(page - 1) * size : page * size]``, so
        ``size`` and ``page`` are read together: shrinking ``size`` on a later
        page moves the window *backwards* into records already retrieved.
        Verified live 2026-08-31 against ``q=catalysis&type=publication&
        sort=bestmatch``: ``page=1&size=25`` returned 25 ids, and
        ``page=2&size=5`` returned ids 6-10 of that same list rather than ids
        26-30.  An earlier version of this method returned
        ``max_papers - len(run.papers)``, which for ``max_papers=30`` issued
        ``page=1&size=25`` then ``page=2&size=5`` and so returned 25 distinct
        records plus 5 duplicates while reporting ``capped (30 papers)`` -- a
        confident count covering records the run never asked for.

        The cost of a constant size is over-fetching at most ``size - 1``
        records on the final page, which :meth:`_fetch_papers` discards.  It
        issues no more requests than the shrinking form did.

        Parameters
        ----------
        max_papers : int | None
            Caller-imposed cap.

        Returns
        -------
        int
            Page size between 1 and :attr:`_max_page_size`.
        """
        ceiling = self._max_page_size
        if max_papers is None:
            return ceiling
        return max(1, min(ceiling, max_papers))

    def _fetch_papers(
        self,
        query: Query,
        max_papers: int | None,
        progress_callback: Callable[[int, int | None], None] | None,
        since: datetime.date | None = None,
        until: datetime.date | None = None,
    ) -> CoveredPapers:
        """Fetch papers from Zenodo with page pagination and coverage.

        Every way this loop can end is tracked separately, because they are
        different facts about the result:

        - the caller's ``max_papers`` was reached -> ``CAPPED``;
        - a page request failed, or its body was unusable, with nothing
          retrieved -> raised, so the base class reports ``FAILED``;
        - the same failure *after* records were retrieved -> ``PARTIAL``
          carrying the error, because the records that did arrive are real;
        - the 10,000-record offset window was reached -> ``PARTIAL``, because
          more records match than this API can page to;
        - ``hits.total`` changed between pages -> ``PARTIAL``, because neither
          total can be checked against what was retrieved;
        - ``links.next`` was absent -> ``from_pagination`` decides, and it
          returns ``COMPLETE`` only when as many records came back as Zenodo
          said existed.

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
        # Fixed for the whole run: see _page_size_for on why a size that
        # tracks the remaining cap rewinds the window.
        size = self._page_size_for(max_papers)

        while True:
            run.page += 1
            params = {
                "q": expression,
                "size": size,
                "page": run.page,
                "sort": _SORT,
                "type": _PUBLICATION_TYPE,
            }

            try:
                payload = self._read_body(self._get(_RECORDS_URL, params))
            except (requests.RequestException, ConnectorError) as exc:
                self._record_page_failure(run, exc)
                break

            seen = self._absorb_page(run, payload)
            if progress_callback is not None:
                progress_callback(run.processed, run.total)

            if not self._should_continue(run, payload, size, max_papers, page_records=seen):
                break

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

        Raises
        ------
        Exception
            The original failure, when nothing had been retrieved.
        """
        if not run.papers:
            raise exc
        log_pagination_request_failure(
            logger,
            connector_label="Zenodo",
            processed=run.processed,
            total=run.total,
            position_label="page",
            position=run.page,
            exc=exc,
        )
        run.truncate(f"page request at page={run.page} failed: {exc}", type(exc).__name__)

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
        run.note_total(self._read_total(payload))
        raw = payload["hits"].get("hits")
        records: list[Any] = raw if isinstance(raw, list) else []
        for record in records:
            paper = self._parse_record(record)
            if paper is not None:
                run.papers.append(paper)
                run.parsed += 1
        run.processed += len(records)
        return len(records)

    @staticmethod
    def _should_continue(
        run: _Retrieval,
        payload: dict[str, Any],
        size: int,
        max_papers: int | None,
        *,
        page_records: int,
    ) -> bool:
        """Decide whether to request another page.

        Every stop other than the cap and a natural exhaustion is recorded as
        a truncation, so that it reaches the coverage status instead of the
        log.

        Parameters
        ----------
        run : _Retrieval
            Bookkeeping for the current run, updated in place.
        payload : dict[str, Any]
            Validated response body for the page just absorbed.
        size : int
            Page size that produced this page.
        max_papers : int | None
            Caller-imposed cap.
        page_records : int
            Raw entries the page just absorbed carried.

        Returns
        -------
        bool
            ``True`` to request another page.
        """
        if max_papers is not None and len(run.papers) >= max_papers:
            return False

        links = payload.get("links")
        has_next = isinstance(links, dict) and bool(links.get("next"))
        if has_next and page_records == 0:
            # A page that carried nothing while still offering a next link
            # cannot advance the run: following it would re-ask the same
            # question until the offset window closed, at 2s per request.
            # OpenAIRE's _advance already stops on exactly this shape; this
            # keeps the two siblings consistent and turns a silent spin into
            # a reported shortfall.
            run.truncate(
                f"Zenodo returned an empty page at page {run.page} while still offering a "
                f"next link, after {run.processed} of "
                f"{run.total if run.total is not None else 'an unknown number of'} "
                f"matching records"
            )
            return False
        if not has_next:
            # Zenodo omits links.next on the last page and on a zero-hit
            # result.  A short page is not consulted: it is not a documented
            # end-of-set signal and the reference port proved that heuristic
            # wrong on a sibling API.
            if run.total is not None and run.processed < run.total:
                run.truncate(
                    f"Zenodo stopped offering a next page after {run.processed} of "
                    f"{run.total} reported records (last page requested: {run.page})"
                )
            return False

        if (run.page + 1) * size > _MAX_RESULT_WINDOW:
            run.truncate(
                f"Zenodo refuses to page beyond {_MAX_RESULT_WINDOW} records "
                f"(HTTP 400 'Invalid querystring parameters'), so retrieval stopped at "
                f"{run.processed} of "
                f"{run.total if run.total is not None else 'an unknown number of'} "
                f"matching records"
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
