"""IACR Cryptology ePrint Archive connector, driven by OAI-PMH.

**This is new code, not a port.**  The donor
(``paper_search_mcp/academic_platforms/iacr.py`` at pinned commit
``234678ab231074a7977320978ee0496dcdaddd1f``) scrapes
``https://eprint.iacr.org/search`` with BeautifulSoup, and that path is
disallowed.  Its error contract is also the one this project exists to remove:
``if response.status_code != 200: return papers`` (``iacr.py:165-167``),
``if not results: return papers`` (``iacr.py:175-177``) and a whole-body
``except Exception: logger.error(...)`` before ``return papers``
(``iacr.py:191-194``) collapse an HTTP error, a markup change and an empty
result into the same empty list.

Verified live 2026-08-31.

**Policy.**  ``https://eprint.iacr.org/robots.txt`` disallows ``/search`` for
every named agent, ends with ``User-agent: * -> Disallow: /``, and says
crawlers must be whitelisted.  Its own opening comment names the sanctioned
route: *"Metadata is available from this site using OAI-PMH, RSS, and ATOM."*
This connector therefore drives OAI-PMH -- and still ships **gated**, because
"whitelisted" is a statement about an agreement with the archive that this code
cannot verify for itself.  Until the operator records that agreement (see
:data:`_OPT_IN_ENV_VAR`) every search reports ``SKIPPED_NO_CREDENTIAL``, which
is the status for a source that was never contacted and can be enabled by
configuration.

**Capability.**  The endpoint cannot be queried.  ``ListSets`` returns
``<error code="noSetHierarchy">``, ``ListMetadataFormats`` offers only
``oai_dc``, and ``ListRecords`` accepts ``from``, ``until`` and
``metadataPrefix``.  So a search harvests a **date window** and filters it
locally, which has three consequences the coverage status must carry:

1. The window is a slice of the archive, not the answer to the query.
2. OAI-PMH's ``from``/``until`` filter on the record's **datestamp**, which for
   this repository is its last-modified time.  A 2019 paper revised last month
   falls in last month's window, and a 2019 paper never revised does not.
3. The archive never reports how many records match a query, because it cannot
   be asked one.

**A result from this connector is therefore never ``COMPLETE``.**  That is
enforced in :meth:`IACRConnector._build_coverage`, not left to a convention.

**Errors do not arrive as errors.**  Every OAI-PMH fault is HTTP 200 with an
``<error code="...">`` element, so ``findall(".//record")`` returns ``[]`` and
a naive implementation reports zero papers.  Four were observed:

===============================  ==========================  ======================
Request                          ``error code``              Meaning here
===============================  ==========================  ======================
``from=BADDATE``                 ``badArgument``             our bug -> ``FAILED``
``metadataPrefix=nosuchformat``  ``cannotDisseminateFormat`` our bug -> ``FAILED``
``verb=Bogus``                   ``badVerb``                 our bug -> ``FAILED``
window with no records           ``noRecordsMatch``          **a genuine empty set**
===============================  ==========================  ======================

``noRecordsMatch`` is the one that matters: the protocol reports "nothing in
this window" as an *error*, and treating it as a failure would be as wrong as
treating the other three as emptiness.

**Pagination.**  ``ListRecords`` returns 500 records and a
``<resumptionToken completeListSize="1975" cursor="500">``.  ``completeListSize``
is the archive's own count **of the window**, not of the query, so it is
reported as context and never used to justify completeness.
"""

from __future__ import annotations

import datetime
import logging
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import requests
from defusedxml import ElementTree as ET

if TYPE_CHECKING:
    from xml.etree.ElementTree import Element  # nosec B405

    from lit_search.query.builder import QueryBuilder

from lit_search.connectors.search_base import SearchConnectorBase
from lit_search.core.author import Author
from lit_search.core.coverage import CoveredPapers, SourceCoverage
from lit_search.core.paper import Database, Paper, PaperType
from lit_search.core.query import Query
from lit_search.core.source import Source, SourceType
from lit_search.exceptions import ConnectorError, MissingApiKeyError
from lit_search.query.builders.iacr import IACRQueryBuilder
from lit_search.utils.normalization import normalize_language, parse_date
from lit_search.utils.pagination_logging import log_pagination_request_failure

logger = logging.getLogger(__name__)

_OAI_URL = "https://eprint.iacr.org/oai"

_OPT_IN_ENV_VAR = "LIT_SEARCH_IACR_OAI_ENABLED"
"""Environment variable recording that IACR has whitelisted this client.

IACR's ``robots.txt`` states that crawlers must be whitelisted and confirmed to
be following the rules.  Whether that has happened is a fact about an agreement
with the archive, so the code cannot establish it -- it can only refuse to
assume it.  Set this to a truthy value once the agreement exists.

Modelling the acknowledgement as a credential is deliberate:
``SKIPPED_NO_CREDENTIAL`` already means "never contacted, determinate, fixable
by configuration", which is exactly this situation, and it keeps the source
visible in the ledger instead of quietly absent.
"""

_TRUTHY = frozenset({"1", "true", "yes", "on", "enabled"})

# XML namespaces used by the OAI-PMH response envelope and the oai_dc payload.
_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/",
    "dc": "http://purl.org/dc/elements/1.1/",
}

# The OAI-PMH error code meaning "this window holds no records", as opposed to
# the codes meaning "your request was malformed".  Verified live: a window in
# 2050 returns ``<error code="noRecordsMatch">No matching records</error>``.
_EMPTY_ERROR_CODE = "noRecordsMatch"

# Default harvest window when the caller supplies no lower bound, in days.  The
# archive runs from 1996 and holds tens of thousands of records; harvesting all
# of it costs hundreds of paced requests, so an unbounded search consults a
# recent window and says so rather than running for half an hour.
_DEFAULT_WINDOW_DAYS = 365

# Hard ceiling on resumption rounds, so a repository that keeps issuing tokens
# ends as an honest PARTIAL rather than running forever.  At 500 records per
# response this bounds one search at 100,000 records.
_MAX_PAGES = 200

_MIN_REQUEST_INTERVAL = 5.0
"""Seconds between requests.

**This is the strictest number IACR publishes.**
``https://eprint.iacr.org/robots.txt`` (re-read 2026-08-31) carries
``Crawl-delay: 5`` in its ``bingbot`` and ``DuckDuckBot/1.1`` blocks.  It is
**not** stated for every named agent: the first block, which names
``Googlebot``, ``Claude-User`` and ``Claude-SearchBot``, carries no
``Crawl-delay`` at all, and neither does the ``ClaudeBot``/``GPTBot`` AI-crawler
block.  So this value is borrowed from the agents the file *does* pace, not
inherited from one that names us -- which is the conservative direction for a
file whose closing rule is ``User-agent: * -> Disallow: /`` and whose comment
says crawlers must be whitelisted.

The archive publishes no separate OAI-PMH rate.  Harvesting is bulk by nature,
so this interval is the main thing keeping a sanctioned protocol from being
used in an unsanctioned way.
"""


class IACROAIError(ConnectorError):
    """An OAI-PMH response that is not a usable record list.

    Every OAI-PMH fault arrives with HTTP 200 and an ``<error>`` element, so
    ``raise_for_status`` is silent and a record search finds nothing.  This is
    raised where that silence would otherwise be read as an empty archive.

    Parameters
    ----------
    message : str
        Message assembled from the ``error`` element or the parse failure.
    code : str | None
        The OAI-PMH ``error code``, when the response carried one.
    """

    def __init__(self, message: str, *, code: str | None = None) -> None:
        """Store the message and the OAI-PMH error code."""
        super().__init__(message)
        self.code = code


@dataclass(slots=True)
class _Harvest:
    """Mutable bookkeeping for one IACR OAI-PMH harvest.

    Attributes
    ----------
    papers : list[Paper]
        Records that parsed **and** matched the query, in harvest order.
    processed : int
        Records seen, whether or not they parsed or matched.
    parsed : int
        Records that became a :class:`~lit_search.core.paper.Paper`.
    matched : int
        Parsed records that satisfied the query predicate.
    pages : int
        Responses consumed.
    window_total : int | None
        ``completeListSize`` from a resumption token: how many records the
        archive says the *window* holds.  Never the number matching the query.
    exhausted : bool
        Whether the resumption chain ran out naturally.
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
    until : datetime.date | None
        Upper bound, or ``None`` for open-ended.
    caller_supplied_since : bool
        Whether *since* came from the caller or was defaulted.  A defaulted
        window is a much larger shortfall and is described differently.
    """

    since: datetime.date
    until: datetime.date | None
    caller_supplied_since: bool

    def describe(self) -> str:
        """Return a human-readable description of the window.

        Returns
        -------
        str
            Description naming the bounds and how *since* was chosen.
        """
        high = self.until.isoformat() if self.until else "the present"
        if self.caller_supplied_since:
            return f"records modified between {self.since.isoformat()} and {high}"
        return (
            f"records modified between {self.since.isoformat()} and {high} -- the "
            f"connector's default {_DEFAULT_WINDOW_DAYS}-day window, because the "
            f"search supplied no start date and the archive runs from 1996"
        )


class IACRConnector(SearchConnectorBase):
    """Connector for the IACR Cryptology ePrint Archive.

    Harvests ``oai_dc`` records over a date window through the archive's
    OAI-PMH endpoint and applies the query locally.  See the module docstring
    for why there is no remote query and why a result is never ``COMPLETE``.
    """

    _credential_attrs: ClassVar[tuple[str, ...]] = ()
    """The archive issues no API key.  The opt-in flag is an acknowledgement,
    not a secret, so it is deliberately **not** listed here: including it would
    put its value into the rate-limit bucket fingerprint and give an opted-in
    process a different clock from an opted-out one, against the same host.
    """

    def __init__(self, query_builder: IACRQueryBuilder | None = None) -> None:
        """Create an IACR searcher.

        Parameters
        ----------
        query_builder : IACRQueryBuilder | None
            Builder used to validate queries and compile the local predicate.
            When ``None`` a default :class:`IACRQueryBuilder` is created.
        """
        super().__init__()
        self._query_builder: IACRQueryBuilder = query_builder or IACRQueryBuilder()

    @property
    def name(self) -> str:
        """Return the database identifier.

        Returns
        -------
        str
            Database name.
        """
        return Database.IACR.value

    @property
    def query_builder(self) -> QueryBuilder:
        """Return the IACR query builder.

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

        Stated explicitly so that any future connector against another
        ``eprint.iacr.org`` route -- the RSS and ATOM feeds the same
        ``robots.txt`` names, or a PDF fetcher -- shares this clock instead of
        doubling the rate against a host that asks to be crawled slowly.

        Returns
        -------
        str
            Quota-authority identifier.
        """
        return "iacr"

    @staticmethod
    def _require_opt_in() -> None:
        """Refuse to contact the archive until the operator has opted in.

        Raises
        ------
        MissingApiKeyError
            When :data:`_OPT_IN_ENV_VAR` is unset or not truthy.  The base
            class turns this into ``SKIPPED_NO_CREDENTIAL``, so the source
            stays visible in the ledger as one that was never asked.
        """
        value = os.environ.get(_OPT_IN_ENV_VAR, "").strip().lower()
        if value not in _TRUTHY:
            raise MissingApiKeyError(_OPT_IN_ENV_VAR)

    # ------------------------------------------------------------------
    # Response handling
    # ------------------------------------------------------------------

    @staticmethod
    def _read_body(response: requests.Response) -> Element:
        """Parse one OAI-PMH response and reject its error forms.

        Parameters
        ----------
        response : requests.Response
            Response returned by the OAI-PMH endpoint.

        Returns
        -------
        Element
            Parsed ``OAI-PMH`` root element, guaranteed not to carry an error
            other than :data:`_EMPTY_ERROR_CODE`.

        Raises
        ------
        IACROAIError
            When the body is not well-formed XML, or carries an ``<error>``
            element describing a malformed request.  Both arrive with HTTP 200
            and would otherwise be read as an archive holding no records.
        """
        try:
            root: Element = ET.fromstring(response.text)
        except ET.ParseError as exc:
            raise IACROAIError(
                f"IACR OAI-PMH returned a body that is not well-formed XML with status "
                f"{response.status_code}: {exc}. Treating it as an empty archive would "
                f"report a failure as zero papers."
            ) from exc

        error = root.find("oai:error", _NS)
        if error is not None:
            code = (error.get("code") or "").strip()
            message = (error.text or "").strip() or "no message supplied"
            if code == _EMPTY_ERROR_CODE:
                # The protocol reports "nothing in this window" as an error.
                # It is a genuine empty result and must not become FAILED.
                return root
            raise IACROAIError(
                f"IACR OAI-PMH reported error '{code}': {message}", code=code or None
            )
        return root

    @staticmethod
    def _is_empty_window(root: Element) -> bool:
        """Whether this response is the protocol's "no records here" answer.

        Parameters
        ----------
        root : Element
            Parsed response root.

        Returns
        -------
        bool
            ``True`` when the response carried ``noRecordsMatch``.
        """
        error = root.find("oai:error", _NS)
        return error is not None and (error.get("code") or "").strip() == _EMPTY_ERROR_CODE

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _record_fields(record: Element) -> dict[str, list[str]]:
        """Extract every ``dc:*`` value from one record, keyed by element name.

        Parameters
        ----------
        record : Element
            One ``<record>`` element.

        Returns
        -------
        dict[str, list[str]]
            Element name (without the ``dc:`` prefix) to its values.  Empty
            when the record carries no ``oai_dc`` metadata, which is how a
            deleted record arrives.
        """
        fields: dict[str, list[str]] = {}
        payload = record.find("oai:metadata/oai_dc:dc", _NS)
        if payload is None:
            return fields
        for child in payload:
            name = child.tag.rsplit("}", 1)[-1]
            text = (child.text or "").strip()
            if text:
                fields.setdefault(name, []).append(text)
        return fields

    def _parse_record(self, fields: dict[str, list[str]]) -> Paper | None:
        """Build a :class:`Paper` from one record's ``dc`` fields.

        Parameters
        ----------
        fields : dict[str, list[str]]
            Values keyed by ``dc`` element name.

        Returns
        -------
        Paper | None
            Parsed paper, or ``None`` when the record has no title or fails
            model validation.  Callers **must** count the ``None`` returns: an
            unparsed record is a coverage loss, not an absent paper.
        """
        title = next(iter(fields.get("title", ())), "").strip()
        if not title:
            return None

        url = next(
            (value for value in fields.get("identifier", ()) if value.startswith("http")), None
        )
        keywords = {value for value in fields.get("subject", ()) if value}
        try:
            return Paper(
                title=title,
                abstract=" ".join(fields.get("description", ())).strip(),
                authors=[Author(name=name) for name in fields.get("creator", ()) if name],
                source=Source(
                    title="IACR Cryptology ePrint Archive",
                    source_type=SourceType.REPOSITORY,
                ),
                publication_date=_first_date(fields.get("date", ())),
                url=url,
                pdf_url=f"{url}.pdf" if url else None,
                keywords=keywords or None,
                found_in={self.name},
                paper_type=PaperType.UNPUBLISHED,
                language=normalize_language(next(iter(fields.get("language", ())), None)),
                # Every ePrint is openly readable; the archive publishes no
                # paywalled records.
                is_open_access=True,
            )
        except ValueError:
            logger.debug("IACR: a harvested record failed model validation.")
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
            Lower bound of the harvest window.  Defaulted when absent.
        until : datetime.date | None
            Upper bound of the harvest window.

        Returns
        -------
        CoveredPapers
            Matching papers and the coverage status describing them.

        Raises
        ------
        MissingApiKeyError
            When the operator has not recorded IACR's whitelisting.
        ConnectorError
            When the first request returned an unusable body.
        requests.RequestException
            When the first request failed at the transport level.
        """
        self._require_opt_in()
        window = _resolve_window(since, until)
        matches = self._query_builder.compile_predicate(query)
        run = _Harvest()
        token: str | None = None

        while True:
            params = (
                {"verb": "ListRecords", "resumptionToken": token}
                if token
                else {
                    "verb": "ListRecords",
                    "metadataPrefix": "oai_dc",
                    "from": _oai_timestamp(window.since),
                    "until": _oai_timestamp(window.until) if window.until else None,
                }
            )

            try:
                root = self._read_body(self._get(_OAI_URL, _without_none(params)))
            except (requests.RequestException, ConnectorError) as exc:
                self._record_page_failure(run, exc, run.pages)
                break

            if self._is_empty_window(root):
                run.exhausted = True
                run.pages += 1
                break

            self._absorb_page(run, root, matches, max_papers)
            if progress_callback is not None:
                progress_callback(run.processed, run.window_total)

            token = self._advance(run, root, max_papers)
            if token is None:
                break

        if progress_callback is not None:
            progress_callback(run.processed, run.window_total)

        kept = run.papers[:max_papers] if max_papers is not None else run.papers
        return CoveredPapers(
            papers=tuple(kept),
            coverage=self._build_coverage(
                run=run, retrieved=len(kept), window=window, max_papers=max_papers
            ),
        )

    @staticmethod
    def _record_page_failure(run: _Harvest, exc: Exception, page: int) -> None:
        """Record a harvest failure, or re-raise when nothing was retrieved.

        With no records in hand there is no partial result to report and the
        failure is the whole story, so it is re-raised for
        :meth:`SearchConnectorBase.search` to name.  With records in hand the
        harvest is a real shortfall and the records are kept.

        Parameters
        ----------
        run : _Harvest
            Bookkeeping for the current harvest.
        exc : Exception
            The failure raised while fetching or reading the response.
        page : int
            Which response failed, for the message.

        Raises
        ------
        Exception
            The original failure, when nothing had been retrieved.
        """
        if not run.papers:
            raise exc
        log_pagination_request_failure(
            logger,
            connector_label="IACR",
            processed=run.processed,
            total=run.window_total,
            position_label="resumption round",
            position=str(page),
            exc=exc,
        )
        run.truncate(f"harvest failed at resumption round {page}: {exc}", type(exc).__name__)

    def _absorb_page(
        self,
        run: _Harvest,
        root: Element,
        matches: Callable[[dict[str, list[str]]], bool],
        max_papers: int | None,
    ) -> None:
        """Fold one response into *run*, keeping the records that match.

        Parameters
        ----------
        run : _Harvest
            Bookkeeping for the current harvest, updated in place.
        root : Element
            Parsed response root.
        matches : Callable[[dict[str, list[str]]], bool]
            Compiled query predicate.
        max_papers : int | None
            Caller-imposed cap; harvesting stops adding papers once reached.
        """
        for record in root.findall("oai:ListRecords/oai:record", _NS):
            run.processed += 1
            fields = self._record_fields(record)
            if not fields:
                # A deleted record carries a header and no metadata.  It is not
                # a parse failure and not a paper, so it is counted as seen and
                # nothing more.
                continue
            paper = self._parse_record(fields)
            if paper is None:
                continue
            run.parsed += 1
            if not matches(fields):
                continue
            run.matched += 1
            if max_papers is None or len(run.papers) < max_papers:
                run.papers.append(paper)
        run.pages += 1

    @staticmethod
    def _advance(run: _Harvest, root: Element, max_papers: int | None) -> str | None:
        """Return the next resumption token, or ``None`` to stop.

        Parameters
        ----------
        run : _Harvest
            Bookkeeping for the current harvest, updated in place.
        root : Element
            Parsed response root.
        max_papers : int | None
            Caller-imposed cap.

        Returns
        -------
        str | None
            Token for the next request, or ``None`` to stop.
        """
        element = root.find("oai:ListRecords/oai:resumptionToken", _NS)
        if element is not None:
            size = element.get("completeListSize")
            if size is not None and size.strip().isdigit():
                run.window_total = int(size.strip())

        if max_papers is not None and len(run.papers) >= max_papers:
            return None

        # An absent or empty token is the protocol's end-of-list signal.
        token = (element.text or "").strip() if element is not None else ""
        if not token:
            run.exhausted = True
            return None

        if run.pages >= _MAX_PAGES:
            run.truncate(
                f"harvest stopped at the connector's {_MAX_PAGES}-response safety limit "
                f"after {run.processed} records, while the archive was still offering a "
                f"resumption token"
            )
            return None
        return token

    def _build_coverage(
        self,
        *,
        run: _Harvest,
        retrieved: int,
        window: _Window,
        max_papers: int | None,
    ) -> SourceCoverage:
        """Describe what the harvest achieved.

        The status is always ``PARTIAL``, and both halves of that are
        deliberate.

        It is never ``COMPLETE`` for a structural reason rather than a cautious
        one: ``COMPLETE`` asserts that everything the source could match for
        this query was retrieved, and this connector never asked the source
        about the query at all.  ``completeListSize`` counts the records in a
        date window, so passing it as a ``reported_total`` would let arithmetic
        that happens to balance stand in for a claim nothing supports.

        It is not ``CAPPED`` either, even when the caller's limit stopped the
        harvest.  ``CAPPED`` means a *known* amount is missing on purpose, and
        it ranks as a milder outcome than ``PARTIAL``; here an unknown amount
        is missing whatever the cap did, because everything outside the window
        was never consulted.  Reporting ``CAPPED`` would make a truncated
        harvest read as healthier than an untruncated one.  The cap is recorded
        in the detail instead.

        Parameters
        ----------
        run : _Harvest
            Bookkeeping for the finished harvest.
        retrieved : int
            Papers actually returned, after the cap.
        window : _Window
            The window that was consulted.
        max_papers : int | None
            Caller-imposed cap, named in the detail when it bound.

        Returns
        -------
        SourceCoverage
            A ``PARTIAL`` record describing the window and the local match.
        """
        notes: list[str] = [
            f"IACR ePrint has no queryable endpoint, so this searched {window.describe()}",
            f"harvested {run.processed} records"
            + (
                f" of the {run.window_total} the archive reports for that window"
                if run.window_total is not None
                else ""
            )
            + f" and matched {run.matched} against the query locally",
            "OAI-PMH filters on last-modified date, so a paper first posted earlier and "
            "never revised falls outside this window",
        ]
        if max_papers is not None and run.matched > retrieved:
            notes.append(
                f"the caller's limit of {max_papers} stopped the harvest, so {run.matched} "
                f"records had already matched within the window when it ended"
            )
        if run.error_detail:
            notes.append(run.error_detail)
        if not run.exhausted and not run.truncated_early:
            notes.append("the resumption chain did not run to its end")
        unparsed = run.processed - run.parsed
        if unparsed > 0:
            notes.append(
                f"{unparsed} of {run.processed} harvested records were deleted records or "
                f"could not be parsed into papers"
            )

        return SourceCoverage.partial(
            self.name,
            retrieved,
            detail="; ".join(notes),
            error=run.error_detail,
            error_type=run.error_type,
        )


def _resolve_window(
    since: datetime.date | None,
    until: datetime.date | None,
) -> _Window:
    """Choose the harvest window, defaulting an absent lower bound.

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
    if since is not None:
        return _Window(since=since, until=until, caller_supplied_since=True)
    anchor = until or datetime.date.today()
    return _Window(
        since=anchor - datetime.timedelta(days=_DEFAULT_WINDOW_DAYS),
        until=until,
        caller_supplied_since=False,
    )


def _oai_timestamp(value: datetime.date) -> str:
    """Render a date as the granularity IACR's repository declares.

    ``Identify`` reports ``<granularity>YYYY-MM-DDThh:mm:ssZ</granularity>``,
    so dates are sent at that granularity rather than as bare dates.

    Parameters
    ----------
    value : datetime.date
        Date to render.

    Returns
    -------
    str
        Timestamp in OAI-PMH's full granularity.
    """
    return f"{value.isoformat()}T00:00:00Z"


def _without_none(params: dict[str, Any]) -> dict[str, Any]:
    """Drop keys whose value is ``None``.

    Parameters
    ----------
    params : dict[str, Any]
        Request parameters, possibly with absent optional bounds.

    Returns
    -------
    dict[str, Any]
        Parameters with ``None`` values removed.
    """
    return {key: value for key, value in params.items() if value is not None}


def _first_date(values: Iterable[str]) -> datetime.date | None:
    """Return the first parsable date among *values*.

    Parameters
    ----------
    values : Iterable[str]
        Raw ``dc:date`` values.  IACR repeats the same timestamp twice.

    Returns
    -------
    datetime.date | None
        Parsed date, or ``None`` when none is usable.
    """
    for value in values:
        parsed = parse_date(value.split("T", 1)[0])
        if parsed is not None:
            return parsed
    return None
