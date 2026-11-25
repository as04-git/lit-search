"""PubMed Central (PMC) searcher implementation.

Ported from ``paper_search_mcp/academic_platforms/pmc.py`` at pinned commit
``234678ab231074a7977320978ee0496dcdaddd1f``.  The port is **not** a mechanical
translation: the donor's ``search()`` wraps the whole esearch-then-esummary
block in ``except requests.RequestException`` / ``except ET.ParseError`` /
``except Exception``, all three falling through to ``return papers``, and
short-circuits with ``if not pmcids: return papers``.  A dead endpoint, a
malformed body, a query NCBI refused and a genuinely unmatched query all reach
the caller as the same empty list.  See ``PORTING-GUIDE.md`` for the
before-and-after.

Five behaviours of the live API drive the design here.  All were observed
directly against ``eutils.ncbi.nlm.nih.gov`` on 2026-08-31, and **every one of
them arrives with HTTP 200**, so ``raise_for_status`` is silent for all five:

- **Errors are XML elements, not statuses.**  An empty term returns
  ``<ERROR>Empty term and query_key - nothing todo</ERROR>``; a bad database
  returns ``<ERROR>Invalid db name specified: notadb</ERROR>``.  Both carry no
  ``<IdList>``, so ``findall('.//Id')`` yields ``[]``.
- **An unknown field tag is dropped and the query is silently re-run.**
  ``cancer[ab]`` returns ``Count`` 4,013,901 -- identical to ``cancer[ALL]`` --
  with the only trace being ``<ErrorList><FieldNotFound>ab</FieldNotFound>``.
  A connector that reused PubMed's ``[ab]`` tag would return four million
  full-text hits for a query scoped to abstracts.  The query builder never
  emits such a tag; this connector treats one arriving anyway as
  ``QUERY_UNSUPPORTED`` rather than as a result.
- **A quoted phrase that is not in the phrase index is re-tokenised.**
  ``<WarningList><QuotedPhraseNotFound>`` says so.  The search still runs, over
  a different query, so the run is reported ``PARTIAL`` with the phrase named.
- **The result set is hard-capped at 9,999 records.**  ``retstart`` above 9,998
  returns ``<ERROR>...'retstart' cannot be larger than 9998...</ERROR>``, and
  the ceiling applies to the history server too, so no amount of paging gets
  past it.
- **``esummary`` loses individual records inline.**  A batch containing an
  unresolvable UID returns the good ``<DocSum>`` elements plus
  ``<ERROR>UID=999999999: cannot get document summary</ERROR>``, at HTTP 200.
  Those are records the source acknowledged and did not deliver, so they are
  counted as a coverage loss rather than skipped.

**What this connector does not retrieve: abstracts.**  It pages
``esummary``, whose ``DocSum`` carries title, authors, journal, date and
identifiers but no abstract.  The alternative is ``efetch``, which for
``db=pmc`` returns the complete JATS article -- 143,966 bytes for a single
paper, measured -- making it unusable as a search path.  This is a *field*
gap, not a *record* gap: every record PMC matched is returned, so it is not
represented in the coverage status, which is a statement about records.  It
does mean PMC records reach a semantic re-rank with title-only text; the
policy for that is open item **D6** in the build plan.

Rate limit: shared with PubMed.  See :attr:`PMCConnector._RATE_LIMIT_SERVICE`.
"""

from __future__ import annotations

import datetime
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import requests
from defusedxml import ElementTree as ET

if TYPE_CHECKING:
    # Element is the concrete type returned by defusedxml at runtime;
    # imported from stdlib only for static type annotations.
    from xml.etree.ElementTree import Element  # nosec B405

    from lit_search.query.builder import QueryBuilder

from lit_search.connectors.search_base import SearchConnectorBase
from lit_search.core.author import Author
from lit_search.core.coverage import CoveredPapers, SourceCoverage
from lit_search.core.paper import Database, Paper, PaperType
from lit_search.core.query import Query
from lit_search.core.source import Source, SourceType
from lit_search.exceptions import ConnectorError, UnsupportedQueryError
from lit_search.query.builders.pmc import PMCQueryBuilder
from lit_search.utils.normalization import normalize_doi, parse_date
from lit_search.utils.pagination_logging import log_pagination_request_failure

logger = logging.getLogger(__name__)

_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

_DB = "pmc"
_PAGE_SIZE = 100

# Verified live 2026-08-31: retstart above 9998 returns an <ERROR> body, and the
# same ceiling applies when paging the history server, so 9,999 records is the
# most any single Entrez query can yield.  NCBI's own message points at EDirect
# for more.
_MAX_RETRIEVABLE = 9999

# Hard ceiling on pagination rounds, as a guard against a page that keeps
# returning nothing while the offset refuses to advance.
_MAX_PAGES = 1000

# Rate limits published at https://www.ncbi.nlm.nih.gov/books/NBK25497/ and read
# live on 2026-08-31: "Without an API key, any site (IP address) posting more
# than 3 requests per second to the E-utilities will receive an error message.
# By including an API key, a site can post up to 10 requests per second."
_MIN_REQUEST_INTERVAL_DEFAULT = 0.34  # ~3 req/s
_MIN_REQUEST_INTERVAL_WITH_KEY = 0.11  # ~10 req/s

# ``<ERROR>`` text fragments that blame the query rather than this connector's
# request shape.  Everything else -- an invalid db name, an out-of-range
# retstart -- is a bug here and must be reported loudly as FAILED, not filed
# under "the source could not express this query".
_QUERY_ERROR_MARKERS = ("empty term", "nothing todo", "can not be translated")

# Prefix Entrez uses for a per-record esummary failure, as opposed to a whole
# -page failure.  Verified: "UID=999999999: cannot get document summary".
_UID_ERROR_PREFIX = "uid="

_PMC_ARTICLE_URL = "https://www.ncbi.nlm.nih.gov/pmc/articles/"


class PMCBodyError(ConnectorError):
    """An HTTP 200 E-utilities response whose XML body reports an error.

    NCBI signals almost everything in the body with a 200 status, so this is
    raised where ``raise_for_status`` would be silent.

    Parameters
    ----------
    message : str
        Message assembled from the body's ``<ERROR>`` text.
    is_query_error : bool
        Whether the body blames the search term rather than the request shape.
        Decides whether the caller reports QUERY_UNSUPPORTED or FAILED.
    """

    def __init__(self, message: str, *, is_query_error: bool = False) -> None:
        """Store the message and the query-versus-request classification."""
        super().__init__(message)
        self.is_query_error = is_query_error


@dataclass(frozen=True, slots=True)
class _SearchHandle:
    """The Entrez history entry one ``esearch`` call created.

    Paging through the history server rather than re-running ``esearch`` with a
    moving ``retstart`` is deliberate: the history entry is a snapshot, so a
    record indexed mid-run cannot shift the window and silently displace a
    record out of the result set.

    Attributes
    ----------
    count : int
        ``<Count>`` -- how many records Entrez says match.
    web_env : str
        ``<WebEnv>`` handle.
    query_key : str
        ``<QueryKey>`` within that handle.
    """

    count: int
    web_env: str
    query_key: str


@dataclass(slots=True)
class _Retrieval:
    """Mutable bookkeeping for one paginated PMC run.

    Exists so that the reasons a run can end stay separable all the way to
    :meth:`PMCConnector._build_coverage`.  Collapsing them into "we have some
    papers" is what turns a truncated run into a confident answer.

    Deliberately a near-twin of ``europe_pmc._Retrieval``.  Hoisting a shared
    bookkeeper is a real option and is recorded as open item **O17** in the
    project specification; resolving it here, in a port, would be resolving an
    open item by guessing.

    Attributes
    ----------
    papers : list[Paper]
        Records parsed so far, in retrieval order.
    processed : int
        Records seen, whether or not they parsed -- including records Entrez
        acknowledged and failed to summarise.
    parsed : int
        Records that became a :class:`~lit_search.core.paper.Paper`.
    pages : int
        ``esummary`` pages consumed.
    total : int | None
        ``<Count>`` from the ``esearch`` response.
    notes : list[str]
        Observations that must reach the coverage status -- a re-tokenised
        phrase, the 9,999-record ceiling.
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
    notes: list[str] = field(default_factory=list)
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


def _classify_error_text(text: str) -> PMCBodyError:
    """Build the error for an Entrez ``<ERROR>`` element.

    Parameters
    ----------
    text : str
        Text of the ``<ERROR>`` element.

    Returns
    -------
    PMCBodyError
        Error carrying the query-versus-request classification.
    """
    lowered = text.lower()
    return PMCBodyError(
        f"NCBI E-utilities reported an error for db={_DB}: {text}",
        is_query_error=any(marker in lowered for marker in _QUERY_ERROR_MARKERS),
    )


def _is_multi_word_phrase(reported: str) -> bool:
    """Whether a phrase Entrez reported as unindexed actually changed the query.

    Entrez raises ``QuotedPhraseNotFound`` whenever a quoted term has no entry
    in its phrase index, **including for a single word**, where the fallback --
    searching the individual words -- is the same search.  Verified live
    2026-08-31: ``"zzqqwibbleflurb"[ti]`` reports the warning and echoes
    ``<QueryTranslation>"zzqqwibbleflurb"[ti]</QueryTranslation>``, identical to
    what was sent, while the multi-word ``"zzqq wibble flurb"`` echoes
    ``"zzqq"[All Fields] AND "wibble"[All Fields] AND "flurb"[All Fields]``.

    Reporting the single-word case would mark an ordinary unmatched term
    ``PARTIAL``, which is the opposite error to the one this fork exists to
    prevent: it would make the honest empty unreachable and train a reader to
    ignore the label.

    Parameters
    ----------
    reported : str
        Text Entrez echoed, for example ``'"machine learning"[ti]'``.

    Returns
    -------
    bool
        ``True`` when the quoted portion holds more than one word.
    """
    first, last = reported.find('"'), reported.rfind('"')
    inner = reported[first + 1 : last] if 0 <= first < last else reported
    return len(inner.split()) > 1


def _element_texts(parent: Element | None, tag: str) -> list[str]:
    """Return the stripped, non-empty texts of *tag* children of *parent*.

    Parameters
    ----------
    parent : Element | None
        Parent element, or ``None``.
    tag : str
        Child tag name.

    Returns
    -------
    list[str]
        Text values.
    """
    if parent is None:
        return []
    return [
        (child.text or "").strip() for child in parent.findall(tag) if (child.text or "").strip()
    ]


class PMCConnector(SearchConnectorBase):
    """Connector for PubMed Central via NCBI E-utilities.

    Uses ``esearch`` (with the history server) followed by ``esummary``:
    https://www.ncbi.nlm.nih.gov/books/NBK25500/

    PMC is not a subset view of PubMed.  Its Entrez index includes ``ARTI``
    (article body), so a term can match text that appears nowhere in a PubMed
    record -- which is the reason to run it alongside PubMed rather than
    instead of it.  It also indexes a *narrower* set of metadata fields; see
    :mod:`lit_search.query.builders.pmc`.
    """

    _RATE_LIMIT_SERVICE = "ncbi"
    """NCBI E-utilities enforce one bucket per API key across *all* of their
    endpoints, and the same key serves PubMed and PMC.

    Scoping the limiter to ``"pmc"`` would give this connector a second clock
    against the same 3 or 10 requests/second allowance that
    :class:`~lit_search.connectors.pubmed.PubmedConnector` already draws on,
    doubling the real rate against one quota.  Both connectors must report
    ``"ncbi"``.
    """

    def __init__(
        self,
        query_builder: PMCQueryBuilder | None = None,
        api_key: str | None = None,
        tool: str | None = None,
        email: str | None = None,
    ) -> None:
        """Create a PMC searcher.

        Parameters
        ----------
        query_builder : PMCQueryBuilder | None
            Builder used to validate and convert queries.  When ``None`` a
            default :class:`PMCQueryBuilder` is created.
        api_key : str | None
            NCBI API key (raises the rate limit from 3 to 10 req/s, and shares
            one bucket with PubMed).
        tool : str | None
            Value of the E-utilities ``tool`` parameter.  NCBI's usage policy
            states that ``tool`` and ``email`` *"must be registered with NCBI"*
            and that unregistered callers "may be blocked".  Left unset the
            parameter is omitted rather than filled with a placeholder: the
            donor connector hardcodes ``openags@example.com``, which
            misidentifies every caller that ports it.
        email : str | None
            Value of the E-utilities ``email`` parameter.  See *tool*.
        """
        super().__init__()
        self._query_builder: PMCQueryBuilder = query_builder or PMCQueryBuilder()
        self._api_key = api_key
        self._tool = tool
        self._email = email
        self._request_interval = (
            _MIN_REQUEST_INTERVAL_WITH_KEY if api_key else _MIN_REQUEST_INTERVAL_DEFAULT
        )

        if not api_key:
            logger.warning(
                "No API key provided for PMC. Without a key, NCBI's limit is 3 requests/second "
                "across all E-utilities endpoints and is shared with the PubMed connector "
                "(instead of 10 req/s with a key). Request a free key at "
                "https://www.ncbi.nlm.nih.gov/account/"
            )
        if not (tool and email):
            logger.warning(
                "PMC: NCBI's usage policy asks that the 'tool' and 'email' parameters be set "
                "and registered, and states that unregistered callers may be blocked. Pass "
                "tool= and email= to PMCConnector to identify this client."
            )

    @property
    def name(self) -> str:
        """Return the database identifier.

        Returns
        -------
        str
            Database name.
        """
        return Database.PMC.value

    @property
    def query_builder(self) -> QueryBuilder:
        """Return the PMC query builder.

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
            Interval in seconds (varies with API key).
        """
        return self._request_interval

    @property
    def rate_limit_service(self) -> str:
        """Return the quota authority, which is NCBI rather than PMC.

        Returns
        -------
        str
            ``"ncbi"``.  See :attr:`_RATE_LIMIT_SERVICE`.
        """
        return self._RATE_LIMIT_SERVICE

    def _prepare_params(self, params: dict) -> dict:
        """Inject the NCBI identification parameters into every request.

        Parameters
        ----------
        params : dict
            Raw query parameters.

        Returns
        -------
        dict
            Parameters with ``api_key``, ``tool`` and ``email`` added when set.
        """
        extra = {
            key: value
            for key, value in (
                ("api_key", self._api_key),
                ("tool", self._tool),
                ("email", self._email),
            )
            if value
        }
        return {**params, **extra} if extra else params

    # ------------------------------------------------------------------
    # Response handling
    # ------------------------------------------------------------------

    @staticmethod
    def _read_xml(response: requests.Response, label: str) -> Element:
        """Parse one E-utilities XML response body.

        Parameters
        ----------
        response : requests.Response
            Response returned by an E-utilities endpoint.
        label : str
            Endpoint name, for the error message.

        Returns
        -------
        Element
            Root element.

        Raises
        ------
        PMCBodyError
            When the body is not parsable XML.  NCBI serves HTML maintenance
            pages with a 200 status, and reading one as an empty result set
            would report an outage as zero papers.
        """
        try:
            root: Element = ET.fromstring(response.text)
        except ET.ParseError as exc:
            raise PMCBodyError(
                f"NCBI {label} returned a body that is not parsable XML "
                f"(status {response.status_code}): {exc}"
            ) from exc
        return root

    @staticmethod
    def _raise_for_error_element(root: Element) -> None:
        """Raise when the response carries a top-level ``<ERROR>``.

        Parameters
        ----------
        root : Element
            Root element of an E-utilities response.

        Raises
        ------
        PMCBodyError
            When an ``<ERROR>`` element is present.
        """
        for text in _element_texts(root, "ERROR"):
            raise _classify_error_text(text)

    @staticmethod
    def _esearch_notes(root: Element) -> list[str]:
        """Collect the ways Entrez silently altered the query.

        Parameters
        ----------
        root : Element
            Root element of an ``esearch`` response.

        Returns
        -------
        list[str]
            Notes that must reach the coverage status.

        Raises
        ------
        UnsupportedQueryError
            When Entrez dropped a field tag.  The search that ran was not the
            search that was asked for, and it returns a plausible count for the
            wrong question -- ``cancer[ab]`` yields the same 4,013,901 records
            as an unscoped search.  Nothing has been retrieved at this point, so
            re-raising here is the whole story.
        """
        errors = root.find("ErrorList")
        warnings = root.find("WarningList")

        dropped = _element_texts(errors, "FieldNotFound")
        if dropped:
            raise UnsupportedQueryError(
                f"PMC does not index the field tag(s) {', '.join(sorted(dropped))}. Entrez "
                "dropped them and re-ran the query against every field, which returns a "
                "plausible count for a different search, so the result is refused rather "
                "than reported."
            )

        notes: list[str] = []
        missing = [
            phrase
            for phrase in _element_texts(errors, "PhraseNotFound")
            + _element_texts(warnings, "QuotedPhraseNotFound")
            if _is_multi_word_phrase(phrase)
        ]
        if missing:
            notes.append(
                f"Entrez found no phrase index entry for {', '.join(sorted(set(missing)))} and "
                "searched the individual words instead, so these records answer a broader "
                "query than the one asked"
            )
        ignored = _element_texts(warnings, "PhraseIgnored")
        if ignored:
            notes.append(
                f"Entrez ignored the term(s) {', '.join(sorted(set(ignored)))}, so the query "
                "that ran is narrower in scope than the one asked"
            )
        return notes

    def _open_search(
        self,
        term: str,
        date_params: dict[str, str],
    ) -> tuple[_SearchHandle, list[str]]:
        """Run ``esearch`` and register the result set on the history server.

        Parameters
        ----------
        term : str
            Entrez term expression.
        date_params : dict[str, str]
            ``datetype`` / ``mindate`` / ``maxdate``, or empty.

        Returns
        -------
        tuple[_SearchHandle, list[str]]
            The history handle and any notes about how Entrez altered the query.

        Raises
        ------
        PMCBodyError
            When the response reports an error or lacks the fields every real
            ``esearch`` response carries.
        UnsupportedQueryError
            When Entrez dropped a field tag from the term.
        """
        params: dict[str, Any] = {
            "db": _DB,
            "term": term,
            "retmode": "xml",
            "usehistory": "y",
            "retmax": 0,
            **date_params,
        }
        # Nothing can have been retrieved yet, so a body that blames the term is
        # the whole story and must reach the ledger as QUERY_UNSUPPORTED rather
        # than as FAILED. Without this the two are indistinguishable to a
        # caller, and one of them invites a pointless retry.
        try:
            root = self._read_xml(self._get(_ESEARCH_URL, params), "esearch")
            self._raise_for_error_element(root)
        except PMCBodyError as exc:
            if exc.is_query_error:
                raise UnsupportedQueryError(str(exc)) from exc
            raise
        notes = self._esearch_notes(root)

        raw_count = (root.findtext("Count") or "").strip()
        if not raw_count.isdigit():
            raise PMCBodyError(
                "NCBI esearch response carried no usable <Count>, so it is not a search "
                "result. Treating it as an empty result set would report a failure as "
                "zero papers."
            )
        count = int(raw_count)
        web_env = (root.findtext("WebEnv") or "").strip()
        query_key = (root.findtext("QueryKey") or "").strip()
        if count > 0 and not (web_env and query_key):
            raise PMCBodyError(
                f"NCBI esearch reported {count} matching records but returned no history "
                "handle, so the records cannot be retrieved. Reporting zero papers here "
                "would hide a result set that exists."
            )
        return _SearchHandle(count=count, web_env=web_env, query_key=query_key), notes

    def _fetch_summary_page(
        self,
        handle: _SearchHandle,
        retstart: int,
        retmax: int,
    ) -> tuple[list[Element], list[str]]:
        """Fetch one ``esummary`` page from the history server.

        Parameters
        ----------
        handle : _SearchHandle
            History handle from :meth:`_open_search`.
        retstart : int
            Offset into the result set.
        retmax : int
            Records requested.

        Returns
        -------
        tuple[list[Element], list[str]]
            The ``<DocSum>`` elements, and the per-record ``<ERROR>`` texts for
            records Entrez acknowledged and could not summarise.

        Raises
        ------
        PMCBodyError
            When the page as a whole failed.
        """
        params: dict[str, Any] = {
            "db": _DB,
            "query_key": handle.query_key,
            "WebEnv": handle.web_env,
            "retstart": retstart,
            "retmax": retmax,
            "retmode": "xml",
        }
        root = self._read_xml(self._get(_ESUMMARY_URL, params), "esummary")
        docsums = root.findall("DocSum")
        uid_errors: list[str] = []
        for text in _element_texts(root, "ERROR"):
            if text.lower().startswith(_UID_ERROR_PREFIX):
                uid_errors.append(text)
            else:
                raise _classify_error_text(text)
        return docsums, uid_errors

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _item_text(docsum: Element, name: str) -> str:
        """Return the text of the named ``<Item>`` in *docsum*.

        Parameters
        ----------
        docsum : Element
            One ``<DocSum>`` element.
        name : str
            Value of the item's ``Name`` attribute.

        Returns
        -------
        str
            Concatenated text, stripped; empty when the item is absent.
        """
        item = docsum.find(f"./Item[@Name='{name}']")
        if item is None:
            return ""
        return "".join(item.itertext()).strip()

    @staticmethod
    def _article_ids(docsum: Element) -> dict[str, str]:
        """Return the ``ArticleIds`` sub-items keyed by their ``Name``.

        Parameters
        ----------
        docsum : Element
            One ``<DocSum>`` element.

        Returns
        -------
        dict[str, str]
            Identifier map, for example ``{"pmcid": "PMC13525871", "doi": ...}``.
        """
        container = docsum.find("./Item[@Name='ArticleIds']")
        if container is None:
            return {}
        ids: dict[str, str] = {}
        for item in container.findall("./Item"):
            key = (item.get("Name") or "").strip().lower()
            value = "".join(item.itertext()).strip()
            if key and value:
                ids[key] = value
        return ids

    @staticmethod
    def _authors(docsum: Element) -> list[Author]:
        """Extract authors from a ``DocSum``.

        ``esummary`` carries names only; affiliations are not part of the
        summary record.

        Parameters
        ----------
        docsum : Element
            One ``<DocSum>`` element.

        Returns
        -------
        list[Author]
            Parsed authors.
        """
        container = docsum.find("./Item[@Name='AuthorList']")
        if container is None:
            return []
        authors: list[Author] = []
        for item in container.findall("./Item"):
            name = "".join(item.itertext()).strip()
            if name:
                authors.append(Author(name=name))
        return authors

    def _pmcid(self, docsum: Element, ids: dict[str, str]) -> str | None:
        """Return the record's PMCID in canonical ``PMC...`` form.

        Parameters
        ----------
        docsum : Element
            One ``<DocSum>`` element.
        ids : dict[str, str]
            Identifier map from :meth:`_article_ids`.

        Returns
        -------
        str | None
            PMCID, or ``None`` when the record carries no usable identifier.
        """
        pmcid = ids.get("pmcid", "").strip()
        if not pmcid:
            uid = (docsum.findtext("Id") or "").strip()
            pmcid = f"PMC{uid}" if uid else ""
        if not pmcid:
            return None
        return pmcid if pmcid.upper().startswith("PMC") else f"PMC{pmcid}"

    def _parse_docsum(self, docsum: Element) -> Paper | None:
        """Parse one ``<DocSum>`` into a :class:`Paper`.

        Parameters
        ----------
        docsum : Element
            One ``<DocSum>`` element.

        Returns
        -------
        Paper | None
            Parsed paper, or ``None`` when a required field is missing or the
            record fails model validation.  Callers **must** count the ``None``
            returns: an unparsed record is a coverage loss, not an absent paper.
        """
        title = self._item_text(docsum, "Title")
        if not title:
            return None
        ids = self._article_ids(docsum)
        pmcid = self._pmcid(docsum, ids)
        if pmcid is None:
            return None

        journal = self._item_text(docsum, "FullJournalName") or self._item_text(docsum, "Source")
        doi = normalize_doi(self._item_text(docsum, "DOI") or ids.get("doi", ""))
        published = self._item_text(docsum, "EPubDate") or self._item_text(docsum, "PubDate")

        try:
            return Paper(
                title=title,
                # esummary carries no abstract; see the module docstring for why
                # efetch is not used to obtain one.
                abstract="",
                authors=self._authors(docsum),
                source=(Source(title=journal, source_type=SourceType.JOURNAL) if journal else None),
                publication_date=parse_date(published or None),
                url=f"{_PMC_ARTICLE_URL}{pmcid}/",
                pdf_url=f"{_PMC_ARTICLE_URL}{pmcid}/pdf/",
                doi=doi,
                found_in={self.name},
                paper_type=PaperType.ARTICLE,
                # PMC is an open-access archive; membership is the criterion,
                # not an inference from the record.
                is_open_access=True,
            )
        except ValueError:
            logger.debug("PMC: record %s failed model validation.", docsum.findtext("Id"))
            return None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @staticmethod
    def _date_params(
        since: datetime.date | None,
        until: datetime.date | None,
    ) -> dict[str, str]:
        """Render the publication-date bounds as E-utilities parameters.

        Entrez accepts full ``YYYY/MM/DD`` precision on ``db=pmc`` (verified
        2026-08-31), so no widening is needed and no bound is lost.

        Parameters
        ----------
        since : datetime.date | None
            Lower publication-date bound.
        until : datetime.date | None
            Upper publication-date bound.

        Returns
        -------
        dict[str, str]
            Parameters to merge into the ``esearch`` request; empty when no
            bound was given.
        """
        if since is None and until is None:
            return {}
        return {
            "datetype": "pdat",
            "mindate": since.strftime("%Y/%m/%d") if since else "1000/01/01",
            "maxdate": until.strftime("%Y/%m/%d") if until else "3000/12/31",
        }

    @staticmethod
    def _retrieval_target(count: int, max_papers: int | None) -> tuple[int, str | None]:
        """Return how many records to page for, and any ceiling note.

        Parameters
        ----------
        count : int
            ``<Count>`` reported by ``esearch``.
        max_papers : int | None
            Caller-imposed cap.

        Returns
        -------
        tuple[int, str | None]
            Target record count, and a note when NCBI's own ceiling -- rather
            than the cap or the result set -- is what bounds the run.
        """
        target = min(count, _MAX_RETRIEVABLE)
        if max_papers is not None:
            target = min(target, max_papers)
        note = None
        if count > _MAX_RETRIEVABLE and (max_papers is None or max_papers > _MAX_RETRIEVABLE):
            note = (
                f"NCBI E-utilities refuse to return more than {_MAX_RETRIEVABLE} records for one "
                f"query, including through the history server, so {count - _MAX_RETRIEVABLE} of "
                f"the {count} reported records are unreachable by this route; NCBI directs "
                f"callers to EDirect for the remainder"
            )
        return target, note

    def _fetch_papers(
        self,
        query: Query,
        max_papers: int | None,
        progress_callback: Callable[[int, int | None], None] | None,
        since: datetime.date | None = None,
        until: datetime.date | None = None,
    ) -> CoveredPapers:
        """Fetch papers from PMC via ``esearch`` + ``esummary``, with coverage.

        Every way this run can end is tracked separately, because they are
        different facts about the result:

        - ``esearch`` failed, or reported an error body -> raised, so the base
          class reports ``FAILED`` (or ``QUERY_UNSUPPORTED`` when Entrez blamed
          or silently rewrote the term);
        - the caller's ``max_papers`` was reached -> ``CAPPED``;
        - an ``esummary`` page failed after records were retrieved ->
          ``PARTIAL`` carrying the error, because the records that did arrive
          are real;
        - NCBI's 9,999-record ceiling bounded the run -> ``PARTIAL``, naming it;
        - Entrez re-tokenised a phrase or ignored a term -> ``PARTIAL``, naming
          the term, because the query that ran is not the query that was asked;
        - records Entrez acknowledged and failed to summarise -> ``PARTIAL``
          with the count;
        - the result set ran out naturally -> ``from_pagination`` decides, and
          it returns ``COMPLETE`` only when as many records came back as Entrez
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
            Only return papers published on or after this date.
        until : datetime.date | None
            Only return papers published on or before this date.

        Returns
        -------
        CoveredPapers
            Retrieved papers and the coverage status describing them.

        Raises
        ------
        UnsupportedQueryError
            When Entrez rejected or silently rewrote the term.
        ConnectorError
            When ``esearch`` failed or returned an unusable body.
        requests.RequestException
            When ``esearch`` failed at the transport level.
        """
        term = self._query_builder.convert_query(query)
        run = _Retrieval()

        handle, notes = self._open_search(term, self._date_params(since, until))
        run.total = handle.count
        run.notes.extend(notes)

        target, ceiling_note = self._retrieval_target(handle.count, max_papers)
        if ceiling_note is not None:
            run.notes.append(ceiling_note)

        self._collect_summaries(run, handle, target, progress_callback)

        if progress_callback is not None:
            progress_callback(run.processed, run.total)

        kept = run.papers[:max_papers] if max_papers is not None else run.papers
        return CoveredPapers(
            papers=tuple(kept),
            coverage=self._build_coverage(run, retrieved=len(kept), max_papers=max_papers),
        )

    def _collect_summaries(
        self,
        run: _Retrieval,
        handle: _SearchHandle,
        target: int,
        progress_callback: Callable[[int, int | None], None] | None,
    ) -> None:
        """Page ``esummary`` until *target* records have been offered.

        The offset advances by the number of records *requested*, not by the
        number that came back, so a page that loses records to per-UID errors
        moves on instead of re-requesting the same window forever.

        Parameters
        ----------
        run : _Retrieval
            Bookkeeping for the current run, updated in place.
        handle : _SearchHandle
            History handle from :meth:`_open_search`.
        target : int
            Number of records to page for.
        progress_callback : Callable[[int, int | None], None] | None
            Progress callback.
        """
        retstart = 0
        while retstart < target:
            retmax = min(_PAGE_SIZE, target - retstart)
            try:
                docsums, uid_errors = self._fetch_summary_page(handle, retstart, retmax)
            except (requests.RequestException, ConnectorError) as exc:
                self._record_page_failure(run, exc, retstart)
                return

            self._absorb_page(run, docsums, uid_errors)
            if progress_callback is not None:
                progress_callback(run.processed, run.total)

            if not docsums and not uid_errors:
                run.truncate(
                    f"NCBI esummary returned neither records nor errors at retstart={retstart}, "
                    f"after {run.processed} of {target} records this run set out to retrieve"
                )
                return
            if run.pages >= _MAX_PAGES:
                run.truncate(
                    f"pagination stopped at the connector's {_MAX_PAGES}-page safety limit "
                    f"after {run.processed} records"
                )
                return
            retstart += retmax

    @staticmethod
    def _record_page_failure(run: _Retrieval, exc: Exception, retstart: int) -> None:
        """Record a page failure, or re-raise when nothing was retrieved.

        The asymmetry is the point of the port.  With no records in hand there
        is no partial result to report and the failure is the whole story, so it
        is re-raised for :meth:`SearchConnectorBase.search` to name; with records
        in hand the run is a real shortfall and the records are kept.  Neither
        path can leave ``_fetch_papers`` as an empty success.

        Parameters
        ----------
        run : _Retrieval
            Bookkeeping for the current run.
        exc : Exception
            The failure raised while fetching or reading the page.
        retstart : int
            Offset the failed request carried, for the message.

        Raises
        ------
        UnsupportedQueryError
            When Entrez blamed the term and nothing had been retrieved.
        Exception
            The original failure, when nothing had been retrieved.
        """
        if not run.papers:
            if isinstance(exc, PMCBodyError) and exc.is_query_error:
                raise UnsupportedQueryError(str(exc)) from exc
            raise exc
        log_pagination_request_failure(
            logger,
            connector_label="PMC",
            processed=run.processed,
            total=run.total,
            position_label="retstart",
            position=retstart,
            exc=exc,
        )
        run.truncate(f"esummary request at retstart={retstart} failed: {exc}", type(exc).__name__)

    def _absorb_page(
        self,
        run: _Retrieval,
        docsums: list[Element],
        uid_errors: list[str],
    ) -> None:
        """Fold one ``esummary`` page into *run*.

        Records Entrez acknowledged and failed to summarise are counted as
        processed but not parsed, which is what makes them show up as a coverage
        loss rather than vanishing.

        Parameters
        ----------
        run : _Retrieval
            Bookkeeping for the current run, updated in place.
        docsums : list[Element]
            ``<DocSum>`` elements the page carried.
        uid_errors : list[str]
            Per-record ``<ERROR>`` texts the page carried.
        """
        for docsum in docsums:
            paper = self._parse_docsum(docsum)
            if paper is not None:
                run.papers.append(paper)
                run.parsed += 1
        run.processed += len(docsums) + len(uid_errors)
        run.pages += 1
        if uid_errors:
            run.notes.append(
                f"NCBI esummary could not return {len(uid_errors)} record(s) it acknowledged "
                f"({'; '.join(uid_errors[:3])}{'; ...' if len(uid_errors) > 3 else ''})"
            )

    def _build_coverage(
        self,
        run: _Retrieval,
        *,
        retrieved: int,
        max_papers: int | None,
    ) -> SourceCoverage:
        """Turn the run's own counters into a coverage status.

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
        notes.extend(run.notes)
        unparsed = run.processed - run.parsed
        if unparsed > 0:
            notes.append(f"{unparsed} of {run.processed} records could not be parsed into papers")
        shortfall = "; ".join(notes) or None

        if run.truncated_early:
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
