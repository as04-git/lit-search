"""arXiv searcher implementation."""

from __future__ import annotations

import contextlib
import datetime
import logging
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

import requests
from defusedxml import ElementTree as ET

if TYPE_CHECKING:
    # Element is the concrete type returned by defusedxml at runtime;
    # we import it from stdlib only for static type annotations.
    from xml.etree.ElementTree import Element  # nosec B405

from lit_search.connectors.doi_lookup_base import DOILookupConnectorBase
from lit_search.connectors.search_base import SearchConnectorBase
from lit_search.connectors.url_lookup_base import URLLookupConnectorBase
from lit_search.core.author import Author
from lit_search.core.citation import IdentifierKind, NativeBatchResult, PaperIdentifier
from lit_search.core.coverage import CoveredPapers, SourceCoverage
from lit_search.core.paper import Database, Paper, PaperType
from lit_search.core.query import Query
from lit_search.core.source import Source, SourceType
from lit_search.query.builder import QueryBuilder
from lit_search.query.builders.arxiv import ArxivQueryBuilder
from lit_search.utils.arxiv_taxonomy import arxiv_category_to_field, arxiv_category_to_subject
from lit_search.utils.normalization import parse_date
from lit_search.utils.pagination_logging import (
    log_pagination_empty_before_total,
    log_pagination_request_failure,
)

logger = logging.getLogger(__name__)

_BASE_URL = "https://export.arxiv.org/api/query"
_PAGE_SIZE = 100
# Keep native ``id_list`` URLs comfortably bounded and give managed callers a
# stable one-request width.  The public batch wrapper chunks larger inputs.
_ID_LIST_BATCH_SIZE = 100
# arXiv recommends at least 3 seconds between requests
_MIN_REQUEST_INTERVAL = 3.0
_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

# Regex to extract the arXiv paper ID from the Atom <id> element.
# Example: http://arxiv.org/abs/1706.03762v5 -> 1706.03762
_ARXIV_ID_RE = re.compile(r"arxiv\.org/abs/([\d.]+)", re.IGNORECASE)

# Regex to extract arXiv paper ID from an arXiv-assigned DOI.
# arXiv assigns DOIs under the 10.48550 prefix: 10.48550/arXiv.1706.03762
_ARXIV_DOI_RE = re.compile(r"^10\.48550/arxiv\.([\d.]+)$", re.IGNORECASE)

# Regex that matches arXiv abstract or PDF landing-page URLs and captures the
# paper ID (without version suffix).  Used by URLLookupConnectorBase.
# Examples:
#   https://arxiv.org/abs/1706.03762
#   https://arxiv.org/abs/1706.03762v3
#   https://arxiv.org/pdf/1706.03762v5
_ARXIV_URL_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?",
    re.IGNORECASE,
)

# Mapping from SourceType to PaperType for arXiv entries.
_ARXIV_PAPER_TYPE_MAP: dict[SourceType, PaperType] = {
    SourceType.JOURNAL: PaperType.ARTICLE,
    SourceType.CONFERENCE: PaperType.INPROCEEDINGS,
    SourceType.BOOK: PaperType.INBOOK,
    SourceType.REPOSITORY: PaperType.UNPUBLISHED,
}


class ArxivConnector(SearchConnectorBase, DOILookupConnectorBase, URLLookupConnectorBase):
    """Connector for the arXiv preprint database.

    Uses the arXiv Atom Feed API:
    https://info.arxiv.org/help/api/user-manual.html

    Rate limit: 3 seconds between requests (as recommended by arXiv).
    """

    _credential_attrs: ClassVar[tuple[str, ...]] = ()
    """arXiv issues no credentials, so the quota is enforced against this host's
    source IP address.  Every arXiv-touching process on the machine therefore
    shares one clock.

    arXiv's terms are stricter than any host-local mechanism can honour: they
    aggregate the limit across "all of the machines under your control".  A
    file lock cannot reach another machine; that would need the network broker
    R8 deliberately rejected.
    """

    def __init__(self, query_builder: ArxivQueryBuilder | None = None) -> None:
        """Create an arXiv searcher.

        Parameters
        ----------
        query_builder : ArxivQueryBuilder | None
            Builder used to validate and convert queries.  When ``None`` a
            default :class:`ArxivQueryBuilder` is created automatically.
        """
        super().__init__()
        self._query_builder: ArxivQueryBuilder = query_builder or ArxivQueryBuilder()

    @property
    def name(self) -> str:
        """Return the database identifier.

        Returns
        -------
        str
            Database name.
        """
        return Database.ARXIV.value

    @property
    def query_builder(self) -> QueryBuilder:
        """Return the arXiv query builder.

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
            Interval in seconds.
        """
        return _MIN_REQUEST_INTERVAL

    # ------------------------------------------------------------------
    # URL lookup
    # ------------------------------------------------------------------

    @property
    def url_pattern(self) -> re.Pattern[str]:
        """Return the regex matching arXiv abstract and PDF URLs.

        Returns
        -------
        re.Pattern[str]
            Compiled regex whose first capture group is the arXiv paper ID.
        """
        return _ARXIV_URL_RE

    def fetch_paper_by_id(self, paper_id: str) -> Paper | None:
        """Fetch a single arXiv paper by its native ID.

        Parameters
        ----------
        paper_id : str
            arXiv paper ID (e.g. ``"1706.03762"``), without version suffix.

        Returns
        -------
        Paper | None
            A populated :class:`~lit_search.core.paper.Paper`, or ``None``
            when the paper is not found or the response cannot be parsed.
        """
        try:
            response = self._get(_BASE_URL, params={"id_list": paper_id, "max_results": 1})
            tree = ET.fromstring(response.text)
        except (requests.RequestException, ET.ParseError):
            logger.debug("arXiv: failed to fetch arXiv ID %s.", paper_id)
            return None

        entries = tree.findall("atom:entry", _NS)
        if not entries:
            logger.debug("arXiv: no entry found for ID %s.", paper_id)
            return None

        return self._parse_paper(entries[0])

    # ------------------------------------------------------------------
    # DOI lookup
    # ------------------------------------------------------------------

    def fetch_paper_by_doi(self, doi: str) -> Paper | None:
        """Fetch a single paper by its DOI from arXiv.

        Only arXiv-assigned DOIs (``10.48550/arXiv.<id>``) are supported.
        For papers whose publisher DOI happens to appear on arXiv the arXiv
        API does not provide a DOI-based search, so ``None`` is returned.

        Parameters
        ----------
        doi : str
            Bare DOI identifier.

        Returns
        -------
        Paper | None
            A populated :class:`~lit_search.core.paper.Paper`, or ``None``
            when the DOI is not an arXiv-native DOI, the paper is not found,
            or the response cannot be parsed.
        """
        m = _ARXIV_DOI_RE.match(doi.strip())
        if not m:
            # Not an arXiv-native DOI: cannot resolve via arXiv API.
            logger.debug("arXiv: DOI %s is not an arXiv-native DOI: skipping.", doi)
            return None

        return self.fetch_paper_by_id(m.group(1))

    def max_paper_batch_size(self) -> int:
        """Return the stable maximum IDs in one native ``id_list`` request."""
        return _ID_LIST_BATCH_SIZE

    def fetch_papers_batch_request(self, paper_ids: list[str]) -> NativeBatchResult:
        """Resolve a bounded set of arXiv IDs in exactly one HTTP request."""
        if len(paper_ids) > _ID_LIST_BATCH_SIZE:
            raise ValueError(
                f"arXiv id_list request accepts at most {_ID_LIST_BATCH_SIZE} IDs, "
                f"got {len(paper_ids)}"
            )
        if not paper_ids:
            return NativeBatchResult(source=self.name, papers=(), requested=0)
        try:
            response = self._get(
                _BASE_URL,
                params={"id_list": ",".join(paper_ids), "max_results": len(paper_ids)},
            )
            tree = ET.fromstring(response.text)
        except (requests.RequestException, ET.ParseError) as exc:
            return NativeBatchResult(
                source=self.name,
                papers=(),
                requested=len(paper_ids),
                request_failures=len(paper_ids),
                error_messages=(f"{type(exc).__name__}: {exc}",),
            )

        papers: list[Paper] = []
        parse_failures = 0
        for entry in tree.findall("atom:entry", _NS):
            paper = self._parse_paper(entry)
            if paper is None:
                parse_failures += 1
            else:
                papers.append(paper)
        unresolved = max(0, len(paper_ids) - len(papers) - parse_failures)
        return NativeBatchResult(
            source=self.name,
            papers=tuple(papers),
            requested=len(paper_ids),
            not_found=unresolved,
            parse_failures=parse_failures,
        )

    def fetch_papers_batch(self, paper_ids: list[str]) -> NativeBatchResult:
        """Resolve arXiv IDs through canonical bounded ``id_list`` requests."""
        outcomes = [
            self.fetch_papers_batch_request(paper_ids[start : start + _ID_LIST_BATCH_SIZE])
            for start in range(0, len(paper_ids), _ID_LIST_BATCH_SIZE)
        ]
        if not outcomes:
            return NativeBatchResult(source=self.name, papers=(), requested=0)
        return NativeBatchResult(
            source=self.name,
            papers=tuple(paper for outcome in outcomes for paper in outcome.papers),
            requested=sum(outcome.requested for outcome in outcomes),
            not_found=sum(outcome.not_found for outcome in outcomes),
            parse_failures=sum(outcome.parse_failures for outcome in outcomes),
            truncated_edge_records=sum(outcome.truncated_edge_records for outcome in outcomes),
            request_failures=sum(outcome.request_failures for outcome in outcomes),
            error_messages=tuple(
                message for outcome in outcomes for message in outcome.error_messages
            ),
        )

    @staticmethod
    def _parse_arxiv_authors(entry: Element) -> list[Author]:
        """Extract authors from an arXiv Atom entry element.

        Parameters
        ----------
        entry : Element
            Atom entry XML element.

        Returns
        -------
        list[Author]
            Parsed author list.
        """
        authors: list[Author] = []
        for author_el in entry.findall("atom:author", _NS):
            name_el = author_el.find("atom:name", _NS)
            if name_el is None or not (name_el.text or "").strip():
                continue
            name = (name_el.text or "").strip()
            affiliation_parts = [
                (aff_el.text or "").strip()
                for aff_el in author_el.findall("arxiv:affiliation", _NS)
                if aff_el is not None and (aff_el.text or "").strip()
            ]
            affiliation = "; ".join(affiliation_parts) if affiliation_parts else None
            authors.append(Author(name=name, affiliation=affiliation))
        return authors

    @staticmethod
    def _parse_arxiv_links(entry: Element) -> tuple[str | None, str | None]:
        """Extract HTML landing-page URL and PDF URL from an arXiv entry.

        Parameters
        ----------
        entry : Element
            Atom entry XML element.

        Returns
        -------
        tuple[str | None, str | None]
            ``(url, pdf_url)`` pair.
        """
        url: str | None = None
        for link_el in entry.findall("atom:link", _NS):
            rel = link_el.get("rel", "")
            href = link_el.get("href", "")
            if rel == "alternate" and href:
                url = href
                break
        if url is None:
            id_el = entry.find("atom:id", _NS)
            if id_el is not None and id_el.text:
                url = id_el.text.strip()
        pdf_url: str | None = None
        for link_el in entry.findall("atom:link", _NS):
            if (link_el.get("title") or "") == "pdf" and link_el.get("href"):
                pdf_url = link_el.get("href") or ""
                break
        return url, pdf_url

    @staticmethod
    def _parse_arxiv_doi(entry: Element, url: str | None) -> str | None:
        """Resolve the DOI for an arXiv entry.

        Prefers the explicit ``<arxiv:doi>`` element (publisher DOI).  When
        absent, derives the canonical arXiv DOI (``10.48550/arXiv.<id>``)
        from the landing-page URL.

        Parameters
        ----------
        entry : Element
            Atom entry XML element.
        url : str | None
            Landing-page URL already extracted from the entry.

        Returns
        -------
        str | None
            Resolved DOI or ``None``.
        """
        doi_el = entry.find("arxiv:doi", _NS)
        if doi_el is not None and doi_el.text:
            return doi_el.text.strip()
        if url:
            m = _ARXIV_ID_RE.search(url)
            if m:
                return f"10.48550/arXiv.{m.group(1)}"
        return None

    @staticmethod
    def _parse_arxiv_source(entry: Element) -> Source:
        """Build a :class:`~lit_search.core.source.Source` from an arXiv entry.

        Papers with a ``<arxiv:journal_ref>`` element were formally published;
        all others are arXiv preprints hosted in a repository.

        Parameters
        ----------
        entry : Element
            Atom entry XML element.

        Returns
        -------
        Source
            Populated source.
        """
        journal_ref_el = entry.find("arxiv:journal_ref", _NS)
        has_journal_ref = (
            journal_ref_el is not None and journal_ref_el.text and journal_ref_el.text.strip()
        )
        if has_journal_ref:
            ref_text = journal_ref_el.text.strip()  # type: ignore[union-attr]
            return Source(
                title=ref_text,
                source_type=_infer_source_type_from_journal_ref(ref_text),
            )
        return Source(title="arXiv", source_type=SourceType.REPOSITORY)

    @staticmethod
    def _parse_arxiv_categories(entry: Element) -> tuple[set[str], set[str]]:
        """Map arXiv category codes to fields of study and subjects.

        Parameters
        ----------
        entry : Element
            Atom entry XML element.

        Returns
        -------
        tuple[set[str], set[str]]
            ``(fields_of_study, subjects)`` sets.
        """
        fields_of_study: set[str] = set()
        subjects: set[str] = set()
        for cat_el in entry.findall("atom:category", _NS):
            term = (cat_el.get("term") or "").strip()
            if not term:
                continue
            field = arxiv_category_to_field(term)
            if field:
                fields_of_study.add(field)
            subject = arxiv_category_to_subject(term)
            if subject:
                subjects.add(subject)
        return fields_of_study, subjects

    def _parse_paper(self, entry: Element) -> Paper | None:
        """Parse a single Atom entry element into a :class:`Paper`.

        Parameters
        ----------
        entry : Element
            Atom entry XML element.

        Returns
        -------
        Paper | None
            Parsed paper or ``None`` when required fields are missing.
        """
        title_el = entry.find("atom:title", _NS)
        abstract_el = entry.find("atom:summary", _NS)
        if title_el is None or not (title_el.text or "").strip():
            return None

        title = (title_el.text or "").strip().replace("\n", " ")
        abstract = (
            (abstract_el.text or "").strip().replace("\n", " ") if abstract_el is not None else ""
        )

        authors = self._parse_arxiv_authors(entry)

        # Published date
        published_el = entry.find("atom:published", _NS)
        pub_date_str = parse_date(
            (published_el.text or "").strip() if published_el is not None else None
        )

        url, pdf_url = self._parse_arxiv_links(entry)
        doi = self._parse_arxiv_doi(entry, url)
        arxiv_match = _ARXIV_ID_RE.search(url or "")
        arxiv_id = arxiv_match.group(1) if arxiv_match else None
        source = self._parse_arxiv_source(entry)

        # Comments: optional free-text note (e.g. "39 pages, 14 figures")
        comment: str | None = None
        comment_el = entry.find("arxiv:comment", _NS)
        if comment_el is not None and comment_el.text and comment_el.text.strip():
            comment = comment_el.text.strip()

        # Infer paper_type from source_type.
        paper_type: PaperType | None = None
        if source is not None and source.source_type is not None:
            paper_type = _ARXIV_PAPER_TYPE_MAP.get(source.source_type)

        fields_of_study, subjects = self._parse_arxiv_categories(entry)

        try:
            paper = Paper(
                title=title,
                abstract=abstract,
                authors=authors,
                source=source,
                publication_date=pub_date_str,
                url=url,
                pdf_url=pdf_url,
                doi=doi,
                comments=comment,
                found_in={self.name},
                paper_type=paper_type,
                fields_of_study=fields_of_study if fields_of_study else None,
                subjects=subjects if subjects else None,
                # All arXiv papers are open access by design.
                is_open_access=True,
                identifiers=(
                    [PaperIdentifier(IdentifierKind.ARXIV, arxiv_id)] if arxiv_id else None
                ),
            )
        except ValueError:
            return None

        return paper

    def _fetch_papers(
        self,
        query: Query,
        max_papers: int | None,
        progress_callback: Callable[[int, int | None], None] | None,
        since: datetime.date | None = None,
        until: datetime.date | None = None,
    ) -> CoveredPapers:
        """Fetch papers from arXiv with pagination, rate limiting and coverage.

        This is the reference implementation of the coverage contract (R7).
        arXiv is a good reference because it reports a result total
        (``opensearch:totalResults``), which is what makes a completeness
        claim justifiable rather than assumed.  Four distinct ways of ending
        the pagination loop are tracked and reported separately:

        - the caller's ``max_papers`` was reached -> ``CAPPED``;
        - a page request failed -> ``FAILED`` if nothing had been retrieved,
          ``PARTIAL`` otherwise, carrying the error;
        - arXiv returned an empty page before the reported total was reached
          (a known behaviour of its paging) -> ``PARTIAL``;
        - the result set ran out naturally -> ``from_pagination`` decides,
          which yields ``COMPLETE`` only when as many records came back as
          arXiv said existed.

        Entries that were seen but could not be parsed are counted and
        reported as a shortfall, so a feed whose schema drifts degrades the
        status instead of quietly returning fewer papers.

        Parameters
        ----------
        query : Query
            Validated query object.
        max_papers : int | None
            Maximum papers to retrieve.
        progress_callback : Callable[[int, int | None], None] | None
            Progress callback.
        since : datetime.date | None
            Only return papers submitted on or after this date.
        until : datetime.date | None
            Only return papers submitted on or before this date.

        Returns
        -------
        CoveredPapers
            Retrieved papers and the coverage status describing them.
        """
        arxiv_query = self._query_builder.convert_query(query)

        # Append arXiv submittedDate range filter when date bounds are given.
        # Use plain spaces: ``requests`` encodes them as ``+`` in the URL,
        # which is what the arXiv API expects.  Literal ``+`` characters were
        # previously double-encoded as ``%2B``, causing the filter to be
        # silently ignored.
        # When ``since`` is not provided we default to 1991-01-01 (arXiv
        # launch year) because year 0000 causes a server error.
        if since or until:
            from_date = since.strftime("%Y%m%d") + "0000" if since else "199101010000"
            to_date = until.strftime("%Y%m%d") + "2359" if until else "999912312359"
            date_filter = f"submittedDate:[{from_date} TO {to_date}]"
            arxiv_query = f"{arxiv_query} AND {date_filter}" if arxiv_query else date_filter
        papers: list[Paper] = []
        processed = 0
        offset = 0
        total: int | None = None
        # Coverage bookkeeping: why the loop stopped, and what was lost.
        truncated_early = False
        error_detail: str | None = None
        error_type: str | None = None

        while True:
            remaining = (max_papers - len(papers)) if max_papers is not None else _PAGE_SIZE
            page_size = min(_PAGE_SIZE, remaining)

            params = {
                "search_query": arxiv_query,
                "start": offset,
                "max_results": page_size,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }

            try:
                response = self._get(_BASE_URL, params)
            except requests.RequestException as exc:
                log_pagination_request_failure(
                    logger,
                    connector_label="arXiv",
                    processed=processed,
                    total=total,
                    position_label="offset",
                    position=offset,
                    exc=exc,
                )
                truncated_early = True
                error_detail = f"page request at offset {offset} failed: {exc}"
                error_type = type(exc).__name__
                break

            tree = ET.fromstring(response.text)

            total_results_el = tree.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
            if total_results_el is not None and total_results_el.text:
                with contextlib.suppress(ValueError):
                    total = int(total_results_el.text.strip())

            entries = tree.findall("atom:entry", _NS)
            if not entries:
                log_pagination_empty_before_total(
                    logger,
                    connector_label="arXiv",
                    processed=processed,
                    total=total,
                    position_label="offset",
                    position=offset,
                )
                if total is not None and processed < total:
                    truncated_early = True
                    error_detail = (
                        f"arXiv returned an empty page at offset {offset} after "
                        f"{processed} of {total} reported entries"
                    )
                break

            for entry in entries:
                paper = self._parse_paper(entry)
                if paper is not None:
                    papers.append(paper)

            processed += len(entries)
            if progress_callback is not None:
                progress_callback(processed, total)

            if max_papers is not None and len(papers) >= max_papers:
                break

            if len(entries) < page_size:
                break

            offset += len(entries)

        # Ensure the progress bar is updated even when the loop exits early
        # (e.g. on the first request returning no entries or a request error),
        # so the bar never stays frozen at its initial 0-paper state.
        if progress_callback is not None:
            progress_callback(processed, total)

        kept = papers[:max_papers] if max_papers is not None else papers
        return CoveredPapers(
            papers=tuple(kept),
            coverage=self._build_coverage(
                retrieved=len(kept),
                parsed=len(papers),
                processed=processed,
                total=total,
                max_papers=max_papers,
                truncated_early=truncated_early,
                error_detail=error_detail,
                error_type=error_type,
            ),
        )

    def _build_coverage(
        self,
        *,
        retrieved: int,
        parsed: int,
        processed: int,
        total: int | None,
        max_papers: int | None,
        truncated_early: bool,
        error_detail: str | None,
        error_type: str | None,
    ) -> SourceCoverage:
        """Turn the pagination loop's own counters into a coverage status.

        Parameters
        ----------
        retrieved : int
            Number of papers actually returned to the caller (post-cap).
        parsed : int
            Number of entries successfully parsed into papers (pre-cap).
        processed : int
            Number of Atom entries seen, parsed or not.
        total : int | None
            Total arXiv reported via ``opensearch:totalResults``.
        max_papers : int | None
            Caller-imposed cap.
        truncated_early : bool
            Whether the loop exited before exhausting the result set for a
            reason other than the cap.
        error_detail : str | None
            Description of the failure that truncated the run.
        error_type : str | None
            Exception class name for that failure.

        Returns
        -------
        SourceCoverage
            The status describing this run.
        """
        notes: list[str] = []
        if error_detail:
            notes.append(error_detail)
        unparsed = processed - parsed
        if unparsed > 0:
            notes.append(f"{unparsed} of {processed} entries could not be parsed into records")
        shortfall = "; ".join(notes) or None

        # Nothing came back and a request failed: this is a failure, not an
        # empty result set. Distinguishing the two is the whole point.
        if retrieved == 0 and error_type is not None and error_detail is not None:
            return SourceCoverage.failed(self.name, error=error_detail, error_type=error_type)

        if truncated_early:
            return SourceCoverage.partial(
                self.name,
                retrieved,
                detail=shortfall or "pagination stopped before the result set was exhausted",
                reported_total=total,
                error=error_detail,
                error_type=error_type,
            )

        return SourceCoverage.from_pagination(
            self.name,
            retrieved=retrieved,
            reported_total=total,
            cap=max_papers,
            shortfall_detail=shortfall,
        )


# ---------------------------------------------------------------------------
# Heuristic patterns to infer source type from arXiv journal_ref text
# ---------------------------------------------------------------------------

_CONFERENCE_RE = re.compile(
    r"\b(?:proceedings?|conference|workshop|symposium)\b" r"|\b(?:proc|conf|symp)\.",
    re.IGNORECASE,
)

_BOOK_RE = re.compile(
    r"\b(?:lecture\s+notes|book|chapter)\b",
    re.IGNORECASE,
)

_JOURNAL_RE = re.compile(
    r"\b(?:"
    r"journal|review[s]?|letters?|transactions?|annals?|bulletin"
    r"|magazine"
    r")\b"
    r"|\bj\."
    r"|\brev\."
    r"|\blett\."
    r"|\btrans\."
    r"|\bann\."
    r"|\bbull\."
    r"|\bmag\.",
    re.IGNORECASE,
)


def _infer_source_type_from_journal_ref(text: str) -> SourceType | None:
    """Infer a :class:`SourceType` from a free-text ``journal_ref`` string.

    The function applies keyword heuristics in priority order:

    1. **CONFERENCE**: contains words like *proceedings*, *conference*,
       *workshop*, or *symposium*.
    2. **BOOK**: contains *lecture notes*, *book*, or *chapter*.
    3. **JOURNAL**: contains common journal indicators such as *journal*,
       *review*, *letters*, *transactions*, abbreviated forms like
       *J.*, *Rev.*, *Lett.*, etc.

    If no pattern matches the text is left unclassified (``None``).

    Parameters
    ----------
    text : str
        The ``journal_ref`` value from an arXiv entry.

    Returns
    -------
    SourceType | None
        Inferred source type or ``None`` when no rule matches.
    """
    if _CONFERENCE_RE.search(text):
        return SourceType.CONFERENCE
    if _BOOK_RE.search(text):
        return SourceType.BOOK
    if _JOURNAL_RE.search(text):
        return SourceType.JOURNAL
    return None
