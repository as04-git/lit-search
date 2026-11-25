"""Paper domain model representing an academic publication."""

from __future__ import annotations

import contextlib
import datetime
import logging
import re
from enum import StrEnum
from typing import Any

from lit_search.exceptions import IdentityConflictError, ModelValidationError

from ..utils.merge import merge_authors, merge_value
from .author import Author
from .citation import (
    CitationEdge,
    CitationTarget,
    IdentifierKind,
    PaperIdentifier,
    merge_edges,
    merge_identifiers,
)
from .identity import has_version_identity_conflict, version_identity_key
from .source import Source

logger = logging.getLogger(__name__)


class PaperType(StrEnum):
    """BibTeX-aligned classification of a paper.

    Each value corresponds directly to a standard BibTeX entry type,
    enabling accurate bibliography generation.

    Attributes
    ----------
    ARTICLE : str
        Journal article (``@article``).
    INPROCEEDINGS : str
        Paper published in conference proceedings (``@inproceedings``).
    INBOOK : str
        Chapter or section in a book (``@inbook``).
    INCOLLECTION : str
        A self-contained part of a book with its own title (``@incollection``).
    BOOK : str
        A complete book (``@book``).
    PHDTHESIS : str
        PhD thesis or doctoral dissertation (``@phdthesis``).
    MASTERSTHESIS : str
        Master's thesis (``@mastersthesis``).
    TECHREPORT : str
        Technical report issued by an institution (``@techreport``).
    UNPUBLISHED : str
        Work not formally published, such as preprints (``@unpublished``).
    MISC : str
        Anything that does not fit the other categories (``@misc``).
    """

    ARTICLE = "article"
    INPROCEEDINGS = "inproceedings"
    INBOOK = "inbook"
    INCOLLECTION = "incollection"
    BOOK = "book"
    PHDTHESIS = "phdthesis"
    MASTERSTHESIS = "mastersthesis"
    TECHREPORT = "techreport"
    UNPUBLISHED = "unpublished"
    MISC = "misc"


class Database(StrEnum):
    """Supported academic database identifiers.

    As a :class:`StrEnum`, each member compares equal to its string value,
    so code such as ``database == "arxiv"`` works without modification.

    ``"web_scraping"`` is intentionally absent: web scraping is a retrieval
    mechanism, not an academic database, and must never be added to a
    :attr:`Paper.found_in` set.
    """

    ARXIV = "arxiv"
    """arXiv preprint server, queried live through its API."""

    ARXIV_BULK = "arxiv_bulk"
    """arXiv metadata read from a local Kaggle snapshot.

    Deliberately a **separate** database from :data:`ARXIV` rather than a mode
    of it.  The two have different recall boundaries -- one is current and
    paginated, the other is complete up to a date and stale after it -- and the
    coverage ledger is keyed on the source, so folding them together would make
    one entry describe two different kinds of incompleteness.  Keeping them
    apart also means ``found_in`` records which of the two actually returned a
    paper.
    """

    BASE_SEARCH = "base_search"
    """BASE (Bielefeld Academic Search Engine).  Degraded: see the connector."""

    BIORXIV = "biorxiv"
    """bioRxiv preprint server.  Degraded: see the connector module."""

    CHEMRXIV = "chemrxiv"
    """chemRxiv chemistry preprint server, reached as a CrossRef filter profile."""

    CORE = "core"
    """CORE aggregator of open-access repository content (api.core.ac.uk)."""

    CROSSREF = "crossref"
    """CrossRef DOI registration authority."""

    DBLP = "dblp"
    """dblp computer science bibliography."""

    DOAJ = "doaj"
    """Directory of Open Access Journals article index."""

    EUROPE_PMC = "europe_pmc"
    """Europe PMC life-sciences literature database (EBI)."""

    GOOGLE_SCHOLAR = "google_scholar"
    """Google Scholar.  Degraded and disabled by default: see the connector."""

    HUGGINGFACE = "huggingface"
    """Hugging Face Papers: ML papers (arXiv ids) submitted to Hugging Face; hybrid
    semantic + keyword search.  Used by ``find`` only; not a Boolean search source."""

    HAL = "hal"
    """HAL, France's national open archive (Hyper Articles en Ligne)."""

    IACR = "iacr"
    """IACR Cryptology ePrint Archive (harvested via OAI-PMH)."""

    IEEE = "ieee"
    """IEEE Xplore digital library."""

    MEDRXIV = "medrxiv"
    """medRxiv preprint server.  Degraded: see the connector module."""

    OPENAIRE = "openaire"
    """OpenAIRE Graph, the European open-science research-product graph."""

    OPENALEX = "openalex"
    """OpenAlex open scholarly graph."""

    OPENREVIEW = "openreview"
    """OpenReview conference proceedings (accepted papers, by venue)."""

    PMC = "pmc"
    """PubMed Central full-text archive (NCBI E-utilities)."""

    PUBMED = "pubmed"
    """PubMed biomedical literature database."""

    SCOPUS = "scopus"
    """Elsevier Scopus abstract and citation database."""

    SEMANTIC_SCHOLAR = "semantic_scholar"
    """Semantic Scholar AI-powered research database."""

    SSRN = "ssrn"
    """SSRN preprint server (Elsevier).  Degraded: see the connector module."""

    WOS = "wos"
    """Web of Science (Clarivate) bibliographic database."""

    ZENODO = "zenodo"
    """Zenodo general-purpose research repository (CERN)."""

    ZOTERO = "zotero"
    """The user's own Zotero library, read through the desktop client's local API."""


# Maximum number of days into the future that a publication date is considered
# plausible.  Dates beyond this threshold are treated as data-quality errors
# from upstream APIs and silently replaced with ``None``.
_MAX_FUTURE_DAYS: int = 365


class Paper:
    """Represents a paper instance."""

    def __init__(
        self,
        title: str,
        abstract: str,
        authors: list[Author],
        source: Source | None,
        publication_date: datetime.date | None,
        url: str | None = None,
        pdf_url: str | None = None,
        doi: str | None = None,
        preprint_doi: str | None = None,
        citations: int | None = None,
        keywords: set[str] | None = None,
        comments: str | None = None,
        page_count: int | None = None,
        page_range: str | None = None,
        found_in: set[str] | None = None,
        paper_type: PaperType | None = None,
        fields_of_study: set[str] | None = None,
        subjects: set[str] | None = None,
        language: str | None = None,
        is_open_access: bool | None = None,
        is_retracted: bool | None = None,
        funders: set[str] | None = None,
        references: list[str] | None = None,
        cited_by: list[str] | None = None,
        identifiers: list[PaperIdentifier] | None = None,
        reference_edges: list[CitationEdge] | None = None,
        cited_by_edges: list[CitationEdge] | None = None,
        citation_edge_totals: dict[str, dict[str, int]] | None = None,
    ) -> None:
        """Create a Paper instance.

        Parameters
        ----------
        title : str
            Paper title.
        abstract : str
            Paper abstract.
        authors : list[Author]
            List of authors.
        source : Source | None
            Source where it was published.
        publication_date : datetime.date | None
            Publication date.
        url : str | None
            URL that references the paper.
        pdf_url : str | None
            Direct URL to PDF file.
        doi : str | None
            Paper DOI.  Where a record has both a publisher DOI and a preprint
            DOI this is the **citable** one, per ``D24``: the canonical
            published record, which is what export and Better BibTeX use.
        preprint_doi : str | None
            DOI of this record's preprint identity, when the record has one
            *in addition to* :paramref:`doi`.  A record whose only identifier
            is a preprint DOI carries it in :paramref:`doi` and leaves this
            ``None``; setting both to the same string says nothing and is
            normalised away.

            This field exists because keeping only one of the two identities
            erases the link ``D24`` version-matching depends on.  An arXiv
            record that names a journal DOI is the *same paper* as the
            preprint every other database indexes as
            ``10.48550/arXiv.<id>``, and a set that dropped one of the two
            reports the preprint and the published version as unrelated
            records -- so a reading list contains both, or a dedup pass
            silently keeps the one whose DOI happened to be first.
        citations : int | None
            Citations count.
        keywords : set[str] | None
            Keywords.
        comments : str | None
            Comments.
        page_count : int | None
            Page count.
        page_range : str | None
            Page range (e.g. ``"223-230"``).
        found_in : set[str] | None
            Databases where found.
        paper_type : PaperType | None
            BibTeX-aligned paper type (informational, not used for filtering).
        fields_of_study : set[str] | None
            Broad knowledge areas (e.g. "Computer Science", "Mathematics").
        subjects : set[str] | None
            More specific disciplinary classifications
            (e.g. "Artificial Intelligence", "Optimization and Control").
        language : str | None
            ISO 639-1 2-letter language code (e.g. ``"en"``, ``"pt"``).
        is_open_access : bool | None
            ``True`` when the paper is freely available online, ``False``
            when it is known to be behind a paywall, ``None`` when unknown.
        is_retracted : bool | None
            ``True`` when the paper was retracted, ``False`` when it is
            known not to be retracted, ``None`` when unknown.
        funders : set[str] | None
            Names of funding agencies or organisations that supported this
            work (e.g. ``{"National Science Foundation", "NIH"}``).  When no
            funding information is available, defaults to an empty set.
        references : list[str] | None
            DOIs of papers cited by this paper (backward references).  Only
            DOIs that are explicitly present in the source data are included;
            references without a DOI are silently omitted.  Defaults to an
            empty list.
        cited_by : list[str] | None
            DOIs of papers that cite this paper (forward references).  Only
            DOIs that are explicitly present in the source data are included.
            Defaults to an empty list.
        identifiers : list[PaperIdentifier] | None
            Typed identifiers supplied by graph sources (Semantic Scholar,
            OpenAlex, arXiv, DOI).  The primary DOI is added automatically.
        reference_edges : list[CitationEdge] | None
            Provenance-bearing backward citation edges.  Unlike
            :paramref:`references`, targets without a DOI remain representable.
        cited_by_edges : list[CitationEdge] | None
            Provenance-bearing forward citation edges.
        citation_edge_totals : dict[str, dict[str, int]] | None
            Per-source reported totals for ``references`` and ``citations``.
            These counters let traversal detect a nested edge list that an API
            truncated even though the paper metadata request itself succeeded.

        Raises
        ------
        ModelValidationError
            If title is missing.
        """
        if not title:
            raise ModelValidationError("Paper's title cannot be null")

        self.title = self._normalize_title(title)
        self.abstract = abstract
        self.authors: list[Author] = list(authors or [])
        self.source = source
        self.publication_date = self._sanitize_date(publication_date)
        self.url = url
        self.pdf_url = pdf_url
        self.doi = doi
        # A preprint DOI that merely repeats the primary DOI records no second
        # identity, and keeping it would make "this record has two identifiers"
        # untrustworthy for every reader that tests the field.
        self.preprint_doi = (
            None
            if (preprint_doi is None or (doi is not None and preprint_doi.lower() == doi.lower()))
            else preprint_doi
        )
        self.citations = citations
        self.keywords = keywords if keywords is not None else set()
        self.comments = comments
        self.page_count = (
            page_count if page_count is not None else self._infer_page_count(page_range)
        )
        self.page_range = page_range
        self.found_in = found_in if found_in is not None else set()
        self.paper_type = paper_type
        self.fields_of_study = fields_of_study if fields_of_study is not None else set()
        self.subjects = subjects if subjects is not None else set()
        self.language = language
        self.is_open_access = is_open_access
        self.is_retracted = is_retracted
        self.funders = funders if funders is not None else set()
        self.references: list[str] = list(references) if references is not None else []
        self.cited_by: list[str] = list(cited_by) if cited_by is not None else []
        self.identifiers: list[PaperIdentifier] = list(identifiers or [])
        if doi:
            self.identifiers = merge_identifiers(
                self.identifiers, [PaperIdentifier(IdentifierKind.DOI, doi)]
            )
            if doi.lower().startswith("10.48550/arxiv."):
                self.identifiers = merge_identifiers(
                    self.identifiers,
                    [PaperIdentifier(IdentifierKind.ARXIV, doi[len("10.48550/arxiv.") :])],
                )
        self.reference_edges: list[CitationEdge] = list(reference_edges or [])
        self.cited_by_edges: list[CitationEdge] = list(cited_by_edges or [])
        self.citation_edge_totals: dict[str, dict[str, int]] = {
            source_name: dict(counts)
            for source_name, counts in (citation_edge_totals or {}).items()
        }

    def __eq__(self, other: object) -> bool:
        """Return whether *other* is this exact in-memory record.

        Scholarly equivalence is deliberately not Python equality.  DOI,
        title, and other bibliographic fields are mutable as records are
        enriched; using any of them here makes a paper's set/dict identity
        change after insertion and silently turns fuzzy title matching into a
        membership operation.

        Parameters
        ----------
        other : object
            Object to compare against.

        Returns
        -------
        bool
            ``True`` if both are :class:`Paper` with matching identity.
        """
        if not isinstance(other, Paper):
            return NotImplemented
        return self is other

    def __hash__(self) -> int:
        """Return the stable object-identity hash for this mutable record.

        Returns
        -------
        int
            Hash value.
        """
        return object.__hash__(self)

    def __repr__(self) -> str:
        """Return a developer-friendly representation.

        Returns
        -------
        str
            Representation string.
        """
        return f"Paper(title={self.title!r}, doi={self.doi!r})"

    def __str__(self) -> str:
        """Return a human-readable citation-like string.

        Format: ``"Author et al. (Year). Title."``
        Falls back gracefully when author or date is missing.

        Returns
        -------
        str
            Friendly citation string.
        """
        parts: list[str] = []
        if self.authors:
            first = self.authors[0].name
            if len(self.authors) > 1:
                parts.append(f"{first} et al.")
            else:
                parts.append(first)
        if self.publication_date is not None:
            parts.append(f"({self.publication_date.year})")
        # Always include the title, terminated with a period
        title = self.title.rstrip(".")
        parts.append(f"{title}.")
        return " ".join(parts)

    def _identity_key(self) -> str | None:
        """Return the legacy DOI-or-title matching key, or ``None``.

        This helper remains temporarily for downstream reducers that have not
        yet migrated to explicit identity policies.  It is not Python object
        identity, source-record identity, or evidence that two records describe
        the same work.  Prefers DOI (lowercased); falls back to lowercased title.

        Returns
        -------
        str | None
            A canonical string key, or ``None`` if no identifier available.
        """
        if self.doi:
            return self.doi.strip().lower()
        if self.title:
            return self.title.strip().lower()
        return None

    @staticmethod
    def _normalize_title(title: str) -> str:
        """Strip HTML tags and normalize whitespace in a paper title.

        Removes any HTML markup (e.g. ``<i>...</i>`` injected by publisher
        APIs) and collapses consecutive whitespace characters (including
        newlines and tabs) into a single space.

        Parameters
        ----------
        title : str
            Raw title string, possibly containing HTML tags or newlines.

        Returns
        -------
        str
            Clean title with HTML removed and whitespace normalized.
        """
        # Remove HTML tags (e.g. <i>, <sub>, <sup>, etc.)
        stripped = re.sub(r"<[^>]+>", "", title)
        # Collapse all whitespace (newlines, tabs, multiple spaces) to a single space
        return " ".join(stripped.split())

    @staticmethod
    def _sanitize_date(
        value: datetime.date | None,
    ) -> datetime.date | None:
        """Return *value* unchanged when plausible, otherwise ``None``.

        Dates more than :data:`_MAX_FUTURE_DAYS` in the future are considered
        data-quality errors from upstream APIs (e.g. OpenAlex placeholder
        dates like 2050-01-01) and are replaced with ``None``.

        Parameters
        ----------
        value : datetime.date | None
            The publication date to validate.

        Returns
        -------
        datetime.date | None
            The original date if plausible, otherwise ``None``.
        """
        if value is None:
            return None
        max_allowed = datetime.date.today() + datetime.timedelta(days=_MAX_FUTURE_DAYS)
        if value > max_allowed:
            logger.debug(
                "Discarding implausible future publication date %s (more than %d days from today).",
                value.isoformat(),
                _MAX_FUTURE_DAYS,
            )
            return None
        return value

    @staticmethod
    def _infer_page_count(page_range: str | None) -> int | None:
        """Try to compute a page count from a hyphen-separated page range.

        If *page_range* matches the pattern ``"<start>-<end>"`` where both
        parts are non-negative integers and ``end >= start``, the page count
        is ``end - start + 1``.  Otherwise ``None`` is returned.

        Parameters
        ----------
        page_range : str | None
            Page range string (e.g. ``"223-230"``).

        Returns
        -------
        int | None
            The computed page count, or ``None`` when *page_range* cannot
            be parsed.
        """
        if not page_range:
            return None

        parts = page_range.split("-")
        if len(parts) != 2:
            return None

        try:
            start = int(parts[0].strip())
            end = int(parts[1].strip())
        except ValueError:
            return None

        if end < start or start < 0:
            return None

        return end - start + 1

    def add_database(self, database_name: str) -> None:
        """Add a database name where the paper was found.

        Parameters
        ----------
        database_name : str
            Database name.

        Returns
        -------
        None
        """
        if database_name:
            self.found_in.add(database_name)

    def _merge_doi_identities(self, paper: Paper) -> None:
        """Fill absent DOI fields after the caller has passed the conflict guard.

        Different non-null version identities never reach this helper:
        :meth:`merge` rejects them before any mutation.  A declared
        :attr:`preprint_doi` is metadata about a separate linked version and is
        retained without promoting it to this record's primary DOI.

        Parameters
        ----------
        paper : Paper
            Another instance of the same paper.
        """
        self.preprint_doi = self.preprint_doi or paper.preprint_doi
        if self.doi is None:
            self.doi = paper.doi
        # A preprint DOI that ended up equal to the primary DOI records no
        # second identity.
        if self.preprint_doi and self.doi and self.preprint_doi.lower() == self.doi.lower():
            self.preprint_doi = None

    def merge(self, paper: Paper) -> None:
        """Merge another paper into this one.

        Parameters
        ----------
        paper : Paper
            Another instance of the same paper.

        Returns
        -------
        None

        Raises
        ------
        IdentityConflictError
            If both records state different non-null publication-version
            identities.  Separate versions must be linked, not merged.
        """
        if has_version_identity_conflict(self, paper):
            raise IdentityConflictError(
                "Refusing to merge paper records with different publication-version "
                f"identities: {version_identity_key(self)!r} and "
                f"{version_identity_key(paper)!r}. Versions must be linked, not merged."
            )

        # Prefer existing dates; if missing, use the incoming one.
        if self.publication_date is None:
            self.publication_date = paper.publication_date

        # Merge scalar fields using shared rules.
        self.title = merge_value(self.title, paper.title)
        # Fill a missing primary identity only.  The guard above prevents this
        # helper from overwriting or discarding a different version identity.
        self._merge_doi_identities(paper)
        self.abstract = merge_value(self.abstract, paper.abstract)
        self.citations = merge_value(self.citations, paper.citations)
        self.comments = merge_value(self.comments, paper.comments)
        self.page_count = merge_value(self.page_count, paper.page_count)
        self.page_range = merge_value(self.page_range, paper.page_range)

        # If page_count is still unknown, try to infer it from page_range.
        if self.page_count is None:
            self.page_count = self._infer_page_count(self.page_range)

        self.url = merge_value(self.url, paper.url)
        self.pdf_url = merge_value(self.pdf_url, paper.pdf_url)

        # Merge authors/keywords as collections while keeping uniqueness.
        # Authors use a token-aware merge to avoid duplicating the same person
        # when different sources represent the name as "First Last" vs "Last, First".
        self.authors = merge_authors(self.authors or [], paper.authors or [])
        self.keywords = merge_value(self.keywords, paper.keywords)
        self.fields_of_study |= paper.fields_of_study
        self.subjects |= paper.subjects
        self.language = merge_value(self.language, paper.language)
        self.is_open_access = merge_value(self.is_open_access, paper.is_open_access)
        self.is_retracted = merge_value(self.is_retracted, paper.is_retracted)
        self.funders |= paper.funders

        # Union reference DOIs preserving insertion order and deduplicating.
        if paper.references:
            seen: set[str] = set(self.references)
            for doi in paper.references:
                if doi not in seen:
                    self.references.append(doi)
                    seen.add(doi)

        # Union cited_by DOIs preserving insertion order and deduplicating.
        if paper.cited_by:
            seen_cb: set[str] = set(self.cited_by)
            for doi in paper.cited_by:
                if doi not in seen_cb:
                    self.cited_by.append(doi)
                    seen_cb.add(doi)

        self.identifiers = merge_identifiers(self.identifiers, paper.identifiers)
        self.reference_edges = merge_edges(self.reference_edges, paper.reference_edges)
        self.cited_by_edges = merge_edges(self.cited_by_edges, paper.cited_by_edges)
        for source_name, counts in paper.citation_edge_totals.items():
            current = self.citation_edge_totals.setdefault(source_name, {})
            for direction, total in counts.items():
                current[direction] = max(current.get(direction, 0), total)

        # Always accumulate databases for traceability.
        self.found_in |= paper.found_in
        if self.source is None:
            self.source = paper.source
        elif paper.source is not None:
            self.source.merge(paper.source)

        self.paper_type = merge_value(self.paper_type, paper.paper_type)

    @staticmethod
    def _from_dict_parse_authors(paper_dict: dict) -> list[Author]:
        """Parse the ``authors`` field from a paper dictionary.

        Parameters
        ----------
        paper_dict : dict
            Raw paper dictionary.

        Returns
        -------
        list[Author]
            Parsed author list.
        """
        raw = paper_dict.get("authors") or []
        if isinstance(raw, (list, set, tuple)):
            return [Author.from_dict(a) for a in raw]
        return [Author.from_dict(raw)]

    @staticmethod
    def _from_dict_parse_date(paper_dict: dict) -> datetime.date | None:
        """Parse the ``publication_date`` field from a paper dictionary.

        Parameters
        ----------
        paper_dict : dict
            Raw paper dictionary.

        Returns
        -------
        datetime.date | None
            Parsed date or ``None``.
        """
        raw = paper_dict.get("publication_date")
        if isinstance(raw, str):
            try:
                return datetime.date.fromisoformat(raw)
            except ValueError:
                return None
        return None

    @staticmethod
    def _from_dict_str_set(paper_dict: dict, key: str) -> set[str]:
        """Parse a string-collection field into a ``set[str]``.

        Handles the case where the raw value is a list/set/tuple, a single
        non-collection value, or missing altogether.

        Parameters
        ----------
        paper_dict : dict
            Raw paper dictionary.
        key : str
            Field name to extract.

        Returns
        -------
        set[str]
            Extracted string set (may be empty).
        """
        raw = paper_dict.get(key) or []
        if isinstance(raw, (list, set, tuple)):
            return {str(item) for item in raw}
        return {str(raw)} if raw else set()

    @staticmethod
    def _from_dict_optional_str(val: Any) -> str | None:
        """Coerce a value to ``str`` or ``None`` when it is already a string.

        Parameters
        ----------
        val : Any
            Raw value to coerce.

        Returns
        -------
        str | None
            String representation or ``None``.
        """
        if val is None:
            return None
        return val if isinstance(val, str) else str(val)

    @staticmethod
    def _from_dict_optional_bool(raw: Any) -> bool | None:
        """Coerce a value to ``bool | None``.

        Parameters
        ----------
        raw : Any
            Raw value.

        Returns
        -------
        bool | None
            ``True``/``False`` or ``None`` when the raw value is ``None``.
        """
        if isinstance(raw, bool):
            return raw
        if raw is not None:
            return bool(raw)
        return None

    @staticmethod
    def _from_dict_paper_type(paper_dict: dict) -> PaperType | None:
        """Parse the ``paper_type`` field from a paper dictionary.

        Parameters
        ----------
        paper_dict : dict
            Raw paper dictionary.

        Returns
        -------
        PaperType | None
            Parsed paper type or ``None``.
        """
        raw = paper_dict.get("paper_type")
        if isinstance(raw, str):
            with contextlib.suppress(ValueError):
                return PaperType(raw)
        return None

    @classmethod
    def from_dict(cls, paper_dict: dict) -> Paper:
        """Create a paper from a dict.

        Parameters
        ----------
        paper_dict : dict
            Paper dictionary.

        Returns
        -------
        Paper
            Paper instance.

        Raises
        ------
        ModelValidationError
            If the title is missing.
        """
        title = paper_dict.get("title")
        if not isinstance(title, str) or not title:
            raise ModelValidationError("Paper's title cannot be null")

        abstract = paper_dict.get("abstract") or ""
        if not isinstance(abstract, str):
            abstract = str(abstract)

        authors = cls._from_dict_parse_authors(paper_dict)

        source_data = paper_dict.get("source")
        source = Source.from_dict(source_data) if isinstance(source_data, dict) else None

        publication_date = cls._from_dict_parse_date(paper_dict)
        url = cls._from_dict_optional_str(paper_dict.get("url"))
        pdf_url = cls._from_dict_optional_str(paper_dict.get("pdf_url"))
        doi = cls._from_dict_optional_str(paper_dict.get("doi"))
        preprint_doi = cls._from_dict_optional_str(paper_dict.get("preprint_doi"))
        language = cls._from_dict_optional_str(paper_dict.get("language"))

        citations = paper_dict.get("citations")
        comments = paper_dict.get("comments")
        page_count = paper_dict.get("page_count")
        page_range = paper_dict.get("page_range")

        raw_refs = paper_dict.get("references") or []
        references = [str(r) for r in raw_refs if isinstance(r, str) and r.strip()]

        raw_cb = paper_dict.get("cited_by") or []
        cited_by = [str(r) for r in raw_cb if isinstance(r, str) and r.strip()]

        raw_identifiers = paper_dict.get("identifiers") or []
        identifiers = [
            PaperIdentifier.from_dict(item) for item in raw_identifiers if isinstance(item, dict)
        ]
        raw_reference_edges = paper_dict.get("reference_edges") or []
        reference_edges = [
            CitationEdge.from_dict(item) for item in raw_reference_edges if isinstance(item, dict)
        ]
        raw_cited_by_edges = paper_dict.get("cited_by_edges") or []
        cited_by_edges = [
            CitationEdge.from_dict(item) for item in raw_cited_by_edges if isinstance(item, dict)
        ]
        raw_edge_totals = paper_dict.get("citation_edge_totals") or {}
        citation_edge_totals = {
            str(source_name): {str(direction): int(total) for direction, total in counts.items()}
            for source_name, counts in raw_edge_totals.items()
            if isinstance(counts, dict)
        }

        keywords = cls._from_dict_str_set(paper_dict, "keywords")
        found_in = cls._from_dict_str_set(paper_dict, "found_in")
        fields_of_study = cls._from_dict_str_set(paper_dict, "fields_of_study")
        subjects = cls._from_dict_str_set(paper_dict, "subjects")
        funders = cls._from_dict_str_set(paper_dict, "funders")

        paper_type = cls._from_dict_paper_type(paper_dict)
        is_open_access = cls._from_dict_optional_bool(paper_dict.get("is_open_access"))
        is_retracted = cls._from_dict_optional_bool(paper_dict.get("is_retracted"))

        return cls(
            title=title,
            abstract=abstract,
            authors=authors,
            source=source,
            publication_date=publication_date,
            url=url,
            pdf_url=pdf_url,
            doi=doi,
            preprint_doi=preprint_doi,
            citations=citations,
            keywords=keywords,
            comments=comments,
            page_count=page_count,
            page_range=page_range,
            found_in=found_in,
            paper_type=paper_type,
            fields_of_study=fields_of_study,
            subjects=subjects,
            language=language,
            is_open_access=is_open_access,
            is_retracted=is_retracted,
            funders=funders,
            references=references,
            cited_by=cited_by,
            identifiers=identifiers,
            reference_edges=reference_edges,
            cited_by_edges=cited_by_edges,
            citation_edge_totals=citation_edge_totals,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this Paper to a plain dictionary.

        Returns
        -------
        dict[str, Any]
            Paper data suitable for JSON serialization.
        """
        return {
            "title": self.title,
            "abstract": self.abstract,
            "authors": [author.to_dict() for author in self.authors],
            "source": (self.source.to_dict() if self.source is not None else None),
            "publication_date": (
                self.publication_date.isoformat() if self.publication_date is not None else None
            ),
            "url": self.url,
            "pdf_url": self.pdf_url,
            "doi": self.doi,
            "preprint_doi": self.preprint_doi,
            "citations": self.citations,
            "keywords": sorted(self.keywords),
            "comments": self.comments,
            "page_count": self.page_count,
            "page_range": self.page_range,
            "found_in": sorted(self.found_in),
            "paper_type": self.paper_type.value if self.paper_type else None,
            "fields_of_study": sorted(self.fields_of_study),
            "subjects": sorted(self.subjects),
            "language": self.language,
            "is_open_access": self.is_open_access,
            "is_retracted": self.is_retracted,
            "funders": sorted(self.funders),
            "references": sorted(self.references),
            "cited_by": sorted(self.cited_by),
            "identifiers": [
                identifier.to_dict()
                for identifier in sorted(self.identifiers, key=lambda item: item.key)
            ],
            "reference_edges": [
                edge.to_dict()
                for edge in sorted(
                    self.reference_edges,
                    key=lambda item: (item.source, tuple(sorted(item.target.keys))),
                )
            ],
            "cited_by_edges": [
                edge.to_dict()
                for edge in sorted(
                    self.cited_by_edges,
                    key=lambda item: (item.source, tuple(sorted(item.target.keys))),
                )
            ],
            "citation_edge_totals": self.citation_edge_totals,
        }

    def citation_target(self) -> CitationTarget:
        """Return this paper as a typed citation target.

        Every :class:`Paper` has a title, so the fallback cannot be empty in
        ordinary persisted records.  The fallback title identifier is not
        used for native routing; it only makes the invariant violation loud if
        a caller constructed a paper without any external identifier.
        """
        if self.identifiers:
            return CitationTarget(tuple(self.identifiers), title=self.title)
        raise ValueError(f"Paper {self.title!r} has no routable citation identifier")


def identity_key(paper: Paper) -> str:
    """Return the legacy DOI-or-title matching string.

    This compatibility helper still feeds downstream handle, deduplication and
    index code.  It must not be confused with strict record identity or inferred
    work equivalence; those consumers require a later explicit migration.

    It deliberately does **not** collapse a preprint and its published version:
    those have different DOIs, so they get different keys, and the relationship
    between them is recorded as an explicit link rather than by making them
    equal (``D24``).

    Parameters
    ----------
    paper : Paper
        Paper to key.

    Returns
    -------
    str
        Lower-cased DOI when the paper has one, otherwise its lower-cased
        title.  Never empty: :class:`Paper` refuses a blank title.
    """
    key = paper._identity_key()
    return key if key is not None else paper.title.strip().lower()
