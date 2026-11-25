"""Connector registry for academic database integrations.

Provides central registries that map source identifiers to their connector
classes, so that runners can discover connectors without importing each one
individually.

To register a **new search connector**:
    1. Create the connector class inheriting from
       :class:`~lit_search.connectors.search_base.SearchConnectorBase`.
    2. Add a corresponding member to
       :class:`~lit_search.core.paper.Database`.
    3. Add an entry to :data:`SEARCH_REGISTRY` below.

    **The coverage contract is not optional and is not a style preference.**
    ``_fetch_papers`` returns a
    :class:`~lit_search.core.coverage.CoveredPapers`: the records **and** an
    explicit :class:`~lit_search.core.coverage.SourceCoverage` saying how
    complete they are.  A connector that returns a bare ``list[Paper]`` is
    rejected when its module is imported, because an empty list is
    indistinguishable from "this database holds no matching paper" and that
    ambiguity is how a literature search reports a confident, incomplete
    answer.

    Concretely, when porting a connector:

    - Do **not** ``except`` a request failure and ``return []``.  Let it
      propagate (the base class turns the known failure classes into
      ``FAILED``), or return
      :meth:`~lit_search.core.coverage.SourceCoverage.partial` when some
      records had already been retrieved.
    - Prefer
      :meth:`~lit_search.core.coverage.SourceCoverage.from_pagination`, which
      applies the cap / shortfall / unverifiable decision procedure from the
      connector's own counters.
    - A source that cannot express the query (a preprint server that only
      filters by category, for example) reports ``QUERY_UNSUPPORTED`` or
      ``PARTIAL``.  It must never be registered as if it answered the query.
    - If you cannot yet justify a status, subclass
      :class:`~lit_search.connectors.uninstrumented_base.UninstrumentedSearchConnectorBase`,
      which reports everything as ``PARTIAL``.  Guessing ``COMPLETE`` is the
      one outcome the design exists to prevent.

To register a **new DOI-lookup connector**:
    1. Create the connector class inheriting from
       :class:`~lit_search.connectors.doi_lookup_base.DOILookupConnectorBase`.
    2. Add a corresponding member to
       :class:`~lit_search.core.paper.Database`.
    3. Add an entry to :data:`DOI_LOOKUP_REGISTRY` below.

To register a **new URL-lookup connector**:
    1. Create the connector class inheriting from
       :class:`~lit_search.connectors.url_lookup_base.URLLookupConnectorBase`.
    2. Add a corresponding member to
       :class:`~lit_search.core.paper.Database`.
    3. Add an entry to :data:`URL_LOOKUP_REGISTRY` below.
"""

from __future__ import annotations

from lit_search.connectors.arxiv import ArxivConnector
from lit_search.connectors.arxiv_bulk import ArxivBulkConnector
from lit_search.connectors.base_search import BaseSearchConnector
from lit_search.connectors.biorxiv import BioRxivConnector, MedRxivConnector
from lit_search.connectors.chemrxiv import ChemRxivConnector
from lit_search.connectors.core_ac_uk import CoreConnector
from lit_search.connectors.crossref import CrossRefConnector
from lit_search.connectors.dblp import DBLPConnector
from lit_search.connectors.doaj import DOAJConnector
from lit_search.connectors.doi_lookup_base import DOILookupConnectorBase
from lit_search.connectors.europe_pmc import EuropePMCConnector
from lit_search.connectors.google_scholar import GoogleScholarConnector
from lit_search.connectors.hal import HALConnector
from lit_search.connectors.iacr import IACRConnector
from lit_search.connectors.ieee import IEEEConnector
from lit_search.connectors.openaire import OpenAIREConnector
from lit_search.connectors.openalex import OpenAlexConnector
from lit_search.connectors.openreview import OpenReviewConnector
from lit_search.connectors.pmc import PMCConnector
from lit_search.connectors.pubmed import PubmedConnector
from lit_search.connectors.scopus import ScopusConnector
from lit_search.connectors.search_base import SearchConnectorBase
from lit_search.connectors.semantic_scholar import SemanticScholarConnector
from lit_search.connectors.ssrn import SSRNConnector
from lit_search.connectors.url_lookup_base import URLLookupConnectorBase
from lit_search.connectors.wos import WosConnector
from lit_search.connectors.zenodo import ZenodoConnector
from lit_search.connectors.zotero import ZoteroConnector
from lit_search.core.paper import Database

# Central mapping of Database identifiers to their search connector classes.
SEARCH_REGISTRY: dict[Database, type[SearchConnectorBase]] = {
    Database.ARXIV: ArxivConnector,
    # The local arXiv snapshot is a separate database from the live API, not a
    # mode of it: their recall boundaries differ in kind, and the ledger is
    # keyed on the source.  It reports SKIPPED_NO_CREDENTIAL until a snapshot is
    # configured, so a search that has none says so instead of omitting arXiv's
    # bulk corpus silently.
    Database.ARXIV_BULK: ArxivBulkConnector,
    # BASE is registered and always skipped.  Its search interface refuses this
    # host with HTTP 200 and an XML error body until the IP is registered, and
    # the OAI-PMH route the donor used now 404s.  Registered so that "BASE was
    # not consulted" is a sentence the ledger says rather than an absence.
    Database.BASE_SEARCH: BaseSearchConnector,
    # bioRxiv and medRxiv have no keyword search.  They are driven by date
    # window with the query applied locally, and are never COMPLETE -- the
    # donor's thirty-day horizon is now reported instead of hidden.
    Database.BIORXIV: BioRxivConnector,
    Database.CHEMRXIV: ChemRxivConnector,
    Database.CORE: CoreConnector,
    Database.DBLP: DBLPConnector,
    Database.DOAJ: DOAJConnector,
    Database.EUROPE_PMC: EuropePMCConnector,
    # Google Scholar ships disabled: scholar.google.com/robots.txt disallows
    # /scholar for every agent, so it reports SKIPPED_NO_CREDENTIAL until the
    # operator sets LIT_SEARCH_GOOGLE_SCHOLAR_ENABLED.  Registered rather than
    # omitted for the same reason as BASE.
    Database.GOOGLE_SCHOLAR: GoogleScholarConnector,
    Database.HAL: HALConnector,
    # IACR is registered even though it reports SKIPPED_NO_CREDENTIAL until the
    # operator records the archive's whitelisting.  Registering it keeps the
    # source visible in every coverage ledger as one that was deliberately not
    # consulted; leaving it out would make "we never asked IACR" and "IACR does
    # not exist" the same absence, which is the ambiguity this fork removes.
    Database.IACR: IACRConnector,
    Database.IEEE: IEEEConnector,
    Database.MEDRXIV: MedRxivConnector,
    Database.OPENAIRE: OpenAIREConnector,
    Database.OPENALEX: OpenAlexConnector,
    # OpenReview reports SKIPPED_NO_CREDENTIAL without OPENREVIEW_USERNAME and
    # OPENREVIEW_PASSWORD, and that is not a policy choice: an anonymous read of
    # its notes endpoint is refused outright (HTTP 403 ChallengeRequiredError,
    # verified 2026-08-31).  It is registered anyway so that "the proceedings
    # were not consulted" appears in every ledger instead of being an absence.
    Database.OPENREVIEW: OpenReviewConnector,
    Database.PMC: PMCConnector,
    Database.PUBMED: PubmedConnector,
    Database.SCOPUS: ScopusConnector,
    Database.SEMANTIC_SCHOLAR: SemanticScholarConnector,
    # SSRN is registered even though it can never run without a credential the
    # project does not have.  That is the point: an unregistered source is
    # invisible, and "SSRN was not consulted" has to be sayable.  It reports
    # SKIPPED_NO_CREDENTIAL by default and is never COMPLETE.
    Database.SSRN: SSRNConnector,
    Database.WOS: WosConnector,
    Database.ZENODO: ZenodoConnector,
    # Zotero is a search source as well as the set a search is subtracted from
    # and the export sink (D19).  Registering it as a searchable database is
    # what makes "the library was consulted, and here is how completely" a
    # sentence the ledger can say -- which matters most for the subtraction,
    # where a silently empty library marks every held paper as new.
    Database.ZOTERO: ZoteroConnector,
}

# Mapping of Database identifiers to DOI-lookup connector classes.
# Europe PMC, DOAJ and PMC are deliberately absent: all three are search
# sources only.  Each could support a lookup -- Europe PMC has a ``DOI:"..."``
# field, DOAJ a ``doi:`` short-hand, PMC an Entrez ``[doi]`` term -- but a
# connector that is registered for a role it has not implemented and tested is
# how a lookup path comes to return None for reasons nobody recorded.
# CrossRef is listed first as the canonical DOI registration authority.
DOI_LOOKUP_REGISTRY: dict[Database, type[DOILookupConnectorBase]] = {
    Database.CROSSREF: CrossRefConnector,
    Database.ARXIV: ArxivConnector,
    Database.IEEE: IEEEConnector,
    Database.OPENALEX: OpenAlexConnector,
    Database.PUBMED: PubmedConnector,
    Database.SCOPUS: ScopusConnector,
    Database.SEMANTIC_SCHOLAR: SemanticScholarConnector,
    Database.WOS: WosConnector,
}

# Mapping of Database identifiers to URL-lookup connector classes.
# CrossRef and Scopus do not implement URLLookupConnectorBase and are absent.
URL_LOOKUP_REGISTRY: dict[Database, type[URLLookupConnectorBase]] = {
    Database.ARXIV: ArxivConnector,
    Database.IEEE: IEEEConnector,
    Database.OPENALEX: OpenAlexConnector,
    Database.PUBMED: PubmedConnector,
    Database.SEMANTIC_SCHOLAR: SemanticScholarConnector,
    Database.WOS: WosConnector,
}
