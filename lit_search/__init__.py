"""Findpapers - Academic paper search and management tool.

The package surface is grouped in three layers.

**Operations.**  :class:`~lit_search.engine.Engine` and the runners behind it.

**Coverage.**  :class:`~lit_search.core.coverage.CoverageLedger` and friends
are the record of what a search reached and what it did not, and
:mod:`lit_search.core.rendering` is how that record is shown.  They are
exported here because a caller who cannot see them from the package surface
will not use them, and an unused ledger is the failure this fork exists to
prevent.

**Handles.**  :class:`~lit_search.store.store.HandleStore` addresses immutable
on-disk sets shared between the command line and any future MCP layer.  Only
the store entry points are re-exported here; the set-algebra, lineage and
ordering surface stays reachable as ``lit_search.store.*`` rather than being
promoted to the top level, because the store's scope is still an open decision
(``O4``/``O12``) and re-exporting it would settle that by accident.

.. warning::

   :func:`~lit_search.utils.persistence.save_to_bibtex` mints its own citation
   keys through ``citation_key_for``.  The project specification (``D7``) makes
   Better BibTeX the single authority for citation keys, so keys produced here
   will silently disagree with a Zotero bibliography.  The function is still
   exported because removing it is a separate, breaking change; treat its
   output as provisional.
"""

from lit_search.core.author import Author
from lit_search.core.citation import (
    CitationEdge,
    CitationTarget,
    IdentifierKind,
    PaperIdentifier,
)
from lit_search.core.coverage import (
    ZERO_COVERAGE_STATUSES,
    CoverageLedger,
    CoverageScope,
    CoverageStatus,
    CoveredPapers,
    DerivationNote,
    ExecutionStatus,
    SourceCoverage,
    difference_coverage,
    intersection_coverage,
    severity,
    union_coverage,
    worst,
)
from lit_search.core.lookup_result import LookupResult
from lit_search.core.paper import Database, Paper, PaperType
from lit_search.core.query import ConnectorType, FilterCode
from lit_search.core.query_family import QueryFamilyRevision
from lit_search.core.query_family_result import QueryFamilyResult
from lit_search.core.rendering import (
    CoverageView,
    coverage_banner,
    coverage_footer,
    coverage_view,
    render_coverage_report,
    render_result,
    result_repr,
    result_summary,
)
from lit_search.core.search_result import SearchResult
from lit_search.core.similar_result import SimilarResult
from lit_search.core.snowball_result import SnowballResult
from lit_search.core.source import Source, SourceType
from lit_search.engine import Engine
from lit_search.exceptions import (
    ConnectorError,
    IdentityConflictError,
    InvalidParameterError,
    LitSearchError,
    MissingApiKeyError,
    ModelValidationError,
    PersistenceError,
    QueryValidationError,
    UnsupportedQueryError,
)
from lit_search.runners.download_runner import DownloadRunner
from lit_search.runners.get_runner import GetRunner
from lit_search.runners.query_family_runner import QueryFamilyRunner
from lit_search.runners.search_runner import SearchRunner
from lit_search.runners.similar_runner import SimilarRunner
from lit_search.runners.snowball_runner import SnowballRunner
from lit_search.store import (
    APP_DIRECTORY_NAME,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    STORE_ROOT_ENV_VAR,
    HandleRecord,
    HandleStore,
    QueryFamilyStore,
    compute_handle_id,
    default_store_root,
)
from lit_search.utils.persistence import (
    load_from_bibtex,
    load_from_csv,
    load_from_json,
    save_to_bibtex,
    save_to_csv,
    save_to_json,
)

__all__ = [
    "APP_DIRECTORY_NAME",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "STORE_ROOT_ENV_VAR",
    "ZERO_COVERAGE_STATUSES",
    "Author",
    "CitationEdge",
    "CitationTarget",
    "ConnectorError",
    "ConnectorType",
    "CoverageLedger",
    "CoverageScope",
    "CoverageStatus",
    "CoverageView",
    "CoveredPapers",
    "Database",
    "DerivationNote",
    "DownloadRunner",
    "Engine",
    "ExecutionStatus",
    "FilterCode",
    "GetRunner",
    "HandleRecord",
    "HandleStore",
    "IdentifierKind",
    "IdentityConflictError",
    "InvalidParameterError",
    "LitSearchError",
    "LookupResult",
    "MissingApiKeyError",
    "ModelValidationError",
    "Paper",
    "PaperIdentifier",
    "PaperType",
    "PersistenceError",
    "QueryFamilyResult",
    "QueryFamilyRevision",
    "QueryFamilyRunner",
    "QueryFamilyStore",
    "QueryValidationError",
    "SearchResult",
    "SearchRunner",
    "SimilarResult",
    "SimilarRunner",
    "SnowballResult",
    "SnowballRunner",
    "Source",
    "SourceCoverage",
    "SourceType",
    "UnsupportedQueryError",
    "compute_handle_id",
    "coverage_banner",
    "coverage_footer",
    "coverage_view",
    "default_store_root",
    "difference_coverage",
    "intersection_coverage",
    "load_from_bibtex",
    "load_from_csv",
    "load_from_json",
    "render_coverage_report",
    "render_result",
    "result_repr",
    "result_summary",
    "save_to_bibtex",
    "save_to_csv",
    "save_to_json",
    "severity",
    "union_coverage",
    "worst",
]
