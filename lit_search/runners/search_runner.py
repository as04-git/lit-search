"""SearchRunner: the main entry point for performing academic paper searches."""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from tqdm import tqdm

from lit_search.connectors import SEARCH_REGISTRY
from lit_search.connectors.search_base import SearchConnectorBase
from lit_search.core.coverage import CoverageLedger, CoverageScope, CoveredPapers, SourceCoverage
from lit_search.core.identity import record_identity_keys, same_record_identity
from lit_search.core.paper import Database, Paper
from lit_search.core.rendering import log_coverage
from lit_search.core.retrieval_audit import verify_coverage_claim
from lit_search.core.search_result import SearchResult
from lit_search.exceptions import (
    CoverageContractError,
    InvalidParameterError,
    MissingApiKeyError,
    UnsupportedQueryError,
)
from lit_search.query.normalizer import QueryNormalizer
from lit_search.query.parser import QueryParser
from lit_search.query.propagator import FilterPropagator
from lit_search.query.validator import QueryValidator
from lit_search.runners.discovery_runner import DEFAULT_ENRICHMENT_DATABASES, DiscoveryRunner
from lit_search.utils.logging_config import configure_verbose_logging
from lit_search.utils.parallel import execute_tasks
from lit_search.utils.progress import make_progress_bar

logger = logging.getLogger(__name__)

# The normal search profile contains remote sources that are useful without
# bespoke local setup, private institutional subscriptions, or deliberately
# slow anonymous access.  ``databases=["all"]`` remains the explicit audit
# sweep over the complete registry.
DEFAULT_SEARCH_DATABASES: tuple[str, ...] = (
    "arxiv",
    "biorxiv",
    "chemrxiv",
    "core",
    "dblp",
    "doaj",
    "europe_pmc",
    "hal",
    "medrxiv",
    "openalex",
    "pmc",
    "pubmed",
    "semantic_scholar",
    "zenodo",
)

# Sources that are useful in ordinary discovery but cannot be instantiated
# anonymously. They join the default only when their actual credential
# material is present, so an unconfigured installation does not make every
# normal search operationally incomplete.
CREDENTIALLED_DEFAULT_SEARCH_DATABASES: tuple[str, ...] = ("openaire",)

# Public SearchRunner parameters that can affect each registered connector.
# Keeping this exhaustive makes reachability testable when a connector is
# added: an empty tuple is an explicit statement that it has no source-specific
# public configuration, not an omission from a hand-written credentials map.
SEARCH_CONNECTOR_CONFIG_PARAMETERS: dict[Database, tuple[str, ...]] = {
    Database.ARXIV: (),
    Database.ARXIV_BULK: ("arxiv_snapshot_path", "arxiv_categories"),
    Database.BASE_SEARCH: (),
    Database.BIORXIV: (),
    Database.CHEMRXIV: ("chemrxiv_email",),
    Database.CORE: ("core_api_key",),
    Database.DBLP: (),
    Database.DOAJ: (),
    Database.EUROPE_PMC: (),
    Database.GOOGLE_SCHOLAR: (),
    Database.HAL: (),
    Database.IACR: (),
    Database.IEEE: ("ieee_api_key",),
    Database.MEDRXIV: (),
    Database.OPENAIRE: (
        "openaire_access_token",
        "openaire_refresh_token",
        "openaire_access_token_expires_in",
    ),
    Database.OPENALEX: ("openalex_api_key", "email"),
    Database.OPENREVIEW: (
        "openreview_username",
        "openreview_password",
        "openreview_venue_ids",
        "openreview_cache_dir",
        "openreview_cache_ttl_seconds",
    ),
    Database.PMC: ("pubmed_api_key", "ncbi_tool", "ncbi_email"),
    Database.PUBMED: ("pubmed_api_key", "ncbi_tool", "ncbi_email"),
    Database.SCOPUS: ("scopus_api_key",),
    Database.SEMANTIC_SCHOLAR: ("semantic_scholar_api_key",),
    Database.SSRN: ("ssrn_api_key",),
    Database.WOS: ("wos_api_key",),
    Database.ZENODO: ("zenodo_access_token",),
    Database.ZOTERO: ("zotero_base_url", "zotero_library"),
}


#: Constructor credential each key-gated database is skipped for, named as the
#: caller would supply it.  A skipped database is recorded in the coverage
#: ledger with this name, so the reader is told what to provide rather than
#: only that something was missing.
_CREDENTIAL_FOR_DATABASE: dict[str, str] = {
    Database.ARXIV_BULK.value: "arxiv_snapshot_path",
    Database.BASE_SEARCH.value: "base_search_ip_registration",
    Database.GOOGLE_SCHOLAR.value: "LIT_SEARCH_GOOGLE_SCHOLAR_ENABLED",
    Database.IACR.value: "LIT_SEARCH_IACR_OAI_ENABLED",
    Database.IEEE.value: "ieee_api_key",
    Database.OPENREVIEW.value: "openreview_username and openreview_password",
    Database.SCOPUS.value: "scopus_api_key",
    Database.SSRN.value: "ssrn_api_key",
    Database.WOS.value: "wos_api_key",
}


def _credential_name(database: str) -> str:
    """Return the credential name *database* is skipped for.

    Parameters
    ----------
    database : str
        Database identifier.

    Returns
    -------
    str
        Credential name, defaulting to ``"<database>_api_key"`` for a
        connector added after this map was written.  The default is a guess
        about the *name*, never about whether the database was reached.
    """
    return _CREDENTIAL_FOR_DATABASE.get(database, f"{database}_api_key")


class SearchRunner(DiscoveryRunner):
    """Public API entry point for running academic paper searches.

    The runner orchestrates the full pipeline:

    1. Parse and validate the query string.
    2. Fetch papers from each configured database searcher.
    3. Merge records only when explicit DOI, arXiv, Semantic Scholar or
       OpenAlex identities establish non-conflicting record sameness. Different
       versions and identifier-free records remain separate for later explicit
       deduplication and version linking.
    Parameters
    ----------
    query : str
        Raw query string (e.g. ``"ti[machine learning] AND abs[deep learning]"``).
    databases : list[str] | None
        Database identifiers to query.  ``None`` selects the usable default
        profile; ``["all"]`` requests every registered search source.
    max_papers_per_database : int | None
        Maximum papers to retrieve from each database.  ``None`` means
        unlimited.
    ieee_api_key : str | None
        IEEE Xplore API key.
    scopus_api_key : str | None
        Elsevier / Scopus API key.
    pubmed_api_key : str | None
        NCBI PubMed API key (increases rate limit).
    openalex_api_key : str | None
        OpenAlex API key.
    email : str | None
        Shared contact identity for Crossref-backed requests.
    semantic_scholar_api_key : str | None
        Semantic Scholar API key (increases rate limit).
    wos_api_key : str | None
        Clarivate Web of Science API key.
    num_workers : int
        Number of parallel workers for running database searchers
        concurrently.  Defaults to ``1``, which runs all searchers
        sequentially.  Values greater than ``1`` enable parallel execution.
    since : dt.date | None
        Only return papers published on or after this date.
    until : dt.date | None
        Only return papers published on or before this date.
    enrichment_databases : list[str] | None
        Databases used to enrich papers after search and filtering.
        ``None`` (default) runs enrichment against ``crossref`` and
        ``web_scraping``, which covers the majority of metadata gaps without
        consuming quota from rate-limited databases.  Accepted values:
        ``"arxiv"``, ``"crossref"``, ``"ieee"``, ``"openalex"``,
        ``"pubmed"``, ``"scopus"``, ``"semantic_scholar"``,
        ``"web_scraping"``, ``"wos"``.
        Pass ``[]`` to disable enrichment entirely.
    max_cited_by : int | None
        Maximum number of citing-paper DOIs to collect per paper when
        ``"openalex"`` or ``"semantic_scholar"`` are in *enrichment_databases*.
        Defaults to ``100``.  ``None`` means no limit: use with caution as
        highly-cited papers may have thousands of citations.  A warning is
        emitted when this value is ``None`` or greater than ``100``.
    proxy : str | None
        Optional HTTP/HTTPS proxy URL forwarded to the enrichment
        :class:`~lit_search.runners.get_runner.GetRunner`.
    ssl_verify : bool
        Whether to verify SSL certificates during enrichment.
        Defaults to ``True``.

    Raises
    ------
    lit_search.exceptions.QueryValidationError
        If the query string fails validation.
    lit_search.exceptions.InvalidParameterError
        If *enrichment_databases* contains unknown database names.

    Examples
    --------
    >>> runner = SearchRunner(
    ...     query="ti[machine learning] AND abs[neural network]",
    ...     databases=["arxiv", "pubmed"],
    ...     max_papers_per_database=50,
    ... )
    >>> result = runner.run()
    >>> papers = result.papers
    """

    def __init__(
        self,
        query: str,
        databases: list[str] | None = None,
        max_papers_per_database: int | None = None,
        ieee_api_key: str | None = None,
        scopus_api_key: str | None = None,
        pubmed_api_key: str | None = None,
        openalex_api_key: str | None = None,
        email: str | None = None,
        semantic_scholar_api_key: str | None = None,
        wos_api_key: str | None = None,
        core_api_key: str | None = None,
        openaire_access_token: str | None = None,
        openaire_refresh_token: str | None = None,
        openaire_access_token_expires_in: float | None = None,
        zenodo_access_token: str | None = None,
        chemrxiv_email: str | None = None,
        ssrn_api_key: str | None = None,
        ncbi_tool: str | None = None,
        ncbi_email: str | None = None,
        openreview_username: str | None = None,
        openreview_password: str | None = None,
        openreview_venue_ids: Sequence[str] | None = None,
        openreview_cache_dir: Path | str | None = None,
        openreview_cache_ttl_seconds: float | None = None,
        arxiv_snapshot_path: str | None = None,
        arxiv_categories: frozenset[str] | None = None,
        zotero_base_url: str | None = None,
        zotero_library: str | None = None,
        num_workers: int = 1,
        since: dt.date | None = None,
        until: dt.date | None = None,
        enrichment_databases: list[str] | None = DEFAULT_ENRICHMENT_DATABASES,
        max_cited_by: int | None = 100,
        proxy: str | None = None,
        ssl_verify: bool = True,
    ) -> None:
        """Initialise search configuration without executing it.

        Parameters
        ----------
        query : str
            Raw query string.
        databases : list[str] | None
            Database identifiers to query.  ``None`` selects the usable
            default profile; ``["all"]`` selects every registered source.
        max_papers_per_database : int | None
            Maximum papers per database.  ``None`` means no limit.
        ieee_api_key : str | None
            IEEE Xplore API key.
        scopus_api_key : str | None
            Elsevier / Scopus API key.
        pubmed_api_key : str | None
            NCBI API key.  It is one key for all of E-utilities, so it is used
            for both the ``pubmed`` and the ``pmc`` connector; supplying it to
            only one of them would split a single NCBI quota across two
            rate-limit buckets.
        openalex_api_key : str | None
            OpenAlex API key.
        email : str | None
            Shared contact identity for Crossref-backed requests and the
            fallback E-utilities contact identity.
        semantic_scholar_api_key : str | None
            Semantic Scholar API key.
        wos_api_key : str | None
            Clarivate Web of Science API key.
        num_workers : int
            Number of parallel workers.  Defaults to ``1``.
        since : dt.date | None
            Lower-bound publication date filter.
        until : dt.date | None
            Upper-bound publication date filter.
        enrichment_databases : list[str] | None
            Databases for post-search enrichment.  Defaults to
            ``DEFAULT_ENRICHMENT_DATABASES`` (``["crossref", "web_scraping"]``).
            ``None`` uses the default.  Pass ``[]`` to disable enrichment entirely.
            Accepted values: ``"arxiv"``, ``"crossref"``, ``"ieee"``,
            ``"openalex"``, ``"pubmed"``, ``"scopus"``,
            ``"semantic_scholar"``, ``"web_scraping"``, ``"wos"``.
        max_cited_by : int | None
            Maximum citing-paper DOIs collected per paper when ``"openalex"``
            or ``"semantic_scholar"`` are in *enrichment_databases*.
            Defaults to ``100``.  ``None`` means no limit.
        proxy : str | None
            Optional HTTP/HTTPS proxy URL for enrichment requests.
        ssl_verify : bool
            Whether to verify SSL certificates during enrichment.

        Raises
        ------
        InvalidParameterError
            If *enrichment_databases* contains unknown database names.
        """
        _enrichment_dbs_set = set(enrichment_databases or [])
        _has_citation_enrich = bool(_enrichment_dbs_set & {"openalex", "semantic_scholar"})
        if _has_citation_enrich and (max_cited_by is None or max_cited_by > 100):
            logger.warning(
                "max_cited_by is %s. Fetching cited-by during enrichment may be very slow "
                "or produce large lists for highly-cited papers. "
                "Consider setting max_cited_by <= 100.",
                max_cited_by if max_cited_by is not None else "None (unlimited)",
            )

        self._results: list[Paper] = []
        self._metrics: dict[str, int | float] = {}
        self._search: SearchResult | None = None
        self._max_cited_by = max_cited_by

        super().__init__(
            since=since,
            until=until,
            ieee_api_key=ieee_api_key,
            scopus_api_key=scopus_api_key,
            pubmed_api_key=pubmed_api_key,
            openalex_api_key=openalex_api_key,
            email=email,
            semantic_scholar_api_key=semantic_scholar_api_key,
            wos_api_key=wos_api_key,
            proxy=proxy,
            ssl_verify=ssl_verify,
            enrichment_databases=enrichment_databases,
        )

        normalizer = QueryNormalizer()
        normalized_query = normalizer.normalize(query)

        self._query_string = normalized_query
        self._max_papers_per_database = max_papers_per_database
        self._num_workers = num_workers

        validator = QueryValidator()
        validator.validate(normalized_query)
        parser = QueryParser()
        self._query = parser.parse(normalized_query)

        # Propagate filter specifiers (e.g. ti, abs) through the query tree
        # so that group-level filters reach child term nodes.
        propagator = FilterPropagator()
        propagator.propagate(self._query)

        self._searchers, self._skipped_databases = self._build_searchers(
            databases=databases,
            ieee_api_key=ieee_api_key,
            scopus_api_key=scopus_api_key,
            pubmed_api_key=pubmed_api_key,
            openalex_api_key=openalex_api_key,
            email=email,
            semantic_scholar_api_key=semantic_scholar_api_key,
            wos_api_key=wos_api_key,
            core_api_key=core_api_key,
            openaire_access_token=openaire_access_token,
            openaire_refresh_token=openaire_refresh_token,
            openaire_access_token_expires_in=openaire_access_token_expires_in,
            zenodo_access_token=zenodo_access_token,
            chemrxiv_email=chemrxiv_email,
            ssrn_api_key=ssrn_api_key,
            ncbi_tool=ncbi_tool,
            ncbi_email=ncbi_email,
            openreview_username=openreview_username,
            openreview_password=openreview_password,
            openreview_venue_ids=openreview_venue_ids,
            openreview_cache_dir=openreview_cache_dir,
            openreview_cache_ttl_seconds=openreview_cache_ttl_seconds,
            arxiv_snapshot_path=arxiv_snapshot_path,
            arxiv_categories=arxiv_categories,
            zotero_base_url=zotero_base_url,
            zotero_library=zotero_library,
            proxy=proxy,
            ssl_verify=ssl_verify,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, verbose: bool = False, show_progress: bool = True) -> SearchResult:
        """Execute the configured pipeline and return the results.

        Can be called multiple times; each call resets previous results.

        Parameters
        ----------
        verbose : bool
            Enable verbose logging and print a summary after execution.
        show_progress : bool
            When ``True`` (default), display tqdm progress bars for each
            database while papers are being fetched.  Set to ``False``
            to suppress progress output (e.g. in non-interactive
            environments or to keep log output clean).

        Returns
        -------
        SearchResult
            Search result object containing papers and metadata.
        """
        _root_logger = logging.getLogger()
        _saved_log_level = _root_logger.level
        if verbose:
            configure_verbose_logging()
        logger.debug("=== SearchRunner Configuration ===")
        logger.debug("Databases: %s", [s.name for s in self._searchers])
        logger.debug("Num workers: %d", self._num_workers)
        logger.debug("Query: %s", self._query_string)
        logger.debug("Max papers per database: %s", self._max_papers_per_database or "none")
        logger.debug("==================================")

        start = perf_counter()
        self._results = []
        metrics: dict[str, int | float] = {
            "total_papers": 0,
            "runtime_in_seconds": 0.0,
        }
        for _skipped in self._skipped_databases:
            metrics[f"total_papers_from_{_skipped}"] = 0

        failed_databases: list[str] = []
        db_runtimes: dict[str, float] = {}
        coverage_by_source: dict[str, SourceCoverage] = {}
        try:
            failed_databases, db_runtimes, coverage_by_source = self._fetch_papers(
                metrics, show_progress=show_progress
            )
        finally:
            for searcher in self._searchers:
                searcher.close()

        before_dedupe = len(self._results)
        self._deduplicate_and_merge(metrics)
        logger.debug(
            "Dedupe: %d -> %d papers (%d merged)",
            before_dedupe,
            len(self._results),
            before_dedupe - len(self._results),
        )

        # Apply post-fetch date filters.  Connectors that only support
        # year-level date filtering may return papers outside the requested
        # range; _matches_filters enforces precise boundaries.
        if self._since is not None or self._until is not None:
            before_filter = len(self._results)
            self._results = [p for p in self._results if self._matches_filters(p)]
            logger.debug(
                "Post-fetch filter: %d -> %d papers (%d removed)",
                before_filter,
                len(self._results),
                before_filter - len(self._results),
            )

        # Enrich the filtered papers via per-paper get() lookups.
        # enrichment_databases=None  -> enrich with all available databases.
        # enrichment_databases=[]    -> skip enrichment entirely.
        if not (
            isinstance(self._enrichment_databases, list) and len(self._enrichment_databases) == 0
        ):
            super()._enrich_papers(
                self._results,
                verbose,
                show_progress=show_progress,
                num_workers=self._num_workers,
                max_cited_by=self._max_cited_by,
            )

        metrics["total_papers"] = len(self._results)
        metrics["runtime_in_seconds"] = perf_counter() - start
        self._metrics = metrics

        logger.debug("=== Results ===")
        logger.debug("Total papers: %d", metrics["total_papers"])
        logger.debug("Runtime: %.2f s", metrics["runtime_in_seconds"])
        for searcher in self._searchers:
            count = int(metrics.get(f"total_papers_from_{searcher.name}", 0))
            logger.debug("  %s: %d papers", searcher.name, count)

        self._search = SearchResult(
            query=self._query_string,
            max_papers_per_database=self._max_papers_per_database,
            processed_at=datetime.now(UTC),
            databases=list(self._requested_databases),
            papers=list(self._results),
            runtime_seconds=self._metrics.get("runtime_in_seconds"),
            runtime_seconds_per_database=db_runtimes or None,
            since=self._since,
            until=self._until,
            failed_databases=failed_databases or None,
            enrichment_databases=list(self._enrichment_databases),
            max_cited_by=self._max_cited_by,
            coverage=self._build_ledger(coverage_by_source),
        )
        # One line, at a level that matches the verdict. The renderings are
        # what guarantee a partial set is noticed; this is so that a caller
        # who never renders the object still gets told.
        log_coverage(self._search, logger)
        _root_logger.setLevel(_saved_log_level)
        return self._search

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_searchers(
        self,
        *,
        databases: list[str] | None,
        ieee_api_key: str | None,
        scopus_api_key: str | None,
        pubmed_api_key: str | None,
        openalex_api_key: str | None,
        email: str | None,
        semantic_scholar_api_key: str | None,
        wos_api_key: str | None,
        core_api_key: str | None,
        openaire_access_token: str | None,
        openaire_refresh_token: str | None,
        openaire_access_token_expires_in: float | None,
        zenodo_access_token: str | None,
        chemrxiv_email: str | None,
        ssrn_api_key: str | None,
        ncbi_tool: str | None,
        ncbi_email: str | None,
        openreview_username: str | None,
        openreview_password: str | None,
        openreview_venue_ids: Sequence[str] | None,
        openreview_cache_dir: Path | str | None,
        openreview_cache_ttl_seconds: float | None,
        arxiv_snapshot_path: str | None,
        arxiv_categories: frozenset[str] | None,
        zotero_base_url: str | None,
        zotero_library: str | None,
        proxy: str | None,
        ssl_verify: bool,
    ) -> tuple[list[SearchConnectorBase], list[str]]:
        """Instantiate the requested searchers.

        Parameters
        ----------
        databases : list[str] | None
            Requested database identifiers.
        ieee_api_key : str | None
            IEEE API key.
        scopus_api_key : str | None
            Scopus API key.
        pubmed_api_key : str | None
            PubMed API key.
        openalex_api_key : str | None
            OpenAlex API key.
        email : str | None
            Shared contact identity for Crossref-backed requests.
        semantic_scholar_api_key : str | None
            Semantic Scholar API key.
        wos_api_key : str | None
            Web of Science API key.

        Returns
        -------
        tuple[list[SearchConnectorBase], list[str]]
            A pair of (available searchers, names of skipped searchers).

        Raises
        ------
        InvalidParameterError
            When an unknown database identifier is provided.
        """
        # Per-database constructor credentials.  Databases with no entry
        # (e.g. Arxiv) are constructed with no arguments.  The class for
        # each Database is looked up in the central SEARCH_REGISTRY so
        # that this runner does not need to import every concrete connector.
        # PMC and PubMed are two Entrez databases behind one NCBI quota, which
        # is why both declare ``rate_limit_service = "ncbi"``.  That shares a
        # clock only when they also share a *credential*: the limiter's bucket
        # is (service, credential fingerprint), so handing the key to one and
        # not the other splits them into ('ncbi', <key>) at 0.11s and
        # ('ncbi', anon) at 0.34s, running concurrently from one IP against one
        # allowance.  NCBI's own policy says the unkeyed half then "will
        # receive an error message", and an intermittent 429 reads as thin
        # recall rather than as a misconfiguration.  So ``pubmed_api_key`` is
        # the NCBI key, and both connectors get it.
        _credentials: dict[Database, dict[str, object]] = {
            Database.ARXIV_BULK: {
                "snapshot_path": arxiv_snapshot_path,
                "categories": arxiv_categories,
            },
            Database.CHEMRXIV: {"email": chemrxiv_email or email},
            Database.CORE: {"api_key": core_api_key},
            Database.IEEE: {"api_key": ieee_api_key},
            Database.OPENAIRE: {
                "access_token": openaire_access_token,
                "refresh_token": openaire_refresh_token,
                "access_token_expires_in": openaire_access_token_expires_in,
            },
            Database.OPENALEX: {"api_key": openalex_api_key, "email": email},
            Database.OPENREVIEW: {
                "username": openreview_username,
                "password": openreview_password,
                "venue_ids": openreview_venue_ids,
                "cache_dir": openreview_cache_dir,
                "cache_ttl_seconds": openreview_cache_ttl_seconds,
            },
            Database.PMC: {
                "api_key": pubmed_api_key,
                "tool": ncbi_tool,
                "email": ncbi_email or email,
            },
            Database.PUBMED: {
                "api_key": pubmed_api_key,
                "tool": ncbi_tool,
                "email": ncbi_email or email,
            },
            Database.SCOPUS: {"api_key": scopus_api_key},
            Database.SEMANTIC_SCHOLAR: {"api_key": semantic_scholar_api_key},
            Database.SSRN: {"api_key": ssrn_api_key},
            Database.WOS: {"api_key": wos_api_key},
            Database.ZENODO: {"access_token": zenodo_access_token},
            Database.ZOTERO: {"base_url": zotero_base_url, "library": zotero_library},
        }

        # Omit ``None`` so connector defaults remain authoritative.  This is
        # important for OpenReview's default cache TTL and Zotero's loopback
        # URL; passing None would override a safe constructor default.
        _credentials = {
            source: {name: value for name, value in values.items() if value is not None}
            for source, values in _credentials.items()
        }

        valid_values = {db.value for db in SEARCH_REGISTRY}

        # Treat an explicit empty list as an error: the caller likely
        # intended to pass ``None`` (select all) instead of ``[]``.
        if databases is not None and len(databases) == 0:
            raise InvalidParameterError(
                "databases must not be an empty list. Pass None for the usable default "
                "profile or ['all'] for every registered database."
            )

        all_names = [db.value for db in SEARCH_REGISTRY]
        self._registered_databases = list(all_names)
        if databases is None:
            raw = list(DEFAULT_SEARCH_DATABASES)
            if openaire_access_token or openaire_refresh_token:
                raw.extend(CREDENTIALLED_DEFAULT_SEARCH_DATABASES)
                self._scope_profile = "default+credentialed"
            else:
                self._scope_profile = "default"
            self._policy_excluded_databases = [name for name in all_names if name not in raw]
            self._requested_databases = list(raw)
        else:
            normalised = [db.strip().lower() for db in databases]
            if normalised == ["all"]:
                raw = all_names
                self._scope_profile = "all"
            elif "all" in normalised:
                raise InvalidParameterError("'all' must be the only entry in databases")
            else:
                raw = normalised
                self._scope_profile = "explicit"
            self._policy_excluded_databases = []
            self._requested_databases = list(raw)
        unknown = [db for db in raw if db not in valid_values]
        if unknown:
            raise InvalidParameterError(
                f"Unknown database(s): {', '.join(unknown)}. "
                f"Accepted values: {', '.join(sorted(valid_values))}"
            )

        searchers: list = []
        skipped: list[str] = []
        for db in raw:
            try:
                searcher = SEARCH_REGISTRY[Database(db)](**_credentials.get(Database(db), {}))
                searcher.configure_transport(proxy=proxy, ssl_verify=ssl_verify)
                searchers.append(searcher)
            except MissingApiKeyError:
                skipped.append(Database(db).value)
                logger.warning(
                    "Skipping '%s': a required API key was not provided.",
                    Database(db).value,
                )
        return searchers, skipped

    def _fetch_papers(
        self,
        metrics: dict[str, int | float],
        *,
        show_progress: bool = True,
    ) -> tuple[list[str], dict[str, float], dict[str, SourceCoverage]]:
        """Fetch papers from all configured searchers.

        Updates *metrics* with per-database paper counts.

        Every searcher contributes exactly one
        :class:`~lit_search.core.coverage.SourceCoverage` record, whatever
        happened to it.  There is no path through this method that adds
        papers to :attr:`_results` without also recording why that count is
        what it is -- which is what stops a database that errored, or one
        that was never asked, from reading as a database that found nothing.

        Parameters
        ----------
        metrics : dict[str, int | float]
            Metrics dict to update in-place.
        show_progress : bool
            Display tqdm progress bars.

        Returns
        -------
        tuple[list[str], dict[str, float], dict[str, SourceCoverage]]
            A triple of (names of databases that failed during the search,
            per-database wall-clock runtime in seconds, per-database coverage
            keyed by connector name).  Databases skipped due to unsupported
            queries are **not** included in failures -- they are in the
            coverage map as ``QUERY_UNSUPPORTED``, which is where the whole
            picture lives.
        """

        num_searchers = len(self._searchers)
        num_workers = min(self._num_workers, num_searchers)

        # Create a persistent progress bar for every database upfront, each
        # pinned to a fixed terminal row via ``position``.  This guarantees
        # all bars are visible simultaneously in both serial and parallel
        # modes: workers update their own bar in-place rather than printing
        # new lines or clearing a shared temporary bar.
        db_bars = [
            make_progress_bar(
                desc=f"Searching - {searcher.name}",
                unit="paper",
                disable=not show_progress,
                leave=True,
                position=i,
            )
            for i, searcher in enumerate(self._searchers)
        ]
        bar_by_searcher = dict(zip(self._searchers, db_bars, strict=True))

        def _run_searcher(searcher: SearchConnectorBase) -> tuple[CoveredPapers, float]:
            """Run a single searcher and update its pre-assigned progress bar.

            Parameters
            ----------
            searcher : SearchConnectorBase
                Connector to execute.

            Returns
            -------
            tuple[CoveredPapers, float]
                Retrieved papers with their coverage, and wall-clock runtime
                in seconds.

            Raises
            ------
            CoverageContractError
                If the connector returned anything other than a
                :class:`~lit_search.core.coverage.CoveredPapers`.  The
                connector base class already refuses this, so reaching it here
                means the connector object was replaced wholesale -- by a test
                double, a monkeypatch, or a caller-supplied stub.  The runner
                re-checks rather than trusting that every object in
                ``self._searchers`` went through the base class.
            """
            pbar = bar_by_searcher[searcher]

            def _cb(current: int, total: int | None) -> None:
                pbar.total = total
                pbar.n = current
                pbar.refresh()

            db_start = perf_counter()
            covered = searcher.search(
                self._query,
                max_papers=self._max_papers_per_database,
                progress_callback=_cb,
                since=self._since,
                until=self._until,
            )
            elapsed = perf_counter() - db_start

            if not isinstance(covered, CoveredPapers):
                raise CoverageContractError(
                    f"{searcher.name}.search returned {type(covered).__name__}, not "
                    "CoveredPapers. A bare sequence carries no coverage, so an upstream "
                    "failure would have been recorded as a database that found nothing."
                )
            # The connector's own COMPLETE claim is re-checked against the cap
            # the *caller* chose, which is not a number the connector authored.
            # SearchConnectorBase.search already does this, but a connector
            # object replaced wholesale never runs it, and the evidence to
            # contradict the claim is sitting right here.
            covered = verify_coverage_claim(
                covered.with_source(str(searcher.name)),
                max_papers=self._max_papers_per_database,
                audit=None,
            )
            papers = list(covered.papers)

            # A connector may stop short of the total it originally reported
            # (e.g. an upstream API error, or silently running out of
            # retrievable pages) without that being the user's own
            # max_papers cap. When that happens, pbar.total still holds the
            # larger, stale estimate, so the bar would otherwise sit stuck
            # partway forever with no indication anything went wrong. Detect
            # that case and surface it via tqdm.write, which - unlike
            # logger.warning - is always visible, not just with --verbose.
            capped_by_user = (
                self._max_papers_per_database is not None
                and len(papers) >= self._max_papers_per_database
            )
            if pbar.total is not None and pbar.n < pbar.total and not capped_by_user:
                tqdm.write(
                    f"{searcher.name}: search stopped early after retrieving "
                    f"{pbar.n} of an estimated {pbar.total} papers. "
                    "Re-run with verbose=True for details."
                )
                pbar.total = pbar.n

            # When the connector exits early (e.g. first request returns an
            # error or zero results) total may still be None, leaving the bar
            # in indeterminate mode.  Force an exit from indeterminate state so
            # the bar looks visually finished rather than frozen.
            if pbar.total is None:
                pbar.total = pbar.n
            # A bar with total == 0 (whether explicitly reported by the API or
            # set above after an early exit) retains the indeterminate
            # '?paper/s' display in tqdm, which looks identical to 'still
            # running'.  Adding a 'done' postfix makes it obvious at a glance
            # that the search completed with no results.
            if pbar.total == 0:
                pbar.set_postfix_str("done")
            pbar.refresh()

            return covered, elapsed

        failed: list[str] = []
        db_runtimes: dict[str, float] = {}
        coverage_by_source: dict[str, SourceCoverage] = {}
        try:
            for searcher, result, error in execute_tasks(
                self._searchers,
                _run_searcher,
                num_workers=num_workers,
                timeout=None,
                use_progress=False,
            ):
                name = str(searcher.name)
                if error is not None or result is None:
                    metrics[f"total_papers_from_{searcher.name}"] = 0
                    coverage_by_source[name] = self._coverage_for_error(name, error)
                    if isinstance(error, UnsupportedQueryError):
                        logger.warning("Skipping '%s': %s", searcher.name, error)
                    else:
                        failed.append(searcher.name)
                        logger.warning("Error fetching from %s: %s", searcher.name, error)
                        logger.debug("Exception details for %s:", searcher.name, exc_info=error)
                else:
                    covered, elapsed = result
                    db_runtimes[searcher.name] = elapsed
                    metrics[f"total_papers_from_{searcher.name}"] = len(covered.papers)
                    coverage_by_source[name] = covered.coverage
                    self._results.extend(covered.papers)
        finally:
            # Ensure every bar exits indeterminate mode before closing,
            # covering the edge case where _run_searcher itself raised an
            # unhandled exception and never got to finalize its own bar.
            for pbar in db_bars:
                if pbar.total is None:
                    pbar.total = pbar.n
                if pbar.total == 0:
                    # refresh=False: avoid the same positional misalignment
                    # that removing pbar.refresh() above is meant to fix.
                    pbar.set_postfix_str("done", refresh=False)
                pbar.close()

        return failed, db_runtimes, coverage_by_source

    @staticmethod
    def _coverage_for_error(source: str, error: Exception | None) -> SourceCoverage:
        """Turn an exception that escaped a searcher into a coverage record.

        The one thing this must never do is return a record that reads as
        "this database was consulted and holds nothing".  Every branch below
        is a zero-coverage status carrying the reason.

        Parameters
        ----------
        source : str
            Connector name.
        error : Exception | None
            The exception raised, or ``None`` when the task produced no
            result and no exception -- which is itself unexplained, and is
            recorded as a failure rather than passed over.

        Returns
        -------
        SourceCoverage
            Zero-coverage record naming what went wrong.
        """
        if isinstance(error, UnsupportedQueryError):
            return SourceCoverage.query_unsupported(source, detail=str(error))
        if isinstance(error, MissingApiKeyError):
            return SourceCoverage.skipped_no_credential(source, credential=_credential_name(source))
        if error is None:
            return SourceCoverage.failed(
                source,
                error="the searcher returned no result and raised no exception",
                error_type="NoResult",
            )
        return SourceCoverage.failed(source, error=str(error), error_type=type(error).__name__)

    def _skipped_coverage(self) -> list[SourceCoverage]:
        """Return coverage records for databases that were never instantiated.

        This is the case ``R1`` names first: before the ledger existed, a
        database dropped for a missing key wrote ``total_papers_from_ieee = 0``
        and was indistinguishable from one that genuinely matched nothing.

        Returns
        -------
        list[SourceCoverage]
            One ``SKIPPED_NO_CREDENTIAL`` record per skipped database.
        """
        credential_skips = [
            SourceCoverage.skipped_no_credential(name, credential=_credential_name(name))
            for name in self._skipped_databases
        ]
        policy_skips = [
            SourceCoverage.skipped_by_policy(
                name,
                detail=(
                    "excluded by the default usable search profile; pass databases=['all'] "
                    "or name this source explicitly to consult it"
                ),
            )
            for name in self._policy_excluded_databases
        ]
        return credential_skips + policy_skips

    def _build_ledger(self, coverage_by_source: dict[str, SourceCoverage]) -> CoverageLedger:
        """Assemble the result set's ledger in a stable, readable order.

        Consulted databases come first in configuration order, then the ones
        that were never asked.  Completion order under parallel execution must
        not change how the ledger reads.

        Parameters
        ----------
        coverage_by_source : dict[str, SourceCoverage]
            Per-database coverage collected by :meth:`_fetch_papers`.

        Returns
        -------
        CoverageLedger
            Ledger covering every configured database, consulted or not.
        """
        entries: list[SourceCoverage] = []
        seen: set[str] = set()
        for searcher in self._searchers:
            name = str(searcher.name)
            entry = coverage_by_source.get(name)
            if entry is None:
                # A searcher that produced no coverage entry at all is a bug
                # in this runner, not a database that found nothing, and it is
                # recorded as such rather than omitted.
                entry = SourceCoverage.failed(
                    name,
                    error="no coverage was recorded for this database by the search runner",
                    error_type="MissingCoverage",
                )
            entries.append(entry)
            seen.add(name)
        entries.extend(entry for name, entry in coverage_by_source.items() if name not in seen)
        entries.extend(self._skipped_coverage())
        return CoverageLedger(
            entries=tuple(entries),
            scope=CoverageScope(
                profile=self._scope_profile,
                selected_sources=tuple(self._requested_databases),
                registered_sources=tuple(self._registered_databases),
            ),
        )

    def _deduplicate_and_merge(self, metrics: dict[str, int | float]) -> None:
        """Merge only records sharing an explicit, non-conflicting identity.

        DOI, arXiv, Semantic Scholar and OpenAlex identifiers may establish
        exact record sameness. Title, year, authors and semantic similarity
        never do: those fields are evidence for a later work-equivalence stage,
        and using them here would destroy version boundaries before that stage
        can record them. Identifier-free records therefore remain separate.

        Parameters
        ----------
        metrics : dict[str, int | float]
            Metrics dict (currently unused; reserved for future statistics).

        Returns
        -------
        None
        """
        results: list[Paper] = []
        by_key: dict[str, list[Paper]] = {}
        for paper in self._results:
            keys = record_identity_keys(paper)
            candidates = {
                id(candidate): candidate for key in keys for candidate in by_key.get(key, ())
            }
            representative = next(
                (
                    candidate
                    for candidate in candidates.values()
                    if same_record_identity(candidate, paper)
                ),
                None,
            )
            if representative is None:
                representative = paper
                results.append(representative)
            else:
                representative.merge(paper)

            for key in record_identity_keys(representative):
                bucket = by_key.setdefault(key, [])
                if all(existing is not representative for existing in bucket):
                    bucket.append(representative)

        self._results = results
