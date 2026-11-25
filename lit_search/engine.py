"""Engine: centralised entry point for all lit-search operations.

An :class:`Engine` instance holds shared configuration (API keys, proxy
settings, timeouts) so that callers configure once and invoke multiple
operations without repeating those details.

Example
-------
>>> from lit_search import Engine
>>> engine = Engine(
...     ieee_api_key="...",
...     scopus_api_key="...",
...     proxy="http://proxy:8080",
... )
>>> result = engine.search("[machine learning]", databases=["arxiv", "ieee"])
>>> engine.download(result.papers, "./pdfs")
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from lit_search import credentials
from lit_search.core.lookup_result import LookupResult
from lit_search.core.paper import Paper
from lit_search.core.query_family import QueryFamilyRevision
from lit_search.core.query_family_result import QueryFamilyResult
from lit_search.core.search_result import SearchResult
from lit_search.core.similar_result import SimilarResult
from lit_search.core.snowball_result import SnowballResult
from lit_search.query.family_compiler import QueryFamilyPlan
from lit_search.runners.discovery_runner import DEFAULT_ENRICHMENT_DATABASES
from lit_search.runners.download_runner import DownloadRunner
from lit_search.runners.get_runner import GetRunner
from lit_search.runners.query_family_runner import EvidenceVerifier, QueryFamilyRunner
from lit_search.runners.search_runner import SearchRunner
from lit_search.runners.similar_runner import SimilarRunner
from lit_search.runners.snowball_runner import _UNSET as _SNOWBALL_UNSET
from lit_search.runners.snowball_runner import SnowballRunner
from lit_search.store.run_state import RunStateStore
from lit_search.store.snapshot_cache import SnapshotCache


class Engine:
    """Centralised facade for lit-search operations.

    Holds shared configuration (API keys, proxy, and SSL settings) that
    would otherwise need to be repeated in every call.  Per-call parameters
    such as *num_workers*, *timeout*, and *verbose* are passed directly to
    each method.

    All parameters fall back to the corresponding setting in
    :mod:`lit_search.credentials` (environment, then credentials file) when not
    supplied explicitly.

    Parameters
    ----------
    ieee_api_key : str | None
        IEEE Xplore API key.  Required to query the ``"ieee"`` database.
        Falls back to ``IEEE_API_KEY``.
    scopus_api_key : str | None
        Elsevier / Scopus API key.  Required to query ``"scopus"``.
        Falls back to ``SCOPUS_API_KEY``.
    pubmed_api_key : str | None
        NCBI PubMed API key.  Optional: increases the rate limit.
        Falls back to ``NCBI_API_KEY``.
    openalex_api_key : str | None
        OpenAlex API key.  Optional.
        Falls back to ``OPENALEX_API_KEY``.
    email : str | None
        Contact email sent to Crossref through its official ``mailto`` request
        parameter and used as the fallback NCBI/ChemRxiv contact identity.
        Falls back to ``LIT_SEARCH_EMAIL``.
    semantic_scholar_api_key : str | None
        Semantic Scholar API key.  Optional: increases the rate limit.
        Falls back to ``SEMANTIC_SCHOLAR_API_KEY``.
    wos_api_key : str | None
        Clarivate Web of Science API key.  Required to query ``"wos"``.
        Falls back to ``WOS_API_KEY``.
    proxy : str | None
        Proxy URL (e.g. ``"http://proxy:8080"``).
        Falls back to ``LIT_SEARCH_PROXY``.
    ssl_verify : bool
        Whether to verify SSL certificates.  Set to ``False`` when using
        institutional proxies that perform SSL inspection.
        Defaults to ``True``.  Falls back to ``LIT_SEARCH_SSL_VERIFY``
        (accepted values: ``"0"``, ``"false"``, ``"no"`` -> ``False``).

    Examples
    --------
    >>> from lit_search import Engine
    >>> engine = Engine(ieee_api_key="my-key", proxy="http://proxy:8080")
    >>> result = engine.search("[deep learning]", databases=["arxiv", "ieee"])
    >>> engine.download(result.papers, "./pdfs", num_workers=4)
    """

    def __init__(
        self,
        *,
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
        openreview_cache_dir: str | None = None,
        openreview_cache_ttl_seconds: float | None = None,
        arxiv_snapshot_path: str | None = None,
        arxiv_categories: frozenset[str] | None = None,
        zotero_base_url: str | None = None,
        zotero_library: str | None = None,
        proxy: str | None = None,
        ssl_verify: bool = True,
    ) -> None:
        """Initialise engine with shared configuration.

        Values not supplied explicitly are resolved from environment
        variables (see class docstring for the mapping).
        """
        self._ieee_api_key = ieee_api_key or credentials.get("IEEE_API_KEY") or None
        self._scopus_api_key = scopus_api_key or credentials.get("SCOPUS_API_KEY") or None
        self._pubmed_api_key = pubmed_api_key or credentials.get("NCBI_API_KEY") or None
        self._openalex_api_key = openalex_api_key or credentials.get("OPENALEX_API_KEY") or None
        self._email = email or credentials.get("LIT_SEARCH_EMAIL") or None
        self._semantic_scholar_api_key = (
            semantic_scholar_api_key or credentials.get("SEMANTIC_SCHOLAR_API_KEY") or None
        )
        self._wos_api_key = wos_api_key or credentials.get("WOS_API_KEY") or None
        self._core_api_key = core_api_key or credentials.get("CORE_API_KEY") or None
        self._openaire_access_token = (
            openaire_access_token or credentials.get("OPENAIRE_ACCESS_TOKEN") or None
        )
        self._openaire_refresh_token = (
            openaire_refresh_token or credentials.get("OPENAIRE_REFRESH_TOKEN") or None
        )
        self._openaire_access_token_expires_in = openaire_access_token_expires_in
        self._zenodo_access_token = (
            zenodo_access_token or credentials.get("ZENODO_ACCESS_TOKEN") or None
        )
        self._chemrxiv_email = (
            chemrxiv_email or credentials.get("LIT_SEARCH_CHEMRXIV_EMAIL") or self._email
        )
        self._ssrn_api_key = ssrn_api_key or credentials.get("SSRN_API_KEY") or None
        self._ncbi_tool = ncbi_tool or credentials.get("LIT_SEARCH_NCBI_TOOL") or None
        self._ncbi_email = ncbi_email or credentials.get("LIT_SEARCH_NCBI_EMAIL") or self._email
        self._openreview_username = (
            openreview_username or credentials.get("OPENREVIEW_USERNAME") or None
        )
        self._openreview_password = (
            openreview_password or credentials.get("OPENREVIEW_PASSWORD") or None
        )
        self._openreview_venue_ids = (
            tuple(openreview_venue_ids) if openreview_venue_ids is not None else None
        )
        self._openreview_cache_dir = (
            openreview_cache_dir or credentials.get("LIT_SEARCH_OPENREVIEW_CACHE_DIR") or None
        )
        self._openreview_cache_ttl_seconds = openreview_cache_ttl_seconds
        self._arxiv_snapshot_path = (
            arxiv_snapshot_path or credentials.get("LIT_SEARCH_ARXIV_SNAPSHOT") or None
        )
        self._arxiv_categories = arxiv_categories
        self._zotero_base_url = (
            zotero_base_url or credentials.get("LIT_SEARCH_ZOTERO_BASE_URL") or None
        )
        self._zotero_library = (
            zotero_library or credentials.get("LIT_SEARCH_ZOTERO_LIBRARY") or None
        )
        self._proxy = proxy or credentials.get("LIT_SEARCH_PROXY") or None

        # ssl_verify: only fall back to env when caller omits the argument
        # (i.e. uses the default True).  Env values "0", "false", "no"
        # are treated as False.
        if ssl_verify is True and (env_ssl := credentials.get("LIT_SEARCH_SSL_VERIFY")):
            self._ssl_verify = env_ssl.lower() not in (
                "0",
                "false",
                "no",
            )
        else:
            self._ssl_verify = ssl_verify

        if not self._ssl_verify:
            logging.getLogger(__name__).warning(
                "SSL certificate verification is disabled. "
                "Connections may be vulnerable to man-in-the-middle attacks."
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        databases: list[str] | None = None,
        max_papers_per_database: int | None = None,
        since: dt.date | None = None,
        until: dt.date | None = None,
        num_workers: int = 1,
        verbose: bool = False,
        show_progress: bool = True,
        enrichment_databases: list[str] | None = DEFAULT_ENRICHMENT_DATABASES,
        max_cited_by: int | None = 100,
    ) -> SearchResult:
        """Search for academic papers across multiple databases.

        Queries one or more academic databases and returns a
        :class:`~lit_search.core.search_result.SearchResult` object with the
        collected papers already deduplicated and merged.

        Query syntax
        ~~~~~~~~~~~~
        Wrap each search term in square brackets and combine them with
        ``AND``, ``OR``, or ``AND NOT`` operators.  Optionally prefix a term or
        group with a **filter code** to restrict where it is matched:

        * ``ti``: title
        * ``abs``: abstract
        * ``key``: keywords
        * ``au``: author
        * ``src``: source (journal / conference)
        * ``aff``: affiliation
        * ``tiabs``: title + abstract (default when no filter is given)
        * ``tiabskey``: title + abstract + keywords

        Example queries::

            "[machine learning]"                              # simple
            "ti[deep learning] AND abs[transformer]"           # with filters
            "[covid-19] AND ([treatment] OR [vaccine])"        # grouping

        Supported databases
        ~~~~~~~~~~~~~~~~~~~
        ``"arxiv"``, ``"ieee"`` (requires API key), ``"openalex"``,
        ``"pubmed"``, ``"scopus"`` (requires API key),
        ``"semantic_scholar"``, ``"wos"`` (requires API key).

        When *databases* is ``None`` the usable default profile is queried.
        Pass ``["all"]`` for an explicit registry-wide audit.

        Parameters
        ----------
        query : str
            Query string following the syntax described above.
        databases : list[str] | None
            Database identifiers to query.  ``None`` selects the usable
            default profile; ``["all"]`` selects every registered source.
        max_papers_per_database : int | None
            Cap on the number of papers retrieved from each database.
            ``None`` means no limit.
        since : datetime.date | None
            Only return papers published on or after this date.  Passed to
            each database connector's API when supported.  ``None`` means
            no lower-bound filter.
        until : datetime.date | None
            Only return papers published on or before this date.  Passed to
            each database connector's API when supported.  ``None`` means
            no upper-bound filter.
        num_workers : int
            Number of parallel workers used to query databases concurrently.
            Defaults to ``1`` (sequential).
        verbose : bool
            When ``True``, emit detailed log messages at DEBUG level.
            Defaults to ``False``.
        show_progress : bool
            When ``True`` (default), display tqdm progress bars while
            papers are being fetched.  Set to ``False`` to suppress
            progress output (e.g. in non-interactive environments or to
            keep log output clean).
        enrichment_databases : list[str] | None
            Databases used to enrich papers after search and filtering.
            Defaults to ``["crossref", "web_scraping"]``, which cover the
            majority of metadata gaps without consuming quota from
            rate-limited databases.  Pass an explicit list to enable
            additional (or different) sources.  Accepted values:
            ``"arxiv"``, ``"crossref"``, ``"ieee"``, ``"openalex"``,
            ``"pubmed"``, ``"scopus"``, ``"semantic_scholar"``,
            ``"web_scraping"``, ``"wos"``.
            Pass ``[]`` to disable enrichment entirely. ``None`` uses the
            default.
        max_cited_by : int | None
            Maximum number of citing-paper DOIs collected per paper when
            ``"openalex"`` or ``"semantic_scholar"`` are in
            *enrichment_databases*.  Defaults to ``100``.  ``None`` means no
            limit: use with caution as highly-cited papers may have thousands
            of citations.  A warning is emitted when this value is ``None`` or
            greater than ``100``.

        Returns
        -------
        SearchResult
            A :class:`~lit_search.core.search_result.SearchResult` object
            whose ``papers`` attribute contains the collected
            :class:`~lit_search.core.paper.Paper` instances.  Save via
            :func:`lit_search.save_to_json` or
            :func:`lit_search.save_to_bibtex`.

        Raises
        ------
        lit_search.exceptions.QueryValidationError
            If *query* has syntax errors (unbalanced brackets, invalid filter
            codes, etc.).
        lit_search.exceptions.InvalidParameterError
            If an unknown database name is passed in *databases*.
        See Also
        --------
        lit_search.runners.search_runner.SearchRunner :
            Lower-level class for when you need access to per-run metrics or
            want to separate configuration from execution.

        Examples
        --------
        Basic search across all available databases:

        >>> from lit_search import Engine
        >>> engine = Engine()
        >>> result = engine.search("[machine learning]")
        >>> print(f"{len(result.papers)} papers found")
        127 papers found

        Targeted search with filters and database selection:

        >>> engine = Engine(ieee_api_key="my-key")
        >>> result = engine.search(
        ...     "ti[transformer] AND abs[attention mechanism]",
        ...     databases=["arxiv", "ieee"],
        ...     max_papers_per_database=50,
        ... )

        Save results to a file:

        >>> import lit_search
        >>> lit_search.save_to_json(result, "my_search.json")
        >>> lit_search.save_to_bibtex(result.papers, "my_search.bib")
        """
        runner = SearchRunner(
            query=query,
            databases=databases,
            max_papers_per_database=max_papers_per_database,
            ieee_api_key=self._ieee_api_key,
            scopus_api_key=self._scopus_api_key,
            pubmed_api_key=self._pubmed_api_key,
            openalex_api_key=self._openalex_api_key,
            email=self._email,
            semantic_scholar_api_key=self._semantic_scholar_api_key,
            wos_api_key=self._wos_api_key,
            core_api_key=self._core_api_key,
            openaire_access_token=self._openaire_access_token,
            openaire_refresh_token=self._openaire_refresh_token,
            openaire_access_token_expires_in=self._openaire_access_token_expires_in,
            zenodo_access_token=self._zenodo_access_token,
            chemrxiv_email=self._chemrxiv_email,
            ssrn_api_key=self._ssrn_api_key,
            ncbi_tool=self._ncbi_tool,
            ncbi_email=self._ncbi_email,
            openreview_username=self._openreview_username,
            openreview_password=self._openreview_password,
            openreview_venue_ids=self._openreview_venue_ids,
            openreview_cache_dir=self._openreview_cache_dir,
            openreview_cache_ttl_seconds=self._openreview_cache_ttl_seconds,
            arxiv_snapshot_path=self._arxiv_snapshot_path,
            arxiv_categories=self._arxiv_categories,
            zotero_base_url=self._zotero_base_url,
            zotero_library=self._zotero_library,
            num_workers=num_workers,
            since=since,
            until=until,
            enrichment_databases=enrichment_databases,
            max_cited_by=max_cited_by,
            proxy=self._proxy,
            ssl_verify=self._ssl_verify,
        )
        return runner.run(verbose=verbose, show_progress=show_progress)

    def search_family(
        self,
        revision: QueryFamilyRevision,
        *,
        sources: Sequence[str],
        plan: QueryFamilyPlan | None = None,
        evidence_verifier: EvidenceVerifier | None = None,
        max_papers_per_database: int | None = None,
        since: dt.date | None = None,
        until: dt.date | None = None,
        num_workers: int = 1,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> QueryFamilyResult:
        """Execute an explicit, provenance-bearing query-family revision.

        Unlike :meth:`search`, this method does not accept a free-form query
        or generate terminology.  Every executable term and source-specific
        translation must already be present in *revision*.  *sources* is
        required so that the family plan has an explicit retrieval boundary;
        no registry-wide or mutable default policy is selected implicitly.

        Each source-by-variant cell receives the same credentials and source
        configuration as an ordinary engine search.  Cell-level enrichment is
        intentionally absent: :class:`QueryFamilyRunner` disables it and a
        later family-level operation may enrich the union once without
        obscuring which retrieval cell found each record.

        Parameters
        ----------
        revision : QueryFamilyRevision
            Immutable query-family revision to validate, compile, and run.
        sources : Sequence[str]
            Exact database identifiers in the declared retrieval scope.
        plan : QueryFamilyPlan | None
            Optional precompiled plan.  The runner verifies it against a
            fresh compilation of *revision* and *sources* before any request.
        evidence_verifier : EvidenceVerifier | None
            Optional provenance verifier called before planning or requests.
        max_papers_per_database : int | None
            Per-cell source result cap. ``None`` means no limit.
        since, until : datetime.date | None
            Optional inclusive publication-date bounds for every cell.
        num_workers : int
            Worker count passed to each cell's single-source search runner.
        verbose, show_progress : bool
            Execution logging and progress-display controls.

        Returns
        -------
        QueryFamilyResult
            Auditable per-cell outcomes plus their explicit-identity union.
        """
        search_runner_kwargs = {
            "max_papers_per_database": max_papers_per_database,
            "ieee_api_key": self._ieee_api_key,
            "scopus_api_key": self._scopus_api_key,
            "pubmed_api_key": self._pubmed_api_key,
            "openalex_api_key": self._openalex_api_key,
            "email": self._email,
            "semantic_scholar_api_key": self._semantic_scholar_api_key,
            "wos_api_key": self._wos_api_key,
            "core_api_key": self._core_api_key,
            "openaire_access_token": self._openaire_access_token,
            "openaire_refresh_token": self._openaire_refresh_token,
            "openaire_access_token_expires_in": self._openaire_access_token_expires_in,
            "zenodo_access_token": self._zenodo_access_token,
            "chemrxiv_email": self._chemrxiv_email,
            "ssrn_api_key": self._ssrn_api_key,
            "ncbi_tool": self._ncbi_tool,
            "ncbi_email": self._ncbi_email,
            "openreview_username": self._openreview_username,
            "openreview_password": self._openreview_password,
            "openreview_venue_ids": self._openreview_venue_ids,
            "openreview_cache_dir": self._openreview_cache_dir,
            "openreview_cache_ttl_seconds": self._openreview_cache_ttl_seconds,
            "arxiv_snapshot_path": self._arxiv_snapshot_path,
            "arxiv_categories": self._arxiv_categories,
            "zotero_base_url": self._zotero_base_url,
            "zotero_library": self._zotero_library,
            "num_workers": num_workers,
            "since": since,
            "until": until,
            "proxy": self._proxy,
            "ssl_verify": self._ssl_verify,
        }
        runner = QueryFamilyRunner(
            revision,
            sources,
            plan=plan,
            search_runner_kwargs=search_runner_kwargs,
            evidence_verifier=evidence_verifier,
        )
        return runner.run(verbose=verbose, show_progress=show_progress)

    def download(
        self,
        papers: list[Paper],
        output_directory: str,
        *,
        num_workers: int = 1,
        timeout: float | None = 30.0,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> dict[str, int | float]:
        """Download PDFs for a list of papers.

        For each paper, all known URLs are tried and HTML landing pages are
        followed to resolve the actual PDF link.  Downloaded files are saved
        to *output_directory* with a ``year-title.pdf`` naming scheme.  When
        a download fails the paper is logged to ``download_log.txt`` inside
        *output_directory*.  Successful downloads are also logged.

        Parameters
        ----------
        papers : list[Paper]
            Papers whose PDFs should be downloaded, typically obtained from
            ``engine.search(...).papers``.
        output_directory : str
            Directory where PDF files and the error log will be written.
            Created automatically if it does not exist.
        num_workers : int
            Number of parallel download workers.  Defaults to ``1``
            (sequential).  Increase to speed up bulk downloads.
        timeout : float | None
            Per-request HTTP timeout in seconds.  ``None`` disables the
            timeout.  Defaults to ``30.0``.
        verbose : bool
            When ``True``, emit detailed log messages at DEBUG level.
            Defaults to ``False``.
        show_progress : bool
            When ``True`` (default), display a tqdm progress bar while
            papers are being downloaded.  Set to ``False`` to suppress
            progress output.

        Returns
        -------
        dict[str, int | float]
            Metrics dictionary with at least the following keys:

            * ``total_papers``: number of papers attempted.
            * ``downloaded_papers``: number of successfully downloaded PDFs.
            * ``runtime_in_seconds``: wall-clock time of the download
              process.

        See Also
        --------
        lit_search.runners.download_runner.DownloadRunner :
            Lower-level class for when you need finer control over the
            download pipeline.

        Examples
        --------
        >>> from lit_search import Engine
        >>> engine = Engine(proxy="http://proxy:8080")
        >>> result = engine.search("[deep learning]", databases=["arxiv"])
        >>> metrics = engine.download(result.papers, "./pdfs")
        >>> print(f"{metrics['downloaded_papers']}/{metrics['total_papers']} downloaded")
        8/10 downloaded
        """
        runner = DownloadRunner(
            papers=papers,
            output_directory=output_directory,
            num_workers=num_workers,
            timeout=timeout,
            proxy=self._proxy,
            ssl_verify=self._ssl_verify,
        )
        return runner.run(verbose=verbose, show_progress=show_progress)

    def get(
        self,
        identifier: str,
        *,
        databases: list[str] | None = None,
        max_cited_by: int | None = 100,
        timeout: float | None = 10.0,
        verbose: bool = False,
    ) -> LookupResult:
        """Fetch a single paper by its DOI or landing-page URL.

        Accepts three forms of identifier:

        * **Bare DOI** (e.g. ``"10.1038/nature12373"``): queries each
          database via its API and merges the results into a single
          :class:`~lit_search.core.paper.Paper`.
        * **DOI URL** (e.g. ``"https://doi.org/10.1038/nature12373"``):
          the ``doi.org`` prefix is stripped and the DOI is resolved
          through the same multi-database path.
        * **Landing-page URL** (e.g. ``"https://arxiv.org/abs/1706.03762"``
          or ``"https://www.nature.com/articles/s41586-021-03819-2"``):
          for URLs belonging to a supported database (arXiv, PubMed, IEEE,
          OpenAlex, Semantic Scholar) the paper is fetched directly via
          that database's API.  For all other URLs the page is downloaded
          and metadata is extracted from the HTML.

        When OpenAlex or Semantic Scholar are among the queried databases,
        ``paper.cited_by`` is populated with the DOIs of papers that cite
        this paper.  OpenAlex is used as the primary source (results are
        sorted by citation count so the most-impactful papers are kept when
        the list is truncated).  Semantic Scholar is the fallback.

        Parameters
        ----------
        identifier : str
            DOI, DOI URL, or paper landing-page URL.
        databases : list[str] | None
            Sources to consult when looking up the paper.  When ``None``
            all available sources are used.  Pass a list to enable only
            the specified ones.  Accepted values: ``"arxiv"``,
            ``"crossref"``, ``"ieee"``, ``"openalex"``, ``"pubmed"``,
            ``"scopus"``, ``"semantic_scholar"``, ``"web_scraping"``,
            ``"wos"``.
        max_cited_by : int | None
            Maximum number of citing-paper DOIs to collect when populating
            ``paper.cited_by``.  Defaults to ``100``.  ``None`` means no
            limit: use with caution as highly-cited papers may have
            thousands of citations.
        timeout : float | None
            HTTP request timeout in seconds.  ``None`` disables the
            timeout.  Defaults to ``10.0``.
        verbose : bool
            When ``True``, emit detailed log messages at DEBUG level.
            Defaults to ``False``.

        Returns
        -------
        LookupResult
            The paper, if found, plus per-source coverage.  Inspect
            :attr:`LookupResult.paper`; an empty result is only evidence of
            absence when its ledger is complete.

        Raises
        ------
        InvalidParameterError
            If *identifier* is a bare DOI that is empty or blank after
            stripping whitespace and URL prefixes.
        InvalidParameterError
            If *databases* is an empty list or contains unknown database
            names.

        See Also
        --------
        lit_search.runners.get_runner.GetRunner :
            Lower-level class that combines URL scraping and DOI-based
            lookups into a single unified pipeline.

        Examples
        --------
        Bare DOI:

        >>> from lit_search import Engine
        >>> engine = Engine()
        >>> paper = engine.get("10.1038/nature12373").paper

        DOI URL:

        >>> paper = engine.get("https://doi.org/10.1038/nature12373").paper

        Landing-page URL (delegates to the arXiv API: no scraping):

        >>> paper = engine.get("https://arxiv.org/abs/1706.03762").paper

        Publisher landing-page URL (HTML scraping fallback):

        >>> paper = engine.get("https://www.nature.com/articles/s41586-021-03819-2").paper
        """
        runner = GetRunner(
            identifier=identifier,
            email=self._email,
            databases=databases,
            ieee_api_key=self._ieee_api_key,
            scopus_api_key=self._scopus_api_key,
            pubmed_api_key=self._pubmed_api_key,
            openalex_api_key=self._openalex_api_key,
            semantic_scholar_api_key=self._semantic_scholar_api_key,
            wos_api_key=self._wos_api_key,
            timeout=timeout,
            proxy=self._proxy,
            ssl_verify=self._ssl_verify,
        )
        return runner.run(verbose=verbose, max_cited_by=max_cited_by)

    def snowball(
        self,
        paper: Paper | Sequence[Paper],
        *,
        max_depth: int = 2,
        direction: Literal["both", "backward", "forward"] = "both",
        max_papers_per_level: int | None = None,
        max_expansion_per_level: int | None = None,
        max_cited_by: int | None = 100,
        databases: list[str] | None = _SNOWBALL_UNSET,  # type: ignore[assignment]
        enrichment_databases: list[str] | None = _SNOWBALL_UNSET,  # type: ignore[assignment]
        since: dt.date | None = None,
        until: dt.date | None = None,
        num_workers: int = 1,
        managed_store_root: Path | str | None = None,
        managed_run_id: str | None = None,
        resume_managed_run: bool = False,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> SnowballResult:
        """Expand one paper or a seed set via iterative citation snowballing.

        Starting from one paper or every member of a supplied sequence, iteratively fetches
        references (backward) and/or citing papers (forward).  The seed paper
        is enriched with the full combined set of *databases* and
        *enrichment_databases*.  Papers discovered during BFS are fetched with
        *databases* only; after all BFS levels and filters are applied, the
        surviving non-seed papers are re-enriched with the *enrichment_databases*
        not already used during discovery.

        Parameters
        ----------
        paper : Paper | Sequence[Paper]
            One seed or a non-empty seed set.  Every seed needs a DOI or
            source-native identifier.  Passing ``engine.search(...).papers``
            is the default recall-oriented protocol.
        max_depth : int
            Maximum number of BFS levels.  ``1`` retrieves only the immediate
            neighbours.  ``2`` (default) also expands papers
            found at level 1, and so on.
        direction : Literal["both", "backward", "forward"]
            ``"backward"`` follows
            :attr:`~lit_search.core.paper.Paper.references` (papers
            cited *by* the current frontier), ``"forward"`` follows
            :attr:`~lit_search.core.paper.Paper.cited_by` (papers that
            *cite* the current frontier), ``"both"`` expands both.
        max_papers_per_level : int | None
            When set, only the *top N* most-cited papers discovered at each
            level are kept in the final result.  The seed paper is never
            filtered.  ``None`` (default) means all discovered papers
            are included.
        max_expansion_per_level : int | None
            When set, only the *top N* most-cited papers from each level are
            used as seeds for the next BFS round.  Papers already added to the
            result are unaffected.  ``None`` (default) means the full frontier
            is expanded.
        max_cited_by : int | None
            Maximum number of citing-paper DOIs to collect per paper during
            seed and frontier enrichment.  Defaults to ``100``.  ``None``
            means no limit: use with caution as highly-cited papers may have
            thousands of citations.  A warning is emitted when this value is
            ``None`` or greater than ``100``.
        databases : list[str] | None
            Databases used for BFS discovery.  Only ``"crossref"``,
            ``"openalex"``, and ``"semantic_scholar"`` are accepted.
            Defaults to ``["semantic_scholar"]`` in every direction.  Pass
            ``None`` to use all three databases. Forward and both directions require at
            least one of ``"openalex"`` or ``"semantic_scholar"``.
        enrichment_databases : list[str] | None
            Databases used to enrich non-seed papers after BFS completes.
            Defaults to ``["crossref", "web_scraping"]``.  Pass ``None`` to
            use the same default.  Pass ``[]`` to disable enrichment
            entirely.  Accepted values: ``"arxiv"``, ``"crossref"``,
            ``"ieee"``, ``"openalex"``, ``"pubmed"``, ``"scopus"``,
            ``"semantic_scholar"``, ``"web_scraping"``, ``"wos"``.
        since : datetime.date | None
            Only include discovered papers published on or after this
            date.  The seed paper is never filtered.  ``None`` (default)
            disables the lower-bound filter.
        until : datetime.date | None
            Only include discovered papers published on or before this
            date.  The seed paper is never filtered.  ``None`` (default)
            disables the upper-bound filter.
        num_workers : int
            Number of parallel
            :class:`~lit_search.runners.get_runner.GetRunner` calls per
            level.  Defaults to ``1`` (sequential).
        managed_store_root : pathlib.Path | str | None
            Shared root for per-run checkpoints and source snapshots. ``None``
            keeps the ordinary unmanaged execution path.
        managed_run_id : str | None
            Explicit run ID to create or resume. A caller that creates an ID
            before execution can persist or display it before network work.
        resume_managed_run : bool
            Open ``managed_run_id`` instead of creating it. Resume refuses any
            semantic-configuration or credential-fingerprint mismatch.
        verbose : bool
            When ``True``, emit detailed log messages at DEBUG level.
        show_progress : bool
            When ``True`` (default), display tqdm progress bars while
            papers are being fetched.  Set to ``False`` to suppress
            progress output.

        Returns
        -------
        SnowballResult
            Container with discoveries in ``papers``, enriched seeds in
            ``seed_papers``, their expanded union in ``all_papers``, and an
            edge-fetch coverage ledger.
            Citation relationships are encoded on each paper via
            :attr:`~lit_search.core.paper.Paper.references` and
            :attr:`~lit_search.core.paper.Paper.cited_by`.  Save via
            ``lit_search.save_to_json(result, path)``.

        Raises
        ------
        InvalidParameterError
            If *paper* has no DOI, or any other parameter is invalid (see
            :class:`~lit_search.runners.snowball_runner.SnowballRunner`).

        See Also
        --------
        lit_search.runners.snowball_runner.SnowballRunner :
            Lower-level class for when you need fine-grained control
            over runner configuration.

        Examples
        --------
        Snowball from a single paper found by DOI:

        >>> from lit_search import Engine
        >>> engine = Engine()
        >>> seed = engine.get("10.1038/nature12373")
        >>> result = engine.snowball(seed, max_depth=1)
        >>> print(f"{len(result.papers)} papers found")
        42 papers found

        Save the result to JSON:

        >>> import lit_search
        >>> lit_search.save_to_json(result, "snowball_result.json")

        Snowball with only backward direction and two BFS levels:

        >>> result = engine.snowball(
        ...     seed,
        ...     max_depth=2,
        ...     direction="backward",
        ... )
        """

        def make_runner(
            *,
            run_state: RunStateStore | None = None,
            snapshot_cache: SnapshotCache | None = None,
        ) -> SnowballRunner:
            return SnowballRunner(
                seed_paper=paper,
                max_depth=max_depth,
                direction=direction,
                max_papers_per_level=max_papers_per_level,
                max_expansion_per_level=max_expansion_per_level,
                max_cited_by=max_cited_by,
                databases=databases,
                enrichment_databases=enrichment_databases,
                openalex_api_key=self._openalex_api_key,
                email=self._email,
                semantic_scholar_api_key=self._semantic_scholar_api_key,
                ieee_api_key=self._ieee_api_key,
                scopus_api_key=self._scopus_api_key,
                pubmed_api_key=self._pubmed_api_key,
                wos_api_key=self._wos_api_key,
                num_workers=num_workers,
                since=since,
                until=until,
                proxy=self._proxy,
                ssl_verify=self._ssl_verify,
                run_state=run_state,
                snapshot_cache=snapshot_cache,
            )

        if managed_store_root is None:
            if managed_run_id is not None or resume_managed_run:
                raise ValueError("managed_run_id/resume_managed_run require managed_store_root")
            return make_runner().run(verbose=verbose, show_progress=show_progress)

        probe = make_runner()
        if resume_managed_run:
            if managed_run_id is None:
                raise ValueError("resuming a managed snowball requires managed_run_id")
            run_state = RunStateStore.resume(
                managed_store_root,
                managed_run_id,
                probe.semantic_spec,
                credentials=probe.credential_identities,
            )
        else:
            run_state = RunStateStore.create(
                managed_store_root,
                probe.semantic_spec,
                credentials=probe.credential_identities,
                run_id=managed_run_id,
            )
        with run_state:
            runner = make_runner(
                run_state=run_state,
                snapshot_cache=SnapshotCache(managed_store_root),
            )
            return runner.run(verbose=verbose, show_progress=show_progress)

    def similar(
        self,
        paper: Paper,
        *,
        databases: list[str] | None = None,
        max_papers_per_database: int | None = None,
        since: dt.date | None = None,
        until: dt.date | None = None,
        enrichment_databases: list[str] | None = DEFAULT_ENRICHMENT_DATABASES,
        max_cited_by: int | None = 100,
        num_workers: int = 1,
        timeout: float | None = 10.0,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> SimilarResult:
        """Find content-similar papers around a single seed paper (single-hop).

        Unlike :meth:`snowball`, which expands the *citation graph*
        (references / cited-by) breadth-first over multiple levels,
        ``similar`` returns papers that are topically or semantically
        related to *paper*, using each source's own content-similarity
        signal:

        * ``"semantic_scholar"`` (highest priority): SPECTER-embedding-based
          recommendations via the Recommendations API.  Despite its name,
          the ``from=all-cs`` pool this method always requests is *not*
          restricted to Computer Science: it is the widest candidate pool
          the API offers (the whole corpus, every field, no date
          restriction).  This is not user-configurable.
        * ``"pubmed"``: NCBI ELink's ``neighbor_score`` related-articles
          score (the PMRA algorithm).  Only applicable to biomedical
          papers; silently contributes nothing when *paper* has no
          resolvable PMID.
        * ``"openalex"`` (lowest priority): the ``related_works`` field
          already embedded in the OpenAlex work record.  A coarser
          topic-tag-overlap signal than the other two.

        This is a **single-hop** operation: there is no ``depth`` parameter
        and no BFS.  Only one seed paper is accepted per call, like
        :meth:`snowball`; loop over multiple seeds yourself if needed.

        All three sources are DOI-anchored (none support looking up a paper
        by title/abstract alone), so *paper* must have a DOI: one without a
        DOI is rejected immediately, before any HTTP calls.

        Results from each source are merged into a single deduplicated list
        via :meth:`~lit_search.core.paper.Paper.merge` (matched by DOI,
        falling back to normalised title, see
        :meth:`~lit_search.core.paper.Paper._identity_key`).  Sources do
        **not** report a comparable similarity score, so results are
        **not** re-ranked by any cross-source score: they are returned in
        source-priority order (Semantic Scholar first, then PubMed-only,
        then OpenAlex-only), with per-paper provenance recorded in
        :attr:`~lit_search.core.paper.Paper.found_in`.

        Like :meth:`search` and :meth:`snowball`, the merged papers are then
        date-filtered (none of the three sources supports native date
        filtering, so this always runs as a post-fetch pass) and enriched
        via per-paper :meth:`get`-style lookups, filling metadata gaps that
        a single source's own parser may have missed.

        Parameters
        ----------
        paper : Paper
            Seed paper to find related papers for.  Typically obtained from
            ``engine.search(...).papers``, ``engine.get(...)``, or
            ``engine.snowball(...).papers``.  Must have a DOI.
        databases : list[str] | None
            Sources to consult, in priority order.  ``None`` (default) uses
            ``["semantic_scholar", "pubmed"]``.  ``"openalex"`` is supported
            but excluded from the default: its ``related_works`` signal is
            coarser and, in practice, noisier than the other two (see
            docs/similar.md).  Pass it explicitly (e.g.
            ``databases=["semantic_scholar", "pubmed", "openalex"]``) to
            include it.
        max_papers_per_database : int | None
            Cap on the number of related papers requested/kept from each
            source before merging.  ``None`` (default) lets each source's
            own natural default apply: Semantic Scholar returns 100 results
            without an explicit limit (accepts higher values, up to
            roughly 500); PubMed's ELink is hard-capped by NCBI at 100
            candidates regardless of any parameter; OpenAlex's
            ``related_works`` is already a short, fixed list (typically
            10-20 entries) with no pagination to request more from.
        since : datetime.date | None
            Only keep related papers published on or after this date.
            ``None`` (default) disables the lower-bound filter.
        until : datetime.date | None
            Only keep related papers published on or before this date.
            ``None`` (default) disables the upper-bound filter.
        enrichment_databases : list[str] | None
            Databases used to enrich related papers after merging and
            filtering.  Defaults to ``["crossref", "web_scraping"]``, which
            cover the majority of metadata gaps without consuming quota
            from rate-limited databases.  Pass an explicit list to enable
            additional (or different) sources.  Accepted values:
            ``"arxiv"``, ``"crossref"``, ``"ieee"``, ``"openalex"``,
            ``"pubmed"``, ``"scopus"``, ``"semantic_scholar"``,
            ``"web_scraping"``, ``"wos"``.
            Pass ``[]`` to disable enrichment entirely.  ``None`` uses the
            default.
        max_cited_by : int | None
            Maximum number of citing-paper DOIs collected per paper when
            ``"openalex"`` or ``"semantic_scholar"`` are in
            *enrichment_databases*.  Defaults to ``100``.  ``None`` means no
            limit: use with caution as highly-cited papers may have
            thousands of citations.  A warning is emitted when this value is
            ``None`` or greater than ``100``.
        num_workers : int
            Number of parallel workers used for the enrichment pass.
            Defaults to ``1`` (sequential).
        timeout : float | None
            HTTP request timeout in seconds.  Defaults to ``10.0``.
        verbose : bool
            When ``True``, emit detailed log messages at DEBUG level.
        show_progress : bool
            When ``True`` (default), display a tqdm progress bar across
            sources.

        Returns
        -------
        SimilarResult
            Container whose ``papers`` attribute holds the merged,
            deduplicated, filtered, and enriched related papers (the seed
            paper itself is excluded, even if a source happens to echo it
            back).  Save via ``lit_search.save_to_json(result, path)``.

        Raises
        ------
        lit_search.exceptions.InvalidParameterError
            If *paper* has no DOI.
        lit_search.exceptions.InvalidParameterError
            If *databases* is an empty list or contains unknown database
            names.
        lit_search.exceptions.InvalidParameterError
            If *enrichment_databases* contains unknown database names.

        See Also
        --------
        lit_search.snowball :
            Multi-hop citation-graph expansion (references / cited-by), a
            different notion of "related" from content similarity.
        lit_search.runners.similar_runner.SimilarRunner :
            Lower-level class for finer-grained control.

        Examples
        --------
        >>> from lit_search import Engine
        >>> engine = Engine()
        >>> seed = engine.get("10.1038/nature12373")
        >>> result = engine.similar(seed, databases=["semantic_scholar", "openalex"])
        >>> print(f"{len(result.papers)} related papers found")
        """
        runner = SimilarRunner(
            paper=paper,
            databases=databases,
            max_papers_per_database=max_papers_per_database,
            since=since,
            until=until,
            enrichment_databases=enrichment_databases,
            max_cited_by=max_cited_by,
            num_workers=num_workers,
            email=self._email,
            openalex_api_key=self._openalex_api_key,
            semantic_scholar_api_key=self._semantic_scholar_api_key,
            pubmed_api_key=self._pubmed_api_key,
            ieee_api_key=self._ieee_api_key,
            scopus_api_key=self._scopus_api_key,
            wos_api_key=self._wos_api_key,
            timeout=timeout,
            proxy=self._proxy,
            ssl_verify=self._ssl_verify,
        )
        return runner.run(verbose=verbose, show_progress=show_progress)
