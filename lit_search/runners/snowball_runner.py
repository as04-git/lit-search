"""SnowballRunner: discover papers via iterative citation snowballing.

Given one paper or a seed set, this runner iteratively fetches references
(backward) and/or citing papers (forward) through typed source-native IDs. The result is a
:class:`~lit_search.core.snowball_result.SnowballResult` containing all
discovered papers with citation relationships encoded on each paper object.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

from lit_search.connectors.arxiv import ArxivConnector
from lit_search.connectors.crossref import CrossRefConnector
from lit_search.connectors.openalex import OpenAlexCitedByPageResult, OpenAlexConnector
from lit_search.connectors.semantic_scholar import SemanticScholarConnector
from lit_search.core.citation import (
    CitationEdge,
    CitationTarget,
    IdentifierKind,
    NativeBatchResult,
    PaperIdentifier,
)
from lit_search.core.coverage import CoverageLedger, CoverageStatus, SourceCoverage
from lit_search.core.identity import record_identity_keys, same_record_identity
from lit_search.core.lookup_result import LookupResult
from lit_search.core.paper import Paper
from lit_search.core.retrieval_audit import auditing
from lit_search.core.snowball_result import SnowballResult
from lit_search.exceptions import InvalidParameterError
from lit_search.runners.discovery_runner import DiscoveryRunner
from lit_search.runners.get_runner import GET_DATABASES, GetRunner
from lit_search.snowball.flow import (
    AcquisitionIntent,
    CurrentCitationCapPolicy,
    RouteBatch,
    TraversalAssertion,
    TraversalLevelPlan,
    alias_groups_match,
    claim_was_attempted,
    plan_frontier,
    reduce_level,
    route_assertions,
    target_was_visited,
    version_aliases,
)
from lit_search.store.run_state import (
    CredentialIdentity,
    RunLifecycle,
    RunPhase,
    RunStateStore,
    WorkUnitState,
    canonical_json,
)
from lit_search.store.snapshot_cache import SnapshotCache, SnapshotQuality
from lit_search.utils.logging_config import configure_verbose_logging
from lit_search.utils.parallel import execute_tasks

logger = logging.getLogger(__name__)

# Sentinel used to distinguish "argument not passed" from explicit None.
_UNSET: object = object()

# Databases supported for snowball traversal.  Only these three can populate
# paper.references (backward) and paper.cited_by (forward), which are required
# for BFS expansion.
SNOWBALL_DATABASES: list[str] = ["crossref", "openalex", "semantic_scholar"]

# Semantic Scholar is the default graph source in both directions.  Its paper
# batch endpoint preserves DOI-less S2 edges and resolves up to 500 typed IDs
# per request; CrossRef remains the journal-DOI route for CrossRef edges.
SNOWBALL_DISCOVERY_DATABASES: list[str] = ["semantic_scholar"]

# Default databases for the final enrichment pass on non-seed papers after
# all BFS levels and filters are applied.  CrossRef provides structured
# metadata; web_scraping fills remaining gaps (abstract, PDF URL, etc.).
SNOWBALL_ENRICHMENT_DATABASES: list[str] = ["crossref", "web_scraping"]

# Databases capable of returning forward citation data (paper.cited_by).
_FORWARD_CAPABLE_DATABASES: frozenset[str] = frozenset({"openalex", "semantic_scholar"})

# Native graph acquisition has an exact provider-request/page boundary in I2b.
# The surrounding aggregate is retained so cache and network paths replay the
# same merged source evidence.  Composite GetRunner DOI/web enrichment remains
# deliberately coarser and is not covered by the exact native-graph claim.
_RUN_REPLAY_GRANULARITY = "mixed_native_request_composite_runner_i2b_v1"
_NATIVE_AGGREGATE_GRANULARITY = "native_resolution_aggregate_i2b_v1"
_PROVIDER_REQUEST_GRANULARITY = "actual_provider_request_page_i2b_v1"

# Native graph snapshots are deliberately short-lived operational acceleration,
# not published evidence.  A source observation older than one day is fetched
# again in the current/exhaustive path.
_SNAPSHOT_TTL_SECONDS = 24 * 60 * 60
_SNAPSHOT_PAYLOAD_SCHEMA = "lit-search/snowball-native-direction/v1"
_SNAPSHOT_REQUEST_SHAPE = "lit-search/snowball-native-graph/v1"


@dataclass
class _CoverageAccumulator:
    """Mutable counters converted to one immutable ledger entry at the end."""

    source: str
    attempted: int = 0
    resolved: int = 0
    not_found: int = 0
    parse_failures: int = 0
    edge_parse_failures: int = 0
    truncated_edges: int = 0
    request_failures: int = 0
    edge_request_failures: int = 0
    errors: list[str] = field(default_factory=list)

    def add(self, outcome: NativeBatchResult) -> None:
        """Accumulate a connector batch outcome."""
        self.attempted += outcome.requested
        self.resolved += len(outcome.papers)
        self.not_found += outcome.not_found
        self.parse_failures += outcome.parse_failures
        self.edge_parse_failures += outcome.edge_parse_failures
        self.truncated_edges += outcome.truncated_edge_records
        self.request_failures += outcome.request_failures
        self.edge_request_failures += outcome.edge_request_failures
        self.errors.extend(outcome.error_messages)

    def to_coverage(self) -> SourceCoverage:
        """Build an actionable source record without hiding partial batches."""
        detail = (
            f"citation traversal attempted {self.attempted} typed target(s), resolved "
            f"{self.resolved}, and received {self.not_found} explicit not-found result(s)"
        )
        if (
            self.request_failures
            or self.parse_failures
            or self.edge_request_failures
            or self.edge_parse_failures
            or self.truncated_edges
        ):
            shortfalls = []
            if self.request_failures:
                shortfalls.append(f"{self.request_failures} request target(s) failed")
            if self.parse_failures:
                shortfalls.append(f"{self.parse_failures} returned record(s) could not be parsed")
            if self.edge_request_failures:
                shortfalls.append(
                    f"{self.edge_request_failures} citation-edge page request(s) failed"
                )
            if self.edge_parse_failures:
                shortfalls.append(
                    f"{self.edge_parse_failures} citation-edge record(s) could not be parsed"
                )
            if self.truncated_edges:
                shortfalls.append(
                    f"at least {self.truncated_edges} citation edge(s) were truncated"
                )
            if self.errors:
                shortfalls.append("; ".join(self.errors[:3]))
            return SourceCoverage.partial(
                self.source,
                self.resolved,
                detail=f"{detail}; " + "; ".join(shortfalls),
            )
        return SourceCoverage(
            source=self.source,
            status=CoverageStatus.COMPLETE,
            retrieved=self.resolved,
            reported_total=self.resolved,
            detail=detail,
        )


@dataclass
class _PolicyResidual:
    """Papers deliberately left out by one bounded traversal policy."""

    name: str
    limit: int
    selected: int = 0
    residual: int = 0
    levels: list[int] = field(default_factory=list)

    def add(self, *, level: int, available: int) -> None:
        """Record one level where the policy bound actually excluded papers."""
        omitted = max(0, available - self.limit)
        if not omitted:
            return
        self.selected += min(available, self.limit)
        self.residual += omitted
        self.levels.append(level)

    def to_coverage(self) -> SourceCoverage:
        """Expose the deliberate residual as non-complete coverage evidence."""
        levels = ", ".join(str(level) for level in self.levels)
        return SourceCoverage.capped(
            f"snowball_policy:{self.name}",
            self.selected,
            cap=self.limit,
            reported_total=self.selected + self.residual,
            detail=(
                f"per-level limit {self.limit} applied at level(s) {levels}; "
                f"{self.residual} paper(s) remain outside this policy's selected set"
            ),
        )


@dataclass(frozen=True, slots=True)
class _OpenAlexForwardEvidence:
    """Merged evidence from a resumable sequence of exact cursor pages."""

    papers: tuple[Paper, ...]
    next_ordinal: int
    request_failures: int = 0
    parse_failures: int = 0
    reported_total: int | None = None
    cursor_cycle: bool = False
    error_messages: tuple[str, ...] = ()


def _validate_snowball_databases(value: list[str], param_name: str) -> list[str]:
    """Validate and normalise a snowball discovery databases list.

    Parameters
    ----------
    value : list[str]
        Raw value to validate.
    param_name : str
        Parameter name used in error messages.

    Returns
    -------
    list[str]
        Normalised (lowercase, stripped) list.

    Raises
    ------
    InvalidParameterError
        If the list is empty or contains unsupported database names.
    """
    if len(value) == 0:
        raise InvalidParameterError(
            f"{param_name} must not be an empty list. Pass None to use the default databases."
        )
    normalised = [db.strip().lower() for db in value]
    unknown = [db for db in normalised if db not in SNOWBALL_DATABASES]
    if unknown:
        raise InvalidParameterError(
            f"Unknown or unsupported database(s) in {param_name}: "
            f"{', '.join(unknown)}. "
            f"Accepted values: {', '.join(sorted(SNOWBALL_DATABASES))}"
        )
    return normalised


def _validate_snowball_enrichment_databases(value: list[str], param_name: str) -> list[str]:
    """Validate and normalise a snowball enrichment databases list.

    Parameters
    ----------
    value : list[str]
        Raw value to validate.
    param_name : str
        Parameter name used in error messages.

    Returns
    -------
    list[str]
        Normalised (lowercase, stripped) list, or an empty list when
        *value* is empty (signals "disable enrichment").

    Raises
    ------
    InvalidParameterError
        If the list contains unknown database names.
    """
    if len(value) == 0:
        return []
    normalised = [db.strip().lower() for db in value]
    unknown = [db for db in normalised if db not in GET_DATABASES]
    if unknown:
        raise InvalidParameterError(
            f"Unknown database(s) in {param_name}: {', '.join(unknown)}. "
            f"Accepted values: {', '.join(sorted(GET_DATABASES))}"
        )
    return normalised


def _validate_snowball_scalar_params(
    max_depth: int,
    max_papers_per_level: int | None,
    max_expansion_per_level: int | None,
    max_cited_by: int | None,
    direction: str,
) -> None:
    """Validate scalar SnowballRunner init parameters and emit warnings.

    Parameters
    ----------
    max_depth : int
        Must be >= 1.
    max_papers_per_level : int | None
        Must be >= 1 when set.
    max_expansion_per_level : int | None
        Must be >= 1 when set.
    max_cited_by : int | None
        Must be >= 1 when set.  Warning emitted when ``None`` or > 100
        and direction includes forward expansion.
    direction : str
        Snowball direction (``"both"``, ``"backward"``, or ``"forward"``).

    Raises
    ------
    InvalidParameterError
        If any parameter is out of range.
    """
    if max_depth < 1:
        raise InvalidParameterError(f"max_depth must be >= 1, got {max_depth}")
    if max_papers_per_level is not None and max_papers_per_level < 1:
        raise InvalidParameterError(
            f"max_papers_per_level must be >= 1 when set, got {max_papers_per_level}"
        )
    if max_expansion_per_level is not None and max_expansion_per_level < 1:
        raise InvalidParameterError(
            f"max_expansion_per_level must be >= 1 when set, got {max_expansion_per_level}"
        )
    if max_cited_by is not None and max_cited_by < 1:
        raise InvalidParameterError(f"max_cited_by must be >= 1 when set, got {max_cited_by}")
    if direction in ("forward", "both") and (max_cited_by is None or max_cited_by > 100):
        logger.warning(
            "max_cited_by is %s. Fetching cited-by may be very slow or produce "
            "large lists for highly-cited papers (forward/both direction). "
            "Consider setting max_cited_by <= 100.",
            max_cited_by if max_cited_by is not None else "None (unlimited)",
        )


class SnowballRunner(DiscoveryRunner):
    """Discover papers around a seed paper via iterative citation snowballing.

    Starting from one paper or every member of a seed set, the runner follows
    provenance-bearing citation targets up to *max_depth* BFS levels. DOI-only
    lists remain a backward-compatibility path.

    The seed paper and frontier papers (those driving the next BFS level) are
    fetched with the combined set of *databases* and *enrichment_databases*
    so they carry full metadata, including ``paper.cited_by``, before each
    expansion round. Papers discovered during BFS are fetched through the
    source that asserted each edge; arXiv identifiers never go to CrossRef.
    After all BFS levels complete
    and date filters are applied, the surviving non-seed papers are
    re-enriched with the databases in *enrichment_databases* that were not
    already used (i.e., those not in *databases*).

    Only ``"crossref"``, ``"openalex"``, and ``"semantic_scholar"`` are valid
    for *databases*: these are the only sources that populate
    ``paper.references`` and ``paper.cited_by`` for BFS expansion.
    Forward direction (``"forward"`` or ``"both"``) requires at least one
    of ``"openalex"`` or ``"semantic_scholar"`` in *databases*.

    Parameters
    ----------
    seed_paper : Paper | Sequence[Paper]
        One paper or a non-empty seed set with routable typed identifiers.
    max_depth : int
        Maximum number of BFS levels. ``2`` is the default full-set protocol.
    direction : Literal["both", "backward", "forward"]
        ``"backward"`` follows :attr:`~lit_search.core.paper.Paper.references`
        (papers cited *by* the current frontier),
        ``"forward"`` follows :attr:`~lit_search.core.paper.Paper.cited_by`
        (papers that *cite* the current frontier),
        ``"both"`` expands in both directions.
    max_papers_per_level : int | None
        When set, only the *top-N* most-cited papers discovered at each
        level are kept in the final result.  The seed paper is never filtered.
        ``None`` (default) keeps all discovered papers.
    max_expansion_per_level : int | None
        When set, only the *top-N* most-cited papers from each level are
        used as seeds for the next BFS round.  Papers already added to the
        result are unaffected.  ``None`` (default) expands the full frontier.
    databases : list[str] | None
        Databases used to enrich seeds and frontier papers.  Only
        ``"crossref"``, ``"openalex"``, and ``"semantic_scholar"`` are
        accepted. Defaults to ``["semantic_scholar"]`` in every direction.
        Pass ``None`` to use all three databases.
    enrichment_databases : list[str] | None
        Databases used to enrich non-seed papers after all BFS levels and
        filters are applied.  Databases already used in discovery are
        skipped (they were already fetched).  Defaults to
        ``["crossref", "web_scraping"]``.  ``None`` uses the default.
        Pass ``[]`` to disable enrichment entirely.  Accepted values:
        ``"arxiv"``, ``"crossref"``, ``"ieee"``, ``"openalex"``,
        ``"pubmed"``, ``"scopus"``, ``"semantic_scholar"``,
        ``"web_scraping"``, ``"wos"``.
    num_workers : int
        Number of parallel :class:`~lit_search.runners.get_runner.GetRunner`
        calls to make per level.  Defaults to ``1`` (sequential).
    since : datetime.date | None
        Only include discovered papers published on or after this date.
        The seed paper is never filtered.  ``None`` disables the filter.
    until : datetime.date | None
        Only include discovered papers published on or before this date.
        The seed paper is never filtered.  ``None`` disables the filter.
    openalex_api_key : str | None
        OpenAlex API key.
    email : str | None
        Contact email for polite-pool access (OpenAlex, CrossRef).
    semantic_scholar_api_key : str | None
        Semantic Scholar API key.
    ieee_api_key : str | None
        IEEE Xplore API key.
    scopus_api_key : str | None
        Elsevier / Scopus API key.
    pubmed_api_key : str | None
        NCBI PubMed API key.
    wos_api_key : str | None
        Clarivate Web of Science API key.
    proxy : str | None
        Optional HTTP/HTTPS proxy URL for all requests.
    ssl_verify : bool
        Whether to verify SSL certificates.  Defaults to ``True``.
    run_state : RunStateStore | None
        Optional I2 replay store. Native graph acquisition checkpoints each
        actual S2/OpenAlex/arXiv batch, Crossref/OpenAlex singleton, and
        OpenAlex cursor page. Composite per-DOI GetRunner/web enrichment stays
        at runner-operation granularity and is not part of that exact claim.
    snapshot_cache : SnapshotCache | None
        Optional shared source-snapshot cache. Only fresh, complete native
        direction snapshots satisfy this current/exhaustive path. Composite
        ``GetRunner`` results are never stored in it.
    """

    def __init__(
        self,
        seed_paper: Paper | Sequence[Paper],
        *,
        max_depth: int = 2,
        direction: Literal["both", "backward", "forward"] = "both",
        max_papers_per_level: int | None = None,
        max_expansion_per_level: int | None = None,
        max_cited_by: int | None = 100,
        databases: list[str] | None = _UNSET,  # type: ignore[assignment]
        enrichment_databases: list[str] | None = _UNSET,  # type: ignore[assignment]
        num_workers: int = 1,
        since: datetime.date | None = None,
        until: datetime.date | None = None,
        openalex_api_key: str | None = None,
        email: str | None = None,
        semantic_scholar_api_key: str | None = None,
        ieee_api_key: str | None = None,
        scopus_api_key: str | None = None,
        pubmed_api_key: str | None = None,
        wos_api_key: str | None = None,
        proxy: str | None = None,
        ssl_verify: bool = True,
        run_state: RunStateStore | None = None,
        snapshot_cache: SnapshotCache | None = None,
    ) -> None:
        """Initialise snowball configuration without executing it.

        Parameters
        ----------
        seed_paper : Paper | Sequence[Paper]
            One paper or a non-empty seed set with routable typed identifiers.
        max_depth : int
            Maximum BFS depth.  Must be >= 1.
        direction : Literal["both", "backward", "forward"]
            Snowball direction(s).
        max_papers_per_level : int | None
            Per-level result cap.  When set, only the top-N most-cited papers
            discovered at each BFS level are kept in the final result.  Seed
            papers are never filtered.  ``None`` means all papers are kept.
        max_expansion_per_level : int | None
            Per-level frontier cap.  When set, only the top-N most-cited papers
            from each level are used as seeds for the next BFS round.  Papers
            already added to the result are unaffected.  ``None`` means the
            full set of discovered papers drives the next level.
        max_cited_by : int | None
            Maximum number of citing-paper DOIs collected per paper when
            populating ``paper.cited_by`` during seed and frontier enrichment.
            Defaults to ``100``.  ``None`` means no limit: use with caution as
            highly-cited classic papers may have thousands of citations.
            A warning is emitted when this value is ``None`` or greater than ``100``.
        databases : list[str] | None
            Databases used to enrich seeds and frontier papers.  Only
            ``"crossref"``, ``"openalex"``, and ``"semantic_scholar"`` are
            accepted. When omitted, defaults to ``["semantic_scholar"]`` in
            every direction. Pass ``None`` to use all three databases.
        enrichment_databases : list[str] | None
            Databases used to enrich non-seed papers after BFS completes.
            When not provided or ``None``, defaults to
            ``["crossref", "web_scraping"]``.  Pass ``[]`` to disable
            enrichment entirely.  Accepted values: ``"arxiv"``,
            ``"crossref"``, ``"ieee"``, ``"openalex"``, ``"pubmed"``,
            ``"scopus"``, ``"semantic_scholar"``, ``"web_scraping"``,
            ``"wos"``.
        num_workers : int
            Number of parallel GetRunner calls per level.
        since : datetime.date | None
            Lower-bound publication date filter for discovered papers.
        until : datetime.date | None
            Upper-bound publication date filter for discovered papers.
        openalex_api_key : str | None
            OpenAlex API key.
        email : str | None
            Contact email for polite-pool access.
        semantic_scholar_api_key : str | None
            Semantic Scholar API key.
        ieee_api_key : str | None
            IEEE Xplore API key.
        scopus_api_key : str | None
            Scopus API key.
        pubmed_api_key : str | None
            PubMed API key.
        wos_api_key : str | None
            Web of Science API key.
        proxy : str | None
            Optional proxy URL.
        ssl_verify : bool
            Whether to verify SSL certificates.
        run_state : RunStateStore | None
            Optional I2 replay state. Native graph requests/pages are exact;
            composite GetRunner/web enrichment remains deliberately coarse.
        snapshot_cache : SnapshotCache | None
            Optional source-snapshot cache. Stale, partial, and truncated
            observations remain misses in this runner.

        Raises
        ------
        InvalidParameterError
            If *max_depth* is less than 1, *max_papers_per_level* is less than 1,
            *max_expansion_per_level* is less than 1, *databases* is an
            empty list, *databases* contains unsupported database identifiers,
            the declared *direction* cannot be satisfied by *databases*
            (e.g. ``"forward"`` with only ``"crossref"``), *seed_paper* is
            not a single :class:`~lit_search.core.paper.Paper`, or it has
            no DOI.
        """
        _validate_snowball_scalar_params(
            max_depth, max_papers_per_level, max_expansion_per_level, max_cited_by, direction
        )

        # Apply defaults for the sentinel-based parameters.
        if databases is _UNSET:
            databases = list(SNOWBALL_DISCOVERY_DATABASES)
        elif databases is None:
            databases = list(SNOWBALL_DATABASES)

        if enrichment_databases is _UNSET or enrichment_databases is None:
            enrichment_databases = list(SNOWBALL_ENRICHMENT_DATABASES)

        databases = _validate_snowball_databases(databases, "databases")
        enrichment_databases = _validate_snowball_enrichment_databases(
            enrichment_databases, "enrichment_databases"
        )

        # Validate that the declared databases can satisfy the requested direction.
        # CrossRef only supports backward references; forward requires OpenAlex or
        # Semantic Scholar.
        if direction in ("forward", "both") and not any(
            db in _FORWARD_CAPABLE_DATABASES for db in databases
        ):
            raise InvalidParameterError(
                f"Direction '{direction}' requires at least one forward-capable database "
                f"({', '.join(sorted(_FORWARD_CAPABLE_DATABASES))}), "
                f"but databases={databases!r}. "
                "Add 'openalex' or 'semantic_scholar' to the databases list."
            )

        # DiscoveryRunner handles shared credentials and date filters.
        # enrichment_databases=[] because GetRunner is invoked directly here.
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
            enrichment_databases=[],
        )

        if isinstance(seed_paper, Paper):
            seed_papers = [seed_paper]
        elif isinstance(seed_paper, Sequence) and seed_paper:
            seed_papers = list(seed_paper)
            if not all(isinstance(item, Paper) for item in seed_papers):
                raise InvalidParameterError("every snowball seed must be a Paper")
        else:
            raise InvalidParameterError("seed_paper must be a Paper or a non-empty sequence")
        if not any(paper.identifiers for paper in seed_papers):
            raise InvalidParameterError(
                "the seed set contains no DOI or source-native identifier, so no member "
                "can be traversed"
            )

        self._seed_papers = seed_papers
        self._max_depth = max_depth
        self._direction = direction
        self._max_papers_per_level = max_papers_per_level
        self._max_expansion_per_level = max_expansion_per_level
        self._max_cited_by = max_cited_by
        self._databases = databases
        self._enrichment_databases: list[str] = enrichment_databases
        self._num_workers = max(num_workers, 1)
        self._coverage: dict[str, _CoverageAccumulator] = {}
        self._policy_residuals: dict[str, _PolicyResidual] = {}
        self._attempted_routes: dict[str, list[frozenset[str]]] = {}
        self._visited_alias_groups: list[frozenset[str]] = []
        self._lookup_coverages: list[CoverageLedger] = []
        self._run_state = run_state
        self._snapshot_cache = snapshot_cache
        self._operation_phase = RunPhase.SEED_FETCH
        self._operation_level: int | None = None
        self._semantic_spec = self._build_semantic_spec()
        if self._run_state is not None:
            self._run_state.validate_semantic_spec(
                self.semantic_spec,
                credentials=self.credential_identities,
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def semantic_spec(self) -> dict[str, Any]:
        """Return a detached copy of the semantic spec frozen at construction."""
        value = json.loads(canonical_json(self._semantic_spec))
        if not isinstance(value, dict):  # pragma: no cover - construction guarantees this
            raise RuntimeError("snowball semantic spec is not an object")
        return value

    def _build_semantic_spec(self) -> dict[str, Any]:
        """Return every membership- or evidence-affecting runner parameter.

        Seed identities are canonical alias groups. Secrets are excluded here
        and represented only by :attr:`credential_identities` fingerprints.
        """
        seed_identities = []
        seed_payload_digests = []
        for paper in self._seed_papers:
            payload_digest = hashlib.blake2b(
                canonical_json(paper.to_dict()).encode("utf-8"), digest_size=16
            ).hexdigest()
            aliases = sorted(record_identity_keys(paper))
            if not aliases:
                aliases = [f"snapshot:{payload_digest}"]
            seed_identities.append(aliases)
            seed_payload_digests.append(payload_digest)
        seed_identities.sort()
        seed_payload_digests.sort()
        return {
            "algorithm": "snowball_runner_operation_replay",
            "replay_granularity": _RUN_REPLAY_GRANULARITY,
            "version": 1,
            "seed_identities": seed_identities,
            "seed_payload_digests": seed_payload_digests,
            "max_depth": self._max_depth,
            "direction": self._direction,
            "max_papers_per_level": self._max_papers_per_level,
            "max_expansion_per_level": self._max_expansion_per_level,
            "max_cited_by": self._max_cited_by,
            "databases": list(self._databases),
            "enrichment_databases": list(self._enrichment_databases),
            "num_workers": self._num_workers,
            "since": self._since.isoformat() if self._since else None,
            "until": self._until.isoformat() if self._until else None,
            "ssl_verify": self._ssl_verify,
            "snapshot_cache_policy": "fresh_complete_directional_v1",
        }

    @property
    def credential_identities(self) -> tuple[CredentialIdentity, ...]:
        """Return credential presence/fingerprints without retaining secrets."""
        materials = (
            ("openalex", self._openalex_api_key),
            ("crossref_email", self._email),
            ("semantic_scholar", self._semantic_scholar_api_key),
            ("ieee", self._ieee_api_key),
            ("scopus", self._scopus_api_key),
            ("pubmed", self._pubmed_api_key),
            ("wos", self._wos_api_key),
            ("proxy", self._proxy),
        )
        return tuple(
            CredentialIdentity.from_material(source, material) for source, material in materials
        )

    def _advance_run_phase(self, phase: RunPhase, *, level: int | None = None) -> None:
        """Advance durable phase, tolerating replay of already-passed stages."""
        if self._run_state is None:
            return
        current = self._run_state.snapshot()
        desired_position = self._phase_position(phase, level)
        current_position = self._phase_position(current.phase, current.level)
        if desired_position <= current_position:
            return
        self._run_state.transition_phase(phase, level=level)

    @staticmethod
    def _phase_position(phase: RunPhase, level: int | None) -> tuple[int, int, int]:
        """Order phases for restart-from-zero replay without backward transitions."""
        if phase is RunPhase.SEED_PLAN:
            return (0, 0, 0)
        if phase is RunPhase.SEED_FETCH:
            return (1, 0, 0)
        if phase in {RunPhase.LEVEL_PLAN, RunPhase.LEVEL_FETCH, RunPhase.LEVEL_REDUCE}:
            rank = {
                RunPhase.LEVEL_PLAN: 0,
                RunPhase.LEVEL_FETCH: 1,
                RunPhase.LEVEL_REDUCE: 2,
            }[phase]
            return (2, level or 0, rank)
        if phase is RunPhase.ENRICH_PLAN:
            return (3, 0, 0)
        if phase is RunPhase.ENRICH_FETCH:
            return (4, 0, 0)
        return (5, 0, 0)

    def _process_seed_papers(
        self,
        visited: set[str],
        verbose: bool,
        show_progress: bool,
    ) -> list[Paper]:
        """Fetch and enrich level-0 seed papers, returning them as the initial frontier.

        Seeds are the first BFS frontier, so they are fetched with the full
        *enrichment_databases* set (all API connectors) to ensure that both
        ``paper.references`` and ``paper.cited_by`` are populated before the
        first expansion round.

        Parameters
        ----------
        visited : set[str]
            Already-seen normalised DOIs.  Updated in place with seed DOIs.
        verbose : bool
            Enable verbose logging.
        show_progress : bool
            Show tqdm progress bars.

        Returns
        -------
        list[Paper]
            Enriched seed papers to be used as the initial BFS frontier.
            Seeds for which GetRunner returned nothing are included as-is.
        """
        if len(self._seed_papers) > 1 or any(not paper.doi for paper in self._seed_papers):
            enriched_seeds = list(self._seed_papers)
            routable_seeds = [paper for paper in enriched_seeds if paper.identifiers]
            unroutable_count = len(enriched_seeds) - len(routable_seeds)
            for source in self._databases:
                if unroutable_count:
                    self._record_coverage(
                        NativeBatchResult(
                            source=source,
                            papers=(),
                            requested=unroutable_count,
                            request_failures=unroutable_count,
                            error_messages=(
                                f"{unroutable_count} seed paper(s) had no DOI or native graph ID",
                            ),
                        )
                    )
                seed_edges = [
                    CitationEdge(target=paper.citation_target(), source=source)
                    for paper in routable_seeds
                ]
                available = self._fetch_targets(seed_edges)
                for seed in enriched_seeds:
                    match = next(
                        (paper for paper in available if same_record_identity(paper, seed)),
                        None,
                    )
                    if match is not None:
                        seed.merge(match)
            return enriched_seeds

        seed_dois = [p.doi for p in self._seed_papers if p.doi]
        enriched: dict[str, Paper] = {}
        if seed_dois:
            # Seeds are enriched with all available databases: both the
            # discovery set and the enrichment set: to ensure full metadata
            # (including references and cited_by) before the first BFS round.
            seed_databases = sorted(set(self._databases + self._enrichment_databases))
            fetched_seeds = self._fetch_dois(
                seed_dois,
                databases=seed_databases,
                desc="Seeds",
                verbose=verbose,
                show_progress=False,
                max_cited_by=self._max_cited_by,
            )
            for paper in fetched_seeds:
                if paper.doi:
                    enriched[paper.doi.strip().lower()] = paper

        # Seeds for which GetRunner returned nothing are included as-is.
        for seed in self._seed_papers:
            if not seed.doi:
                continue
            norm = seed.doi.strip().lower()
            if norm not in enriched:
                enriched[norm] = seed

        enriched_values = list(enriched.values())
        self._record_frontier_edge_coverage(enriched_values)
        return enriched_values

    def _record_frontier_edge_coverage(self, papers: list[Paper]) -> None:
        """Audit edge-list acquisition for legacy per-paper seed lookups."""
        for source in self._databases:
            if source in self._coverage:
                continue
            resolved_papers: list[Paper] = []
            truncated = 0
            for paper in papers:
                totals = paper.citation_edge_totals.get(source)
                if totals is None:
                    continue
                resolved_papers.append(paper)
                returned = 0
                expected = 0
                if self._direction in ("both", "backward"):
                    expected += totals.get("references", 0)
                    returned += sum(1 for edge in paper.reference_edges if edge.source == source)
                if self._direction in ("both", "forward") and source in _FORWARD_CAPABLE_DATABASES:
                    expected += totals.get("citations", 0)
                    returned += sum(1 for edge in paper.cited_by_edges if edge.source == source)
                truncated += max(0, expected - returned)
            missing = len(papers) - len(resolved_papers)
            self._record_coverage(
                NativeBatchResult(
                    source=source,
                    papers=tuple(resolved_papers),
                    requested=len(papers),
                    request_failures=missing,
                    truncated_edge_records=truncated,
                    error_messages=(
                        (
                            f"{missing} seed lookup(s) produced no edge-coverage evidence "
                            f"for {source}"
                        ),
                    )
                    if missing
                    else (),
                )
            )

    @staticmethod
    def _identifier_for_source(target: CitationTarget, source: str) -> str | None:
        """Choose the identifier syntax accepted by the asserting source."""
        if source == "semantic_scholar":
            if value := target.get(IdentifierKind.SEMANTIC_SCHOLAR):
                return value
            if value := target.get(IdentifierKind.ARXIV):
                return f"ARXIV:{value}"
            if value := target.get(IdentifierKind.DOI):
                return f"DOI:{value}"
            return None
        if source == "openalex":
            return target.get(IdentifierKind.OPENALEX)
        if source == "arxiv":
            return target.get(IdentifierKind.ARXIV)
        if source == "crossref":
            return target.get(IdentifierKind.DOI)
        return None

    @staticmethod
    def _legacy_edge_source(paper: Paper) -> str:
        """Assign the narrowest defensible provenance to old DOI-only fields."""
        return "legacy_doi"

    @staticmethod
    def _paper_aliases(paper: Paper) -> frozenset[str]:
        """Return raw routable aliases plus canonical record-identity aliases."""
        return frozenset(
            {identifier.key for identifier in paper.identifiers} | set(record_identity_keys(paper))
        )

    @staticmethod
    def _version_aliases(aliases: frozenset[str]) -> frozenset[str]:
        """Compatibility wrapper around the pure flow identity helper."""
        return version_aliases(aliases)

    @classmethod
    def _alias_groups_match(cls, left: frozenset[str], right: frozenset[str]) -> bool:
        """Compatibility wrapper around conflict-aware flow identity."""
        return alias_groups_match(left, right)

    def _target_was_visited(self, aliases: frozenset[str], visited: set[str]) -> bool:
        """Compatibility wrapper around conflict-aware visited matching."""
        return target_was_visited(aliases, visited, self._visited_alias_groups)

    def _remember_visited_group(self, aliases: frozenset[str]) -> None:
        """Remember one resolved alias group without unioning conflicts."""
        if aliases and aliases not in self._visited_alias_groups:
            self._visited_alias_groups.append(aliases)

    @classmethod
    def _claim_was_attempted(
        cls, aliases: frozenset[str], attempted: Sequence[frozenset[str]]
    ) -> bool:
        """Compatibility wrapper around conflict-aware route matching."""
        return claim_was_attempted(aliases, attempted)

    def _collect_candidate_targets(
        self, frontier: list[Paper], visited: set[str]
    ) -> list[CitationEdge]:
        """Compatibility view of a pure assertion-preserving frontier plan."""
        plan = self._plan_frontier(frontier, visited=visited, level=1)
        self._reserve_plan(plan)
        return [
            CitationEdge(target=assertion.target, source=assertion.asserting_source)
            for assertion in plan.assertions
        ]

    def _plan_frontier(
        self,
        frontier: Sequence[Paper],
        *,
        visited: set[str],
        level: int,
    ) -> TraversalLevelPlan:
        """Build one immutable plan from the runner's current traversal state."""
        return plan_frontier(
            frontier,
            level=level,
            direction=self._direction,
            visited=visited,
            visited_alias_groups=self._visited_alias_groups,
            attempted_routes=self._attempted_routes,
            canonical=self._run_state is not None,
        )

    def _reserve_plan(self, plan: TraversalLevelPlan) -> None:
        """Reserve each unique route after the side-effect-free plan exists."""
        for source, aliases in plan.reservations:
            alias_group = frozenset(aliases)
            attempted = self._attempted_routes.setdefault(source, [])
            if alias_group not in attempted:
                attempted.append(alias_group)

    def _record_coverage(self, outcome: NativeBatchResult) -> None:
        """Merge a native fetch outcome into this run's coverage counters."""
        accumulator = self._coverage.setdefault(
            outcome.source, _CoverageAccumulator(source=outcome.source)
        )
        accumulator.add(outcome)

    @staticmethod
    def _paper_key(paper: Paper) -> str:
        """Return a deterministic diagnostic key without using title identity.

        The result store adds a suffix when this base key collides.  That is
        necessary because conflicting records can legitimately share a
        provider alias, and distinct identifier-free records can have an
        identical snapshot.
        """
        keys = sorted(record_identity_keys(paper))
        if keys:
            version_keys = [key for key in keys if key.startswith(("doi:", "arxiv:"))]
            return version_keys[0] if version_keys else keys[0]
        digest = hashlib.blake2b(
            canonical_json(paper.to_dict()).encode("utf-8"), digest_size=16
        ).hexdigest()
        return f"unidentified:{digest}"

    @staticmethod
    def _matching_paper_key(all_papers: dict[str, Paper], paper: Paper) -> str | None:
        """Return the stored key for an explicitly identical record, if any."""
        return next(
            (key for key, existing in all_papers.items() if same_record_identity(existing, paper)),
            None,
        )

    @classmethod
    def _store_paper(cls, all_papers: dict[str, Paper], paper: Paper) -> str:
        """Store a record or merge it only under explicit record identity.

        Returns the collision-safe storage token.  The token is deliberately
        separate from bibliographic identity: it may carry a numeric suffix
        when conflicting or unidentified records share the same base key.
        """
        existing_key = cls._matching_paper_key(all_papers, paper)
        if existing_key is not None:
            all_papers[existing_key].merge(paper)
            return existing_key

        base_key = cls._paper_key(paper)
        storage_key = base_key
        suffix = 2
        while storage_key in all_papers:
            storage_key = f"{base_key}#{suffix}"
            suffix += 1
        all_papers[storage_key] = paper
        return storage_key

    @classmethod
    def _coalesce_papers(cls, papers: Sequence[Paper]) -> list[Paper]:
        """Coalesce a sequence only where explicit record identity permits."""
        records: dict[str, Paper] = {}
        for paper in papers:
            cls._store_paper(records, paper)
        return list(records.values())

    def _plan_native_operation(
        self,
        *,
        source: str,
        targets: Sequence[CitationTarget],
        include_edges: bool,
        ordinal: int,
    ) -> tuple[str | None, NativeBatchResult | None]:
        """Plan or replay one cache/network-equivalent native aggregate.

        This unit is intentionally not labelled as a provider request.  On a
        cold path its child units checkpoint each actual request/page; on a
        warm path it records the equivalent merged source evidence without
        inventing provider I/O.
        """
        if self._run_state is None:
            return None, None
        target_groups = sorted(canonical_json(sorted(target.keys)) for target in targets)
        unit_id = self._run_state.plan_work_unit(
            phase=self._operation_phase,
            level=self._operation_level,
            source=source,
            request_kind="native_resolution_aggregate",
            ordinal=ordinal,
            targets=target_groups,
            semantic_context={
                "include_edges": include_edges,
                "direction": self._direction,
                "max_cited_by": self._max_cited_by,
                "granularity": _NATIVE_AGGREGATE_GRANULARITY,
                "target_alias_groups": target_groups,
            },
        )
        unit = self._run_state.work_unit(unit_id)
        if unit.state in {
            WorkUnitState.SUCCEEDED,
            WorkUnitState.NOT_FOUND,
            WorkUnitState.FAILED,
            WorkUnitState.UNROUTABLE,
        }:
            if unit.outcome is None or "native_batch_result" not in unit.outcome:
                raise RuntimeError(f"replayable native unit {unit_id} has no persisted outcome")
            return unit_id, NativeBatchResult.from_dict(unit.outcome["native_batch_result"])
        self._run_state.begin_work_unit(unit_id)
        return unit_id, None

    @staticmethod
    def _native_work_state(outcome: NativeBatchResult) -> WorkUnitState:
        """Map a coverage-bearing native outcome to operational terminal state."""
        if outcome.requested and outcome.not_found == outcome.requested and outcome.complete:
            return WorkUnitState.NOT_FOUND
        if outcome.request_failures and not outcome.papers:
            return WorkUnitState.FAILED
        if not outcome.papers and outcome.requested == 0:
            return WorkUnitState.UNROUTABLE
        return WorkUnitState.SUCCEEDED

    @staticmethod
    def _native_coverage_payload(outcome: NativeBatchResult) -> dict[str, Any]:
        """Persist the counters needed to audit a replayed native operation."""
        return {
            "source": outcome.source,
            "complete": outcome.complete,
            "requested": outcome.requested,
            "resolved": len(outcome.papers),
            "not_found": outcome.not_found,
            "parse_failures": outcome.parse_failures,
            "truncated_edge_records": outcome.truncated_edge_records,
            "request_failures": outcome.request_failures,
            "edge_parse_failures": outcome.edge_parse_failures,
            "edge_request_failures": outcome.edge_request_failures,
            "error_messages": list(outcome.error_messages),
        }

    def _commit_native_operation(self, unit_id: str | None, outcome: NativeBatchResult) -> None:
        """Persist terminal provider evidence before local consumption."""
        if self._run_state is None or unit_id is None:
            return
        unit = self._run_state.work_unit(unit_id)
        if unit.state is WorkUnitState.IN_FLIGHT:
            self._run_state.complete_work_unit(
                unit_id,
                self._native_work_state(outcome),
                outcome={"native_batch_result": outcome.to_dict()},
                coverage=self._native_coverage_payload(outcome),
            )

    def _commit_native_reduction(self, unit_id: str | None, outcome: NativeBatchResult) -> None:
        """Mark replayed native evidence consumed after local accounting."""
        if self._run_state is None or unit_id is None:
            return
        self._run_state.commit_reduction(
            unit_id,
            "snowball_fetch_targets",
            {
                "paper_keys": sorted(self._paper_key(paper) for paper in outcome.papers),
                "coverage": self._native_coverage_payload(outcome),
            },
        )

    def _execute_native_request(
        self,
        *,
        source: str,
        request_kind: str,
        targets: Sequence[str],
        ordinal: int,
        semantic_context: dict[str, Any],
        fetch: Callable[[], NativeBatchResult],
    ) -> NativeBatchResult:
        """Execute or replay exactly one native provider request.

        Connector retry attempts remain part of this one logical request.  A
        process interruption leaves the unit in flight, so recovery may repeat
        that request; terminal parsed evidence is never reissued.
        """
        if self._run_state is None:
            return fetch()
        context = {
            **semantic_context,
            "granularity": _PROVIDER_REQUEST_GRANULARITY,
        }
        unit_id = self._run_state.plan_work_unit(
            phase=self._operation_phase,
            level=self._operation_level,
            source=source,
            request_kind=request_kind,
            ordinal=ordinal,
            targets=targets,
            semantic_context=context,
        )
        unit = self._run_state.work_unit(unit_id)
        if unit.state in {
            WorkUnitState.SUCCEEDED,
            WorkUnitState.NOT_FOUND,
            WorkUnitState.FAILED,
            WorkUnitState.UNROUTABLE,
        }:
            if unit.outcome is None or "native_batch_result" not in unit.outcome:
                raise RuntimeError(f"replayable provider unit {unit_id} has no persisted outcome")
            outcome = NativeBatchResult.from_dict(unit.outcome["native_batch_result"])
        else:
            self._run_state.begin_work_unit(unit_id)
            outcome = fetch()
            if outcome.source != source:
                raise RuntimeError(
                    f"provider unit for {source} returned evidence for {outcome.source}"
                )
            self._commit_native_operation(unit_id, outcome)
        self._commit_native_reduction(unit_id, outcome)
        return outcome

    def _execute_openalex_cited_by_page(
        self,
        connector: OpenAlexConnector,
        *,
        openalex_id: str,
        cursor: str,
        page_index: int,
        ordinal: int,
        sort_by_citations: bool,
    ) -> OpenAlexCitedByPageResult:
        """Execute or replay one exact OpenAlex cited-by cursor page."""
        if self._run_state is None:
            return connector.fetch_cited_by_page(
                openalex_id,
                cursor,
                sort_by_citations=sort_by_citations,
            )
        target = PaperIdentifier(IdentifierKind.OPENALEX, openalex_id).key
        context = {
            "request_schema": "openalex_cited_by_page_v1",
            "granularity": _PROVIDER_REQUEST_GRANULARITY,
            "cursor": cursor,
            "page_index": page_index,
            "page_size": 200,
            "sort_by_citations": sort_by_citations,
            "max_cited_by": self._max_cited_by,
        }
        unit_id = self._run_state.plan_work_unit(
            phase=self._operation_phase,
            level=self._operation_level,
            source="openalex",
            request_kind="cited_by_page",
            ordinal=ordinal,
            targets=[target],
            semantic_context=context,
        )
        unit = self._run_state.work_unit(unit_id)
        if unit.state in {
            WorkUnitState.SUCCEEDED,
            WorkUnitState.NOT_FOUND,
            WorkUnitState.FAILED,
            WorkUnitState.UNROUTABLE,
        }:
            if unit.outcome is None or "openalex_cited_by_page" not in unit.outcome:
                raise RuntimeError(f"replayable cited-by unit {unit_id} has no persisted outcome")
            outcome = OpenAlexCitedByPageResult.from_dict(
                dict(unit.outcome["openalex_cited_by_page"])
            )
        else:
            self._run_state.begin_work_unit(unit_id)
            outcome = connector.fetch_cited_by_page(
                openalex_id,
                cursor,
                sort_by_citations=sort_by_citations,
            )
            state = WorkUnitState.FAILED if outcome.request_failures else WorkUnitState.SUCCEEDED
            coverage = {
                "source": "openalex",
                "complete": outcome.complete,
                "retrieved": len(outcome.papers),
                "raw_result_count": outcome.raw_result_count,
                "reported_total": outcome.reported_total,
                "parse_failures": outcome.parse_failures,
                "request_failures": outcome.request_failures,
                "cursor": outcome.cursor,
                "next_cursor": outcome.next_cursor,
                "error_messages": list(outcome.error_messages),
            }
            self._run_state.complete_work_unit(
                unit_id,
                state,
                outcome={"openalex_cited_by_page": outcome.to_dict()},
                coverage=coverage,
            )
        self._run_state.commit_reduction(
            unit_id,
            "snowball_openalex_cited_by_page",
            {
                "paper_keys": sorted(self._paper_key(paper) for paper in outcome.papers),
                "cursor": outcome.cursor,
                "next_cursor": outcome.next_cursor,
                "parse_failures": outcome.parse_failures,
                "request_failures": outcome.request_failures,
            },
        )
        return outcome

    def _fetch_openalex_cited_by_resumable(
        self,
        connector: OpenAlexConnector,
        paper: Paper,
        *,
        ordinal: int,
        paper_index: int,
    ) -> _OpenAlexForwardEvidence:
        """Fetch one paper's citing works as canonical cursor-page units."""
        native_id = next(
            (
                identifier.value
                for identifier in paper.identifiers
                if identifier.kind is IdentifierKind.OPENALEX
            ),
            None,
        )
        if native_id is None:
            return _OpenAlexForwardEvidence((), ordinal)
        openalex_id = f"https://openalex.org/{native_id}"
        cursor: str | None = "*"
        page_index = 0
        seen_cursors: set[str] = set()
        all_papers: list[Paper] = []
        request_failures = 0
        parse_failures = 0
        reported_total: int | None = None
        errors: list[str] = []
        stopped_explicitly = False
        sort_by_citations = self._max_cited_by is not None
        while cursor is not None and cursor not in seen_cursors:
            seen_cursors.add(cursor)
            outcome = self._execute_openalex_cited_by_page(
                connector,
                openalex_id=openalex_id,
                cursor=cursor,
                page_index=page_index,
                ordinal=ordinal,
                sort_by_citations=sort_by_citations,
            )
            ordinal += 1
            all_papers.extend(outcome.papers)
            request_failures += outcome.request_failures
            parse_failures += outcome.parse_failures
            if outcome.reported_total is not None:
                reported_total = outcome.reported_total
            errors.extend(outcome.error_messages)
            if outcome.request_failures:
                stopped_explicitly = True
                break
            if self._max_cited_by is not None and len(all_papers) >= self._max_cited_by:
                all_papers = all_papers[: self._max_cited_by]
                stopped_explicitly = True
                break
            cursor = outcome.next_cursor
            page_index += 1
        cursor_cycle = not stopped_explicitly and cursor is not None and cursor in seen_cursors
        if cursor_cycle:
            errors.append(
                f"OpenAlex repeated cited-by cursor {cursor!r}; pagination stopped before "
                "closure could be established"
            )
        logger.debug(
            "OpenAlex native cited-by: paper index %d, %d page(s), %d paper(s)",
            paper_index,
            len(seen_cursors),
            len(all_papers),
        )
        return _OpenAlexForwardEvidence(
            papers=tuple(all_papers),
            next_ordinal=ordinal,
            request_failures=request_failures,
            parse_failures=parse_failures,
            reported_total=reported_total,
            cursor_cycle=cursor_cycle,
            error_messages=tuple(errors),
        )

    @staticmethod
    def _merge_native_results(
        source: str, outcomes: Sequence[NativeBatchResult]
    ) -> NativeBatchResult:
        """Merge canonically ordered exact-request outcomes without losing counters."""
        return NativeBatchResult(
            source=source,
            papers=tuple(paper for outcome in outcomes for paper in outcome.papers),
            requested=sum(outcome.requested for outcome in outcomes),
            not_found=sum(outcome.not_found for outcome in outcomes),
            parse_failures=sum(outcome.parse_failures for outcome in outcomes),
            truncated_edge_records=sum(outcome.truncated_edge_records for outcome in outcomes),
            request_failures=sum(outcome.request_failures for outcome in outcomes),
            edge_parse_failures=sum(outcome.edge_parse_failures for outcome in outcomes),
            edge_request_failures=sum(outcome.edge_request_failures for outcome in outcomes),
            error_messages=tuple(
                message for outcome in outcomes for message in outcome.error_messages
            ),
        )

    @staticmethod
    def _cache_directions(source: str, *, include_edges: bool, direction: str) -> tuple[str, ...]:
        """Return independently fresh graph directions required from ``source``."""
        if not include_edges:
            return ()
        directions: list[str] = []
        if direction in {"both", "backward"} and source in {
            "crossref",
            "openalex",
            "semantic_scholar",
        }:
            directions.append("backward")
        if direction in {"both", "forward"} and source in _FORWARD_CAPABLE_DATABASES:
            directions.append("forward")
        return tuple(directions)

    def _cache_request_shape(self, source: str, direction: str) -> str:
        """Describe provider evidence whose payloads are interchangeable."""
        return canonical_json(
            {
                "schema": _SNAPSHOT_REQUEST_SHAPE,
                "source": source,
                "direction": direction,
                "max_cited_by": self._max_cited_by if direction == "forward" else None,
            }
        )

    @staticmethod
    def _source_native_kind(source: str) -> IdentifierKind | None:
        """Return the namespace that is native to one graph source."""
        return {
            "semantic_scholar": IdentifierKind.SEMANTIC_SCHOLAR,
            "openalex": IdentifierKind.OPENALEX,
            "crossref": IdentifierKind.DOI,
            "arxiv": IdentifierKind.ARXIV,
        }.get(source)

    @classmethod
    def _direct_native_key(cls, source: str, target: CitationTarget) -> str | None:
        """Return a target's explicit source-native identity, if present."""
        kind = cls._source_native_kind(source)
        if kind is None or (value := target.get(kind)) is None:
            return None
        return PaperIdentifier(kind, value).key

    def _cache_native_key(self, source: str, target: CitationTarget) -> str | None:
        """Resolve one target without choosing through a conflicting alias."""
        if direct := self._direct_native_key(source, target):
            return direct
        if self._snapshot_cache is None:
            return None
        resolved: set[str] = set()
        for alias in sorted(target.keys):
            resolution = self._snapshot_cache.resolve_alias(source, alias)
            if resolution.conflict:
                return None
            resolved.update(resolution.native_keys)
        return next(iter(resolved)) if len(resolved) == 1 else None

    @staticmethod
    def _directional_paper(paper: Paper, source: str, direction: str) -> Paper:
        """Project a provider record to one independently fresh edge direction."""
        payload = paper.to_dict()
        counts = paper.citation_edge_totals.get(source, {})
        if direction == "backward":
            payload["cited_by"] = []
            payload["cited_by_edges"] = []
            key = "references"
        else:
            payload["references"] = []
            payload["reference_edges"] = []
            key = "citations"
        payload["citation_edge_totals"] = {source: {key: counts[key]}} if key in counts else {}
        return Paper.from_dict(payload)

    def _project_native_outcome(
        self,
        outcome: NativeBatchResult,
        *,
        include_edges: bool,
    ) -> NativeBatchResult:
        """Make cold provider records use the same directional form as warm hits."""
        if self._snapshot_cache is None:
            return outcome
        directions = self._cache_directions(
            outcome.source,
            include_edges=include_edges,
            direction=self._direction,
        )
        if not directions:
            return outcome
        projected: list[Paper] = []
        for paper in outcome.papers:
            value = self._directional_paper(paper, outcome.source, directions[0])
            for direction in directions[1:]:
                value.merge(self._directional_paper(paper, outcome.source, direction))
            projected.append(value)
        return NativeBatchResult(
            source=outcome.source,
            papers=tuple(projected),
            requested=outcome.requested,
            not_found=outcome.not_found,
            parse_failures=outcome.parse_failures,
            truncated_edge_records=outcome.truncated_edge_records,
            request_failures=outcome.request_failures,
            edge_parse_failures=outcome.edge_parse_failures,
            edge_request_failures=outcome.edge_request_failures,
            error_messages=outcome.error_messages,
        )

    def _read_cached_target(
        self,
        source: str,
        target: CitationTarget,
        *,
        include_edges: bool,
    ) -> Paper | None:
        """Read only a fresh, complete set of required source directions."""
        if self._snapshot_cache is None:
            return None
        directions = self._cache_directions(
            source,
            include_edges=include_edges,
            direction=self._direction,
        )
        if not directions or (native_key := self._cache_native_key(source, target)) is None:
            return None

        papers: list[Paper] = []
        for direction in directions:
            snapshot = self._snapshot_cache.get_graph_snapshot(
                source=source,
                native_key=native_key,
                direction=direction,
                request_shape=self._cache_request_shape(source, direction),
                allow_stale=False,
                require_complete=True,
            )
            if snapshot is None or not snapshot.satisfies_complete:
                return None
            payload = snapshot.payload
            if not isinstance(payload, dict) or payload.get("schema") != _SNAPSHOT_PAYLOAD_SCHEMA:
                return None
            if (
                payload.get("source") != source
                or payload.get("native_key") != native_key
                or payload.get("direction") != direction
                or not isinstance(payload.get("paper"), dict)
            ):
                return None
            try:
                paper = Paper.from_dict(payload["paper"])
            except (KeyError, TypeError, ValueError):
                return None
            if not self._alias_groups_match(target.keys, self._paper_aliases(paper)):
                return None
            papers.append(paper)

        combined = papers[0]
        for paper in papers[1:]:
            if not same_record_identity(combined, paper):
                return None
            combined.merge(paper)
        return combined

    @staticmethod
    def _edge_counts(paper: Paper, source: str, direction: str) -> tuple[int | None, int]:
        """Return provider-reported and actually materialized edge counts."""
        count_key = "references" if direction == "backward" else "citations"
        reported = paper.citation_edge_totals.get(source, {}).get(count_key)
        edges = paper.reference_edges if direction == "backward" else paper.cited_by_edges
        returned = sum(1 for edge in edges if edge.source == source)
        return reported, returned

    @staticmethod
    def _snapshot_quality(reported: int | None, returned: int) -> SnapshotQuality:
        """Classify one direction without turning a lower bound into closure."""
        if reported is None:
            return SnapshotQuality.PARTIAL
        if returned < reported:
            return SnapshotQuality.TRUNCATED
        if returned == reported:
            return SnapshotQuality.COMPLETE
        return SnapshotQuality.PARTIAL

    def _write_native_snapshots(
        self,
        source: str,
        papers: Sequence[Paper],
        *,
        include_edges: bool,
    ) -> None:
        """Offer source observations to cache without endangering traversal."""
        if self._snapshot_cache is None:
            return
        directions = self._cache_directions(
            source,
            include_edges=include_edges,
            direction=self._direction,
        )
        if not directions:
            return
        native_kind = self._source_native_kind(source)
        if native_kind is None:
            return
        for paper in papers:
            native_identifier = next(
                (identifier for identifier in paper.identifiers if identifier.kind is native_kind),
                None,
            )
            if native_identifier is None:
                continue
            native_key = native_identifier.key
            aliases = tuple(sorted(identifier.key for identifier in paper.identifiers))
            for direction in directions:
                directional = self._directional_paper(paper, source, direction)
                reported, returned = self._edge_counts(directional, source, direction)
                quality = self._snapshot_quality(reported, returned)
                payload = {
                    "schema": _SNAPSHOT_PAYLOAD_SCHEMA,
                    "source": source,
                    "native_key": native_key,
                    "direction": direction,
                    "paper": directional.to_dict(),
                }
                try:
                    result = self._snapshot_cache.put_graph_snapshot(
                        source=source,
                        native_key=native_key,
                        direction=direction,
                        request_shape=self._cache_request_shape(source, direction),
                        payload=payload,
                        quality=quality,
                        reported_count=reported,
                        returned_count=returned,
                        ttl_seconds=_SNAPSHOT_TTL_SECONDS,
                        aliases=aliases,
                    )
                except Exception as exc:
                    logger.warning(
                        "Could not cache successful %s %s snapshot for %s: %s",
                        source,
                        direction,
                        native_key,
                        exc,
                    )
                    continue
                if not result.stored:
                    logger.info(
                        "Snapshot cache refused %s %s for %s: %s",
                        source,
                        direction,
                        native_key,
                        result.reason or "managed-data budget",
                    )

    @classmethod
    def _combine_cached_and_network(
        cls,
        source: str,
        targets: Sequence[CitationTarget],
        cached: Sequence[tuple[CitationTarget, Paper]],
        network: NativeBatchResult,
        *,
        requested: int,
        unroutable: int,
    ) -> NativeBatchResult:
        """Restore positional accounting after resolving only cache misses remotely."""
        cached_remaining = list(cached)
        network_remaining = list(network.papers)
        ordered: list[Paper] = []
        for target in targets:
            cached_index = next(
                (index for index, (item, _) in enumerate(cached_remaining) if item is target),
                None,
            )
            if cached_index is not None:
                _, paper = cached_remaining.pop(cached_index)
                ordered.append(paper)
                continue
            network_index = next(
                (
                    index
                    for index, paper in enumerate(network_remaining)
                    if cls._alias_groups_match(target.keys, cls._paper_aliases(paper))
                ),
                None,
            )
            if network_index is not None:
                ordered.append(network_remaining.pop(network_index))
        ordered.extend(paper for _, paper in cached_remaining)
        ordered.extend(network_remaining)
        return NativeBatchResult(
            source=source,
            papers=tuple(ordered),
            requested=requested,
            not_found=network.not_found,
            parse_failures=network.parse_failures,
            truncated_edge_records=network.truncated_edge_records,
            request_failures=network.request_failures + unroutable,
            edge_parse_failures=network.edge_parse_failures,
            edge_request_failures=network.edge_request_failures,
            error_messages=(
                *network.error_messages,
                *(
                    (
                        f"{unroutable} asserted edge target(s) had no identifier accepted by "
                        f"{source}",
                    )
                    if unroutable
                    else ()
                ),
            ),
        )

    def _fetch_native_source(
        self,
        source: str,
        targets: Sequence[CitationTarget],
        *,
        include_edges: bool,
    ) -> NativeBatchResult:
        """Fetch cache misses through exact provider-request/page units."""
        if source == "semantic_scholar":
            s2_routable = sorted(
                (
                    (canonical_json(sorted(target.keys)), identifier)
                    for target in targets
                    if (identifier := self._identifier_for_source(target, source)) is not None
                ),
                key=lambda item: item[0],
            )
            semantic_connector = SemanticScholarConnector(api_key=self._semantic_scholar_api_key)
            try:
                include_references = include_edges and self._direction in ("both", "backward")
                include_citations = include_edges and self._direction in ("both", "forward")
                max_batch = getattr(semantic_connector, "max_paper_batch_size", None)
                batch_limit = (
                    max_batch(include_edges=include_edges)
                    if callable(max_batch)
                    else max(1, len(s2_routable))
                )
                exact_fetch = getattr(semantic_connector, "fetch_papers_batch_request", None)
                if exact_fetch is None:
                    exact_fetch = semantic_connector.fetch_papers_batch
                outcomes: list[NativeBatchResult] = []
                for chunk_index, start in enumerate(range(0, len(s2_routable), batch_limit)):
                    chunk = s2_routable[start : start + batch_limit]
                    alias_groups = [alias_group for alias_group, _ in chunk]
                    identifiers = [identifier for _, identifier in chunk]
                    identifier_tuple = tuple(identifiers)

                    def _fetch_semantic_batch(
                        identifier_tuple: tuple[str, ...] = identifier_tuple,
                    ) -> NativeBatchResult:
                        result = exact_fetch(
                            list(identifier_tuple),
                            include_edges=include_edges,
                            include_references=include_references,
                            include_citations=include_citations,
                        )
                        if not isinstance(result, NativeBatchResult):
                            raise TypeError("Semantic Scholar exact batch returned invalid result")
                        return result

                    outcome = self._execute_native_request(
                        source=source,
                        request_kind="paper_batch",
                        targets=alias_groups,
                        ordinal=chunk_index,
                        semantic_context={
                            "request_schema": "semantic_scholar_paper_batch_v1",
                            "chunk_index": chunk_index,
                            "batch_limit": batch_limit,
                            "identifier_order": identifiers,
                            "include_edges": include_edges,
                            "include_references": include_references,
                            "include_citations": include_citations,
                        },
                        fetch=_fetch_semantic_batch,
                    )
                    outcomes.append(outcome)
                return self._merge_native_results(source, outcomes)
            finally:
                semantic_connector.close()
        if source == "openalex":
            native_targets = sorted(
                (
                    (canonical_json(sorted(target.keys)), identifier)
                    for target in targets
                    if (identifier := self._identifier_for_source(target, source)) is not None
                ),
                key=lambda item: item[0],
            )
            doi_fallbacks = sorted(
                (
                    (canonical_json(sorted(target.keys)), doi)
                    for target in targets
                    if target.get(IdentifierKind.OPENALEX) is None
                    for doi in (target.get(IdentifierKind.DOI),)
                    if doi is not None
                ),
                key=lambda item: item[0],
            )
            openalex_connector = OpenAlexConnector(
                api_key=self._openalex_api_key, email=self._email
            )
            try:
                include_forward = include_edges and self._direction in ("both", "forward")
                batch_limit = openalex_connector.max_paper_batch_size()
                resolution_outcomes: list[NativeBatchResult] = []
                request_ordinal = 0
                for chunk_index, start in enumerate(range(0, len(native_targets), batch_limit)):
                    chunk = native_targets[start : start + batch_limit]
                    alias_groups = [alias_group for alias_group, _ in chunk]
                    identifiers = [identifier for _, identifier in chunk]
                    identifier_tuple = tuple(identifiers)

                    def _fetch_openalex_batch(
                        identifier_tuple: tuple[str, ...] = identifier_tuple,
                    ) -> NativeBatchResult:
                        return openalex_connector.fetch_papers_batch_request(
                            list(identifier_tuple),
                            include_forward=include_forward,
                        )

                    resolution_outcomes.append(
                        self._execute_native_request(
                            source=source,
                            request_kind="works_by_ids",
                            targets=alias_groups,
                            ordinal=request_ordinal,
                            semantic_context={
                                "request_schema": "openalex_works_by_ids_v1",
                                "chunk_index": chunk_index,
                                "batch_limit": batch_limit,
                                "identifier_order": identifiers,
                                "include_forward": include_forward,
                            },
                            fetch=_fetch_openalex_batch,
                        )
                    )
                    request_ordinal += 1

                for fallback_index, (alias_group, doi) in enumerate(doi_fallbacks):

                    def _fetch_openalex_doi(doi: str = doi) -> NativeBatchResult:
                        try:
                            with auditing(source) as audit:
                                paper = openalex_connector.fetch_paper_by_doi(doi)
                        except Exception as exc:
                            return NativeBatchResult(
                                source=source,
                                papers=(),
                                requested=1,
                                request_failures=1,
                                error_messages=(f"{doi}: {type(exc).__name__}: {exc}",),
                            )
                        if audit.requests_failed:
                            if all(item.status_code == 404 for item in audit.requests_failed):
                                return NativeBatchResult(
                                    source=source, papers=(), requested=1, not_found=1
                                )
                            return NativeBatchResult(
                                source=source,
                                papers=(),
                                requested=1,
                                request_failures=1,
                                error_messages=tuple(
                                    f"{doi}: {item.describe()}" for item in audit.requests_failed
                                ),
                            )
                        if paper is None:
                            return NativeBatchResult(
                                source=source,
                                papers=(),
                                requested=1,
                                parse_failures=1,
                                error_messages=(
                                    f"OpenAlex DOI lookup for {doi} returned no record or failed",
                                ),
                            )
                        return NativeBatchResult(source=source, papers=(paper,), requested=1)

                    resolution_outcomes.append(
                        self._execute_native_request(
                            source=source,
                            request_kind="work_by_doi",
                            targets=[alias_group],
                            ordinal=request_ordinal,
                            semantic_context={
                                "request_schema": "openalex_work_by_doi_v1",
                                "fallback_index": fallback_index,
                                "doi": doi.strip().lower(),
                            },
                            fetch=_fetch_openalex_doi,
                        )
                    )
                    request_ordinal += 1

                outcome = self._merge_native_results(source, resolution_outcomes)
                source_papers = list(outcome.papers)
                omitted_forward = outcome.truncated_edge_records
                forward_request_failures = 0
                forward_parse_failures = 0
                forward_errors: list[str] = []
                if include_edges and self._direction in ("both", "forward"):
                    omitted_forward = 0
                    for paper_index, paper in enumerate(sorted(source_papers, key=self._paper_key)):
                        forward = self._fetch_openalex_cited_by_resumable(
                            openalex_connector,
                            paper,
                            ordinal=request_ordinal,
                            paper_index=paper_index,
                        )
                        request_ordinal = forward.next_ordinal
                        forward_request_failures += forward.request_failures
                        forward_parse_failures += forward.parse_failures
                        forward_errors.extend(forward.error_messages)
                        for citing_paper in forward.papers:
                            if citing_paper.identifiers:
                                paper.cited_by_edges.append(
                                    CitationEdge(
                                        target=citing_paper.citation_target(),
                                        source="openalex",
                                    )
                                )
                        expected = (
                            forward.reported_total
                            if forward.reported_total is not None
                            else paper.citation_edge_totals.get("openalex", {}).get("citations")
                        )
                        if isinstance(expected, int):
                            omitted_forward += max(0, expected - len(forward.papers))
                        else:
                            omitted_forward += forward.parse_failures
                        if forward.cursor_cycle and not omitted_forward:
                            # The provider published no usable total, so the
                            # exact missing count is unknowable. Preserve a
                            # lower-bound truncation signal rather than zero.
                            omitted_forward += 1
                return NativeBatchResult(
                    source="openalex",
                    papers=tuple(source_papers),
                    requested=outcome.requested,
                    not_found=outcome.not_found,
                    parse_failures=outcome.parse_failures,
                    truncated_edge_records=omitted_forward,
                    request_failures=outcome.request_failures,
                    edge_parse_failures=(outcome.edge_parse_failures + forward_parse_failures),
                    edge_request_failures=(
                        outcome.edge_request_failures + forward_request_failures
                    ),
                    error_messages=(*outcome.error_messages, *forward_errors),
                )
            finally:
                openalex_connector.close()
        if source == "arxiv":
            arxiv_routable: list[tuple[str, str]] = []
            for target in targets:
                arxiv_id = target.get(IdentifierKind.ARXIV)
                target_doi = target.get(IdentifierKind.DOI)
                if arxiv_id:
                    arxiv_routable.append((canonical_json(sorted(target.keys)), arxiv_id))
                elif target_doi and target_doi.lower().startswith("10.48550/arxiv."):
                    arxiv_routable.append(
                        (
                            canonical_json(sorted(target.keys)),
                            target_doi[len("10.48550/arxiv.") :],
                        )
                    )
            arxiv_routable.sort(key=lambda item: item[0])
            arxiv_connector = ArxivConnector()
            try:
                batch_limit = arxiv_connector.max_paper_batch_size()
                outcomes = []
                for chunk_index, start in enumerate(range(0, len(arxiv_routable), batch_limit)):
                    chunk = arxiv_routable[start : start + batch_limit]
                    alias_groups = [alias_group for alias_group, _ in chunk]
                    identifiers = [identifier for _, identifier in chunk]
                    identifier_tuple = tuple(identifiers)

                    def _fetch_arxiv_batch(
                        identifier_tuple: tuple[str, ...] = identifier_tuple,
                    ) -> NativeBatchResult:
                        return arxiv_connector.fetch_papers_batch_request(list(identifier_tuple))

                    outcomes.append(
                        self._execute_native_request(
                            source=source,
                            request_kind="id_list",
                            targets=alias_groups,
                            ordinal=chunk_index,
                            semantic_context={
                                "request_schema": "arxiv_id_list_v1",
                                "chunk_index": chunk_index,
                                "batch_limit": batch_limit,
                                "identifier_order": identifiers,
                            },
                            fetch=_fetch_arxiv_batch,
                        )
                    )
                return self._merge_native_results(source, outcomes)
            finally:
                arxiv_connector.close()
        if source == "crossref":
            crossref_routable = sorted(
                (
                    (canonical_json(sorted(target.keys)), doi)
                    for target in targets
                    for doi in (target.get(IdentifierKind.DOI),)
                    if doi is not None
                ),
                key=lambda item: item[0],
            )
            crossref_connector = CrossRefConnector(email=self._email)
            try:
                outcomes = []
                for request_index, (alias_group, doi) in enumerate(crossref_routable):

                    def _fetch_crossref_doi(doi: str = doi) -> NativeBatchResult:
                        try:
                            with auditing(source) as audit:
                                paper = crossref_connector.fetch_paper_by_doi(doi)
                        except Exception as exc:
                            return NativeBatchResult(
                                source=source,
                                papers=(),
                                requested=1,
                                request_failures=1,
                                error_messages=(f"{doi}: {type(exc).__name__}: {exc}",),
                            )
                        if audit.requests_failed and all(
                            item.status_code == 404 for item in audit.requests_failed
                        ):
                            return NativeBatchResult(
                                source=source, papers=(), requested=1, not_found=1
                            )
                        if audit.requests_failed:
                            return NativeBatchResult(
                                source=source,
                                papers=(),
                                requested=1,
                                request_failures=1,
                                error_messages=tuple(
                                    f"{doi}: {item.describe()}" for item in audit.requests_failed
                                ),
                            )
                        if paper is None:
                            return NativeBatchResult(
                                source=source,
                                papers=(),
                                requested=1,
                                parse_failures=1,
                                error_messages=(
                                    f"Crossref DOI lookup for {doi} returned an unreadable record",
                                ),
                            )
                        return NativeBatchResult(source=source, papers=(paper,), requested=1)

                    outcomes.append(
                        self._execute_native_request(
                            source=source,
                            request_kind="work_by_doi",
                            targets=[alias_group],
                            ordinal=request_index,
                            semantic_context={
                                "request_schema": "crossref_work_by_doi_v1",
                                "request_index": request_index,
                                "doi": doi.strip().lower(),
                            },
                            fetch=_fetch_crossref_doi,
                        )
                    )
                return self._merge_native_results(source, outcomes)
            finally:
                crossref_connector.close()
        return NativeBatchResult(
            source=source,
            papers=(),
            requested=len(targets),
            request_failures=len(targets),
            error_messages=(
                f"unsupported citation-edge source {source!r}; no connector route was attempted",
            ),
        )

    def acquire(
        self,
        batches: tuple[RouteBatch, ...],
        *,
        intent: AcquisitionIntent,
    ) -> Sequence[Paper]:
        """Implement the flow acquisition port with the existing runtime.

        Route batches retain assertion provenance, while provider execution is
        coalesced by execution source exactly as before the extraction.  The
        fragile request/page replay, aggregate commit, cache-write, and local
        reduction ordering below is intentionally unchanged.
        """
        include_edges = intent is AcquisitionIntent.TRAVERSAL_EDGES
        grouped: dict[str, list[CitationTarget]] = {}
        for batch in batches:
            source_targets = grouped.setdefault(batch.execution_source, [])
            for target in batch.targets:
                if not any(
                    self._alias_groups_match(target.keys, existing.keys)
                    for existing in source_targets
                ):
                    source_targets.append(target)

        papers: list[Paper] = []
        source_groups: list[tuple[str, list[CitationTarget]]] = [
            (source, list(targets)) for source, targets in grouped.items()
        ]
        if self._run_state is not None:
            source_groups = sorted(source_groups, key=lambda item: item[0])
        for source_ordinal, (source, unsorted_targets) in enumerate(source_groups):
            routed_targets = unsorted_targets
            if self._run_state is not None:
                routed_targets = sorted(
                    unsorted_targets,
                    key=lambda target: canonical_json(sorted(target.keys)),
                )
            if source == "legacy_doi":
                dois = [
                    doi
                    for target in routed_targets
                    if (doi := target.get(IdentifierKind.DOI)) is not None
                ]
                papers.extend(
                    self._fetch_dois(
                        dois,
                        databases=["crossref"],
                        desc="Legacy DOI discovery",
                        verbose=False,
                        show_progress=False,
                        coverage_source="legacy_doi",
                    )
                )
                continue

            unit_id, replayed = self._plan_native_operation(
                source=source,
                targets=routed_targets,
                include_edges=include_edges,
                ordinal=source_ordinal,
            )
            if replayed is not None:
                outcome = replayed
                network_papers_to_cache: tuple[Paper, ...] = ()
            else:
                cached: list[tuple[CitationTarget, Paper]] = []
                misses: list[CitationTarget] = []
                for target in routed_targets:
                    paper = self._read_cached_target(
                        source,
                        target,
                        include_edges=include_edges,
                    )
                    if paper is None:
                        misses.append(target)
                    else:
                        cached.append((target, paper))
                if misses:
                    network = self._fetch_native_source(
                        source,
                        misses,
                        include_edges=include_edges,
                    )
                    network = self._project_native_outcome(network, include_edges=include_edges)
                    network_papers_to_cache = network.papers
                else:
                    network = NativeBatchResult(source=source, papers=(), requested=0)
                    network_papers_to_cache = ()
                outcome = self._combine_cached_and_network(
                    source,
                    routed_targets,
                    cached,
                    network,
                    requested=len(routed_targets),
                    unroutable=len(misses) - network.requested,
                )
            self._commit_native_operation(unit_id, outcome)
            # The aggregate is durable before cache mutation.  A crash during
            # snapshot writes therefore replays run evidence rather than
            # replanning different cache-miss chunks.
            if network_papers_to_cache:
                self._write_native_snapshots(
                    source,
                    network_papers_to_cache,
                    include_edges=include_edges,
                )
            self._record_coverage(outcome)
            papers.extend(outcome.papers)
            self._commit_native_reduction(unit_id, outcome)
        return self._coalesce_papers(papers)

    def _fetch_targets(
        self, edges: list[CitationEdge], *, include_edges: bool = True
    ) -> list[Paper]:
        """Compatibility adapter for tests and seed acquisition using edges."""
        assertions = tuple(
            TraversalAssertion(
                target=edge.target,
                target_key=canonical_json(sorted(edge.target.keys)),
                aliases=tuple(sorted(edge.target.keys)),
                level=1,
                direction="backward",
                asserting_source=edge.source,
                parent_key="compatibility-edge-input",
                parent_aliases=(),
            )
            for edge in edges
        )
        plan = TraversalLevelPlan(
            level=1,
            assertions=assertions,
            reservations=tuple(dict.fromkeys(assertion.route_key for assertion in assertions)),
        )
        batches = route_assertions(plan, canonical=self._run_state is not None)
        intent = (
            AcquisitionIntent.TRAVERSAL_EDGES if include_edges else AcquisitionIntent.PROJECTION
        )
        return list(self.acquire(batches, intent=intent))

    def run(self, verbose: bool = False, show_progress: bool = True) -> SnowballResult:
        """Execute the snowball and return the result.

        Can be called multiple times; each call is independent.

        Parameters
        ----------
        verbose : bool
            Enable verbose logging.
        show_progress : bool
            When ``True`` (default), display tqdm progress bars for each
            snowball level.  Set to ``False`` to suppress progress output.

        Returns
        -------
        SnowballResult
            Container with all discovered papers (including seeds).
            Citation relationships are encoded on each paper via
            :attr:`~lit_search.core.paper.Paper.references` and
            :attr:`~lit_search.core.paper.Paper.cited_by`.
        """
        _root_logger = logging.getLogger()
        _saved_log_level = _root_logger.level
        if verbose:
            configure_verbose_logging()

        logger.debug("=== SnowballRunner Configuration ===")
        logger.debug("Seed paper DOI: %s", self._seed_papers[0].doi)
        logger.debug("Max depth: %d", self._max_depth)
        logger.debug("Direction: %s", self._direction)
        logger.debug(
            "Top N per level: %s",
            str(self._max_papers_per_level) if self._max_papers_per_level else "unlimited",
        )
        logger.debug("Discovery databases: %s", self._databases or "all")
        logger.debug("Enrichment databases: %s", self._enrichment_databases or "all")
        logger.debug("Num workers: %d", self._num_workers)
        logger.debug("=====================================")

        start = perf_counter()
        self._coverage = {}
        self._policy_residuals = {}
        self._attempted_routes = {}
        self._visited_alias_groups = []
        self._lookup_coverages = []

        if self._run_state is not None:
            lifecycle = self._run_state.snapshot().lifecycle
            if lifecycle is not RunLifecycle.ACTIVE:
                raise RuntimeError(
                    f"run {self._run_state.run_id} cannot execute while lifecycle is "
                    f"{lifecycle.value}"
                )
            self._run_state.recover_in_flight()
            self._advance_run_phase(RunPhase.SEED_FETCH)
            self._operation_phase = RunPhase.SEED_FETCH
            self._operation_level = None

        # Every known alias is visited.  A DOI and an S2 ID for the same target
        # must not schedule two fetches merely because their strings differ.
        visited: set[str] = {
            identifier.key for paper in self._seed_papers for identifier in paper.identifiers
        }
        for paper in self._seed_papers:
            self._remember_visited_group(self._paper_aliases(paper))

        # Canonical paper objects keyed by collision-safe storage tokens.
        # Bibliographic identity is always checked explicitly before insertion.
        all_papers: dict[str, Paper] = {}

        # --- Level 0: fetch/enrich seed papers using full enrichment databases ---
        enriched_seeds = self._process_seed_papers(visited, verbose, show_progress)

        # Add enriched seeds to all_papers so they participate in the final
        # web-scraping pass and are available for any post-processing.
        seed_keys: set[str] = set()
        for seed in enriched_seeds:
            seed_keys.add(self._store_paper(all_papers, seed))
            aliases = self._paper_aliases(seed)
            visited.update(aliases)
            self._remember_visited_group(aliases)

        frontier: list[Paper] = enriched_seeds

        # --- BFS levels 1 ... max_depth ---
        for level in range(1, self._max_depth + 1):
            if not frontier:
                logger.debug("Level %d: empty frontier: stopping.", level)
                break

            self._advance_run_phase(RunPhase.LEVEL_PLAN, level=level)
            self._advance_run_phase(RunPhase.LEVEL_FETCH, level=level)
            self._operation_phase = RunPhase.LEVEL_FETCH
            self._operation_level = level
            frontier = self._process_bfs_level(
                level=level,
                frontier=frontier,
                visited=visited,
                all_papers=all_papers,
                verbose=verbose,
                show_progress=show_progress,
            )
            self._advance_run_phase(RunPhase.LEVEL_REDUCE, level=level)

        # --- Final enrichment of non-seed papers ---
        self._advance_run_phase(RunPhase.ENRICH_PLAN)
        self._advance_run_phase(RunPhase.ENRICH_FETCH)
        self._operation_phase = RunPhase.ENRICH_FETCH
        self._operation_level = None
        if all_papers:
            self._run_final_enrichment(
                all_papers, seed_keys, verbose=verbose, show_progress=show_progress
            )

        elapsed = perf_counter() - start

        logger.debug("=== Snowball Results ===")
        logger.debug("Total papers discovered: %d", len(all_papers))
        logger.debug("Runtime: %.2f s", elapsed)
        logger.debug("========================")

        _root_logger.setLevel(_saved_log_level)
        discovered_papers = [p for key, p in all_papers.items() if key not in seed_keys]
        coverage_entries = [item.to_coverage() for item in self._coverage.values()]
        coverage_entries.extend(
            item.to_coverage() for item in self._policy_residuals.values() if item.residual
        )
        coverage_entries.extend(
            entry for ledger in self._lookup_coverages for entry in ledger.entries
        )
        coverage_derivations = [
            note for ledger in self._lookup_coverages for note in ledger.derivations
        ]
        coverage = (
            CoverageLedger(
                entries=tuple(coverage_entries),
                derivations=tuple(coverage_derivations),
            )
            if coverage_entries or coverage_derivations
            else None
        )

        result = SnowballResult(
            seed_paper=enriched_seeds[0],
            max_depth=self._max_depth,
            direction=self._direction,
            since=self._since,
            until=self._until,
            databases=self._databases,
            max_papers_per_level=self._max_papers_per_level,
            max_expansion_per_level=self._max_expansion_per_level,
            papers=discovered_papers,
            processed_at=datetime.datetime.now(datetime.UTC),
            runtime_seconds=elapsed,
            enrichment_databases=list(self._enrichment_databases),
            max_cited_by=self._max_cited_by,
            seed_papers=enriched_seeds,
            coverage=coverage,
            run_id=self._run_state.run_id if self._run_state is not None else None,
        )
        if self._run_state is not None:
            self._advance_run_phase(RunPhase.FINALIZE)
            self._run_state.transition_lifecycle(
                RunLifecycle.COMPLETED,
                reason="snowball runner finished its declared depth and enrichment",
            )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_bfs_level(
        self,
        *,
        level: int,
        frontier: list[Paper],
        visited: set[str],
        all_papers: dict[str, Paper],
        verbose: bool,
        show_progress: bool,
    ) -> list[Paper]:
        """Run one BFS level: discover, filter, store, and enrich the next frontier.

        Parameters
        ----------
        level : int
            Current BFS level (1-based).
        frontier : list[Paper]
            Papers whose citation lists drive this level's expansion.
        visited : set[str]
            Normalised DOIs already scheduled or processed; mutated in place.
        all_papers : dict[str, Paper]
            Canonical result store; updated in place with new papers.
        verbose : bool
            Forwarded to each :meth:`GetRunner.run` call.
        show_progress : bool
            Whether to display a tqdm progress bar.

        Returns
        -------
        list[Paper]
            The frontier to use for the next BFS level.  Returns an empty list
            when there are no more candidates to expand.
        """
        plan = self._plan_frontier(frontier, visited=visited, level=level)
        self._reserve_plan(plan)

        logger.debug(
            "Level %d/%d: %d citation assertions (%d reserved routes) from %d frontier papers.",
            level,
            self._max_depth,
            len(plan.assertions),
            len(plan.reservations),
            len(frontier),
        )

        if not plan.assertions:
            return []

        batches = route_assertions(plan, canonical=self._run_state is not None)
        intent = (
            AcquisitionIntent.TRAVERSAL_EDGES
            if level < self._max_depth
            else AcquisitionIntent.PROJECTION
        )
        fetched = list(self.acquire(batches, intent=intent))
        policy = CurrentCitationCapPolicy(
            max_papers_per_level=self._max_papers_per_level,
            max_expansion_per_level=self._max_expansion_per_level,
        )
        reduction = reduce_level(
            plan,
            fetched,
            eligibility=self._matches_filters,
            policy=policy,
        )

        # A target becomes globally visited only after some route resolved it.
        # Failed routes remain eligible when a later edge asserts the same work
        # through a different source.  All alternate assertions collected for
        # this level have already been attempted by the time this executes.
        for paper in fetched:
            resolved_aliases = self._paper_aliases(paper)
            visited.update(resolved_aliases)
            self._remember_visited_group(resolved_aliases)
            for assertion in reduction.resolved_assertions:
                aliases = frozenset(assertion.aliases)
                if self._alias_groups_match(aliases, resolved_aliases):
                    visited.update(aliases)
                    self._remember_visited_group(aliases | resolved_aliases)

        # Same-source duplicates within one level should be coalesced, but a
        # failed route must not poison the target for the rest of the run. Drop
        # unresolved route reservations after the batch so a later frontier may
        # legitimately retry them. Resolved targets are protected by ``visited``.
        unresolved_routes = {assertion.route_key for assertion in reduction.unresolved_assertions}
        resolved_routes = {assertion.route_key for assertion in reduction.resolved_assertions}
        for source, route_aliases in unresolved_routes.difference(resolved_routes):
            attempted = self._attempted_routes.get(source)
            if attempted is not None:
                alias_group = frozenset(route_aliases)
                if alias_group in attempted:
                    attempted.remove(alias_group)

        if self._max_papers_per_level is not None:
            residual = self._policy_residuals.setdefault(
                "max_papers_per_level",
                _PolicyResidual("max_papers_per_level", self._max_papers_per_level),
            )
            residual.add(
                level=level,
                available=(
                    len(reduction.policy.result_members)
                    + len(reduction.policy.excluded_from_result)
                ),
            )

        for p in reduction.policy.result_members:
            self._store_paper(all_papers, p)

        valid_count = len(reduction.policy.result_members) + len(
            reduction.policy.excluded_from_result
        )

        logger.debug(
            "Level %d/%d complete: %d/%d papers passed date filters%s%s.",
            level,
            self._max_depth,
            valid_count,
            len(fetched),
            f" (top {self._max_papers_per_level} kept in result)"
            if self._max_papers_per_level
            else "",
            f" (top {self._max_expansion_per_level} used for expansion)"
            if self._max_expansion_per_level
            else "",
        )

        if self._max_expansion_per_level is not None and level < self._max_depth:
            residual = self._policy_residuals.setdefault(
                "max_expansion_per_level",
                _PolicyResidual("max_expansion_per_level", self._max_expansion_per_level),
            )
            residual.add(
                level=level,
                available=(
                    len(reduction.policy.expansion_members) + len(reduction.policy.unexpanded)
                ),
            )
        expansion_base = list(reduction.policy.expansion_members)

        # Re-enrich frontier papers before the next BFS round so that
        # paper.references and paper.cited_by are fully populated.
        # Skipped at the last level since there is no further expansion.
        needs_enrichment = [
            paper
            for paper in expansion_base
            if not any(source in paper.citation_edge_totals for source in self._databases)
        ]
        if level < self._max_depth and needs_enrichment:
            enriched_missing = self._enrich_frontier(
                needs_enrichment,
                level=level,
                all_papers=all_papers,
                verbose=verbose,
                show_progress=show_progress,
            )
            expansion_base = [
                next(
                    (
                        enriched
                        for enriched in enriched_missing
                        if same_record_identity(enriched, paper)
                    ),
                    paper,
                )
                for paper in expansion_base
            ]

        return expansion_base

    def _fetch_dois(
        self,
        dois: list[str],
        databases: list[str] | None,
        *,
        desc: str,
        verbose: bool,
        show_progress: bool,
        max_cited_by: int | None = None,
        coverage_source: str | None = None,
    ) -> list[Paper]:
        """Fetch papers for a list of DOIs using GetRunner.

        Each DOI is resolved via a separate
        :class:`~lit_search.runners.get_runner.GetRunner` call.  When
        *num_workers* is greater than 1, calls are made in parallel.

        Parameters
        ----------
        dois : list[str]
            DOI strings to fetch.
        databases : list[str] | None
            Database filter passed to each :class:`GetRunner`.  ``None``
            means all available sources.
        desc : str
            Human-readable label for the tqdm progress bar.
        verbose : bool
            Forwarded to each :meth:`GetRunner.run` call.
        show_progress : bool
            Whether to display a tqdm progress bar.
        max_cited_by : int | None
            Forwarded to :meth:`GetRunner.run` to cap cited-by pagination.
            ``None`` means no limit.
        coverage_source : str | None
            When set, record terminal accounting for this DOI batch under the
            supplied coverage source.  Used by legacy DOI-only traversal.

        Returns
        -------
        list[Paper]
            Successfully fetched papers. Empty lookups retain their coverage;
            only a completely covered empty lookup counts as explicit
            not-found evidence.
        """
        ieee_api_key = self._ieee_api_key
        scopus_api_key = self._scopus_api_key
        pubmed_api_key = self._pubmed_api_key
        openalex_api_key = self._openalex_api_key
        email = self._email
        semantic_scholar_api_key = self._semantic_scholar_api_key
        wos_api_key = self._wos_api_key
        proxy = self._proxy
        ssl_verify = self._ssl_verify

        def _fetch_task(doi: str) -> LookupResult | Paper | None:
            """Fetch a single paper by DOI via GetRunner.

            Parameters
            ----------
            doi : str
                DOI to fetch.

            Returns
            -------
            LookupResult | Paper | None
                Coverage-bearing lookup result.  Bare papers and ``None`` are
                accepted only for backward-compatible test doubles.
            """
            runner = GetRunner(
                identifier=doi,
                email=email,
                databases=databases,
                ieee_api_key=ieee_api_key,
                scopus_api_key=scopus_api_key,
                pubmed_api_key=pubmed_api_key,
                openalex_api_key=openalex_api_key,
                semantic_scholar_api_key=semantic_scholar_api_key,
                wos_api_key=wos_api_key,
                proxy=proxy,
                ssl_verify=ssl_verify,
            )
            return runner.run(verbose=verbose, max_cited_by=max_cited_by)

        if self._run_state is not None:
            return self._fetch_dois_resumable(
                dois,
                databases,
                desc=desc,
                show_progress=show_progress,
                max_cited_by=max_cited_by,
                coverage_source=coverage_source,
                fetch_task=_fetch_task,
            )

        results: list[Paper] = []
        not_found = 0
        request_failures = 0
        errors: list[str] = []
        for doi, result, error in execute_tasks(
            dois,
            _fetch_task,
            num_workers=self._num_workers,
            timeout=None,
            progress_total=len(dois),
            progress_unit="paper",
            progress_desc=desc,
            use_progress=show_progress,
        ):
            if error is not None:
                logger.warning("Failed to fetch DOI '%s': %s", doi, error)
                request_failures += 1
                errors.append(f"{doi}: {type(error).__name__}: {error}")
            elif isinstance(result, LookupResult):
                self._lookup_coverages.append(result.coverage)
                if result.paper is not None:
                    results.append(result.paper)
                elif result.coverage.is_complete:
                    not_found += 1
                else:
                    request_failures += 1
                    errors.append(
                        f"{doi}: lookup returned no paper with "
                        f"{result.coverage.status.value} coverage"
                    )
            elif result is not None:
                logger.warning("GetRunner returned a legacy bare Paper without lookup coverage")
                results.append(result)
            else:
                request_failures += 1
                errors.append(f"{doi}: legacy lookup returned no paper and no coverage evidence")
        if coverage_source is not None:
            self._record_coverage(
                NativeBatchResult(
                    source=coverage_source,
                    papers=tuple(results),
                    requested=len(dois),
                    not_found=not_found,
                    request_failures=request_failures,
                    error_messages=tuple(errors),
                )
            )
        return results

    def _fetch_dois_resumable(
        self,
        dois: Sequence[str],
        databases: list[str] | None,
        *,
        desc: str,
        show_progress: bool,
        max_cited_by: int | None,
        coverage_source: str | None,
        fetch_task: Callable[[str], LookupResult | Paper | None],
    ) -> list[Paper]:
        """Execute or replay canonical per-DOI GetRunner operations.

        Planning and state transitions stay on the caller thread. Worker
        threads only execute ``fetch_task``; their results are committed here.
        """
        if self._run_state is None:  # pragma: no cover - guarded by caller
            raise RuntimeError("resumable DOI fetch requires run state")
        canonical_dois = sorted({doi.strip().lower(): doi.strip() for doi in dois}.items())
        planned: list[tuple[str, str]] = []
        pending: list[tuple[str, str]] = []
        context = {
            "databases": sorted(databases) if databases is not None else None,
            "max_cited_by": max_cited_by,
            "role": desc,
            "coverage_source": coverage_source,
            "granularity": _RUN_REPLAY_GRANULARITY,
        }
        terminal_states = {
            WorkUnitState.SUCCEEDED,
            WorkUnitState.NOT_FOUND,
            WorkUnitState.FAILED,
            WorkUnitState.UNROUTABLE,
        }
        for ordinal, (normalised, original) in enumerate(canonical_dois):
            unit_id = self._run_state.plan_work_unit(
                phase=self._operation_phase,
                level=self._operation_level,
                source="get_runner",
                request_kind="doi_lookup",
                ordinal=ordinal,
                targets=[f"doi:{normalised}"],
                semantic_context=context,
            )
            planned.append((unit_id, original))
            if self._run_state.work_unit(unit_id).state in terminal_states:
                continue
            self._run_state.begin_work_unit(unit_id)
            pending.append((unit_id, original))

        def _execute(item: tuple[str, str]) -> LookupResult | Paper | None:
            return fetch_task(item[1])

        for item, result, error in execute_tasks(
            pending,
            _execute,
            num_workers=self._num_workers,
            timeout=None,
            progress_total=len(pending),
            progress_unit="paper",
            progress_desc=desc,
            use_progress=show_progress,
        ):
            unit_id, doi = item
            state, outcome, coverage = self._lookup_operation_outcome(doi, result, error)
            self._run_state.complete_work_unit(
                unit_id,
                state,
                outcome=outcome,
                coverage=coverage,
            )

        results: list[Paper] = []
        not_found = 0
        request_failures = 0
        errors: list[str] = []
        for unit_id, doi in planned:
            unit = self._run_state.work_unit(unit_id)
            if unit.outcome is None:
                raise RuntimeError(f"replayable DOI unit {unit_id} has no persisted outcome")
            kind = str(unit.outcome.get("kind"))
            paper: Paper | None = None
            if kind == "lookup_result":
                lookup = LookupResult.from_dict(dict(unit.outcome["value"]))
                self._lookup_coverages.append(lookup.coverage)
                paper = lookup.paper
                if paper is None and lookup.coverage.is_complete:
                    not_found += 1
                elif paper is None:
                    request_failures += 1
                    errors.append(
                        f"{doi}: lookup returned no paper with "
                        f"{lookup.coverage.status.value} coverage"
                    )
            elif kind == "paper":
                logger.warning("GetRunner returned a legacy bare Paper without lookup coverage")
                paper = Paper.from_dict(dict(unit.outcome["value"]))
            elif kind == "error":
                message = str(unit.outcome.get("message") or "provider operation failed")
                error_type = str(unit.outcome.get("error_type") or "Exception")
                logger.warning("Failed to fetch DOI '%s': %s", doi, message)
                request_failures += 1
                errors.append(f"{doi}: {error_type}: {message}")
            else:
                request_failures += 1
                errors.append(f"{doi}: legacy lookup returned no paper and no coverage evidence")
            if paper is not None:
                results.append(paper)
            self._run_state.commit_reduction(
                unit_id,
                "snowball_fetch_doi",
                {
                    "kind": kind,
                    "paper_key": self._paper_key(paper) if paper is not None else None,
                    "coverage_source": coverage_source,
                },
            )

        if coverage_source is not None:
            self._record_coverage(
                NativeBatchResult(
                    source=coverage_source,
                    papers=tuple(results),
                    requested=len(canonical_dois),
                    not_found=not_found,
                    request_failures=request_failures,
                    error_messages=tuple(errors),
                )
            )
        return results

    @staticmethod
    def _lookup_operation_outcome(
        doi: str,
        result: LookupResult | Paper | None,
        error: Exception | None,
    ) -> tuple[WorkUnitState, dict[str, Any], dict[str, Any]]:
        """Serialize one GetRunner outcome without losing its coverage."""
        if error is not None:
            outcome = {
                "kind": "error",
                "error_type": type(error).__name__,
                "message": str(error),
            }
            return WorkUnitState.FAILED, outcome, {"status": "failed", **outcome}
        if isinstance(result, LookupResult):
            state = (
                WorkUnitState.SUCCEEDED
                if result.paper is not None
                else WorkUnitState.NOT_FOUND
                if result.coverage.is_complete
                else WorkUnitState.FAILED
            )
            return (
                state,
                {"kind": "lookup_result", "value": result.to_dict()},
                result.coverage.to_dict(),
            )
        if isinstance(result, Paper):
            return (
                WorkUnitState.SUCCEEDED,
                {"kind": "paper", "value": result.to_dict()},
                {"status": "unknown", "reason": "legacy bare Paper"},
            )
        return (
            WorkUnitState.FAILED,
            {"kind": "none", "doi": doi},
            {"status": "unknown", "reason": "legacy None without coverage"},
        )

    def _enrich_frontier(
        self,
        papers: list[Paper],
        *,
        level: int,
        all_papers: dict[str, Paper],
        verbose: bool,
        show_progress: bool,
    ) -> list[Paper]:
        """Re-enrich frontier papers with the full combined set of databases.

        Papers that will drive the next BFS round are re-fetched using the
        union of *databases* and *enrichment_databases*, the same strategy
        applied to level-0 seeds.  This ensures that ``paper.references`` and
        ``paper.cited_by`` are fully populated before the next expansion.

        The enriched versions replace their entries in *all_papers* in place.

        Parameters
        ----------
        papers : list[Paper]
            Frontier papers to re-enrich.
        level : int
            Current BFS level (used in the progress bar label).
        all_papers : dict[str, Paper]
            Canonical result store; updated in place with enriched versions.
        verbose : bool
            Forwarded to each :meth:`GetRunner.run` call.
        show_progress : bool
            Whether to display a tqdm progress bar.

        Returns
        -------
        list[Paper]
            Enriched frontier papers.  Papers for which enrichment returned
            nothing fall back to their original (discovery-fetch) version.
        """
        dois = [p.doi for p in papers if p.doi]
        if not dois:
            return papers

        frontier_databases = sorted(set(self._databases + self._enrichment_databases))
        enriched_list = self._fetch_dois(
            dois,
            databases=frontier_databases,
            desc=f"Level {level}/{self._max_depth} [enrichment]",
            verbose=verbose,
            show_progress=show_progress,
            max_cited_by=self._max_cited_by,
        )
        enriched_map: dict[str, Paper] = {}
        for p in enriched_list:
            if p.doi:
                norm = p.doi.strip().lower()
                key = self._matching_paper_key(all_papers, p)
                if key is None:
                    # Expansion may include a paper intentionally excluded by
                    # max_papers_per_level.  Enrich it for traversal without
                    # re-inserting it into the result store.
                    enriched_map[norm] = p
                else:
                    all_papers[key].merge(p)
                    enriched_map[norm] = all_papers[key]

        return [enriched_map.get(p.doi.strip().lower(), p) for p in papers]  # type: ignore[union-attr]

    def _run_final_enrichment(
        self,
        all_papers: dict[str, Paper],
        seed_keys: set[str],
        *,
        verbose: bool,
        show_progress: bool,
    ) -> None:
        """Re-enrich non-seed papers with enrichment-only databases.

        After all BFS levels and filters are applied, papers that were
        discovered during BFS have only been fetched with the discovery
        databases.  This method enriches them with the databases declared in
        *enrichment_databases* that were **not** already used during
        discovery, avoiding redundant API calls.

        Seed papers are excluded because they were already enriched with the
        full combined set at the start.

        Parameters
        ----------
        all_papers : dict[str, Paper]
            Canonical result store; updated in place with enriched data.
        seed_keys : set[str]
            Collision-safe storage keys of seed papers to exclude from enrichment.
        verbose : bool
            Forwarded to each :meth:`GetRunner.run` call.
        show_progress : bool
            Whether to display a tqdm progress bar.
        """
        # Only use databases that haven't already been used in BFS discovery.
        discovery_dbs = set(self._databases)
        enrichment_only_dbs = [db for db in self._enrichment_databases if db not in discovery_dbs]
        if not enrichment_only_dbs:
            return

        non_seed_dois = [
            paper.doi
            for storage_key, paper in all_papers.items()
            if storage_key not in seed_keys and paper.doi
        ]
        if not non_seed_dois:
            return

        logger.debug(
            "Final enrichment pass: %d non-seed papers, databases: %s",
            len(non_seed_dois),
            enrichment_only_dbs,
        )
        enriched = self._fetch_dois(
            non_seed_dois,
            databases=enrichment_only_dbs,
            desc="Enrichment",
            verbose=verbose,
            show_progress=show_progress,
            max_cited_by=self._max_cited_by,
        )
        for enriched_paper in enriched:
            if enriched_paper.doi:
                key = self._matching_paper_key(all_papers, enriched_paper)
                if key is not None:
                    all_papers[key].merge(enriched_paper)
