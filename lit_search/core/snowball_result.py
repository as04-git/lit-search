"""SnowballResult: container for snowballing metadata and discovered papers."""

from __future__ import annotations

import contextlib
import datetime
from typing import Any, Literal

from ..utils.version import package_version
from .coverage import CoverageLedger
from .identity import same_record_identity
from .paper import Paper
from .rendering import (
    coverage_view,
    render_coverage_report,
    render_result,
    result_repr,
    result_summary,
)


class SnowballResult:
    """Represents a snowball configuration and its discovered papers.

    Holds the seed-set reference, traversal settings, and the flat list of
    :class:`~lit_search.core.paper.Paper` objects *discovered* during the BFS
    traversal. Seeds are **not** included in :attr:`papers`: they are
    accessible via :attr:`seed_papers`, with :attr:`seed_paper` retaining the
    first for compatibility. Citation relationships are encoded
    directly on each paper via
    :attr:`~lit_search.core.paper.Paper.references` and
    :attr:`~lit_search.core.paper.Paper.cited_by`.

    Parameters
    ----------
    seed_paper : Paper
        Enriched seed paper (fetched via GetRunner; original used as fallback
        when lookup fails).  It is **not** duplicated in :attr:`papers`.
    max_depth : int
        Maximum BFS traversal depth used.
    direction : Literal["both", "backward", "forward"]
        Snowball direction(s) used.
    since : datetime.date | None
        Lower-bound publication date filter applied to discovered papers.
    until : datetime.date | None
        Upper-bound publication date filter applied to discovered papers.
    databases : list[str] | None
        Database identifiers passed to
        :class:`~lit_search.runners.get_runner.GetRunner` for each lookup.
        ``None`` means all available databases were used.
    max_papers_per_level : int | None
        Per-level result cap.  Only the *top N* most-cited papers discovered
        at each level are kept in the final result.  Seed papers are never
        filtered.  ``None`` means no cap was applied.
    max_expansion_per_level : int | None
        Per-level frontier cap.  Only the *top N* most-cited papers from each
        level are used as seeds for the next BFS round.  All papers already
        added to the result are unaffected.  ``None`` means no cap.
    papers : list[Paper] | None
        Papers *discovered* during BFS traversal (seed excluded).
        ``None`` is treated as an empty list.
    processed_at : datetime.datetime | None
        Timestamp at which the snowball was executed.  Defaults to
        :func:`datetime.datetime.now`.
    runtime_seconds : float | None
        Wall-clock runtime of the snowball pipeline.
    seed_papers : list[Paper] | None
        Full enriched seed set. Defaults to ``[seed_paper]``.
    coverage : CoverageLedger | None
        Per-source edge-fetch coverage. ``None`` only for a legacy path that
        produced no auditable evidence.
    run_id : str | None
        Operational run identifier when the traversal used managed checkpoint
        state. This references mutable resume state; it is not a result handle.
    """

    def __init__(
        self,
        seed_paper: Paper,
        max_depth: int,
        direction: Literal["both", "backward", "forward"],
        since: datetime.date | None = None,
        until: datetime.date | None = None,
        databases: list[str] | None = None,
        max_papers_per_level: int | None = None,
        max_expansion_per_level: int | None = None,
        papers: list[Paper] | None = None,
        processed_at: datetime.datetime | None = None,
        runtime_seconds: float | None = None,
        enrichment_databases: list[str] | None = None,
        max_cited_by: int | None = None,
        seed_papers: list[Paper] | None = None,
        coverage: CoverageLedger | None = None,
        run_id: str | None = None,
    ) -> None:
        """Create a SnowballResult instance.

        Parameters
        ----------
        seed_paper : Paper
            Enriched seed paper used as the BFS starting point.
        max_depth : int
            Maximum BFS traversal depth.
        direction : Literal["both", "backward", "forward"]
            Snowball direction(s).
        since : datetime.date | None
            Lower-bound publication date filter.
        until : datetime.date | None
            Upper-bound publication date filter.
        databases : list[str] | None
            Database identifiers used for each lookup.
        max_papers_per_level : int | None
            Per-level result cap (top-N most-cited papers kept per level).
        max_expansion_per_level : int | None
            Per-level frontier cap (top-N most-cited papers used as next-level seeds).
        papers : list[Paper] | None
            Discovered papers (seed excluded).
        processed_at : datetime.datetime | None
            Execution timestamp.  Defaults to now (UTC).
        runtime_seconds : float | None
            Wall-clock runtime.
        enrichment_databases : list[str] | None
            Databases used for the final post-BFS enrichment pass on discovered
            papers.
        max_cited_by : int | None
            Maximum number of citing-paper DOIs collected per paper during
            forward snowball expansion.
        run_id : str | None
            Managed operational run identifier, when present.
        """
        self.seed_paper: Paper = seed_paper
        self.seed_papers: list[Paper] = list(seed_papers or [seed_paper])
        self.max_depth = max_depth
        self.direction = direction
        self.since = since
        self.until = until
        self.databases = databases
        self.max_papers_per_level = max_papers_per_level
        self.max_expansion_per_level = max_expansion_per_level
        self.papers: list[Paper] = list(papers) if papers is not None else []
        self.runtime_seconds = runtime_seconds
        self.enrichment_databases = enrichment_databases
        self.max_cited_by = max_cited_by
        self.coverage = coverage
        self.run_id = run_id

        if processed_at is None:
            processed_at = datetime.datetime.now(datetime.UTC)
        if processed_at.tzinfo is None:
            processed_at = processed_at.replace(tzinfo=datetime.UTC)
        self.processed_at = processed_at

    #: Explanation used only when a legacy traversal could not produce a
    #: ledger. Native source traversal sets :attr:`coverage` instead.
    coverage_unavailable_reason = (
        "this snowball used a legacy DOI-only lookup path with no auditable edge-fetch "
        "outcome, so an empty frontier is not distinguishable from a failed lookup"
    )

    def headline(self) -> str:
        """Return the one-line description of the traversal.

        Returns
        -------
        str
            Seed, direction, depth and paper count.
        """
        seed = self.seed_paper.doi or self.seed_paper.title or "(unidentified seed)"
        seed_label = seed if len(self.seed_papers) == 1 else f"{len(self.seed_papers)} seed papers"
        return (
            f"snowball {self.direction} depth {self.max_depth} from {seed_label} "
            f"-- {len(self.papers)} papers discovered"
        )

    @property
    def all_papers(self) -> list[Paper]:
        """Return seeds plus discoveries, coalesced only by explicit identity.

        Identifier-free records are kept unless the exact same object occurs
        twice.  A shared title is evidence for possible work equivalence, not
        a safe record identity.
        """
        combined: list[Paper] = []
        for paper in [*self.seed_papers, *self.papers]:
            if any(same_record_identity(paper, existing) for existing in combined):
                continue
            combined.append(paper)
        return combined

    def render_coverage(self) -> str:
        """Render this set's coverage as honest multi-line text.

        Native traversal renders its per-source ledger. A legacy DOI-only
        traversal reports UNKNOWN and names that gap.

        Returns
        -------
        str
            Coverage report.
        """
        return render_coverage_report(coverage_view(self))

    def render(self, *, limit: int | None = 10) -> str:
        """Render the whole set: coverage, then papers, then coverage again.

        Parameters
        ----------
        limit : int | None
            Maximum number of papers to list.  ``None`` lists all of them.

        Returns
        -------
        str
            Multi-line report.
        """
        return render_result(self, headline=self.headline(), limit=limit)

    def __repr__(self) -> str:
        """Return a representation whose shape reveals incomplete coverage.

        Returns
        -------
        str
            Single-line ``repr``.
        """
        return result_repr(
            self,
            seed=self.seed_paper.doi,
            direction=self.direction,
            max_depth=self.max_depth,
            papers=len(self.papers),
        )

    def __str__(self) -> str:
        """Return the short display: coverage and a paper count, never papers.

        Returns
        -------
        str
            Multi-line summary.
        """
        return result_summary(self, headline=self.headline())

    def add_paper(self, paper: Paper) -> None:
        """Append a paper to the result.

        Parameters
        ----------
        paper : Paper
            Paper to add.

        Returns
        -------
        None
        """
        self.papers.append(paper)

    def remove_paper(self, paper: Paper) -> None:
        """Remove the first explicitly identical paper if present.

        Parameters
        ----------
        paper : Paper
            Paper to remove.

        Returns
        -------
        None
        """
        for index, existing in enumerate(self.papers):
            if same_record_identity(existing, paper):
                del self.papers[index]
                return

    def to_dict(self) -> dict[str, Any]:
        """Serialize the snowball result to a dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary with ``"metadata"`` and ``"papers"`` keys, suitable
            for JSON serialization.
        """
        return {
            "metadata": {
                "seed_paper": {"doi": self.seed_paper.doi, "title": self.seed_paper.title},
                "seed_papers": [paper.to_dict() for paper in self.seed_papers],
                "max_depth": self.max_depth,
                "direction": self.direction,
                "since": self.since.isoformat() if self.since else None,
                "until": self.until.isoformat() if self.until else None,
                "databases": self.databases,
                "enrichment_databases": self.enrichment_databases,
                "max_cited_by": self.max_cited_by,
                "max_papers_per_level": self.max_papers_per_level,
                "max_expansion_per_level": self.max_expansion_per_level,
                "timestamp": self.processed_at.astimezone(datetime.UTC).isoformat(),
                "version": package_version(),
                "runtime_seconds": self.runtime_seconds,
                "coverage": self.coverage.to_dict() if self.coverage is not None else None,
                "run_id": self.run_id,
            },
            "papers": [paper.to_dict() for paper in self.papers],
        }

    @classmethod
    def from_dict(cls, data: dict) -> SnowballResult:
        """Reconstruct a SnowballResult from a dictionary.

        Accepts the format produced by :meth:`to_dict`.

        Parameters
        ----------
        data : dict
            Dictionary with ``"metadata"`` and ``"papers"`` keys.

        Returns
        -------
        SnowballResult
            Reconstructed instance.
        """
        metadata = data.get("metadata", {})
        raw_papers = data.get("papers", [])

        processed_at: datetime.datetime | None = None
        ts = metadata.get("timestamp")
        if isinstance(ts, str):
            with contextlib.suppress(ValueError):
                processed_at = datetime.datetime.fromisoformat(ts)

        since: datetime.date | None = None
        since_str = metadata.get("since")
        if isinstance(since_str, str):
            with contextlib.suppress(ValueError):
                since = datetime.date.fromisoformat(since_str)

        until: datetime.date | None = None
        until_str = metadata.get("until")
        if isinstance(until_str, str):
            with contextlib.suppress(ValueError):
                until = datetime.date.fromisoformat(until_str)

        raw_seed_papers = metadata.get("seed_papers") or []
        seed_papers = [Paper.from_dict(item) for item in raw_seed_papers]
        if seed_papers:
            seed_paper = seed_papers[0]
        else:
            # Backward compatibility with the original single-seed summary.
            seed_data = metadata.get("seed_paper") or {}
            seed_paper = Paper(
                title=seed_data.get("title") or "(unknown seed paper)",
                abstract="",
                authors=[],
                source=None,
                publication_date=None,
                doi=seed_data.get("doi"),
            )
            seed_papers = [seed_paper]

        raw_coverage = metadata.get("coverage")
        coverage = (
            CoverageLedger.from_dict(raw_coverage) if isinstance(raw_coverage, dict) else None
        )

        return cls(
            seed_paper=seed_paper,
            max_depth=metadata.get("max_depth", 1),
            direction=metadata.get("direction", "both"),
            since=since,
            until=until,
            databases=metadata.get("databases"),
            max_papers_per_level=metadata.get("max_papers_per_level"),
            max_expansion_per_level=metadata.get("max_expansion_per_level"),
            papers=[Paper.from_dict(p) for p in raw_papers],
            processed_at=processed_at,
            runtime_seconds=metadata.get("runtime_seconds"),
            enrichment_databases=metadata.get("enrichment_databases"),
            max_cited_by=metadata.get("max_cited_by"),
            seed_papers=seed_papers,
            coverage=coverage,
            run_id=metadata.get("run_id"),
        )
