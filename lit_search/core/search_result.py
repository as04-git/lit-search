"""Search result container that aggregates papers from multiple databases."""

from __future__ import annotations

import contextlib
import datetime
from typing import Any

from ..utils.version import package_version
from .coverage import CoverageLedger, CoverageStatus, ExecutionStatus
from .identity import same_record_identity
from .paper import Paper
from .rendering import (
    CoverageView,
    coverage_view,
    render_coverage_report,
    render_result,
    result_repr,
    result_summary,
)


class SearchResult:
    """Represents a search configuration and results."""

    def __init__(
        self,
        query: str,
        since: datetime.date | None = None,
        until: datetime.date | None = None,
        max_papers_per_database: int | None = None,
        processed_at: datetime.datetime | None = None,
        databases: list[str] | None = None,
        papers: list[Paper] | None = None,
        runtime_seconds: float | None = None,
        runtime_seconds_per_database: dict[str, float] | None = None,
        failed_databases: list[str] | None = None,
        enrichment_databases: list[str] | None = None,
        max_cited_by: int | None = None,
        coverage: CoverageLedger | None = None,
    ) -> None:
        """Create a SearchResult instance.

        Parameters
        ----------
        query : str
            Search query.
        since : datetime.date | None
            Lower bound date.
        until : datetime.date | None
            Upper bound date.
        max_papers_per_database : int | None
            Maximum papers per database.
        processed_at : datetime.datetime | None
            Processing timestamp.
        databases : list[str] | None
            Database identifiers.
        papers : list[Paper] | None
            Initial papers.
        runtime_seconds : float | None
            Total runtime of the search pipeline.
        runtime_seconds_per_database : dict[str, float] | None
            Runtime in seconds for each database.
        failed_databases : list[str] | None
            Database identifiers that failed during search (network error,
            connector error, etc.).  ``None`` means the information was not
            recorded (e.g. loaded from an older save).
        enrichment_databases : list[str] | None
            Databases used for post-search paper enrichment.
        max_cited_by : int | None
            Maximum number of citing-paper DOIs collected per paper during
            enrichment.
        coverage : CoverageLedger | None
            Per-source retrieval coverage for this set (R7).  ``None`` means
            no coverage was recorded -- a result loaded from an older save,
            or one assembled by hand -- and is deliberately **not** the same
            as "everything was complete": :attr:`coverage_status` reports it
            as :data:`~lit_search.core.coverage.CoverageStatus.UNKNOWN`.

            This field supersedes *failed_databases* as the coverage record.
            *failed_databases* is kept because it is part of the existing
            save format, but it only ever recorded thrown errors, which is a
            strict subset of the ways a search can fall short.
        """
        self.query = query
        self.since = since
        self.until = until
        self.max_papers_per_database = max_papers_per_database
        processed_at = (
            processed_at if processed_at is not None else datetime.datetime.now(datetime.UTC)
        )
        if processed_at.tzinfo is None:
            processed_at = processed_at.replace(tzinfo=datetime.UTC)
        self.processed_at = processed_at
        self.databases = databases
        self.papers: list[Paper] = papers or []
        self.runtime_seconds = runtime_seconds
        self.runtime_seconds_per_database: dict[str, float] = dict(
            runtime_seconds_per_database or {}
        )
        self.failed_databases: list[str] = list(failed_databases or [])
        self.enrichment_databases = enrichment_databases
        self.max_cited_by = max_cited_by
        self.coverage: CoverageLedger = coverage if coverage is not None else CoverageLedger()

    @property
    def coverage_status(self) -> CoverageStatus:
        """Aggregate coverage of this result set.

        Returns
        -------
        CoverageStatus
            The worst per-source status recorded, or
            :data:`~lit_search.core.coverage.CoverageStatus.UNKNOWN` when no
            coverage was recorded at all.
        """
        return self.coverage.status

    @property
    def is_complete(self) -> bool:
        """Whether every source consulted was retrieved completely.

        Returns
        -------
        bool
            ``True`` only when :attr:`coverage_status` is
            :data:`~lit_search.core.coverage.CoverageStatus.COMPLETE`.
        """
        return self.coverage.is_complete

    @property
    def execution_status(self) -> ExecutionStatus:
        """Whether the declared search plan completed operationally."""
        return self.coverage.execution_status

    @property
    def execution_is_complete(self) -> bool:
        """Whether the declared search plan had no unplanned shortfall."""
        return self.coverage.execution_is_complete

    def render_coverage(self) -> str:
        """Render this set's coverage as honest multi-line text.

        An incomplete set is banner-wrapped and its shortfalls are repeated
        below the detail, so the output differs from a complete set's in
        shape rather than only in wording.

        Returns
        -------
        str
            Report naming every source consulted and every shortfall.
        """
        return render_coverage_report(self._coverage_view())

    def render(self, *, limit: int | None = 10) -> str:
        """Render the whole set: coverage, then papers, then coverage again.

        This is the full display used by the command line.  There is no
        variant that lists papers without coverage.

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

    def _coverage_view(self) -> CoverageView:
        """Return this set's coverage as a renderable view.

        Returns
        -------
        CoverageView
            View over :attr:`coverage`.
        """
        return coverage_view(self)

    def headline(self) -> str:
        """Return the one-line description of the operation.

        The databases named are the ones the **ledger** covers, not the ones
        that were successfully instantiated.  A database dropped for a missing
        credential never becomes a searcher, so :attr:`databases` does not
        list it -- and a headline reading "across arxiv" for a run the user
        asked to span three databases quietly narrows the question that was
        answered.  The ledger knows about every database that was asked for,
        reached or not, so it is what the headline counts.

        Returns
        -------
        str
            Query and database summary.
        """
        names = (
            list(self.coverage.scope.selected_sources)
            if self.coverage.scope is not None
            else list(self.coverage.sources) or list(self.databases or ())
        )
        databases = ", ".join(names) if names else "no database"
        return f"search {self.query!r} across {databases} -- {len(self.papers)} papers"

    def __repr__(self) -> str:
        """Return a representation whose shape reveals incomplete coverage.

        Returns
        -------
        str
            Single-line ``repr``.
        """
        return result_repr(self, query=self.query, papers=len(self.papers))

    def __str__(self) -> str:
        """Return the short display: coverage and a paper count, never papers.

        Returns
        -------
        str
            Multi-line summary.
        """
        return result_summary(self, headline=self.headline())

    def add_paper(self, paper: Paper) -> None:
        """Add a paper to the results.

        Parameters
        ----------
        paper : Paper
            Paper to add.
        """
        self.papers.append(paper)

    def remove_paper(self, paper: Paper) -> None:
        """Remove the first result with the same explicit record identity.

        A rehydrated/provider clone can therefore remove a stored record when
        they share an explicit DOI or provider-native identifier.  Title
        similarity is never used.  For identifier-free records, only the exact
        in-memory object matches.

        Parameters
        ----------
        paper : Paper
            Paper to remove.
        """
        for index, candidate in enumerate(self.papers):
            if same_record_identity(candidate, paper):
                del self.papers[index]
                break

    def to_dict(self) -> dict[str, Any]:
        """Serialize search to a dictionary representation.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the search.
        """
        metadata = {
            "query": self.query,
            "since": self.since.isoformat() if self.since else None,
            "until": self.until.isoformat() if self.until else None,
            "databases": self.databases,
            "max_papers_per_database": self.max_papers_per_database,
            "enrichment_databases": self.enrichment_databases,
            "max_cited_by": self.max_cited_by,
            "timestamp": self.processed_at.astimezone(datetime.UTC).isoformat(),
            "version": package_version(),
            "runtime_seconds": self.runtime_seconds,
            "runtime_seconds_per_database": dict(self.runtime_seconds_per_database),
            "failed_databases": self.failed_databases,
            "coverage": self.coverage.to_dict(),
        }
        return {
            "metadata": metadata,
            "papers": [paper.to_dict() for paper in self.papers],
        }

    @classmethod
    def from_dict(cls, data: dict) -> SearchResult:
        """Reconstruct a SearchResult from a dictionary.

        Accepts the format produced by :meth:`to_dict` (and by
        :func:`~lit_search.utils.persistence.save_to_json`).

        Parameters
        ----------
        data : dict
            Dictionary with ``"metadata"`` and ``"papers"`` keys.

        Returns
        -------
        SearchResult
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

        return cls(
            query=metadata.get("query", ""),
            since=since,
            until=until,
            databases=metadata.get("databases"),
            max_papers_per_database=metadata.get("max_papers_per_database"),
            processed_at=processed_at,
            papers=[Paper.from_dict(p) for p in raw_papers],
            runtime_seconds=metadata.get("runtime_seconds"),
            runtime_seconds_per_database=metadata.get("runtime_seconds_per_database"),
            failed_databases=metadata.get("failed_databases"),
            enrichment_databases=metadata.get("enrichment_databases"),
            max_cited_by=metadata.get("max_cited_by"),
            coverage=(
                CoverageLedger.from_dict(metadata["coverage"])
                if isinstance(metadata.get("coverage"), dict)
                else None
            ),
        )
