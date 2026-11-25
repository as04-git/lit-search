"""SimilarResult: container for a single-hop content-similarity lookup."""

from __future__ import annotations

import contextlib
import datetime
from typing import Any

from ..utils.version import package_version
from .identity import same_record_identity
from .paper import Paper
from .rendering import (
    coverage_view,
    render_coverage_report,
    render_result,
    result_repr,
    result_summary,
)


class SimilarResult:
    """Represents a content-similarity lookup around a single seed paper.

    Unlike :class:`~lit_search.core.snowball_result.SnowballResult`, which
    covers a multi-level BFS citation-graph traversal over one or more seed
    papers, this holds the result of a single-hop lookup around exactly one
    seed paper: papers judged topically/semantically similar by each queried
    source's own signal (Semantic Scholar SPECTER-based recommendations,
    PubMed's PMRA related-articles score, OpenAlex's ``related_works``).

    Parameters
    ----------
    seed_paper : Paper
        The paper the similarity lookup was performed around.
    databases : list[str] | None
        Sources that were requested, in priority order.  ``None`` means the
        default two sources were used (``["semantic_scholar", "pubmed"]``);
        OpenAlex is supported but excluded from the default because its
        ``related_works`` signal proved noisier in testing, so it only
        appears here when requested explicitly.
    max_papers_per_database : int | None
        Cap on the number of related papers requested/kept from each source
        before merging.  ``None`` means each source's own natural default
        was used.
    since : datetime.date | None
        Lower-bound publication date filter applied to related papers.
    until : datetime.date | None
        Upper-bound publication date filter applied to related papers.
    enrichment_databases : list[str] | None
        Databases used to enrich related papers after merging and filtering.
    max_cited_by : int | None
        Maximum number of citing-paper DOIs collected per paper during
        enrichment (only relevant when ``"openalex"`` or
        ``"semantic_scholar"`` are in *enrichment_databases*).
    papers : list[Paper] | None
        Related papers found, deduplicated and merged across sources.  The
        seed paper itself is never included.  ``None`` is treated as an
        empty list.
    processed_at : datetime.datetime | None
        Timestamp at which the lookup was executed.  Defaults to
        :func:`datetime.datetime.now` (UTC).
    runtime_seconds : float | None
        Wall-clock runtime of the lookup.
    failed_databases : list[str] | None
        Sources that were queried but raised an error.  ``None`` is treated
        as an empty list.
    skipped_databases : list[str] | None
        Sources that were not queried because they do not apply to this seed
        paper (e.g. PubMed when no PMID could be resolved).  Distinct from
        *failed_databases*: a skip is an expected outcome, not an error.
        ``None`` is treated as an empty list.
    """

    def __init__(
        self,
        seed_paper: Paper,
        databases: list[str] | None = None,
        max_papers_per_database: int | None = None,
        since: datetime.date | None = None,
        until: datetime.date | None = None,
        enrichment_databases: list[str] | None = None,
        max_cited_by: int | None = None,
        papers: list[Paper] | None = None,
        processed_at: datetime.datetime | None = None,
        runtime_seconds: float | None = None,
        failed_databases: list[str] | None = None,
        skipped_databases: list[str] | None = None,
    ) -> None:
        """Create a SimilarResult instance.

        Parameters
        ----------
        seed_paper : Paper
            The paper the similarity lookup was performed around.
        databases : list[str] | None
            Sources that were requested, in priority order.
        max_papers_per_database : int | None
            Cap on the number of related papers kept from each source.
        since : datetime.date | None
            Lower-bound publication date filter applied to related papers.
        until : datetime.date | None
            Upper-bound publication date filter applied to related papers.
        enrichment_databases : list[str] | None
            Databases used to enrich related papers after merging/filtering.
        max_cited_by : int | None
            Maximum number of citing-paper DOIs collected per paper during
            enrichment.
        papers : list[Paper] | None
            Related papers found (seed excluded).
        processed_at : datetime.datetime | None
            Execution timestamp.  Defaults to now (UTC).
        runtime_seconds : float | None
            Wall-clock runtime.
        failed_databases : list[str] | None
            Sources that were queried but raised an error.
        skipped_databases : list[str] | None
            Sources that were not applicable to this seed paper.
        """
        self.seed_paper = seed_paper
        self.databases = databases
        self.max_papers_per_database = max_papers_per_database
        self.since = since
        self.until = until
        self.enrichment_databases = enrichment_databases
        self.max_cited_by = max_cited_by
        self.papers: list[Paper] = list(papers) if papers is not None else []
        self.runtime_seconds = runtime_seconds
        self.failed_databases: list[str] = list(failed_databases) if failed_databases else []
        self.skipped_databases: list[str] = list(skipped_databases) if skipped_databases else []

        if processed_at is None:
            processed_at = datetime.datetime.now(datetime.UTC)
        if processed_at.tzinfo is None:
            processed_at = processed_at.replace(tzinfo=datetime.UTC)
        self.processed_at = processed_at

    #: Why this container carries no coverage ledger.  Read by
    #: :func:`~lit_search.core.rendering.coverage_view` so that every
    #: rendering states the specific gap rather than a generic absence.
    coverage_unavailable_reason = (
        "the similar runner records no coverage ledger (R14): only thrown "
        "errors reach failed_databases, so a source that returned a short or "
        "empty list reports the same as one that returned everything"
    )

    def headline(self) -> str:
        """Return the one-line description of the lookup.

        Returns
        -------
        str
            Seed, sources and paper count.
        """
        seed = self.seed_paper.doi or self.seed_paper.title or "(unidentified seed)"
        sources = ", ".join(self.databases) if self.databases else "default sources"
        return f"similar to {seed} via {sources} -- {len(self.papers)} papers"

    def render_coverage(self) -> str:
        """Render this set's coverage as honest multi-line text.

        The similar runner records no coverage ledger today; its
        ``failed_databases`` and ``skipped_databases`` lists are reported
        here so the gap is visible rather than implied.

        Returns
        -------
        str
            Coverage report, followed by the two legacy lists.
        """
        report = render_coverage_report(coverage_view(self))
        extra = [
            f"  failed sources : {', '.join(self.failed_databases) or 'none recorded'}",
            f"  skipped sources: {', '.join(self.skipped_databases) or 'none recorded'}",
        ]
        return "\n".join([report, *extra])

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
            papers=len(self.papers),
            failed=list(self.failed_databases),
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
        """Remove the first result with the same explicit record identity.

        A rehydrated/provider clone can therefore remove a stored record when
        they share an explicit DOI or provider-native identifier.  Title
        similarity is never used.  For identifier-free records, only the exact
        in-memory object matches.

        Parameters
        ----------
        paper : Paper
            Paper to remove.

        Returns
        -------
        None
        """
        for index, candidate in enumerate(self.papers):
            if same_record_identity(candidate, paper):
                del self.papers[index]
                break

    def to_dict(self) -> dict[str, Any]:
        """Serialize the similar result to a dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary with ``"metadata"`` and ``"papers"`` keys, suitable
            for JSON serialization.
        """
        return {
            "metadata": {
                "seed_paper": {"doi": self.seed_paper.doi, "title": self.seed_paper.title},
                "databases": self.databases,
                "max_papers_per_database": self.max_papers_per_database,
                "since": self.since.isoformat() if self.since else None,
                "until": self.until.isoformat() if self.until else None,
                "enrichment_databases": self.enrichment_databases,
                "max_cited_by": self.max_cited_by,
                "failed_databases": self.failed_databases,
                "skipped_databases": self.skipped_databases,
                "timestamp": self.processed_at.astimezone(datetime.UTC).isoformat(),
                "version": package_version(),
                "runtime_seconds": self.runtime_seconds,
            },
            "papers": [paper.to_dict() for paper in self.papers],
        }

    @classmethod
    def from_dict(cls, data: dict) -> SimilarResult:
        """Reconstruct a SimilarResult from a dictionary.

        Accepts the format produced by :meth:`to_dict`.

        Parameters
        ----------
        data : dict
            Dictionary with ``"metadata"`` and ``"papers"`` keys.

        Returns
        -------
        SimilarResult
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

        # Reconstruct a minimal seed Paper object from the stored summary.
        seed_data = metadata.get("seed_paper") or {}
        doi = seed_data.get("doi")
        title = seed_data.get("title") or ""
        seed_paper = Paper(
            title=title or "(unknown seed paper)",
            abstract="",
            authors=[],
            source=None,
            publication_date=None,
            doi=doi,
        )

        return cls(
            seed_paper=seed_paper,
            databases=metadata.get("databases"),
            max_papers_per_database=metadata.get("max_papers_per_database"),
            since=since,
            until=until,
            enrichment_databases=metadata.get("enrichment_databases"),
            max_cited_by=metadata.get("max_cited_by"),
            papers=[Paper.from_dict(p) for p in raw_papers],
            processed_at=processed_at,
            runtime_seconds=metadata.get("runtime_seconds"),
            failed_databases=metadata.get("failed_databases"),
            skipped_databases=metadata.get("skipped_databases"),
        )
