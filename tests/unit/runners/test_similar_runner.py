"""Unit tests for SimilarRunner."""

from __future__ import annotations

import datetime
from unittest.mock import patch

import pytest
import requests

from lit_search.core.citation import IdentifierKind, PaperIdentifier
from lit_search.core.paper import Paper
from lit_search.core.retrieval_audit import record_request_attempt, record_request_failure
from lit_search.exceptions import InvalidParameterError
from lit_search.runners.similar_runner import (
    DEFAULT_SIMILAR_DATABASES,
    SIMILAR_DATABASES,
    SimilarRunner,
)


@pytest.fixture(autouse=True)
def _no_enrichment():
    """Patch out enrichment so unit tests do not make real HTTP requests.

    Mirrors the equivalent fixture in ``test_search_runner.py``: SimilarRunner
    inherits ``DiscoveryRunner._enrich_papers``, and without this patch the
    default ``enrichment_databases=["crossref", "web_scraping"]`` would try
    real network access for every test.
    """
    with patch(
        "lit_search.runners.discovery_runner.DiscoveryRunner._enrich_papers",
        return_value=None,
    ):
        yield


def _make_paper(
    title: str = "Test",
    doi: str | None = "10.1000/seed",
    publication_date: datetime.date | None = datetime.date(2024, 1, 1),
) -> Paper:
    """Create a minimal Paper for testing.

    Parameters
    ----------
    title : str
        Paper title.
    doi : str | None
        Optional DOI.
    publication_date : datetime.date | None
        Optional publication date.

    Returns
    -------
    Paper
        A minimal paper instance.
    """
    return Paper(
        title=title,
        abstract="Abstract",
        authors=[],
        source=None,
        publication_date=publication_date,
        doi=doi,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestSimilarRunnerValidation:
    """Tests for the databases parameter validation."""

    def test_none_uses_default_databases(self) -> None:
        """None selects the default databases (semantic_scholar, pubmed), not openalex."""
        runner = SimilarRunner(paper=_make_paper())
        assert runner._active_databases == DEFAULT_SIMILAR_DATABASES

    def test_explicit_all_databases_includes_openalex(self) -> None:
        """Passing all three explicitly includes openalex, unlike the default."""
        runner = SimilarRunner(paper=_make_paper(), databases=SIMILAR_DATABASES)
        assert runner._active_databases == SIMILAR_DATABASES

    def test_empty_list_raises(self) -> None:
        """An empty databases list raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError):
            SimilarRunner(paper=_make_paper(), databases=[])

    def test_unknown_database_raises(self) -> None:
        """An unknown database name raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError):
            SimilarRunner(paper=_make_paper(), databases=["not-a-real-database"])

    def test_subset_preserves_priority_order(self) -> None:
        """A subset is reordered to match SIMILAR_DATABASES priority."""
        runner = SimilarRunner(paper=_make_paper(), databases=["openalex", "semantic_scholar"])
        assert runner._active_databases == ["semantic_scholar", "openalex"]


# ---------------------------------------------------------------------------
# DOI requirement
# ---------------------------------------------------------------------------


class TestSimilarRunnerNoDoi:
    """Tests for the seed-paper-without-DOI rejection."""

    def test_no_doi_raises_without_http_calls(self) -> None:
        """A seed paper with no DOI is rejected at construction time; no source is touched."""
        with (
            patch(
                "lit_search.connectors.semantic_scholar.SemanticScholarConnector.fetch_related"
            ) as mock_ss,
            patch("lit_search.connectors.openalex.OpenAlexConnector.fetch_related") as mock_oa,
            patch("lit_search.connectors.pubmed.PubmedConnector.fetch_related") as mock_pm,
            pytest.raises(InvalidParameterError),
        ):
            SimilarRunner(paper=_make_paper(doi=None), databases=SIMILAR_DATABASES)

        mock_ss.assert_not_called()
        mock_oa.assert_not_called()
        mock_pm.assert_not_called()


# ---------------------------------------------------------------------------
# Merge / dedup / provenance / ordering
# ---------------------------------------------------------------------------


class TestSimilarRunnerMerge:
    """Tests for cross-source merge, dedup, provenance, and priority order."""

    def test_merges_duplicate_across_sources_with_combined_provenance(self) -> None:
        """A paper found by two sources is merged into one, provenance from both."""
        seed = _make_paper(doi="10.1000/seed")
        shared = _make_paper("Shared Paper", doi="10.1000/shared")
        ss_only = _make_paper("SS Only", doi="10.1000/ss-only")

        runner = SimilarRunner(paper=seed, databases=["semantic_scholar", "openalex"])

        with (
            patch.object(
                runner._semantic_scholar,
                "fetch_related",
                return_value=[ss_only, shared],
            ),
            patch.object(
                runner._openalex,
                "fetch_related",
                return_value=[_make_paper("Shared Paper", doi="10.1000/shared")],
            ),
        ):
            result = runner.run(show_progress=False)

        by_doi = {p.doi: p for p in result.papers}
        assert set(by_doi) == {"10.1000/ss-only", "10.1000/shared"}
        assert by_doi["10.1000/shared"].found_in == {"semantic_scholar", "openalex"}
        assert by_doi["10.1000/ss-only"].found_in == {"semantic_scholar"}

    def test_identifier_free_same_title_candidates_both_survive(self) -> None:
        """A title fallback cannot erase a DOI-less recommendation."""
        seed = _make_paper(doi="10.1000/seed")
        first = _make_paper("Generic Result", doi=None)
        second = _make_paper("Generic Result", doi=None)
        first.abstract = "First"
        second.abstract = "Second"
        runner = SimilarRunner(paper=seed, databases=["semantic_scholar", "openalex"])

        with (
            patch.object(runner._semantic_scholar, "fetch_related", return_value=[first]),
            patch.object(runner._openalex, "fetch_related", return_value=[second]),
        ):
            result = runner.run(show_progress=False)

        assert len(result.papers) == 2
        assert [paper.abstract for paper in result.papers] == ["First", "Second"]

    def test_shared_native_identity_merges_doi_less_candidates(self) -> None:
        """An exact provider-native record alias remains a valid merge key."""
        seed = _make_paper(doi="10.1000/seed")
        identifier = PaperIdentifier(IdentifierKind.SEMANTIC_SCHOLAR, "S2-shared")
        first = Paper(
            title="Shared",
            abstract="",
            authors=[],
            source=None,
            publication_date=None,
            identifiers=[identifier],
        )
        second = Paper(
            title="Shared richer title",
            abstract="Richer",
            authors=[],
            source=None,
            publication_date=None,
            identifiers=[identifier],
        )
        runner = SimilarRunner(paper=seed, databases=["semantic_scholar", "openalex"])

        with (
            patch.object(runner._semantic_scholar, "fetch_related", return_value=[first]),
            patch.object(runner._openalex, "fetch_related", return_value=[second]),
        ):
            result = runner.run(show_progress=False)

        assert len(result.papers) == 1
        assert result.papers[0].abstract == "Richer"
        assert result.papers[0].found_in == {"semantic_scholar", "openalex"}

    def test_shared_aggregator_alias_does_not_hide_conflicting_dois(self) -> None:
        """Two version identities survive a contradictory aggregator alias."""
        seed = _make_paper(doi="10.1000/seed")
        identifier = PaperIdentifier(IdentifierKind.SEMANTIC_SCHOLAR, "S2-conflict")
        first = Paper(
            title="Shared",
            abstract="First",
            authors=[],
            source=None,
            publication_date=None,
            doi="10.1000/first",
            identifiers=[identifier],
        )
        second = Paper(
            title="Shared",
            abstract="Second",
            authors=[],
            source=None,
            publication_date=None,
            doi="10.1000/second",
            identifiers=[identifier],
        )
        runner = SimilarRunner(paper=seed, databases=["semantic_scholar", "openalex"])

        with (
            patch.object(runner._semantic_scholar, "fetch_related", return_value=[first]),
            patch.object(runner._openalex, "fetch_related", return_value=[second]),
        ):
            result = runner.run(show_progress=False)

        assert [paper.doi for paper in result.papers] == ["10.1000/first", "10.1000/second"]

    def test_priority_order_ss_first_then_pubmed_only_then_openalex_only(self) -> None:
        """Results follow source priority: SS, then PubMed-only, then OpenAlex-only."""
        seed = _make_paper(doi="10.1000/seed")
        ss_paper = _make_paper("From SS", doi="10.1000/ss")
        pm_paper = _make_paper("From PubMed", doi="10.1000/pm")
        oa_paper = _make_paper("From OpenAlex", doi="10.1000/oa")

        runner = SimilarRunner(paper=seed, databases=SIMILAR_DATABASES)

        with (
            patch.object(runner._semantic_scholar, "fetch_related", return_value=[ss_paper]),
            patch.object(runner._pubmed, "_resolve_pmid", return_value="12345"),
            patch.object(runner._pubmed, "fetch_related", return_value=[pm_paper]),
            patch.object(runner._openalex, "fetch_related", return_value=[oa_paper]),
        ):
            result = runner.run(show_progress=False)

        assert [p.doi for p in result.papers] == [
            "10.1000/ss",
            "10.1000/pm",
            "10.1000/oa",
        ]

    def test_excludes_seed_paper_when_echoed_back(self) -> None:
        """A source echoing the seed paper back does not add it to the result."""
        seed = _make_paper("Seed", doi="10.1000/seed")
        runner = SimilarRunner(paper=seed, databases=["semantic_scholar"])

        with patch.object(
            runner._semantic_scholar,
            "fetch_related",
            return_value=[_make_paper("Seed", doi="10.1000/seed")],
        ):
            result = runner.run(show_progress=False)

        assert result.papers == []


# ---------------------------------------------------------------------------
# PubMed skip vs failed
# ---------------------------------------------------------------------------


class TestSimilarRunnerPubmedSkipVsFailed:
    """Tests distinguishing PubMed inapplicability (skip) from errors (failed)."""

    def test_pubmed_skipped_when_pmid_not_resolvable(self) -> None:
        """No resolvable PMID marks PubMed as skipped, not failed."""
        seed = _make_paper(doi="10.1000/seed")
        runner = SimilarRunner(paper=seed, databases=["pubmed"])

        with (
            patch.object(runner._pubmed, "_resolve_pmid", return_value=None),
            patch.object(runner._pubmed, "fetch_related") as mock_fetch,
        ):
            result = runner.run(show_progress=False)

        assert result.skipped_databases == ["pubmed"]
        assert result.failed_databases == []
        mock_fetch.assert_not_called()

    def test_pubmed_resolution_transport_failure_is_failed_not_skipped(self) -> None:
        """A swallowed NCBI failure must not masquerade as DOI inapplicability."""
        seed = _make_paper(doi="10.1000/seed")
        runner = SimilarRunner(paper=seed, databases=["pubmed"])

        def _failed_resolution(_doi: str) -> None:
            error = requests.ConnectionError("NCBI unavailable")
            record_request_attempt()
            record_request_failure(error)
            return None

        with (
            patch.object(runner._pubmed, "_resolve_pmid", side_effect=_failed_resolution),
            patch.object(runner._pubmed, "fetch_related") as mock_fetch,
        ):
            result = runner.run(show_progress=False)

        assert result.failed_databases == ["pubmed"]
        assert result.skipped_databases == []
        mock_fetch.assert_not_called()


# ---------------------------------------------------------------------------
# Per-source error isolation
# ---------------------------------------------------------------------------


class TestSimilarRunnerErrorIsolation:
    """Tests that one source's failure does not abort the others."""

    def test_source_exception_is_isolated(self) -> None:
        """A source raising an exception is recorded as failed; others still run."""
        seed = _make_paper(doi="10.1000/seed")
        oa_paper = _make_paper("From OpenAlex", doi="10.1000/oa")
        runner = SimilarRunner(paper=seed, databases=["semantic_scholar", "openalex"])

        with (
            patch.object(
                runner._semantic_scholar, "fetch_related", side_effect=RuntimeError("boom")
            ),
            patch.object(runner._openalex, "fetch_related", return_value=[oa_paper]),
        ):
            result = runner.run(show_progress=False)

        assert result.failed_databases == ["semantic_scholar"]
        assert [p.doi for p in result.papers] == ["10.1000/oa"]

    def test_swallowed_source_request_failure_is_recorded(self) -> None:
        """A connector returning [] after a request failure is still failed."""
        seed = _make_paper(doi="10.1000/seed")
        runner = SimilarRunner(paper=seed, databases=["semantic_scholar"])

        def _swallowed_failure(_paper: Paper, _limit: int | None) -> list[Paper]:
            error = requests.ConnectionError("recommendations unavailable")
            record_request_attempt()
            record_request_failure(error)
            return []

        with patch.object(
            runner._semantic_scholar,
            "fetch_related",
            side_effect=_swallowed_failure,
        ):
            result = runner.run(show_progress=False)

        assert result.failed_databases == ["semantic_scholar"]
        assert result.skipped_databases == []


# ---------------------------------------------------------------------------
# max_papers_per_database forwarding
# ---------------------------------------------------------------------------


class TestSimilarRunnerMaxPapersForwarding:
    """Tests that max_papers_per_database is forwarded to each source."""

    def test_forwarded_to_each_active_connector(self) -> None:
        """Each connector's fetch_related receives the configured cap."""
        seed = _make_paper(doi="10.1000/seed")
        runner = SimilarRunner(
            paper=seed, databases=["semantic_scholar", "openalex"], max_papers_per_database=7
        )

        with (
            patch.object(runner._semantic_scholar, "fetch_related", return_value=[]) as mock_ss,
            patch.object(runner._openalex, "fetch_related", return_value=[]) as mock_oa,
        ):
            runner.run(show_progress=False)

        mock_ss.assert_called_once_with(seed, 7)
        mock_oa.assert_called_once_with(seed, 7)


# ---------------------------------------------------------------------------
# Date filtering (since/until), inherited from DiscoveryRunner
# ---------------------------------------------------------------------------


class TestSimilarRunnerDateFilter:
    """Tests for the post-fetch since/until filter, applied after merging.

    None of the three similarity sources supports native date filtering, so
    this always runs as a post-fetch pass, mirroring SearchRunner's own
    exact-boundary post-fetch filter.
    """

    def test_no_filter_returns_all_papers(self) -> None:
        """With since=None and until=None all merged papers are returned."""
        seed = _make_paper(doi="10.1000/seed")
        a = _make_paper("A", doi="10.1000/a", publication_date=datetime.date(2020, 1, 1))
        b = _make_paper("B", doi="10.1000/b", publication_date=datetime.date(2022, 6, 15))
        c = _make_paper("C", doi="10.1000/c", publication_date=None)
        runner = SimilarRunner(paper=seed, databases=["semantic_scholar"])

        with patch.object(runner._semantic_scholar, "fetch_related", return_value=[a, b, c]):
            result = runner.run(show_progress=False)

        assert len(result.papers) == 3

    def test_since_excludes_older_papers(self) -> None:
        """Papers published before `since` are removed after merging."""
        seed = _make_paper(doi="10.1000/seed")
        old = _make_paper("Old", doi="10.1000/old", publication_date=datetime.date(2021, 12, 31))
        new = _make_paper("New", doi="10.1000/new", publication_date=datetime.date(2022, 6, 1))
        runner = SimilarRunner(
            paper=seed, databases=["semantic_scholar"], since=datetime.date(2022, 1, 1)
        )

        with patch.object(runner._semantic_scholar, "fetch_related", return_value=[old, new]):
            result = runner.run(show_progress=False)

        assert len(result.papers) == 1
        assert result.papers[0].title == "New"

    def test_until_excludes_newer_papers(self) -> None:
        """Papers published after `until` are removed after merging."""
        seed = _make_paper(doi="10.1000/seed")
        old = _make_paper("Old", doi="10.1000/old", publication_date=datetime.date(2019, 3, 1))
        new = _make_paper("New", doi="10.1000/new", publication_date=datetime.date(2024, 1, 1))
        runner = SimilarRunner(
            paper=seed, databases=["semantic_scholar"], until=datetime.date(2020, 12, 31)
        )

        with patch.object(runner._semantic_scholar, "fetch_related", return_value=[old, new]):
            result = runner.run(show_progress=False)

        assert len(result.papers) == 1
        assert result.papers[0].title == "Old"

    def test_since_excludes_papers_with_no_date(self) -> None:
        """Papers without a publication date are excluded when since is set."""
        seed = _make_paper(doi="10.1000/seed")
        undated = _make_paper("Undated", doi="10.1000/undated", publication_date=None)
        runner = SimilarRunner(
            paper=seed, databases=["semantic_scholar"], since=datetime.date(2020, 1, 1)
        )

        with patch.object(runner._semantic_scholar, "fetch_related", return_value=[undated]):
            result = runner.run(show_progress=False)

        assert result.papers == []

    def test_result_metadata_records_since_until(self) -> None:
        """The since/until values used are recorded on the returned SimilarResult."""
        seed = _make_paper(doi="10.1000/seed")
        since = datetime.date(2020, 1, 1)
        until = datetime.date(2023, 12, 31)
        runner = SimilarRunner(paper=seed, databases=["semantic_scholar"], since=since, until=until)

        with patch.object(runner._semantic_scholar, "fetch_related", return_value=[]):
            result = runner.run(show_progress=False)

        assert result.since == since
        assert result.until == until


# ---------------------------------------------------------------------------
# Enrichment, inherited from DiscoveryRunner
# ---------------------------------------------------------------------------


class TestSimilarRunnerEnrichment:
    """Tests for the post-fetch, post-filter enrichment pass."""

    def test_enrichment_called_with_filtered_papers(self) -> None:
        """_enrich_papers is invoked with the merged (and filtered) paper list."""
        seed = _make_paper(doi="10.1000/seed")
        related = _make_paper("Related", doi="10.1000/related")
        runner = SimilarRunner(
            paper=seed, databases=["semantic_scholar"], max_cited_by=50, num_workers=3
        )

        with (
            patch.object(runner._semantic_scholar, "fetch_related", return_value=[related]),
            patch(
                "lit_search.runners.discovery_runner.DiscoveryRunner._enrich_papers"
            ) as mock_enrich,
        ):
            result = runner.run(show_progress=False)

        mock_enrich.assert_called_once()
        args, kwargs = mock_enrich.call_args
        assert args[0] == result.papers
        assert kwargs["num_workers"] == 3
        assert kwargs["max_cited_by"] == 50

    def test_enrichment_databases_empty_list_disables_enrichment(self) -> None:
        """enrichment_databases=[] disables the enrichment pass entirely."""
        seed = _make_paper(doi="10.1000/seed")
        related = _make_paper("Related", doi="10.1000/related")
        runner = SimilarRunner(paper=seed, databases=["semantic_scholar"], enrichment_databases=[])

        with (
            patch.object(runner._semantic_scholar, "fetch_related", return_value=[related]),
            patch(
                "lit_search.runners.discovery_runner.DiscoveryRunner._enrich_papers"
            ) as mock_enrich,
        ):
            runner.run(show_progress=False)

        mock_enrich.assert_not_called()

    def test_result_metadata_records_enrichment_databases(self) -> None:
        """The resolved enrichment_databases list is recorded on the result."""
        seed = _make_paper(doi="10.1000/seed")
        runner = SimilarRunner(
            paper=seed, databases=["semantic_scholar"], enrichment_databases=["crossref"]
        )

        with patch.object(runner._semantic_scholar, "fetch_related", return_value=[]):
            result = runner.run(show_progress=False)

        assert result.enrichment_databases == ["crossref"]

    def test_unknown_enrichment_database_raises(self) -> None:
        """An unknown enrichment database name raises InvalidParameterError."""
        seed = _make_paper(doi="10.1000/seed")
        with pytest.raises(InvalidParameterError):
            SimilarRunner(paper=seed, enrichment_databases=["not-a-real-database"])

    def test_max_cited_by_warning_when_citation_enrichment_unbounded(self, caplog) -> None:
        """A warning is logged when citation-capable enrichment has no cap."""
        seed = _make_paper(doi="10.1000/seed")
        with caplog.at_level("WARNING"):
            SimilarRunner(
                paper=seed,
                enrichment_databases=["openalex"],
                max_cited_by=None,
            )

        assert any("max_cited_by" in record.message for record in caplog.records)
