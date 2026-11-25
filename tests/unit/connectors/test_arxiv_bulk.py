"""Unit tests for ArxivBulkConnector.

These run under ``pytest-socket`` with ``--disable-socket``.  This connector
makes no HTTP request at all, so the fixtures are snapshot files on disk.

The test carrying the most weight is
:meth:`TestCoverageIsHonest.test_failure_and_emptiness_are_different_statuses`:
everything else can pass while an unreadable snapshot still reports "arXiv holds
no matching paper", and that one cannot.
:meth:`TestCoverageIsHonest.test_a_full_scan_is_never_complete` is the second,
because a snapshot is stale by construction and a ``COMPLETE`` from it is the
motivating failure with the network removed.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from lit_search.connectors.arxiv_bulk import ArxivBulkConnector
from lit_search.core.coverage import CoverageStatus, CoveredPapers
from lit_search.core.paper import Database, PaperType
from lit_search.loaders.arxiv_snapshot import SNAPSHOT_PATH_ENV_VAR, write_manifest

from .conftest import _parse

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _raw(
    identifier: str,
    title: str,
    *,
    abstract: str = "A study of neural network robustness under distribution shift.",
    categories: str = "cs.LG",
    doi: str | None = None,
    created: str = "Thu, 17 Jun 2021 17:37:18 GMT",
) -> str:
    """Build one snapshot line.

    Parameters
    ----------
    identifier : str
        arXiv identifier.
    title : str
        Paper title.
    abstract : str
        Abstract text.
    categories : str
        Space-separated categories.
    doi : str | None
        Publisher DOI, when the record has one.
    created : str
        RFC 2822 timestamp for version 1.

    Returns
    -------
    str
        One JSON line.
    """
    record = {
        "id": identifier,
        "title": title,
        "abstract": abstract,
        "authors": "A Researcher",
        "authors_parsed": [["Researcher", "A", ""]],
        "categories": categories,
        "journal-ref": "",
        "comments": "8 pages",
        "versions": [{"version": "v1", "created": created}],
        "update_date": "2021-06-18",
    }
    if doi is not None:
        record["doi"] = doi
    return json.dumps(record)


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    """Write a three-record snapshot and return its path."""
    path = tmp_path / "arxiv-metadata-oai-snapshot.json"
    path.write_text(
        "\n".join(
            [
                _raw("2106.09685", "Low-Rank Adaptation of Large Language Models"),
                _raw(
                    "1706.03762",
                    "Attention Is All You Need",
                    categories="cs.CL cs.LG",
                    doi="10.5555/3295222.3295349",
                ),
                _raw("2301.00001", "Elliptic Regularity", categories="math.AP"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_manifest(
        str(path),
        version_number=301,
        version_created_at="2026-08-29T23:51:40.527Z",
    )
    return path


@pytest.fixture(autouse=True)
def _no_ambient_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a developer's real snapshot out of these tests."""
    monkeypatch.delenv(SNAPSHOT_PATH_ENV_VAR, raising=False)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


class TestCoverageIsHonest:
    """What the connector says about how complete its answer is."""

    def test_failure_and_emptiness_are_different_statuses(self, tmp_path: Path) -> None:
        """The regression test for this connector.

        An unreadable snapshot and a query that matched nothing must not
        produce the same status.  If they ever do, an outage is being reported
        as arXiv holding no matching paper.
        """
        good = tmp_path / "good.json"
        good.write_text(_raw("2106.09685", "Low-Rank Adaptation") + "\n", encoding="utf-8")

        empty = ArxivBulkConnector(snapshot_path=str(good)).search(
            _parse("[nothing whatsoever matches this]")
        )
        broken = ArxivBulkConnector(snapshot_path=str(tmp_path / "absent.json")).search(
            _parse("[low-rank]")
        )

        assert empty.status is CoverageStatus.PARTIAL
        assert empty.papers == ()
        assert broken.status is CoverageStatus.FAILED
        assert empty.status is not broken.status

    def test_a_full_scan_is_never_complete(self, snapshot: Path) -> None:
        """Even reading every record end to end does not justify COMPLETE.

        The snapshot publishes no total, and its own date bounds what it can
        answer.  ``COMPLETE`` would claim arXiv was exhausted.
        """
        result = ArxivBulkConnector(snapshot_path=str(snapshot), categories=frozenset()).search(
            _parse("[a]")
        )
        assert result.status is CoverageStatus.PARTIAL

    def test_the_detail_names_the_snapshot_version_and_date(self, snapshot: Path) -> None:
        """Provenance reaches the coverage ledger, not just the log."""
        result = ArxivBulkConnector(snapshot_path=str(snapshot)).search(_parse("[adaptation]"))
        detail = result.coverage.detail or ""
        assert "Cornell-University/arxiv" in detail
        assert "version 301" in detail
        assert "2026-08-29T23:51:40.527Z" in detail
        assert "posted or revised after that date is absent" in detail

    def test_without_a_manifest_the_detail_refuses_to_call_it_a_snapshot_date(
        self, tmp_path: Path
    ) -> None:
        """A file mtime is labelled as one, so nobody reads it as a cutoff."""
        path = tmp_path / "snap.json"
        path.write_text(_raw("2106.09685", "Low-Rank Adaptation") + "\n", encoding="utf-8")
        result = ArxivBulkConnector(snapshot_path=str(path)).search(_parse("[adaptation]"))
        detail = result.coverage.detail or ""
        assert "an unrecorded version" in detail
        assert "modification time" in detail

    def test_the_category_restriction_is_reported(self, snapshot: Path) -> None:
        """A recall boundary the caller did not choose has to be visible."""
        result = ArxivBulkConnector(snapshot_path=str(snapshot)).search(_parse("[a]"))
        detail = result.coverage.detail or ""
        assert "restricted to arXiv categories" in detail
        assert "cs.LG" in detail

    def test_no_snapshot_configured_is_skipped_not_empty(self) -> None:
        """An unconfigured corpus is a skip that names what to configure."""
        result = ArxivBulkConnector().search(_parse("[adaptation]"))
        assert result.status is CoverageStatus.SKIPPED_NO_CREDENTIAL
        assert result.coverage.missing_credential == SNAPSHOT_PATH_ENV_VAR

    def test_the_cap_is_reported_and_does_not_become_capped(self, snapshot: Path) -> None:
        """A cap is named in the detail, and the status stays PARTIAL.

        ``CAPPED`` ranks milder than ``PARTIAL``, so using it here would make a
        truncated scan of a stale corpus read as healthier than an untruncated
        one.
        """
        result = ArxivBulkConnector(snapshot_path=str(snapshot), categories=frozenset()).search(
            _parse("[a]"), max_papers=1
        )
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 1
        assert "stopped after 1 papers" in (result.coverage.detail or "")

    def test_unreadable_lines_are_reported_as_a_loss(self, tmp_path: Path) -> None:
        """A line the scan could not read is a coverage loss, not an absence."""
        path = tmp_path / "snap.json"
        path.write_text(
            "\n".join([_raw("2106.09685", "Low-Rank Adaptation"), "{ not json"]) + "\n",
            encoding="utf-8",
        )
        result = ArxivBulkConnector(snapshot_path=str(path)).search(_parse("[adaptation]"))
        assert "could not be read as records" in (result.coverage.detail or "")

    def test_search_returns_the_contract_type(self, snapshot: Path) -> None:
        """The connector cannot hand back a bare list."""
        assert isinstance(
            ArxivBulkConnector(snapshot_path=str(snapshot)).search(_parse("[adaptation]")),
            CoveredPapers,
        )


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


class TestRecords:
    """What the papers themselves carry."""

    def test_both_doi_identities_reach_the_paper(self, snapshot: Path) -> None:
        """The published DOI is citable and the arXiv DOI is kept beside it."""
        result = ArxivBulkConnector(snapshot_path=str(snapshot)).search(_parse("[attention]"))
        (paper,) = result.papers
        assert paper.doi == "10.5555/3295222.3295349"
        assert paper.preprint_doi == "10.48550/arXiv.1706.03762"
        assert paper.paper_type is PaperType.ARTICLE

    def test_an_unpublished_record_carries_one_identity(self, snapshot: Path) -> None:
        """With no journal DOI the arXiv DOI is primary and not duplicated."""
        result = ArxivBulkConnector(snapshot_path=str(snapshot)).search(_parse("[low-rank]"))
        (paper,) = result.papers
        assert paper.doi == "10.48550/arXiv.2106.09685"
        assert paper.preprint_doi is None
        assert paper.paper_type is PaperType.UNPUBLISHED

    def test_found_in_records_the_bulk_source_not_the_api_one(self, snapshot: Path) -> None:
        """The snapshot is its own database, so provenance says which answered."""
        result = ArxivBulkConnector(snapshot_path=str(snapshot)).search(_parse("[low-rank]"))
        (paper,) = result.papers
        assert paper.found_in == {Database.ARXIV_BULK.value}
        assert Database.ARXIV.value not in paper.found_in

    def test_categories_become_subjects_and_are_searchable(self, snapshot: Path) -> None:
        """arXiv categories are the corpus's controlled vocabulary."""
        result = ArxivBulkConnector(snapshot_path=str(snapshot)).search(_parse("key[cs.CL]"))
        assert [p.title for p in result.papers] == ["Attention Is All You Need"]
        assert "cs.CL" in result.papers[0].subjects


# ---------------------------------------------------------------------------
# Query handling
# ---------------------------------------------------------------------------


class TestQueryHandling:
    """Matching, date bounds, and rejected filters."""

    def test_boolean_and_not_is_honoured(self, snapshot: Path) -> None:
        """Local matching supports the full grammar, including AND NOT."""
        result = ArxivBulkConnector(snapshot_path=str(snapshot), categories=frozenset()).search(
            _parse("[robustness] AND NOT [attention]")
        )
        assert [p.title for p in result.papers] == [
            "Low-Rank Adaptation of Large Language Models",
            "Elliptic Regularity",
        ]

    def test_date_bounds_filter_on_the_submission_date(self, tmp_path: Path) -> None:
        """The bound is applied to v1's date, which is what the dump carries."""
        path = tmp_path / "snap.json"
        path.write_text(
            "\n".join(
                [
                    _raw("2106.09685", "Older", created="Thu, 17 Jun 2021 17:37:18 GMT"),
                    _raw("2401.00002", "Newer", created="Mon, 1 Jan 2024 00:00:00 GMT"),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        result = ArxivBulkConnector(snapshot_path=str(path)).search(
            _parse("[robustness]"), since=datetime.date(2023, 1, 1)
        )
        assert [p.title for p in result.papers] == ["Newer"]
        assert "date range was applied" in (result.coverage.detail or "")

    def test_an_unanswerable_filter_is_rejected_rather_than_widened(self, snapshot: Path) -> None:
        """No affiliation exists in the dump, so ``[aff]`` is QUERY_UNSUPPORTED."""
        result = ArxivBulkConnector(snapshot_path=str(snapshot)).search(_parse("aff[MIT]"))
        assert result.status is CoverageStatus.QUERY_UNSUPPORTED
        assert result.papers == ()
