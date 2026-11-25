"""Unit tests for the Kaggle arXiv snapshot loader.

These run under ``pytest-socket`` with ``--disable-socket``; nothing here
touches the network.  Snapshot lines mirror the field set the Kaggle dataset's
own description lists (read 2026-08-31), and the Kaggle view payload mirrors a
real response from the public dataset-view endpoint.

The tests that carry the most weight are the two that pin the defects this
loader exists to remove:

- :meth:`TestDoiIdentity.test_publisher_and_arxiv_dois_both_survive` -- the
  identity ``D24`` needs, which paperscraper's normalisation destroys.
- :meth:`TestIdentifierParsing.test_archive_name_containing_v_is_not_truncated`
  -- the ``solv-int`` case, where the obvious implementation mints a DOI that
  does not resolve.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from lit_search.loaders.arxiv_snapshot import (
    ARXIV_ML_CATEGORIES,
    MANIFEST_SUFFIX,
    SnapshotProvenance,
    SnapshotScan,
    SnapshotUnavailableError,
    arxiv_doi_for,
    iter_records,
    normalize_record,
    parse_kaggle_view,
    resolve_snapshot_path,
    strip_version_suffix,
    write_manifest,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _raw(
    *,
    identifier: str = "2106.09685",
    title: str = "LoRA: Low-Rank Adaptation of Large Language Models",
    doi: str | None = None,
    categories: str = "cs.CL cs.AI cs.LG",
    versions: list[dict[str, str]] | None = None,
) -> dict:
    """Build one raw snapshot record.

    Parameters
    ----------
    identifier : str
        The ``id`` field.
    title : str
        The ``title`` field.
    doi : str | None
        The ``doi`` field, absent when ``None``.
    categories : str
        Space-separated ``categories`` field.
    versions : list[dict[str, str]] | None
        The ``versions`` list.

    Returns
    -------
    dict
        Raw record in the Kaggle dump's shape.
    """
    record = {
        "id": identifier,
        "submitter": "Edward Hu",
        "authors": "Edward J. Hu and Yelong Shen",
        "title": title,
        "comments": "Draft V2",
        "journal-ref": "",
        "doi": doi,
        "abstract": "  An important paradigm of natural language processing\n  consists of ...",
        "categories": categories,
        "license": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
        "versions": versions
        if versions is not None
        else [
            {"version": "v1", "created": "Thu, 17 Jun 2021 17:37:18 GMT"},
            {"version": "v2", "created": "Fri, 16 Oct 2021 15:20:11 GMT"},
        ],
        "update_date": "2021-10-17",
        "authors_parsed": [["Hu", "Edward J.", ""], ["Shen", "Yelong", ""]],
    }
    if doi is None:
        record.pop("doi")
    return record


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    """Write a small snapshot file and return its path."""
    path = tmp_path / "arxiv-metadata-oai-snapshot.json"
    lines = [
        json.dumps(_raw()),
        json.dumps(_raw(identifier="2301.00001", categories="math.AP", title="Elliptic things")),
        json.dumps(_raw(identifier="1706.03762", categories="cs.CL", doi="10.5555/3295222")),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Identifier parsing
# ---------------------------------------------------------------------------


class TestIdentifierParsing:
    """The versionless identifier, and the DOI built from it."""

    def test_new_style_identifier_is_unchanged(self) -> None:
        """A modern identifier carries no version suffix and is left alone."""
        assert strip_version_suffix("2106.09685") == "2106.09685"

    def test_trailing_version_suffix_is_removed(self) -> None:
        """A ``v<n>`` suffix is stripped when one is present."""
        assert strip_version_suffix("2106.09685v3") == "2106.09685"

    def test_archive_name_containing_v_is_not_truncated(self) -> None:
        """``solv-int`` survives, where ``id.split("v")[0]`` destroys it.

        paperscraper computes ``str(record["id"]).split("v")[0]``, which turns
        ``solv-int/9506005`` into ``sol`` and mints
        ``10.48550/arXiv.sol``.  Checked live 2026-08-31: the correct DOI
        resolves through doi.org and the truncated one is a 404.
        """
        assert strip_version_suffix("solv-int/9506005") == "solv-int/9506005"
        donor_id = "solv-int/9506005"
        assert donor_id.split("v")[0] == "sol"
        assert arxiv_doi_for(strip_version_suffix("solv-int/9506005")) == (
            "10.48550/arXiv.solv-int/9506005"
        )

    def test_old_style_identifier_is_unchanged(self) -> None:
        """An old-style archive/number identifier round-trips."""
        assert strip_version_suffix("hep-th/9901001") == "hep-th/9901001"


# ---------------------------------------------------------------------------
# The DOI identity problem
# ---------------------------------------------------------------------------


class TestDoiIdentity:
    """Both identities survive normalisation, which is the whole point."""

    def test_publisher_and_arxiv_dois_both_survive(self) -> None:
        """A published record keeps its arXiv DOI as well as the journal one."""
        record = normalize_record(_raw(identifier="1706.03762", doi="10.5555/3295222"))
        assert record is not None
        assert record.publisher_doi == "10.5555/3295222"
        assert record.arxiv_doi == "10.48550/arXiv.1706.03762"
        # D24: the published record is the citable one, the preprint is linked.
        assert record.citable_doi == "10.5555/3295222"
        assert record.preprint_doi == "10.48550/arXiv.1706.03762"

    def test_unpublished_record_has_one_identity_and_says_so(self) -> None:
        """With no journal DOI the arXiv DOI is primary, not a second identity."""
        record = normalize_record(_raw())
        assert record is not None
        assert record.publisher_doi is None
        assert record.citable_doi == "10.48550/arXiv.2106.09685"
        assert record.preprint_doi is None

    def test_paperscraper_would_have_kept_only_one(self) -> None:
        """Pin the donor behaviour this loader replaces.

        Not a test of our code -- a test of the claim in the module docstring,
        so that the reason for the extra field cannot quietly stop being true.
        """
        raw = _raw(identifier="1706.03762", doi="10.5555/3295222")
        donor_doi = raw.get("doi") or f"10.48550/arXiv.{raw['id'].split('v')[0]}"
        assert donor_doi == "10.5555/3295222"
        record = normalize_record(raw)
        assert record is not None
        assert record.arxiv_doi not in {donor_doi}


# ---------------------------------------------------------------------------
# Record normalisation
# ---------------------------------------------------------------------------


class TestNormalization:
    """Field extraction from a raw snapshot line."""

    def test_versions_supply_submission_and_update_dates(self) -> None:
        """v1 gives the submission date and the last version the update date."""
        record = normalize_record(_raw())
        assert record is not None
        assert record.submitted == datetime.date(2021, 6, 17)
        assert record.updated == datetime.date(2021, 10, 16)
        assert record.latest_version == 2
        assert record.version_count == 2

    def test_authors_come_from_the_structured_field(self) -> None:
        """``authors_parsed`` is preferred over the free-text ``authors``."""
        record = normalize_record(_raw())
        assert record is not None
        assert record.authors == ("Edward J. Hu", "Yelong Shen")

    def test_authors_fall_back_to_the_flat_string(self) -> None:
        """Without ``authors_parsed`` the flat string is split."""
        raw = _raw()
        raw.pop("authors_parsed")
        record = normalize_record(raw)
        assert record is not None
        assert record.authors == ("Edward J. Hu", "Yelong Shen")

    def test_whitespace_in_the_abstract_is_collapsed(self) -> None:
        """The dump wraps abstracts with leading spaces and newlines."""
        record = normalize_record(_raw())
        assert record is not None
        assert record.abstract.startswith("An important paradigm")
        assert "\n" not in record.abstract

    def test_a_record_with_no_title_is_unparsable(self) -> None:
        """A record with no title is ``None``, for the caller to count."""
        assert normalize_record(_raw(title="")) is None

    def test_urls_are_derived_from_the_identifier(self) -> None:
        """Abstract and PDF URLs follow from the versionless identifier."""
        record = normalize_record(_raw())
        assert record is not None
        assert record.abs_url == "https://arxiv.org/abs/2106.09685"
        assert record.pdf_url == "https://arxiv.org/pdf/2106.09685"


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


class TestScanning:
    """Streaming the snapshot, and counting what could not be read."""

    def test_category_filter_keeps_cross_listed_records(self, snapshot: Path) -> None:
        """A record is in scope when any of its categories is."""
        scan = SnapshotScan()
        records = list(iter_records(str(snapshot), categories=ARXIV_ML_CATEGORIES, scan=scan))
        assert [r.arxiv_id for r in records] == ["2106.09685", "1706.03762"]
        assert scan.lines == 3
        assert scan.in_scope == 2

    def test_no_category_filter_reads_everything(self, snapshot: Path) -> None:
        """``categories=None`` keeps every record."""
        scan = SnapshotScan()
        records = list(iter_records(str(snapshot), categories=None, scan=scan))
        assert len(records) == 3

    def test_unreadable_and_unparsable_lines_are_counted_separately(self, tmp_path: Path) -> None:
        """A lost line is a coverage loss, and the two kinds are distinguished."""
        path = tmp_path / "snap.json"
        path.write_text(
            "\n".join(
                [
                    json.dumps(_raw()),
                    "{not json at all",
                    json.dumps({"id": "2101.00001", "categories": "cs.LG"}),
                    json.dumps(["a list, not an object"]),
                    "",
                ]
            ),
            encoding="utf-8",
        )
        scan = SnapshotScan()
        records = list(iter_records(str(path), categories=None, scan=scan))
        assert len(records) == 1
        assert scan.lines == 4
        assert scan.unreadable == 2
        assert scan.unparsed == 1
        assert scan.lost == 3

    def test_a_missing_file_raises_rather_than_yielding_nothing(self, tmp_path: Path) -> None:
        """An absent snapshot is not an arXiv that holds no papers."""
        with pytest.raises(SnapshotUnavailableError):
            list(iter_records(str(tmp_path / "nope.json"), scan=SnapshotScan()))


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    """The snapshot's date and version, and how honestly they are reported."""

    def test_manifest_supplies_dataset_version_and_date(self, snapshot: Path) -> None:
        """With a manifest the provenance names Kaggle's own version and date."""
        write_manifest(
            str(snapshot),
            version_number=301,
            version_created_at="2026-08-29T23:51:40.527Z",
            downloaded_at="2026-08-30T09:00:00+00:00",
        )
        provenance = SnapshotProvenance.for_file(str(snapshot))
        assert provenance.version_number == 301
        assert provenance.as_of == "2026-08-29T23:51:40.527Z"
        assert provenance.dataset_ref == "Cornell-University/arxiv"
        described = provenance.describe()
        assert "Cornell-University/arxiv" in described
        assert "version 301" in described
        assert "published by Kaggle at" in described

    def test_without_a_manifest_the_date_basis_says_what_it_is(self, snapshot: Path) -> None:
        """A file mtime is reported as a file mtime, never as a snapshot date."""
        provenance = SnapshotProvenance.for_file(str(snapshot))
        assert provenance.version_number is None
        assert provenance.file_mtime is not None
        assert provenance.as_of == provenance.file_mtime
        described = provenance.describe()
        assert "an unrecorded version" in described
        assert "modification time" in described

    def test_a_corrupt_manifest_degrades_rather_than_raising(self, snapshot: Path) -> None:
        """An unreadable manifest falls back and says so, instead of failing."""
        Path(f"{snapshot}{MANIFEST_SUFFIX}").write_text("{ broken", encoding="utf-8")
        provenance = SnapshotProvenance.for_file(str(snapshot))
        assert provenance.version_number is None
        assert "modification time" in provenance.describe()

    def test_manifest_cannot_misdescribe_the_file_that_was_opened(self, snapshot: Path) -> None:
        """Path and size come from the opened file, not from the manifest."""
        write_manifest(str(snapshot), version_number=42, dataset_ref="someone/else")
        provenance = SnapshotProvenance.for_file(str(snapshot))
        assert provenance.path == str(snapshot)
        assert provenance.size_bytes == snapshot.stat().st_size

    def test_kaggle_view_payload_yields_version_and_creation_date(self) -> None:
        """Parse the public dataset-view body into manifest fields.

        Field names and shape are from a live response on 2026-08-31.
        """
        fields = parse_kaggle_view(
            {
                "ref": "Cornell-University/arxiv",
                "currentVersionNumber": 301,
                "lastUpdated": "2026-08-29T23:51:40.527Z",
                "versions": [
                    {"versionNumber": 301, "creationDate": "2026-08-29T23:51:40.527Z"},
                    {"versionNumber": 300, "creationDate": "2026-08-22T23:51:43.117Z"},
                ],
            }
        )
        assert fields == {
            "dataset_ref": "Cornell-University/arxiv",
            "version_number": 301,
            "version_created_at": "2026-08-29T23:51:40.527Z",
        }

    def test_kaggle_view_without_a_version_reports_none_rather_than_guessing(self) -> None:
        """An absent version stays absent instead of being inferred."""
        fields = parse_kaggle_view({"ref": "Cornell-University/arxiv"})
        assert fields["version_number"] is None
        assert fields["version_created_at"] is None


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


class TestPathResolution:
    """Finding the snapshot, and refusing to guess which one it is."""

    def test_an_explicit_file_resolves(self, snapshot: Path) -> None:
        """A path naming a file resolves to that file."""
        assert resolve_snapshot_path(str(snapshot)) == str(snapshot.resolve())

    def test_a_directory_with_one_json_resolves(self, snapshot: Path) -> None:
        """A directory holding exactly one candidate is unambiguous."""
        assert resolve_snapshot_path(str(snapshot.parent)) == str(snapshot.resolve())

    def test_a_directory_with_two_candidates_refuses(self, snapshot: Path) -> None:
        """Refuse rather than pick the biggest, which is paperscraper's rule.

        Picking the largest file silently reads last month's snapshot when this
        month's download was interrupted.
        """
        (snapshot.parent / "other.json").write_text("{}", encoding="utf-8")
        assert resolve_snapshot_path(str(snapshot.parent)) is None

    def test_nothing_configured_resolves_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No configuration is ``None``, for the connector to turn into a skip."""
        monkeypatch.delenv("LIT_SEARCH_ARXIV_SNAPSHOT", raising=False)
        assert resolve_snapshot_path(None) is None
