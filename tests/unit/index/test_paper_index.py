"""Tests for the LanceDB paper index.

The load-bearing claims under test, in the order they would bite:

1. The vector width comes from the model, and an index built with one model
   refuses to open under another.  A dimension check alone is not enough.
2. The full-text index is created with the current, non-deprecated API.
3. Rows added after the index are still returned, because the flat-scan
   fallback is left in place -- and ``.fast_search()``, which removes it, is
   never called by this project.  The second half is the one worth a test: it
   demonstrates the omission rather than asserting the absence of a call.
4. Model identity and revision are stored on every row, not just the
   dimension.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path

import pytest

from lit_search.core.citation import IdentifierKind, PaperIdentifier
from lit_search.core.paper import Paper
from lit_search.index.embedding import DeterministicEmbedder
from lit_search.index.paper_index import (
    MANIFEST_NAME,
    PaperIndex,
    PaperIndexError,
    key_predicate,
)
from lit_search.index.schema import SEARCH_TEXT_COLUMN, VECTOR_COLUMN, paper_key


class TestDimensionAndIdentity:
    """The vector space is read from the model, never assumed."""

    @pytest.mark.parametrize("dimension", [8, 32, 77])
    def test_vector_width_follows_the_model(
        self, tmp_path: Path, corpus: list[Paper], dimension: int
    ) -> None:
        """An index's vector column is as wide as the loaded model, not a constant."""
        index = PaperIndex(tmp_path / f"index{dimension}", DeterministicEmbedder(dimension))
        index.add(corpus)
        field = index.table().schema.field(VECTOR_COLUMN)
        assert field.type.list_size == dimension

    def test_reopening_with_another_model_is_refused(
        self, tmp_path: Path, corpus: list[Paper]
    ) -> None:
        """Two models are not interchangeable even at the same dimension."""
        root = tmp_path / "index"
        PaperIndex(root, DeterministicEmbedder(16)).add(corpus)

        class Impostor(DeterministicEmbedder):
            """A different model that happens to produce the same width."""

            def __init__(self) -> None:
                """Create an embedder claiming a different identity."""
                super().__init__(dimension=16)
                object.__setattr__(self._identity, "model_id", "other/model")

        with pytest.raises(PaperIndexError, match="was built with"):
            PaperIndex(root, Impostor())

    def test_a_foreign_manifest_is_refused(self, tmp_path: Path) -> None:
        """A file that is not our manifest is not treated as an index."""
        root = tmp_path / "index"
        root.mkdir(parents=True)
        (root / MANIFEST_NAME).write_text('{"schema": "someone/else"}', encoding="utf-8")
        with pytest.raises(PaperIndexError, match="not a lit-search/index manifest"):
            PaperIndex(root, DeterministicEmbedder(8))

    def test_every_row_carries_model_identity_and_revision(
        self, index: PaperIndex, corpus: list[Paper]
    ) -> None:
        """Storing the dimension alone would not detect a mixed-model table."""
        index.add(corpus)
        rows = (
            index.table()
            .search()
            .select(["model_id", "model_revision", "embedding_dim"])
            .limit(10)
            .to_list()
        )
        assert rows
        for row in rows:
            assert row["model_id"] == index.identity.model_id
            assert row["model_revision"] == (index.identity.model_revision or "")
            assert row["embedding_dim"] == index.identity.dimension


class TestFullTextIndex:
    """Keyword retrieval needs an index, and the API to build it has moved."""

    def test_creation_uses_the_current_api(self, index: PaperIndex, corpus: list[Paper]) -> None:
        """``create_fts_index`` is deprecated on the sync table as of 0.25.0."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            index.add(corpus)
        names = [entry.name for entry in index.table().list_indices()]
        assert any(SEARCH_TEXT_COLUMN in str(name) for name in names)

    def test_text_lane_is_unavailable_before_the_index_exists(
        self, index: PaperIndex, corpus: list[Paper]
    ) -> None:
        """Without an index the keyword lane returns nothing, and health says so."""
        index.table().add(_rows_for(index, corpus))
        assert index.health().has_text_index is False
        assert index.text_lane("specification", 10) == []

    def test_health_reports_the_unindexed_tail(
        self, tmp_path: Path, corpus: list[Paper], make_paper: Callable[..., Paper]
    ) -> None:
        """Rows added past the index are counted, not assumed away."""
        index = PaperIndex(tmp_path / "index", DeterministicEmbedder(16), auto_optimize=False)
        index.add(corpus)
        assert index.health().unindexed_rows == 0
        index.add([make_paper("A Late Arrival on Hoare Logic", "10.0/late")])
        assert index.health().unindexed_rows == 1
        assert index.optimize().unindexed_rows == 0


class TestFastSearchIsForbidden:
    """The reason ``.fast_search()`` is banned, demonstrated rather than asserted."""

    def test_the_normal_lane_returns_rows_added_after_the_index(
        self, tmp_path: Path, corpus: list[Paper], make_paper: Callable[..., Paper]
    ) -> None:
        """The flat-scan fallback is what keeps a result complete."""
        index = PaperIndex(tmp_path / "index", DeterministicEmbedder(16), auto_optimize=False)
        index.add(corpus)
        index.add([make_paper("Hoare-Style Reasoning, Late Addition", "10.0/late")])
        assert index.health().unindexed_rows == 1

        found = {row["paper_key"] for row in index.text_lane("hoare", 20)}
        assert "10.0/late" in found

    def test_fast_search_would_silently_drop_them(
        self, tmp_path: Path, corpus: list[Paper], make_paper: Callable[..., Paper]
    ) -> None:
        """The omission is real, which is why the project never calls it."""
        index = PaperIndex(tmp_path / "index", DeterministicEmbedder(16), auto_optimize=False)
        index.add(corpus)
        index.add([make_paper("Hoare-Style Reasoning, Late Addition", "10.0/late")])

        table = index.table()
        fast = (
            table.search("hoare", query_type="fts", fts_columns=SEARCH_TEXT_COLUMN)
            .select(["paper_key", "_score"])
            .limit(20)
            .fast_search()
            .to_list()
        )
        assert "10.0/late" not in {row["paper_key"] for row in fast}

    def test_no_module_in_the_project_calls_it(self) -> None:
        """A grep, so that a future edit reintroducing the call fails here.

        The pattern excludes a leading backtick so that the modules explaining
        *why* the call is banned do not trip their own rule.
        """
        import re

        call = re.compile(r"(?<!`)\.fast_search\(")
        package = Path(__file__).resolve().parents[3] / "lit_search"
        offenders = [
            str(path.relative_to(package))
            for path in package.rglob("*.py")
            if call.search(path.read_text(encoding="utf-8"))
        ]
        assert offenders == []


class TestIngest:
    """Ingest is idempotent by record key and reports degraded text."""

    def test_adding_the_same_papers_twice_writes_nothing_new(
        self, index: PaperIndex, corpus: list[Paper]
    ) -> None:
        """Re-ingesting a handle must not duplicate its rows."""
        first = index.add(corpus)
        second = index.add(corpus)
        assert first.written == len(corpus)
        assert second.written == 0
        assert second.skipped_existing == len(corpus)
        assert index.count() == len(corpus)

    def test_records_embedded_on_incomplete_text_are_named(
        self, index: PaperIndex, corpus: list[Paper]
    ) -> None:
        """A title-only embedding is a weaker measurement and must be visible."""
        report = index.add(corpus)
        degraded = " ".join(report.degraded_text)
        assert "10.0/preguss" in degraded
        assert "abstract" in degraded

    def test_byte_distinct_identifier_free_same_title_records_get_distinct_keys(self) -> None:
        """A title collision cannot suppress a distinct snapshot row at ingest."""
        papers = [
            Paper(
                title="Annual Report",
                abstract=abstract,
                authors=[],
                source=None,
                publication_date=None,
            )
            for abstract in ("First edition", "Second edition")
        ]

        keys = {paper_key(paper) for paper in papers}
        assert len(keys) == 2
        assert all(key.startswith("snapshot:") for key in keys)

    def test_identifier_free_snapshot_keys_are_deterministic_within_the_index(self) -> None:
        """Re-ingesting the same serialized snapshot is idempotent."""
        paper = Paper(
            title="Unidentified",
            abstract="Stable snapshot",
            authors=[],
            source=None,
            publication_date=None,
        )

        assert paper_key(paper) == paper_key(Paper.from_dict(paper.to_dict()))

    def test_provider_native_identity_keys_a_doi_less_record(self) -> None:
        """A source-native ID is preferred over snapshot content."""
        paper = Paper(
            title="Native record",
            abstract="",
            authors=[],
            source=None,
            publication_date=None,
            identifiers=[PaperIdentifier(IdentifierKind.OPENALEX, "W123")],
        )

        assert paper_key(paper) == "openalex:w123"


class TestLaneFiltering:
    """Restricting a lane to a key set, and the empty-set trap."""

    def test_no_restriction_and_empty_restriction_differ(self) -> None:
        """ "Restrict to nothing" and "do not restrict" are opposite instructions."""
        assert key_predicate(None) is None
        assert key_predicate([]) == "1 = 0"

    def test_quotes_in_a_key_cannot_break_the_predicate(self) -> None:
        """A key containing a quote is escaped, not injected."""
        assert key_predicate(["o'brien"]) == "paper_key IN ('o''brien')"

    def test_lanes_honour_the_restriction(self, index: PaperIndex, corpus: list[Paper]) -> None:
        """Every lane can be scoped to one stored set's members."""
        index.add(corpus)
        keys = ["10.48550/arxiv.2604.11556"]
        vector = index._embedder.encode_query("specification")
        assert {row["paper_key"] for row in index.vector_lane(vector, 10, keys=keys)} == set(keys)
        assert {row["paper_key"] for row in index.text_lane("hoare", 10, keys=keys)} <= set(keys)
        assert {row["paper_key"] for row in index.exact_lane(["FM-Agent"], 10, keys=keys)} <= set(
            keys
        )


def _rows_for(index: PaperIndex, papers: list[Paper]) -> list[dict]:
    """Build raw rows for *papers*, bypassing :meth:`PaperIndex.add`.

    Used only to reach the state "table has rows but no full-text index",
    which ``add`` deliberately never leaves behind.

    Parameters
    ----------
    index : PaperIndex
        Index the rows belong to.
    papers : list[Paper]
        Papers to build rows for.

    Returns
    -------
    list[dict]
        Rows matching the index schema.
    """
    from lit_search.index.schema import build_row
    from lit_search.index.text import build_paper_text

    texts = [build_paper_text(paper, index.text_policy) for paper in papers]
    vectors = index._embedder.encode_documents([text.search_text for text in texts])
    return [
        build_row(
            paper,
            text,
            vector,
            identity=index.identity,
            policy=index.text_policy,
            source_handle=None,
        )
        for paper, text, vector in zip(papers, texts, vectors, strict=True)
    ]
