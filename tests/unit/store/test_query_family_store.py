"""Tests for immutable query-family revision persistence and lineage."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from lit_search.core.query_family import QueryFamilyRevision
from lit_search.store.query_family_store import (
    AmbiguousQueryFamilyRevisionError,
    QueryFamilyLineageError,
    QueryFamilyRevisionIntegrityError,
    QueryFamilyRevisionNotFoundError,
    QueryFamilyStore,
)
from lit_search.store.store import HandleStore


def _revision(
    question: str,
    *parents: QueryFamilyRevision,
) -> QueryFamilyRevision:
    return QueryFamilyRevision(
        research_question=question,
        parent_revision_ids=tuple(parent.revision_id for parent in parents),
    )


class TestStorage:
    """A revision round-trips without acquiring a mutable name."""

    def test_store_is_rooted_beside_handles_and_created_lazily(self, tmp_path: Path) -> None:
        handles = HandleStore(tmp_path / "cache")
        store = QueryFamilyStore(handles)
        assert store.root == handles.root
        assert store.revisions_directory == handles.root / "query-families"
        assert not store.revisions_directory.exists()
        assert store.list_ids() == []

    def test_put_get_exists_and_list(self, tmp_path: Path) -> None:
        store = QueryFamilyStore(tmp_path)
        revision = _revision("Which vocabulary crosses the boundary?")

        revision_id = store.put(revision)

        assert revision_id == revision.revision_id
        assert store.exists(revision_id)
        assert store.get(revision_id).to_dict() == revision.to_dict()
        assert store.list_ids() == [revision_id]
        assert store.path_for(revision_id) == (tmp_path / "query-families" / f"{revision_id}.json")

    def test_identical_retry_is_a_no_op(self, tmp_path: Path) -> None:
        store = QueryFamilyStore(tmp_path)
        revision = _revision("Repeatable revision")
        revision_id = store.put(revision)
        before = store.path_for(revision_id).stat().st_mtime_ns

        assert store.put(revision) == revision_id
        assert store.path_for(revision_id).stat().st_mtime_ns == before

    def test_existing_tampered_file_is_never_overwritten(self, tmp_path: Path) -> None:
        store = QueryFamilyStore(tmp_path)
        revision = _revision("Immutable revision")
        revision_id = store.put(revision)
        path = store.path_for(revision_id)
        corrupted = '{"corrupted": true}'
        path.write_text(corrupted, encoding="utf-8")

        with pytest.raises(QueryFamilyRevisionIntegrityError, match="strict verification"):
            store.put(revision)
        assert path.read_text(encoding="utf-8") == corrupted

    def test_atomic_write_failure_leaves_no_target_or_scratch_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = QueryFamilyStore(tmp_path)
        revision = _revision("Crash boundary")

        def exploding_replace(source: object, target: object) -> None:
            raise OSError("simulated rename crash")

        monkeypatch.setattr(os, "replace", exploding_replace)
        with pytest.raises(OSError, match="simulated rename crash"):
            store.put(revision)

        assert not store.path_for(revision.revision_id).exists()
        assert list(store.revisions_directory.iterdir()) == []


class TestStrictReads:
    """The filename, schema, and semantic digest must all agree."""

    def test_missing_and_malformed_identifiers_are_refused(self, tmp_path: Path) -> None:
        store = QueryFamilyStore(tmp_path)
        with pytest.raises(QueryFamilyRevisionNotFoundError, match="No query-family"):
            store.get("qf_" + "0" * 32)
        with pytest.raises(QueryFamilyRevisionIntegrityError, match="not a query-family"):
            store.path_for("../../escape")
        assert not store.exists("../../escape")

    def test_unknown_schema_field_is_refused_on_get(self, tmp_path: Path) -> None:
        store = QueryFamilyStore(tmp_path)
        revision = _revision("Strict schema")
        revision_id = store.put(revision)
        path = store.path_for(revision_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["future_meaning"] = True
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(QueryFamilyRevisionIntegrityError, match="unrecognised key"):
            store.get(revision_id)

    def test_changed_semantics_are_refused_on_get(self, tmp_path: Path) -> None:
        store = QueryFamilyStore(tmp_path)
        revision = _revision("Original question")
        revision_id = store.put(revision)
        path = store.path_for(revision_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["research_question"] = "Quietly changed"
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(QueryFamilyRevisionIntegrityError, match="recomputed"):
            store.get(revision_id)

    def test_valid_revision_in_wrong_filename_is_refused(self, tmp_path: Path) -> None:
        store = QueryFamilyStore(tmp_path)
        first = _revision("First")
        second = _revision("Second")
        store.revisions_directory.mkdir(parents=True)
        store.path_for(first.revision_id).write_text(json.dumps(second.to_dict()), encoding="utf-8")

        with pytest.raises(QueryFamilyRevisionIntegrityError, match="contains revision"):
            store.get(first.revision_id)

    def test_list_ignores_unrelated_and_malformed_files(self, tmp_path: Path) -> None:
        store = QueryFamilyStore(tmp_path)
        revision = _revision("Only valid name")
        revision_id = store.put(revision)
        (store.revisions_directory / "notes.json").write_text("{}", encoding="utf-8")
        (store.revisions_directory / "qf_not-a-digest.json").write_text("{}", encoding="utf-8")
        assert store.list_ids() == [revision_id]


class TestPrefixResolution:
    """Prefixes are ergonomic lookup, never mutable aliases."""

    def test_full_and_unambiguous_prefix_resolve(self, tmp_path: Path) -> None:
        store = QueryFamilyStore(tmp_path)
        revision_id = store.put(_revision("Resolvable"))
        assert store.resolve(revision_id) == revision_id
        assert store.resolve(revision_id[:12]) == revision_id
        assert store.resolve(revision_id[3:15]) == revision_id

    def test_short_nonhex_and_unknown_prefixes_are_refused(self, tmp_path: Path) -> None:
        store = QueryFamilyStore(tmp_path)
        store.put(_revision("Known"))
        with pytest.raises(QueryFamilyRevisionNotFoundError, match="shorter"):
            store.resolve("ab")
        with pytest.raises(QueryFamilyRevisionNotFoundError, match="not hexadecimal"):
            store.resolve("nothex")
        with pytest.raises(QueryFamilyRevisionNotFoundError, match="No query-family"):
            store.resolve("ffff")

    def test_ambiguous_prefix_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = QueryFamilyStore(tmp_path)
        first = "qf_" + "a" * 32
        second = "qf_" + "a" * 31 + "b"
        monkeypatch.setattr(store, "list_ids", lambda: [first, second])
        with pytest.raises(AmbiguousQueryFamilyRevisionError, match="matches 2 revisions"):
            store.resolve("aaaa")


class TestLineage:
    """Only a complete acyclic revision DAG is reportable as lineage."""

    def test_put_requires_every_direct_parent(self, tmp_path: Path) -> None:
        store = QueryFamilyStore(tmp_path)
        absent_parent = _revision("Absent parent")
        child = _revision("Child", absent_parent)
        with pytest.raises(QueryFamilyLineageError, match="missing parent"):
            store.put(child)
        assert store.list_ids() == []

    def test_diamond_lineage_is_deterministic_breadth_first(self, tmp_path: Path) -> None:
        store = QueryFamilyStore(tmp_path)
        base = _revision("Base")
        left = _revision("Left", base)
        right = _revision("Right", base)
        top = _revision("Top", right, left)
        for revision in (base, left, right, top):
            store.put(revision)

        ordered_parents = sorted((left.revision_id, right.revision_id))
        lineage = store.lineage(top.revision_id)

        assert lineage.root == top.revision_id
        assert lineage.nodes == (top.revision_id, *ordered_parents, base.revision_id)
        assert lineage.edges == (
            (top.revision_id, ordered_parents[0]),
            (top.revision_id, ordered_parents[1]),
            (ordered_parents[0], base.revision_id),
            (ordered_parents[1], base.revision_id),
        )
        assert store.ancestors(top.revision_id) == [*ordered_parents, base.revision_id]

    def test_missing_transitive_parent_is_refused(self, tmp_path: Path) -> None:
        store = QueryFamilyStore(tmp_path)
        base = _revision("Base")
        child = _revision("Child", base)
        store.put(base)
        store.put(child)
        store.path_for(base.revision_id).unlink()

        with pytest.raises(QueryFamilyLineageError, match="missing parent"):
            store.lineage(child.revision_id)

    def test_cycle_is_refused_even_if_storage_was_corrupted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = QueryFamilyStore(tmp_path)
        first_id = "qf_" + "1" * 32
        second_id = "qf_" + "2" * 32
        base = _revision("Synthetic corrupt graph")
        records = {
            first_id: replace(base, parent_revision_ids=(second_id,)),
            second_id: replace(base, parent_revision_ids=(first_id,)),
        }
        monkeypatch.setattr(store, "exists", lambda revision_id: revision_id in records)
        monkeypatch.setattr(store, "get", lambda revision_id: records[revision_id])

        with pytest.raises(QueryFamilyLineageError, match="contains a cycle"):
            store.lineage(first_id)
