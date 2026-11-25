"""Tests for the on-disk handle store: durability, immutability, lookup."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from lit_search.exceptions import (
    AmbiguousHandleError,
    HandleIntegrityError,
    HandleNotFoundError,
    HandleSchemaError,
)
from lit_search.store.lineage import Operand, Operation, OperationName
from lit_search.store.record import HandleRecord
from lit_search.store.store import (
    APP_DIRECTORY_NAME,
    STORE_ROOT_ENV_VAR,
    HandleStore,
    default_store_root,
)


class TestRoot:
    """Every process must resolve the same cache root."""

    def test_environment_override_wins(self, monkeypatch, tmp_path: Path) -> None:
        """An explicit home overrides the XDG default."""
        monkeypatch.setenv(STORE_ROOT_ENV_VAR, str(tmp_path / "elsewhere"))
        assert default_store_root() == tmp_path / "elsewhere"

    def test_xdg_cache_is_used_when_no_override(self, monkeypatch, tmp_path: Path) -> None:
        """Without an override the store lives under the XDG cache directory."""
        monkeypatch.delenv(STORE_ROOT_ENV_VAR, raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        assert default_store_root() == tmp_path / APP_DIRECTORY_NAME

    def test_home_cache_is_the_final_fallback(self, monkeypatch) -> None:
        """With neither variable set the store lives under ~/.cache."""
        monkeypatch.delenv(STORE_ROOT_ENV_VAR, raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        assert default_store_root() == Path.home() / ".cache" / APP_DIRECTORY_NAME

    def test_store_is_not_created_until_written(self, tmp_path: Path) -> None:
        """Opening a store does not create directories."""
        store = HandleStore(tmp_path / "unused")
        assert not store.root.exists()
        assert store.list_ids() == []


class TestPutAndGet:
    """A handle survives the round trip through the filesystem."""

    def test_put_then_get(self, store, make_record, papers) -> None:
        """A stored handle reloads with the same identity and members."""
        handle_id = store.put(make_record(papers, label="round one"))
        loaded = store.get(handle_id)
        assert loaded.handle_id == handle_id
        assert loaded.label == "round one"
        assert [p.doi for p in loaded.members] == [p.doi for p in papers]

    def test_handle_lands_in_its_own_file(self, store, make_record, papers) -> None:
        """One JSON file per handle, named by its content address."""
        handle_id = store.put(make_record(papers))
        path = store.handles_directory / f"{handle_id}.json"
        assert path.is_file()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 1
        assert payload["handle_id"] == handle_id

    def test_store_is_readable_by_a_second_instance(self, tmp_path, make_record, papers) -> None:
        """Handles are shared through the filesystem, not through process state."""
        writer = HandleStore(tmp_path / "cache")
        handle_id = writer.put(make_record(papers))
        reader = HandleStore(tmp_path / "cache")
        assert reader.get(handle_id).handle_id == handle_id

    def test_missing_handle_raises(self, store) -> None:
        """A handle that was never written is reported as absent."""
        with pytest.raises(HandleNotFoundError, match="No handle"):
            store.get("h_" + "0" * 32)

    def test_exists_reports_presence(self, store, make_record, papers) -> None:
        """Presence is a filesystem question with a straight answer."""
        handle_id = store.put(make_record(papers))
        assert store.exists(handle_id)
        assert not store.exists("h_" + "1" * 32)


class TestImmutability:
    """Every operation writes a new handle; nothing is edited in place."""

    def test_rewriting_identical_content_is_a_no_op(self, store, make_record, papers) -> None:
        """A crashed and restarted operation does not duplicate anything."""
        record = make_record(papers)
        first = store.put(record)
        before = store.path_for(first).stat().st_mtime_ns
        assert store.put(record) == first
        assert store.path_for(first).stat().st_mtime_ns == before
        assert store.list_ids() == [first]

    def test_different_content_takes_a_different_address(self, store, make_record, papers) -> None:
        """Two distinct sets never contend for the same file."""
        first = store.put(make_record(papers))
        second = store.put(make_record(papers[:2]))
        assert first != second
        assert sorted(store.list_ids()) == sorted([first, second])

    def test_colliding_address_with_different_content_is_refused(
        self, store, make_record, papers
    ) -> None:
        """An immutable handle is never silently overwritten."""
        record = make_record(papers)
        handle_id = store.put(record)
        store.path_for(handle_id).write_text('{"schema": "tampered"}', encoding="utf-8")
        with pytest.raises(HandleIntegrityError, match="refusing to overwrite"):
            store.put(record)


class TestDanglingLineage:
    """A lineage edge that points at nothing fails when it is created."""

    def test_put_refuses_an_operand_not_in_the_store(self, store, papers) -> None:
        """The failure lands on the writer, not on a later audit."""
        record = HandleRecord(
            operation=Operation(
                name=OperationName.FILTER,
                operands=(Operand.of_handle("h_" + "a" * 32, "base"),),
                parameters={"description": "top 2"},
            ),
            members=tuple(papers[:2]),
        )
        with pytest.raises(HandleNotFoundError, match="which is not in the store"):
            store.put(record)

    def test_put_accepts_an_operand_that_exists(self, store, make_record, papers) -> None:
        """A well-formed edge is written without complaint."""
        base = store.put(make_record(papers))
        derived = HandleRecord(
            operation=Operation(
                name=OperationName.FILTER,
                operands=(Operand.of_handle(base, "base"),),
                parameters={"description": "first two"},
            ),
            members=tuple(papers[:2]),
        )
        assert store.put(derived)


class TestIntegrity:
    """A file that does not hash to its own name is not loaded."""

    def test_hand_edited_file_is_detected(self, store, make_record, papers) -> None:
        """Editing a stored handle by hand is caught on read."""
        handle_id = store.put(make_record(papers))
        path = store.path_for(handle_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["label"] = "quietly renamed"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(HandleIntegrityError, match="does not match its own content"):
            store.get(handle_id)

    def test_truncated_file_is_reported_as_bad_json(self, store, make_record, papers) -> None:
        """A partial file is refused rather than partially believed."""
        handle_id = store.put(make_record(papers))
        path = store.path_for(handle_id)
        path.write_text(path.read_text(encoding="utf-8")[:120], encoding="utf-8")
        with pytest.raises(HandleSchemaError, match="not valid JSON"):
            store.get(handle_id)

    def test_non_object_file_is_refused(self, store, make_record, papers) -> None:
        """A JSON array is not a handle."""
        handle_id = store.put(make_record(papers))
        store.path_for(handle_id).write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(HandleSchemaError, match="expected an object"):
            store.get(handle_id)


class TestAtomicity:
    """A crash mid-write never leaves a truncated handle behind."""

    def test_failed_rename_leaves_no_file_and_no_temporary(
        self, store, make_record, papers, monkeypatch
    ) -> None:
        """The destination is untouched and the scratch file is cleaned up."""
        store.handles_directory.mkdir(parents=True, exist_ok=True)

        def exploding_replace(source, target):  # type: ignore[no-untyped-def]
            raise OSError("simulated crash during rename")

        monkeypatch.setattr(os, "replace", exploding_replace)
        with pytest.raises(OSError, match="simulated crash"):
            store.put(make_record(papers))

        assert store.list_ids() == []
        assert list(store.handles_directory.iterdir()) == []

    def test_failed_rename_preserves_previous_content(self, tmp_path, monkeypatch) -> None:
        """A crash leaves the old file whole rather than half-overwritten."""
        from lit_search.utils.persistence import atomic_write_text

        target = tmp_path / "handle.json"
        target.write_text('{"good": true}', encoding="utf-8")

        def exploding_replace(source, target_path):  # type: ignore[no-untyped-def]
            raise OSError("simulated crash during rename")

        monkeypatch.setattr(os, "replace", exploding_replace)
        with pytest.raises(OSError, match="simulated crash"):
            atomic_write_text(target, "x" * 10_000)

        assert target.read_text(encoding="utf-8") == '{"good": true}'
        assert list(tmp_path.iterdir()) == [target]

    def test_write_is_visible_only_as_whole_content(self, store, make_record, papers) -> None:
        """The file that appears at the destination is always complete."""
        handle_id = store.put(make_record(papers))
        payload = json.loads(store.path_for(handle_id).read_text(encoding="utf-8"))
        assert len(payload["members"]) == len(papers)


class TestResolve:
    """An abbreviated identifier expands, or says why it cannot."""

    def test_full_identifier_resolves(self, store, make_record, papers) -> None:
        """A full identifier is returned unchanged."""
        handle_id = store.put(make_record(papers))
        assert store.resolve(handle_id) == handle_id

    def test_prefix_resolves(self, store, make_record, papers) -> None:
        """A distinctive prefix is enough."""
        handle_id = store.put(make_record(papers))
        assert store.resolve(handle_id[:12]) == handle_id

    def test_prefix_without_marker_resolves(self, store, make_record, papers) -> None:
        """The h_ marker may be omitted when typing."""
        handle_id = store.put(make_record(papers))
        assert store.resolve(handle_id[2:14]) == handle_id

    def test_ambiguous_prefix_is_refused(self, store, monkeypatch) -> None:
        """Picking one of two candidate sets silently is exactly the enemy."""
        first = "h_" + "a" * 32
        second = "h_" + "a" * 31 + "b"
        monkeypatch.setattr(HandleStore, "list_ids", lambda self: [first, second])
        with pytest.raises(AmbiguousHandleError, match="matches 2 handles"):
            store.resolve("aaaa")

    def test_too_short_prefix_is_refused(self, store, make_record, papers) -> None:
        """A prefix too short to identify anything safely is refused."""
        store.put(make_record(papers))
        with pytest.raises(HandleNotFoundError, match="too short"):
            store.resolve("ab")

    def test_unmatched_prefix_is_refused(self, store, make_record, papers) -> None:
        """Nothing matching is an error, not an empty result."""
        store.put(make_record(papers))
        with pytest.raises(HandleNotFoundError, match="No handle matching"):
            store.resolve("ffffffff")


class TestIdentifierValidation:
    """An identifier is used as a filename and is validated as one."""

    @pytest.mark.parametrize(
        "bad",
        ["../../etc/passwd", "h_../escape", "h_" + "z" * 32, "h_abc", "no_prefix", "h_" + "A" * 32],
    )
    def test_malformed_identifiers_are_refused(self, store, bad) -> None:
        """A malformed identifier never reaches the filesystem."""
        with pytest.raises(HandleSchemaError, match="not a handle identifier"):
            store.path_for(bad)

    def test_exists_is_false_for_malformed_identifiers(self, store) -> None:
        """Presence checks do not raise on garbage input."""
        assert not store.exists("../../etc/passwd")

    def test_list_ids_ignores_unrelated_files(self, store, make_record, papers) -> None:
        """Stray files in the cache directory are not mistaken for handles."""
        handle_id = store.put(make_record(papers))
        (store.handles_directory / "notes.txt").write_text("hello", encoding="utf-8")
        (store.handles_directory / "h_short.json").write_text("{}", encoding="utf-8")
        assert store.list_ids() == [handle_id]


def test_canonical_text_is_stable(store, make_record, papers) -> None:
    """Two reads of the same handle produce identical canonical text."""
    handle_id = store.put(make_record(papers))
    assert store.canonical_text(handle_id) == store.canonical_text(handle_id)
