"""Tests for crash-safe file writing.

The pattern being replaced, ``open(path, "w")``, truncates the destination
before the new content is serialized.  A crash between those two moments
destroys the previous file and leaves an unparsable fragment in its place.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from lit_search.utils.persistence import atomic_write_json, atomic_write_text


class TestAtomicWriteText:
    """Writes land whole or not at all."""

    def test_creates_the_file(self, tmp_path: Path) -> None:
        """The happy path writes the requested content."""
        target = tmp_path / "out.txt"
        atomic_write_text(target, "hello")
        assert target.read_text(encoding="utf-8") == "hello"

    def test_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        """A nested destination does not need to exist beforehand."""
        target = tmp_path / "a" / "b" / "out.txt"
        atomic_write_text(target, "hello")
        assert target.read_text(encoding="utf-8") == "hello"

    def test_overwrites_existing_content(self, tmp_path: Path) -> None:
        """A successful write replaces the previous file."""
        target = tmp_path / "out.txt"
        target.write_text("old", encoding="utf-8")
        atomic_write_text(target, "new")
        assert target.read_text(encoding="utf-8") == "new"

    def test_leaves_no_temporary_file_behind(self, tmp_path: Path) -> None:
        """The scratch file is renamed away, never left in the directory."""
        target = tmp_path / "out.txt"
        atomic_write_text(target, "hello")
        assert [path.name for path in tmp_path.iterdir()] == ["out.txt"]

    def test_previous_content_survives_a_crash(self, tmp_path: Path, monkeypatch) -> None:
        """This is the property the base's truncating writer does not have."""
        target = tmp_path / "out.json"
        target.write_text('{"papers": [1, 2, 3]}', encoding="utf-8")

        def exploding_replace(source, destination):  # type: ignore[no-untyped-def]
            raise OSError("simulated crash")

        monkeypatch.setattr(os, "replace", exploding_replace)
        with pytest.raises(OSError, match="simulated crash"):
            atomic_write_text(target, "x" * 100_000)

        assert target.read_text(encoding="utf-8") == '{"papers": [1, 2, 3]}'
        assert [path.name for path in tmp_path.iterdir()] == ["out.json"]

    def test_serialization_failure_leaves_no_file(self, tmp_path: Path) -> None:
        """Serializing before touching the destination keeps it intact."""
        target = tmp_path / "out.json"
        with pytest.raises(TypeError):
            atomic_write_json(target, {"bad": object()})
        assert not target.exists()

    def test_temporary_file_is_removed_when_writing_fails(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A failure inside the write does not leak a scratch file."""
        target = tmp_path / "out.txt"
        real_fsync = os.fsync

        def exploding_fsync(descriptor):  # type: ignore[no-untyped-def]
            raise OSError("simulated disk failure")

        monkeypatch.setattr(os, "fsync", exploding_fsync)
        with pytest.raises(OSError, match="simulated disk failure"):
            atomic_write_text(target, "hello")
        monkeypatch.setattr(os, "fsync", real_fsync)

        assert list(tmp_path.iterdir()) == []


class TestAtomicWriteJson:
    """JSON is serialized in full before the destination is touched."""

    def test_writes_readable_json(self, tmp_path: Path) -> None:
        """The result parses back to the original payload."""
        target = tmp_path / "out.json"
        atomic_write_json(target, {"a": 1, "b": ["x"]})
        assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": ["x"]}

    def test_preserves_non_ascii(self, tmp_path: Path) -> None:
        """Non-ASCII characters are written as themselves, not escaped.

        The non-ASCII characters are spelled as escapes so that the *source*
        stays ASCII, which the repository's own style gate requires, while the
        *data* under test is genuinely non-ASCII.  Writing them literally would
        test the same thing and fail ``tools/check_ascii.py``.
        """
        target = tmp_path / "out.json"
        word = "\u00fcberpr\u00fcfung"
        atomic_write_json(target, {"title": f"Prufung uber {word}"})
        assert word in target.read_text(encoding="utf-8")


class TestExistingSaversAreAtomic:
    """The library's own savers inherit the guarantee."""

    def test_save_to_json_survives_a_crash(self, tmp_path: Path, monkeypatch) -> None:
        """save_to_json no longer truncates before it serializes."""
        from lit_search.core.paper import Paper
        from lit_search.core.search_result import SearchResult
        from lit_search.utils.persistence import load_from_json, save_to_json

        target = tmp_path / "result.json"
        first = SearchResult(
            query="[a]",
            papers=[
                Paper(
                    title="Kept",
                    abstract="",
                    authors=[],
                    source=None,
                    publication_date=None,
                    doi="10.1/1",
                )
            ],
        )
        save_to_json(first, str(target))

        def exploding_replace(source, destination):  # type: ignore[no-untyped-def]
            raise OSError("simulated crash")

        monkeypatch.setattr(os, "replace", exploding_replace)
        with pytest.raises(OSError, match="simulated crash"):
            save_to_json(SearchResult(query="[b]"), str(target))

        reloaded = load_from_json(str(target))
        assert isinstance(reloaded, SearchResult)
        assert [paper.title for paper in reloaded.papers] == ["Kept"]
