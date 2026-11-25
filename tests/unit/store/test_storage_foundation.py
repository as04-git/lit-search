"""Composition checks for cache, run state, and immutable handles."""

from __future__ import annotations

from pathlib import Path

from lit_search.store import HandleStore, RunStateStore, SnapshotCache


def test_operational_stores_share_a_root_without_becoming_handle_storage(
    tmp_path: Path,
) -> None:
    """Cache and checkpoints occupy distinct paths beside immutable handles."""
    handles = HandleStore(tmp_path)
    cache = SnapshotCache(tmp_path)

    with RunStateStore.create(
        tmp_path,
        {"mode": "exhaustive", "max_depth": 2},
        run_id="r_0123456789abcdef0123456789abcdef",
    ) as run:
        assert run.path == tmp_path / "runs" / run.run_id / "run.sqlite"

    assert cache.path == tmp_path / "graph-cache" / "cache.sqlite"
    assert handles.handles_directory == tmp_path / "handles"
    assert handles.list_ids() == []
    assert not handles.handles_directory.exists()
