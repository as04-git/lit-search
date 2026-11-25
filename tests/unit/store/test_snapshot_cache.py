"""Falsifying tests for the bounded source-snapshot cache."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from lit_search.store.snapshot_cache import (
    DEFAULT_BUDGET_BYTES,
    MAX_BUDGET_BYTES,
    SCHEMA_VERSION,
    PhysicalUsage,
    SnapshotCache,
    SnapshotCacheSchemaError,
    SnapshotCacheValueError,
    SnapshotQuality,
)


def _graph(
    cache: SnapshotCache,
    *,
    native_key: str = "semantic_scholar:s2-a",
    payload: object | None = None,
    quality: SnapshotQuality = SnapshotQuality.COMPLETE,
    reported: int | None = 1,
    returned: int = 1,
    fetched_at: float = 100.0,
    ttl: float = 10.0,
    aliases: tuple[str, ...] = ("doi:10.1000/a",),
):
    """Write one Semantic Scholar backward snapshot for concise tests."""
    return cache.put_graph_snapshot(
        source="semantic_scholar",
        native_key=native_key,
        direction="backward",
        request_shape="edges/v1:uncapped",
        payload={"edges": ["semantic_scholar:s2-b"]} if payload is None else payload,
        quality=quality,
        reported_count=reported,
        returned_count=returned,
        ttl_seconds=ttl,
        fetched_at=fetched_at,
        aliases=aliases,
    )


def _read_graph(cache: SnapshotCache, **kwargs):
    """Read the default graph key while allowing policy overrides."""
    return cache.get_graph_snapshot(
        source="semantic_scholar",
        native_key=kwargs.pop("native_key", "semantic_scholar:s2-a"),
        direction="backward",
        request_shape="edges/v1:uncapped",
        **kwargs,
    )


def _noise(items: int) -> dict[str, list[str]]:
    """Return deterministic, poorly compressible JSON for budget tests."""
    return {
        "values": [
            hashlib.sha256(f"snapshot-{index}".encode()).hexdigest() for index in range(items)
        ]
    }


class TestSchemaAndConfiguration:
    """The on-disk meaning and owner ceiling must be explicit."""

    def test_database_lives_under_store_root_and_uses_wal(self, tmp_path: Path) -> None:
        cache = SnapshotCache(tmp_path)

        assert cache.path == tmp_path / "graph-cache" / "cache.sqlite"
        with cache._connect() as connection:
            assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            rows = dict(connection.execute("SELECT key, value FROM cache_meta"))
        assert rows["schema_version"] == str(SCHEMA_VERSION)

    def test_default_budget_is_one_gib_and_above_five_gib_is_refused(self, tmp_path: Path) -> None:
        assert SnapshotCache(tmp_path / "default").budget_bytes == DEFAULT_BUDGET_BYTES
        with pytest.raises(SnapshotCacheValueError, match="5 GiB"):
            SnapshotCache(tmp_path / "too-large", budget_bytes=MAX_BUDGET_BYTES + 1)

    def test_unknown_schema_version_is_refused(self, tmp_path: Path) -> None:
        cache = SnapshotCache(tmp_path)
        with sqlite3.connect(cache.path) as connection:
            connection.execute(
                "UPDATE cache_meta SET value = ? WHERE key = 'schema_version'", ("999",)
            )

        with pytest.raises(SnapshotCacheSchemaError, match="no migration"):
            SnapshotCache(tmp_path)

    def test_existing_unknown_sqlite_database_is_not_adopted(self, tmp_path: Path) -> None:
        path = tmp_path / "graph-cache" / "cache.sqlite"
        path.parent.mkdir(parents=True)
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE someone_elses_data(value TEXT)")

        with pytest.raises(SnapshotCacheSchemaError, match="refusing to claim"):
            SnapshotCache(tmp_path)

        with sqlite3.connect(path) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
        assert tables == {"someone_elses_data"}


class TestAliases:
    """Alias conflicts remain visible instead of last-writer winning."""

    def test_conflicting_source_native_identities_are_both_retained(self, tmp_path: Path) -> None:
        cache = SnapshotCache(tmp_path)

        assert _graph(
            cache,
            native_key="s2-a",
            aliases=("DOI:10.1000/A",),
            fetched_at=1,
        ).stored
        assert _graph(
            cache,
            native_key="s2-b",
            aliases=("doi:10.1000/a",),
            fetched_at=2,
        ).stored
        second = cache.resolve_alias("semantic_scholar", "doi:10.1000/a")

        assert second.native_keys == ("s2-a", "s2-b")
        assert second.conflict

    def test_alias_requires_a_typed_namespace(self, tmp_path: Path) -> None:
        cache = SnapshotCache(tmp_path)
        with pytest.raises(SnapshotCacheValueError, match="namespace-qualified"):
            _graph(cache, aliases=("10.1000/a",))

    def test_refresh_replaces_owned_aliases_instead_of_growing_an_unbounded_side_table(
        self, tmp_path: Path
    ) -> None:
        cache = SnapshotCache(tmp_path)
        assert _graph(cache, aliases=("doi:10.1000/a", "arxiv:2601.00001")).stored
        assert _graph(cache, aliases=("doi:10.1000/a",), fetched_at=105).stored

        assert cache.resolve_alias("semantic_scholar", "arxiv:2601.00001").native_keys == ()
        assert cache.resolve_alias("semantic_scholar", "doi:10.1000/a").native_keys == (
            "semantic_scholar:s2-a",
        )

    def test_many_aliases_are_budgeted_and_refusal_leaves_no_side_rows(
        self, tmp_path: Path
    ) -> None:
        cache = SnapshotCache(tmp_path, budget_bytes=128 * 1024)
        aliases = tuple(f"doi:10.1000/generated-{index}" for index in range(2_000))

        refused = _graph(cache, aliases=aliases)

        assert not refused.stored
        assert cache.resolve_alias("semantic_scholar", aliases[0]).native_keys == ()
        assert cache.usage().by_tier.get("graph", 0) == 0

    def test_post_write_physical_overrun_rolls_back_aliases_and_snapshot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = SnapshotCache(tmp_path, budget_bytes=128 * 1024)
        monkeypatch.setattr(cache, "_physical_room", lambda *_: True)
        monkeypatch.setattr(
            cache,
            "physical_usage",
            lambda: PhysicalUsage(
                database_bytes=cache.budget_bytes + 1,
                wal_bytes=0,
                database_apparent_bytes=cache.budget_bytes + 1,
                wal_apparent_bytes=0,
            ),
        )

        refused = _graph(cache, aliases=("doi:10.1000/a",))

        assert not refused.stored
        assert cache.resolve_alias("semantic_scholar", "doi:10.1000/a").native_keys == ()
        assert _read_graph(cache, now=101) is None


class TestGraphFreshnessAndQuality:
    """A lower bound, stale observation, and complete snapshot are not equivalent."""

    def test_expired_is_a_miss_unless_stale_is_explicit(self, tmp_path: Path) -> None:
        cache = SnapshotCache(tmp_path)
        assert _graph(cache).stored

        assert _read_graph(cache, now=110.0) is None
        stale = _read_graph(cache, now=110.0, allow_stale=True)

        assert stale is not None
        assert stale.stale
        assert stale.is_lower_bound
        assert not stale.satisfies_complete

    def test_partial_is_reusable_but_cannot_satisfy_require_complete(self, tmp_path: Path) -> None:
        cache = SnapshotCache(tmp_path)
        assert _graph(
            cache,
            quality=SnapshotQuality.PARTIAL,
            reported=10,
            returned=7,
        ).stored

        lower_bound = _read_graph(cache, now=105.0)

        assert lower_bound is not None
        assert lower_bound.quality is SnapshotQuality.PARTIAL
        assert lower_bound.is_lower_bound
        assert _read_graph(cache, now=105.0, require_complete=True) is None

    def test_complete_zero_is_valid_but_uncorroborated_complete_is_refused(
        self, tmp_path: Path
    ) -> None:
        cache = SnapshotCache(tmp_path)

        assert _graph(cache, payload={"edges": []}, reported=0, returned=0).stored
        complete_zero = _read_graph(cache, now=105.0, require_complete=True)
        assert complete_zero is not None
        assert complete_zero.payload == {"edges": []}

        with pytest.raises(SnapshotCacheValueError, match="equal reported_count"):
            _graph(cache, reported=None, returned=0)

    def test_refresh_replaces_source_snapshot_instead_of_unioning_edges(
        self, tmp_path: Path
    ) -> None:
        cache = SnapshotCache(tmp_path)
        assert _graph(cache, payload={"edges": ["old"]}).stored
        assert _graph(cache, payload={"edges": ["new"]}, fetched_at=106.0).stored

        refreshed = _read_graph(cache, now=107.0)

        assert refreshed is not None
        assert refreshed.payload == {"edges": ["new"]}


class TestMetadataAndNegatives:
    """Retrieval metadata and explicit negatives carry independent freshness."""

    def test_metadata_field_sets_do_not_substitute_for_one_another(self, tmp_path: Path) -> None:
        cache = SnapshotCache(tmp_path)
        result = cache.put_metadata_snapshot(
            source="crossref",
            native_key="doi:10.1000/a",
            field_set="abstract+authors/v1",
            payload={"abstract": "A"},
            quality=SnapshotQuality.COMPLETE,
            ttl_seconds=100,
            fetched_at=10,
        )

        assert result.stored
        assert (
            cache.get_metadata_snapshot(
                source="crossref",
                native_key="doi:10.1000/a",
                field_set="abstract+authors/v1",
                now=11,
            )
            is not None
        )
        assert (
            cache.get_metadata_snapshot(
                source="crossref",
                native_key="doi:10.1000/a",
                field_set="abstract+authors+urls/v1",
                now=11,
            )
            is None
        )

    def test_not_found_expires_and_failure_cannot_poison_negative_cache(
        self, tmp_path: Path
    ) -> None:
        cache = SnapshotCache(tmp_path)
        result = cache.record_negative(
            source="semantic_scholar",
            requested_key="doi:10.1000/missing",
            outcome="not_found",
            ttl_seconds=5,
            detail="HTTP 404",
            observed_at=20,
        )

        assert result.stored
        hit = cache.get_not_found("semantic_scholar", "doi:10.1000/missing", now=24)
        assert hit is not None
        assert hit.detail == "HTTP 404"
        assert cache.get_not_found("semantic_scholar", "doi:10.1000/missing", now=25) is None

        with pytest.raises(SnapshotCacheValueError, match="failures remain retryable"):
            cache.record_negative(
                source="semantic_scholar",
                requested_key="doi:10.1000/failed",
                outcome="failed",
                ttl_seconds=5,
            )


class TestBudgetAndEviction:
    """Bounded storage prefers disposable tiers and refuses without data loss."""

    def test_prune_evicts_metadata_before_fresh_graph_even_when_graph_is_older(
        self, tmp_path: Path
    ) -> None:
        cache = SnapshotCache(tmp_path, budget_bytes=2 * 1024**2)
        assert _graph(cache, fetched_at=1).stored
        metadata = cache.put_metadata_snapshot(
            source="crossref",
            native_key="doi:10.1000/a",
            field_set="abstract/v1",
            payload={"abstract": "A"},
            quality=SnapshotQuality.COMPLETE,
            ttl_seconds=1_000,
            fetched_at=2,
        )
        assert metadata.stored
        graph_bytes = cache.usage().by_tier["graph"]

        pruned = cache.prune(graph_bytes, now=3)

        assert pruned.evicted[0].startswith("metadata:")
        assert _read_graph(cache, now=3) is not None

    def test_expired_negative_is_evicted_before_approved_snapshot_tiers(
        self, tmp_path: Path
    ) -> None:
        cache = SnapshotCache(tmp_path, budget_bytes=2 * 1024**2)
        assert _graph(cache, fetched_at=1, ttl=1_000).stored
        negative = cache.record_negative(
            source="semantic_scholar",
            requested_key="doi:10.1000/missing",
            outcome="not_found",
            ttl_seconds=1,
            observed_at=1,
        )
        assert negative.stored
        graph_bytes = cache.usage().by_tier["graph"]

        pruned = cache.prune(graph_bytes, now=2)

        assert pruned.evicted[0].startswith("negative:")
        assert _read_graph(cache, now=2) is not None

    def test_fresh_negative_is_evicted_after_fresh_graph(self, tmp_path: Path) -> None:
        cache = SnapshotCache(tmp_path, budget_bytes=2 * 1024**2)
        assert _graph(cache, fetched_at=1, ttl=1_000).stored
        negative = cache.record_negative(
            source="semantic_scholar",
            requested_key="doi:10.1000/missing",
            outcome="not_found",
            ttl_seconds=1_000,
            observed_at=1,
        )
        assert negative.stored
        negative_bytes = cache.usage().by_tier["negative"]

        pruned = cache.prune(negative_bytes, now=2)

        assert pruned.evicted[0].startswith("graph:")
        assert cache.get_not_found("semantic_scholar", "doi:10.1000/missing", now=2) is not None

    def test_over_budget_write_is_explicit_and_preserves_previous_snapshot(
        self, tmp_path: Path
    ) -> None:
        cache = SnapshotCache(tmp_path, budget_bytes=256 * 1024)
        assert _graph(cache, payload={"edges": ["old"]}).stored

        refused = _graph(cache, payload=_noise(10_000), fetched_at=105)

        assert not refused.stored
        assert refused.reason is not None
        assert "not cached" in refused.reason
        preserved = _read_graph(cache, now=106)
        assert preserved is not None
        assert preserved.payload == {"edges": ["old"]}

    def test_write_pressure_evicts_to_low_water_with_deterministic_tier_order(
        self, tmp_path: Path
    ) -> None:
        cache = SnapshotCache(tmp_path, budget_bytes=512 * 1024, low_watermark=0.90)
        assert _graph(cache, native_key="semantic_scholar:s2-a", payload=_noise(5_000)).stored
        metadata = cache.put_metadata_snapshot(
            source="crossref",
            native_key="doi:10.1000/b",
            field_set="full/v1",
            payload=_noise(5_000),
            quality=SnapshotQuality.COMPLETE,
            ttl_seconds=1_000,
            fetched_at=101,
        )
        assert metadata.stored

        pressured = _graph(
            cache,
            native_key="semantic_scholar:s2-c",
            payload=_noise(5_000),
            fetched_at=102,
            aliases=("doi:10.1000/c",),
        )

        assert pressured.stored
        assert pressured.evicted
        assert pressured.evicted[0].startswith("metadata:")
        assert pressured.usage.logical_bytes <= cache.budget_bytes
        assert pressured.usage.physical.total_bytes <= cache.budget_bytes

    def test_physical_usage_observes_database_and_wal(self, tmp_path: Path) -> None:
        cache = SnapshotCache(tmp_path)
        assert _graph(cache).stored

        usage = cache.physical_usage()

        assert usage.database_apparent_bytes == cache.path.stat().st_size
        assert usage.database_bytes == cache.path.stat().st_blocks * 512
        assert usage.wal_bytes >= 0
        assert usage.total_bytes >= usage.database_bytes > 0
