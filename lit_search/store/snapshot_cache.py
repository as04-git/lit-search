"""Bounded, source-snapshot cache for citation traversal.

The cache deliberately stores a source's latest observation, not a merged
``Paper``.  A refresh can therefore remove an obsolete edge or correct a
reported total.  Immutable handles remain the publication boundary; this
module is disposable operational acceleration underneath them.

SQLite provides the cross-process transaction and crash boundary.  Network
requests must happen outside these transactions: callers fetch first and then
offer the successful payload to :class:`SnapshotCache`.  A budget refusal is a
normal :class:`CacheWriteResult`, never an exception that could erase the
successful network result.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from lit_search.exceptions import PersistenceError

SCHEMA_NAME = "lit-search/source-snapshot-cache"
SCHEMA_VERSION = 1

DEFAULT_BUDGET_BYTES = 1024**3
MAX_BUDGET_BYTES = 5 * 1024**3
DEFAULT_LOW_WATERMARK = 0.90
DEFAULT_BUSY_TIMEOUT_MS = 5_000

_CACHE_SUBDIRECTORY = "graph-cache"
_CACHE_FILENAME = "cache.sqlite"
_ENTRY_OVERHEAD_BYTES = 512
_ALIAS_OVERHEAD_BYTES = 128
_MIN_HEADROOM_BYTES = 16 * 1024
_MAX_HEADROOM_BYTES = 64 * 1024**2


class SnapshotCacheError(PersistenceError, RuntimeError):
    """Base error for a cache that cannot safely honour its contract."""


class SnapshotCacheSchemaError(SnapshotCacheError):
    """Raised when an existing cache has an unknown schema or version."""


class SnapshotCacheValueError(SnapshotCacheError, ValueError):
    """Raised when a caller attempts to write an internally false record."""


class SnapshotQuality(StrEnum):
    """How much of the requested source snapshot was observed."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    TRUNCATED = "truncated"


class CacheTier(StrEnum):
    """Managed-data tiers in eviction order."""

    NEGATIVE = "negative"
    DERIVED = "derived"
    METADATA = "metadata"
    GRAPH = "graph"


@dataclass(frozen=True, slots=True)
class SnapshotRead:
    """One graph or metadata snapshot returned by the cache."""

    payload: Any
    quality: SnapshotQuality
    reported_count: int | None
    returned_count: int | None
    fetched_at: float
    expires_at: float
    digest: str
    stale: bool

    @property
    def is_lower_bound(self) -> bool:
        """Whether this observation cannot support a current completeness claim."""
        return self.stale or self.quality is not SnapshotQuality.COMPLETE

    @property
    def satisfies_complete(self) -> bool:
        """Whether this is fresh evidence of a complete source response."""
        return not self.stale and self.quality is SnapshotQuality.COMPLETE


@dataclass(frozen=True, slots=True)
class NegativeRead:
    """A fresh, explicit source not-found observation."""

    source: str
    requested_key: str
    detail: str | None
    observed_at: float
    expires_at: float


@dataclass(frozen=True, slots=True)
class AliasResolution:
    """All source-native identities asserted for one typed alias."""

    source: str
    alias_key: str
    native_keys: tuple[str, ...]

    @property
    def conflict(self) -> bool:
        """Whether the source has associated this alias with multiple identities."""
        return len(self.native_keys) > 1


@dataclass(frozen=True, slots=True)
class PhysicalUsage:
    """Observed SQLite storage, including the active WAL."""

    database_bytes: int
    wal_bytes: int
    database_apparent_bytes: int
    wal_apparent_bytes: int

    @property
    def total_bytes(self) -> int:
        """Return allocated database plus WAL bytes."""
        return self.database_bytes + self.wal_bytes

    @property
    def apparent_total_bytes(self) -> int:
        """Return logical file lengths, which may exceed allocated sparse blocks."""
        return self.database_apparent_bytes + self.wal_apparent_bytes


@dataclass(frozen=True, slots=True)
class CacheUsage:
    """Logical managed data and observed physical SQLite usage."""

    logical_bytes: int
    entries: int
    by_tier: dict[str, int]
    physical: PhysicalUsage


@dataclass(frozen=True, slots=True)
class CacheWriteResult:
    """Outcome of offering one already-fetched payload to the cache."""

    stored: bool
    reason: str | None
    evicted: tuple[str, ...]
    usage: CacheUsage


@dataclass(frozen=True, slots=True)
class PruneResult:
    """Outcome of deterministic cache pruning."""

    evicted: tuple[str, ...]
    usage: CacheUsage


@dataclass(frozen=True, slots=True)
class _EncodedPayload:
    """Canonical and compressed JSON prepared before opening a transaction."""

    blob: bytes
    digest: str
    logical_bytes: int


class _BudgetRefusal(Exception):
    """Internal transaction-abort signal carrying a user-facing reason."""


class SnapshotCache:
    """Schema-versioned, bounded cache rooted beside immutable handles."""

    def __init__(
        self,
        store_root: Path | str,
        *,
        budget_bytes: int = DEFAULT_BUDGET_BYTES,
        low_watermark: float = DEFAULT_LOW_WATERMARK,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    ) -> None:
        """Open or create a cache without changing any result handle.

        ``budget_bytes`` applies only to disposable managed data.  Immutable
        handles are reported and managed elsewhere and are never evicted here.
        """
        if budget_bytes <= 0:
            raise SnapshotCacheValueError("budget_bytes must be positive")
        if budget_bytes > MAX_BUDGET_BYTES:
            raise SnapshotCacheValueError(f"budget_bytes may not exceed {MAX_BUDGET_BYTES} (5 GiB)")
        if not 0 < low_watermark < 1:
            raise SnapshotCacheValueError("low_watermark must be between 0 and 1")
        if busy_timeout_ms < 0:
            raise SnapshotCacheValueError("busy_timeout_ms must not be negative")

        self._store_root = Path(store_root).expanduser()
        self._directory = self._store_root / _CACHE_SUBDIRECTORY
        self._path = self._directory / _CACHE_FILENAME
        self._budget_bytes = budget_bytes
        self._low_watermark = low_watermark
        self._busy_timeout_ms = busy_timeout_ms
        self._directory.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @property
    def path(self) -> Path:
        """Return the SQLite database path."""
        return self._path

    @property
    def budget_bytes(self) -> int:
        """Return the configured managed-data ceiling."""
        return self._budget_bytes

    def _connect(self) -> sqlite3.Connection:
        """Open one configured connection with no hidden long-lived transaction."""
        connection = sqlite3.connect(
            self._path,
            timeout=self._busy_timeout_ms / 1000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA wal_autocheckpoint = 100")
        return connection

    def _initialize(self) -> None:
        """Create the schema transactionally or refuse an unknown one."""
        with self._connect() as connection:
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
            }
            if tables:
                if "cache_meta" not in tables:
                    raise SnapshotCacheSchemaError(
                        f"existing SQLite database at {self._path} has tables {sorted(tables)} "
                        "but no cache_meta; refusing to claim an unknown database"
                    )
                metadata = dict(connection.execute("SELECT key, value FROM cache_meta").fetchall())
                self._check_schema(metadata)
                if tables != _EXPECTED_TABLES:
                    raise SnapshotCacheSchemaError(
                        f"cache schema {SCHEMA_VERSION} requires tables "
                        f"{sorted(_EXPECTED_TABLES)}, found {sorted(tables)}"
                    )
                connection.execute("PRAGMA journal_mode = WAL")
                return
            connection.execute("PRAGMA journal_mode = WAL")
            try:
                # ``executescript`` commits any pending transaction before it
                # starts.  Put BEGIN in the script itself so schema creation is
                # one real transaction rather than a collection of autocommits.
                connection.executescript(f"BEGIN IMMEDIATE;\n{_SCHEMA_SQL}")
                rows = dict(connection.execute("SELECT key, value FROM cache_meta").fetchall())
                if not rows:
                    connection.executemany(
                        "INSERT INTO cache_meta(key, value) VALUES (?, ?)",
                        (("schema", SCHEMA_NAME), ("schema_version", str(SCHEMA_VERSION))),
                    )
                else:
                    self._check_schema(rows)
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    @staticmethod
    def _check_schema(metadata: dict[str, str]) -> None:
        """Reject a cache whose persisted meaning this reader does not know."""
        if metadata.get("schema") != SCHEMA_NAME:
            raise SnapshotCacheSchemaError(
                f"cache schema is {metadata.get('schema')!r}, expected {SCHEMA_NAME!r}"
            )
        raw_version = metadata.get("schema_version")
        if raw_version != str(SCHEMA_VERSION):
            raise SnapshotCacheSchemaError(
                f"cache schema version is {raw_version!r}, expected {SCHEMA_VERSION}; "
                "no migration is registered"
            )

    # ------------------------------------------------------------------
    # Typed aliases
    # ------------------------------------------------------------------

    def resolve_alias(self, source: str, alias_key: str) -> AliasResolution:
        """Resolve a typed alias, returning every conflict deterministically."""
        source = _required("source", source)
        alias_key = _typed_alias(alias_key)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT native_key FROM aliases
                WHERE source = ? AND alias_key = ?
                ORDER BY native_key
                """,
                (source, alias_key),
            ).fetchall()
        return AliasResolution(source, alias_key, tuple(str(row[0]) for row in rows))

    # ------------------------------------------------------------------
    # Graph snapshots
    # ------------------------------------------------------------------

    def put_graph_snapshot(
        self,
        *,
        source: str,
        native_key: str,
        direction: str,
        request_shape: str,
        payload: Any,
        quality: SnapshotQuality,
        reported_count: int | None,
        returned_count: int,
        ttl_seconds: float,
        fetched_at: float | None = None,
        aliases: tuple[str, ...] = (),
    ) -> CacheWriteResult:
        """Replace one source/direction observation, never merge it with the old one."""
        source = _required("source", source)
        native_key = _required("native_key", native_key)
        direction = _direction(direction)
        request_shape = _required("request_shape", request_shape)
        quality = SnapshotQuality(quality)
        _validate_counts(quality, reported_count, returned_count)
        stamp, expires = _freshness(fetched_at, ttl_seconds)
        alias_keys = tuple(_typed_alias(alias) for alias in aliases)
        encoded = _encode(payload)
        encoded = _EncodedPayload(
            blob=encoded.blob,
            digest=encoded.digest,
            logical_bytes=encoded.logical_bytes
            + sum(len(alias.encode("utf-8")) + _ALIAS_OVERHEAD_BYTES for alias in alias_keys),
        )
        key = (source, native_key, direction, request_shape)

        def write(connection: sqlite3.Connection, entry_id: int) -> None:
            connection.execute(
                """
                INSERT INTO graph_snapshots(
                    entry_id, source, native_key, direction, request_shape
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source, native_key, direction, request_shape)
                DO UPDATE SET entry_id = excluded.entry_id
                """,
                (entry_id, *key),
            )
            connection.execute("DELETE FROM aliases WHERE entry_id = ?", (entry_id,))
            for alias in alias_keys:
                connection.execute(
                    """
                    INSERT INTO aliases(
                        entry_id, source, alias_key, native_key, first_seen_at, last_seen_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (entry_id, source, alias, native_key, stamp, stamp),
                )

        return self._put_snapshot(
            tier=CacheTier.GRAPH,
            lookup_table="graph_snapshots",
            lookup_where=(
                "source = ? AND native_key = ? AND direction = ? AND request_shape = ?",
                key,
            ),
            encoded=encoded,
            quality=quality,
            reported_count=reported_count,
            returned_count=returned_count,
            fetched_at=stamp,
            expires_at=expires,
            writer=write,
        )

    def get_graph_snapshot(
        self,
        *,
        source: str,
        native_key: str,
        direction: str,
        request_shape: str,
        allow_stale: bool = False,
        require_complete: bool = False,
        now: float | None = None,
    ) -> SnapshotRead | None:
        """Read a graph observation under explicit stale/completeness policy."""
        key = (
            _required("source", source),
            _required("native_key", native_key),
            _direction(direction),
            _required("request_shape", request_shape),
        )
        return self._get_snapshot(
            table="graph_snapshots",
            where="source = ? AND native_key = ? AND direction = ? AND request_shape = ?",
            parameters=key,
            allow_stale=allow_stale,
            require_complete=require_complete,
            now=_timestamp(now),
        )

    # ------------------------------------------------------------------
    # Retrieval metadata snapshots
    # ------------------------------------------------------------------

    def put_metadata_snapshot(
        self,
        *,
        source: str,
        native_key: str,
        field_set: str,
        payload: Any,
        quality: SnapshotQuality,
        ttl_seconds: float,
        fetched_at: float | None = None,
    ) -> CacheWriteResult:
        """Replace one independently fresh retrieval-metadata field set."""
        source = _required("source", source)
        native_key = _required("native_key", native_key)
        field_set = _required("field_set", field_set)
        quality = SnapshotQuality(quality)
        stamp, expires = _freshness(fetched_at, ttl_seconds)
        encoded = _encode(payload)
        key = (source, native_key, field_set)

        def write(connection: sqlite3.Connection, entry_id: int) -> None:
            connection.execute(
                """
                INSERT INTO metadata_snapshots(entry_id, source, native_key, field_set)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(source, native_key, field_set)
                DO UPDATE SET entry_id = excluded.entry_id
                """,
                (entry_id, *key),
            )

        return self._put_snapshot(
            tier=CacheTier.METADATA,
            lookup_table="metadata_snapshots",
            lookup_where=("source = ? AND native_key = ? AND field_set = ?", key),
            encoded=encoded,
            quality=quality,
            reported_count=None,
            returned_count=None,
            fetched_at=stamp,
            expires_at=expires,
            writer=write,
        )

    def get_metadata_snapshot(
        self,
        *,
        source: str,
        native_key: str,
        field_set: str,
        allow_stale: bool = False,
        require_complete: bool = False,
        now: float | None = None,
    ) -> SnapshotRead | None:
        """Read metadata without letting one field set satisfy another."""
        key = (
            _required("source", source),
            _required("native_key", native_key),
            _required("field_set", field_set),
        )
        return self._get_snapshot(
            table="metadata_snapshots",
            where="source = ? AND native_key = ? AND field_set = ?",
            parameters=key,
            allow_stale=allow_stale,
            require_complete=require_complete,
            now=_timestamp(now),
        )

    # ------------------------------------------------------------------
    # Explicit negative cache
    # ------------------------------------------------------------------

    def record_negative(
        self,
        *,
        source: str,
        requested_key: str,
        outcome: str,
        ttl_seconds: float,
        detail: str | None = None,
        observed_at: float | None = None,
    ) -> CacheWriteResult:
        """Cache only an explicit not-found; failures are rejected loudly."""
        if outcome != "not_found":
            raise SnapshotCacheValueError(
                "only explicit 'not_found' outcomes are negative-cacheable; "
                "failures remain retryable evidence"
            )
        source = _required("source", source)
        requested_key = _typed_alias(requested_key)
        stamp, expires = _freshness(observed_at, ttl_seconds)
        encoded = _encode({"detail": detail})
        key = (source, requested_key)

        def write(connection: sqlite3.Connection, entry_id: int) -> None:
            connection.execute(
                """
                INSERT INTO negative_outcomes(entry_id, source, requested_key)
                VALUES (?, ?, ?)
                ON CONFLICT(source, requested_key)
                DO UPDATE SET entry_id = excluded.entry_id
                """,
                (entry_id, *key),
            )

        return self._put_snapshot(
            tier=CacheTier.NEGATIVE,
            lookup_table="negative_outcomes",
            lookup_where=("source = ? AND requested_key = ?", key),
            encoded=encoded,
            quality=None,
            reported_count=None,
            returned_count=None,
            fetched_at=stamp,
            expires_at=expires,
            writer=write,
        )

    def get_not_found(
        self,
        source: str,
        requested_key: str,
        *,
        now: float | None = None,
    ) -> NegativeRead | None:
        """Return a fresh explicit not-found; expired negatives are always misses."""
        source = _required("source", source)
        requested_key = _typed_alias(requested_key)
        stamp = _timestamp(now)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT e.entry_id, e.payload, e.fetched_at, e.expires_at
                FROM negative_outcomes n
                JOIN cache_entries e ON e.entry_id = n.entry_id
                WHERE n.source = ? AND n.requested_key = ?
                """,
                (source, requested_key),
            ).fetchone()
            if row is None or stamp >= float(row["expires_at"]):
                return None
            connection.execute(
                "UPDATE cache_entries SET last_accessed = ? WHERE entry_id = ?",
                (stamp, int(row["entry_id"])),
            )
            payload = _decode(bytes(row["payload"]))
        detail = payload.get("detail") if isinstance(payload, dict) else None
        return NegativeRead(
            source=source,
            requested_key=requested_key,
            detail=str(detail) if detail is not None else None,
            observed_at=float(row["fetched_at"]),
            expires_at=float(row["expires_at"]),
        )

    # ------------------------------------------------------------------
    # Budget, eviction, and telemetry
    # ------------------------------------------------------------------

    def physical_usage(self) -> PhysicalUsage:
        """Observe database and WAL file sizes rather than infer them from payloads."""
        wal_path = Path(f"{self._path}-wal")
        database, database_apparent = _allocated_and_apparent(self._path)
        wal, wal_apparent = _allocated_and_apparent(wal_path)
        return PhysicalUsage(
            database_bytes=database,
            wal_bytes=wal,
            database_apparent_bytes=database_apparent,
            wal_apparent_bytes=wal_apparent,
        )

    def usage(self) -> CacheUsage:
        """Return managed logical bytes by tier and observed physical bytes."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT tier, COUNT(*) AS entries, COALESCE(SUM(logical_bytes), 0) AS bytes
                FROM cache_entries GROUP BY tier
                """
            ).fetchall()
        by_tier = {str(row["tier"]): int(row["bytes"]) for row in rows}
        return CacheUsage(
            logical_bytes=sum(by_tier.values()),
            entries=sum(int(row["entries"]) for row in rows),
            by_tier=by_tier,
            physical=self.physical_usage(),
        )

    def prune(self, target_bytes: int = 0, *, now: float | None = None) -> PruneResult:
        """Evict deterministically to at most ``target_bytes`` logical bytes."""
        if target_bytes < 0 or target_bytes > self._budget_bytes:
            raise SnapshotCacheValueError("target_bytes must be between zero and budget_bytes")
        stamp = _timestamp(now)
        evicted: list[str] = []
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                logical = self._logical_bytes(connection)
                while logical > target_bytes:
                    candidate = self._next_eviction(connection, stamp, excluded_entry_id=None)
                    if candidate is None:
                        break
                    evicted.append(self._delete_entry(connection, candidate))
                    logical = self._logical_bytes(connection)
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        self._checkpoint()
        return PruneResult(tuple(evicted), self.usage())

    def _put_snapshot(
        self,
        *,
        tier: CacheTier,
        lookup_table: str,
        lookup_where: tuple[str, tuple[str, ...]],
        encoded: _EncodedPayload,
        quality: SnapshotQuality | None,
        reported_count: int | None,
        returned_count: int | None,
        fetched_at: float,
        expires_at: float,
        writer: Callable[[sqlite3.Connection, int], None],
    ) -> CacheWriteResult:
        """Run one bounded replacement transaction and translate refusal to data."""
        evicted: list[str] = []
        reason: str | None = None
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                clause, parameters = lookup_where
                existing = connection.execute(
                    f"SELECT entry_id FROM {lookup_table} WHERE {clause}",
                    parameters,
                ).fetchone()
                existing_id = int(existing[0]) if existing is not None else None
                old_bytes = 0
                if existing_id is not None:
                    old_row = connection.execute(
                        "SELECT logical_bytes FROM cache_entries WHERE entry_id = ?",
                        (existing_id,),
                    ).fetchone()
                    old_bytes = int(old_row[0]) if old_row is not None else 0

                evicted.extend(
                    self._make_room(
                        connection,
                        incoming_bytes=encoded.logical_bytes,
                        replaced_bytes=old_bytes,
                        now=fetched_at,
                        excluded_entry_id=existing_id,
                    )
                )
                if existing_id is None:
                    cursor = connection.execute(
                        """
                        INSERT INTO cache_entries(
                            tier, quality, reported_count, returned_count, fetched_at,
                            expires_at, last_accessed, payload, payload_digest, logical_bytes
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            tier.value,
                            quality.value if quality is not None else None,
                            reported_count,
                            returned_count,
                            fetched_at,
                            expires_at,
                            fetched_at,
                            encoded.blob,
                            encoded.digest,
                            encoded.logical_bytes,
                        ),
                    )
                    if cursor.lastrowid is None:
                        raise SnapshotCacheError("SQLite did not return an entry id")
                    entry_id = int(cursor.lastrowid)
                else:
                    entry_id = existing_id
                    connection.execute(
                        """
                        UPDATE cache_entries SET
                            tier = ?, quality = ?, reported_count = ?, returned_count = ?,
                            fetched_at = ?, expires_at = ?, last_accessed = ?, payload = ?,
                            payload_digest = ?, logical_bytes = ?
                        WHERE entry_id = ?
                        """,
                        (
                            tier.value,
                            quality.value if quality is not None else None,
                            reported_count,
                            returned_count,
                            fetched_at,
                            expires_at,
                            fetched_at,
                            encoded.blob,
                            encoded.digest,
                            encoded.logical_bytes,
                            entry_id,
                        ),
                    )
                writer(connection, entry_id)
                if self.physical_usage().total_bytes > self._budget_bytes:
                    raise _BudgetRefusal(
                        "the transactional database/WAL grew beyond the physical budget; "
                        "the cache transaction was rolled back and the fetched payload was "
                        "not cached"
                    )
                connection.commit()
            except _BudgetRefusal as error:
                connection.rollback()
                evicted.clear()
                reason = str(error)
            except BaseException:
                connection.rollback()
                raise

        if reason is not None:
            self._checkpoint()
            return CacheWriteResult(False, reason, (), self.usage())
        self._checkpoint()
        return CacheWriteResult(True, None, tuple(evicted), self.usage())

    def _get_snapshot(
        self,
        *,
        table: str,
        where: str,
        parameters: tuple[str, ...],
        allow_stale: bool,
        require_complete: bool,
        now: float,
    ) -> SnapshotRead | None:
        """Shared policy-aware graph/metadata read path."""
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT e.* FROM {table} s
                JOIN cache_entries e ON e.entry_id = s.entry_id
                WHERE {where}
                """,
                parameters,
            ).fetchone()
            if row is None:
                return None
            stale = now >= float(row["expires_at"])
            quality = SnapshotQuality(str(row["quality"]))
            if stale and not allow_stale:
                return None
            if require_complete and (stale or quality is not SnapshotQuality.COMPLETE):
                return None
            connection.execute(
                "UPDATE cache_entries SET last_accessed = ? WHERE entry_id = ?",
                (now, int(row["entry_id"])),
            )
            payload = _decode(bytes(row["payload"]))
            digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
            if digest != str(row["payload_digest"]):
                raise SnapshotCacheSchemaError(
                    f"cache payload {row['entry_id']} failed its digest check"
                )
        return SnapshotRead(
            payload=payload,
            quality=quality,
            reported_count=(
                int(row["reported_count"]) if row["reported_count"] is not None else None
            ),
            returned_count=(
                int(row["returned_count"]) if row["returned_count"] is not None else None
            ),
            fetched_at=float(row["fetched_at"]),
            expires_at=float(row["expires_at"]),
            digest=str(row["payload_digest"]),
            stale=stale,
        )

    def _make_room(
        self,
        connection: sqlite3.Connection,
        *,
        incoming_bytes: int,
        replaced_bytes: int,
        now: float,
        excluded_entry_id: int | None,
    ) -> list[str]:
        """Evict to the low-water mark or refuse without modifying the cache."""
        if incoming_bytes > self._budget_bytes:
            raise _BudgetRefusal(
                f"snapshot needs {incoming_bytes} managed bytes, above the "
                f"{self._budget_bytes}-byte budget; the fetched payload was not cached"
            )
        current = self._logical_bytes(connection) - replaced_bytes
        projected = current + incoming_bytes
        target = self._budget_bytes
        if projected > self._budget_bytes:
            target = max(incoming_bytes, int(self._budget_bytes * self._low_watermark))

        evicted: list[str] = []
        while current + incoming_bytes > target or not self._physical_room(
            connection, incoming_bytes, replaced_bytes
        ):
            candidate = self._next_eviction(connection, now, excluded_entry_id)
            if candidate is None:
                raise _BudgetRefusal(
                    "physical database/WAL headroom or managed-data budget is exhausted; "
                    "the fetched payload remains available to the caller but was not cached"
                )
            evicted.append(self._delete_entry(connection, candidate))
            current = self._logical_bytes(connection) - replaced_bytes
        return evicted

    def _physical_room(
        self,
        connection: sqlite3.Connection,
        incoming_bytes: int,
        replaced_bytes: int,
    ) -> bool:
        """Conservatively include observed files, reusable pages, and WAL headroom."""
        page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
        free_pages = int(connection.execute("PRAGMA freelist_count").fetchone()[0])
        reusable = free_pages * page_size + replaced_bytes
        growth = max(0, incoming_bytes - reusable)
        headroom = min(
            _MAX_HEADROOM_BYTES,
            max(_MIN_HEADROOM_BYTES, self._budget_bytes // 20),
        )
        return self.physical_usage().total_bytes + growth + headroom <= self._budget_bytes

    @staticmethod
    def _logical_bytes(connection: sqlite3.Connection) -> int:
        """Return accounted compressed payload bytes plus row overhead."""
        return int(
            connection.execute(
                "SELECT COALESCE(SUM(logical_bytes), 0) FROM cache_entries"
            ).fetchone()[0]
        )

    @staticmethod
    def _next_eviction(
        connection: sqlite3.Connection,
        now: float,
        excluded_entry_id: int | None,
    ) -> sqlite3.Row | None:
        """Select the next victim under the approved deterministic tier order."""
        exclusion = "" if excluded_entry_id is None else "WHERE entry_id != ?"
        parameters: tuple[float | int, ...]
        if excluded_entry_id is None:
            parameters = (now, now, now, now)
        else:
            parameters = (excluded_entry_id, now, now, now, now)
        row = connection.execute(
            f"""
            SELECT entry_id, tier, payload_digest FROM cache_entries
            {exclusion}
            ORDER BY
                CASE
                    WHEN tier = 'negative' AND expires_at <= ? THEN 0
                    WHEN tier = 'derived' AND expires_at <= ? THEN 1
                    WHEN tier = 'derived' THEN 2
                    WHEN tier = 'metadata' AND expires_at <= ? THEN 3
                    WHEN tier = 'metadata' THEN 4
                    WHEN tier = 'graph' AND expires_at <= ? THEN 5
                    WHEN tier = 'graph' THEN 6
                    ELSE 7
                END,
                last_accessed ASC,
                entry_id ASC
            LIMIT 1
            """,
            parameters,
        ).fetchone()
        return cast("sqlite3.Row | None", row)

    @staticmethod
    def _delete_entry(connection: sqlite3.Connection, row: sqlite3.Row) -> str:
        """Delete one entry and return a stable audit label."""
        entry_id = int(row["entry_id"])
        label = f"{row['tier']}:{entry_id}:{str(row['payload_digest'])[:12]}"
        connection.execute("DELETE FROM cache_entries WHERE entry_id = ?", (entry_id,))
        return label

    def _checkpoint(self) -> None:
        """Bound WAL growth when no concurrent reader prevents a checkpoint."""
        try:
            with self._connect() as connection:
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.OperationalError:
            # The next writer still observes and budgets the live WAL.  A busy
            # reader delaying compaction is not data loss and must not convert a
            # successful cache write into a failed traversal.
            return


def _required(name: str, value: str) -> str:
    """Normalize a required string or reject it."""
    normalized = value.strip()
    if not normalized:
        raise SnapshotCacheValueError(f"{name} must not be blank")
    return normalized


def _typed_alias(value: str) -> str:
    """Require the namespace-qualified key shape used by ``PaperIdentifier``."""
    alias = _required("alias_key", value).lower()
    namespace, separator, identifier = alias.partition(":")
    if not separator or not namespace or not identifier:
        raise SnapshotCacheValueError(
            "alias_key must be namespace-qualified, for example 'doi:10.1000/example'"
        )
    return alias


def _direction(value: str) -> str:
    """Restrict graph snapshots to one independently fresh direction."""
    direction = value.strip().lower()
    if direction not in {"backward", "forward"}:
        raise SnapshotCacheValueError("direction must be 'backward' or 'forward'")
    return direction


def _timestamp(value: float | None) -> float:
    """Return a finite epoch timestamp, defaulting to the current wall clock."""
    stamp = time.time() if value is None else float(value)
    if stamp < 0 or not math.isfinite(stamp):
        raise SnapshotCacheValueError("timestamp must be a finite non-negative number")
    return stamp


def _freshness(fetched_at: float | None, ttl_seconds: float) -> tuple[float, float]:
    """Validate a positive TTL and return observation and expiry timestamps."""
    if ttl_seconds <= 0 or not math.isfinite(ttl_seconds):
        raise SnapshotCacheValueError("ttl_seconds must be positive")
    stamp = _timestamp(fetched_at)
    return stamp, stamp + float(ttl_seconds)


def _validate_counts(
    quality: SnapshotQuality, reported_count: int | None, returned_count: int
) -> None:
    """Make a complete graph claim carry arithmetic that supports it."""
    if returned_count < 0 or (reported_count is not None and reported_count < 0):
        raise SnapshotCacheValueError("edge counts must not be negative")
    if quality is SnapshotQuality.COMPLETE and (
        reported_count is None or returned_count != reported_count
    ):
        raise SnapshotCacheValueError(
            "a complete graph snapshot requires equal reported_count and returned_count"
        )
    if quality is SnapshotQuality.TRUNCATED and (
        reported_count is None or returned_count >= reported_count
    ):
        raise SnapshotCacheValueError(
            "a truncated graph snapshot requires returned_count < reported_count"
        )


def _canonical_bytes(payload: Any) -> bytes:
    """Serialize JSON deterministically before hashing and compression."""
    try:
        text = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise SnapshotCacheValueError(f"snapshot payload is not canonical JSON: {error}") from error
    return text.encode("utf-8")


def _encode(payload: Any) -> _EncodedPayload:
    """Hash canonical JSON and compress it for bounded local storage."""
    raw = _canonical_bytes(payload)
    blob = zlib.compress(raw)
    return _EncodedPayload(
        blob=blob,
        digest=hashlib.sha256(raw).hexdigest(),
        logical_bytes=len(blob) + _ENTRY_OVERHEAD_BYTES,
    )


def _decode(blob: bytes) -> Any:
    """Decode a stored payload, refusing corrupt compressed or JSON data."""
    try:
        return json.loads(zlib.decompress(blob).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SnapshotCacheSchemaError(f"cached payload is corrupt: {error}") from error


def _allocated_and_apparent(path: Path) -> tuple[int, int]:
    """Return allocated blocks and apparent length for one SQLite file."""
    if not path.exists():
        return 0, 0
    stat = path.stat()
    blocks = getattr(stat, "st_blocks", None)
    allocated = int(blocks) * 512 if blocks is not None else int(stat.st_size)
    return allocated, int(stat.st_size)


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cache_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cache_entries (
    entry_id INTEGER PRIMARY KEY,
    tier TEXT NOT NULL CHECK (tier IN ('negative', 'derived', 'metadata', 'graph')),
    quality TEXT CHECK (quality IN ('complete', 'partial', 'truncated') OR quality IS NULL),
    reported_count INTEGER CHECK (reported_count IS NULL OR reported_count >= 0),
    returned_count INTEGER CHECK (returned_count IS NULL OR returned_count >= 0),
    fetched_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    last_accessed REAL NOT NULL,
    payload BLOB NOT NULL,
    payload_digest TEXT NOT NULL,
    logical_bytes INTEGER NOT NULL CHECK (logical_bytes > 0)
);

CREATE TABLE IF NOT EXISTS aliases (
    entry_id INTEGER NOT NULL REFERENCES cache_entries(entry_id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    alias_key TEXT NOT NULL,
    native_key TEXT NOT NULL,
    first_seen_at REAL NOT NULL,
    last_seen_at REAL NOT NULL,
    PRIMARY KEY (entry_id, alias_key)
);

CREATE TABLE IF NOT EXISTS graph_snapshots (
    entry_id INTEGER PRIMARY KEY REFERENCES cache_entries(entry_id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    native_key TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('backward', 'forward')),
    request_shape TEXT NOT NULL,
    UNIQUE (source, native_key, direction, request_shape)
);

CREATE TABLE IF NOT EXISTS metadata_snapshots (
    entry_id INTEGER PRIMARY KEY REFERENCES cache_entries(entry_id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    native_key TEXT NOT NULL,
    field_set TEXT NOT NULL,
    UNIQUE (source, native_key, field_set)
);

CREATE TABLE IF NOT EXISTS negative_outcomes (
    entry_id INTEGER PRIMARY KEY REFERENCES cache_entries(entry_id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    requested_key TEXT NOT NULL,
    UNIQUE (source, requested_key)
);

CREATE INDEX IF NOT EXISTS cache_entries_eviction_idx
ON cache_entries(tier, expires_at, last_accessed, entry_id);

CREATE INDEX IF NOT EXISTS aliases_native_idx
ON aliases(source, native_key);

CREATE INDEX IF NOT EXISTS aliases_lookup_idx
ON aliases(source, alias_key);
"""

_EXPECTED_TABLES = frozenset(
    {
        "aliases",
        "cache_entries",
        "cache_meta",
        "graph_snapshots",
        "metadata_snapshots",
        "negative_outcomes",
    }
)


__all__ = [
    "DEFAULT_BUDGET_BYTES",
    "MAX_BUDGET_BYTES",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "AliasResolution",
    "CacheTier",
    "CacheUsage",
    "CacheWriteResult",
    "NegativeRead",
    "PhysicalUsage",
    "PruneResult",
    "SnapshotCache",
    "SnapshotCacheError",
    "SnapshotCacheSchemaError",
    "SnapshotCacheValueError",
    "SnapshotQuality",
    "SnapshotRead",
]
