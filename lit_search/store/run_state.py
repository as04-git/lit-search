"""Crash-safe operational state for resumable traversal runs.

Immutable handles are the boundary for completed paper sets.  A long-running
traversal needs a different kind of persistence: mutable coordination state
that is committed before and after each provider request.  This module owns
that deliberately narrow substrate.  It does not execute requests or publish
handles.

Each run lives in ``<store-root>/runs/<run-id>/run.sqlite``.  SQLite WAL and
short transactions provide crash consistency.  The append-only ``events``
table is the authoritative operational trace; the other tables are
transactionally maintained projections used for efficient resume and status
queries.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sqlite3
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from lit_search.exceptions import PersistenceError

RUN_SCHEMA_NAME = "lit-search/run"
RUN_SCHEMA_VERSION = 1
RUNS_SUBDIRECTORY = "runs"
RUN_DATABASE_NAME = "run.sqlite"

_RUN_ID = re.compile(r"r_[0-9a-f]{32}\Z")
_DIGEST_SIZE = 16


class RunStateError(PersistenceError, RuntimeError):
    """Base class for run-state failures."""


class RunSchemaError(RunStateError):
    """The database schema is absent, damaged, or unsupported."""


class SemanticSpecMismatchError(RunStateError):
    """Resume was requested with a different semantic configuration."""


class InvalidRunTransitionError(RunStateError):
    """A lifecycle, phase, target, or work-unit transition is invalid."""


class RunLeaseError(RunStateError):
    """Another live owner holds the run lease."""


class RunLifecycle(StrEnum):
    """Operational lifecycle, independent of traversal phase."""

    ACTIVE = "active"
    WAITING = "waiting"
    PAUSED = "paused"
    FAILED = "failed"
    COMPLETED = "completed"


class RunPhase(StrEnum):
    """Durable traversal phase."""

    SEED_PLAN = "seed_plan"
    SEED_FETCH = "seed_fetch"
    LEVEL_PLAN = "level_plan"
    LEVEL_FETCH = "level_fetch"
    LEVEL_REDUCE = "level_reduce"
    ENRICH_PLAN = "enrich_plan"
    ENRICH_FETCH = "enrich_fetch"
    FINALIZE = "finalize"


class TargetState(StrEnum):
    """Independent target states; no overloaded ``visited`` bit."""

    DISCOVERED = "discovered"
    SCHEDULED = "scheduled"
    RESOLVED = "resolved"
    NOT_FOUND = "not_found"
    FAILED = "failed"
    UNROUTABLE = "unroutable"
    EXPANDED = "expanded"
    RESIDUAL = "residual"


class WorkUnitState(StrEnum):
    """Request-sized work-unit state."""

    PLANNED = "planned"
    IN_FLIGHT = "in_flight"
    SUCCEEDED = "succeeded"
    NOT_FOUND = "not_found"
    FAILED = "failed"
    UNROUTABLE = "unroutable"


_TERMINAL_WORK_STATES = frozenset(
    {
        WorkUnitState.SUCCEEDED,
        WorkUnitState.NOT_FOUND,
        WorkUnitState.FAILED,
        WorkUnitState.UNROUTABLE,
    }
)

_LIFECYCLE_TRANSITIONS: dict[RunLifecycle, frozenset[RunLifecycle]] = {
    RunLifecycle.ACTIVE: frozenset(
        {RunLifecycle.WAITING, RunLifecycle.PAUSED, RunLifecycle.FAILED, RunLifecycle.COMPLETED}
    ),
    RunLifecycle.WAITING: frozenset(
        {RunLifecycle.ACTIVE, RunLifecycle.PAUSED, RunLifecycle.FAILED, RunLifecycle.COMPLETED}
    ),
    RunLifecycle.PAUSED: frozenset(
        {RunLifecycle.ACTIVE, RunLifecycle.FAILED, RunLifecycle.COMPLETED}
    ),
    RunLifecycle.FAILED: frozenset(),
    RunLifecycle.COMPLETED: frozenset(),
}

_PHASE_TRANSITIONS: dict[RunPhase, frozenset[RunPhase]] = {
    RunPhase.SEED_PLAN: frozenset({RunPhase.SEED_FETCH, RunPhase.FINALIZE}),
    RunPhase.SEED_FETCH: frozenset({RunPhase.LEVEL_PLAN, RunPhase.ENRICH_PLAN, RunPhase.FINALIZE}),
    RunPhase.LEVEL_PLAN: frozenset(
        {
            RunPhase.LEVEL_FETCH,
            RunPhase.LEVEL_REDUCE,
            RunPhase.ENRICH_PLAN,
            RunPhase.FINALIZE,
        }
    ),
    RunPhase.LEVEL_FETCH: frozenset({RunPhase.LEVEL_REDUCE, RunPhase.FINALIZE}),
    RunPhase.LEVEL_REDUCE: frozenset(
        {RunPhase.LEVEL_PLAN, RunPhase.ENRICH_PLAN, RunPhase.FINALIZE}
    ),
    RunPhase.ENRICH_PLAN: frozenset({RunPhase.ENRICH_FETCH, RunPhase.FINALIZE}),
    RunPhase.ENRICH_FETCH: frozenset({RunPhase.FINALIZE}),
    RunPhase.FINALIZE: frozenset(),
}

_TARGET_TRANSITIONS: dict[TargetState, frozenset[TargetState]] = {
    TargetState.DISCOVERED: frozenset(
        {TargetState.SCHEDULED, TargetState.UNROUTABLE, TargetState.RESIDUAL}
    ),
    TargetState.SCHEDULED: frozenset(
        {
            TargetState.RESOLVED,
            TargetState.NOT_FOUND,
            TargetState.FAILED,
            TargetState.UNROUTABLE,
            TargetState.RESIDUAL,
        }
    ),
    TargetState.RESOLVED: frozenset({TargetState.EXPANDED, TargetState.RESIDUAL}),
    TargetState.NOT_FOUND: frozenset({TargetState.SCHEDULED, TargetState.RESIDUAL}),
    TargetState.FAILED: frozenset({TargetState.SCHEDULED, TargetState.RESIDUAL}),
    TargetState.UNROUTABLE: frozenset({TargetState.SCHEDULED, TargetState.RESIDUAL}),
    TargetState.EXPANDED: frozenset(),
    TargetState.RESIDUAL: frozenset({TargetState.SCHEDULED}),
}


@dataclass(frozen=True, slots=True)
class CredentialIdentity:
    """Non-secret identity of one credential scope."""

    source: str
    present: bool
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        source = self.source.strip().lower()
        if not source:
            raise ValueError("credential source must not be blank")
        if self.present and not self.fingerprint:
            raise ValueError("a present credential requires a fingerprint")
        if not self.present and self.fingerprint is not None:
            raise ValueError("an absent credential must not carry a fingerprint")
        object.__setattr__(self, "source", source)

    @classmethod
    def from_material(cls, source: str, material: str | None) -> CredentialIdentity:
        """Build an identity without retaining credential material."""
        if material is None or not material.strip():
            return cls(source=source, present=False)
        fingerprint = hashlib.sha256(material.strip().encode("utf-8")).hexdigest()[:16]
        return cls(source=source, present=True, fingerprint=fingerprint)

    def to_dict(self) -> dict[str, Any]:
        """Return the only credential representation allowed on disk."""
        return {
            "source": self.source,
            "present": self.present,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    """Materialized state used by status and resume callers."""

    run_id: str
    lifecycle: RunLifecycle
    phase: RunPhase
    level: int | None
    generation: int
    semantic_digest: str
    lease_owner: str | None
    lease_expires_at: dt.datetime | None


@dataclass(frozen=True, slots=True)
class RunEvent:
    """One append-only operational event."""

    sequence: int
    event_type: str
    payload: dict[str, Any]
    recorded_at: dt.datetime


@dataclass(frozen=True, slots=True)
class WorkUnit:
    """One provider-request-sized unit."""

    unit_id: str
    phase: RunPhase
    level: int | None
    source: str
    request_kind: str
    ordinal: int
    plan_sequence: int
    targets: tuple[str, ...]
    semantic_context: dict[str, Any]
    state: WorkUnitState
    attempts: int
    outcome: dict[str, Any] | None
    coverage: dict[str, Any] | None


def canonical_json(value: Any) -> str:
    """Serialize a semantic value deterministically."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def semantic_run_digest(
    semantic_spec: Mapping[str, Any],
    credentials: Iterable[CredentialIdentity] = (),
) -> str:
    """Hash the exact immutable run specification and credential identities.

    Operational timestamps and retry history are not arguments to this function
    and therefore do not enter the run-spec digest. Callers must keep them out
    of ``semantic_spec``; unlike :meth:`RunStateStore.semantic_outcome_digest`,
    this function deliberately does not guess which supplied configuration
    fields the caller intended to be semantic.
    """
    credential_payload = [
        item.to_dict() for item in sorted(credentials, key=lambda item: item.source)
    ]
    payload = {"semantic_spec": dict(semantic_spec), "credentials": credential_payload}
    digest = hashlib.blake2b(canonical_json(payload).encode("utf-8"), digest_size=_DIGEST_SIZE)
    return digest.hexdigest()


def deterministic_work_unit_id(
    *,
    phase: RunPhase,
    level: int | None,
    source: str,
    request_kind: str,
    targets: Sequence[str],
    semantic_context: Mapping[str, Any] | None = None,
) -> str:
    """Return a stable ID independent of target input ordering."""
    canonical_targets = sorted(set(targets))
    payload = {
        "phase": phase.value,
        "level": level,
        "source": source.strip().lower(),
        "request_kind": request_kind.strip().lower(),
        "targets": canonical_targets,
        "semantic_context": dict(semantic_context or {}),
    }
    digest = hashlib.blake2b(canonical_json(payload).encode("utf-8"), digest_size=_DIGEST_SIZE)
    return f"wu_{digest.hexdigest()}"


class RunStateStore:
    """Transactional state for one resumable run."""

    def __init__(self, path: Path, connection: sqlite3.Connection) -> None:
        self.path = path
        self._connection = connection
        self._connection.row_factory = sqlite3.Row

    @classmethod
    def create(
        cls,
        store_root: Path | str,
        semantic_spec: Mapping[str, Any],
        *,
        credentials: Iterable[CredentialIdentity] = (),
        run_id: str | None = None,
        created_at: dt.datetime | None = None,
    ) -> RunStateStore:
        """Create a run with an immutable semantic specification."""
        resolved_run_id = run_id or f"r_{uuid.uuid4().hex}"
        _validate_run_id(resolved_run_id)
        path = (
            Path(store_root).expanduser() / RUNS_SUBDIRECTORY / resolved_run_id / RUN_DATABASE_NAME
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise RunStateError(f"run {resolved_run_id} already exists at {path}")

        credential_items = tuple(credentials)
        spec_json = canonical_json(dict(semantic_spec))
        credentials_json = canonical_json(
            [item.to_dict() for item in sorted(credential_items, key=lambda item: item.source)]
        )
        digest = semantic_run_digest(semantic_spec, credential_items)
        timestamp = _as_utc(created_at or dt.datetime.now(dt.UTC))
        connection = _connect(path)
        store = cls(path, connection)
        try:
            store._create_schema()
            with connection:
                metadata = {
                    "schema_name": RUN_SCHEMA_NAME,
                    "schema_version": str(RUN_SCHEMA_VERSION),
                    "run_id": resolved_run_id,
                    "semantic_spec": spec_json,
                    "credentials": credentials_json,
                    "semantic_digest": digest,
                    "created_at": timestamp.isoformat(),
                }
                connection.executemany(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items()
                )
                connection.execute(
                    """
                    INSERT INTO run_state(
                        singleton, lifecycle, phase, level, generation, lease_owner,
                        lease_expires_at
                    ) VALUES (1, ?, ?, NULL, 0, NULL, NULL)
                    """,
                    (RunLifecycle.ACTIVE.value, RunPhase.SEED_PLAN.value),
                )
                store._append_event(
                    "run_created",
                    {
                        "run_id": resolved_run_id,
                        "semantic_digest": digest,
                        "lifecycle": RunLifecycle.ACTIVE.value,
                        "phase": RunPhase.SEED_PLAN.value,
                    },
                    recorded_at=timestamp,
                )
        except BaseException:
            connection.close()
            path.unlink(missing_ok=True)
            raise
        return store

    @classmethod
    def resume(
        cls,
        store_root: Path | str,
        run_id: str,
        semantic_spec: Mapping[str, Any],
        *,
        credentials: Iterable[CredentialIdentity] = (),
    ) -> RunStateStore:
        """Open an existing run only when semantic configuration is identical."""
        _validate_run_id(run_id)
        path = Path(store_root).expanduser() / RUNS_SUBDIRECTORY / run_id / RUN_DATABASE_NAME
        if not path.is_file():
            raise RunStateError(f"run {run_id} does not exist at {path}")
        connection = _connect(path)
        store = cls(path, connection)
        try:
            store._check_schema(run_id)
            expected = semantic_run_digest(semantic_spec, credentials)
            actual = store._metadata("semantic_digest")
            if expected != actual:
                raise SemanticSpecMismatchError(
                    f"run {run_id} was created with semantic digest {actual}, got {expected}; "
                    "semantic changes require a forked run"
                )
        except BaseException:
            connection.close()
            raise
        return store

    @property
    def run_id(self) -> str:
        """Return this run's stable identifier."""
        return self._metadata("run_id")

    @property
    def semantic_spec(self) -> dict[str, Any]:
        """Return a detached copy of the immutable semantic specification."""
        value = json.loads(self._metadata("semantic_spec"))
        if not isinstance(value, dict):  # pragma: no cover - schema corruption guard
            raise RunSchemaError("metadata.semantic_spec is not an object")
        return value

    @property
    def credential_identities(self) -> tuple[CredentialIdentity, ...]:
        """Return stored non-secret credential identities."""
        raw = json.loads(self._metadata("credentials"))
        if not isinstance(raw, list):  # pragma: no cover - schema corruption guard
            raise RunSchemaError("metadata.credentials is not an array")
        return tuple(
            CredentialIdentity(
                source=str(item["source"]),
                present=bool(item["present"]),
                fingerprint=item.get("fingerprint"),
            )
            for item in raw
        )

    def validate_semantic_spec(
        self,
        semantic_spec: Mapping[str, Any],
        *,
        credentials: Iterable[CredentialIdentity] = (),
    ) -> None:
        """Refuse runner attachment when semantic configuration differs."""
        expected = semantic_run_digest(semantic_spec, credentials)
        actual = self._metadata("semantic_digest")
        if expected != actual:
            raise SemanticSpecMismatchError(
                f"run {self.run_id} was created with semantic digest {actual}, got {expected}; "
                "semantic changes require a forked run"
            )

    def snapshot(self) -> RunSnapshot:
        """Return the current materialized lifecycle and phase."""
        row = self._connection.execute("SELECT * FROM run_state WHERE singleton = 1").fetchone()
        if row is None:  # pragma: no cover - schema corruption guard
            raise RunSchemaError("run_state singleton is missing")
        lease_raw = row["lease_expires_at"]
        return RunSnapshot(
            run_id=self.run_id,
            lifecycle=RunLifecycle(row["lifecycle"]),
            phase=RunPhase(row["phase"]),
            level=row["level"],
            generation=int(row["generation"]),
            semantic_digest=self._metadata("semantic_digest"),
            lease_owner=row["lease_owner"],
            lease_expires_at=dt.datetime.fromisoformat(lease_raw) if lease_raw else None,
        )

    def transition_lifecycle(
        self,
        lifecycle: RunLifecycle,
        *,
        reason: str,
        recorded_at: dt.datetime | None = None,
    ) -> None:
        """Apply one valid lifecycle transition and append its event."""
        if not reason.strip():
            raise ValueError("lifecycle transition reason must not be blank")
        with self._immediate_transaction():
            current = self.snapshot()
            if lifecycle not in _LIFECYCLE_TRANSITIONS[current.lifecycle]:
                raise InvalidRunTransitionError(
                    f"lifecycle cannot move from {current.lifecycle.value} to {lifecycle.value}"
                )
            if lifecycle is RunLifecycle.COMPLETED and current.phase is not RunPhase.FINALIZE:
                raise InvalidRunTransitionError("a run can complete only from the finalize phase")
            self._connection.execute(
                """
                UPDATE run_state
                SET lifecycle = ?, generation = generation + 1
                WHERE singleton = 1
                """,
                (lifecycle.value,),
            )
            self._append_event(
                "lifecycle_changed",
                {"from": current.lifecycle.value, "to": lifecycle.value, "reason": reason},
                recorded_at=recorded_at,
            )

    def transition_phase(
        self,
        phase: RunPhase,
        *,
        level: int | None = None,
        recorded_at: dt.datetime | None = None,
    ) -> None:
        """Apply one valid traversal-phase transition."""
        _validate_phase_level(phase, level)
        with self._immediate_transaction():
            current = self.snapshot()
            if current.lifecycle is not RunLifecycle.ACTIVE:
                raise InvalidRunTransitionError(
                    f"phase cannot change while lifecycle is {current.lifecycle.value}"
                )
            if phase not in _PHASE_TRANSITIONS[current.phase]:
                raise InvalidRunTransitionError(
                    f"phase cannot move from {current.phase.value} to {phase.value}"
                )
            _validate_level_progression(current.phase, current.level, phase, level)
            self._connection.execute(
                """
                UPDATE run_state
                SET phase = ?, level = ?, generation = generation + 1
                WHERE singleton = 1
                """,
                (phase.value, level),
            )
            self._append_event(
                "phase_changed",
                {
                    "from": current.phase.value,
                    "from_level": current.level,
                    "to": phase.value,
                    "to_level": level,
                },
                recorded_at=recorded_at,
            )

    def acquire_lease(
        self,
        owner: str,
        *,
        ttl_seconds: float,
        now: dt.datetime | None = None,
    ) -> None:
        """Acquire or renew the crash-expiring active lease."""
        if not owner.strip():
            raise ValueError("lease owner must not be blank")
        if ttl_seconds <= 0:
            raise ValueError("lease ttl_seconds must be positive")
        timestamp = _as_utc(now or dt.datetime.now(dt.UTC))
        expires = timestamp + dt.timedelta(seconds=ttl_seconds)
        with self._immediate_transaction():
            current = self.snapshot()
            if current.lifecycle in {RunLifecycle.FAILED, RunLifecycle.COMPLETED}:
                raise RunLeaseError(f"cannot lease a {current.lifecycle.value} run")
            if (
                current.lease_owner is not None
                and current.lease_owner != owner
                and current.lease_expires_at is not None
                and current.lease_expires_at > timestamp
            ):
                raise RunLeaseError(
                    f"run {self.run_id} is leased by {current.lease_owner} until "
                    f"{current.lease_expires_at.isoformat()}"
                )
            self._connection.execute(
                """
                UPDATE run_state
                SET lease_owner = ?, lease_expires_at = ?, generation = generation + 1
                WHERE singleton = 1
                """,
                (owner, expires.isoformat()),
            )
            self._append_event(
                "lease_acquired",
                {"owner": owner, "expires_at": expires.isoformat()},
                recorded_at=timestamp,
            )

    def release_lease(self, owner: str, *, now: dt.datetime | None = None) -> None:
        """Release a lease only by its current owner."""
        with self._immediate_transaction():
            current = self.snapshot()
            if current.lease_owner != owner:
                raise RunLeaseError(
                    f"lease is held by {current.lease_owner!r}, not requested owner {owner!r}"
                )
            self._connection.execute(
                """
                UPDATE run_state
                SET lease_owner = NULL, lease_expires_at = NULL, generation = generation + 1
                WHERE singleton = 1
                """
            )
            self._append_event("lease_released", {"owner": owner}, recorded_at=now)

    def discover_target(
        self, target_key: str, *, evidence: Mapping[str, Any] | None = None
    ) -> bool:
        """Retain one distinct discovery assertion for a normalized target.

        Target identity and discovery evidence are deliberately separate. A
        target may be asserted by several parents, sources, directions, or
        targeting lanes. An exact canonical repeat is a no-op; a distinct
        assertion is appended without resetting the target's state.
        """
        key = target_key.strip().lower()
        if not key:
            raise ValueError("target key must not be blank")
        evidence_json = canonical_json(dict(evidence or {}))
        evidence_digest = hashlib.blake2b(
            evidence_json.encode("utf-8"), digest_size=_DIGEST_SIZE
        ).hexdigest()
        with self._immediate_transaction():
            self._require_active()
            existing_evidence = self._connection.execute(
                """
                SELECT evidence_json FROM target_evidence
                WHERE target_key = ? AND evidence_digest = ?
                """,
                (key, evidence_digest),
            ).fetchone()
            if existing_evidence is not None:
                if existing_evidence["evidence_json"] != evidence_json:
                    raise RunStateError(f"discovery-evidence digest collision for target {key}")
                return False
            target_exists = (
                self._connection.execute(
                    "SELECT 1 FROM targets WHERE target_key = ?", (key,)
                ).fetchone()
                is not None
            )
            event_type = "target_evidence_added" if target_exists else "target_discovered"
            cursor = self._append_event(
                event_type,
                {
                    "target_key": key,
                    "evidence_digest": evidence_digest,
                    "evidence": dict(evidence or {}),
                },
            )
            if not target_exists:
                self._connection.execute(
                    """
                    INSERT INTO targets(target_key, state, first_event)
                    VALUES (?, ?, ?)
                    """,
                    (key, TargetState.DISCOVERED.value, cursor),
                )
            self._connection.execute(
                """
                INSERT INTO target_evidence(
                    target_key, evidence_digest, evidence_json, first_event
                ) VALUES (?, ?, ?, ?)
                """,
                (key, evidence_digest, evidence_json, cursor),
            )
        return True

    def transition_target(
        self,
        target_key: str,
        state: TargetState,
        *,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        """Apply an explicit target-state transition."""
        key = target_key.strip().lower()
        with self._immediate_transaction():
            self._require_active()
            row = self._connection.execute(
                "SELECT state FROM targets WHERE target_key = ?", (key,)
            ).fetchone()
            if row is None:
                raise RunStateError(f"unknown target {key}")
            current = TargetState(row["state"])
            if state not in _TARGET_TRANSITIONS[current]:
                raise InvalidRunTransitionError(
                    f"target {key} cannot move from {current.value} to {state.value}"
                )
            self._connection.execute(
                "UPDATE targets SET state = ? WHERE target_key = ?", (state.value, key)
            )
            self._append_event(
                "target_state_changed",
                {
                    "target_key": key,
                    "from": current.value,
                    "to": state.value,
                    "detail": dict(detail or {}),
                },
            )

    def target_states(self) -> tuple[tuple[str, TargetState], ...]:
        """Return targets in canonical key order."""
        rows = self._connection.execute(
            "SELECT target_key, state FROM targets ORDER BY target_key"
        ).fetchall()
        return tuple((row["target_key"], TargetState(row["state"])) for row in rows)

    def target_evidence(self, target_key: str) -> tuple[dict[str, Any], ...]:
        """Return all distinct discovery assertions in canonical order."""
        key = target_key.strip().lower()
        if not key:
            raise ValueError("target key must not be blank")
        if (
            self._connection.execute(
                "SELECT 1 FROM targets WHERE target_key = ?", (key,)
            ).fetchone()
            is None
        ):
            raise RunStateError(f"unknown target {key}")
        rows = self._connection.execute(
            """
            SELECT evidence_json FROM target_evidence
            WHERE target_key = ? ORDER BY evidence_json
            """,
            (key,),
        ).fetchall()
        return tuple(json.loads(row["evidence_json"]) for row in rows)

    def plan_work_unit(
        self,
        *,
        phase: RunPhase,
        level: int | None,
        source: str,
        request_kind: str,
        ordinal: int,
        targets: Sequence[str],
        semantic_context: Mapping[str, Any] | None = None,
    ) -> str:
        """Commit a deterministic provider request plan before any I/O."""
        _validate_phase_level(phase, level)
        if ordinal < 0:
            raise ValueError("work-unit ordinal must not be negative")
        source_value = source.strip().lower()
        kind_value = request_kind.strip().lower()
        if not source_value or not kind_value:
            raise ValueError("work-unit source and request_kind must not be blank")
        canonical_targets = tuple(sorted({target.strip().lower() for target in targets}))
        if not canonical_targets or any(not target for target in canonical_targets):
            raise ValueError("work unit requires non-blank targets")
        context = dict(semantic_context or {})
        unit_id = deterministic_work_unit_id(
            phase=phase,
            level=level,
            source=source_value,
            request_kind=kind_value,
            targets=canonical_targets,
            semantic_context=context,
        )
        targets_json = canonical_json(canonical_targets)
        context_json = canonical_json(context)
        with self._immediate_transaction():
            self._require_active()
            row = self._connection.execute(
                "SELECT * FROM work_units WHERE unit_id = ?", (unit_id,)
            ).fetchone()
            if row is not None:
                expected = (
                    phase.value,
                    level,
                    source_value,
                    kind_value,
                    ordinal,
                    targets_json,
                    context_json,
                )
                actual = tuple(
                    row[key]
                    for key in (
                        "phase",
                        "level",
                        "source",
                        "request_kind",
                        "ordinal",
                        "targets_json",
                        "semantic_context_json",
                    )
                )
                if actual != expected:
                    raise RunStateError(f"work-unit ID collision or changed plan for {unit_id}")
                return unit_id
            plan_sequence = int(
                self._connection.execute(
                    "SELECT COALESCE(MAX(plan_sequence), -1) + 1 FROM work_units"
                ).fetchone()[0]
            )
            self._connection.execute(
                """
                INSERT INTO work_units(
                    unit_id, phase, level, source, request_kind, ordinal, plan_sequence,
                    targets_json, semantic_context_json, state, attempts, outcome_json,
                    coverage_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL)
                """,
                (
                    unit_id,
                    phase.value,
                    level,
                    source_value,
                    kind_value,
                    ordinal,
                    plan_sequence,
                    targets_json,
                    context_json,
                    WorkUnitState.PLANNED.value,
                ),
            )
            self._append_event(
                "work_unit_planned",
                {
                    "unit_id": unit_id,
                    "phase": phase.value,
                    "level": level,
                    "source": source_value,
                    "request_kind": kind_value,
                    "ordinal": ordinal,
                    "plan_sequence": plan_sequence,
                    "targets": list(canonical_targets),
                    "semantic_context": context,
                },
            )
        return unit_id

    def begin_work_unit(self, unit_id: str) -> int:
        """Commit an attempt marker immediately before provider I/O."""
        with self._immediate_transaction():
            self._require_active()
            row = self._work_row(unit_id)
            state = WorkUnitState(row["state"])
            if state is not WorkUnitState.PLANNED:
                raise InvalidRunTransitionError(
                    f"work unit {unit_id} cannot begin from {state.value}"
                )
            attempt = int(row["attempts"]) + 1
            self._connection.execute(
                "UPDATE work_units SET state = ?, attempts = ? WHERE unit_id = ?",
                (WorkUnitState.IN_FLIGHT.value, attempt, unit_id),
            )
            self._append_event("work_unit_started", {"unit_id": unit_id, "attempt": attempt})
        return attempt

    def complete_work_unit(
        self,
        unit_id: str,
        state: WorkUnitState,
        *,
        outcome: Mapping[str, Any],
        coverage: Mapping[str, Any],
    ) -> bool:
        """Commit one terminal provider outcome and coverage atomically."""
        if state not in _TERMINAL_WORK_STATES:
            raise ValueError(f"{state.value} is not a terminal work-unit state")
        outcome_json = canonical_json(dict(outcome))
        coverage_json = canonical_json(dict(coverage))
        with self._immediate_transaction():
            self._require_active()
            row = self._work_row(unit_id)
            current = WorkUnitState(row["state"])
            if current in _TERMINAL_WORK_STATES:
                if (
                    current is state
                    and row["outcome_json"] == outcome_json
                    and row["coverage_json"] == coverage_json
                ):
                    return False
                raise InvalidRunTransitionError(
                    f"work unit {unit_id} already has terminal outcome {current.value}"
                )
            if current is not WorkUnitState.IN_FLIGHT:
                raise InvalidRunTransitionError(
                    f"work unit {unit_id} cannot finish from {current.value}"
                )
            self._connection.execute(
                """
                UPDATE work_units
                SET state = ?, outcome_json = ?, coverage_json = ?
                WHERE unit_id = ?
                """,
                (state.value, outcome_json, coverage_json, unit_id),
            )
            self._append_event(
                "work_unit_finished",
                {
                    "unit_id": unit_id,
                    "attempt": int(row["attempts"]),
                    "state": state.value,
                    "outcome": dict(outcome),
                    "coverage": dict(coverage),
                },
            )
        return True

    def recover_in_flight(self, *, reason: str = "process ended before outcome commit") -> int:
        """Make indeterminate provider attempts retryable without erasing history."""
        if not reason.strip():
            raise ValueError("recovery reason must not be blank")
        with self._immediate_transaction():
            self._require_active()
            rows = self._connection.execute(
                """
                SELECT unit_id, attempts FROM work_units
                WHERE state = ? ORDER BY plan_sequence
                """,
                (WorkUnitState.IN_FLIGHT.value,),
            ).fetchall()
            for row in rows:
                self._connection.execute(
                    "UPDATE work_units SET state = ? WHERE unit_id = ?",
                    (WorkUnitState.PLANNED.value, row["unit_id"]),
                )
                self._append_event(
                    "work_unit_attempt_indeterminate",
                    {
                        "unit_id": row["unit_id"],
                        "attempt": int(row["attempts"]),
                        "reason": reason,
                    },
                )
        return len(rows)

    def commit_reduction(
        self,
        unit_id: str,
        reduction_key: str,
        payload: Mapping[str, Any],
    ) -> bool:
        """Commit a local reduction exactly once."""
        key = reduction_key.strip()
        if not key:
            raise ValueError("reduction key must not be blank")
        payload_json = canonical_json(dict(payload))
        with self._immediate_transaction():
            self._require_active()
            row = self._work_row(unit_id)
            state = WorkUnitState(row["state"])
            if state not in _TERMINAL_WORK_STATES:
                raise InvalidRunTransitionError(
                    f"work unit {unit_id} cannot reduce from {state.value}"
                )
            existing = self._connection.execute(
                """
                SELECT payload_json FROM reductions
                WHERE unit_id = ? AND reduction_key = ?
                """,
                (unit_id, key),
            ).fetchone()
            if existing is not None:
                if existing["payload_json"] == payload_json:
                    return False
                raise InvalidRunTransitionError(
                    f"reduction {key!r} for {unit_id} was already committed differently"
                )
            self._connection.execute(
                """
                INSERT INTO reductions(unit_id, reduction_key, payload_json)
                VALUES (?, ?, ?)
                """,
                (unit_id, key, payload_json),
            )
            self._append_event(
                "reduction_committed",
                {"unit_id": unit_id, "reduction_key": key, "payload": dict(payload)},
            )
        return True

    def work_units(self) -> tuple[WorkUnit, ...]:
        """Return units in deterministic planned order."""
        rows = self._connection.execute(
            "SELECT * FROM work_units ORDER BY plan_sequence"
        ).fetchall()
        return tuple(_work_unit_from_row(row) for row in rows)

    def work_unit(self, unit_id: str) -> WorkUnit:
        """Return one materialized work unit by deterministic ID."""
        return _work_unit_from_row(self._work_row(unit_id))

    def semantic_outcome_digest(self) -> str:
        """Hash semantic state while ignoring time, leases, and retry history.

        Two runs with the same planned units, target outcomes, terminal provider
        evidence, coverage, and reductions produce the same digest even when
        one crashed and retried. Event timestamps, attempt counts, leases,
        generation numbers, and runtime/progress fields are excluded.
        """
        target_rows = self._connection.execute(
            "SELECT target_key, state FROM targets ORDER BY target_key"
        ).fetchall()
        evidence_rows = self._connection.execute(
            """
            SELECT target_key, evidence_json FROM target_evidence
            ORDER BY target_key, evidence_json
            """
        ).fetchall()
        evidence_by_target: dict[str, list[Any]] = {}
        for row in evidence_rows:
            evidence_by_target.setdefault(row["target_key"], []).append(
                _semantic_projection(json.loads(row["evidence_json"]))
            )
        work_rows = self._connection.execute(
            "SELECT * FROM work_units ORDER BY plan_sequence"
        ).fetchall()
        reduction_rows = self._connection.execute(
            """
            SELECT unit_id, reduction_key, payload_json
            FROM reductions ORDER BY unit_id, reduction_key
            """
        ).fetchall()
        payload = {
            "semantic_spec_digest": self._metadata("semantic_digest"),
            "targets": [
                {
                    "target_key": row["target_key"],
                    "state": row["state"],
                    "evidence": evidence_by_target.get(row["target_key"], []),
                }
                for row in target_rows
            ],
            "work_units": [
                {
                    "unit_id": row["unit_id"],
                    "phase": row["phase"],
                    "level": row["level"],
                    "source": row["source"],
                    "request_kind": row["request_kind"],
                    "ordinal": row["ordinal"],
                    "targets": json.loads(row["targets_json"]),
                    "semantic_context": _semantic_projection(
                        json.loads(row["semantic_context_json"])
                    ),
                    "state": row["state"],
                    "outcome": _semantic_projection(json.loads(row["outcome_json"]))
                    if row["outcome_json"] is not None
                    else None,
                    "coverage": _semantic_projection(json.loads(row["coverage_json"]))
                    if row["coverage_json"] is not None
                    else None,
                }
                for row in work_rows
            ],
            "reductions": [
                {
                    "unit_id": row["unit_id"],
                    "reduction_key": row["reduction_key"],
                    "payload": _semantic_projection(json.loads(row["payload_json"])),
                }
                for row in reduction_rows
            ],
        }
        digest = hashlib.blake2b(canonical_json(payload).encode("utf-8"), digest_size=_DIGEST_SIZE)
        return digest.hexdigest()

    def events(self) -> tuple[RunEvent, ...]:
        """Return the authoritative append-only trace in sequence order."""
        rows = self._connection.execute(
            "SELECT sequence, event_type, payload_json, recorded_at FROM events ORDER BY sequence"
        ).fetchall()
        return tuple(
            RunEvent(
                sequence=int(row["sequence"]),
                event_type=row["event_type"],
                payload=json.loads(row["payload_json"]),
                recorded_at=dt.datetime.fromisoformat(row["recorded_at"]),
            )
            for row in rows
        )

    def close(self) -> None:
        """Checkpoint the WAL and close the database."""
        self._connection.execute("PRAGMA wal_checkpoint(PASSIVE)")
        self._connection.close()

    def __enter__(self) -> RunStateStore:
        """Return this open store."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Close the store on context exit."""
        self.close()

    @contextmanager
    def _immediate_transaction(self) -> Iterator[None]:
        """Secure SQLite's writer slot before any validation read.

        A deferred transaction begins on its first write, so a preceding read
        can validate stale state while another process commits. ``IMMEDIATE``
        obtains the reserved writer lock first; competing mutation waits and
        then validates against the committed winner.
        """
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    def _metadata(self, key: str) -> str:
        row = self._connection.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            raise RunSchemaError(f"required metadata key {key!r} is missing")
        return str(row["value"])

    def _work_row(self, unit_id: str) -> sqlite3.Row:
        row = self._connection.execute(
            "SELECT * FROM work_units WHERE unit_id = ?", (unit_id,)
        ).fetchone()
        if row is None:
            raise RunStateError(f"unknown work unit {unit_id}")
        return cast(sqlite3.Row, row)

    def _require_active(self) -> None:
        """Refuse operational mutation outside the active lifecycle."""
        lifecycle = self.snapshot().lifecycle
        if lifecycle is not RunLifecycle.ACTIVE:
            raise InvalidRunTransitionError(
                f"operational state cannot mutate while lifecycle is {lifecycle.value}"
            )

    def _append_event(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        recorded_at: dt.datetime | None = None,
    ) -> int:
        timestamp = _as_utc(recorded_at or dt.datetime.now(dt.UTC))
        cursor = self._connection.execute(
            "INSERT INTO events(event_type, payload_json, recorded_at) VALUES (?, ?, ?)",
            (event_type, canonical_json(dict(payload)), timestamp.isoformat()),
        )
        if cursor.lastrowid is None:  # pragma: no cover - SQLite always supplies it
            raise RunStateError("SQLite did not return an event sequence")
        return int(cursor.lastrowid)

    def _check_schema(self, run_id: str) -> None:
        try:
            user_version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        except sqlite3.DatabaseError as exc:
            raise RunSchemaError(f"cannot read run database schema: {exc}") from exc
        if user_version != RUN_SCHEMA_VERSION:
            raise RunSchemaError(
                f"run schema version {user_version} is unsupported; expected {RUN_SCHEMA_VERSION}"
            )
        if self._metadata("schema_name") != RUN_SCHEMA_NAME:
            raise RunSchemaError("database is not a lit-search run store")
        if int(self._metadata("schema_version")) != RUN_SCHEMA_VERSION:
            raise RunSchemaError("run metadata and SQLite schema versions disagree")
        if self._metadata("run_id") != run_id:
            raise RunSchemaError("run ID does not match its directory")

    def _create_schema(self) -> None:
        self._connection.executescript(
            f"""
            PRAGMA user_version = {RUN_SCHEMA_VERSION};

            CREATE TABLE metadata(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;

            CREATE TABLE run_state(
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                lifecycle TEXT NOT NULL,
                phase TEXT NOT NULL,
                level INTEGER,
                generation INTEGER NOT NULL,
                lease_owner TEXT,
                lease_expires_at TEXT
            );

            CREATE TABLE events(
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                recorded_at TEXT NOT NULL
            );

            CREATE TRIGGER events_no_update
            BEFORE UPDATE ON events
            BEGIN
                SELECT RAISE(ABORT, 'run events are append-only');
            END;

            CREATE TRIGGER events_no_delete
            BEFORE DELETE ON events
            BEGIN
                SELECT RAISE(ABORT, 'run events are append-only');
            END;

            CREATE TABLE targets(
                target_key TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                first_event INTEGER NOT NULL REFERENCES events(sequence)
            ) WITHOUT ROWID;

            CREATE TABLE target_evidence(
                target_key TEXT NOT NULL REFERENCES targets(target_key),
                evidence_digest TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                first_event INTEGER NOT NULL REFERENCES events(sequence),
                PRIMARY KEY(target_key, evidence_digest)
            ) WITHOUT ROWID;

            CREATE TABLE work_units(
                unit_id TEXT PRIMARY KEY,
                phase TEXT NOT NULL,
                level INTEGER,
                source TEXT NOT NULL,
                request_kind TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                plan_sequence INTEGER NOT NULL UNIQUE,
                targets_json TEXT NOT NULL,
                semantic_context_json TEXT NOT NULL,
                state TEXT NOT NULL,
                attempts INTEGER NOT NULL,
                outcome_json TEXT,
                coverage_json TEXT
            ) WITHOUT ROWID;

            CREATE UNIQUE INDEX work_units_plan_order
            ON work_units(phase, COALESCE(level, -1), ordinal, unit_id);

            CREATE TABLE reductions(
                unit_id TEXT NOT NULL REFERENCES work_units(unit_id),
                reduction_key TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY(unit_id, reduction_key)
            ) WITHOUT ROWID;
            """
        )


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30.0, isolation_level="DEFERRED")
    connection.execute("PRAGMA foreign_keys = ON")
    mode = str(connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]).lower()
    if mode != "wal":
        connection.close()
        raise RunSchemaError(f"SQLite refused WAL mode for {path}; got {mode!r}")
    connection.execute("PRAGMA synchronous = FULL")
    return connection


def _validate_run_id(run_id: str) -> None:
    if _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run ID must have the form r_<32 lowercase hexadecimal characters>")


def _validate_phase_level(phase: RunPhase, level: int | None) -> None:
    is_level_phase = phase in {
        RunPhase.LEVEL_PLAN,
        RunPhase.LEVEL_FETCH,
        RunPhase.LEVEL_REDUCE,
    }
    if is_level_phase and (level is None or level < 1):
        raise ValueError(f"phase {phase.value} requires a positive level")
    if not is_level_phase and level is not None:
        raise ValueError(f"phase {phase.value} must not carry a level")


def _validate_level_progression(
    old_phase: RunPhase,
    old_level: int | None,
    new_phase: RunPhase,
    new_level: int | None,
) -> None:
    if old_phase is RunPhase.SEED_FETCH and new_phase is RunPhase.LEVEL_PLAN and new_level != 1:
        raise InvalidRunTransitionError("the first traversal level must be level 1")
    if (
        old_phase in {RunPhase.LEVEL_PLAN, RunPhase.LEVEL_FETCH}
        and new_phase in {RunPhase.LEVEL_FETCH, RunPhase.LEVEL_REDUCE}
        and new_level != old_level
    ):
        raise InvalidRunTransitionError("planning, fetching, and reducing must stay at one level")
    if (
        old_phase is RunPhase.LEVEL_REDUCE
        and new_phase is RunPhase.LEVEL_PLAN
        and (old_level is None or new_level != old_level + 1)
    ):
        raise InvalidRunTransitionError("the next planned level must increment by one")


def _as_utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.UTC)
    return value.astimezone(dt.UTC)


_TRANSIENT_OUTCOME_KEYS = frozenset(
    {
        "attempt",
        "attempt_count",
        "attempt_history",
        "attempts",
        "created_at",
        "elapsed_seconds",
        "finished_at",
        "lease_expires_at",
        "lease_owner",
        "progress",
        "recorded_at",
        "retry_history",
        "runtime_seconds",
        "started_at",
        "transient_retry_history",
        "updated_at",
    }
)


def _semantic_projection(value: Any) -> Any:
    """Remove operational-only fields from persisted outcome evidence."""
    if isinstance(value, Mapping):
        return {
            str(key): _semantic_projection(item)
            for key, item in value.items()
            if str(key) not in _TRANSIENT_OUTCOME_KEYS
        }
    if isinstance(value, list | tuple):
        return [_semantic_projection(item) for item in value]
    return value


def _work_unit_from_row(row: sqlite3.Row) -> WorkUnit:
    outcome_raw = row["outcome_json"]
    coverage_raw = row["coverage_json"]
    return WorkUnit(
        unit_id=row["unit_id"],
        phase=RunPhase(row["phase"]),
        level=row["level"],
        source=row["source"],
        request_kind=row["request_kind"],
        ordinal=int(row["ordinal"]),
        plan_sequence=int(row["plan_sequence"]),
        targets=tuple(json.loads(row["targets_json"])),
        semantic_context=json.loads(row["semantic_context_json"]),
        state=WorkUnitState(row["state"]),
        attempts=int(row["attempts"]),
        outcome=json.loads(outcome_raw) if outcome_raw is not None else None,
        coverage=json.loads(coverage_raw) if coverage_raw is not None else None,
    )


__all__ = [
    "RUN_DATABASE_NAME",
    "RUN_SCHEMA_NAME",
    "RUN_SCHEMA_VERSION",
    "CredentialIdentity",
    "InvalidRunTransitionError",
    "RunEvent",
    "RunLeaseError",
    "RunLifecycle",
    "RunPhase",
    "RunSchemaError",
    "RunSnapshot",
    "RunStateError",
    "RunStateStore",
    "SemanticSpecMismatchError",
    "TargetState",
    "WorkUnit",
    "WorkUnitState",
    "canonical_json",
    "deterministic_work_unit_id",
    "semantic_run_digest",
]
