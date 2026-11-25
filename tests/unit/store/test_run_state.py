"""Contract tests for crash-safe traversal run state."""

from __future__ import annotations

import datetime as dt
import sqlite3
import threading
from pathlib import Path

import pytest

from lit_search.store.run_state import (
    CredentialIdentity,
    InvalidRunTransitionError,
    RunLeaseError,
    RunLifecycle,
    RunPhase,
    RunSchemaError,
    RunStateStore,
    SemanticSpecMismatchError,
    TargetState,
    WorkUnitState,
    deterministic_work_unit_id,
    semantic_run_digest,
)

RUN_ID = "r_0123456789abcdef0123456789abcdef"


def _spec(*, depth: int = 2) -> dict[str, object]:
    return {
        "mode": "interactive",
        "direction": "both",
        "max_depth": depth,
        "sources": ["semantic_scholar"],
        "targeting_policy": {"version": 1, "budget": 500},
    }


def _create(tmp_path: Path, *, depth: int = 2) -> RunStateStore:
    return RunStateStore.create(tmp_path, _spec(depth=depth), run_id=RUN_ID)


class TestSchemaAndSemanticIdentity:
    def test_create_uses_wal_and_expected_location(self, tmp_path: Path) -> None:
        with _create(tmp_path) as store:
            assert store.path == tmp_path / "runs" / RUN_ID / "run.sqlite"
            connection = sqlite3.connect(store.path)
            try:
                assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
                assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
            finally:
                connection.close()

    def test_run_spec_digest_is_mapping_order_independent_and_exact(self) -> None:
        first = {
            "mode": "interactive",
            "sources": ["semantic_scholar", "openalex"],
            "budget": {"requests": 30, "seconds": 600},
        }
        reordered = {
            "budget": {"seconds": 600, "requests": 30},
            "sources": ["semantic_scholar", "openalex"],
            "mode": "interactive",
        }
        assert semantic_run_digest(first) == semantic_run_digest(reordered)

        with_timestamp = dict(first, created_at="2026-08-31T10:00:00+00:00")
        assert semantic_run_digest(first) != semantic_run_digest(with_timestamp)

    def test_runtime_history_is_not_part_of_the_spec_api(self, tmp_path: Path) -> None:
        with _create(tmp_path) as store:
            digest = store.snapshot().semantic_digest
            store.acquire_lease("worker-a", ttl_seconds=10)
            store.release_lease("worker-a")
            store.transition_phase(RunPhase.SEED_FETCH)
            assert store.snapshot().semantic_digest == digest

    def test_resume_refuses_semantic_config_change(self, tmp_path: Path) -> None:
        with _create(tmp_path):
            pass
        with pytest.raises(SemanticSpecMismatchError, match="require a forked run"):
            RunStateStore.resume(tmp_path, RUN_ID, _spec(depth=3))

    def test_resume_refuses_unknown_schema_version(self, tmp_path: Path) -> None:
        with _create(tmp_path) as store:
            path = store.path
        connection = sqlite3.connect(path)
        try:
            connection.execute("PRAGMA user_version = 999")
        finally:
            connection.close()
        with pytest.raises(RunSchemaError, match="version 999 is unsupported"):
            RunStateStore.resume(tmp_path, RUN_ID, _spec())

    def test_only_credential_fingerprint_is_persisted(self, tmp_path: Path) -> None:
        secret = "a-real-secret-value"
        identity = CredentialIdentity.from_material("semantic_scholar", secret)
        with RunStateStore.create(
            tmp_path,
            _spec(),
            run_id=RUN_ID,
            credentials=(identity,),
        ) as store:
            assert store.credential_identities == (identity,)
        raw = (tmp_path / "runs" / RUN_ID / "run.sqlite").read_bytes()
        assert secret.encode() not in raw


class TestLifecycleAndPhase:
    def test_valid_phase_and_lifecycle_path(self, tmp_path: Path) -> None:
        with _create(tmp_path) as store:
            store.transition_phase(RunPhase.SEED_FETCH)
            store.transition_phase(RunPhase.LEVEL_PLAN, level=1)
            store.transition_phase(RunPhase.LEVEL_FETCH, level=1)
            store.transition_phase(RunPhase.LEVEL_REDUCE, level=1)
            store.transition_phase(RunPhase.LEVEL_PLAN, level=2)
            store.transition_phase(RunPhase.FINALIZE)
            store.transition_lifecycle(RunLifecycle.COMPLETED, reason="depth limit reached")
            snapshot = store.snapshot()
            assert snapshot.lifecycle is RunLifecycle.COMPLETED
            assert snapshot.phase is RunPhase.FINALIZE

    def test_invalid_transition_is_refused_without_an_event(self, tmp_path: Path) -> None:
        with _create(tmp_path) as store:
            before = store.events()
            with pytest.raises(InvalidRunTransitionError, match="seed_plan to level_fetch"):
                store.transition_phase(RunPhase.LEVEL_FETCH, level=1)
            assert store.events() == before
            store.transition_phase(RunPhase.SEED_FETCH)
            assert store.snapshot().phase is RunPhase.SEED_FETCH

    def test_completed_run_cannot_be_leased_or_reactivated(self, tmp_path: Path) -> None:
        with _create(tmp_path) as store:
            store.transition_phase(RunPhase.FINALIZE)
            store.transition_lifecycle(RunLifecycle.COMPLETED, reason="policy satisfied")
            with pytest.raises(RunLeaseError, match="completed"):
                store.acquire_lease("worker", ttl_seconds=30)
            with pytest.raises(InvalidRunTransitionError, match="completed to active"):
                store.transition_lifecycle(RunLifecycle.ACTIVE, reason="try again")

    def test_terminal_run_refuses_target_work_and_reduction_mutation(self, tmp_path: Path) -> None:
        with _create(tmp_path) as store:
            target = "s2:seed"
            store.discover_target(target)
            unit_id = store.plan_work_unit(
                phase=RunPhase.SEED_FETCH,
                level=None,
                source="semantic_scholar",
                request_kind="seed_batch",
                ordinal=0,
                targets=[target],
            )
            store.begin_work_unit(unit_id)
            store.complete_work_unit(
                unit_id,
                WorkUnitState.SUCCEEDED,
                outcome={"papers": 1},
                coverage={"status": "complete"},
            )
            store.transition_phase(RunPhase.FINALIZE)
            store.transition_lifecycle(RunLifecycle.COMPLETED, reason="policy satisfied")

            with pytest.raises(InvalidRunTransitionError, match="completed"):
                store.discover_target("s2:late")
            with pytest.raises(InvalidRunTransitionError, match="completed"):
                store.transition_target(target, TargetState.SCHEDULED)
            with pytest.raises(InvalidRunTransitionError, match="completed"):
                store.plan_work_unit(
                    phase=RunPhase.SEED_FETCH,
                    level=None,
                    source="semantic_scholar",
                    request_kind="seed_batch",
                    ordinal=1,
                    targets=["s2:late"],
                )
            with pytest.raises(InvalidRunTransitionError, match="completed"):
                store.commit_reduction(unit_id, "late", {"paper": target})

    def test_live_lease_refuses_another_owner_but_expired_lease_is_recoverable(
        self, tmp_path: Path
    ) -> None:
        now = dt.datetime(2026, 8, 31, tzinfo=dt.UTC)
        with _create(tmp_path) as store:
            store.acquire_lease("worker-a", ttl_seconds=10, now=now)
            with pytest.raises(RunLeaseError, match="worker-a"):
                store.acquire_lease("worker-b", ttl_seconds=10, now=now)
            store.acquire_lease("worker-b", ttl_seconds=10, now=now + dt.timedelta(seconds=11))
            assert store.snapshot().lease_owner == "worker-b"

    def test_competing_connection_cannot_steal_lease_after_stale_read(self, tmp_path: Path) -> None:
        now = dt.datetime(2026, 8, 31, tzinfo=dt.UTC)
        with _create(tmp_path) as store:
            path = store.path
            holder = sqlite3.connect(path, timeout=5.0)
            contender_connection = sqlite3.connect(path, timeout=5.0, check_same_thread=False)
            contender = RunStateStore(path, contender_connection)
            began_immediate = threading.Event()
            contender_connection.set_trace_callback(
                lambda statement: (
                    began_immediate.set()
                    if statement.strip().upper() == "BEGIN IMMEDIATE"
                    else None
                )
            )
            errors: list[BaseException] = []

            holder.execute("BEGIN IMMEDIATE")
            holder.execute(
                """
                UPDATE run_state SET lease_owner = ?, lease_expires_at = ?
                WHERE singleton = 1
                """,
                ("worker-a", (now + dt.timedelta(minutes=1)).isoformat()),
            )

            def compete() -> None:
                try:
                    contender.acquire_lease("worker-b", ttl_seconds=10, now=now)
                except BaseException as error:  # Captured for assertion in the parent thread.
                    errors.append(error)

            thread = threading.Thread(target=compete)
            thread.start()
            assert began_immediate.wait(timeout=2.0)
            holder.commit()
            thread.join(timeout=5.0)
            try:
                assert not thread.is_alive()
                assert len(errors) == 1
                assert isinstance(errors[0], RunLeaseError)
                assert contender.snapshot().lease_owner == "worker-a"
            finally:
                contender.close()
                holder.close()


class TestTargetsAndWorkUnits:
    def test_target_retains_distinct_parent_and_source_assertions(self, tmp_path: Path) -> None:
        first = {"parent": "s2:parent-b", "source": "semantic_scholar"}
        second = {"parent": "s2:parent-a", "source": "openalex"}
        with _create(tmp_path) as store:
            assert store.discover_target("s2:child", evidence=first)
            single_assertion_digest = store.semantic_outcome_digest()
            assert store.discover_target("s2:child", evidence=second)
            plural_assertion_digest = store.semantic_outcome_digest()
            event_count = len(store.events())

            assert not store.discover_target(
                "s2:child",
                evidence={"source": "semantic_scholar", "parent": "s2:parent-b"},
            )
            assert len(store.events()) == event_count
            assert store.target_states() == (("s2:child", TargetState.DISCOVERED),)
            assert store.target_evidence("S2:CHILD") == (second, first)
            assert plural_assertion_digest != single_assertion_digest
            assert store.semantic_outcome_digest() == plural_assertion_digest

    def test_targets_are_distinct_states_in_canonical_order(self, tmp_path: Path) -> None:
        with _create(tmp_path) as store:
            store.discover_target("doi:Z")
            store.discover_target("doi:A")
            store.transition_target("doi:A", TargetState.SCHEDULED)
            store.transition_target("doi:A", TargetState.RESOLVED)
            store.transition_target("doi:A", TargetState.RESIDUAL)
            assert store.target_states() == (
                ("doi:a", TargetState.RESIDUAL),
                ("doi:z", TargetState.DISCOVERED),
            )

    def test_invalid_target_transition_is_refused(self, tmp_path: Path) -> None:
        with _create(tmp_path) as store:
            store.discover_target("doi:a")
            with pytest.raises(InvalidRunTransitionError, match="discovered to expanded"):
                store.transition_target("doi:a", TargetState.EXPANDED)

    def test_work_unit_id_is_canonical_and_plan_is_idempotent(self, tmp_path: Path) -> None:
        first = deterministic_work_unit_id(
            phase=RunPhase.LEVEL_FETCH,
            level=1,
            source="semantic_scholar",
            request_kind="edge_batch",
            targets=["s2:z", "s2:a", "s2:a"],
        )
        second = deterministic_work_unit_id(
            phase=RunPhase.LEVEL_FETCH,
            level=1,
            source="SEMANTIC_SCHOLAR",
            request_kind="EDGE_BATCH",
            targets=["s2:a", "s2:z"],
        )
        assert first == second

        with _create(tmp_path) as store:
            unit_id = store.plan_work_unit(
                phase=RunPhase.LEVEL_FETCH,
                level=1,
                source="semantic_scholar",
                request_kind="edge_batch",
                ordinal=0,
                targets=["s2:z", "s2:a"],
            )
            event_count = len(store.events())
            assert (
                store.plan_work_unit(
                    phase=RunPhase.LEVEL_FETCH,
                    level=1,
                    source="semantic_scholar",
                    request_kind="edge_batch",
                    ordinal=0,
                    targets=["s2:a", "s2:z"],
                )
                == unit_id
            )
            assert len(store.events()) == event_count
            assert store.work_units()[0].targets == ("s2:a", "s2:z")

    def test_work_units_follow_plan_sequence_not_phase_text(self, tmp_path: Path) -> None:
        with _create(tmp_path) as store:
            planned = [
                store.plan_work_unit(
                    phase=phase,
                    level=level,
                    source="semantic_scholar",
                    request_kind=f"request-{ordinal}",
                    ordinal=ordinal,
                    targets=[f"s2:{ordinal}"],
                )
                for ordinal, (phase, level) in enumerate(
                    [
                        (RunPhase.SEED_FETCH, None),
                        (RunPhase.LEVEL_FETCH, 1),
                        (RunPhase.ENRICH_FETCH, None),
                    ]
                )
            ]
            assert [unit.unit_id for unit in store.work_units()] == planned
            assert [unit.plan_sequence for unit in store.work_units()] == [0, 1, 2]

    def test_crash_reopen_preserves_plan_and_retries_indeterminate_attempt(
        self, tmp_path: Path
    ) -> None:
        store = _create(tmp_path)
        unit_id = store.plan_work_unit(
            phase=RunPhase.SEED_FETCH,
            level=None,
            source="semantic_scholar",
            request_kind="seed_batch",
            ordinal=0,
            targets=["s2:seed"],
        )
        assert store.begin_work_unit(unit_id) == 1
        store.close()  # Process ended after request start, before outcome commit.

        with RunStateStore.resume(tmp_path, RUN_ID, _spec()) as resumed:
            assert resumed.work_units()[0].state is WorkUnitState.IN_FLIGHT
            assert resumed.recover_in_flight() == 1
            assert resumed.work_units()[0].state is WorkUnitState.PLANNED
            assert resumed.begin_work_unit(unit_id) == 2
            assert resumed.complete_work_unit(
                unit_id,
                WorkUnitState.SUCCEEDED,
                outcome={"papers": 1},
                coverage={"status": "complete", "resolved": 1},
            )
            assert resumed.work_units()[0].attempts == 2
            assert any(
                event.event_type == "work_unit_attempt_indeterminate" for event in resumed.events()
            )

    def test_terminal_outcome_and_reduction_are_exactly_once(self, tmp_path: Path) -> None:
        with _create(tmp_path) as store:
            unit_id = store.plan_work_unit(
                phase=RunPhase.SEED_FETCH,
                level=None,
                source="semantic_scholar",
                request_kind="seed_batch",
                ordinal=0,
                targets=["s2:seed"],
            )
            store.begin_work_unit(unit_id)
            assert store.complete_work_unit(
                unit_id,
                WorkUnitState.SUCCEEDED,
                outcome={"paper_ids": ["s2:seed"]},
                coverage={"status": "complete"},
            )
            assert not store.complete_work_unit(
                unit_id,
                WorkUnitState.SUCCEEDED,
                outcome={"paper_ids": ["s2:seed"]},
                coverage={"status": "complete"},
            )
            assert store.commit_reduction(unit_id, "paper:s2:seed", {"merged": True})
            assert not store.commit_reduction(unit_id, "paper:s2:seed", {"merged": True})
            with pytest.raises(InvalidRunTransitionError, match="committed differently"):
                store.commit_reduction(unit_id, "paper:s2:seed", {"merged": False})

    def test_semantic_outcome_is_equal_after_crash_and_retry(self, tmp_path: Path) -> None:
        uninterrupted_root = tmp_path / "uninterrupted"
        retried_root = tmp_path / "retried"
        second_run_id = "r_fedcba9876543210fedcba9876543210"

        with RunStateStore.create(uninterrupted_root, _spec(), run_id=RUN_ID) as uninterrupted:
            first_unit = uninterrupted.plan_work_unit(
                phase=RunPhase.SEED_FETCH,
                level=None,
                source="semantic_scholar",
                request_kind="seed_batch",
                ordinal=0,
                targets=["s2:seed"],
            )
            uninterrupted.begin_work_unit(first_unit)
            uninterrupted.complete_work_unit(
                first_unit,
                WorkUnitState.SUCCEEDED,
                outcome={
                    "paper_ids": ["s2:seed"],
                    "finished_at": "2026-08-31T10:00:00+00:00",
                    "runtime_seconds": 1.2,
                },
                coverage={"status": "complete", "retry_history": []},
            )
            uninterrupted.commit_reduction(first_unit, "paper:s2:seed", {"merged": True})
            expected = uninterrupted.semantic_outcome_digest()

        retried = RunStateStore.create(retried_root, _spec(), run_id=second_run_id)
        second_unit = retried.plan_work_unit(
            phase=RunPhase.SEED_FETCH,
            level=None,
            source="semantic_scholar",
            request_kind="seed_batch",
            ordinal=0,
            targets=["s2:seed"],
        )
        retried.begin_work_unit(second_unit)
        retried.close()
        with RunStateStore.resume(retried_root, second_run_id, _spec()) as resumed:
            resumed.recover_in_flight()
            resumed.begin_work_unit(second_unit)
            resumed.complete_work_unit(
                second_unit,
                WorkUnitState.SUCCEEDED,
                outcome={
                    "paper_ids": ["s2:seed"],
                    "finished_at": "2026-08-31T11:00:00+00:00",
                    "runtime_seconds": 9.8,
                    "attempt_history": ["indeterminate", "succeeded"],
                },
                coverage={"status": "complete", "retry_history": ["crash"]},
            )
            resumed.commit_reduction(second_unit, "paper:s2:seed", {"merged": True})
            assert resumed.semantic_outcome_digest() == expected


class TestAuthoritativeEvents:
    def test_events_are_append_only_even_via_direct_sql(self, tmp_path: Path) -> None:
        with _create(tmp_path) as store:
            connection = sqlite3.connect(store.path)
            try:
                with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                    connection.execute("UPDATE events SET event_type = 'tampered'")
                with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                    connection.execute("DELETE FROM events")
            finally:
                connection.close()

    def test_state_changes_and_request_commits_have_ordered_events(self, tmp_path: Path) -> None:
        with _create(tmp_path) as store:
            unit_id = store.plan_work_unit(
                phase=RunPhase.SEED_FETCH,
                level=None,
                source="semantic_scholar",
                request_kind="seed_batch",
                ordinal=0,
                targets=["s2:seed"],
            )
            store.begin_work_unit(unit_id)
            store.complete_work_unit(
                unit_id,
                WorkUnitState.NOT_FOUND,
                outcome={"not_found": 1},
                coverage={"status": "complete"},
            )
            events = store.events()
            assert [event.sequence for event in events] == list(range(1, len(events) + 1))
            assert [event.event_type for event in events][-3:] == [
                "work_unit_planned",
                "work_unit_started",
                "work_unit_finished",
            ]
