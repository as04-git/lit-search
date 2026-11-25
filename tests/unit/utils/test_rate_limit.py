"""Unit tests for the credential-scoped, cross-process rate limiter.

The cross-process contention tests live in
``tests/unit/utils/test_rate_limit_cross_process.py`` because they spawn real
subprocesses and are slower.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess  # nosec B404: fixed argv, no shell, test-only.
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lit_search.connectors.arxiv import ArxivConnector
from lit_search.connectors.connector_base import ConnectorBase
from lit_search.connectors.crossref import CrossRefConnector
from lit_search.connectors.pubmed import PubmedConnector
from lit_search.connectors.semantic_scholar import SemanticScholarConnector
from lit_search.connectors.web_scraping import WebScrapingConnector
from lit_search.exceptions import (
    ConnectorError,
    RateLimitLockTimeoutError,
    RateLimitStoreUnavailableError,
)
from lit_search.utils import rate_limit as rl

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubConnector(ConnectorBase):
    """Minimal concrete connector used to exercise the base-class wiring."""

    def __init__(self, api_key: str | None = None, interval: float = 0.0) -> None:
        """Store the credential and the declared interval."""
        super().__init__()
        self._api_key = api_key
        self._interval = interval

    @property
    def name(self) -> str:
        """Return the connector name."""
        return "stub_service"

    @property
    def min_request_interval(self) -> float:
        """Return the declared interval."""
        return self._interval


def _scope(service: str = "svc", *credentials: str | None) -> rl.RateLimitScope:
    """Build a scope with less ceremony."""
    return rl.RateLimitScope.build(service, credentials)


_REPO_ROOT = Path(__file__).resolve().parents[3]

# Holds the lock for one scope until killed.  Run in a real subprocess rather
# than a forked child: ``os.fork`` in a multi-threaded interpreter is deprecated
# from Python 3.12, and a separate interpreter is a truer model of the situation
# being tested anyway.
_HOLDER = """
import sys
import time

from lit_search.utils.rate_limit import RateLimitScope, _locked_state

scope = RateLimitScope.build(sys.argv[1], [sys.argv[2]] if sys.argv[2] else [])
with _locked_state(scope, 5.0):
    sys.stdout.write("held\\n")
    sys.stdout.flush()
    time.sleep(60)
"""

#: Signature of the ``hold_lock`` fixture: takes a scope, returns the live
#: holder process once it has confirmed the lock is taken.
_HoldLock = Callable[[rl.RateLimitScope], "subprocess.Popen[str]"]


@pytest.fixture
def hold_lock(tmp_path: Path) -> Iterator[_HoldLock]:
    """Yield a factory that starts a subprocess holding a scope's lock.

    Yields
    ------
    _HoldLock
        Callable taking a scope and returning the running holder, already past
        the point of having acquired the lock.  Any holder still alive at
        teardown is killed.
    """
    script = tmp_path / "hold_lock.py"
    script.write_text(_HOLDER, encoding="utf-8")
    started: list[subprocess.Popen[str]] = []

    def _start(scope: rl.RateLimitScope) -> subprocess.Popen[str]:
        credential = "" if scope.is_anonymous else "k"
        process = subprocess.Popen(  # nosec B603: fixed argv from test constants, no shell.
            [sys.executable, str(script), scope.service, credential],
            cwd=str(_REPO_ROOT),
            env={
                **os.environ,
                rl.ENV_LOCK_DIR: str(rl.rate_limit_dir()),
                "PYTHONPATH": str(_REPO_ROOT),
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        started.append(process)
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "held"
        return process

    yield _start

    for process in started:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)


# ---------------------------------------------------------------------------
# Scope identity
# ---------------------------------------------------------------------------


class TestRateLimitScope:
    """Credential identity derivation."""

    def test_same_credential_same_scope(self) -> None:
        """Identical credentials on one service produce one bucket."""
        assert _scope("s2", "secret-key") == _scope("s2", "secret-key")

    def test_different_credentials_are_different_scopes(self) -> None:
        """Two keys against one service do not share a clock."""
        assert _scope("s2", "key-a") != _scope("s2", "key-b")

    def test_same_credential_different_service_is_different_scope(self) -> None:
        """One key reused at two services gets one bucket per service."""
        assert _scope("s2", "k") != _scope("openalex", "k")

    def test_missing_credential_falls_back_to_anonymous(self) -> None:
        """No credential means the shared unauthenticated host bucket."""
        scope = _scope("arxiv")
        assert scope.is_anonymous
        assert scope.fingerprint == rl.ANONYMOUS_FINGERPRINT
        assert scope.key == "arxiv:anon"

    def test_blank_and_none_credentials_are_dropped(self) -> None:
        """Whitespace and ``None`` do not count as credentials."""
        assert _scope("svc", None, "   ", "") == _scope("svc")

    def test_credential_never_appears_in_key_or_filename(self) -> None:
        """The raw key must not reach a log line or a path name."""
        scope = _scope("s2", "super-secret-token")
        assert "super-secret-token" not in scope.key
        assert "super-secret-token" not in scope.filename

    def test_service_is_normalised_to_path_safe_characters(self) -> None:
        """Odd service names cannot escape the lock directory."""
        scope = _scope("../Evil Service/", "k")
        assert scope.service == ".._evil_service_"
        assert "/" not in scope.filename

    def test_empty_service_is_rejected(self) -> None:
        """An empty service name is a programming error, not a default."""
        with pytest.raises(ValueError, match="must not be empty"):
            _scope("   ")

    def test_credential_ordering_is_significant(self) -> None:
        """Multi-part credentials hash in declaration order."""
        assert _scope("svc", "a", "b") != _scope("svc", "b", "a")


class TestRateLimitQuota:
    """Quota validation."""

    @pytest.mark.parametrize(("limit", "window"), [(0, 60.0), (-1, 60.0), (10, 0.0), (10, -5.0)])
    def test_invalid_quota_rejected(self, limit: int, window: float) -> None:
        """Non-positive limits or windows are rejected at construction."""
        with pytest.raises(ValueError, match="must be positive"):
            rl.RateLimitQuota(limit=limit, window_seconds=window)


# ---------------------------------------------------------------------------
# Lock directory
# ---------------------------------------------------------------------------


class TestLockDirectory:
    """Resolution and failure behaviour of the shared state directory."""

    def test_env_override_wins(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The environment variable relocates the shared state."""
        target = tmp_path / "elsewhere"
        monkeypatch.setenv(rl.ENV_LOCK_DIR, str(target))
        assert rl.rate_limit_dir() == target
        assert target.is_dir()

    def test_xdg_cache_home_used_when_no_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without an override the shared state lands under the cache home."""
        monkeypatch.delenv(rl.ENV_LOCK_DIR, raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        assert rl.rate_limit_dir() == tmp_path / "lit-search" / "ratelimit"

    def test_unwritable_directory_fails_loudly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unusable store raises instead of silently disabling pacing."""
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("", encoding="utf-8")
        monkeypatch.setenv(rl.ENV_LOCK_DIR, str(blocker / "sub"))
        with pytest.raises(RateLimitStoreUnavailableError):
            rl.rate_limit_dir()

    def test_store_failure_is_not_a_connector_error(self) -> None:
        """Limiter failures must not be swallowed as per-database failures.

        ``SearchConnectorBase.search`` converts ``ConnectorError`` into an empty
        result list.  An unpaced run must never be mistaken for a database with
        nothing to say, so the rate-limit errors sit outside that hierarchy.
        """
        assert not issubclass(RateLimitStoreUnavailableError, ConnectorError)
        assert not issubclass(RateLimitLockTimeoutError, ConnectorError)


# ---------------------------------------------------------------------------
# Reservation semantics (single process)
# ---------------------------------------------------------------------------


class TestReserveSlot:
    """Interval pacing within one process."""

    def test_zero_interval_without_quota_never_touches_disk(self, tmp_path: Path) -> None:
        """A connector declaring no limit pays no file I/O."""
        scope = _scope("noop")
        wait = rl.reserve_slot(scope, 0.0)
        assert wait.waited == 0.0
        assert wait.reason == "none"
        assert not (rl.rate_limit_dir() / scope.filename).exists()

    def test_first_reservation_does_not_wait(self) -> None:
        """A cold bucket serves immediately."""
        wait = rl.reserve_slot(_scope("svc", "k"), 0.5)
        assert wait.waited == 0.0
        assert wait.reason == "none"

    def test_second_reservation_waits_the_interval(self) -> None:
        """Consecutive reservations are spaced by the declared interval."""
        scope = _scope("svc", "k")
        rl.reserve_slot(scope, 0.3)
        started = time.monotonic()
        wait = rl.reserve_slot(scope, 0.3)
        elapsed = time.monotonic() - started
        assert wait.reason == "interval"
        assert wait.waited == pytest.approx(0.3, abs=0.05)
        assert elapsed >= 0.28

    def test_distinct_credentials_do_not_block_each_other(self) -> None:
        """Two keys on one service run in parallel, as their quotas allow."""
        rl.reserve_slot(_scope("svc", "key-a"), 5.0)
        started = time.monotonic()
        rl.reserve_slot(_scope("svc", "key-b"), 5.0)
        assert time.monotonic() - started < 0.5

    def test_separate_instances_share_one_clock(self) -> None:
        """Object churn does not reset the clock.

        This is the single-process half of the defect: the snowball path builds
        a fresh ``GetRunner``, and therefore fresh connectors, for every DOI.
        With a per-instance float the interval was never enforced between DOIs.
        """
        first = _StubConnector(api_key="shared-key", interval=0.3)
        second = _StubConnector(api_key="shared-key", interval=0.3)
        assert first.rate_limit_scope == second.rate_limit_scope

        first._rate_limit()
        started = time.monotonic()
        second._rate_limit()
        assert time.monotonic() - started >= 0.28

    def test_state_file_is_created_once_per_scope(self) -> None:
        """One bucket means one file, reused."""
        scope = _scope("svc", "k")
        rl.reserve_slot(scope, 0.01)
        rl.reserve_slot(scope, 0.01)
        assert list(rl.rate_limit_dir().iterdir()) == [rl.rate_limit_dir() / scope.filename]


# ---------------------------------------------------------------------------
# Stale / corrupt shared state
# ---------------------------------------------------------------------------


class TestStaleState:
    """Recovery from corrupt records and stepped clocks."""

    @pytest.mark.parametrize(
        "payload",
        [b"", b"not json", b"[]", b'{"v":999,"next_slot":1e18}', b'{"v":1,"next_slot":"soon"}'],
    )
    def test_unreadable_state_is_treated_as_cold(self, payload: bytes) -> None:
        """Garbage on disk must not wedge or mispace the limiter."""
        scope = _scope("svc", "k")
        path = rl.rate_limit_dir() / scope.filename
        path.write_bytes(payload)
        wait = rl.reserve_slot(scope, 0.05)
        assert wait.waited == 0.0

    def test_far_future_slot_is_discarded(self, caplog: pytest.LogCaptureFixture) -> None:
        """A slot beyond any legitimate reservation is a stepped clock.

        Trusting it would stall every process for an unbounded time, so the
        record is reset and the reset is logged.
        """
        scope = _scope("svc", "k")
        path = rl.rate_limit_dir() / scope.filename
        far_future = time.time() + 86_400
        path.write_text(
            json.dumps({"v": 1, "next_slot": far_future, "window_start": 0.0, "count": 0}),
            encoding="utf-8",
        )
        with caplog.at_level(logging.WARNING, logger="lit_search.utils.rate_limit"):
            wait = rl.reserve_slot(scope, 0.05)
        assert wait.waited == 0.0
        assert any("implausible state" in r.message for r in caplog.records)

    def test_slot_just_inside_the_horizon_is_honoured(self) -> None:
        """The stale guard must not discard a legitimate reservation."""
        scope = _scope("svc", "k")
        path = rl.rate_limit_dir() / scope.filename
        legitimate = time.time() + 0.4
        path.write_text(
            json.dumps({"v": 1, "next_slot": legitimate, "window_start": 0.0, "count": 0}),
            encoding="utf-8",
        )
        wait = rl.reserve_slot(scope, 0.5)
        assert wait.reason == "interval"
        assert wait.waited > 0.2

    def test_past_slot_is_served_immediately(self) -> None:
        """A stale-in-the-past record simply means the bucket is free."""
        scope = _scope("svc", "k")
        path = rl.rate_limit_dir() / scope.filename
        path.write_text(
            json.dumps({"v": 1, "next_slot": time.time() - 3600, "window_start": 0.0, "count": 0}),
            encoding="utf-8",
        )
        assert rl.reserve_slot(scope, 1.0).waited == 0.0


# ---------------------------------------------------------------------------
# Quota windows
# ---------------------------------------------------------------------------


class TestQuota:
    """Budget-shaped limits layered on top of the interval."""

    def test_requests_inside_the_budget_are_not_stalled(self) -> None:
        """Under budget, only the interval applies."""
        scope = _scope("openaire", "k")
        quota = rl.RateLimitQuota(limit=3, window_seconds=3600.0)
        for _ in range(3):
            assert rl.reserve_slot(scope, 0.0, quota=quota).reason != "quota"

    def test_exhausted_budget_stalls_to_the_next_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fourth request in a 3-per-window budget waits for the window.

        Without this, a connector paced perfectly at its interval still burns
        an hourly allowance in seconds and then returns errors for the rest of
        the hour -- which reads as empty recall, not as a limit.
        """
        scope = _scope("openaire", "k")
        quota = rl.RateLimitQuota(limit=3, window_seconds=3600.0)
        slept: list[float] = []
        monkeypatch.setattr(rl, "_sleep", slept.append)

        for _ in range(3):
            rl.reserve_slot(scope, 0.0, quota=quota)
        wait = rl.reserve_slot(scope, 0.0, quota=quota)

        assert wait.reason == "quota"
        assert wait.waited == pytest.approx(3600.0, abs=1.0)
        assert slept[-1] == pytest.approx(3600.0, abs=1.0)

    def test_window_rolls_over_when_time_advances(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A fresh window restores the full budget."""
        scope = _scope("openaire", "k")
        quota = rl.RateLimitQuota(limit=2, window_seconds=100.0)
        clock = [1_000_000.0]
        monkeypatch.setattr(rl, "_now", lambda: clock[0])
        monkeypatch.setattr(rl, "_sleep", lambda _s: None)

        rl.reserve_slot(scope, 0.0, quota=quota)
        rl.reserve_slot(scope, 0.0, quota=quota)
        clock[0] += 101.0
        assert rl.reserve_slot(scope, 0.0, quota=quota).reason == "none"


class TestRefusalBackoff:
    """A 429 seen by one process slows every process on the bucket."""

    def test_pause_then_widened_interval_then_recovery(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scope = _scope("semantic_scholar", "k")
        clock = [1_000_000.0]
        monkeypatch.setattr(rl, "_now", lambda: clock[0])
        monkeypatch.setattr(rl, "_sleep", lambda _s: None)

        assert rl.penalize(scope, 5.0) == 2.0
        first = rl.reserve_slot(scope, 1.0)
        assert first.waited == pytest.approx(5.0)
        clock[0] += 5.0
        assert rl.reserve_slot(scope, 1.0).waited == pytest.approx(2.0)

        clock[0] += rl.SLOWDOWN_SECONDS + 10
        rl.reserve_slot(scope, 1.0)
        assert rl.reserve_slot(scope, 1.0).waited == pytest.approx(1.0)

    def test_repeated_refusals_escalate_up_to_a_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        scope = _scope("arxiv", None)
        monkeypatch.setattr(rl, "_now", lambda: 1_000_000.0)
        factors = [rl.penalize(scope, 1.0) for _ in range(5)]
        assert factors == [2.0, 4.0, 8.0, 8.0, 8.0]


# ---------------------------------------------------------------------------
# Surfacing (D6): visible, never actionable
# ---------------------------------------------------------------------------


class TestSurfacing:
    """Waits must be diagnosable after the fact."""

    def test_active_limit_is_announced_once_per_scope(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The limit in force is stated the first time a bucket is used."""
        scope = _scope("s2", "k")
        with caplog.at_level(logging.INFO, logger="lit_search.utils.rate_limit"):
            rl.reserve_slot(scope, 0.01)
            rl.reserve_slot(scope, 0.01)
        announcements = [r for r in caplog.records if "min interval" in r.message]
        assert len(announcements) == 1
        assert "s2:" in announcements[0].message

    def test_anonymous_scope_is_named_as_such(self) -> None:
        """A missing credential is visible, so a mis-declaration is catchable."""
        assert "no credential" in rl.describe_limit(_scope("arxiv"), 3.0, None)
        assert "credential-scoped" in rl.describe_limit(_scope("s2", "k"), 1.1, None)

    def test_short_wait_logs_at_debug_and_long_wait_at_info(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Routine pacing does not flood; an unusual stall is noticed."""
        monkeypatch.setattr(rl, "_sleep", lambda _s: None)
        with caplog.at_level(logging.DEBUG, logger="lit_search.utils.rate_limit"):
            short = _scope("svc", "short")
            rl.reserve_slot(short, 0.5)
            rl.reserve_slot(short, 0.5)
            long_scope = _scope("svc", "long")
            rl.reserve_slot(long_scope, 30.0)
            rl.reserve_slot(long_scope, 30.0)
        levels = {r.levelno for r in caplog.records if "waiting" in r.message}
        assert levels == {logging.DEBUG, logging.INFO}

    def test_stats_accumulate_per_scope(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Totals are recorded so a slow run can be explained.

        ``_sleep`` is stubbed out, so the reservations queue without wall-clock
        time passing: the waits are 0s, 2s and 4s.  That is the correct
        behaviour to assert -- each caller is told how long it must wait for the
        slot it reserved, not how long the previous caller waited.
        """
        monkeypatch.setattr(rl, "_sleep", lambda _s: None)
        scope = _scope("s2", "k")
        for _ in range(3):
            rl.reserve_slot(scope, 2.0)

        (stats,) = rl.rate_limit_stats()
        assert stats.scope_key == scope.key
        assert stats.requests == 3
        assert stats.waits == 2
        assert stats.total_wait_seconds == pytest.approx(6.0, abs=0.2)
        assert stats.max_wait_seconds == pytest.approx(4.0, abs=0.2)

    def test_report_names_the_scope_and_the_total(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The rendered report is what a runner prints at the end of a run."""
        monkeypatch.setattr(rl, "_sleep", lambda _s: None)
        scope = _scope("s2", "k")
        rl.reserve_slot(scope, 1.1)
        rl.reserve_slot(scope, 1.1)
        report = rl.format_rate_limit_report()
        assert scope.key in report
        assert "interval 1.10s" in report

    def test_report_is_empty_when_nothing_waited(self) -> None:
        """No wait, no noise."""
        rl.reserve_slot(_scope("svc", "k"), 0.01)
        assert rl.format_rate_limit_report() == ""

    def test_stats_are_ordered_worst_first(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The scope costing the most time is the one to look at."""
        monkeypatch.setattr(rl, "_sleep", lambda _s: None)
        cheap, costly = _scope("a", "k"), _scope("b", "k")
        rl.reserve_slot(cheap, 0.1)
        rl.reserve_slot(cheap, 0.1)
        rl.reserve_slot(costly, 9.0)
        rl.reserve_slot(costly, 9.0)
        assert [s.scope_key for s in rl.rate_limit_stats()] == [costly.key, cheap.key]


# ---------------------------------------------------------------------------
# Connector wiring
# ---------------------------------------------------------------------------


class TestConnectorScopes:
    """The scopes the shipped connectors actually declare."""

    def test_semantic_scholar_key_is_the_bucket(self) -> None:
        """One key, one clock, regardless of which endpoint is called."""
        keyed = SemanticScholarConnector(api_key="abc")
        assert keyed.rate_limit_scope.service == "semantic_scholar"
        assert not keyed.rate_limit_scope.is_anonymous
        assert keyed.rate_limit_scope == SemanticScholarConnector(api_key="abc").rate_limit_scope
        assert keyed.rate_limit_scope != SemanticScholarConnector(api_key="xyz").rate_limit_scope

    def test_semantic_scholar_without_key_is_anonymous(self) -> None:
        """An unkeyed client draws on the shared unauthenticated pool."""
        assert SemanticScholarConnector().rate_limit_scope.is_anonymous

    def test_pubmed_reports_ncbi_as_the_quota_authority(self) -> None:
        """PubMed and a future PMC connector must share one NCBI bucket."""
        assert PubmedConnector(api_key="k").rate_limit_service == "ncbi"
        assert PubmedConnector(api_key="k").rate_limit_scope.service == "ncbi"

    def test_crossref_is_keyed_by_polite_pool_email(self) -> None:
        """CrossRef's identity is the mailto, not an API key."""
        polite = CrossRefConnector(email="a@example.edu")
        assert not polite.rate_limit_scope.is_anonymous
        assert polite.rate_limit_scope != CrossRefConnector(email="b@example.edu").rate_limit_scope
        assert CrossRefConnector().rate_limit_scope.is_anonymous

    def test_arxiv_is_host_scoped(self) -> None:
        """No credential exists, so every arXiv caller on this host shares one clock."""
        scope = ArxivConnector().rate_limit_scope
        assert scope.is_anonymous
        assert scope.key == "arxiv:anon"

    def test_web_scraping_opts_out(self) -> None:
        """Per-host limits are not expressible as one credential scope."""
        connector = WebScrapingConnector()
        assert connector.min_request_interval == 0.0
        assert connector._credential_attrs == ()

    def test_scope_is_cached_per_instance(self) -> None:
        """Scope derivation happens once, not on every request."""
        connector = _StubConnector(api_key="k")
        assert connector.rate_limit_scope is connector.rate_limit_scope

    def test_undeclared_credential_falls_back_conservatively(self) -> None:
        """A connector that hides its key over-throttles; it never over-sends."""

        class _Undeclared(_StubConnector):
            def __init__(self) -> None:
                super().__init__()
                self._token = "undeclared-key"

        assert _Undeclared().rate_limit_scope.is_anonymous

    def test_request_quota_defaults_to_none(self) -> None:
        """Quotas are opt-in; interval-only services declare nothing."""
        assert _StubConnector().request_quota is None

    def test_rate_limit_is_invoked_before_each_http_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every request start, including retries, reserves a slot."""
        reserved: list[str] = []
        monkeypatch.setattr(
            rl,
            "reserve_slot",
            lambda scope, interval, **_kw: (
                reserved.append(scope.key) or rl.RateLimitWait(scope.key, interval, 0.0, "none")
            ),
        )
        monkeypatch.setattr(
            "lit_search.connectors.connector_base.reserve_slot",
            lambda scope, interval, **_kw: (
                reserved.append(scope.key) or rl.RateLimitWait(scope.key, interval, 0.0, "none")
            ),
        )
        connector = _StubConnector(api_key="k", interval=1.0)
        response = MagicMock(status_code=200, reason="OK", headers={}, content=b"{}")
        response.ok = True
        connector._http_session = MagicMock()
        connector._http_session.get.return_value = response

        connector._get("https://example.invalid/x")

        assert reserved == [connector.rate_limit_scope.key]


# ---------------------------------------------------------------------------
# Lock mechanics
# ---------------------------------------------------------------------------


class TestLockMechanics:
    """Behaviour of the file lock itself."""

    def test_lock_is_released_after_a_reservation(self) -> None:
        """The lock is not held across the sleep, only across the bookkeeping."""
        scope = _scope("svc", "k")
        rl.reserve_slot(scope, 0.01)
        with rl._locked_state(scope, 1.0) as (fd, state):
            assert state.next_slot > 0
            assert isinstance(fd, int)

    def test_state_file_permissions_are_owner_only(self) -> None:
        """Fingerprints are not secrets, but the store stays private anyway."""
        scope = _scope("svc", "k")
        rl.reserve_slot(scope, 0.01)
        mode = (rl.rate_limit_dir() / scope.filename).stat().st_mode
        assert mode & 0o077 == 0

    def test_write_then_read_round_trips(self) -> None:
        """The record survives a lock/unlock cycle intact."""
        scope = _scope("svc", "k")
        with rl._locked_state(scope, 1.0) as (fd, _state):
            rl._write_state(fd, rl._State(next_slot=12.5, window_start=3.0, count=7))
        with rl._locked_state(scope, 1.0) as (_fd, state):
            assert (state.next_slot, state.window_start, state.count) == (12.5, 3.0, 7)

    def test_shorter_record_does_not_leave_trailing_bytes(self) -> None:
        """The file is truncated on write, so a stale tail cannot be read back."""
        scope = _scope("svc", "k")
        with rl._locked_state(scope, 1.0) as (fd, _state):
            rl._write_state(fd, rl._State(next_slot=1234567890.123456, count=999999))
        with rl._locked_state(scope, 1.0) as (fd, _state):
            rl._write_state(fd, rl._State())
        raw = (rl.rate_limit_dir() / scope.filename).read_bytes()
        assert json.loads(raw.decode("utf-8"))["count"] == 0

    def test_held_lock_times_out_loudly(self, hold_lock: _HoldLock) -> None:
        """A live but wedged holder produces an error, never a silent bypass.

        A holder that *dies* needs no timeout: the kernel drops its ``flock``
        when the file descriptor closes.  This covers the other case, where the
        holder is alive and stuck.  Timing out loudly is deliberate -- carrying
        on without the lock would send unpaced requests, and the symptom of that
        (429 storms turning into empty results) is invisible.
        """
        scope = _scope("svc", "k")
        holder = hold_lock(scope)
        with pytest.raises(RateLimitLockTimeoutError, match="wedged holder"):
            rl.reserve_slot(scope, 0.1, lock_timeout=0.3)
        holder.kill()

    def test_lock_survives_holder_death(self, hold_lock: _HoldLock) -> None:
        """SIGKILL while holding the lock must not wedge the bucket.

        This is why the mechanism is ``flock`` rather than an exclusive-create
        lock file: there is no stale lock to detect, because the kernel releases
        the lock when the dead process's file descriptor is closed.
        """
        scope = _scope("svc", "k")
        holder = hold_lock(scope)
        holder.kill()

        started = time.monotonic()
        rl.reserve_slot(scope, 0.01, lock_timeout=5.0)
        assert time.monotonic() - started < 1.0
