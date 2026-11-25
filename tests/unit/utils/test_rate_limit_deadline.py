"""A slot past the caller's deadline is refused without being reserved."""

from __future__ import annotations

import time

import pytest

from lit_search.exceptions import RateLimitBusyError
from lit_search.utils import rate_limit as rl


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rl, "_sleep", lambda _s: None)


def _scope() -> rl.RateLimitScope:
    return rl.RateLimitScope.build("deadline_svc", ("key",))


def test_slot_within_deadline_is_reserved() -> None:
    scope = _scope()
    with rl.request_deadline(time.time() + 30):
        rl.reserve_slot(scope, 5.0)
        rl.reserve_slot(scope, 5.0)  # second slot 5 s out, inside 30 s


def test_slot_past_deadline_raises_and_reserves_nothing() -> None:
    scope = _scope()
    rl.reserve_slot(scope, 10.0)  # next free slot is now ~10 s away
    with rl.request_deadline(time.time() + 2), pytest.raises(RateLimitBusyError) as caught:
        rl.reserve_slot(scope, 10.0)
    assert caught.value.retry_after > 5
    assert "nothing was sent" in str(caught.value)
    # The refused call did not push the queue: the next caller still gets the ~10 s slot.
    wait = rl.reserve_slot(scope, 10.0)
    assert wait.waited < 11


def test_deadline_is_lifted_outside_the_context() -> None:
    scope = _scope()
    rl.reserve_slot(scope, 10.0)
    with rl.request_deadline(time.time() + 1), pytest.raises(RateLimitBusyError):
        rl.reserve_slot(scope, 10.0)
    rl.reserve_slot(scope, 10.0)  # no deadline: waits (sleep is stubbed) instead of raising


def test_penalize_respects_max_factor() -> None:
    scope = rl.RateLimitScope.build("refusal_svc", ("key",))
    assert rl.penalize(scope, 0.1, slow_seconds=60, max_factor=1.5) == 1.5
    assert rl.penalize(scope, 0.1, slow_seconds=60, max_factor=1.5) == 1.5
    assert rl.penalize(scope, 0.1, slow_seconds=60, max_factor=1.0) == 1.0


def test_semantic_scholar_retries_quickly_without_long_slowdown() -> None:
    from lit_search.connectors.semantic_scholar import SemanticScholarConnector

    assert SemanticScholarConnector._refusal_max_factor <= 1.5
    assert SemanticScholarConnector._refusal_slow_seconds <= 60
    assert SemanticScholarConnector._rate_limit_base_delay <= 1.0
