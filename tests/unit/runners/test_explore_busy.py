"""A busy rate limiter fails one step of a move, not the whole move."""

from __future__ import annotations

from unittest.mock import MagicMock

from lit_search.core.coverage import CoverageStatus
from lit_search.exceptions import RateLimitBusyError
from lit_search.runners.explore_runner import run_related


def test_related_reports_busy_steps_and_returns() -> None:
    connector = MagicMock()
    connector._post.side_effect = RateLimitBusyError("semantic_scholar:k", 30.0, 2.0)
    outcome = run_related(connector, ["ARXIV:2401.00001"])
    failed = [s for s in outcome.steps if s.status is CoverageStatus.FAILED]
    assert {s.source for s in failed} == {"related:recommendations", "related:edges"}
    assert all("nothing was sent" in (s.error or "") for s in failed)
    assert outcome.papers == []
