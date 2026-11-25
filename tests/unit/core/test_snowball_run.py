"""Semantic contract tests for snowball modes, termination, and residuals."""

from __future__ import annotations

import datetime as dt
from dataclasses import FrozenInstanceError

import pytest

from lit_search.core.snowball_run import (
    ResidualEntry,
    ResidualReason,
    SnowballMode,
    TerminationReason,
    TraversalTermination,
)


def _residual(
    target: str = "semantic_scholar:child",
    *,
    reason: ResidualReason = ResidualReason.UNSELECTED,
    level: int = 2,
    direction: str = "forward",
    source: str = "semantic_scholar",
    selected: bool = False,
    expanded: bool = False,
    detail: str | None = None,
) -> ResidualEntry:
    return ResidualEntry(
        target_key=target,
        aliases=(target, "doi:10.0/child"),
        reason=reason,
        level=level,
        direction=direction,  # type: ignore[arg-type]
        asserting_source=source,
        parent_keys=("semantic_scholar:parent-b", "semantic_scholar:parent-a"),
        lanes=("semantic", "bridge"),
        selected=selected,
        expanded=expanded,
        work_unit_id="WU_ABCD",
        detail=detail,
    )


def test_enum_values_are_stable_public_words() -> None:
    assert [mode.value for mode in SnowballMode] == ["interactive", "exhaustive"]
    assert TerminationReason.FRONTIER_EXHAUSTED.value == "frontier_exhausted"
    assert ResidualReason.QUOTA_DEFERRED.value == "quota_deferred"


def test_residual_entry_is_immutable_canonical_and_round_trips() -> None:
    entry = ResidualEntry(
        target_key=" Semantic_Scholar:Child ",
        aliases=("DOI:10.0/Child", "semantic_scholar:CHILD", "doi:10.0/child"),
        reason=ResidualReason.PENDING,
        level=2,
        direction="forward",
        asserting_source=" Semantic_Scholar ",
        parent_keys=("s2:parent-b", "S2:Parent-A"),
        lanes=("Semantic", "bridge", "semantic"),
        selected=True,
        work_unit_id=" WU_ABCD ",
    )

    assert entry.aliases == ("doi:10.0/child", "semantic_scholar:child")
    assert entry.parent_keys == ("s2:parent-a", "s2:parent-b")
    assert entry.lanes == ("bridge", "semantic")
    assert entry.asserting_source == "semantic_scholar"
    assert entry.work_unit_id == "wu_abcd"
    assert ResidualEntry.from_dict(entry.to_dict()) == entry
    with pytest.raises(FrozenInstanceError):
        entry.level = 3  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"expanded": True}, "must have been selected"),
        ({"selected": True}, "unselected residual"),
        ({"level": 0}, "positive integer"),
        ({"direction": "sideways"}, "backward.*forward"),
        ({"reason": ResidualReason.FAILED}, "actionable detail"),
    ],
)
def test_residual_entry_refuses_inconsistent_states(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _residual(**kwargs)  # type: ignore[arg-type]


def test_failed_residual_requires_and_preserves_actionable_detail() -> None:
    entry = _residual(
        reason=ResidualReason.FAILED,
        selected=True,
        detail=" provider request failed after retry budget ",
    )
    assert entry.detail == "provider request failed after retry budget"


def test_termination_canonicalizes_scope_and_residual_order() -> None:
    later = _residual("semantic_scholar:z", level=2)
    earlier = _residual(
        "openalex:a",
        level=1,
        direction="backward",
        source="openalex",
    )
    termination = TraversalTermination(
        mode=SnowballMode.INTERACTIVE,
        reason=TerminationReason.TARGET_BUDGET,
        operationally_complete=True,
        closure_claimed=False,
        max_depth=2,
        directions=("forward", "backward", "forward"),
        sources=("Semantic_Scholar", "openalex", "semantic_scholar"),
        freshness_policy=" fresh-complete-v1 ",
        residual=(later, earlier),
    )

    assert termination.directions == ("backward", "forward")
    assert termination.sources == ("openalex", "semantic_scholar")
    assert termination.freshness_policy == "fresh-complete-v1"
    assert termination.residual == (earlier, later)
    assert termination.residual_count == 2
    assert termination.bounded
    assert TraversalTermination.from_dict(termination.to_dict()) == termination


def test_closure_requires_a_complete_scoped_fresh_observation() -> None:
    observed = dt.datetime(2026, 8, 31, 5, 30, tzinfo=dt.timezone(dt.timedelta(hours=-5)))
    termination = TraversalTermination(
        mode=SnowballMode.EXHAUSTIVE,
        reason=TerminationReason.DEPTH_LIMIT,
        operationally_complete=True,
        closure_claimed=True,
        max_depth=2,
        directions=("backward", "forward"),
        sources=("semantic_scholar",),
        freshness_policy="fresh-complete-v1",
        observed_at=observed,
    )

    assert termination.closure_claimed
    assert termination.observed_at == dt.datetime(2026, 8, 31, 10, 30, tzinfo=dt.UTC)
    assert TraversalTermination.from_dict(termination.to_dict()) == termination


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"operationally_complete": False}, "operational completion"),
        ({"reason": TerminationReason.REQUEST_BUDGET}, "cannot support closure"),
        ({"residual": (_residual(),)}, "residual work remains"),
        ({"sources": ()}, "source scope"),
        ({"freshness_policy": None}, "freshness policy"),
        ({"observed_at": None}, "observation timestamp"),
    ],
)
def test_closure_refuses_missing_or_contradictory_evidence(
    changes: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "mode": SnowballMode.EXHAUSTIVE,
        "reason": TerminationReason.FRONTIER_EXHAUSTED,
        "operationally_complete": True,
        "closure_claimed": True,
        "sources": ("semantic_scholar",),
        "freshness_policy": "fresh-complete-v1",
        "observed_at": dt.datetime(2026, 8, 31, tzinfo=dt.UTC),
    }
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        TraversalTermination(**values)  # type: ignore[arg-type]


def test_quota_wait_is_exhaustive_and_not_operationally_complete() -> None:
    waiting = TraversalTermination(
        mode=SnowballMode.EXHAUSTIVE,
        reason=TerminationReason.QUOTA_WAIT,
        operationally_complete=False,
        closure_claimed=False,
        sources=("semantic_scholar",),
        residual=(
            _residual(
                reason=ResidualReason.QUOTA_DEFERRED,
                selected=True,
                detail="daily allowance protected",
            ),
        ),
    )
    assert waiting.bounded

    with pytest.raises(ValueError, match="exhaustive-mode"):
        TraversalTermination(
            mode=SnowballMode.INTERACTIVE,
            reason=TerminationReason.QUOTA_WAIT,
            operationally_complete=False,
            closure_claimed=False,
        )
    with pytest.raises(ValueError, match="not operationally complete"):
        TraversalTermination(
            mode=SnowballMode.EXHAUSTIVE,
            reason=TerminationReason.QUOTA_WAIT,
            operationally_complete=True,
            closure_claimed=False,
        )


def test_finished_partial_is_a_terminal_nonclosure_verdict() -> None:
    result = TraversalTermination(
        mode=SnowballMode.EXHAUSTIVE,
        reason=TerminationReason.USER_FINISHED_PARTIAL,
        operationally_complete=True,
        closure_claimed=False,
        residual=(_residual(),),
    )
    assert result.bounded
    assert not result.closure_claimed

    with pytest.raises(ValueError, match="operationally complete"):
        TraversalTermination(
            mode=SnowballMode.EXHAUSTIVE,
            reason=TerminationReason.USER_FINISHED_PARTIAL,
            operationally_complete=False,
            closure_claimed=False,
        )


def test_from_dict_refuses_schema_drift_and_wrong_scalar_types() -> None:
    payload = _residual().to_dict()
    payload["invented"] = True
    with pytest.raises(ValueError, match="unknown field"):
        ResidualEntry.from_dict(payload)

    termination = TraversalTermination(
        mode=SnowballMode.INTERACTIVE,
        reason=TerminationReason.POLICY_SATISFIED,
        operationally_complete=True,
        closure_claimed=False,
    ).to_dict()
    termination["max_depth"] = True
    with pytest.raises(TypeError, match="must be an integer"):
        TraversalTermination.from_dict(termination)


def test_exact_duplicate_residual_is_refused() -> None:
    entry = _residual()
    with pytest.raises(ValueError, match="exact duplicate"):
        TraversalTermination(
            mode=SnowballMode.INTERACTIVE,
            reason=TerminationReason.TARGET_BUDGET,
            operationally_complete=True,
            closure_claimed=False,
            residual=(entry, entry),
        )
