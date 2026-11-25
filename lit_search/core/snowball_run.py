"""Pure semantic contract for snowball modes, termination, and residual work.

The operational run database and the immutable result artifact have different
jobs, but they must describe the same traversal verdict.  These types are the
small shared vocabulary between them.  They deliberately perform no storage,
network, runner, or CLI work.

Closure is a stronger claim than operational completion.  A bounded
interactive run can finish successfully with a residual frontier; it cannot
claim closure.  Conversely, a closure claim must name its bounded observation
scope and contain no residual work.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, cast


class SnowballMode(StrEnum):
    """Public traversal modes with intentionally different completion claims."""

    INTERACTIVE = "interactive"
    EXHAUSTIVE = "exhaustive"


class TerminationReason(StrEnum):
    """Why traversal stopped or checkpointed."""

    FRONTIER_EXHAUSTED = "frontier_exhausted"
    DEPTH_LIMIT = "depth_limit"
    TIME_BUDGET = "time_budget"
    REQUEST_BUDGET = "request_budget"
    TARGET_BUDGET = "target_budget"
    POLICY_SATISFIED = "policy_satisfied"
    QUOTA_WAIT = "quota_wait"
    USER_FINISHED_PARTIAL = "user_finished_partial"


class ResidualReason(StrEnum):
    """Why one discovered target remains outside the expanded result."""

    UNSELECTED = "unselected"
    HYDRATION_DEFERRED = "hydration_deferred"
    QUOTA_DEFERRED = "quota_deferred"
    FAILED = "failed"
    TRUNCATED = "truncated"
    STALE = "stale"
    UNROUTABLE = "unroutable"
    PENDING = "pending"


TraversalDirection = Literal["backward", "forward"]
_DIRECTION_ORDER: dict[str, int] = {"backward": 0, "forward": 1}
_DETAIL_REQUIRED = frozenset(
    {
        ResidualReason.FAILED,
        ResidualReason.TRUNCATED,
        ResidualReason.STALE,
        ResidualReason.UNROUTABLE,
    }
)
_NON_CLOSING_REASONS = frozenset(
    {
        TerminationReason.TIME_BUDGET,
        TerminationReason.REQUEST_BUDGET,
        TerminationReason.TARGET_BUDGET,
        TerminationReason.POLICY_SATISFIED,
        TerminationReason.QUOTA_WAIT,
        TerminationReason.USER_FINISHED_PARTIAL,
    }
)


@dataclass(frozen=True, slots=True)
class ResidualEntry:
    """One reconstructable target left outside completed graph expansion.

    One target may legitimately have several entries when distinct sources or
    directions asserted it and those assertions ended differently.  Parent and
    lane collections live on an entry so exact repeated assertions can be
    coalesced without losing plurality.
    """

    target_key: str
    aliases: tuple[str, ...]
    reason: ResidualReason
    level: int
    direction: TraversalDirection
    asserting_source: str
    parent_keys: tuple[str, ...] = ()
    lanes: tuple[str, ...] = ()
    selected: bool = False
    expanded: bool = False
    work_unit_id: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        """Normalize unordered identity evidence and enforce truthful states."""
        target_key = _normalised_required(self.target_key, "target_key")
        aliases = _canonical_strings(self.aliases, "aliases", required=True)
        if isinstance(self.level, bool) or self.level < 1:
            raise ValueError("residual level must be a positive integer")
        direction = self.direction.strip().lower()
        if direction not in _DIRECTION_ORDER:
            raise ValueError("residual direction must be 'backward' or 'forward'")
        source = _normalised_required(self.asserting_source, "asserting_source")
        parent_keys = _canonical_strings(self.parent_keys, "parent_keys")
        lanes = _canonical_strings(self.lanes, "lanes")
        work_unit_id = (
            _normalised_required(self.work_unit_id, "work_unit_id")
            if self.work_unit_id is not None
            else None
        )
        detail = self.detail.strip() if self.detail is not None else None
        if detail == "":
            detail = None
        if self.expanded and not self.selected:
            raise ValueError("an expanded residual target must have been selected")
        if self.reason is ResidualReason.UNSELECTED and (self.selected or self.expanded):
            raise ValueError("an unselected residual target cannot be selected or expanded")
        if self.reason in _DETAIL_REQUIRED and detail is None:
            raise ValueError(f"residual reason {self.reason.value!r} requires actionable detail")

        object.__setattr__(self, "target_key", target_key)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "direction", cast(TraversalDirection, direction))
        object.__setattr__(self, "asserting_source", source)
        object.__setattr__(self, "parent_keys", parent_keys)
        object.__setattr__(self, "lanes", lanes)
        object.__setattr__(self, "work_unit_id", work_unit_id)
        object.__setattr__(self, "detail", detail)

    @property
    def sort_key(self) -> tuple[Any, ...]:
        """Return the canonical semantic order used by termination records."""
        return (
            self.level,
            _DIRECTION_ORDER[self.direction],
            self.asserting_source,
            self.target_key,
            self.reason.value,
            self.aliases,
            self.parent_keys,
            self.lanes,
            self.selected,
            self.expanded,
            self.work_unit_id or "",
            self.detail or "",
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""
        return {
            "target_key": self.target_key,
            "aliases": list(self.aliases),
            "reason": self.reason.value,
            "level": self.level,
            "direction": self.direction,
            "asserting_source": self.asserting_source,
            "parent_keys": list(self.parent_keys),
            "lanes": list(self.lanes),
            "selected": self.selected,
            "expanded": self.expanded,
            "work_unit_id": self.work_unit_id,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResidualEntry:
        """Reconstruct an entry while refusing an unknown contract shape."""
        data = _strict_mapping(payload, _RESIDUAL_KEYS, "residual")
        return cls(
            target_key=_required_str(data, "target_key", "residual"),
            aliases=_string_tuple(data, "aliases", "residual"),
            reason=ResidualReason(_required_str(data, "reason", "residual")),
            level=_required_int(data, "level", "residual"),
            direction=cast(TraversalDirection, _required_str(data, "direction", "residual")),
            asserting_source=_required_str(data, "asserting_source", "residual"),
            parent_keys=_string_tuple(data, "parent_keys", "residual"),
            lanes=_string_tuple(data, "lanes", "residual"),
            selected=_required_bool(data, "selected", "residual"),
            expanded=_required_bool(data, "expanded", "residual"),
            work_unit_id=_optional_str(data, "work_unit_id", "residual"),
            detail=_optional_str(data, "detail", "residual"),
        )


@dataclass(frozen=True, slots=True)
class TraversalTermination:
    """A traversal verdict, its declared scope, and its complete residual.

    ``operationally_complete`` says whether the declared operation stopped
    normally.  ``closure_claimed`` additionally says that every required unit
    inside the recorded source/depth/direction/freshness observation scope was
    resolved.  The latter is intentionally impossible for bounded, waiting,
    explicitly partial, or residual-bearing outcomes.
    """

    mode: SnowballMode
    reason: TerminationReason
    operationally_complete: bool
    closure_claimed: bool
    max_depth: int = 1
    directions: tuple[TraversalDirection, ...] = ("backward", "forward")
    sources: tuple[str, ...] = ()
    freshness_policy: str | None = None
    observed_at: dt.datetime | None = None
    residual: tuple[ResidualEntry, ...] = ()

    def __post_init__(self) -> None:
        """Canonicalize scope and refuse internally inconsistent verdicts."""
        if isinstance(self.max_depth, bool) or self.max_depth < 1:
            raise ValueError("termination max_depth must be a positive integer")
        directions = _canonical_directions(self.directions)
        sources = _canonical_strings(self.sources, "sources")
        freshness = self.freshness_policy.strip() if self.freshness_policy is not None else None
        if freshness == "":
            freshness = None
        observed_at = _as_utc(self.observed_at) if self.observed_at is not None else None
        residual = tuple(sorted(self.residual, key=lambda item: item.sort_key))
        if any(not isinstance(item, ResidualEntry) for item in residual):
            raise TypeError("termination residual must contain only ResidualEntry values")
        if len(set(residual)) != len(residual):
            raise ValueError("termination residual contains an exact duplicate entry")

        if self.closure_claimed and not self.operationally_complete:
            raise ValueError("closure requires operational completion")
        if self.closure_claimed and self.reason in _NON_CLOSING_REASONS:
            raise ValueError(f"termination reason {self.reason.value!r} cannot support closure")
        if self.closure_claimed and residual:
            raise ValueError("closure cannot be claimed while residual work remains")
        if self.closure_claimed and not sources:
            raise ValueError("closure requires a non-empty declared source scope")
        if self.closure_claimed and freshness is None:
            raise ValueError("closure requires an explicit freshness policy")
        if self.closure_claimed and observed_at is None:
            raise ValueError("closure requires an observation timestamp")
        if self.reason is TerminationReason.QUOTA_WAIT:
            if self.mode is not SnowballMode.EXHAUSTIVE:
                raise ValueError("quota_wait is an exhaustive-mode lifecycle outcome")
            if self.operationally_complete:
                raise ValueError("a quota-waiting run is not operationally complete")
        if (
            self.reason is TerminationReason.USER_FINISHED_PARTIAL
            and not self.operationally_complete
        ):
            raise ValueError("an explicitly finished-partial run is operationally complete")

        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "freshness_policy", freshness)
        object.__setattr__(self, "observed_at", observed_at)
        object.__setattr__(self, "residual", residual)

    @property
    def residual_count(self) -> int:
        """Return the number of reconstructable residual assertions."""
        return len(self.residual)

    @property
    def bounded(self) -> bool:
        """Whether policy or resource bounds, rather than closure, stopped work."""
        return self.reason in _NON_CLOSING_REASONS

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""
        return {
            "mode": self.mode.value,
            "reason": self.reason.value,
            "operationally_complete": self.operationally_complete,
            "closure_claimed": self.closure_claimed,
            "max_depth": self.max_depth,
            "directions": list(self.directions),
            "sources": list(self.sources),
            "freshness_policy": self.freshness_policy,
            "observed_at": self.observed_at.isoformat() if self.observed_at is not None else None,
            "residual": [item.to_dict() for item in self.residual],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TraversalTermination:
        """Reconstruct a verdict while refusing an unknown contract shape."""
        data = _strict_mapping(payload, _TERMINATION_KEYS, "termination")
        observed_raw = _optional_str(data, "observed_at", "termination")
        try:
            observed_at = dt.datetime.fromisoformat(observed_raw) if observed_raw else None
        except ValueError as error:
            raise ValueError("termination.observed_at must be an ISO-8601 datetime") from error
        raw_residual = data.get("residual")
        if not isinstance(raw_residual, list):
            raise TypeError("termination.residual must be a list")
        return cls(
            mode=SnowballMode(_required_str(data, "mode", "termination")),
            reason=TerminationReason(_required_str(data, "reason", "termination")),
            operationally_complete=_required_bool(data, "operationally_complete", "termination"),
            closure_claimed=_required_bool(data, "closure_claimed", "termination"),
            max_depth=_required_int(data, "max_depth", "termination"),
            directions=tuple(
                cast(TraversalDirection, item)
                for item in _string_tuple(data, "directions", "termination")
            ),
            sources=_string_tuple(data, "sources", "termination"),
            freshness_policy=_optional_str(data, "freshness_policy", "termination"),
            observed_at=observed_at,
            residual=tuple(
                ResidualEntry.from_dict(_as_mapping(item, f"termination.residual[{index}]"))
                for index, item in enumerate(raw_residual)
            ),
        )


_RESIDUAL_KEYS = frozenset(
    {
        "target_key",
        "aliases",
        "reason",
        "level",
        "direction",
        "asserting_source",
        "parent_keys",
        "lanes",
        "selected",
        "expanded",
        "work_unit_id",
        "detail",
    }
)
_TERMINATION_KEYS = frozenset(
    {
        "mode",
        "reason",
        "operationally_complete",
        "closure_claimed",
        "max_depth",
        "directions",
        "sources",
        "freshness_policy",
        "observed_at",
        "residual",
    }
)


def _normalised_required(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalised = value.strip().lower()
    if not normalised:
        raise ValueError(f"{name} must not be blank")
    return normalised


def _canonical_strings(
    values: tuple[str, ...], name: str, *, required: bool = False
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    canonical = tuple(sorted({_normalised_required(value, name) for value in values}))
    if required and not canonical:
        raise ValueError(f"{name} must not be empty")
    return canonical


def _canonical_directions(
    values: tuple[TraversalDirection, ...],
) -> tuple[TraversalDirection, ...]:
    if not isinstance(values, tuple):
        raise TypeError("directions must be a tuple")
    normalized: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError("directions must contain strings")
        direction = value.strip().lower()
        if direction not in _DIRECTION_ORDER:
            raise ValueError("directions may contain only 'backward' and 'forward'")
        normalized.add(direction)
    if not normalized:
        raise ValueError("directions must not be empty")
    return tuple(
        cast(TraversalDirection, item)
        for item in sorted(normalized, key=lambda item: _DIRECTION_ORDER[item])
    )


def _as_utc(value: dt.datetime) -> dt.datetime:
    if not isinstance(value, dt.datetime):
        raise TypeError("observed_at must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("observed_at must be timezone-aware")
    return value.astimezone(dt.UTC)


def _as_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{context} keys must be strings")
    return cast(Mapping[str, Any], value)


def _strict_mapping(
    payload: Mapping[str, Any], expected: frozenset[str], context: str
) -> Mapping[str, Any]:
    data = _as_mapping(payload, context)
    actual = set(data)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise ValueError(f"{context}: missing required field(s): {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{context}: unknown field(s): {', '.join(unknown)}")
    return data


def _required_str(data: Mapping[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{context}.{key} must be a string")
    return value


def _optional_str(data: Mapping[str, Any], key: str, context: str) -> str | None:
    value = data.get(key)
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{context}.{key} must be a string or null")
    return value


def _required_bool(data: Mapping[str, Any], key: str, context: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise TypeError(f"{context}.{key} must be a boolean")
    return value


def _required_int(data: Mapping[str, Any], key: str, context: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context}.{key} must be an integer")
    return value


def _string_tuple(data: Mapping[str, Any], key: str, context: str) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{context}.{key} must be a list of strings")
    return tuple(value)


__all__ = [
    "ResidualEntry",
    "ResidualReason",
    "SnowballMode",
    "TerminationReason",
    "TraversalDirection",
    "TraversalTermination",
]
