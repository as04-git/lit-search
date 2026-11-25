"""Pure planning, routing, policy, and reduction for citation traversal.

This module is the deterministic seam around the impure snowball acquisition
runtime.  It deliberately knows nothing about connectors, credentials,
request replay, caches, persistence, or enrichment.  The runner adapts the
existing provider machinery to :class:`AcquisitionPort`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

from lit_search.core.citation import (
    CitationEdge,
    CitationTarget,
    IdentifierKind,
    PaperIdentifier,
)
from lit_search.core.identity import record_identity_keys
from lit_search.core.paper import Paper
from lit_search.core.snowball_run import TraversalDirection

SnowballDirection = Literal["backward", "forward", "both"]


def _canonical_json(value: object) -> str:
    """Return a stable JSON representation without importing store code."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_target(target: CitationTarget) -> CitationTarget:
    """Return a target whose identifier order cannot depend on provider order."""
    return CitationTarget(
        identifiers=tuple(sorted(target.identifiers, key=lambda item: item.key)),
        title=target.title,
    )


def _target_key(target: CitationTarget) -> str:
    """Return the canonical alias-group key for one target."""
    return _canonical_json(sorted(target.keys))


def _paper_aliases(paper: Paper) -> frozenset[str]:
    """Return every explicit routable and record-identity alias."""
    return frozenset(
        {identifier.key for identifier in paper.identifiers} | set(record_identity_keys(paper))
    )


def _paper_key(paper: Paper) -> str:
    """Return a deterministic diagnostic key without title identity."""
    aliases = sorted(record_identity_keys(paper))
    if aliases:
        version_aliases = [key for key in aliases if key.startswith(("doi:", "arxiv:"))]
        return version_aliases[0] if version_aliases else aliases[0]
    digest = hashlib.blake2b(
        _canonical_json(paper.to_dict()).encode("utf-8"), digest_size=16
    ).hexdigest()
    return f"unidentified:{digest}"


def version_aliases(aliases: frozenset[str] | Sequence[str]) -> frozenset[str]:
    """Return every version-like alias as non-equivalence relationship evidence."""
    versions: set[str] = set()
    for alias in aliases:
        if alias.startswith("arxiv:"):
            versions.add(alias.lower())
        elif alias.startswith("doi:"):
            doi = alias[4:].strip().lower()
            if doi.startswith("10.48550/arxiv."):
                versions.add(f"arxiv:{doi[len('10.48550/arxiv.') :]}")
            else:
                versions.add(f"doi:{doi}")
    return frozenset(versions)


def primary_version_alias(aliases: frozenset[str] | Sequence[str]) -> str | None:
    """Return the canonical primary version represented by an alias group.

    A non-synthetic DOI is preferred.  If none exists, a synthetic arXiv DOI
    and a native arXiv ID canonicalise to the same primary key.  Additional
    version-like aliases are relationship evidence, not exact identity.
    """
    aliases_set = frozenset(aliases)
    publisher_dois = sorted(
        alias.lower()
        for alias in aliases_set
        if alias.startswith("doi:") and not alias[4:].strip().lower().startswith("10.48550/arxiv.")
    )
    if publisher_dois:
        return publisher_dois[0]
    arxiv_aliases = sorted(
        alias for alias in version_aliases(aliases_set) if alias.startswith("arxiv:")
    )
    return arxiv_aliases[0] if arxiv_aliases else None


def alias_groups_match(
    left: frozenset[str] | Sequence[str],
    right: frozenset[str] | Sequence[str],
) -> bool:
    """Return exact record identity under the canonical primary-version rule."""
    left_set = frozenset(left)
    right_set = frozenset(right)
    left_primary = primary_version_alias(left_set)
    right_primary = primary_version_alias(right_set)
    if left_primary is not None or right_primary is not None:
        return left_primary is not None and left_primary == right_primary
    return bool(left_set.intersection(right_set))


def target_was_visited(
    aliases: frozenset[str],
    visited: frozenset[str] | set[str],
    visited_alias_groups: Sequence[frozenset[str]],
) -> bool:
    """Check visited aliases without letting a provider collision hide a version."""
    overlap = aliases.intersection(visited)
    if not overlap:
        return False
    known_overlap: set[str] = set()
    for group in visited_alias_groups:
        shared = aliases.intersection(group)
        if not shared:
            continue
        known_overlap.update(shared)
        if alias_groups_match(aliases, group):
            return True
    unaccounted = overlap.difference(known_overlap)
    if not unaccounted:
        return False
    target_primary = primary_version_alias(aliases)
    if target_primary is None:
        return True
    # A legacy flat set may establish the version itself, but an ungrouped
    # provider alias cannot wildcard-match a versioned target.
    return primary_version_alias(unaccounted) == target_primary


def claim_was_attempted(aliases: frozenset[str], attempted: Sequence[frozenset[str]]) -> bool:
    """Return whether an exactly identical route claim was attempted before."""
    return any(alias_groups_match(aliases, group) for group in attempted)


@dataclass(frozen=True, slots=True)
class TraversalAssertion:
    """One parent/direction/source assertion of a citation target."""

    target: CitationTarget
    target_key: str
    aliases: tuple[str, ...]
    level: int
    direction: TraversalDirection
    asserting_source: str
    parent_key: str
    parent_aliases: tuple[str, ...]
    lanes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Canonicalise immutable identity-bearing fields."""
        if self.level < 1:
            raise ValueError("TraversalAssertion.level must be at least 1")
        source = self.asserting_source.strip().lower()
        if not source:
            raise ValueError("TraversalAssertion.asserting_source must not be blank")
        if self.direction not in ("backward", "forward"):
            raise ValueError("TraversalAssertion.direction must be backward or forward")
        target = self.target
        aliases = tuple(sorted(set(self.aliases or target.keys)))
        if not aliases:
            raise ValueError("TraversalAssertion.aliases must not be empty")
        if frozenset(aliases) != target.keys:
            raise ValueError("TraversalAssertion.aliases must exactly match target aliases")
        expected_key = _target_key(target)
        if self.target_key != expected_key:
            raise ValueError("TraversalAssertion.target_key must be canonical")
        parent_aliases = tuple(sorted(set(self.parent_aliases)))
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "asserting_source", source)
        object.__setattr__(self, "parent_aliases", parent_aliases)
        object.__setattr__(self, "lanes", tuple(sorted(set(self.lanes))))

    @property
    def route_key(self) -> tuple[str, tuple[str, ...]]:
        """Return the source-qualified reservation identity."""
        return self.asserting_source, self.aliases


@dataclass(frozen=True, slots=True)
class TraversalLevelPlan:
    """Immutable assertions and route reservations for one BFS level."""

    level: int
    assertions: tuple[TraversalAssertion, ...]
    reservations: tuple[tuple[str, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        """Reject reservations not supported by an assertion."""
        if self.level < 1:
            raise ValueError("TraversalLevelPlan.level must be at least 1")
        if any(assertion.level != self.level for assertion in self.assertions):
            raise ValueError("every assertion must belong to the plan level")
        assertion_routes = {assertion.route_key for assertion in self.assertions}
        if len(self.reservations) != len(set(self.reservations)):
            raise ValueError("plan reservations must be unique")
        if set(self.reservations) != assertion_routes:
            raise ValueError("plan reservations must exactly account for assertion routes")


@dataclass(frozen=True, slots=True)
class RouteBatch:
    """One execution route retaining all assertions behind provider work."""

    requested_source: str
    execution_source: str
    assertions: tuple[TraversalAssertion, ...]
    targets: tuple[CitationTarget, ...]

    def __post_init__(self) -> None:
        """Validate source and target accounting."""
        requested = self.requested_source.strip().lower()
        execution = self.execution_source.strip().lower()
        if not requested or not execution:
            raise ValueError("RouteBatch sources must not be blank")
        if any(assertion.asserting_source != requested for assertion in self.assertions):
            raise ValueError("RouteBatch assertions must share requested_source")
        assertion_keys = {assertion.target_key for assertion in self.assertions}
        target_keys = [_target_key(target) for target in self.targets]
        if len(target_keys) != len(set(target_keys)):
            raise ValueError("RouteBatch targets must be deduplicated")
        if not set(target_keys).issubset(assertion_keys):
            raise ValueError("RouteBatch targets must originate from assertions")
        if any(
            not any(
                alias_groups_match(frozenset(assertion.aliases), target.keys)
                for target in self.targets
            )
            for assertion in self.assertions
        ):
            raise ValueError("RouteBatch targets must account for every assertion")
        object.__setattr__(self, "requested_source", requested)
        object.__setattr__(self, "execution_source", execution)


class AcquisitionIntent(StrEnum):
    """How much graph evidence the caller authorises acquisition to fetch."""

    PROJECTION = "projection"
    TRAVERSAL_EDGES = "traversal_edges"


@runtime_checkable
class AcquisitionPort(Protocol):
    """Impure boundary implemented by the current runner adapter."""

    def acquire(
        self,
        batches: tuple[RouteBatch, ...],
        *,
        intent: AcquisitionIntent,
    ) -> Sequence[Paper]:
        """Resolve routed targets under an explicit graph-acquisition intent."""
        ...


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Result membership and next-frontier selection for eligible papers."""

    result_members: tuple[Paper, ...]
    expansion_members: tuple[Paper, ...]
    excluded_from_result: tuple[Paper, ...]
    unexpanded: tuple[Paper, ...]


@dataclass(frozen=True, slots=True)
class LevelReduction:
    """Pure resolution and policy outcome for one acquired level."""

    resolved_assertions: tuple[TraversalAssertion, ...]
    unresolved_assertions: tuple[TraversalAssertion, ...]
    policy: PolicyDecision


@dataclass(frozen=True, slots=True)
class CurrentCitationCapPolicy:
    """Compatibility policy reproducing the runner's current citation caps."""

    max_papers_per_level: int | None
    max_expansion_per_level: int | None

    def __post_init__(self) -> None:
        """Validate the two independent optional bounds."""
        if self.max_papers_per_level is not None and self.max_papers_per_level < 1:
            raise ValueError("max_papers_per_level must be positive")
        if self.max_expansion_per_level is not None and self.max_expansion_per_level < 1:
            raise ValueError("max_expansion_per_level must be positive")

    @staticmethod
    def _select(papers: tuple[Paper, ...], limit: int | None) -> tuple[Paper, ...]:
        """Apply the historic stable citation sort only when a cap exists."""
        if limit is None:
            return papers
        return tuple(sorted(papers, key=lambda paper: paper.citations or 0, reverse=True)[:limit])

    def select(self, papers: Sequence[Paper], *, level: int) -> PolicyDecision:
        """Select result and expansion members independently."""
        if level < 1:
            raise ValueError("level must be at least 1")
        available = tuple(papers)
        result_members = self._select(available, self.max_papers_per_level)
        expansion_members = self._select(available, self.max_expansion_per_level)
        result_ids = {id(paper) for paper in result_members}
        expansion_ids = {id(paper) for paper in expansion_members}
        return PolicyDecision(
            result_members=result_members,
            expansion_members=expansion_members,
            excluded_from_result=tuple(paper for paper in available if id(paper) not in result_ids),
            unexpanded=tuple(paper for paper in available if id(paper) not in expansion_ids),
        )


def _directions(direction: SnowballDirection) -> tuple[TraversalDirection, ...]:
    """Expand the runner's direction option into individual directions."""
    if direction == "backward":
        return ("backward",)
    if direction == "forward":
        return ("forward",)
    if direction == "both":
        return ("backward", "forward")
    raise ValueError("direction must be backward, forward, or both")


def _directional_edges(paper: Paper, direction: TraversalDirection) -> tuple[CitationEdge, ...]:
    """Return typed assertions plus legacy DOI-only compatibility edges."""
    if direction == "backward":
        typed_edges = paper.reference_edges
        legacy_dois = paper.references
    else:
        typed_edges = paper.cited_by_edges
        legacy_dois = paper.cited_by
    edges = list(typed_edges)
    typed_doi_keys = {
        f"doi:{doi.lower()}"
        for edge in typed_edges
        if (doi := edge.target.get(IdentifierKind.DOI)) is not None
    }
    for doi in legacy_dois:
        key = f"doi:{doi.strip().lower()}"
        if key in typed_doi_keys:
            continue
        target = CitationTarget((PaperIdentifier(IdentifierKind.DOI, doi),), title=None)
        edges.append(CitationEdge(target=target, source="legacy_doi"))
    return tuple(edges)


def _assertion_sort_key(assertion: TraversalAssertion) -> tuple[object, ...]:
    """Return a total, provenance-bearing canonical assertion order."""
    return (
        assertion.target_key,
        assertion.asserting_source,
        assertion.direction,
        assertion.parent_key,
        assertion.parent_aliases,
        assertion.lanes,
        assertion.target.title or "",
        _canonical_json(assertion.target.to_dict()),
    )


def plan_frontier(
    frontier: Sequence[Paper],
    *,
    level: int,
    direction: SnowballDirection,
    visited: frozenset[str] | set[str] = frozenset(),
    visited_alias_groups: Sequence[frozenset[str]] = (),
    attempted_routes: Mapping[str, Sequence[frozenset[str]]] | None = None,
    canonical: bool = False,
) -> TraversalLevelPlan:
    """Enumerate distinct parent/direction/source assertions without side effects.

    Prior route reservations suppress only work attempted before this plan.
    Plural assertions discovered together remain in the plan; provider work is
    deduplicated later by :func:`route_assertions`.
    """
    if level < 1:
        raise ValueError("level must be at least 1")
    attempted_routes = attempted_routes or {}
    assertions: list[TraversalAssertion] = []
    reservations: list[tuple[str, tuple[str, ...]]] = []
    seen_reservations: set[tuple[str, tuple[str, ...]]] = set()

    for parent in frontier:
        parent_aliases = tuple(sorted(_paper_aliases(parent)))
        parent_key = _paper_key(parent)
        for traversal_direction in _directions(direction):
            for edge in _directional_edges(parent, traversal_direction):
                target = edge.target
                aliases_set = target.keys
                if target_was_visited(aliases_set, visited, visited_alias_groups):
                    continue
                prior_attempts = attempted_routes.get(edge.source, ())
                if claim_was_attempted(aliases_set, prior_attempts):
                    continue
                aliases = tuple(sorted(aliases_set))
                assertion = TraversalAssertion(
                    target=target,
                    target_key=_target_key(target),
                    aliases=aliases,
                    level=level,
                    direction=traversal_direction,
                    asserting_source=edge.source,
                    parent_key=parent_key,
                    parent_aliases=parent_aliases,
                )
                assertions.append(assertion)
                if assertion.route_key not in seen_reservations:
                    seen_reservations.add(assertion.route_key)
                    reservations.append(assertion.route_key)

    if canonical:
        assertions.sort(key=_assertion_sort_key)
        reservations.sort()
    return TraversalLevelPlan(
        level=level,
        assertions=tuple(assertions),
        reservations=tuple(reservations),
    )


def _execution_source(assertion: TraversalAssertion) -> str:
    """Apply the sole current source-routing exception."""
    doi = assertion.target.get(IdentifierKind.DOI)
    if (
        assertion.asserting_source in {"crossref", "legacy_doi"}
        and doi
        and doi.lower().startswith("10.48550/arxiv.")
    ):
        return "arxiv"
    return assertion.asserting_source


def route_assertions(
    plan: TraversalLevelPlan, *, canonical: bool = False
) -> tuple[RouteBatch, ...]:
    """Group assertions into source routes and deduplicate provider targets."""
    grouped: dict[tuple[str, str], list[TraversalAssertion]] = {}
    for assertion in plan.assertions:
        key = (assertion.asserting_source, _execution_source(assertion))
        grouped.setdefault(key, []).append(assertion)
    group_items = list(grouped.items())
    if canonical:
        group_items.sort(key=lambda item: item[0])

    batches: list[RouteBatch] = []
    for (requested_source, execution_source), assertions in group_items:
        if canonical:
            assertions = sorted(assertions, key=_assertion_sort_key)
        unique_targets: list[CitationTarget] = []
        for assertion in assertions:
            if not any(
                alias_groups_match(assertion.target.keys, target.keys) for target in unique_targets
            ):
                unique_targets.append(assertion.target)
        targets = unique_targets
        if canonical:
            targets.sort(key=_target_key)
        batches.append(
            RouteBatch(
                requested_source=requested_source,
                execution_source=execution_source,
                assertions=tuple(assertions),
                targets=tuple(targets),
            )
        )
    return tuple(batches)


def reduce_level(
    plan: TraversalLevelPlan,
    acquired: Sequence[Paper],
    *,
    eligibility: Callable[[Paper], bool],
    policy: CurrentCitationCapPolicy,
) -> LevelReduction:
    """Resolve assertions, apply eligibility, and make a pure policy decision."""
    acquired_aliases = tuple(_paper_aliases(paper) for paper in acquired)
    resolved: list[TraversalAssertion] = []
    unresolved: list[TraversalAssertion] = []
    for assertion in plan.assertions:
        destination = (
            resolved
            if any(
                alias_groups_match(frozenset(assertion.aliases), aliases)
                for aliases in acquired_aliases
            )
            else unresolved
        )
        destination.append(assertion)
    eligible = tuple(paper for paper in acquired if eligibility(paper))
    return LevelReduction(
        resolved_assertions=tuple(resolved),
        unresolved_assertions=tuple(unresolved),
        policy=policy.select(eligible, level=plan.level),
    )


__all__ = [
    "AcquisitionIntent",
    "AcquisitionPort",
    "CurrentCitationCapPolicy",
    "LevelReduction",
    "PolicyDecision",
    "RouteBatch",
    "TraversalAssertion",
    "TraversalLevelPlan",
    "alias_groups_match",
    "claim_was_attempted",
    "plan_frontier",
    "primary_version_alias",
    "reduce_level",
    "route_assertions",
    "target_was_visited",
    "version_aliases",
]
