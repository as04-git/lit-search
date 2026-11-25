"""Honest scoring for a frozen adversarial benchmark run."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .benchmark import (
    AcquisitionStatus,
    BenchmarkSuite,
    ExecutionIssueKind,
    ExecutionOutcome,
    IssueState,
    RankingStatus,
    RunManifest,
    Stratum,
)


@dataclass(frozen=True, slots=True)
class RecallScore:
    """Tri-valued recall counts and bounds.

    ``point`` is intentionally absent whenever one or more targets are
    unknown.  ``lower_bound`` treats unknowns as missed and ``upper_bound``
    treats them as acquired, but neither bound is mislabeled as the observed
    recall.
    """

    total: int
    acquired: int
    not_acquired: int
    unknown: int
    lower_bound: float
    upper_bound: float
    point: float | None


@dataclass(frozen=True, slots=True)
class SurfacingScore:
    """Separate ranking counts at the run's frozen cutoff."""

    total: int
    ranked: int
    not_in_top_k: int
    not_run: int
    unknown: int


@dataclass(frozen=True, slots=True)
class ExecutionScore:
    """Typed execution outcomes without collapsing unknown into failure."""

    total_cases: int
    complete_cases: int
    incomplete_cases: int
    unknown_cases: int
    known_shortfalls_by_kind: dict[ExecutionIssueKind, int]
    unknowns_by_kind: dict[ExecutionIssueKind, int]


@dataclass(frozen=True, slots=True)
class BenchmarkScore:
    """Score report for one run against one exact suite freeze."""

    suite_id: str
    suite_freeze_digest: str
    run_id: str
    critical_paper_recall: RecallScore
    recall_by_stratum: dict[Stratum, RecallScore]
    surfacing: SurfacingScore
    execution: ExecutionScore


def _recall(statuses: list[AcquisitionStatus]) -> RecallScore:
    """Convert tri-valued target states to counts and honest bounds."""

    counts = Counter(statuses)
    total = len(statuses)
    acquired = counts[AcquisitionStatus.ACQUIRED]
    not_acquired = counts[AcquisitionStatus.NOT_ACQUIRED]
    unknown = counts[AcquisitionStatus.UNKNOWN]
    lower = acquired / total if total else 0.0
    upper = (acquired + unknown) / total if total else 0.0
    return RecallScore(
        total=total,
        acquired=acquired,
        not_acquired=not_acquired,
        unknown=unknown,
        lower_bound=lower,
        upper_bound=upper,
        point=None if unknown else lower,
    )


def score_run(suite: BenchmarkSuite, run: RunManifest) -> BenchmarkScore:
    """Validate and score *run* against the exact frozen *suite*.

    Raises
    ------
    BenchmarkManifestError
        If the suite changed after freezing, the run names another freeze, or
        the run violates track, target, execution, or ranking invariants.
    """

    run.validate_against(suite)
    case_runs = {case.case_id: case for case in run.cases}
    all_statuses: list[AcquisitionStatus] = []
    by_stratum: dict[Stratum, list[AcquisitionStatus]] = {stratum: [] for stratum in Stratum}
    ranking_counts: Counter[RankingStatus] = Counter()
    execution_counts: Counter[ExecutionOutcome] = Counter()
    incomplete_kinds: Counter[ExecutionIssueKind] = Counter()
    unknown_kinds: Counter[ExecutionIssueKind] = Counter()

    for case in suite.cases:
        case_run = case_runs[case.case_id]
        execution_counts[case_run.execution] += 1
        statuses = [target.acquisition_status for target in case_run.targets]
        all_statuses.extend(statuses)
        for stratum in case.strata:
            by_stratum[stratum].extend(statuses)
        for target in case_run.targets:
            ranking_counts[target.ranking.status] += 1
        for issue in case_run.execution_issues:
            destination = unknown_kinds if issue.state is IssueState.UNKNOWN else incomplete_kinds
            destination[issue.kind] += 1

    return BenchmarkScore(
        suite_id=suite.suite_id,
        suite_freeze_digest=suite.freeze_digest,
        run_id=run.run_id,
        critical_paper_recall=_recall(all_statuses),
        recall_by_stratum={
            stratum: _recall(statuses) for stratum, statuses in by_stratum.items() if statuses
        },
        surfacing=SurfacingScore(
            total=sum(ranking_counts.values()),
            ranked=ranking_counts[RankingStatus.RANKED],
            not_in_top_k=ranking_counts[RankingStatus.NOT_IN_TOP_K],
            not_run=ranking_counts[RankingStatus.NOT_RUN],
            unknown=ranking_counts[RankingStatus.UNKNOWN],
        ),
        execution=ExecutionScore(
            total_cases=len(run.cases),
            complete_cases=execution_counts[ExecutionOutcome.COMPLETE],
            incomplete_cases=execution_counts[ExecutionOutcome.INCOMPLETE],
            unknown_cases=execution_counts[ExecutionOutcome.UNKNOWN],
            known_shortfalls_by_kind=dict(incomplete_kinds),
            unknowns_by_kind=dict(unknown_kinds),
        ),
    )


__all__ = [
    "BenchmarkScore",
    "ExecutionScore",
    "RecallScore",
    "SurfacingScore",
    "score_run",
]
