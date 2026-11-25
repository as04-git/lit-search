from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from lit_search.core.coverage import CoverageStatus, ExecutionStatus
from lit_search.evaluation.benchmark import (
    AcquisitionMechanism,
    AcquisitionStatus,
    BenchmarkCase,
    BenchmarkManifestError,
    BenchmarkSuite,
    BenchmarkTrack,
    CaseRun,
    CitationReachability,
    CriticalTarget,
    DiscoveryEvidence,
    ExecutionIssue,
    ExecutionIssueKind,
    ExecutionOutcome,
    IssueState,
    ProviderObservation,
    ProviderPresence,
    RankingOutcome,
    RankingStatus,
    RunManifest,
    Stratum,
    TargetIdentity,
    TargetOutcome,
)
from lit_search.evaluation.scoring import score_run

_D0 = "sha256:" + "0" * 64
_D1 = "sha256:" + "1" * 64


def _case(index: int, primary: Stratum, *secondary: Stratum) -> BenchmarkCase:
    disconnected = Stratum.CITATION_DISCONNECTED in {primary, *secondary}
    return BenchmarkCase(
        case_id=f"synthetic-{index}",
        primary_stratum=primary,
        secondary_strata=secondary,
        research_question=f"Synthetic question {index}",
        initial_formulation=f"ordinary user vocabulary {index}",
        initial_query_family_digest=_D0,
        initial_query_visible_terms=(f"ordinary-term-{index}",),
        prohibited_answer_terms=(f"hidden-answer-{index}",),
        critical_targets=(
            CriticalTarget(
                target_id=f"target-{index}",
                acceptable_identities=(TargetIdentity("doi", f"10.0000/synthetic.{index}"),),
                relevance_evidence_digest=_D1,
            ),
        ),
        freeze_timestamp="2026-08-31T00:00:00Z",
        source_snapshot_digests=(_D0,),
        citation_reachability=(
            CitationReachability.NO_USABLE_PATH
            if disconnected
            else CitationReachability.PATH_EXISTS
        ),
        citation_evidence_digest=_D1,
        provider_observations=(
            ProviderObservation(
                source="synthetic-provider",
                presence=(
                    ProviderPresence.ABSENT
                    if primary is Stratum.AGGREGATOR_ASYMMETRY
                    else ProviderPresence.PRESENT
                ),
                coverage_status=CoverageStatus.COMPLETE,
                execution_status=ExecutionStatus.COMPLETE,
                evidence_digest=_D0,
            ),
        ),
        relevance_protocol_digest=_D1,
        max_wall_seconds=600,
        max_provider_requests=100,
    )


def _suite() -> BenchmarkSuite:
    suite = BenchmarkSuite(
        suite_id="fully-synthetic-contract-example",
        cases=(
            _case(1, Stratum.CITATION_DISCONNECTED),
            _case(2, Stratum.RECENT_UNCITED, Stratum.CITATION_DISCONNECTED),
            _case(3, Stratum.TERMINOLOGY_COMMUNITY_SHIFT),
            _case(4, Stratum.AGGREGATOR_ASYMMETRY),
            _case(5, Stratum.ACTOR_VENUE_CLASSIFICATION),
            _case(6, Stratum.IDENTIFIER_VERSION_STRESS),
            _case(7, Stratum.AMBIGUOUS_LANGUAGE),
            _case(8, Stratum.QUESTION_MODEL_REVISION),
        ),
        freeze_digest=_D0,
    ).with_computed_freeze_digest()
    suite.validate()
    return suite


def _target_outcome(
    target_id: str,
    status: AcquisitionStatus,
    *,
    mechanism: AcquisitionMechanism = AcquisitionMechanism.ORIGINAL_QUERY,
) -> TargetOutcome:
    evidence = (
        (
            DiscoveryEvidence(
                mechanism=mechanism,
                operation_id="synthetic-operation",
                evidence_digest=_D0,
            ),
        )
        if status is AcquisitionStatus.ACQUIRED
        else ()
    )
    ranking = (
        RankingOutcome(RankingStatus.RANKED, 1)
        if status is AcquisitionStatus.ACQUIRED
        else RankingOutcome(
            RankingStatus.NOT_RUN
            if status is AcquisitionStatus.NOT_ACQUIRED
            else RankingStatus.UNKNOWN,
            None,
        )
    )
    return TargetOutcome(
        target_id=target_id,
        acquisition_status=status,
        observed_identity=(
            TargetIdentity("doi", f"10.0000/synthetic.{target_id.removeprefix('target-')}")
            if status is AcquisitionStatus.ACQUIRED
            else None
        ),
        discovery_evidence=evidence,
        ranking=ranking,
    )


def _run(
    suite: BenchmarkSuite,
    statuses: tuple[AcquisitionStatus, ...] | None = None,
) -> RunManifest:
    actual = statuses or (AcquisitionStatus.ACQUIRED,) * len(suite.cases)
    return RunManifest(
        run_id="synthetic-run",
        suite_id=suite.suite_id,
        suite_freeze_digest=suite.freeze_digest,
        track=BenchmarkTrack.ORIGINAL_QUERY,
        ranking_cutoff=10,
        candidate_union_digest=_D1,
        baseline_candidate_union_digest=None,
        cases=tuple(
            CaseRun(
                case_id=case.case_id,
                execution=ExecutionOutcome.COMPLETE,
                execution_issues=(),
                targets=(_target_outcome(case.critical_targets[0].target_id, status),),
            )
            for case, status in zip(suite.cases, actual, strict=True)
        ),
    )


def test_valid_suite_round_trips_with_verified_freeze() -> None:
    suite = _suite()

    restored = BenchmarkSuite.from_dict(suite.to_dict())

    assert restored == suite
    assert restored.freeze_digest == restored.expected_freeze_digest()


def test_suite_rejects_definition_change_after_freeze() -> None:
    suite = _suite()
    changed_case = replace(suite.cases[0], research_question="Changed after freeze")
    changed = replace(suite, cases=(changed_case, *suite.cases[1:]))

    with pytest.raises(BenchmarkManifestError, match="definitions changed after freeze"):
        changed.validate()


def test_suite_enforces_size_required_strata_and_two_disconnected_cases() -> None:
    suite = _suite()

    with pytest.raises(BenchmarkManifestError, match="between 8 and 12"):
        replace(suite, cases=suite.cases[:-1]).with_computed_freeze_digest().validate()

    missing_question_revision = replace(
        suite,
        cases=(*suite.cases[:-1], _case(9, Stratum.RECENT_UNCITED)),
    ).with_computed_freeze_digest()
    with pytest.raises(BenchmarkManifestError, match="question_model_revision"):
        missing_question_revision.validate()

    only_one_disconnected = replace(
        suite,
        cases=(
            suite.cases[0],
            _case(2, Stratum.RECENT_UNCITED),
            *suite.cases[2:],
        ),
    ).with_computed_freeze_digest()
    with pytest.raises(BenchmarkManifestError, match="at least two citation-disconnected"):
        only_one_disconnected.validate()


def test_disconnected_tag_requires_scoped_no_path_evidence() -> None:
    with pytest.raises(BenchmarkManifestError, match="requires scoped no_usable_path"):
        replace(
            _case(1, Stratum.CITATION_DISCONNECTED),
            citation_reachability=CitationReachability.UNKNOWN,
        )


def test_initial_formulation_cannot_leak_answer_terms() -> None:
    case = _case(1, Stratum.CITATION_DISCONNECTED)
    with pytest.raises(BenchmarkManifestError, match="run-visible initial input leaks"):
        replace(case, initial_formulation="Please find HIDDEN-ANSWER-1")


def test_research_question_cannot_leak_answer_terms() -> None:
    case = _case(1, Stratum.CITATION_DISCONNECTED)
    with pytest.raises(BenchmarkManifestError, match="run-visible initial input leaks"):
        replace(case, research_question="Please find HIDDEN-ANSWER-1")


def test_initial_query_family_terms_cannot_leak_answer_terms() -> None:
    case = _case(1, Stratum.CITATION_DISCONNECTED)
    with pytest.raises(BenchmarkManifestError, match="run-visible initial input leaks"):
        replace(case, initial_query_visible_terms=("HIDDEN-ANSWER-1",))


def test_critical_target_requires_identifier_and_evidence_digest() -> None:
    with pytest.raises(BenchmarkManifestError, match="at least one identity"):
        CriticalTarget("target", (), _D0)
    with pytest.raises(BenchmarkManifestError, match="expected sha256"):
        CriticalTarget("target", (TargetIdentity("doi", "10.0/x"),), "not-a-digest")


def test_same_identity_cannot_be_weighted_as_two_targets_in_one_case() -> None:
    case = _case(1, Stratum.CITATION_DISCONNECTED)
    duplicate = replace(case.critical_targets[0], target_id="target-duplicate")

    with pytest.raises(BenchmarkManifestError, match="cannot count as multiple"):
        replace(case, critical_targets=(*case.critical_targets, duplicate))


def test_provider_absence_requires_complete_observation() -> None:
    with pytest.raises(BenchmarkManifestError, match="absence requires complete"):
        ProviderObservation(
            source="synthetic-provider",
            presence=ProviderPresence.ABSENT,
            coverage_status=CoverageStatus.FAILED,
            execution_status=ExecutionStatus.INCOMPLETE,
            evidence_digest=_D0,
        )


def test_aggregator_asymmetry_requires_typed_absence_or_misrepresentation() -> None:
    case = _case(4, Stratum.AGGREGATOR_ASYMMETRY)
    present = replace(case.provider_observations[0], presence=ProviderPresence.PRESENT)

    with pytest.raises(BenchmarkManifestError, match="absent or misrepresented"):
        replace(case, provider_observations=(present,))


def test_unknown_recall_is_neither_zero_nor_success() -> None:
    suite = _suite()
    statuses = (
        AcquisitionStatus.ACQUIRED,
        AcquisitionStatus.NOT_ACQUIRED,
        AcquisitionStatus.UNKNOWN,
        *(AcquisitionStatus.ACQUIRED for _ in range(5)),
    )
    score = score_run(suite, _run(suite, statuses))

    assert score.critical_paper_recall.acquired == 6
    assert score.critical_paper_recall.not_acquired == 1
    assert score.critical_paper_recall.unknown == 1
    assert score.critical_paper_recall.lower_bound == 0.75
    assert score.critical_paper_recall.upper_bound == 0.875
    assert score.critical_paper_recall.point is None
    disconnected = score.recall_by_stratum[Stratum.CITATION_DISCONNECTED]
    assert disconnected.total == 2
    assert disconnected.point == 0.5


def test_execution_unknowns_remain_typed_and_separate() -> None:
    suite = _suite()
    run = _run(suite)
    issue = ExecutionIssue(
        kind=ExecutionIssueKind.GRAPH_WORK_UNIT,
        state=IssueState.UNKNOWN,
        unit_id="openalex:cursor:2",
        evidence_digest=_D0,
    )
    changed_case = replace(
        run.cases[0], execution=ExecutionOutcome.UNKNOWN, execution_issues=(issue,)
    )
    changed = replace(run, cases=(changed_case, *run.cases[1:]))

    score = score_run(suite, changed)

    assert score.execution.complete_cases == 7
    assert score.execution.unknown_cases == 1
    assert score.execution.incomplete_cases == 0
    assert score.execution.unknowns_by_kind == {ExecutionIssueKind.GRAPH_WORK_UNIT: 1}


def test_execution_summary_must_agree_with_typed_issues() -> None:
    with pytest.raises(BenchmarkManifestError, match="contradicts typed issues"):
        CaseRun(
            case_id="case",
            execution=ExecutionOutcome.COMPLETE,
            execution_issues=(
                ExecutionIssue(
                    ExecutionIssueKind.SOURCE,
                    IssueState.INCOMPLETE,
                    "source:x",
                    _D0,
                ),
            ),
            targets=(),
        )


def test_ranking_cannot_manufacture_acquisition() -> None:
    with pytest.raises(BenchmarkManifestError, match="known absent target"):
        TargetOutcome(
            target_id="target",
            acquisition_status=AcquisitionStatus.NOT_ACQUIRED,
            observed_identity=None,
            discovery_evidence=(),
            ranking=RankingOutcome(RankingStatus.RANKED, 1),
        )


def test_acquired_target_must_name_an_acceptable_frozen_identity() -> None:
    suite = _suite()
    run = _run(suite)
    target = replace(
        run.cases[0].targets[0],
        observed_identity=TargetIdentity("doi", "10.0000/wrong-version"),
    )
    changed_case = replace(run.cases[0], targets=(target,))

    with pytest.raises(BenchmarkManifestError, match="acceptable frozen version identity"):
        score_run(suite, replace(run, cases=(changed_case, *run.cases[1:])))


def test_semantic_ranking_track_must_preserve_candidate_membership() -> None:
    suite = _suite()
    run = replace(
        _run(suite),
        track=BenchmarkTrack.SEMANTIC_RERANKING_ONLY,
        baseline_candidate_union_digest=_D0,
        candidate_union_digest=_D1,
    )

    with pytest.raises(BenchmarkManifestError, match="cannot claim recall"):
        score_run(suite, run)


def test_track_rejects_forbidden_discovery_mechanism() -> None:
    suite = _suite()
    run = _run(suite)
    target = _target_outcome(
        run.cases[0].targets[0].target_id,
        AcquisitionStatus.ACQUIRED,
        mechanism=AcquisitionMechanism.CITATION_TRAVERSAL,
    )
    changed_case = replace(run.cases[0], targets=(target,))

    with pytest.raises(BenchmarkManifestError, match="forbidden in original_query"):
        score_run(suite, replace(run, cases=(changed_case, *run.cases[1:])))


def test_run_refuses_different_suite_freeze() -> None:
    suite = _suite()

    with pytest.raises(BenchmarkManifestError, match="different suite freeze digest"):
        score_run(suite, replace(_run(suite), suite_freeze_digest=_D0))


def test_run_cannot_omit_cases_or_frozen_targets() -> None:
    suite = _suite()
    run = _run(suite)

    with pytest.raises(BenchmarkManifestError, match="run cases do not match suite"):
        score_run(suite, replace(run, cases=run.cases[:-1]))

    without_target = replace(run.cases[0], targets=())
    with pytest.raises(BenchmarkManifestError, match="target outcomes do not match"):
        score_run(suite, replace(run, cases=(without_target, *run.cases[1:])))


def test_run_manifest_round_trip_preserves_observed_identity() -> None:
    suite = _suite()
    run = _run(suite)

    restored = RunManifest.from_dict(run.to_dict())
    restored.validate_against(suite)

    assert restored == run
    assert restored.cases[0].targets[0].observed_identity == TargetIdentity(
        "doi", "10.0000/synthetic.1"
    )


def test_strict_parser_rejects_unknown_fields() -> None:
    suite_data = _suite().to_dict()
    suite_data["quietly_ignored"] = True

    with pytest.raises(BenchmarkManifestError, match="unknown key"):
        BenchmarkSuite.from_dict(suite_data)


def test_checked_in_case_template_is_strictly_parseable_and_obviously_synthetic() -> None:
    path = Path("docs/benchmarks/synthetic-case.template.json")
    data = json.loads(path.read_text(encoding="utf-8"))

    case = BenchmarkCase.from_dict(data, "synthetic template")

    assert case.case_id.startswith("synthetic-")
    assert case.critical_targets[0].target_id.startswith("synthetic-")
    assert case.critical_targets[0].acceptable_identities[0].value == "10.0000/not-a-real-doi"
