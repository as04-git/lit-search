"""I2a operation-level replay tests for ``SnowballRunner``."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, ClassVar

import pytest

from lit_search.connectors.openalex import OpenAlexCitedByPageResult
from lit_search.core.citation import (
    CitationEdge,
    CitationTarget,
    IdentifierKind,
    NativeBatchResult,
    PaperIdentifier,
)
from lit_search.core.paper import Paper
from lit_search.runners.snowball_runner import SnowballRunner
from lit_search.store.run_state import (
    RunLifecycle,
    RunStateStore,
    SemanticSpecMismatchError,
    WorkUnitState,
    canonical_json,
)

RUN_ID = "r_11111111111111111111111111111111"
SECOND_RUN_ID = "r_22222222222222222222222222222222"


def _paper(
    title: str,
    s2_id: str,
    *,
    references: tuple[CitationTarget, ...] = (),
    doi: str | None = None,
) -> Paper:
    return Paper(
        title=title,
        abstract="frozen abstract",
        authors=[],
        source=None,
        publication_date=dt.date(2026, 1, 1),
        doi=doi,
        identifiers=[PaperIdentifier(IdentifierKind.SEMANTIC_SCHOLAR, s2_id)],
        reference_edges=[
            CitationEdge(target=target, source="semantic_scholar") for target in references
        ],
        citation_edge_totals={"semantic_scholar": {"references": len(references), "citations": 0}},
    )


def _seed_input(*, with_embedded_edge: bool = False) -> Paper:
    target = CitationTarget.from_values(semantic_scholar_id="S2")
    references = (target,) if with_embedded_edge and target is not None else ()
    return _paper("Seed", "S1", references=references)


def _runner(seed: Paper, *, run_state: RunStateStore | None = None) -> SnowballRunner:
    return SnowballRunner(
        seed,
        max_depth=1,
        databases=["semantic_scholar"],
        enrichment_databases=[],
        run_state=run_state,
    )


def _create_store(root: Path, run_id: str = RUN_ID) -> RunStateStore:
    probe = _runner(_seed_input())
    return RunStateStore.create(
        root,
        probe.semantic_spec,
        credentials=probe.credential_identities,
        run_id=run_id,
    )


def _result_signature(result: Any) -> tuple[tuple[str, ...], str]:
    members = tuple(sorted(canonical_json(paper.to_dict()) for paper in result.all_papers))
    coverage = canonical_json(result.coverage.to_dict() if result.coverage is not None else None)
    return members, coverage


class _FrozenSemanticConnector:
    calls: ClassVar[list[tuple[str, ...]]] = []
    interrupt_s2_once: ClassVar[bool] = False
    interrupted: ClassVar[bool] = False

    def __init__(self, **_kwargs: object) -> None:
        pass

    def fetch_papers_batch(self, identifiers: list[str], **_kwargs: object) -> NativeBatchResult:
        request = tuple(identifiers)
        self.calls.append(request)
        if request == ("S2",) and self.interrupt_s2_once and not self.interrupted:
            type(self).interrupted = True
            raise KeyboardInterrupt("frozen interruption")
        if request == ("S1",):
            target = CitationTarget.from_values(semantic_scholar_id="S2")
            assert target is not None
            papers = (_paper("Seed", "S1", references=(target,)),)
        elif request == ("S2",):
            papers = (_paper("Child", "S2"),)
        else:  # pragma: no cover - makes an unexpected route loud
            raise AssertionError(f"unexpected frozen request {request!r}")
        return NativeBatchResult(
            source="semantic_scholar",
            papers=papers,
            requested=len(identifiers),
        )

    def close(self) -> None:
        pass


class _FakeGetRunner:
    calls: ClassVar[list[str]] = []

    def __init__(self, *, identifier: str, **_kwargs: object) -> None:
        self.identifier = identifier

    def run(self, **_kwargs: object) -> Paper:
        self.calls.append(self.identifier)
        return _paper("DOI result", "S-DOI", doi=self.identifier)


class _ChunkedSemanticConnector:
    calls: ClassVar[list[tuple[str, ...]]] = []
    interrupt_final_chunk_once: ClassVar[bool] = False
    interrupted: ClassVar[bool] = False

    def __init__(self, **_kwargs: object) -> None:
        pass

    @staticmethod
    def max_paper_batch_size(*, include_edges: bool = True) -> int:
        return 250 if include_edges else 500

    def fetch_papers_batch_request(
        self, identifiers: list[str], **_kwargs: object
    ) -> NativeBatchResult:
        request = tuple(identifiers)
        self.calls.append(request)
        if (
            request != ("S1",)
            and len(request) == 1
            and self.interrupt_final_chunk_once
            and not self.interrupted
        ):
            type(self).interrupted = True
            raise KeyboardInterrupt("interrupt final S2 request chunk")
        if request == ("S1",):
            references = tuple(
                target
                for index in range(251)
                if (target := CitationTarget.from_values(semantic_scholar_id=f"C{index:03d}"))
                is not None
            )
            papers = (_paper("Seed", "S1", references=references),)
        else:
            papers = tuple(_paper(f"Child {identifier}", identifier) for identifier in identifiers)
        return NativeBatchResult(
            source="semantic_scholar",
            papers=papers,
            requested=len(identifiers),
        )

    def close(self) -> None:
        pass


class _PagedOpenAlexConnector:
    batch_calls: ClassVar[list[tuple[str, ...]]] = []
    page_calls: ClassVar[list[str]] = []
    interrupt_second_page_once: ClassVar[bool] = False
    fail_second_page: ClassVar[bool] = False
    interrupted: ClassVar[bool] = False

    def __init__(self, **_kwargs: object) -> None:
        pass

    @staticmethod
    def max_paper_batch_size() -> int:
        return 50

    def fetch_papers_batch_request(
        self, identifiers: list[str], **_kwargs: object
    ) -> NativeBatchResult:
        self.batch_calls.append(tuple(identifiers))
        focal = Paper(
            title="OpenAlex focal",
            abstract="",
            authors=[],
            source=None,
            publication_date=None,
            identifiers=[PaperIdentifier(IdentifierKind.OPENALEX, "W1")],
            citation_edge_totals={"openalex": {"references": 0, "citations": 3}},
        )
        return NativeBatchResult(source="openalex", papers=(focal,), requested=1)

    def fetch_cited_by_page(
        self,
        openalex_id: str,
        cursor: str,
        *,
        sort_by_citations: bool = False,
    ) -> OpenAlexCitedByPageResult:
        assert openalex_id == "https://openalex.org/W1"
        assert sort_by_citations is True
        self.page_calls.append(cursor)
        if cursor == "CURSOR2" and self.interrupt_second_page_once and not self.interrupted:
            type(self).interrupted = True
            raise KeyboardInterrupt("interrupt second OpenAlex cursor page")
        if cursor == "CURSOR2" and self.fail_second_page:
            return OpenAlexCitedByPageResult(
                source="openalex",
                openalex_id=openalex_id,
                cursor=cursor,
                papers=(),
                raw_result_count=0,
                next_cursor=None,
                request_failures=1,
                error_messages=("HTTPError: 503",),
            )
        ids = ("W2", "W3") if cursor == "*" else ("W4",)
        papers = tuple(
            Paper(
                title=f"Citing {identifier}",
                abstract="",
                authors=[],
                source=None,
                publication_date=None,
                identifiers=[PaperIdentifier(IdentifierKind.OPENALEX, identifier)],
            )
            for identifier in ids
        )
        return OpenAlexCitedByPageResult(
            source="openalex",
            openalex_id=openalex_id,
            cursor=cursor,
            papers=papers,
            raw_result_count=len(papers),
            next_cursor="CURSOR2" if cursor == "*" else None,
            reported_total=3,
        )

    def close(self) -> None:
        pass


class _SingletonCrossRefConnector:
    calls: ClassVar[list[str]] = []
    interrupt_second_once: ClassVar[bool] = False
    interrupted: ClassVar[bool] = False

    def __init__(self, **_kwargs: object) -> None:
        pass

    def fetch_paper_by_doi(self, doi: str) -> Paper:
        self.calls.append(doi)
        if doi == "10.0/b" and self.interrupt_second_once and not self.interrupted:
            type(self).interrupted = True
            raise KeyboardInterrupt("interrupt second Crossref singleton")
        return Paper(
            title=f"Crossref {doi}",
            abstract="",
            authors=[],
            source=None,
            publication_date=None,
            doi=doi,
            citation_edge_totals={"crossref": {"references": 0}},
        )

    def close(self) -> None:
        pass


def test_uninterrupted_and_resumed_native_runs_are_semantically_equal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "lit_search.runners.snowball_runner.SemanticScholarConnector",
        _FrozenSemanticConnector,
    )

    uninterrupted_root = tmp_path / "uninterrupted"
    with _create_store(uninterrupted_root) as uninterrupted_store:
        _FrozenSemanticConnector.calls = []
        _FrozenSemanticConnector.interrupt_s2_once = False
        uninterrupted_runner = _runner(_seed_input(), run_state=uninterrupted_store)
        frozen_spec = uninterrupted_runner.semantic_spec
        expected_result = uninterrupted_runner.run(show_progress=False)
        assert uninterrupted_runner.semantic_spec == frozen_spec
        expected_signature = _result_signature(expected_result)
        expected_digest = uninterrupted_store.semantic_outcome_digest()
        assert uninterrupted_store.snapshot().lifecycle is RunLifecycle.COMPLETED

    resumed_root = tmp_path / "resumed"
    interrupted_store = _create_store(resumed_root, SECOND_RUN_ID)
    _FrozenSemanticConnector.calls = []
    _FrozenSemanticConnector.interrupt_s2_once = True
    _FrozenSemanticConnector.interrupted = False
    with pytest.raises(KeyboardInterrupt, match="frozen interruption"):
        _runner(_seed_input(), run_state=interrupted_store).run(show_progress=False)
    assert interrupted_store.snapshot().lifecycle is RunLifecycle.ACTIVE
    interrupted_units = interrupted_store.work_units()
    assert [(unit.request_kind, unit.state) for unit in interrupted_units] == [
        ("native_resolution_aggregate", WorkUnitState.SUCCEEDED),
        ("paper_batch", WorkUnitState.SUCCEEDED),
        ("native_resolution_aggregate", WorkUnitState.IN_FLIGHT),
        ("paper_batch", WorkUnitState.IN_FLIGHT),
    ]
    interrupted_store.close()

    probe = _runner(_seed_input())
    with RunStateStore.resume(
        resumed_root,
        SECOND_RUN_ID,
        probe.semantic_spec,
        credentials=probe.credential_identities,
    ) as resumed_store:
        resumed_result = _runner(_seed_input(), run_state=resumed_store).run(show_progress=False)
        assert _result_signature(resumed_result) == expected_signature
        assert resumed_store.semantic_outcome_digest() == expected_digest
        assert resumed_store.snapshot().lifecycle is RunLifecycle.COMPLETED

    assert _FrozenSemanticConnector.calls.count(("S1",)) == 1
    assert _FrozenSemanticConnector.calls.count(("S2",)) == 2


def test_completed_s2_request_chunk_is_not_reissued_after_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 251-position graph batch checkpoints its 250- and 1-ID POSTs separately."""
    monkeypatch.setattr(
        "lit_search.runners.snowball_runner.SemanticScholarConnector",
        _ChunkedSemanticConnector,
    )

    def chunk_runner(run_state: RunStateStore | None = None) -> SnowballRunner:
        return SnowballRunner(
            _seed_input(),
            max_depth=2,
            databases=["semantic_scholar"],
            enrichment_databases=[],
            run_state=run_state,
        )

    def chunk_store(root: Path, run_id: str) -> RunStateStore:
        probe = chunk_runner()
        return RunStateStore.create(
            root,
            probe.semantic_spec,
            credentials=probe.credential_identities,
            run_id=run_id,
        )

    uninterrupted_root = tmp_path / "chunk-uninterrupted"
    with chunk_store(uninterrupted_root, RUN_ID) as uninterrupted_store:
        _ChunkedSemanticConnector.calls = []
        _ChunkedSemanticConnector.interrupt_final_chunk_once = False
        expected_result = chunk_runner(uninterrupted_store).run(show_progress=False)
        expected_signature = _result_signature(expected_result)
        expected_digest = uninterrupted_store.semantic_outcome_digest()
        request_units = [
            unit for unit in uninterrupted_store.work_units() if unit.request_kind == "paper_batch"
        ]
        assert [len(unit.targets) for unit in request_units] == [1, 250, 1]
        assert all(
            unit.semantic_context["granularity"] == "actual_provider_request_page_i2b_v1"
            for unit in request_units
        )

    resumed_root = tmp_path / "chunk-resumed"
    interrupted_store = chunk_store(resumed_root, SECOND_RUN_ID)
    _ChunkedSemanticConnector.calls = []
    _ChunkedSemanticConnector.interrupt_final_chunk_once = True
    _ChunkedSemanticConnector.interrupted = False
    with pytest.raises(KeyboardInterrupt, match="interrupt final S2 request chunk"):
        chunk_runner(interrupted_store).run(show_progress=False)
    interrupted_store.close()

    probe = chunk_runner()
    with RunStateStore.resume(
        resumed_root,
        SECOND_RUN_ID,
        probe.semantic_spec,
        credentials=probe.credential_identities,
    ) as resumed_store:
        resumed_result = chunk_runner(resumed_store).run(show_progress=False)
        assert _result_signature(resumed_result) == expected_signature
        assert resumed_store.semantic_outcome_digest() == expected_digest

    calls = _ChunkedSemanticConnector.calls
    assert sum(request == ("S1",) for request in calls) == 1
    assert sum(len(request) == 250 for request in calls) == 1
    assert sum(len(request) == 1 and request != ("S1",) for request in calls) == 2


def test_completed_openalex_cursor_page_is_not_reissued_after_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The persisted next cursor deterministically plans only the unfinished page."""
    monkeypatch.setattr(
        "lit_search.runners.snowball_runner.OpenAlexConnector",
        _PagedOpenAlexConnector,
    )
    seed = Paper(
        title="OpenAlex seed",
        abstract="",
        authors=[],
        source=None,
        publication_date=None,
        identifiers=[PaperIdentifier(IdentifierKind.OPENALEX, "W1")],
    )

    def make_runner(run_state: RunStateStore | None = None) -> SnowballRunner:
        return SnowballRunner(
            seed,
            max_depth=1,
            direction="forward",
            databases=["openalex"],
            enrichment_databases=[],
            max_cited_by=10,
            run_state=run_state,
        )

    target = seed.citation_target()
    first_probe = make_runner()
    with RunStateStore.create(
        tmp_path / "oa-uninterrupted",
        first_probe.semantic_spec,
        credentials=first_probe.credential_identities,
        run_id=RUN_ID,
    ) as uninterrupted_store:
        _PagedOpenAlexConnector.batch_calls = []
        _PagedOpenAlexConnector.page_calls = []
        _PagedOpenAlexConnector.interrupt_second_page_once = False
        _PagedOpenAlexConnector.fail_second_page = False
        expected = make_runner(uninterrupted_store)._fetch_native_source(
            "openalex", [target], include_edges=True
        )
        expected_digest = uninterrupted_store.semantic_outcome_digest()
        assert [
            edge.target.get(IdentifierKind.OPENALEX) for edge in expected.papers[0].cited_by_edges
        ] == [
            "W2",
            "W3",
            "W4",
        ]

    resumed_root = tmp_path / "oa-resumed"
    interrupted_probe = make_runner()
    interrupted_store = RunStateStore.create(
        resumed_root,
        interrupted_probe.semantic_spec,
        credentials=interrupted_probe.credential_identities,
        run_id=SECOND_RUN_ID,
    )
    _PagedOpenAlexConnector.batch_calls = []
    _PagedOpenAlexConnector.page_calls = []
    _PagedOpenAlexConnector.interrupt_second_page_once = True
    _PagedOpenAlexConnector.fail_second_page = False
    _PagedOpenAlexConnector.interrupted = False
    with pytest.raises(KeyboardInterrupt, match="interrupt second OpenAlex cursor page"):
        make_runner(interrupted_store)._fetch_native_source(
            "openalex", [target], include_edges=True
        )
    interrupted_store.close()

    resume_probe = make_runner()
    with RunStateStore.resume(
        resumed_root,
        SECOND_RUN_ID,
        resume_probe.semantic_spec,
        credentials=resume_probe.credential_identities,
    ) as resumed_store:
        resumed_store.recover_in_flight()
        resumed = make_runner(resumed_store)._fetch_native_source(
            "openalex", [target], include_edges=True
        )
        assert resumed.to_dict() == expected.to_dict()
        assert resumed_store.semantic_outcome_digest() == expected_digest
        page_units = [
            unit for unit in resumed_store.work_units() if unit.request_kind == "cited_by_page"
        ]
        assert [unit.semantic_context["cursor"] for unit in page_units] == ["*", "CURSOR2"]

    assert _PagedOpenAlexConnector.batch_calls == [("W1",)]
    assert _PagedOpenAlexConnector.page_calls == ["*", "CURSOR2", "CURSOR2"]


def test_openalex_page_failure_survives_native_aggregate_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed edge page cannot disappear behind a resolved focal target."""
    monkeypatch.setattr(
        "lit_search.runners.snowball_runner.OpenAlexConnector",
        _PagedOpenAlexConnector,
    )
    seed = Paper(
        title="OpenAlex seed",
        abstract="",
        authors=[],
        source=None,
        publication_date=None,
        identifiers=[PaperIdentifier(IdentifierKind.OPENALEX, "W1")],
    )
    runner = SnowballRunner(
        seed,
        max_depth=1,
        direction="forward",
        databases=["openalex"],
        enrichment_databases=[],
        max_cited_by=10,
    )
    _PagedOpenAlexConnector.batch_calls = []
    _PagedOpenAlexConnector.page_calls = []
    _PagedOpenAlexConnector.interrupt_second_page_once = False
    _PagedOpenAlexConnector.fail_second_page = True

    outcome = runner._fetch_native_source("openalex", [seed.citation_target()], include_edges=True)

    assert outcome.edge_request_failures == 1
    assert not outcome.complete
    assert "503" in " ".join(outcome.error_messages)
    _PagedOpenAlexConnector.fail_second_page = False


def test_completed_crossref_singleton_is_not_reissued_after_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Crossref source group is durably decomposed into one unit per DOI GET."""
    monkeypatch.setattr(
        "lit_search.runners.snowball_runner.CrossRefConnector",
        _SingletonCrossRefConnector,
    )
    seed = _seed_input()

    def make_runner(run_state: RunStateStore | None = None) -> SnowballRunner:
        return SnowballRunner(
            seed,
            max_depth=1,
            direction="backward",
            databases=["crossref"],
            enrichment_databases=[],
            run_state=run_state,
        )

    targets = [
        CitationTarget((PaperIdentifier(IdentifierKind.DOI, doi),)) for doi in ("10.0/a", "10.0/b")
    ]
    probe = make_runner()
    with RunStateStore.create(
        tmp_path / "crossref-uninterrupted",
        probe.semantic_spec,
        credentials=probe.credential_identities,
        run_id=RUN_ID,
    ) as uninterrupted_store:
        _SingletonCrossRefConnector.calls = []
        _SingletonCrossRefConnector.interrupt_second_once = False
        expected = make_runner(uninterrupted_store)._fetch_native_source(
            "crossref", targets, include_edges=True
        )
        expected_digest = uninterrupted_store.semantic_outcome_digest()

    resumed_root = tmp_path / "crossref-resumed"
    interrupted_probe = make_runner()
    interrupted_store = RunStateStore.create(
        resumed_root,
        interrupted_probe.semantic_spec,
        credentials=interrupted_probe.credential_identities,
        run_id=SECOND_RUN_ID,
    )
    _SingletonCrossRefConnector.calls = []
    _SingletonCrossRefConnector.interrupt_second_once = True
    _SingletonCrossRefConnector.interrupted = False
    with pytest.raises(KeyboardInterrupt, match="interrupt second Crossref singleton"):
        make_runner(interrupted_store)._fetch_native_source("crossref", targets, include_edges=True)
    interrupted_store.close()

    resume_probe = make_runner()
    with RunStateStore.resume(
        resumed_root,
        SECOND_RUN_ID,
        resume_probe.semantic_spec,
        credentials=resume_probe.credential_identities,
    ) as resumed_store:
        resumed_store.recover_in_flight()
        resumed = make_runner(resumed_store)._fetch_native_source(
            "crossref", targets, include_edges=True
        )
        assert resumed.to_dict() == expected.to_dict()
        assert resumed_store.semantic_outcome_digest() == expected_digest
        units = [unit for unit in resumed_store.work_units() if unit.request_kind == "work_by_doi"]
        assert [unit.semantic_context["doi"] for unit in units] == ["10.0/a", "10.0/b"]

    assert _SingletonCrossRefConnector.calls == ["10.0/a", "10.0/b", "10.0/b"]


def test_seed_payload_change_is_a_semantic_mismatch(tmp_path: Path) -> None:
    with (
        _create_store(tmp_path) as store,
        pytest.raises(SemanticSpecMismatchError, match="require a forked run"),
    ):
        _runner(_seed_input(with_embedded_edge=True), run_state=store)


def test_native_unit_plan_is_input_order_stable_and_preserves_alias_groups(
    tmp_path: Path,
) -> None:
    first_target = CitationTarget(
        (
            PaperIdentifier(IdentifierKind.SEMANTIC_SCHOLAR, "A"),
            PaperIdentifier(IdentifierKind.DOI, "10.0/a"),
        )
    )
    second_target = CitationTarget((PaperIdentifier(IdentifierKind.SEMANTIC_SCHOLAR, "B"),))

    with _create_store(tmp_path / "first", RUN_ID) as first_store:
        first_runner = _runner(_seed_input(), run_state=first_store)
        first_id, _ = first_runner._plan_native_operation(
            source="semantic_scholar",
            targets=[first_target, second_target],
            include_edges=True,
            ordinal=0,
        )

    with _create_store(tmp_path / "second", SECOND_RUN_ID) as second_store:
        second_runner = _runner(_seed_input(), run_state=second_store)
        second_id, _ = second_runner._plan_native_operation(
            source="semantic_scholar",
            targets=[second_target, first_target],
            include_edges=True,
            ordinal=0,
        )
        context = second_store.work_units()[0].semantic_context

    regrouped_first = CitationTarget((PaperIdentifier(IdentifierKind.SEMANTIC_SCHOLAR, "A"),))
    regrouped_second = CitationTarget(
        (
            PaperIdentifier(IdentifierKind.DOI, "10.0/a"),
            PaperIdentifier(IdentifierKind.SEMANTIC_SCHOLAR, "B"),
        )
    )
    third_run_id = "r_33333333333333333333333333333333"
    with _create_store(tmp_path / "regrouped", third_run_id) as regrouped_store:
        regrouped_runner = _runner(_seed_input(), run_state=regrouped_store)
        regrouped_id, _ = regrouped_runner._plan_native_operation(
            source="semantic_scholar",
            targets=[regrouped_first, regrouped_second],
            include_edges=True,
            ordinal=0,
        )

    assert first_id == second_id
    assert regrouped_id != first_id
    assert context["target_alias_groups"] == sorted(
        [canonical_json(sorted(first_target.keys)), canonical_json(sorted(second_target.keys))]
    )


def test_source_group_ordinals_are_canonical_under_edge_reordering(tmp_path: Path) -> None:
    first = CitationTarget((PaperIdentifier(IdentifierKind.SEMANTIC_SCHOLAR, "A"),))
    second = CitationTarget((PaperIdentifier(IdentifierKind.SEMANTIC_SCHOLAR, "B"),))

    def planned(root: Path, run_id: str, edges: list[CitationEdge]) -> list[tuple[str, int, str]]:
        with _create_store(root, run_id) as store:
            runner = _runner(_seed_input(), run_state=store)
            runner._fetch_targets(edges)
            return [(unit.source, unit.ordinal, unit.unit_id) for unit in store.work_units()]

    forward = [
        CitationEdge(target=first, source="z-source"),
        CitationEdge(target=second, source="a-source"),
    ]
    reverse = list(reversed(forward))
    first_plan = planned(tmp_path / "source-first", RUN_ID, forward)
    second_plan = planned(tmp_path / "source-second", SECOND_RUN_ID, reverse)
    assert first_plan == second_plan
    assert [(source, ordinal) for source, ordinal, _ in first_plan] == [
        ("a-source", 0),
        ("z-source", 1),
    ]


def test_doi_operations_replay_and_none_preserves_legacy_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("lit_search.runners.snowball_runner.GetRunner", _FakeGetRunner)
    _FakeGetRunner.calls = []
    with _create_store(tmp_path / "resumable") as store:
        resumable = _runner(_seed_input(), run_state=store)
        first = resumable._fetch_dois(
            ["10.0/X"],
            ["crossref"],
            desc="test DOI replay",
            verbose=False,
            show_progress=False,
        )
        second = resumable._fetch_dois(
            ["10.0/x"],
            ["crossref"],
            desc="test DOI replay",
            verbose=False,
            show_progress=False,
        )
        assert [paper.doi for paper in first] == ["10.0/X"]
        assert [paper.doi for paper in second] == ["10.0/X"]
        assert _FakeGetRunner.calls == ["10.0/X"]

    _FakeGetRunner.calls = []
    legacy = _runner(_seed_input(), run_state=None)
    legacy._fetch_dois(
        ["10.0/X"],
        ["crossref"],
        desc="legacy one",
        verbose=False,
        show_progress=False,
    )
    legacy._fetch_dois(
        ["10.0/X"],
        ["crossref"],
        desc="legacy two",
        verbose=False,
        show_progress=False,
    )
    assert _FakeGetRunner.calls == ["10.0/X", "10.0/X"]
