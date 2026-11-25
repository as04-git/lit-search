"""Strict source-snapshot reuse for ``SnowballRunner``."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, Literal

import pytest

from lit_search.core.citation import (
    CitationEdge,
    CitationTarget,
    IdentifierKind,
    NativeBatchResult,
    PaperIdentifier,
)
from lit_search.core.paper import Paper
from lit_search.runners import snowball_runner as snowball_module
from lit_search.runners.snowball_runner import SnowballRunner
from lit_search.store.run_state import RunStateStore
from lit_search.store.snapshot_cache import AliasResolution, SnapshotCache, SnapshotQuality


def _target(
    *,
    s2_id: str | None = None,
    doi: str | None = None,
) -> CitationTarget:
    value = CitationTarget.from_values(semantic_scholar_id=s2_id, doi=doi)
    assert value is not None
    return value


def _paper(
    title: str,
    s2_id: str,
    *,
    doi: str | None = None,
    references: tuple[CitationTarget, ...] = (),
    citations: tuple[CitationTarget, ...] = (),
) -> Paper:
    return Paper(
        title=title,
        abstract=f"{title} abstract",
        authors=[],
        source=None,
        publication_date=dt.date(2026, 1, 1),
        doi=doi,
        identifiers=[PaperIdentifier(IdentifierKind.SEMANTIC_SCHOLAR, s2_id)],
        reference_edges=[
            CitationEdge(target=item, source="semantic_scholar") for item in references
        ],
        cited_by_edges=[CitationEdge(target=item, source="semantic_scholar") for item in citations],
        citation_edge_totals={
            "semantic_scholar": {
                "references": len(references),
                "citations": len(citations),
            }
        },
    )


def _runner(
    cache: SnapshotCache | None,
    *,
    direction: Literal["both", "backward", "forward"] = "both",
    run_state: RunStateStore | None = None,
) -> SnowballRunner:
    return SnowballRunner(
        _paper("Seed", "SEED"),
        max_depth=1,
        direction=direction,
        databases=["semantic_scholar"],
        enrichment_databases=[],
        snapshot_cache=cache,
        run_state=run_state,
    )


class _CompleteConnector:
    calls: ClassVar[list[tuple[str, ...]]] = []
    paper: ClassVar[Paper] = _paper("unset", "UNSET")

    def __init__(self, **_kwargs: object) -> None:
        pass

    def fetch_papers_batch(self, identifiers: list[str], **_kwargs: object) -> NativeBatchResult:
        self.calls.append(tuple(identifiers))
        return NativeBatchResult(
            source="semantic_scholar",
            papers=(self.paper,),
            requested=len(identifiers),
        )

    def close(self) -> None:
        pass


def _edge(target: CitationTarget) -> list[CitationEdge]:
    return [CitationEdge(target=target, source="semantic_scholar")]


def test_warm_complete_snapshots_match_cold_paper_and_source_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = SnapshotCache(tmp_path)
    reference = _target(s2_id="REF")
    citation = _target(s2_id="CITE")
    _CompleteConnector.calls = []
    _CompleteConnector.paper = _paper(
        "Resolved",
        "WORK",
        doi="10.1000/work",
        references=(reference,),
        citations=(citation,),
    )
    monkeypatch.setattr(snowball_module, "SemanticScholarConnector", _CompleteConnector)

    target = _target(s2_id="WORK", doi="10.1000/work")
    cold_runner = _runner(cache)
    cold = cold_runner._fetch_targets(_edge(target))
    warm_runner = _runner(cache)
    warm = warm_runner._fetch_targets(_edge(target))

    assert _CompleteConnector.calls == [("WORK",)]
    assert [paper.to_dict() for paper in warm] == [paper.to_dict() for paper in cold]
    assert warm_runner._coverage["semantic_scholar"].to_coverage().to_dict() == (
        cold_runner._coverage["semantic_scholar"].to_coverage().to_dict()
    )


@pytest.mark.parametrize(
    ("quality", "reported", "returned", "fetched_at", "ttl_seconds"),
    [
        (SnapshotQuality.PARTIAL, None, 0, None, 3600.0),
        (SnapshotQuality.TRUNCATED, 1, 0, None, 3600.0),
        (SnapshotQuality.COMPLETE, 0, 0, 0.0, 1.0),
    ],
)
def test_lower_bound_or_stale_snapshot_cannot_satisfy_current_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quality: SnapshotQuality,
    reported: int | None,
    returned: int,
    fetched_at: float | None,
    ttl_seconds: float,
) -> None:
    cache = SnapshotCache(tmp_path)
    runner = _runner(cache, direction="backward")
    cached_paper = _paper("Cached lower bound", "CACHED", doi="10.1000/alias")
    directional = runner._directional_paper(cached_paper, "semantic_scholar", "backward")
    cache.put_graph_snapshot(
        source="semantic_scholar",
        native_key="semantic_scholar:cached",
        direction="backward",
        request_shape=runner._cache_request_shape("semantic_scholar", "backward"),
        payload={
            "schema": snowball_module._SNAPSHOT_PAYLOAD_SCHEMA,
            "source": "semantic_scholar",
            "native_key": "semantic_scholar:cached",
            "direction": "backward",
            "paper": directional.to_dict(),
        },
        quality=quality,
        reported_count=reported,
        returned_count=returned,
        ttl_seconds=ttl_seconds,
        fetched_at=fetched_at,
        aliases=("doi:10.1000/alias",),
    )

    _CompleteConnector.calls = []
    _CompleteConnector.paper = _paper("Fresh", "FRESH", doi="10.1000/alias")
    monkeypatch.setattr(snowball_module, "SemanticScholarConnector", _CompleteConnector)
    result = runner._fetch_targets(_edge(_target(doi="10.1000/alias")))

    assert [paper.title for paper in result] == ["Fresh"]
    assert _CompleteConnector.calls == [("DOI:10.1000/alias",)]


def test_conflicting_aliases_are_cache_misses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = SnapshotCache(tmp_path)
    runner = _runner(cache, direction="backward")
    for native_id in ("A", "B"):
        paper = _paper(f"Cached {native_id}", native_id, doi="10.1000/conflict")
        directional = runner._directional_paper(paper, "semantic_scholar", "backward")
        native_key = f"semantic_scholar:{native_id.lower()}"
        cache.put_graph_snapshot(
            source="semantic_scholar",
            native_key=native_key,
            direction="backward",
            request_shape=runner._cache_request_shape("semantic_scholar", "backward"),
            payload={
                "schema": snowball_module._SNAPSHOT_PAYLOAD_SCHEMA,
                "source": "semantic_scholar",
                "native_key": native_key,
                "direction": "backward",
                "paper": directional.to_dict(),
            },
            quality=SnapshotQuality.COMPLETE,
            reported_count=0,
            returned_count=0,
            ttl_seconds=3600,
            aliases=("doi:10.1000/conflict",),
        )

    resolution = cache.resolve_alias("semantic_scholar", "doi:10.1000/conflict")
    assert resolution.conflict
    _CompleteConnector.calls = []
    _CompleteConnector.paper = _paper("Fresh", "FRESH", doi="10.1000/conflict")
    monkeypatch.setattr(snowball_module, "SemanticScholarConnector", _CompleteConnector)

    result = runner._fetch_targets(_edge(_target(doi="10.1000/conflict")))

    assert [paper.title for paper in result] == ["Fresh"]
    assert _CompleteConnector.calls == [("DOI:10.1000/conflict",)]


def test_shared_provider_snapshot_cannot_satisfy_conflicting_doi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A native ID match cannot launder a different stated publication version."""
    cache = SnapshotCache(tmp_path)
    runner = _runner(cache, direction="backward")
    cached = _paper("Cached wrong version", "SHARED", doi="10.1000/first")
    directional = runner._directional_paper(cached, "semantic_scholar", "backward")
    cache.put_graph_snapshot(
        source="semantic_scholar",
        native_key="semantic_scholar:shared",
        direction="backward",
        request_shape=runner._cache_request_shape("semantic_scholar", "backward"),
        payload={
            "schema": snowball_module._SNAPSHOT_PAYLOAD_SCHEMA,
            "source": "semantic_scholar",
            "native_key": "semantic_scholar:shared",
            "direction": "backward",
            "paper": directional.to_dict(),
        },
        quality=SnapshotQuality.COMPLETE,
        reported_count=0,
        returned_count=0,
        ttl_seconds=3600,
        aliases=("doi:10.1000/first", "semantic_scholar:shared"),
    )

    _CompleteConnector.calls = []
    _CompleteConnector.paper = _paper("Fresh right version", "SHARED", doi="10.1000/second")
    monkeypatch.setattr(snowball_module, "SemanticScholarConnector", _CompleteConnector)

    result = runner._fetch_targets(_edge(_target(s2_id="SHARED", doi="10.1000/second")))

    assert [paper.title for paper in result] == ["Fresh right version"]
    assert _CompleteConnector.calls == [("SHARED",)]


def test_terminal_run_replay_precedes_mutated_shared_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = SnapshotCache(tmp_path / "store")
    probe = _runner(cache)
    store = RunStateStore.create(
        tmp_path / "store",
        probe.semantic_spec,
        credentials=probe.credential_identities,
        run_id="r_99999999999999999999999999999999",
    )
    try:
        runner = _runner(cache, run_state=store)
        _CompleteConnector.calls = []
        _CompleteConnector.paper = _paper("Persisted A", "WORK")
        monkeypatch.setattr(snowball_module, "SemanticScholarConnector", _CompleteConnector)
        target = _target(s2_id="WORK")
        first = runner._fetch_targets(_edge(target))

        replacement = _paper("Cache B", "WORK")
        for direction in ("backward", "forward"):
            directional = runner._directional_paper(replacement, "semantic_scholar", direction)
            cache.put_graph_snapshot(
                source="semantic_scholar",
                native_key="semantic_scholar:work",
                direction=direction,
                request_shape=runner._cache_request_shape("semantic_scholar", direction),
                payload={
                    "schema": snowball_module._SNAPSHOT_PAYLOAD_SCHEMA,
                    "source": "semantic_scholar",
                    "native_key": "semantic_scholar:work",
                    "direction": direction,
                    "paper": directional.to_dict(),
                },
                quality=SnapshotQuality.COMPLETE,
                reported_count=0,
                returned_count=0,
                ttl_seconds=3600,
                aliases=("semantic_scholar:work",),
            )

        replayed = runner._fetch_targets(_edge(target))
    finally:
        store.close()

    assert [paper.title for paper in first] == ["Persisted A"]
    assert [paper.title for paper in replayed] == ["Persisted A"]
    assert _CompleteConnector.calls == [("WORK",)]


class _RefusingCache:
    writes = 0

    def resolve_alias(self, source: str, alias: str) -> AliasResolution:
        return AliasResolution(source, alias, ())

    def get_graph_snapshot(self, **_kwargs: object) -> None:
        return None

    def put_graph_snapshot(self, **_kwargs: object) -> SimpleNamespace:
        type(self).writes += 1
        return SimpleNamespace(stored=False, reason="test budget refusal")


def test_cache_write_refusal_does_not_fail_successful_traversal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _RefusingCache()
    runner = _runner(cache)  # type: ignore[arg-type]
    _RefusingCache.writes = 0
    _CompleteConnector.calls = []
    _CompleteConnector.paper = _paper("Network survives", "WORK")
    monkeypatch.setattr(snowball_module, "SemanticScholarConnector", _CompleteConnector)

    result = runner._fetch_targets(_edge(_target(s2_id="WORK")))

    assert [paper.title for paper in result] == ["Network survives"]
    assert _RefusingCache.writes == 2


class _BareGetRunner:
    calls: ClassVar[list[str]] = []

    def __init__(self, *, identifier: str, **_kwargs: object) -> None:
        self.identifier = identifier

    def run(self, **_kwargs: object) -> Paper:
        self.calls.append(self.identifier)
        return _paper("Composite lookup", "LOOKUP", doi=self.identifier)


def test_composite_get_runner_results_are_not_source_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = SnapshotCache(tmp_path)
    runner = _runner(cache)
    _BareGetRunner.calls = []
    monkeypatch.setattr(snowball_module, "GetRunner", _BareGetRunner)

    for _ in range(2):
        runner._fetch_dois(
            ["10.1000/composite"],
            ["crossref", "semantic_scholar"],
            desc="composite",
            verbose=False,
            show_progress=False,
        )

    assert _BareGetRunner.calls == ["10.1000/composite", "10.1000/composite"]
    assert cache.usage().entries == 0
