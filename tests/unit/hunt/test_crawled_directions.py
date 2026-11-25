"""A snowball direction counts as crawled only when its step answered."""

from __future__ import annotations

from types import SimpleNamespace

from lit_search.cli.hunt import _crawled_directions
from lit_search.core.coverage import SourceCoverage


def _session() -> SimpleNamespace:
    return SimpleNamespace(
        candidates={
            "c1": SimpleNamespace(paper=SimpleNamespace(title="Specula: Scaling formal specs")),
            "c2": SimpleNamespace(paper=SimpleNamespace(title="FM-Agent: formal methods")),
        }
    )


def test_failed_direction_stays_in_queue() -> None:
    steps = [
        SourceCoverage.complete(
            "snowball:forward:Specula: Scaling formal specs", 5, reported_total=5
        ),
        SourceCoverage.failed("snowball:backward:Specula: Scaling formal specs", error="429"),
    ]
    got = _crawled_directions(_session(), ["c1"], steps, True, True)  # type: ignore[arg-type]
    assert got == {"c1": ["forward"]}


def test_seed_matched_by_title_not_by_others_success() -> None:
    steps = [
        SourceCoverage.complete(
            "snowball:forward:Specula: Scaling formal specs", 5, reported_total=5
        ),
        SourceCoverage.failed("snowball:forward:FM-Agent: formal methods", error="429"),
    ]
    got = _crawled_directions(_session(), ["c1", "c2"], steps, True, False)  # type: ignore[arg-type]
    assert got == {"c1": ["forward"], "c2": []}


def test_unmatched_titles_count_only_without_failures() -> None:
    ok = [SourceCoverage.complete("snowball:forward:Other title from S2", 3, reported_total=3)]
    assert _crawled_directions(_session(), ["c1"], ok, True, False) == {"c1": ["forward"]}  # type: ignore[arg-type]
    mixed = [*ok, SourceCoverage.failed("snowball:forward:Another", error="429")]
    assert _crawled_directions(_session(), ["c1"], mixed, True, False) == {"c1": []}  # type: ignore[arg-type]
