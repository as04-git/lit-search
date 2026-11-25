"""Refusal statistics from the request log."""

from __future__ import annotations

from lit_search.evaluation.requests import analyse, render_requests


def test_refusals_binned_by_gap_and_trailing_count() -> None:
    entries = [{"t": float(t), "scope": "s2:k", "status": 200} for t in range(0, 30, 2)]
    entries += [
        {"t": 30.2, "scope": "s2:k", "status": 429},
        {"t": 30.4, "scope": "s2:k", "status": 429},
    ]
    entries.append({"t": 31.0, "scope": "s2:k", "status": "Timeout"})
    report = analyse(entries)["s2:k"]
    assert report.requests == 18
    assert report.refused == 2
    assert report.transport == 1
    assert report.by_gap[0.5].refused == 1  # 30.4 came 0.2 s after 30.2
    assert report.by_gap[5.0].refused == 1  # 30.2 came 2.2 s after 28
    assert report.by_gap[2.0].refused == 0
    assert report.by_count[(1.0, 0)].refused == 1
    assert report.by_count[(1.0, 1)].refused == 1
    text = render_requests({"s2:k": report})
    assert "2 refused" in text
