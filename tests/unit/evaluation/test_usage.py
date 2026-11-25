"""Tests for usage statistics over hunt sessions."""

from __future__ import annotations

import json
from pathlib import Path

from lit_search.evaluation.usage import failure_class, load_sessions, render, summarize


def _session(started: float = 1_792_000_000.0) -> dict:
    return {
        "started": started,
        "moves": [
            {
                "number": 1,
                "kind": "find",
                "seconds": 20.0,
                "args": {"text": "q"},
                "steps": [
                    {"source": "openalex", "status": "capped"},
                    {
                        "source": "semantic_scholar",
                        "status": "failed",
                        "error": "no answer within the 20s deadline",
                        "error_type": "Timeout",
                    },
                    {"source": "arxiv", "status": "failed", "error": "429 Client Error"},
                ],
            },
            {
                "number": 2,
                "kind": "related",
                "seconds": 10.0,
                "args": {"seeds": "c1", "not": None},
                "steps": [{"source": "related:recommendations", "status": "complete"}],
            },
        ],
        "candidates": {
            "c1": {
                "paper": {"title": "Paper One"},
                "routes": [{"move": 1, "label": "find#1 (openalex#1)"}],
                "mark": "core",
            },
            "c4": {
                "paper": {"title": "Paper Four"},
                "routes": [{"move": 1, "label": "find#3 (arxiv#2,openalex#5)"}],
                "mark": "keep",
            },
            "c2": {
                "paper": {"title": "Paper Two"},
                "routes": [
                    {"move": 2, "label": "related#1"},
                    {"move": 1, "label": "find#1 (openalex#2)"},
                ],
                "mark": "keep",
            },
            "c3": {
                "paper": {"title": "Paper Three"},
                "routes": [{"move": 2, "label": "related#2"}],
                "mark": "drop",
            },
        },
    }


def test_failure_class_names_kinds() -> None:
    assert failure_class({"error": "429 Client Error"}) == "429"
    assert failure_class({"error": "no answer within the 20s deadline"}) == "deadline"
    assert failure_class({"error": "404 Client Error"}) == "404"
    assert failure_class({"error": "HTTP 503 from x"}) == "5xx"
    assert failure_class({"error": "boom"}) == "other"


def test_summarize_credits_first_route_and_counts_failures() -> None:
    report = summarize([_session()])
    assert report.hunts == 1
    assert report.kept == 3
    assert report.by_source["arxiv"].reached_kept == 1  # second source in a merged label
    assert report.by_source["arxiv"].exclusive == 0
    find = report.by_move["find"]
    related = report.by_move["related"]
    assert find.first_kept == 2  # c1, c4
    assert related.first_kept == 1  # c2 reached first by related
    assert find.reached == 3
    assert find.reached_kept == 3
    assert find.failed == {"deadline": 1, "429": 1}
    assert report.by_source["semantic_scholar"].failed == {"deadline": 1}
    assert report.by_source["openalex"].exclusive_kept == 1  # c1: only openalex, only find
    assert report.related_with_negatives == 0


def test_repeat_share_counts_papers_seen_in_several_hunts() -> None:
    report = summarize([_session(), _session()])
    assert report.repeat_share == 0.5


def test_load_sessions_filters_by_date_and_skips_bad_files(tmp_path: Path) -> None:
    import datetime as dt

    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "session.json").write_text(json.dumps(_session(started=0.0)))
    (tmp_path / "new").mkdir()
    (tmp_path / "new" / "session.json").write_text(json.dumps(_session()))
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "session.json").write_text("{not json")
    assert len(load_sessions(tmp_path)) == 2
    assert len(load_sessions(tmp_path, since=dt.date(2020, 1, 1))) == 1


def test_render_mentions_moves_and_caveat() -> None:
    text = render(summarize([_session()]))
    assert "related moves using negatives: 0/1" in text
    assert "Caveats" in text
