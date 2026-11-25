"""Recorded moves are rebuilt as CLI arguments with stable seed ids."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from lit_search.evaluation import replay


class _Original:
    def __init__(self, candidates: dict[str, Any]) -> None:
        self.candidates = candidates


@pytest.fixture
def original(monkeypatch: pytest.MonkeyPatch) -> _Original:
    monkeypatch.setattr(replay, "s2_request_id_for_paper", lambda paper: f"ARXIV:{paper.arxiv}")
    return _Original(
        {
            "c3": SimpleNamespace(paper=SimpleNamespace(arxiv="2401.00001")),
            "c7": SimpleNamespace(paper=SimpleNamespace(arxiv="2401.00002")),
        }
    )


def test_find_from_replay_record(original: _Original) -> None:
    move = {
        "kind": "find",
        "args": {"text": "a b"},
        "replay": {
            "params": {"text": ["a", "b"], "k": 15, "since": "2024-01-01", "databases": ["arxiv"]}
        },
    }
    argv = replay.move_argv(move, original, "hunt-x")  # type: ignore[arg-type]
    assert argv == [
        "hunt",
        "find",
        "hunt-x",
        "--brief",
        "--quiet",
        "a",
        "b",
        "-k",
        "15",
        "--since",
        "2024-01-01",
        "-d",
        "arxiv",
    ]


def test_related_translates_candidate_ids(original: _Original) -> None:
    move = {"kind": "related", "args": {"seeds": "c3 c7", "not": None}}
    argv = replay.move_argv(move, original, "hunt-x")  # type: ignore[arg-type]
    assert argv is not None
    assert argv[-2:] == ["ARXIV:2401.00001", "ARXIV:2401.00002"]


def test_snowball_direction_from_old_args(original: _Original) -> None:
    move = {
        "kind": "snowball",
        "args": {"seeds": "c3", "direction": "forward", "since": "2025-06-01"},
    }
    argv = replay.move_argv(move, original, "hunt-x")  # type: ignore[arg-type]
    assert argv is not None
    assert "--forward" in argv
    assert argv[-2:] == ["--since", "2025-06-01"]


def test_unknown_seed_and_judging_moves_are_not_replayed(original: _Original) -> None:
    assert replay.move_argv({"kind": "related", "args": {"seeds": "c99"}}, original, "h") is None  # type: ignore[arg-type]
    assert replay.move_argv({"kind": "mark", "args": {}}, original, "h") is None  # type: ignore[arg-type]
    retry = {"kind": "find", "args": {"text": "a", "retry_of": 2}}
    assert replay.move_argv(retry, original, "h") is None  # type: ignore[arg-type]


def test_add_splits_titles(original: _Original) -> None:
    move = {"kind": "add", "args": {"refs": "Title One | Title Two"}}
    argv = replay.move_argv(move, original, "h")  # type: ignore[arg-type]
    assert argv is not None
    assert argv[-2:] == ["Title One", "Title Two"]
