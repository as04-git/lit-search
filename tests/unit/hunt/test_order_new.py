"""Move output follows the relevance service when it answers, the hunt's score otherwise."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lit_search.cli import hunt


def _session() -> SimpleNamespace:
    cands = {
        f"c{i}": SimpleNamespace(paper=SimpleNamespace(title=f"T{i}", abstract="a"), score=i)
        for i in range(1, 5)
    }
    return SimpleNamespace(
        candidates=cands,
        data={"question": "q"},
        score=lambda c: c.score,
    )


def test_falls_back_to_score_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hunt, "relevance_scores", lambda q, docs: None)
    order, by = hunt._order_new(_session(), ["c1", "c2", "c3", "c4"])  # type: ignore[arg-type]
    assert order == ["c4", "c3", "c2", "c1"]
    assert by == "score"


def test_uses_relevance_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    # documents arrive in score order (c4, c3, c2, c1); relevance prefers c1 then c3
    monkeypatch.setattr(hunt, "relevance_scores", lambda q, docs: ([0.1, 0.5, 0.2, 0.9], "m"))
    order, by = hunt._order_new(_session(), ["c1", "c2", "c3", "c4"])  # type: ignore[arg-type]
    assert order == ["c1", "c3", "c2", "c4"]
    assert by == "relevance:m"


def test_relevance_client_returns_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    from lit_search.utils.relevance import relevance_scores

    monkeypatch.delenv("LIT_SEARCH_RERANK_URL", raising=False)
    assert relevance_scores("q", ["d"]) is None


def test_relevance_client_returns_none_when_down(monkeypatch: pytest.MonkeyPatch) -> None:
    from lit_search.utils.relevance import relevance_scores

    monkeypatch.setenv("LIT_SEARCH_RERANK_URL", "http://127.0.0.1:9")
    assert relevance_scores("q", ["d"], timeout=0.5) is None
