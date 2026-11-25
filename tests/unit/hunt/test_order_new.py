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
    monkeypatch.setenv("LIT_SEARCH_RELEVANCE", "local")
    assert relevance_scores("q", ["d"]) is None


def test_relevance_client_returns_none_when_down(monkeypatch: pytest.MonkeyPatch) -> None:
    from lit_search.utils.relevance import relevance_scores

    monkeypatch.setenv("LIT_SEARCH_RERANK_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("LIT_SEARCH_RELEVANCE", "local")
    assert relevance_scores("q", ["d"], timeout=0.5) is None


class _Response:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        import json

        return json.dumps(self.payload).encode()


def test_voyage_scores_are_aligned_to_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    from lit_search.utils import relevance

    sent: list[object] = []

    def fake_urlopen(request: object, timeout: float) -> _Response:
        sent.append(request)
        return _Response(
            {
                "data": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.2},
                ],
                "model": "rerank-3",
            }
        )

    monkeypatch.setenv("VOYAGE_API_KEY", "k")
    monkeypatch.setenv("LIT_SEARCH_RELEVANCE", "auto")
    monkeypatch.setattr(relevance.urllib.request, "urlopen", fake_urlopen)
    assert relevance.relevance_scores("q", ["a", "b"]) == ([0.2, 0.9], "voyage/rerank-3")
    assert sent[0].full_url == relevance.VOYAGE_URL  # type: ignore[attr-defined]


def test_voyage_refusal_falls_back_to_local(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error

    from lit_search.utils import relevance

    def fake_urlopen(request: object, timeout: float) -> _Response:
        if request.full_url == relevance.VOYAGE_URL:  # type: ignore[attr-defined]
            raise urllib.error.HTTPError(relevance.VOYAGE_URL, 429, "busy", {}, None)  # type: ignore[arg-type]
        return _Response({"scores": [0.5, 0.1], "model": "local-ce"})

    monkeypatch.setenv("VOYAGE_API_KEY", "k")
    monkeypatch.setenv("LIT_SEARCH_RERANK_URL", "http://127.0.0.1:8797")
    monkeypatch.setenv("LIT_SEARCH_RELEVANCE", "auto")
    monkeypatch.setattr(relevance.urllib.request, "urlopen", fake_urlopen)
    assert relevance.relevance_scores("q", ["a", "b"]) == ([0.5, 0.1], "local-ce")


def test_relevance_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from lit_search.utils import relevance

    monkeypatch.setenv("VOYAGE_API_KEY", "k")
    monkeypatch.setenv("LIT_SEARCH_RELEVANCE", "off")
    assert relevance.relevance_scores("q", ["a"]) is None
    assert relevance.relevance_backend() == "off"
