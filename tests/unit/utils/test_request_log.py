"""Every connector request leaves one line in the shared request log."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lit_search.connectors.arxiv import ArxivConnector
from lit_search.utils import request_log


def test_record_writes_one_line_without_query_string(tmp_path: Path) -> None:
    request_log.record(
        scope="svc:abc",
        method="GET",
        url="https://api.x.org/p?key=SECRET",
        sent=1.0,
        status=429,
        latency=0.2,
    )
    lines = request_log.log_path().read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["path"] == "api.x.org/p"
    assert entry["status"] == 429
    assert "SECRET" not in lines[0]


def test_disabled_by_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(request_log.ENV_DISABLE, "0")
    request_log.record(scope="s", method="GET", url="https://a/b", sent=0, status=200, latency=0)
    assert not request_log.log_path().exists()


def test_connector_request_is_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    connector = ArxivConnector()
    response = MagicMock(status_code=200, ok=True, headers={}, content=b"")
    session = MagicMock()
    session.get.return_value = response
    monkeypatch.setattr(connector, "_get_session", lambda: session)
    monkeypatch.setattr(connector, "_rate_limit", lambda: None)
    connector._get("https://export.arxiv.org/api/query", {"search_query": "x"})
    entry = json.loads(request_log.log_path().read_text().splitlines()[-1])
    assert entry["path"] == "export.arxiv.org/api/query"
    assert entry["scope"].startswith("arxiv")
