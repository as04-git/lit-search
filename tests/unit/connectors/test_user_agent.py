"""Every connector session identifies the tool, with the contact address when set."""

from __future__ import annotations

import pytest

from lit_search.connectors.arxiv import ArxivConnector
from lit_search.connectors.connector_base import user_agent


def test_user_agent_includes_contact_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIT_SEARCH_EMAIL", "someone@example.org")
    agent = user_agent()
    assert agent.startswith("lit-search/")
    assert "mailto:someone@example.org" in agent


def test_user_agent_omits_contact_when_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    monkeypatch.delenv("LIT_SEARCH_EMAIL", raising=False)
    monkeypatch.setenv("LIT_SEARCH_CREDENTIALS", str(tmp_path / "none.env"))
    assert "mailto" not in user_agent()


def test_connector_session_sends_the_tool_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIT_SEARCH_EMAIL", "someone@example.org")
    session = ArxivConnector()._get_session()
    assert session.headers["User-Agent"] == user_agent()
