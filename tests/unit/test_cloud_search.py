"""Offline checks for the bounded cloud runner and its credential-safe export."""

from __future__ import annotations

import datetime as dt
import json
from urllib.parse import quote

import pytest

from lit_search.core.coverage import CoverageLedger, SourceCoverage
from lit_search.core.search_result import SearchResult
from lit_search.exceptions import QueryValidationError
from tools import cloud_search


def test_default_window_is_fourteen_days() -> None:
    """A blank window is bounded and based on UTC date supplied by the runner."""
    result = cloud_search.read_inputs(
        {"LIT_SEARCH_QUERY": "[program synthesis]"}, dt.date(2026, 10, 1)
    )
    assert result.since == dt.date(2026, 9, 18)
    assert result.until == dt.date(2026, 10, 1)


@pytest.mark.parametrize(
    "changes",
    [
        {"LIT_SEARCH_QUERY": ""},
        {"LIT_SEARCH_QUERY": "[unclosed"},
        {"LIT_SEARCH_QUERY": "[one]\nAND [two]"},
        {"LIT_SEARCH_QUERY": "[" + "x" * 512 + "]"},
        {"LIT_SEARCH_SINCE": "20260901"},
        {"LIT_SEARCH_SINCE": "2026-08-01"},
        {"LIT_SEARCH_SINCE": "2026-10-02"},
        {"LIT_SEARCH_UNTIL": "2026-10-02"},
        {"LIT_SEARCH_UNTIL": "2026-02-30"},
    ],
)
def test_bad_inputs_rejected_before_search(changes: dict[str, str]) -> None:
    """Invalid syntax, unbounded windows and future publication dates fail closed."""
    env = {"LIT_SEARCH_QUERY": "[program synthesis]", **changes}
    with pytest.raises((ValueError, QueryValidationError)):
        cloud_search.read_inputs(env, dt.date(2026, 10, 1))


def test_export_excludes_raw_errors_and_redacts_paper_data(make_paper) -> None:
    """Even echoed keys and encoded credentials cannot enter the exported fields."""
    secret = "fixture/token+with=punct"
    paper = make_paper(
        title=f"Paper {secret}",
        abstract=f"Encoded {quote(secret, safe='')} and {secret}",
        url=f"https://user:password@example.org/paper?api_key={quote(secret, safe='')}#token",
    )
    result = SearchResult(
        query="[test]",
        papers=[paper],
        coverage=CoverageLedger(
            entries=(
                SourceCoverage.failed(
                    "openalex",
                    error=f"https://api.openalex.org/works?api_key={secret}",
                    error_type=f"deliberate-{secret}",
                ),
            )
        ),
    )
    inputs = cloud_search.SearchInputs("[test]", dt.date(2026, 9, 1), dt.date(2026, 9, 14))
    payload = cloud_search.export_result(result, inputs, (secret,))
    serialized = json.dumps(payload)
    assert secret not in serialized
    assert quote(secret, safe="") not in serialized
    assert "api_key" not in serialized
    assert "password" not in serialized
    assert "error_type" not in serialized
    assert "deliberate" not in serialized
    assert payload["papers"][0]["url"] == "https://example.org/paper"
    assert payload["execution_status"] == "incomplete"
    assert payload["coverage"][0]["status"] == "failed"


def test_main_passes_bounded_plan_and_exports(tmp_path, monkeypatch, make_paper) -> None:
    """The job uses exactly two sources, a cap of 25 and no metadata enrichment."""
    captured = {}

    class FakeEngine:
        """Return a deterministic result without touching the network."""

        def __init__(self, **kwargs):
            captured["credentials"] = set(kwargs)

        def search(self, query, **kwargs):
            captured.update(kwargs)
            return SearchResult(
                query=query,
                papers=[make_paper()],
                coverage=CoverageLedger(
                    entries=tuple(
                        SourceCoverage.complete(source, retrieved=1, reported_total=1)
                        for source in cloud_search.SOURCES
                    )
                ),
            )

    monkeypatch.setattr(cloud_search, "Engine", FakeEngine)
    env = {
        "LIT_SEARCH_QUERY": "[program synthesis]",
        "LIT_SEARCH_SINCE": "2026-09-01",
        "LIT_SEARCH_UNTIL": "2026-09-14",
        "LIT_SEARCH_OUTPUT_DIR": str(tmp_path),
        **dict.fromkeys(cloud_search.CREDENTIALS, "dummy-credential"),
    }
    assert cloud_search.main(env) == 0
    assert captured["databases"] == ["openalex", "semantic_scholar"]
    assert captured["max_papers_per_database"] == 25
    assert captured["enrichment_databases"] == []
    assert captured["verbose"] is False
    assert captured["show_progress"] is False
    assert json.loads((tmp_path / "result.json").read_text())["paper_count"] == 1


def test_missing_credentials_does_not_start_engine(tmp_path, monkeypatch) -> None:
    """Missing repository secrets must not silently consume anonymous quota."""

    def forbidden_engine(**kwargs):
        raise AssertionError("Engine must not be constructed")

    monkeypatch.setattr(cloud_search, "Engine", forbidden_engine)
    assert (
        cloud_search.main({"LIT_SEARCH_QUERY": "[test]", "LIT_SEARCH_OUTPUT_DIR": str(tmp_path)})
        == 1
    )
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["reason"] == "missing_openalex_or_semantic_scholar_credential"


def test_provider_exception_is_not_persisted(tmp_path, monkeypatch) -> None:
    """An unknown error leaves a fixed diagnostic code with no secret or traceback."""

    def failing_engine(**kwargs):
        raise RuntimeError("secret-in-authenticated-request-url")

    monkeypatch.setattr(cloud_search, "Engine", failing_engine)
    env = {
        "LIT_SEARCH_QUERY": "[test]",
        "LIT_SEARCH_OUTPUT_DIR": str(tmp_path),
        **dict.fromkeys(cloud_search.CREDENTIALS, "dummy-key"),
    }
    assert cloud_search.main(env) == 1
    text = (tmp_path / "result.json").read_text()
    assert "secret-in-authenticated-request-url" not in text
    assert json.loads(text)["reason"] == "search_or_safe_export_failed"
