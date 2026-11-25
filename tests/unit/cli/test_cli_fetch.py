"""Tests for ``lit-search fetch``: rendering, exit codes and handle input."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lit_search.cli import fetch as cli_fetch
from lit_search.cli.main import main
from lit_search.cli.output import EXIT_ERROR, EXIT_INCOMPLETE, EXIT_OK, EXIT_USAGE
from lit_search.fetch.http import HttpResponse
from lit_search.fetch.model import Attempt, AttemptResult, FetchOutcome, FetchStatus, FetchTarget
from lit_search.store.lineage import Operation, OperationName
from lit_search.store.record import HandleRecord
from lit_search.store.store import HandleStore

PDF = b"%PDF-1.5\n" + b"y" * 2000 + b"\n%%EOF\n"


class _Transport:
    """Serve arXiv PDFs; 404 for everything else."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, url: str, *, timeout: float) -> HttpResponse:
        self.calls.append(url)
        if url.startswith("https://arxiv.org/pdf/"):
            return HttpResponse(200, url, "application/pdf", PDF)
        return HttpResponse(404, url, "text/html", b"<html></html>")

    def close(self) -> None:
        """Match CurlTransport."""


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch) -> _Transport:
    """Replace the network transport for every fetcher the CLI builds."""
    fake = _Transport()
    monkeypatch.setattr("lit_search.fetch.fetcher.CurlTransport", lambda: fake)
    monkeypatch.delenv("LIT_SEARCH_SCIHUB", raising=False)
    monkeypatch.delenv("LIT_SEARCH_SCIHUB_MIRRORS", raising=False)
    return fake


class TestRender:
    """One line per paper, carrying what was tried."""

    def test_downloaded_line(self) -> None:
        outcome = FetchOutcome(
            FetchTarget(label="2310.06825"),
            FetchStatus.DOWNLOADED,
            path="/p/a.pdf",
            via="arxiv",
            size=1_300_000,
        )
        assert (
            cli_fetch.render_outcome(outcome) == "ok   2310.06825  via arxiv  -> /p/a.pdf (1.2 MB)"
        )

    def test_miss_line_lists_attempts_and_disabled_scihub(self) -> None:
        outcome = FetchOutcome(
            FetchTarget(label="10.1/y", doi="10.1/y"),
            FetchStatus.NOT_FOUND,
            attempts=(
                Attempt("arxiv", None, AttemptResult.SKIPPED, "no arXiv id"),
                Attempt("openalex", "https://api", AttemptResult.MISS, "no OA location"),
                Attempt(
                    "publisher",
                    "https://doi.org/10.1/y",
                    AttemptResult.LOOKUP,
                    "landing page -> citation_pdf_url",
                ),
                Attempt("publisher", "https://pub/y.pdf", AttemptResult.BLOCKED, "HTTP 403"),
                Attempt("scihub", None, AttemptResult.SKIPPED, "disabled"),
            ),
        )
        assert cli_fetch.render_outcome(outcome) == (
            "miss 10.1/y  tried: openalex(no OA location), "
            "publisher(landing page -> citation_pdf_url; HTTP 403), scihub(disabled)"
        )


class TestCommand:
    """Exit codes and output of the command itself."""

    def test_downloads_and_summarises(self, tmp_path: Path, transport: _Transport, capsys) -> None:
        code = main(["fetch", "2310.06825", "-o", str(tmp_path)])
        out = capsys.readouterr().out.splitlines()
        assert code == EXIT_OK
        assert out[0].startswith("ok   2310.06825  via arxiv")
        assert "1 downloaded" in out[-1]
        assert (tmp_path / "arxiv-2310.06825.pdf").read_bytes() == PDF

    def test_miss_exits_zero_but_strict_exits_three(
        self, tmp_path: Path, transport: _Transport, capsys
    ) -> None:
        assert main(["fetch", "10.1/missing", "-o", str(tmp_path)]) == EXIT_OK
        assert "scihub(disabled)" in capsys.readouterr().out
        assert main(["fetch", "10.1/missing", "-o", str(tmp_path), "--strict"]) == EXIT_INCOMPLETE

    def test_json_payload(self, tmp_path: Path, transport: _Transport, capsys) -> None:
        code = main(["fetch", "2310.06825", "10.1/missing", "-o", str(tmp_path), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert code == EXIT_OK
        assert payload["summary"] == {
            "downloaded": 1,
            "already_present": 0,
            "not_found": 1,
            "failed": 0,
            "unresolved": 0,
        }
        assert payload["scihub_enabled"] is False
        statuses = [r["status"] for r in payload["results"]]
        assert statuses == ["downloaded", "not_found"]
        assert payload["results"][1]["attempts"][-1] == {
            "step": "scihub",
            "url": None,
            "result": "skipped",
            "reason": "disabled",
        }

    def test_scihub_env_enables_the_step(
        self, tmp_path: Path, transport: _Transport, capsys, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LIT_SEARCH_SCIHUB", "1")
        monkeypatch.setenv("LIT_SEARCH_SCIHUB_MIRRORS", "https://m.test")
        main(["fetch", "10.1/missing", "-o", str(tmp_path)])
        assert "https://m.test/10.1/missing" in transport.calls
        assert "scihub[m.test](HTTP 404)" in capsys.readouterr().out

    def test_bad_mirror_config_cannot_run(
        self, tmp_path: Path, transport: _Transport, capsys, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LIT_SEARCH_SCIHUB_MIRRORS", "not-a-url")
        assert main(["fetch", "10.1/x", "-o", str(tmp_path)]) == EXIT_ERROR
        assert "not an http" in capsys.readouterr().err

    def test_usage_errors(self, tmp_path: Path, transport: _Transport, capsys) -> None:
        assert main(["fetch", "-o", str(tmp_path)]) == EXIT_USAGE
        assert main(["fetch", "deep learning", "-o", str(tmp_path)]) == EXIT_USAGE
        assert "not a DOI or arXiv id" in capsys.readouterr().err
        assert transport.calls == []

    def test_unknown_handle_cannot_run(self, tmp_path: Path, transport: _Transport, capsys) -> None:
        code = main(
            [
                "fetch",
                "--handle",
                "h_nope",
                "--store-root",
                str(tmp_path / "s"),
                "-o",
                str(tmp_path),
            ]
        )
        assert code == EXIT_ERROR

    def test_handle_papers_with_limit(
        self, tmp_path: Path, transport: _Transport, capsys, make_paper
    ) -> None:
        store = HandleStore(tmp_path / "store")
        papers = (
            make_paper("A", doi="10.48550/arXiv.2401.00001"),
            make_paper("B", doi="10.48550/arXiv.2401.00002"),
            make_paper("C", doi="10.48550/arXiv.2401.00003"),
        )
        handle_id = store.put(
            HandleRecord(
                operation=Operation(name=OperationName.SEARCH, parameters={"query": "[q]"}),
                members=papers,
            )
        )
        code = main(
            [
                "fetch",
                "--handle",
                handle_id,
                "--store-root",
                str(tmp_path / "store"),
                "--limit",
                "2",
                "-o",
                str(tmp_path / "pdfs"),
            ]
        )
        out = capsys.readouterr().out
        assert code == EXIT_OK
        assert out.count("via arxiv") == 2
        assert "fetched first 2 of 3 papers; --limit 0 for all" in out

    def test_default_directory_honours_lit_search_home(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("LIT_SEARCH_HOME", str(tmp_path))
        assert cli_fetch.default_pdf_directory() == tmp_path / "pdfs"
        monkeypatch.delenv("LIT_SEARCH_HOME")
        assert cli_fetch.default_pdf_directory() == Path.home() / ".local/share/lit-search/pdfs"
