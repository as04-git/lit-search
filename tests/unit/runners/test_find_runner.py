"""Tests for the instant relevance search runner and the ``find`` command."""

from __future__ import annotations

import datetime as dt
import threading
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import requests

from lit_search.cli.main import main
from lit_search.core.coverage import CoverageStatus, CoveredPapers, SourceCoverage
from lit_search.core.paper import Paper
from lit_search.exceptions import InvalidParameterError
from lit_search.runners import find_runner
from lit_search.runners.find_runner import (
    FindRequest,
    _arxiv_query,
    _fuse,
    _ranked_coverage,
    run_find,
)


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch) -> None:
    """Keep cooldown state out of the real cache."""
    monkeypatch.setenv("LIT_SEARCH_HOME", str(tmp_path))


def _paper(title: str, doi: str | None = None, year: int = 2025) -> Paper:
    return Paper(
        title=title,
        abstract="",
        authors=[],
        source=None,
        publication_date=dt.date(year, 1, 1),
        doi=doi,
    )


def _ok(source: str, *papers: Paper) -> CoveredPapers:
    return CoveredPapers(
        papers=tuple(papers),
        coverage=SourceCoverage.capped(source, len(papers), cap=25, reported_total=1000),
    )


class TestArxivQuery:
    def test_terms_are_anded_and_stopwords_dropped(self) -> None:
        assert _arxiv_query("state tracking in linear attention") == (
            "all:state AND all:tracking AND all:linear AND all:attention"
        )

    def test_quoted_spans_stay_phrases(self) -> None:
        assert _arxiv_query('"delta rule" state') == 'all:"delta rule" AND (all:state)'

    def test_no_terms_is_an_error(self) -> None:
        with pytest.raises(InvalidParameterError):
            _arxiv_query("the of and")


_ARXIV_STUB: Any = SimpleNamespace(name="arxiv")


class TestArxivStrictZero:
    """A long free-text query AND-ed term by term matches nothing on arXiv.

    Seen 2026-10-18: three resolvers' 8-10 word queries all came back from
    arXiv as ``ok, all 0 matches`` while OpenAlex found the papers, and the
    same query returns 0 from arXiv when it is not rate-limited.  The zero was
    the strict AND, not a missing literature.
    """

    LONG = "state tracking linear RNN negative eigenvalues Householder language model"

    @staticmethod
    def _fake(relaxed_papers: list[Paper]):
        calls: list[str] = []

        def fake_request(connector, query, request):
            calls.append(query)
            if len(calls) == 1:
                return [], 0, 0
            return list(relaxed_papers), len(relaxed_papers), 5000

        return fake_request, calls

    def test_strict_zero_falls_back_to_relaxed_matches(self) -> None:
        hit = _paper("Householder state tracking in linear RNNs")
        hit.abstract = "negative eigenvalues let a linear RNN language model track state"
        fake, calls = self._fake([hit])
        with patch.object(find_runner, "_arxiv_request", fake):
            result = find_runner._search_arxiv(_ARXIV_STUB, FindRequest(self.LONG))
        assert len(calls) == 2
        assert [p.title for p in result.papers] == [hit.title]
        assert result.coverage.status is CoverageStatus.PARTIAL
        assert "relaxed" in (result.coverage.detail or "")

    def test_relaxed_hits_must_cover_half_the_terms(self) -> None:
        on_topic = _paper("Householder state tracking in linear RNNs")
        on_topic.abstract = "negative eigenvalues let a linear RNN language model track state"
        off_topic = _paper("A Yiddish language model and benchmark")
        fake, _ = self._fake([off_topic, on_topic])
        with patch.object(find_runner, "_arxiv_request", fake):
            result = find_runner._search_arxiv(_ARXIV_STUB, FindRequest(self.LONG))
        assert [p.title for p in result.papers] == [on_topic.title]

    def test_short_query_zero_stays_a_complete_zero(self) -> None:
        fake, calls = self._fake([_paper("anything at all here")])
        with patch.object(find_runner, "_arxiv_request", fake):
            result = find_runner._search_arxiv(_ARXIV_STUB, FindRequest("Householder tracking"))
        assert len(calls) == 1
        assert result.papers == ()
        assert result.coverage.status is CoverageStatus.COMPLETE


class TestRankedCoverage:
    def test_full_page_is_capped_with_total(self) -> None:
        coverage = _ranked_coverage("s", retrieved=25, seen=25, requested=25, total=900)
        assert coverage.status is CoverageStatus.CAPPED
        assert coverage.reported_total == 900

    def test_short_page_matching_total_is_complete(self) -> None:
        coverage = _ranked_coverage("s", retrieved=7, seen=7, requested=25, total=7)
        assert coverage.status is CoverageStatus.COMPLETE

    def test_unparsed_record_is_partial(self) -> None:
        coverage = _ranked_coverage("s", retrieved=6, seen=7, requested=25, total=7)
        assert coverage.status is CoverageStatus.PARTIAL
        assert "could not be parsed" in (coverage.detail or "")


class TestFuse:
    def test_agreement_across_sources_ranks_first(self) -> None:
        shared_a = _paper("A shared paper about state tracking", doi="10.1/shared")
        shared_b = _paper("A shared paper about state tracking", doi="10.1/shared")
        solo = _paper("A solo paper that only one source has", doi="10.1/solo")
        papers, ranks, _ = _fuse({"x": (solo, shared_a), "y": (shared_b,)})
        assert papers[0].doi == "10.1/shared"
        assert ranks[id(papers[0])] == {"x": 2, "y": 1}

    def test_same_title_without_identifiers_merges(self) -> None:
        with_doi = _paper("Length-Independent State Tracking", doi="10.48550/arXiv.2609.1")
        bare = _paper("Length-independent state tracking")
        papers, _, title_merges = _fuse({"x": (with_doi,), "y": (bare,)})
        assert len(papers) == 1
        assert title_merges == 1

    def test_same_title_with_different_dois_stays_separate(self) -> None:
        preprint = _paper("A paper with two versions here", doi="10.48550/arXiv.1")
        journal = _paper("A paper with two versions here", doi="10.1000/journal.1")
        papers, _, title_merges = _fuse({"x": (preprint,), "y": (journal,)})
        assert len(papers) == 2
        assert title_merges == 0


class TestRunFind:
    def _run(self, searchers: dict[str, Any], deadline: float = 5.0) -> find_runner.FindOutcome:
        connectors = {name: object() for name in searchers}
        with patch.dict(find_runner._SEARCHERS, searchers, clear=True):
            return run_find(FindRequest("state tracking"), connectors, deadline_seconds=deadline)

    def test_network_failure_is_reported_and_others_answer(self) -> None:
        def broken(_connector: Any, _request: FindRequest) -> CoveredPapers:
            raise requests.ConnectionError("down")

        outcome = self._run(
            {
                "good": lambda c, r: _ok("good", _paper("Good paper title here", "10.1/g")),
                "bad": broken,
            }
        )
        statuses = {c.source: c.status for c in outcome.result.coverage}
        assert statuses == {"good": CoverageStatus.CAPPED, "bad": CoverageStatus.FAILED}
        assert len(outcome.result.papers) == 1
        assert outcome.tracebacks == {}

    def test_unexpected_error_keeps_its_traceback(self) -> None:
        def buggy(_connector: Any, _request: FindRequest) -> CoveredPapers:
            raise KeyError("missing field")

        outcome = self._run({"buggy": buggy})
        coverage = outcome.result.coverage.get("buggy")
        assert coverage is not None
        assert coverage.status is CoverageStatus.FAILED
        assert "unexpected error (bug)" in (coverage.error or "")
        assert "KeyError" in outcome.tracebacks["buggy"]

    def test_source_past_deadline_is_failed_not_awaited(self) -> None:
        release = threading.Event()

        def slow(_connector: Any, _request: FindRequest) -> CoveredPapers:
            release.wait(5)
            return _ok("slow")

        try:
            outcome = self._run(
                {
                    "slow": slow,
                    "fast": lambda c, r: _ok("fast", _paper("Fast paper title", "10.1/f")),
                },
                deadline=0.2,
            )
        finally:
            release.set()
        coverage = outcome.result.coverage.get("slow")
        assert coverage is not None
        assert coverage.status is CoverageStatus.FAILED
        assert "deadline" in (coverage.error or "")
        assert len(outcome.result.papers) == 1

    def test_empty_query_rejected(self) -> None:
        with pytest.raises(InvalidParameterError):
            run_find(FindRequest("   "), {})


class TestFindCommand:
    def _outcome(self, *coverages: SourceCoverage) -> find_runner.FindOutcome:
        connectors = {c.source: object() for c in coverages}
        searchers = {
            c.source: (lambda cov: lambda _c, _r: CoveredPapers(papers=(), coverage=cov))(c)
            for c in coverages
        }
        if any(c.status is not CoverageStatus.FAILED for c in coverages):
            first = next(c for c in coverages if c.status is not CoverageStatus.FAILED)
            searchers[first.source] = lambda _c, _r: _ok(
                first.source, _paper("Some paper title", "10.1/p")
            )
        with patch.dict(find_runner._SEARCHERS, searchers, clear=True):
            return run_find(FindRequest("q"), connectors)

    def _main(self, outcome: find_runner.FindOutcome, *extra: str) -> int:
        with (
            patch("lit_search.cli.find.build_find_connectors", return_value={}),
            patch("lit_search.cli.find.run_find", return_value=outcome),
        ):
            return main(["find", "state", "tracking", *extra])

    def test_failed_source_still_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        outcome = self._outcome(
            SourceCoverage.capped("arxiv", 1, cap=25, reported_total=40),
            SourceCoverage.failed("openaire", error="HTTP 503", error_type="HTTPError"),
        )
        assert self._main(outcome) == 0
        out = capsys.readouterr().out
        assert "FAIL openaire" in out
        assert "fell short (openaire)" in out

    def test_strict_turns_shortfall_into_exit_three(self) -> None:
        outcome = self._outcome(
            SourceCoverage.capped("arxiv", 1, cap=25, reported_total=40),
            SourceCoverage.failed("openaire", error="HTTP 503", error_type="HTTPError"),
        )
        assert self._main(outcome, "--strict") == 3

    def test_every_source_failed_exits_one(self) -> None:
        outcome = self._outcome(
            SourceCoverage.failed("arxiv", error="down", error_type="ConnectionError"),
        )
        assert self._main(outcome) == 1


class TestReviewRegressions:
    """Defects found in the 2026-10-10 Codex review."""

    def test_full_page_with_parse_loss_is_a_shortfall(self) -> None:
        coverage = _ranked_coverage("s", retrieved=24, seen=25, requested=25, total=100)
        assert coverage.status is CoverageStatus.PARTIAL
        assert "could not be parsed" in (coverage.detail or "")

    def test_full_page_equal_to_total_is_complete(self) -> None:
        coverage = _ranked_coverage("s", retrieved=25, seen=25, requested=25, total=25)
        assert coverage.status is CoverageStatus.COMPLETE

    def test_title_match_with_conflicting_arxiv_ids_does_not_crash(self) -> None:
        from lit_search.core.citation import IdentifierKind, PaperIdentifier

        left = _paper("A paper title that is long enough")
        right = _paper("A paper title that is long enough")
        left.identifiers = [PaperIdentifier(IdentifierKind.ARXIV, "2601.00001")]
        right.identifiers = [PaperIdentifier(IdentifierKind.ARXIV, "2601.00002")]
        papers, _, title_merges = _fuse({"x": (left,), "y": (right,)})
        assert len(papers) == 2
        assert title_merges == 0

    def test_unknown_year_does_not_title_merge(self) -> None:
        dated = _paper("Some long distinctive paper title")
        undated = _paper("Some long distinctive paper title")
        undated.publication_date = None
        papers, _, _ = _fuse({"x": (dated,), "y": (undated,)})
        assert len(papers) == 2

    def test_late_source_does_not_hold_the_process_open(self) -> None:
        import subprocess
        import sys

        script = (
            "import time\n"
            "from unittest.mock import patch\n"
            "from lit_search.runners import find_runner\n"
            "from lit_search.runners.find_runner import FindRequest, run_find\n"
            "def slow(c, r):\n"
            "    time.sleep(30)\n"
            "with patch.dict(find_runner._SEARCHERS, {'slow': slow}, clear=True):\n"
            "    run_find(FindRequest('q'), {'slow': object()}, deadline_seconds=0.2)\n"
        )
        started = time.monotonic()
        subprocess.run([sys.executable, "-c", script], check=True, timeout=20)
        assert time.monotonic() - started < 10


class TestTermCoverage:
    def test_off_topic_top_hit_sinks_below_on_topic_paper(self) -> None:
        off = _paper("Search for scalar top quarks in charm jets", doi="10.1/off")
        on = _paper("Diagonal state space models fail at parity", doi="10.1/on")
        papers, _, _ = _fuse(
            {"x": (off, on)}, find_runner.query_terms("diagonal state space parity")
        )
        assert papers[0].doi == "10.1/on"

    def test_terms_drop_stopwords_and_stem(self) -> None:
        assert find_runner.query_terms("the tracking of states") == ["tracki", "states"]

    def test_single_source_self_deposit_sinks(self) -> None:
        deposit = _paper("Diagonal state space parity framework", doi="10.5281/zenodo.123")
        paper = _paper("Diagonal state space models and parity", doi="10.1/real")
        papers, _, _ = _fuse(
            {"x": (deposit, paper)}, find_runner.query_terms("diagonal state space parity")
        )
        assert papers[0].doi == "10.1/real"


class TestCooldown:
    def test_refused_source_is_skipped_on_the_next_search(self) -> None:
        def refused(_connector: Any, _request: FindRequest) -> CoveredPapers:
            raise requests.HTTPError("429 Client Error: Too Many Requests")

        calls = []

        def counting(_connector: Any, _request: FindRequest) -> CoveredPapers:
            calls.append(1)
            return _ok("arxiv")

        with patch.dict(find_runner._SEARCHERS, {"arxiv": refused}, clear=True):
            first = run_find(FindRequest("q"), {"arxiv": object()})
        with patch.dict(find_runner._SEARCHERS, {"arxiv": counting}, clear=True):
            second = run_find(FindRequest("q"), {"arxiv": object()})

        assert first.result.coverage.get("arxiv").status is CoverageStatus.FAILED
        skipped = second.result.coverage.get("arxiv")
        assert skipped.status is CoverageStatus.SKIPPED_BY_POLICY
        assert "cooling down" in (skipped.detail or "")
        assert calls == []

    def test_ordinary_failure_starts_no_cooldown(self) -> None:
        def broken(_connector: Any, _request: FindRequest) -> CoveredPapers:
            raise requests.ConnectionError("connection reset")

        with patch.dict(find_runner._SEARCHERS, {"arxiv": broken}, clear=True):
            run_find(FindRequest("q"), {"arxiv": object()})
        with patch.dict(find_runner._SEARCHERS, {"arxiv": lambda c, r: _ok("arxiv")}, clear=True):
            second = run_find(FindRequest("q"), {"arxiv": object()})
        assert second.result.coverage.get("arxiv").status is CoverageStatus.CAPPED

    def test_semantic_scholar_load_shedding_starts_no_cooldown(self) -> None:
        def refused(_connector: Any, _request: FindRequest) -> CoveredPapers:
            raise requests.HTTPError("429 Client Error: Too Many Requests")

        with patch.dict(find_runner._SEARCHERS, {"semantic_scholar": refused}, clear=True):
            run_find(FindRequest("q"), {"semantic_scholar": object()})
        ok = {"semantic_scholar": lambda c, r: _ok("semantic_scholar")}
        with patch.dict(find_runner._SEARCHERS, ok, clear=True):
            second = run_find(FindRequest("q"), {"semantic_scholar": object()})
        assert second.result.coverage.get("semantic_scholar").status is CoverageStatus.CAPPED


class TestHuggingFace:
    def test_item_becomes_arxiv_keyed_paper(self) -> None:
        item = {
            "paper": {
                "id": "2201.02177",
                "title": "Grokking: Generalization\n  Beyond Overfitting",
                "summary": "We study generalization.",
                "publishedAt": "2022-01-06T18:43:37.000Z",
                "authors": [{"name": "Alethea Power"}],
            }
        }
        paper = find_runner._parse_hf_item(item)
        assert paper is not None
        assert paper.title == "Grokking: Generalization Beyond Overfitting"
        assert paper.doi == "10.48550/arXiv.2201.02177"
        assert paper.publication_date == dt.date(2022, 1, 6)

    def test_semantic_hit_is_not_penalised_for_missing_query_words(self) -> None:
        semantic = _paper("Explaining delayed generalization via circuits", doi="10.1/sem")
        lexical = _paper("Grokking parity small datasets", doi="10.1/lex")
        papers, _, _ = _fuse(
            {"huggingface": (semantic,), "x": (lexical,)},
            find_runner.query_terms("grokking parity small datasets"),
        )
        assert papers[0].doi == "10.1/sem"

    def test_date_filter_is_reported_unsupported(self) -> None:
        client = find_runner.HuggingFacePapers()
        result = find_runner._search_huggingface(
            client, FindRequest("q", since=dt.date(2025, 1, 1))
        )
        assert result.coverage.status is CoverageStatus.QUERY_UNSUPPORTED
