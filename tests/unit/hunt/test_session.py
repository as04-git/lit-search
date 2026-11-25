"""Tests for hunt sessions and the ``hunt`` command."""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from lit_search.cli.main import main
from lit_search.core.coverage import CoveredPapers, SourceCoverage
from lit_search.core.paper import Paper
from lit_search.exceptions import InvalidParameterError
from lit_search.hunt.session import HuntSession
from lit_search.runners import find_runner


def _paper(title: str, doi: str | None = None, year: int = 2025) -> Paper:
    return Paper(
        title=title,
        abstract="",
        authors=[],
        source=None,
        publication_date=dt.date(year, 1, 1),
        doi=doi,
    )


def _absorb(session: HuntSession, key: str, *papers: Paper) -> list[str]:
    record = session.absorb(
        kind="find",
        key=key,
        args={"text": key},
        started=time.time(),
        steps=[SourceCoverage.complete("arxiv", len(papers), reported_total=len(papers))],
        papers=[(p, f"find#{i}", 1.0 / i, []) for i, p in enumerate(papers, start=1)],
    )
    return record.new


class TestSession:
    def test_same_paper_from_two_moves_is_one_candidate(self, tmp_path: Path) -> None:
        session = HuntSession.start("q", profile="prior-art", root=tmp_path)
        first = _absorb(session, "a", _paper("Shared paper title here", "10.1/x"))
        second = _absorb(session, "b", _paper("Shared paper title here", "10.1/x"))
        assert first == ["c1"]
        assert second == []
        assert len(session.candidates["c1"].routes) == 2

    def test_repeated_move_is_detected(self, tmp_path: Path) -> None:
        session = HuntSession.start("q", profile="prior-art", root=tmp_path)
        _absorb(session, "same key", _paper("Some paper title here", "10.1/a"))
        assert session.previous_move("same key") is not None
        assert session.previous_move("other key") is None

    def test_kept_candidates_rank_first(self, tmp_path: Path) -> None:
        session = HuntSession.start("q", profile="prior-art", root=tmp_path)
        _absorb(
            session,
            "a",
            _paper("Strong first hit title", "10.1/a"),
            _paper("Weaker later hit", "10.1/b"),
        )
        session.mark(["c2"], "keep", "on target")
        assert [c.cid for c in session.ranked()] == ["c2", "c1"]
        session.mark(["c1"], "drop")
        assert [c.cid for c in session.ranked()] == ["c2"]

    def test_state_survives_reload(self, tmp_path: Path) -> None:
        session = HuntSession.start("q", profile="frontier", root=tmp_path)
        _absorb(session, "a", _paper("Persisted paper title", "10.1/p"))
        session.mark(["c1"], "keep", "why")
        loaded = HuntSession.load(session.data["id"][:20], root=tmp_path)
        assert loaded.candidates["c1"].note == "why"
        assert loaded.data["moves"][0]["new"] == ["c1"]

    def test_two_writers_do_not_lose_each_others_changes(self, tmp_path: Path) -> None:
        session = HuntSession.start("q", profile="prior-art", root=tmp_path)
        _absorb(
            session,
            "a",
            _paper("First paper title here", "10.1/a"),
            _paper("Second paper title here", "10.1/b"),
        )
        hunt_id = session.data["id"]
        agent = HuntSession.load(hunt_id, tmp_path)
        background = HuntSession.load(hunt_id, tmp_path)
        _absorb(background, "b", _paper("Third paper title here", "10.1/c"))
        agent.mark(["c1"], "keep", "on target")
        final = HuntSession.load(hunt_id, tmp_path)
        assert final.candidates["c1"].mark == "keep"
        assert "c3" in final.candidates
        assert len(final.data["moves"]) == 2

    def test_unknown_candidate_is_an_error(self, tmp_path: Path) -> None:
        session = HuntSession.start("q", profile="orient", root=tmp_path)
        with pytest.raises(InvalidParameterError):
            session.resolve("c99")

    def test_frontier_prefers_recent(self, tmp_path: Path) -> None:
        session = HuntSession.start("q", profile="frontier", root=tmp_path)
        recent = dt.date.today().year
        _absorb(
            session,
            "a",
            _paper("An old paper title here", "10.1/o", 2015),
            _paper("A new paper title here", "10.1/n", recent),
        )
        assert session.ranked()[0].paper.doi == "10.1/n"


class TestHuntCommand:
    def _run(self, tmp_path: Path, *argv: str) -> int:
        with patch.dict("os.environ", {"LIT_SEARCH_HOME": str(tmp_path)}):
            return main(list(argv))

    def test_end_to_end_with_budget_refusal_and_report(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert (
            self._run(
                tmp_path, "hunt", "start", "my", "question", "--profile", "prior-art", "--json"
            )
            == 0
        )
        hunt_id = json.loads(capsys.readouterr().out)["hunt"]

        fake = {
            "arxiv": lambda c, r: CoveredPapers(
                papers=(_paper("A relevant paper title", "10.1/r"),),
                coverage=SourceCoverage.complete("arxiv", 1, reported_total=1),
            )
        }
        with (
            patch.dict(find_runner._SEARCHERS, fake, clear=True),
            patch("lit_search.cli.hunt.build_find_connectors", return_value={"arxiv": object()}),
        ):
            assert (
                self._run(tmp_path, "hunt", "find", hunt_id, "relevant", "paper", "-d", "arxiv")
                == 0
            )
            out = capsys.readouterr().out
            assert "+1 new" in out
            assert "c1" in out
            assert (
                self._run(tmp_path, "hunt", "find", hunt_id, "relevant", "paper", "-d", "arxiv")
                == 0
            )
            assert "already ran as move 1" in capsys.readouterr().out

            session = HuntSession.load(hunt_id, root=tmp_path / "hunts")
            session.data["deadline"] = time.time() - 5
            session.save()
            assert self._run(tmp_path, "hunt", "find", hunt_id, "another", "-d", "arxiv") == 3
            assert "budget spent" in capsys.readouterr().out

        assert (
            self._run(tmp_path, "hunt", "mark", hunt_id, "c1", "--keep", "--note", "on target") == 0
        )
        assert self._run(tmp_path, "hunt", "report", hunt_id) == 0
        report = capsys.readouterr().out
        assert "A relevant paper title" in report
        assert "why: on target" in report
        assert "relevance sources never queried" in report
        assert (tmp_path / "hunts" / hunt_id / "report.md").exists()


class TestRetry:
    def test_failed_source_is_listed_rerun_and_resolved(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def run(*argv: str) -> int:
            with patch.dict("os.environ", {"LIT_SEARCH_HOME": str(tmp_path)}):
                return main(list(argv))

        assert run("hunt", "start", "q", "--profile", "prior-art", "--json") == 0
        hunt_id = json.loads(capsys.readouterr().out)["hunt"]
        refused = {
            "arxiv": lambda c, r: CoveredPapers(
                papers=(), coverage=SourceCoverage.failed("arxiv", error="HTTP 429 Too Many")
            )
        }
        served = {
            "arxiv": lambda c, r: CoveredPapers(
                papers=(_paper("A paper found on retry", "10.1/r"),),
                coverage=SourceCoverage.complete("arxiv", 1, reported_total=1),
            )
        }
        connectors = patch(
            "lit_search.cli.hunt.build_find_connectors", return_value={"arxiv": object()}
        )
        with patch.dict(find_runner._SEARCHERS, refused, clear=True), connectors:
            run("hunt", "find", hunt_id, "some", "words", "-d", "arxiv")
        capsys.readouterr()

        assert run("hunt", "retry", hunt_id) == 0
        listing = capsys.readouterr().out
        assert "move 1 find" in listing
        assert "429" in listing

        with patch.dict(find_runner._SEARCHERS, served, clear=True), connectors:
            assert run("hunt", "retry", hunt_id, "--all") == 3
            assert "0 fixed, 1 still failing" in capsys.readouterr().out
            cooldowns = tmp_path / "find-cooldown.json"
            cooldowns.write_text("{}", encoding="utf-8")
            assert run("hunt", "retry", hunt_id, "--all") == 0
        assert "1 fixed, 0 still failing" in capsys.readouterr().out

        session = HuntSession.load(hunt_id, root=tmp_path / "hunts")
        assert session.data["moves"][0]["resolved_by"] == 3
        assert session.data["moves"][2]["args"]["retry_of"] == 1
        assert session.failed_moves() == []
        assert session.shortfalls() == []
        assert session.candidates["c1"].paper.title == "A paper found on retry"


class TestJudgmentTiers:
    def test_core_outranks_keep(self, tmp_path: Path) -> None:
        session = HuntSession.start("q", profile="prior-art", root=tmp_path)
        _absorb(
            session,
            "a",
            _paper("Strong first hit title", "10.1/a"),
            _paper("Weaker second hit title", "10.1/b"),
            _paper("Weakest third hit title", "10.1/c"),
        )
        session.mark(["c1"], "keep")
        session.mark(["c3"], "core")
        assert [c.cid for c in session.ranked()][:2] == ["c3", "c1"]

    def test_report_waits_for_80_percent_of_budget(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with patch.dict("os.environ", {"LIT_SEARCH_HOME": str(tmp_path)}):
            assert main(["hunt", "start", "q", "--profile", "frontier", "--json"]) == 0
            hunt_id = json.loads(capsys.readouterr().out)["hunt"]
            assert main(["hunt", "report", hunt_id]) == 3
            assert "the report opens at 80% of the budget" in capsys.readouterr().out
            session = HuntSession.load(hunt_id, tmp_path / "hunts")
            session.data["started"] -= 0.85 * session.data["minutes"] * 60
            session.save()
            assert main(["hunt", "report", hunt_id]) == 0

    def test_minimum_share_is_set_per_hunt(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with patch.dict("os.environ", {"LIT_SEARCH_HOME": str(tmp_path)}):
            args = ["hunt", "start", "q", "--profile", "prior-art", "--min-share", "50", "--json"]
            assert main(args) == 0
            hunt_id = json.loads(capsys.readouterr().out)["hunt"]
            session = HuntSession.load(hunt_id, tmp_path / "hunts")
            session.data["started"] -= 0.4 * session.data["minutes"] * 60
            session.save()
            assert main(["hunt", "report", hunt_id]) == 3
            assert "opens at 50%" in capsys.readouterr().out
            session = HuntSession.load(hunt_id, tmp_path / "hunts")
            session.data["started"] -= 0.15 * session.data["minutes"] * 60
            session.save()
            assert main(["hunt", "report", hunt_id]) == 0

    def test_known_item_may_report_at_once(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with patch.dict("os.environ", {"LIT_SEARCH_HOME": str(tmp_path)}):
            assert main(["hunt", "start", "q", "--profile", "known-item", "--json"]) == 0
            hunt_id = json.loads(capsys.readouterr().out)["hunt"]
            assert main(["hunt", "report", hunt_id]) == 0

    def test_agent_order_leads_the_ranking(self, tmp_path: Path) -> None:
        session = HuntSession.start("q", profile="prior-art", root=tmp_path)
        _absorb(
            session,
            "a",
            _paper("First paper title here", "10.1/a"),
            _paper("Second paper title here", "10.1/b"),
            _paper("Third paper title here", "10.1/c"),
        )
        session.mark(["c1", "c2", "c3"], "keep")
        session.set_order(["c3", "c1"])
        assert [c.cid for c in session.ranked()] == ["c3", "c1", "c2"]


class TestSection:
    TEXT = (
        "Title\nAbstract\nWe do things.\n1\nIntroduction\nWe propose X.\nContributions: Y.\n"
        "2\nRelated Work\nPrior work did Z.\n3\nMethod\nDetails.\n"
    )

    def test_cuts_numbered_sections(self) -> None:
        from lit_search.cli.hunt import section

        intro = section(self.TEXT, "intro")
        assert intro is not None
        assert "We propose X." in intro
        assert "Prior work" not in intro
        related = section(self.TEXT, "related")
        assert related is not None
        assert "Prior work did Z." in related
        assert "Details" not in related

    def test_missing_heading_is_none(self) -> None:
        from lit_search.cli.hunt import section

        assert section("no headings at all", "conclusion") is None


class TestStrictMode:
    def test_moves_need_plan_note_and_judgment(self, tmp_path: Path) -> None:
        session = HuntSession.start("q", profile="prior-art", root=tmp_path, strict=True)
        assert any("plan" in b for b in session.move_blockers())
        with pytest.raises(InvalidParameterError):
            session.add_note("- only one community", "plan")
        session.add_note("- SE: spec inference\n- PL: contracts\n- FM: model checking", "plan")
        assert session.move_blockers() == []
        _absorb(
            session,
            "a",
            _paper("First new paper title", "10.1/a"),
            _paper("Second new paper title", "10.1/b"),
        )
        blockers = session.move_blockers()
        assert any("note on move 1" in b for b in blockers)
        assert any("c1 c2" in b for b in blockers)
        session.add_note("move 1 found two SE papers; next: FM vocabulary")
        session.mark(["c1"], "keep", "partial overlap")
        session.mark(["c2"], "drop")
        assert session.move_blockers() == []

    def test_report_needs_moves_review_budget_and_final_note(self, tmp_path: Path) -> None:
        session = HuntSession.start("q", profile="prior-art", root=tmp_path, strict=True)
        blockers = " | ".join(session.report_blockers())
        assert "3 `find` moves" in blockers
        assert "snowballing in both directions" in blockers
        assert "snippets" in blockers
        assert "final note" in blockers

    def test_core_papers_must_be_snowballed_both_ways(self, tmp_path: Path) -> None:
        session = HuntSession.start("q", profile="prior-art", root=tmp_path, strict=True)
        _absorb(session, "a", _paper("A paper that does the idea", "10.1/a"))
        session.mark(["c1"], "core", "same mechanism")
        assert session.snowball_queue() == {"forward": ["c1"], "backward": ["c1"]}
        assert any(
            "snowball every core paper both ways: c1" in b for b in session.report_blockers()
        )
        session.data["crawled"] = {"c1": ["forward", "backward"]}
        assert session.snowball_queue() == {"forward": [], "backward": []}
        assert not any("snowball every core" in b for b in session.report_blockers())

    def test_non_strict_session_has_no_blockers(self, tmp_path: Path) -> None:
        session = HuntSession.start("q", profile="prior-art", root=tmp_path)
        assert session.move_blockers() == []
        assert session.report_blockers() == []


class TestHuntWorkspaceAndHandles:
    """Hunts read and write the handles the building-block commands use."""

    def _run(self, tmp_path: Path, *argv: str) -> int:
        with patch.dict("os.environ", {"LIT_SEARCH_HOME": str(tmp_path)}):
            return main(list(argv))

    def _start(self, tmp_path: Path, capsys: pytest.CaptureFixture[str], *extra: str) -> str:
        argv = ("hunt", "start", "q", "--profile", "known-item", "--json", *extra)
        assert self._run(tmp_path, *argv) == 0
        return str(json.loads(capsys.readouterr().out)["hunt"])

    def _find(self, tmp_path: Path, hunt_id: str, *papers: Paper) -> None:
        fake = {
            "arxiv": lambda c, r: CoveredPapers(
                papers=papers,
                coverage=SourceCoverage.complete("arxiv", len(papers), reported_total=len(papers)),
            )
        }
        with (
            patch.dict(find_runner._SEARCHERS, fake, clear=True),
            patch("lit_search.cli.hunt.build_find_connectors", return_value={"arxiv": object()}),
        ):
            assert self._run(tmp_path, "hunt", "find", hunt_id, "anything", "-d", "arxiv") == 0

    def _all_handle(self, tmp_path: Path, hunt_id: str, capsys: pytest.CaptureFixture[str]) -> str:
        assert self._run(tmp_path, "hunt", "report", hunt_id) == 0
        out = capsys.readouterr().out
        return out.split("all candidates handle: ")[1].split()[0]

    def test_report_handle_seeds_and_excludes_later_hunts(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        known = _paper("A paper the first hunt found", "10.1/known")
        fresh = _paper("A paper only the second hunt finds", "10.1/fresh")
        first = self._start(tmp_path, capsys)
        self._find(tmp_path, first, known)
        capsys.readouterr()
        handle = self._all_handle(tmp_path, first, capsys)

        second = self._start(tmp_path, capsys, "--exclude", handle)
        self._find(tmp_path, second, known, fresh)
        out = capsys.readouterr().out
        assert "+1 new" in out
        assert "1 excluded as known" in out
        session = HuntSession.load(second, root=tmp_path / "hunts")
        assert [c.paper.title for c in session.candidates.values()] == [fresh.title]

        third = self._start(tmp_path, capsys)
        assert self._run(tmp_path, "hunt", "add", third, "--from", handle[:10]) == 0
        session = HuntSession.load(third, root=tmp_path / "hunts")
        assert [c.paper.title for c in session.candidates.values()] == [known.title]
        assert session.candidates["c1"].routes[0]["label"] == f"from {handle}"

    def test_exclude_mid_hunt_keeps_existing_candidates(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        known = _paper("A paper the first hunt found", "10.1/known")
        first = self._start(tmp_path, capsys)
        self._find(tmp_path, first, known)
        capsys.readouterr()
        handle = self._all_handle(tmp_path, first, capsys)
        second = self._start(tmp_path, capsys)
        _absorb(HuntSession.load(second, root=tmp_path / "hunts"), "a", known)
        assert self._run(tmp_path, "hunt", "exclude", second, handle) == 0
        assert "1 of 1 papers newly excluded" in capsys.readouterr().out
        session = HuntSession.load(second, root=tmp_path / "hunts")
        assert list(session.candidates) == ["c1"]
        record = session.absorb(
            kind="find",
            key="b",
            args={},
            started=time.time(),
            steps=[],
            papers=[(_paper("A paper the first hunt found", None), "find#1", 1.0, [])],
        )
        assert record.new == []
        assert record.again == ["c1"]

    def test_search_move_adds_boolean_results_with_coverage(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from lit_search.core.coverage import CoverageLedger
        from lit_search.core.search_result import SearchResult

        hunt_id = self._start(tmp_path, capsys)
        papers = [_paper("Model checking with language models", "10.1/mc")]
        result = SearchResult(
            query="abs[model checking]",
            papers=papers,
            coverage=CoverageLedger(
                [
                    SourceCoverage.complete("openalex", 1, reported_total=1),
                    SourceCoverage.failed("dblp", error="boom", error_type="HTTPError"),
                ]
            ),
        )
        with patch("lit_search.engine.Engine.search", return_value=result) as search:
            assert self._run(tmp_path, "hunt", "search", hunt_id, "abs[model checking]") == 0
        assert search.call_args.kwargs["databases"] == ["openalex", "arxiv"]
        out = capsys.readouterr().out
        assert "move 1 search" in out
        assert "1/2 sources answered" in out
        session = HuntSession.load(hunt_id, root=tmp_path / "hunts")
        assert session.candidates["c1"].routes[0]["label"] == "search#1"
        assert [m["number"] for m in session.failed_moves()] == [1]


class TestStepAccounting:
    def test_unsupported_query_is_not_counted_as_answered(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from lit_search.core.coverage import CoverageLedger
        from lit_search.core.search_result import SearchResult

        with patch.dict("os.environ", {"LIT_SEARCH_HOME": str(tmp_path)}):
            assert main(["hunt", "start", "q", "--profile", "known-item", "--json"]) == 0
            hunt_id = json.loads(capsys.readouterr().out)["hunt"]
            result = SearchResult(
                query="[x]",
                papers=[_paper("Some paper title here", "10.1/x")],
                coverage=CoverageLedger(
                    [
                        SourceCoverage.complete("openalex", 1, reported_total=1),
                        SourceCoverage.query_unsupported("dblp", detail="no abstracts"),
                    ]
                ),
            )
            with patch("lit_search.engine.Engine.search", return_value=result):
                assert main(["hunt", "search", hunt_id, "[x]", "-d", "openalex", "-d", "dblp"]) == 0
        out = capsys.readouterr().out
        assert "1/2 sources answered" in out
        assert "unsupported dblp: no abstracts" in out

    def test_search_defaults_follow_the_query_filters(self) -> None:
        from lit_search.cli.hunt import _search_defaults

        assert _search_defaults("abs[state tracking]") == ["openalex", "arxiv"]
        assert _search_defaults("[state tracking]") == [
            "openalex",
            "semantic_scholar",
            "arxiv",
            "dblp",
        ]


class TestOpenSnowball:
    def test_open_routes_carry_a_snowball_when_semantic_scholar_refuses(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from lit_search.runners.explore_runner import ExploreOutcome

        with patch.dict("os.environ", {"LIT_SEARCH_HOME": str(tmp_path)}):
            assert main(["hunt", "start", "q", "--profile", "prior-art", "--json"]) == 0
            hunt_id = json.loads(capsys.readouterr().out)["hunt"]
            session = HuntSession.load(hunt_id, root=tmp_path / "hunts")
            _absorb(
                session, "a", _paper("The seed paper of this hunt", "10.48550/arxiv.2501.00001")
            )

            citer = _paper("A later paper citing the seed", "10.1/c")
            opened = ExploreOutcome(
                papers=[citer],
                steps=[
                    SourceCoverage.partial(
                        "snowball:forward:openalex-fulltext:The seed paper of this hunt",
                        1,
                        detail="x",
                    ),
                    SourceCoverage.complete(
                        "snowball:backward:arxiv-source:The seed paper of this hunt",
                        0,
                        reported_total=0,
                    ),
                ],
                routes={id(citer): ["cites c1 [fulltext]"]},
            )
            refused = ExploreOutcome(
                steps=[SourceCoverage.failed("snowball:seeds", error="429", error_type="HTTPError")]
            )
            with (
                patch("lit_search.cli.hunt.run_open_snowball", return_value=opened),
                patch("lit_search.cli.hunt.run_snowball", return_value=(refused, [])),
                patch("lit_search.cli.hunt.s2_connector"),
            ):
                assert main(["hunt", "snowball", hunt_id, "c1"]) == 0
        out = capsys.readouterr().out
        assert "+1 new" in out
        assert "FAIL snowball:seeds" in out
        session = HuntSession.load(hunt_id, root=tmp_path / "hunts")
        assert session.candidates["c2"].routes[0]["label"] == "snowball: cites c1 [fulltext]"
        assert sorted(session.data["crawled"]["c1"]) == ["backward", "forward"]
