"""Coverage, vocabulary, cross-hunt memory, keeper ordering and the crawl."""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from lit_search.cli import hunt as hunt_cli
from lit_search.cli.main import main
from lit_search.core.coverage import SourceCoverage
from lit_search.core.paper import Paper
from lit_search.hunt import insight
from lit_search.hunt.memory import JudgmentMemory, describe
from lit_search.hunt.session import HuntSession
from lit_search.runners.explore_runner import ExploreOutcome


def _paper(title: str, doi: str, abstract: str = "") -> Paper:
    return Paper(
        title=title,
        abstract=abstract,
        authors=[],
        source=None,
        publication_date=dt.date(2025, 1, 1),
        doi=doi,
    )


def _move(
    session: HuntSession, kind: str, text: str, steps: list[SourceCoverage], *papers: Paper
) -> list[str]:
    return session.absorb(
        kind=kind,
        key=f"{kind}|{text}",
        args={"text": text} if kind != "search" else {"query": text},
        started=time.time(),
        steps=steps,
        papers=[(p, f"{kind}#{i}", 1.0, []) for i, p in enumerate(papers, start=1)],
    ).new


def test_provider_of_names_the_service() -> None:
    assert insight.provider_of("snowball:forward:openalex-fulltext:T", "snowball") == "openalex"
    assert (
        insight.provider_of("snowball:backward:arxiv-source:T", "snowball")
        == "arxiv (LaTeX source)"
    )
    assert insight.provider_of("snowball:forward:Some title", "snowball") == "semantic_scholar"
    assert insight.provider_of("related:recommendations", "related") == "semantic_scholar"
    assert insight.provider_of("arxiv", "find") == "arxiv"
    assert insight.provider_of("add:h_1234", "add") == "handles"


def test_coverage_and_vocabulary(tmp_path: Path) -> None:
    session = HuntSession.start("linear rnn state tracking", profile="prior-art", root=tmp_path)
    abstract = "negative eigenvalues let householder products track permutations"
    _move(
        session,
        "find",
        "parity linear rnn",
        [
            SourceCoverage.complete("arxiv", 2, reported_total=2),
            SourceCoverage.failed("semantic_scholar", error="429", error_type="HTTPError"),
        ],
        _paper("Householder products for permutations", "10.1/a", abstract),
        _paper("Eigenvalues beyond the unit interval", "10.1/b", abstract),
    )
    _move(
        session,
        "search",
        "ti[state space]",
        [SourceCoverage.query_unsupported("dblp", detail="no abstracts")],
        _paper("Unrelated coffee paper", "10.1/c", "espresso extraction grind size"),
    )
    session.mark(["c1", "c2"], "keep")
    rows = {r["provider"]: r for r in insight.coverage_rows(session)}
    assert rows["arxiv"]["answered"] == 1
    assert rows["arxiv"]["papers"] == 2
    assert rows["semantic_scholar"]["failed"] == 1
    assert rows["dblp"]["not_run"] == 1
    phrasing = insight.phrasings(session)[0]
    assert (phrasing.text, phrasing.kept_first, phrasing.kept_reached) == (
        "parity linear rnn",
        2,
        2,
    )
    terms = insight.unexplored_terms(session)
    assert "householder" in " ".join(terms)
    assert not any("parity" in t or "rnn" in t for t in terms)


def _hunt_with_judgment(
    root: Path, question: str, title: str, doi: str, mark: str, note: str
) -> str:
    session = HuntSession.start(question, profile="prior-art", root=root)
    _move(
        session,
        "find",
        question,
        [SourceCoverage.complete("arxiv", 1, reported_total=1)],
        _paper(title, doi),
    )
    session.mark(["c1"], mark, note)
    return str(session.data["id"])


def test_memory_reports_earlier_judgments_but_not_the_current_hunt(tmp_path: Path) -> None:
    old = _hunt_with_judgment(
        tmp_path, "q one", "A paper seen twice across hunts", "10.1/x", "keep", "same mechanism"
    )
    now = _hunt_with_judgment(
        tmp_path, "q two", "A paper seen twice across hunts", "10.1/x", "drop", ""
    )
    memory = JudgmentMemory(tmp_path)
    paper = _paper("A paper seen twice across hunts", "10.1/x")
    judged = memory.lookup(paper, exclude_hunt=now)
    assert [(j.hunt, j.mark) for j in judged] == [(old, "keep")]
    assert describe(judged) == f'kept in {old} ("same mechanism")'
    by_title = memory.lookup(
        _paper("A Paper Seen Twice Across Hunts", "10.9/other"), exclude_hunt=now
    )
    assert [j.hunt for j in by_title] == [old]
    assert (tmp_path / "judgments.json").exists()
    assert JudgmentMemory(tmp_path).lookup(paper, exclude_hunt=now) == judged


def test_keeper_order_puts_like_papers_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hunt_cli, "relevance_scores", lambda q, docs: None)
    session = HuntSession.start("q", profile="prior-art", root=tmp_path)
    _move(
        session,
        "find",
        "q",
        [SourceCoverage.complete("arxiv", 3, reported_total=3)],
        _paper(
            "Householder reflections for state tracking", "10.1/k", "householder state tracking"
        ),
        _paper("Espresso extraction study", "10.1/e", "coffee grind"),
        _paper("State tracking with householder products", "10.1/l", "householder products state"),
    )
    session.mark(["c1"], "keep")
    unjudged = [c for c in session.ranked() if c.mark is None]
    ordered, label = hunt_cli._keeper_order(session, unjudged)
    assert [c.cid for c in ordered] == ["c3", "c2"]
    assert "word overlap" in label


def test_crawl_goes_level_by_level_and_records_itself(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hunt_cli, "relevance_scores", lambda q, docs: None)
    graph = {
        "Seed paper of the crawl": ["Level one paper alpha", "Level one paper beta"],
        "Level one paper alpha": ["Level two paper gamma"],
        "Level one paper beta": [],
    }
    dois = {
        t: f"10.1/{i}"
        for i, t in enumerate(
            ["Seed paper of the crawl", *(t for ts in graph.values() for t in ts)]
        )
    }

    def fake_open(seeds, **kwargs):  # type: ignore[no-untyped-def]
        outcome = ExploreOutcome()
        for seed in seeds:
            outcome.steps.append(
                SourceCoverage.complete(
                    f"snowball:forward:openalex-fulltext:{seed.title[:40]}", 1, reported_total=1
                )
            )
            for title in graph.get(seed.title, []):
                paper = _paper(title, dois[title])
                outcome.papers.append(paper)
                outcome.routes[id(paper)] = [f"cites {seed.label} [fulltext]"]
        return outcome

    with patch.dict("os.environ", {"LIT_SEARCH_HOME": str(tmp_path)}):
        assert main(["hunt", "start", "q", "--profile", "prior-art", "--json"]) == 0
        hunt_id = json.loads(capsys.readouterr().out)["hunt"]
        session = HuntSession.load(hunt_id, root=tmp_path / "hunts")
        _move(
            session,
            "find",
            "q",
            [SourceCoverage.complete("arxiv", 1, reported_total=1)],
            _paper("Seed paper of the crawl", dois["Seed paper of the crawl"]),
        )
        session.mark(["c1"], "keep")
        with patch("lit_search.cli.hunt.run_open_snowball", side_effect=fake_open):
            assert (
                main(["hunt", "crawl", hunt_id, "--via", "open", "--forward", "--depth", "3"]) == 0
            )
    out = capsys.readouterr().out
    assert "crawl 1 level 1: 1 seeds, 1 moves, +2 new" in out
    assert "crawl 1 level 2: 2 seeds, 2 moves, +1 new" in out
    assert "crawl 1 stopped: no new papers" in out
    session = HuntSession.load(hunt_id, root=tmp_path / "hunts")
    crawl = session.data["crawls"][0]
    assert crawl["depth_reached"] == 3
    assert [level["new"] for level in crawl["levels"]] == [2, 1, 0]
    assert any("crawl from c1" in line for line in insight.coverage_lines(session))
