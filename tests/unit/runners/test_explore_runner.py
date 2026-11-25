"""Tests for the bounded exploration moves (related, snippets)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from lit_search.connectors.semantic_scholar import SemanticScholarConnector
from lit_search.core.coverage import CoverageStatus
from lit_search.exceptions import InvalidParameterError
from lit_search.runners.explore_runner import (
    run_related,
    run_snippets,
    run_snowball,
    s2_request_id,
)


def _item(paper_id: str, title: str) -> dict[str, Any]:
    return {
        "paperId": paper_id,
        "title": title,
        "externalIds": {"DOI": f"10.9/{paper_id}"},
        "publicationDate": "2025-01-01",
        "authors": [],
    }


def _response(payload: Any) -> MagicMock:
    response = MagicMock()
    response.json.return_value = payload
    return response


class TestRequestIds:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("2605.07755", "ARXIV:2605.07755"),
            ("arXiv:2605.07755v2", "ARXIV:2605.07755v2"),
            ("10.48550/arXiv.2605.07755", "ARXIV:2605.07755"),
            ("10.1038/nature14539", "DOI:10.1038/nature14539"),
            ("doi:10.1/x", "DOI:10.1/x"),
            ("CorpusId:123", "CorpusId:123"),
            ("a" * 40, "a" * 40),
        ],
    )
    def test_forms(self, raw: str, expected: str) -> None:
        assert s2_request_id(raw) == expected

    def test_garbage_rejected(self) -> None:
        with pytest.raises(InvalidParameterError):
            s2_request_id("not an id")


class TestRelated:
    def test_routes_rank_multiply_connected_papers_first(self) -> None:
        connector = SemanticScholarConnector(api_key="k")
        recs = {"recommendedPapers": [_item("r1", "Recommended only paper")]}
        edges = [
            {
                "paperId": "s1",
                "title": "Seed one",
                "referenceCount": 1,
                "citationCount": 0,
                "references": [{"paperId": "shared"}],
                "citations": [],
            },
            {
                "paperId": "s2",
                "title": "Seed two",
                "referenceCount": 1,
                "citationCount": 0,
                "references": [{"paperId": "shared"}],
                "citations": [],
            },
        ]
        metadata = [_item("shared", "Paper both seeds cite")]
        with patch.object(
            connector,
            "_post",
            side_effect=[_response(recs), _response(edges), _response(metadata)],
        ):
            outcome = run_related(connector, ["s1", "s2"], recommendations=5)

        titles = [p.title for p in outcome.papers]
        assert titles[0] == "Paper both seeds cite"
        assert outcome.routes[id(outcome.papers[0])] == ["cited by 2 seed(s)"]
        statuses = {s.source: s.status for s in outcome.steps}
        assert statuses["related:edges"] is CoverageStatus.COMPLETE

    def test_truncated_edges_and_unknown_seed_are_reported(self) -> None:
        connector = SemanticScholarConnector(api_key="k")
        edges = [
            None,
            {
                "paperId": "s2",
                "title": "Famous seed",
                "referenceCount": 0,
                "citationCount": 5000,
                "references": [],
                "citations": [{"paperId": "c1"}],
            },
        ]
        with patch.object(
            connector,
            "_post",
            side_effect=[_response(edges), _response([_item("c1", "A citing paper")])],
        ):
            outcome = run_related(connector, ["missing", "s2"], recommendations=0)

        edge_step = next(s for s in outcome.steps if s.source == "related:edges")
        assert edge_step.status is CoverageStatus.PARTIAL
        assert "not in Semantic Scholar: missing" in (edge_step.detail or "")
        assert "fewer edges than its counts" in (edge_step.detail or "")

    def test_failed_recommendations_do_not_stop_edges(self) -> None:
        connector = SemanticScholarConnector(api_key="k")
        edges = [
            {
                "paperId": "s1",
                "title": "Seed",
                "referenceCount": 1,
                "citationCount": 0,
                "references": [{"paperId": "x"}],
                "citations": [],
            }
        ]
        with patch.object(
            connector,
            "_post",
            side_effect=[
                requests.HTTPError("429 Too Many Requests"),
                _response(edges),
                _response([_item("x", "Referenced paper")]),
            ],
        ):
            outcome = run_related(connector, ["s1"], recommendations=5)

        statuses = {s.source: s.status for s in outcome.steps}
        assert statuses["related:recommendations"] is CoverageStatus.FAILED
        assert [p.title for p in outcome.papers] == ["Referenced paper"]


class TestSnippets:
    def test_papers_carry_their_sentences(self) -> None:
        connector = SemanticScholarConnector(api_key="k")
        hits = {
            "data": [
                {"paper": {"corpusId": 1}, "snippet": {"text": "Diagonal models fail parity."}},
                {"paper": {"corpusId": 1}, "snippet": {"text": "A second sentence."}},
                {"paper": {"corpusId": 2}, "snippet": {"text": "Other paper."}},
            ]
        }
        metadata = [_item("p1", "First paper"), _item("p2", "Second paper")]
        with (
            patch.object(connector, "_get", return_value=_response(hits)),
            patch.object(connector, "_post", return_value=_response(metadata)),
        ):
            outcome = run_snippets(connector, "parity", limit=10)

        assert [p.title for p in outcome.papers] == ["First paper", "Second paper"]
        assert outcome.evidence[id(outcome.papers[0])] == [
            "Diagonal models fail parity.",
            "A second sentence.",
        ]

    def test_search_failure_is_reported(self) -> None:
        connector = SemanticScholarConnector(api_key="k")
        with patch.object(connector, "_get", side_effect=requests.HTTPError("429")):
            outcome = run_snippets(connector, "parity")

        assert outcome.papers == []
        assert outcome.steps[0].status is CoverageStatus.FAILED


class TestNeighbourCap:
    def test_dropped_neighbours_are_reported(self) -> None:
        connector = SemanticScholarConnector(api_key="k")
        edges = [
            {
                "paperId": "s1",
                "title": "Seed",
                "referenceCount": 3,
                "citationCount": 0,
                "references": [{"paperId": "a"}, {"paperId": "b"}, {"paperId": "c"}],
                "citations": [],
            }
        ]
        with patch.object(
            connector,
            "_post",
            side_effect=[_response(edges), _response([_item("a", "Kept neighbour paper")])],
        ):
            outcome = run_related(connector, ["s1"], recommendations=0, max_neighbours=1)

        cap = next(s for s in outcome.steps if s.source == "related:neighbours")
        assert cap.status is CoverageStatus.CAPPED
        assert cap.reported_total == 3
        assert "2 dropped" in (cap.detail or "")


def _seed(paper_id: str, title: str, citations: int = 0, references: int = 0) -> dict[str, Any]:
    return {
        "paperId": paper_id,
        "title": title,
        "citationCount": citations,
        "referenceCount": references,
    }


def _edge(
    paper_id: str, title: str, date: str, contexts: list[str] | None = None
) -> dict[str, Any]:
    return {
        "paper": {
            "paperId": paper_id,
            "title": title,
            "publicationDate": date,
            "year": int(date[:4]),
        },
        "contexts": contexts or [],
        "intents": [],
        "isInfluential": False,
    }


def _page(direction: str, edges: list[dict[str, Any]]) -> MagicMock:
    key = "citingPaper" if direction == "citations" else "citedPaper"
    return _response({"data": [{**e, key: e.pop("paper")} for e in edges]})


class TestSnowball:
    def test_date_window_coupling_rank_and_citation_sentences(self) -> None:
        import datetime

        connector = SemanticScholarConnector(api_key="k")
        seeds = [_seed("s1", "Seed one", citations=3), _seed("s2", "Seed two", citations=1)]
        s1_citers = [
            _edge("old", "Old citer", "2020-05-01"),
            _edge("solo", "Recent citer of one seed", "2026-03-01", ["builds on [1]"]),
            _edge("both", "Recent citer of both seeds", "2026-02-01", ["extends [3]"]),
        ]
        s2_citers = [_edge("both", "Recent citer of both seeds", "2026-02-01")]
        metadata = [
            _item("both", "Recent citer of both seeds"),
            _item("solo", "Recent citer of one seed"),
        ]
        with (
            patch.object(connector, "_post", side_effect=[_response(seeds), _response(metadata)]),
            patch.object(
                connector,
                "_get",
                side_effect=[_page("citations", s1_citers), _page("citations", s2_citers)],
            ),
        ):
            outcome, resolved = run_snowball(
                connector,
                ["ARXIV:1", "ARXIV:2"],
                backward=False,
                since=datetime.date(2026, 1, 1),
                labels={"ARXIV:1": "c1", "ARXIV:2": "c2"},
            )

        assert [s.paper_id for s in resolved] == ["s1", "s2"]
        titles = [p.title for p in outcome.papers]
        assert titles == ["Recent citer of both seeds", "Recent citer of one seed"]
        top = outcome.papers[0]
        assert outcome.routes[id(top)] == ["cites c1", "cites c2"]
        assert outcome.evidence[id(top)] == ["[cites c1] extends [3]"]

    def test_missing_reference_list_is_a_reported_shortfall(self) -> None:
        connector = SemanticScholarConnector(api_key="k")
        with (
            patch.object(
                connector,
                "_post",
                side_effect=[_response([_seed("s1", "Preprint", references=40)])],
            ),
            patch.object(connector, "_get", side_effect=[_response({"data": []})]),
        ):
            outcome, _ = run_snowball(connector, ["ARXIV:1"], forward=False)

        assert outcome.papers == []
        (step,) = outcome.steps
        assert step.status is CoverageStatus.PARTIAL
        assert "no reference list" in (step.detail or "")

    def test_neighbours_beyond_keep_are_reported_as_capped(self) -> None:
        connector = SemanticScholarConnector(api_key="k")
        citers = [_edge(f"p{i}", f"Citer {i}", "2025-01-01") for i in range(5)]
        metadata = [_item(f"p{i}", f"Citer {i}") for i in range(2)]
        with (
            patch.object(
                connector,
                "_post",
                side_effect=[_response([_seed("s1", "Seed", citations=5)]), _response(metadata)],
            ),
            patch.object(connector, "_get", side_effect=[_page("citations", citers)]),
        ):
            outcome, _ = run_snowball(connector, ["ARXIV:1"], backward=False, keep=2)

        assert len(outcome.papers) == 2
        kept = next(s for s in outcome.steps if s.source == "snowball:kept")
        assert kept.status is CoverageStatus.CAPPED
        assert kept.reported_total == 5

    def test_rate_limited_edge_listing_is_a_failure_not_an_empty_list(self) -> None:
        connector = SemanticScholarConnector(api_key="k")
        refused = requests.HTTPError("429 Too Many Requests")
        with (
            patch.object(
                connector,
                "_post",
                side_effect=[_response([_seed("s1", "Preprint", references=40)])],
            ),
            patch.object(connector, "_get", side_effect=refused),
        ):
            outcome, _ = run_snowball(connector, ["ARXIV:1"], forward=False)

        (step,) = outcome.steps
        assert step.status is CoverageStatus.FAILED
        assert "no reference list" not in (step.detail or "")
