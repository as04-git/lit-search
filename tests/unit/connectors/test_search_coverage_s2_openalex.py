"""Coverage statuses reported by the Semantic Scholar and OpenAlex search paths."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import requests

from lit_search.connectors.openalex import OpenAlexConnector
from lit_search.connectors.semantic_scholar import SemanticScholarConnector
from lit_search.core.coverage import CoverageStatus


def _s2_item(paper_id: str, title: str | None = "A paper") -> dict[str, Any]:
    return {
        "paperId": paper_id,
        "title": title,
        "externalIds": {"DOI": f"10.1234/{paper_id}"},
        "publicationDate": "2024-01-01",
        "authors": [],
    }


def _s2_page(items: list[dict[str, Any]], total: int, token: str | None) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {"total": total, "data": items, "token": token}
    return response


def _oa_work(work_id: str) -> dict[str, Any]:
    return {
        "id": f"https://openalex.org/{work_id}",
        "title": f"Work {work_id}",
        "doi": f"https://doi.org/10.5555/{work_id}",
        "publication_date": "2024-01-01",
        "abstract_inverted_index": None,
        "authorships": [],
        "cited_by_count": 0,
        "open_access": {},
        "locations": [],
        "primary_location": {},
        "concepts": [],
        "keywords": [],
        "type": "article",
    }


def _oa_page(work_ids: list[str], count: int) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {
        "meta": {"count": count, "next_cursor": None},
        "results": [_oa_work(w) for w in work_ids],
    }
    return response


class TestSemanticScholarSearchCoverage:
    def test_all_reported_records_retrieved_is_complete(self, simple_query) -> None:
        connector = SemanticScholarConnector()
        page = _s2_page([_s2_item("a"), _s2_item("b")], total=2, token=None)
        with (
            patch.object(connector, "_get", return_value=page),
            patch.object(connector, "_enrich_author_affiliations"),
        ):
            result = connector.search(simple_query)

        assert result.coverage.status is CoverageStatus.COMPLETE
        assert result.coverage.reported_total == 2

    def test_cap_below_total_is_capped(self, simple_query) -> None:
        connector = SemanticScholarConnector()
        page = _s2_page([_s2_item("a"), _s2_item("b")], total=50, token="next")
        with patch.object(connector, "_get", return_value=page):
            result = connector.search(simple_query, max_papers=2)

        assert result.coverage.status is CoverageStatus.CAPPED
        assert result.coverage.reported_total == 50

    def test_failure_after_first_page_is_partial_with_error(self, simple_query) -> None:
        connector = SemanticScholarConnector()
        first = _s2_page([_s2_item("a")], total=5, token="next")
        with (
            patch("lit_search.connectors.semantic_scholar._PAGE_SIZE", 1),
            patch.object(
                connector, "_get", side_effect=[first, requests.ConnectionError("reset by peer")]
            ),
        ):
            result = connector.search(simple_query)

        assert result.coverage.status is CoverageStatus.PARTIAL
        assert result.coverage.error_type == "ConnectionError"
        assert len(result.papers) == 1

    def test_failure_on_first_page_is_failed(self, simple_query) -> None:
        connector = SemanticScholarConnector()
        with patch.object(connector, "_get", side_effect=requests.Timeout("slow")):
            result = connector.search(simple_query)

        assert result.coverage.status is CoverageStatus.FAILED
        assert result.papers == ()

    def test_unparseable_record_is_a_shortfall(self, simple_query) -> None:
        connector = SemanticScholarConnector()
        page = _s2_page([_s2_item("a"), _s2_item("b", title=None)], total=2, token=None)
        with patch.object(connector, "_get", return_value=page):
            result = connector.search(simple_query)

        assert result.coverage.status is CoverageStatus.PARTIAL
        assert "could not be parsed" in (result.coverage.detail or "")


class TestOpenAlexSearchCoverage:
    def test_single_query_complete(self, simple_query) -> None:
        connector = OpenAlexConnector()
        with patch.object(connector, "_get", return_value=_oa_page(["W1", "W2"], count=2)):
            result = connector.search(simple_query)

        assert result.coverage.status is CoverageStatus.COMPLETE
        assert result.coverage.reported_total == 2

    def test_or_branches_all_complete_is_complete(self, or_query) -> None:
        connector = OpenAlexConnector()
        pages = [_oa_page(["W1", "W2"], count=2), _oa_page(["W2", "W3"], count=2)]
        with patch.object(connector, "_get", side_effect=pages):
            result = connector.search(or_query)

        assert result.coverage.status is CoverageStatus.COMPLETE
        assert len(result.papers) == 3

    def test_one_failed_or_branch_makes_union_partial(self, or_query) -> None:
        connector = OpenAlexConnector()
        side_effect = [_oa_page(["W1"], count=1), requests.ConnectionError("down")]
        with patch.object(connector, "_get", side_effect=side_effect):
            result = connector.search(or_query)

        assert result.coverage.status is CoverageStatus.PARTIAL
        assert "OR branch 2 failed" in (result.coverage.detail or "")
        assert len(result.papers) == 1

    def test_every_or_branch_failed_is_failed(self, or_query) -> None:
        connector = OpenAlexConnector()
        side_effect = [requests.ConnectionError("down"), requests.ConnectionError("down")]
        with patch.object(connector, "_get", side_effect=side_effect):
            result = connector.search(or_query)

        assert result.coverage.status is CoverageStatus.FAILED

    def test_short_count_is_partial(self, simple_query) -> None:
        connector = OpenAlexConnector()
        with patch.object(connector, "_get", return_value=_oa_page(["W1"], count=3)):
            result = connector.search(simple_query)

        assert result.coverage.status is CoverageStatus.PARTIAL
