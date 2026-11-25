"""Unit tests for the bioRxiv and medRxiv connectors.

These run under ``pytest-socket`` with ``--disable-socket``; every HTTP call is
mocked.  Payloads mirror responses observed against ``api.biorxiv.org`` and
``api.medrxiv.org`` on 2026-08-31, including the four HTTP-200 error shapes.

Two tests carry the most weight:

- :meth:`TestCoverageIsHonest.test_failure_and_emptiness_are_different_statuses`,
  the regression test for this connector as a whole.
- :meth:`TestDonorRegression.test_a_boolean_query_is_not_turned_into_a_category`,
  which pins the specific defect that made this a new module rather than a
  port: the donor set ``category = query.lower().replace(' ', '_')`` and
  reported the resulting miss as an empty preprint server.
"""

from __future__ import annotations

import datetime
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from lit_search.connectors.biorxiv import (
    BioRxivConnector,
    MedRxivConnector,
    XRxivBodyError,
)
from lit_search.core.coverage import CoverageStatus, CoveredPapers, ExecutionStatus
from lit_search.core.paper import Database, PaperType

from .conftest import _parse

# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def _posting(
    doi: str = "10.64898/2026.07.31.741992",
    *,
    title: str = "Sound-shape correspondences in macaques",
    abstract: str = "Humans reliably associate speech-like sounds with visual shapes.",
    category: str = "animal behavior and cognition",
    published: str = "NA",
    version: str = "1",
    date: str = "2026-08-01",
    authors: str = "Loconsole, M.; Xue, C.",
) -> dict[str, Any]:
    """Build one ``collection`` entry.

    Parameters
    ----------
    doi : str
        Preprint DOI.
    title : str
        Title.
    abstract : str
        Abstract.
    category : str
        bioRxiv subject category.
    published : str
        Publisher DOI, or the literal ``"NA"``.
    version : str
        Version label.
    date : str
        Posting date.
    authors : str
        Semicolon-separated author string.

    Returns
    -------
    dict[str, Any]
        Record in the API's shape.
    """
    return {
        "doi": doi,
        "title": title,
        "authors": authors,
        "author_corresponding": "Maria Loconsole",
        "author_corresponding_institution": "University of Padova",
        "date": date,
        "version": version,
        "type": "new results",
        "license": "cc_by_nc_nd",
        "category": category,
        "jatsxml": "https://www.biorxiv.org/content/early/x.source.xml",
        "abstract": abstract,
        "published": published,
        "server": "biorxiv",
    }


def _body(
    records: list[dict[str, Any]],
    *,
    status: str = "ok",
    total: object = None,
) -> dict[str, Any]:
    """Build one response envelope.

    Parameters
    ----------
    records : list[dict[str, Any]]
        Collection entries.
    status : str
        ``messages[0].status``.
    total : object
        ``messages[0].total``.  The live API sends this as an ``int`` on some
        responses and as a decimal ``str`` on others, so tests exercise both.

    Returns
    -------
    dict[str, Any]
        Response body.
    """
    message: dict[str, Any] = {"status": status, "cursor": 0, "count": len(records)}
    if total is not None:
        message["total"] = total
    return {"messages": [message], "collection": records}


def _response(payload: dict[str, Any]) -> MagicMock:
    """Wrap a payload in a mock response.

    Parameters
    ----------
    payload : dict[str, Any]
        Body to return from ``.json()``.

    Returns
    -------
    MagicMock
        Mock response.
    """
    response = MagicMock(spec=requests.Response)
    response.status_code = 200
    response.json.return_value = payload
    response.text = json.dumps(payload)
    return response


def _search(connector, responses: list[Any], **kwargs) -> CoveredPapers:
    """Run a search against a scripted sequence of ``_get`` outcomes.

    Parameters
    ----------
    connector : BioRxivConnector | MedRxivConnector
        Connector under test.
    responses : list[Any]
        Values (or exceptions) ``_get`` yields in order.
    **kwargs
        Forwarded to ``search``.

    Returns
    -------
    CoveredPapers
        The result.
    """
    query = kwargs.pop("query", None) or _parse("[sound-shape]")
    with patch.object(connector, "_get", side_effect=responses):
        result: CoveredPapers = connector.search(query, **kwargs)
    return result


@pytest.fixture
def connector() -> BioRxivConnector:
    """Return a bioRxiv connector."""
    return BioRxivConnector()


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


class TestCoverageIsHonest:
    """What the connector says about how complete its answer is."""

    def test_failure_and_emptiness_are_different_statuses(
        self, connector: BioRxivConnector
    ) -> None:
        """The regression test for this connector.

        Both of these arrive as HTTP 200 with an empty ``collection``.  Under
        the donor's contract they are the same value, and a rejected request
        reads as a preprint server holding no matching paper.
        """
        empty = _search(connector, [_response(_body([], status="no posts found"))])
        rejected = _search(
            BioRxivConnector(),
            [_response(_body([], status="Both dates must be in yyyy-mm-dd format"))],
        )

        assert empty.status is CoverageStatus.PARTIAL
        assert empty.papers == ()
        assert rejected.status is CoverageStatus.FAILED
        assert empty.status is not rejected.status

    def test_a_normal_result_is_partial_never_complete(self, connector: BioRxivConnector) -> None:
        """Retrieving the whole window still does not answer the query fully."""
        result = _search(connector, [_response(_body([_posting()], total=1)), _response(_body([]))])
        assert result.status is CoverageStatus.PARTIAL
        assert result.coverage.execution_status is ExecutionStatus.COMPLETE
        assert len(result.papers) == 1

    def test_the_window_is_named_in_the_detail(self, connector: BioRxivConnector) -> None:
        """The horizon the donor hid is the first thing the ledger reports."""
        result = _search(connector, [_response(_body([], status="no posts found"))])
        detail = result.coverage.detail or ""
        assert "has no keyword search" in detail
        assert "postings dated" in detail
        assert "default 30-day window" in detail

    def test_a_caller_supplied_window_is_reported_without_the_default_note(
        self, connector: BioRxivConnector
    ) -> None:
        """An explicit window is described as the caller's, not as a default."""
        result = _search(
            connector,
            [_response(_body([], status="no posts found"))],
            since=datetime.date(2024, 1, 1),
            until=datetime.date(2024, 1, 31),
        )
        detail = result.coverage.detail or ""
        assert "postings dated 2024-01-01 to 2024-01-31" in detail
        assert "default 30-day window" not in detail

    def test_the_window_total_is_carried_but_does_not_justify_completeness(
        self, connector: BioRxivConnector
    ) -> None:
        """``total`` counts the window's postings, not the query's matches."""
        result = _search(
            connector, [_response(_body([_posting()], total="1")), _response(_body([]))]
        )
        assert result.status is CoverageStatus.PARTIAL
        assert "of the 1 the server reports for that window" in (result.coverage.detail or "")

    def test_a_mid_pagination_failure_keeps_records_and_reports_partial(
        self, connector: BioRxivConnector
    ) -> None:
        """Records already in hand are real and must not be discarded."""
        result = _search(
            connector,
            [
                _response(_body([_posting()], total=90)),
                requests.ConnectionError("connection reset"),
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 1
        assert "failed" in (result.coverage.detail or "")
        assert result.coverage.error_type == "ConnectionError"
        assert result.coverage.execution_status is ExecutionStatus.INCOMPLETE

    def test_a_first_page_failure_with_nothing_retrieved_is_failed(
        self, connector: BioRxivConnector
    ) -> None:
        """With no records the failure is the whole story."""
        result = _search(connector, [requests.ConnectionError("dns failure")])
        assert result.status is CoverageStatus.FAILED
        assert result.papers == ()

    @pytest.mark.parametrize(
        "status",
        ["Both dates must be in yyyy-mm-dd format", "Server not recognized"],
    )
    def test_each_http_200_error_body_becomes_failed(self, status: str) -> None:
        """Every HTTP-200 rejection is a failure, never an empty result."""
        result = _search(BioRxivConnector(), [_response(_body([], status=status))])
        assert result.status is CoverageStatus.FAILED
        assert status.lower() in (result.coverage.error or "").lower()

    def test_a_body_with_no_collection_is_rejected(self, connector: BioRxivConnector) -> None:
        """An unrecognised shape raises rather than defaulting to no papers."""
        result = _search(connector, [_response({"messages": [{"status": "ok"}]})])
        assert result.status is CoverageStatus.FAILED
        assert "no 'collection' list" in (result.coverage.error or "")

    def test_a_non_json_body_is_rejected(self, connector: BioRxivConnector) -> None:
        """A body that is not JSON is a failure, not zero papers."""
        response = MagicMock(spec=requests.Response)
        response.status_code = 200
        response.json.side_effect = ValueError("no json")
        response.text = "<html>maintenance</html>"
        result = _search(connector, [response])
        assert result.status is CoverageStatus.FAILED

    def test_unparsed_records_are_reported(self, connector: BioRxivConnector) -> None:
        """A posting seen and not parsed is a coverage loss, not an absence."""
        untitled = _posting()
        untitled["title"] = ""
        result = _search(
            connector,
            [_response(_body([_posting(), untitled], total=2)), _response(_body([]))],
        )
        assert "1 of 2 postings could not be parsed" in (result.coverage.detail or "")

    def test_search_returns_the_contract_type(self, connector: BioRxivConnector) -> None:
        """The connector cannot hand back a bare list."""
        result = _search(connector, [_response(_body([], status="no posts found"))])
        assert isinstance(result, CoveredPapers)


# ---------------------------------------------------------------------------
# The donor's defect
# ---------------------------------------------------------------------------


class TestDonorRegression:
    """The specific behaviour that made this a rewrite rather than a port."""

    def test_a_boolean_query_is_not_turned_into_a_category(
        self, connector: BioRxivConnector
    ) -> None:
        """A multi-term query matches on content, not on a fabricated category.

        The donor would have asked for the category
        ``sound-shape_and_macaques`` and found nothing.
        """
        result = _search(
            connector,
            [
                _response(_body([_posting()], total=1)),
                _response(_body([])),
            ],
            query=_parse("[sound-shape] AND [macaques]"),
        )
        assert len(result.papers) == 1

    def test_the_category_is_searchable_on_its_own(self, connector: BioRxivConnector) -> None:
        """The donor's one true insight survives: category is a real field."""
        result = _search(
            connector,
            [_response(_body([_posting()], total=1)), _response(_body([]))],
            query=_parse("key[animal behavior]"),
        )
        assert len(result.papers) == 1

    def test_no_request_carries_the_query(self, connector: BioRxivConnector) -> None:
        """The query never reaches the wire, because the API cannot take one."""
        with patch.object(
            connector,
            "_get",
            side_effect=[_response(_body([], status="no posts found"))],
        ) as mocked:
            connector.search(_parse("[sound-shape] AND [macaques]"))
        (url,), _ = mocked.call_args
        assert "sound-shape" not in url
        assert "/details/biorxiv/" in url


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


class TestRecords:
    """What the papers themselves carry."""

    def test_a_published_preprint_keeps_both_dois(self, connector: BioRxivConnector) -> None:
        """``published`` supplies D24's link, and both identities are kept."""
        result = _search(
            connector,
            [
                _response(
                    _body(
                        [
                            _posting(
                                doi="10.1101/2023.09.16.558066",
                                published="10.1098/rstb.2022.0467",
                            )
                        ],
                        total=1,
                    )
                ),
                _response(_body([])),
            ],
        )
        (paper,) = result.papers
        assert paper.doi == "10.1098/rstb.2022.0467"
        assert paper.preprint_doi == "10.1101/2023.09.16.558066"
        assert paper.paper_type is PaperType.ARTICLE

    def test_the_na_literal_is_not_mistaken_for_a_doi(self, connector: BioRxivConnector) -> None:
        """``published`` is the string "NA", not null, when unpublished."""
        result = _search(connector, [_response(_body([_posting()], total=1)), _response(_body([]))])
        (paper,) = result.papers
        assert paper.doi == "10.64898/2026.07.31.741992"
        assert paper.preprint_doi is None
        assert paper.paper_type is PaperType.UNPUBLISHED

    def test_authors_are_split_on_semicolons(self, connector: BioRxivConnector) -> None:
        """The API's author string is semicolon-separated, not comma."""
        result = _search(connector, [_response(_body([_posting()], total=1)), _response(_body([]))])
        (paper,) = result.papers
        assert [a.name for a in paper.authors] == ["Loconsole, M.", "Xue, C."]

    def test_found_in_names_the_right_server(self) -> None:
        """medRxiv records are attributed to medRxiv, not to bioRxiv."""
        result = _search(
            MedRxivConnector(),
            [_response(_body([_posting()], total=1)), _response(_body([]))],
        )
        (paper,) = result.papers
        assert paper.found_in == {Database.MEDRXIV.value}


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


class TestWiring:
    """Rate-limit scope and request shape."""

    def test_both_servers_share_one_rate_limit_scope(self) -> None:
        """One operator, one API stack, no published rate: share the clock."""
        assert BioRxivConnector().rate_limit_service == MedRxivConnector().rate_limit_service

    def test_the_rate_is_declared_and_not_zero(self) -> None:
        """A zero interval would silently exempt a real remote host."""
        assert BioRxivConnector().min_request_interval > 0

    def test_pagination_advances_by_the_records_returned(self, connector: BioRxivConnector) -> None:
        """The cursor is a record offset, and it must track what was served."""
        with patch.object(
            connector,
            "_get",
            side_effect=[
                _response(_body([_posting(doi=f"10.64898/x{i}") for i in range(30)], total=45)),
                _response(_body([_posting(doi=f"10.64898/y{i}") for i in range(15)], total=45)),
            ],
        ) as mocked:
            connector.search(_parse("[sound-shape]"))
        urls = [call.args[0] for call in mocked.call_args_list]
        assert urls[0].endswith("/0")
        assert urls[1].endswith("/30")

    def test_the_body_error_type_is_a_connector_error(self) -> None:
        """The base class only converts ``ConnectorError`` into ``FAILED``."""
        assert issubclass(XRxivBodyError, Exception)
        result = _search(BioRxivConnector(), [_response(_body([], status="Server not recognized"))])
        assert result.status is CoverageStatus.FAILED
