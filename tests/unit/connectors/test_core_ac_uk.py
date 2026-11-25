"""Unit tests for CoreConnector.

These run under ``pytest-socket`` with ``--disable-socket`` (see
``pyproject.toml``), so every HTTP call is mocked and no test can reach the
network.

**A caveat this file used to carry, now withdrawn.** It said CORE refuses
keyless search -- HTTP 429 with a zero-byte body -- and therefore that no
fixture here could be checked against a real response.  Re-checked live
2026-08-31: a keyless ``GET /v3/search/works/?q=catalysis&limit=2`` returns
HTTP 200, and its envelope is ``limit`` / ``offset`` / ``results`` /
``searchId`` / ``totalHits`` with work keys ``publishedDate``,
``yearPublished``, ``downloadUrl``, ``fieldOfStudy``, ``sourceFulltextUrls``,
``citationCount``, ``documentType``, ``journals`` -- **camelCase, matching the
documentation page (``api.core.ac.uk/docs/v3``) and not the OpenAPI document
(``api.core.ac.uk/swagger/v3.json``), which says ``total_hits`` /
``published_date`` / ``year_published``**.  The camelCase fixtures here are
therefore the real shape.  The connector still reads both spellings and
:class:`TestBothFieldSpellingsAreRead` still keeps that true, as cheap
redundancy rather than as a hedge against an unknown.

The test that carries the most weight is
:meth:`TestCoverageIsHonest.test_failure_and_emptiness_are_different_statuses`.
Everything else here can pass while the connector still reports an upstream
outage as a confident zero; that one cannot.
"""

from __future__ import annotations

import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from lit_search.connectors import core_ac_uk
from lit_search.connectors.core_ac_uk import CoreBodyError, CoreConnector
from lit_search.core.coverage import CoverageStatus, CoveredPapers
from lit_search.core.paper import Database, PaperType
from lit_search.core.query import FilterCode, NodeType, Query, QueryNode
from lit_search.core.source import SourceType
from lit_search.query.builders.core_ac_uk import CoreQueryBuilder

from .conftest import _parse

_API_KEY = "test-core-key"

# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def _record(index: int, **overrides: Any) -> dict[str, Any]:
    """Build one CORE work, in the documentation's camelCase spelling.

    Parameters
    ----------
    index : int
        Used to make the title and identifiers distinct.
    **overrides : Any
        Fields to replace or add.

    Returns
    -------
    dict[str, Any]
        A record shaped like a ``results`` entry.
    """
    record: dict[str, Any] = {
        "id": 100000 + index,
        "title": f"Synthetic Work {index}.",
        "abstract": f"Synthetic abstract {index}.",
        "doi": f"10.1000/test.{index}",
        "publishedDate": "2023-05-04T00:00:00",
        "yearPublished": 2023,
        "downloadUrl": f"https://core.ac.uk/download/{index}.pdf",
        "publisher": "Elsevier",
        "language": {"code": "en", "name": "English"},
        "documentType": "research",
        "citationCount": index,
        "authors": [{"name": f"Author {index}"}],
        "subjects": ["catalysis"],
        "fieldOfStudy": "Chemistry",
        "journals": {},
        "links": [{"type": "display", "url": f"https://core.ac.uk/works/{index}"}],
    }
    record.update(overrides)
    return record


def _page(records: list[dict[str, Any]], total: int | None, **overrides: Any) -> dict[str, Any]:
    """Build one search-response body.

    Parameters
    ----------
    records : list[dict[str, Any]]
        Records for this page.
    total : int | None
        Value of ``totalHits``.  ``None`` omits the key entirely, which is how
        a serialization mismatch would present.
    **overrides : Any
        Extra envelope fields, used to test the snake_case spelling.

    Returns
    -------
    dict[str, Any]
        Response body.
    """
    body: dict[str, Any] = {"results": records, "limit": len(records), "offset": 0}
    if total is not None:
        body["totalHits"] = total
    body.update(overrides)
    return body


def _response(body: Any, status_code: int = 200) -> MagicMock:
    """Wrap *body* in a mock :class:`requests.Response`.

    Parameters
    ----------
    body : Any
        Object returned by ``.json()``.
    status_code : int
        HTTP status to report.

    Returns
    -------
    MagicMock
        Mock response.
    """
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = body
    return mock


@pytest.fixture
def connector() -> CoreConnector:
    """Return a CORE connector holding a key."""
    return CoreConnector(api_key=_API_KEY)


def _search(connector: CoreConnector, responses: list[Any], **kwargs: Any) -> CoveredPapers:
    """Run a search against a scripted sequence of ``_get`` outcomes.

    Parameters
    ----------
    connector : CoreConnector
        Connector under test.
    responses : list[Any]
        Values or exceptions for successive ``_get`` calls.
    **kwargs : Any
        Forwarded to ``search`` (``max_papers``, ``since``, ``until``).

    Returns
    -------
    CoveredPapers
        The search outcome.
    """
    query = kwargs.pop("query", None) or _parse("[machine learning]")
    with (
        patch.object(connector, "_rate_limit"),
        patch.object(connector, "_get", side_effect=responses),
    ):
        return connector.search(query, **kwargs)


# ---------------------------------------------------------------------------
# Declared contract
# ---------------------------------------------------------------------------


class TestDeclaredContract:
    """The connector declares what the base classes require of it."""

    def test_name_is_the_registered_database_identifier(self, connector) -> None:
        """The ledger key matches the Database member."""
        assert connector.name == Database.CORE.value

    def test_interval_is_the_strictest_published_tier(self, connector) -> None:
        """Three of CORE's four published tiers cap at 10 requests a minute.

        Nothing in an API key says which tier it belongs to, so the strictest
        keyed one binds.  The build plan's 2.0s is 30/min and exceeds three of
        the four; it is deliberately not used.
        """
        assert connector.min_request_interval == 6.0

    def test_registered_in_the_search_registry(self) -> None:
        """A connector nobody can select is not ported."""
        from lit_search.connectors import SEARCH_REGISTRY

        assert SEARCH_REGISTRY[Database.CORE] is CoreConnector

    def test_quota_is_declared_because_the_limit_is_a_daily_budget(self, connector) -> None:
        """Interval pacing alone spends a daily token budget and then errors."""
        quota = connector.request_quota
        assert (quota.limit, quota.window_seconds) == (1000, 86400.0)

    def test_rate_limit_service_is_shared_across_core_endpoints(self, connector) -> None:
        """The token bucket belongs to the key, not to one endpoint."""
        assert connector.rate_limit_service == "core"

    def test_the_key_changes_the_rate_limit_bucket(self) -> None:
        """The key is the quota's identity, so it must reach the scope."""
        anonymous = CoreConnector()
        keyed = CoreConnector(api_key=_API_KEY)
        assert anonymous.rate_limit_scope.fingerprint == "anon"
        assert keyed.rate_limit_scope.fingerprint != "anon"
        assert keyed.rate_limit_scope.service == anonymous.rate_limit_scope.service


# ---------------------------------------------------------------------------
# The four required cases, plus the one that matters
# ---------------------------------------------------------------------------


class TestCoverageIsHonest:
    """Each way a search can end reports a different, actionable status."""

    def test_normal_result_is_complete(self, connector) -> None:
        """Retrieving as many records as totalHits reports is COMPLETE."""
        result = _search(connector, [_response(_page([_record(0), _record(1)], total=2))])
        assert result.status is CoverageStatus.COMPLETE
        assert len(result.papers) == 2
        assert result.coverage.reported_total == 2

    def test_upstream_failure_is_failed(self, connector) -> None:
        """A dead first request is FAILED, and the error is carried.

        The donor reaches this same point at ``if response is None: return
        papers`` after three exhausted retries -- a successful empty search,
        with no exception raised anywhere.
        """
        result = _search(connector, [requests.ConnectionError("core unreachable")])
        assert result.status is CoverageStatus.FAILED
        assert result.papers == ()
        assert "unreachable" in (result.coverage.error or "")
        assert result.coverage.error_type == "ConnectionError"

    def test_empty_but_successful_search_is_complete(self, connector) -> None:
        """A genuine zero-hit query is COMPLETE with no papers."""
        result = _search(connector, [_response(_page([], total=0))])
        assert result.status is CoverageStatus.COMPLETE
        assert result.papers == ()
        assert result.coverage.reported_total == 0

    def test_capped_result_is_capped(self, connector) -> None:
        """Stopping at max_papers is CAPPED and names the cap."""
        result = _search(
            connector,
            [_response(_page([_record(i) for i in range(3)], total=900))],
            max_papers=3,
        )
        assert result.status is CoverageStatus.CAPPED
        assert len(result.papers) == 3
        assert result.coverage.cap == 3
        assert result.coverage.reported_total == 900

    def test_keyless_search_issues_a_request(self) -> None:
        """CORE's documented anonymous tier is a real public code path."""
        keyless = CoreConnector()
        with (
            patch.object(keyless, "_rate_limit"),
            patch.object(keyless, "_get", return_value=_response(_page([], total=0))) as mocked,
        ):
            result = keyless.search(_parse("[machine learning]"))
        assert result.status is CoverageStatus.COMPLETE
        assert result.papers == ()
        mocked.assert_called_once()

    def test_failure_and_emptiness_are_different_statuses(self, connector) -> None:
        """The regression test for the whole port.

        The donor reaches all three of these through the same ``return
        papers``: ``except requests.RequestException``, ``except Exception``,
        and ``if response is None``.  Here the outage, the empty repository and
        the absent key must be three distinct statuses.
        """
        failure = _search(connector, [requests.ConnectionError("boom")])
        empty = _search(connector, [_response(_page([], total=0))])
        keyless = CoreConnector()
        with (
            patch.object(keyless, "_rate_limit"),
            patch.object(keyless, "_get", return_value=_response(_page([], total=0))),
        ):
            anonymous_empty = keyless.search(_parse("[machine learning]"))

        assert failure.papers == empty.papers == anonymous_empty.papers == ()
        assert failure.status is CoverageStatus.FAILED
        assert empty.status is CoverageStatus.COMPLETE
        assert anonymous_empty.status is CoverageStatus.COMPLETE
        assert empty.coverage.is_complete
        assert not failure.coverage.is_complete
        assert anonymous_empty.coverage.is_complete


# ---------------------------------------------------------------------------
# The 200-with-an-unusable-body family
# ---------------------------------------------------------------------------


class TestTwoHundredWithAnUnusableBody:
    """A 200 that is not a search result must not read as zero records."""

    def test_missing_results_is_failed_not_empty(self, connector) -> None:
        """Without ``results`` the body is not a search result at all."""
        result = _search(connector, [_response({"totalHits": 5})])
        assert result.status is CoverageStatus.FAILED
        assert "results" in (result.coverage.error or "")

    def test_error_message_body_is_failed(self, connector) -> None:
        """A rejected key returns ``{"message": "..."}`` with no results.

        Verified live 2026-08-31 with an invalid bearer token: HTTP 401
        ``{"message": "The API key you provided is not valid."}``.  The body is
        treated as an error whatever the status, so a gateway rewriting a 401
        to a 200 cannot turn a rejected key into an empty database.
        """
        result = _search(
            connector, [_response({"message": "The API key you provided is not valid."})]
        )
        assert result.status is CoverageStatus.FAILED
        assert "not valid" in (result.coverage.error or "")

    def test_results_of_the_wrong_type_is_failed(self, connector) -> None:
        """``results`` as an object rather than a list is a schema change."""
        result = _search(connector, [_response({"results": {"a": 1}, "totalHits": 1})])
        assert result.status is CoverageStatus.FAILED

    def test_non_json_body_is_failed(self, connector) -> None:
        """An HTML error page from a proxy is a failure, not an empty result."""
        broken = MagicMock()
        broken.status_code = 200
        broken.json.side_effect = ValueError("Expecting value")
        result = _search(connector, [broken])
        assert result.status is CoverageStatus.FAILED

    def test_json_array_body_is_failed(self, connector) -> None:
        """A JSON body of the wrong shape is a failure."""
        result = _search(connector, [_response([1, 2, 3])])
        assert result.status is CoverageStatus.FAILED

    def test_body_error_is_a_connector_error(self) -> None:
        """The body error is catchable as a ConnectorError by the base class."""
        from lit_search.exceptions import ConnectorError

        assert isinstance(CoreBodyError("x"), ConnectorError)


# ---------------------------------------------------------------------------
# The spelling conflict CORE's own two documents create
# ---------------------------------------------------------------------------


class TestBothFieldSpellingsAreRead:
    """CORE's OpenAPI document and its docs page disagree on field naming.

    Neither could be checked live, so both are read.  If only one were, the
    other's arrival would silently produce papers with blank dates and a run
    with no reported total -- and a run with no reported total *cannot* claim
    COMPLETE, which is the safety net that makes the ambiguity survivable
    rather than dangerous.
    """

    def test_snake_case_envelope_total_is_read(self, connector) -> None:
        """The OpenAPI document names it ``total_hits``."""
        body = _page([_record(0)], total=None, total_hits=1)
        result = _search(connector, [_response(body)])
        assert result.status is CoverageStatus.COMPLETE
        assert result.coverage.reported_total == 1

    def test_camel_case_envelope_total_is_read(self, connector) -> None:
        """The documentation page names it ``totalHits``."""
        result = _search(connector, [_response(_page([_record(0)], total=1))])
        assert result.status is CoverageStatus.COMPLETE
        assert result.coverage.reported_total == 1

    def test_neither_spelling_present_cannot_claim_complete(self, connector) -> None:
        """This is the case the ambiguity actually protects against.

        If CORE ships a third spelling, or renames the field, the connector
        reads no total -- and reports PARTIAL, not a confident COMPLETE built
        on its own paper count.
        """
        result = _search(
            connector,
            [
                _response(_page([_record(0)], total=None)),
                _response(_page([], total=None)),
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert result.coverage.reported_total is None
        assert "no result total" in (result.coverage.detail or "")

    def test_without_a_total_pagination_runs_to_an_empty_page(self, connector) -> None:
        """With no total there is nothing to stop at except exhaustion.

        Stopping after one page because the count was unreadable would turn a
        serialization mismatch into a silently truncated search.
        """
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(
                connector,
                "_get",
                side_effect=[
                    _response(_page([_record(0)], total=None)),
                    _response(_page([_record(1)], total=None)),
                    _response(_page([], total=None)),
                ],
            ) as mocked,
        ):
            result = connector.search(_parse("[machine learning]"))
        assert len(result.papers) == 2
        assert mocked.call_count == 3

    def test_snake_case_record_fields_are_read(self, connector) -> None:
        """A record in the OpenAPI spelling still yields a dated paper."""
        record = {
            "id": 1,
            "title": "Snake Case Work.",
            "abstract": "Body.",
            "published_date": "2021-03-02T00:00:00",
            "year_published": 2021,
            "download_url": "https://core.ac.uk/download/1.pdf",
            "field_of_study": "Physics",
            "citation_count": 4,
            "document_type": "research",
        }
        paper = connector._parse_item(record)
        assert paper is not None
        assert paper.publication_date == datetime.date(2021, 3, 2)
        assert paper.pdf_url == "https://core.ac.uk/download/1.pdf"
        assert paper.citations == 4
        assert paper.keywords == {"Physics"}


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class TestOffsetPagination:
    """Every abnormal stop reaches the coverage status instead of the log."""

    def test_full_page_then_empty_page_ends_the_run(self, connector) -> None:
        """An empty page is CORE's only end-of-set signal."""
        result = _search(
            connector,
            [
                _response(_page([_record(0), _record(1)], total=3)),
                _response(_page([_record(2)], total=3)),
            ],
        )
        assert result.status is CoverageStatus.COMPLETE
        assert len(result.papers) == 3

    def test_short_page_alone_does_not_end_pagination(self, connector) -> None:
        """A page smaller than the limit keeps going until the total is met.

        CORE documents no guarantee that a page is full, and the reference port
        proved the short-page heuristic wrong on a sibling API.
        """
        result = _search(
            connector,
            [
                _response(_page([_record(0)], total=2)),
                _response(_page([_record(1)], total=2)),
            ],
        )
        assert len(result.papers) == 2
        assert result.status is CoverageStatus.COMPLETE

    def test_empty_page_before_the_total_is_partial(self, connector) -> None:
        """CORE stopped delivering before it delivered what it promised."""
        result = _search(
            connector,
            [
                _response(_page([_record(0)], total=50)),
                _response(_page([], total=50)),
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert result.coverage.reported_total == 50
        assert "empty page" in (result.coverage.detail or "")

    def test_failure_after_some_pages_is_partial_not_failed(self, connector) -> None:
        """Records that did arrive are real, so the run is PARTIAL."""
        result = _search(
            connector,
            [
                _response(_page([_record(0), _record(1)], total=90)),
                requests.ConnectionError("dropped mid-run"),
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 2
        assert result.coverage.error_type == "ConnectionError"
        assert "dropped mid-run" in (result.coverage.detail or "")

    def test_total_drift_between_pages_is_partial(self, connector) -> None:
        """A moving index means neither total can be checked."""
        result = _search(
            connector,
            [
                _response(_page([_record(0), _record(1)], total=3)),
                _response(_page([_record(2)], total=7)),
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert "changed mid-retrieval" in (result.coverage.detail or "")

    def test_unparsable_records_are_counted_as_a_shortfall(self, connector) -> None:
        """A record seen but not parsed is a coverage loss, not an absence."""
        result = _search(
            connector,
            [_response(_page([_record(0), _record(1, title="  ")], total=2))],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 1
        assert "could not be parsed" in (result.coverage.detail or "")

    def test_the_offset_ceiling_is_a_reported_shortfall(self, connector) -> None:
        """The connector's own deep-pagination bound is stated, not hidden.

        The bound is the connector's safety stop, not a measured CORE limit --
        CORE served ``offset=100000`` with real records on 2026-08-31 -- so
        reaching it must degrade the status and say whose bound it was.  A
        truncation reported as an exhaustion is the failure this fork exists to
        prevent, and that stays true when the truncation is ours.
        """
        full_page = [_record(index) for index in range(100)]
        pages = [_response(_page(full_page, total=50_000_000)) for _ in range(10)]
        with patch.object(core_ac_uk, "_MAX_OFFSET", 500):
            result = _search(connector, pages)
        assert result.status is CoverageStatus.PARTIAL
        assert "500" in (result.coverage.detail or "")
        assert "connector's own" in (result.coverage.detail or "")

    def test_the_offset_ceiling_is_not_ten_thousand(self) -> None:
        """Regression: the 10,000 guess was falsified live on 2026-08-31.

        ``GET /v3/search/works/?q=catalysis&limit=10&offset=10000`` returned
        HTTP 200 with ten real records, as did the same request at
        ``offset=100000``.  A 10,000 bound therefore capped CORE recall on any
        broad query and reported a shortfall where none had occurred.
        """
        assert core_ac_uk._MAX_OFFSET > 10_000

    def test_offset_advances_by_the_records_actually_seen(self, connector) -> None:
        """Advancing by the requested limit would skip records on a short page."""
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(
                connector,
                "_get",
                side_effect=[
                    _response(_page([_record(0)], total=2)),
                    _response(_page([_record(1)], total=2)),
                ],
            ) as mocked,
        ):
            connector.search(_parse("[machine learning]"))
        assert mocked.call_args_list[0].args[1]["offset"] == 0
        assert mocked.call_args_list[1].args[1]["offset"] == 1

    def test_coverage_count_always_matches_the_papers_returned(self, connector) -> None:
        """CoveredPapers rejects a mismatch, so this asserts the accounting."""
        result = _search(
            connector,
            [_response(_page([_record(i) for i in range(5)], total=99))],
            max_papers=2,
        )
        assert result.coverage.retrieved == len(result.papers) == 2


# ---------------------------------------------------------------------------
# Request shaping
# ---------------------------------------------------------------------------


class TestRequestShaping:
    """Parameters and headers sent upstream match what CORE accepts."""

    def _capture(self, connector, **kwargs) -> dict[str, Any]:
        """Run one search and return the parameters of the first request."""
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(connector, "_get", return_value=_response(_page([], total=0))) as mocked,
        ):
            connector.search(_parse("[machine learning]"), **kwargs)
        params: dict[str, Any] = mocked.call_args.args[1]
        return params

    def test_page_size_never_exceeds_one_hundred(self, connector) -> None:
        """The donor's own documented ceiling, and the conservative choice."""
        assert 1 <= self._capture(connector, max_papers=100_000)["limit"] <= 100

    def test_page_size_is_at_least_one_when_the_cap_is_one(self, connector) -> None:
        """A cap of 1 must not produce limit=0."""
        assert self._capture(connector, max_papers=1)["limit"] >= 1

    def test_date_bounds_become_a_year_published_filter(self, connector) -> None:
        """CORE's documented searchable date field for works is a year."""
        params = self._capture(
            connector,
            since=datetime.date(2020, 6, 1),
            until=datetime.date(2021, 3, 15),
        )
        assert "yearPublished>=2020" in params["q"]
        assert "yearPublished<=2021" in params["q"]
        assert "[" not in params["q"].split("yearPublished", 1)[1]

    @pytest.mark.parametrize(
        ("since", "until"),
        [
            (datetime.date(2020, 6, 1), None),
            (None, datetime.date(2021, 3, 15)),
            (datetime.date(2020, 6, 1), datetime.date(2021, 3, 15)),
        ],
    )
    def test_date_bounds_never_use_the_lucene_range_form(self, connector, since, until) -> None:
        """Regression: ``yearPublished:[a TO b]`` is HTTP 500 on CORE.

        Verified live 2026-08-31: ``yearPublished:[2020 TO 2024]`` returns
        HTTP 500 -- *"Invalid expression: A binary operator with incompatible
        types was detected. Found operand types 'Edm.Int32' and 'Edm.String'"*
        -- because CORE runs Azure Cognitive Search and ``yearPublished`` is an
        integer field.  The open-ended ``*`` bound the connector used to emit
        failed identically.  Emitting that form made **every** date-bounded
        CORE search end as ``FAILED``.
        """
        rendered = self._capture(connector, since=since, until=until)["q"]
        assert "yearPublished:[" not in rendered
        assert "*" not in rendered

    def test_no_date_filter_when_no_bounds_are_given(self, connector) -> None:
        """An unbounded search must not silently acquire a date restriction."""
        assert "yearPublished" not in self._capture(connector)["q"]

    def test_the_key_is_sent_as_a_bearer_header(self, connector) -> None:
        """A configured key must actually reach the request."""
        assert connector._prepare_headers({})["Authorization"] == f"Bearer {_API_KEY}"

    def test_no_authorization_header_without_a_key(self) -> None:
        """The keyless path never reaches a request, but must not send a blank."""
        assert "Authorization" not in CoreConnector()._prepare_headers({})


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    """Field mapping from a CORE work to a Paper."""

    def test_core_fields(self, connector) -> None:
        """Title, abstract, DOI, date, URLs and citations are mapped."""
        paper = connector._parse_item(_record(7))
        assert paper is not None
        assert paper.title == "Synthetic Work 7."
        assert paper.abstract == "Synthetic abstract 7."
        assert paper.doi == "10.1000/test.7"
        assert paper.publication_date == datetime.date(2023, 5, 4)
        assert paper.url == "https://core.ac.uk/works/7"
        assert paper.pdf_url == "https://core.ac.uk/download/7.pdf"
        assert paper.citations == 7
        assert paper.language == "en"
        assert paper.paper_type is PaperType.ARTICLE
        assert Database.CORE in paper.found_in

    def test_year_only_records_fall_back_to_january(self, connector) -> None:
        """A work with only a publication year still carries a date."""
        paper = connector._parse_item(_record(1, publishedDate=None))
        assert paper is not None
        assert paper.publication_date == datetime.date(2023, 1, 1)

    def test_journal_becomes_a_journal_source(self, connector) -> None:
        """A named journal outranks the publisher."""
        record = _record(2, journals=[{"title": "Nature", "identifiers": ["0028-0836"]}])
        paper = connector._parse_item(record)
        assert paper is not None
        assert paper.source is not None
        assert paper.source.title == "Nature"
        assert paper.source.source_type is SourceType.JOURNAL

    def test_publisher_is_the_repository_fallback(self, connector) -> None:
        """Without a journal the publisher is used, and labelled a repository."""
        paper = connector._parse_item(_record(3))
        assert paper is not None
        assert paper.source is not None
        assert paper.source.title == "Elsevier"
        assert paper.source.source_type is SourceType.REPOSITORY

    def test_subjects_and_field_of_study_both_become_keywords(self, connector) -> None:
        """Two separate CORE fields carry topical terms."""
        paper = connector._parse_item(_record(4))
        assert paper is not None
        assert paper.keywords == {"catalysis", "Chemistry"}

    def test_open_access_is_not_asserted(self, connector) -> None:
        """CORE indexes open repositories but makes no per-record claim.

        Asserting ``is_open_access=True`` from the aggregator's remit rather
        than from the record would be a claim about a paper made from a fact
        about a database.
        """
        paper = connector._parse_item(_record(5))
        assert paper is not None
        assert paper.is_open_access is None

    def test_untitled_record_is_rejected(self, connector) -> None:
        """A record with no title cannot become a Paper."""
        assert connector._parse_item(_record(0, title="")) is None

    def test_non_dict_record_is_rejected(self, connector) -> None:
        """A malformed entry is counted as unparsed, not crashed on."""
        assert connector._parse_item("not a record") is None

    def test_url_falls_back_to_the_doi(self, connector) -> None:
        """A record with no display link still gets a resolvable URL."""
        paper = connector._parse_item(_record(6, links=[]))
        assert paper is not None
        assert paper.url == "https://doi.org/10.1000/test.6"

    def test_non_pdf_download_url_is_not_offered_as_a_pdf(self, connector) -> None:
        """A landing page is not a PDF, and calling it one breaks download."""
        paper = connector._parse_item(_record(8, downloadUrl="https://core.ac.uk/works/8"))
        assert paper is not None
        assert paper.pdf_url is None


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------


class TestQueryBuilder:
    """Only documented searchable fields are emitted."""

    @pytest.fixture
    def builder(self) -> CoreQueryBuilder:
        """Return a CORE query builder."""
        return CoreQueryBuilder()

    def test_default_filter_is_title_abstract_not_keywords(self, builder) -> None:
        """CORE documents no searchable keyword field for works.

        Declaring tiabskey unsupported makes the default fall back to the two
        fields CORE really does index, rather than claiming a keyword search
        that never happens.
        """
        assert builder.supports_filter(FilterCode.TITLE_ABSTRACT_KEYWORDS) is False
        converted = builder.convert_query(_parse("[machine learning]"))
        assert converted == '(title:"machine learning" OR abstract:"machine learning")'

    @pytest.mark.parametrize(
        ("query_string", "expected"),
        [
            ("ti[catalysis]", 'title:"catalysis"'),
            ("abs[catalysis]", 'abstract:"catalysis"'),
            ("au[David]", 'authors:"David"'),
        ],
    )
    def test_each_filter_maps_to_a_documented_field(self, builder, query_string, expected) -> None:
        """Each field carries a worked example in CORE's own documentation."""
        assert builder.convert_query(_parse(query_string)) == expected

    def test_connectors_map_to_core_operators(self, builder) -> None:
        """AND, OR and AND NOT render as boolean operators."""
        assert "NOT" in builder.convert_query(_parse("ti[a] AND NOT ti[b]"))

    def test_wildcard_terms_are_rejected(self, builder) -> None:
        """Untestable without a key, and the siblings disagree on behaviour."""
        result = builder.validate_query(_parse("ti[catalys*]"))
        assert result.is_valid is False
        assert "wildcard" in (result.error_message or "")

    def test_unsupported_filter_is_rejected(self, builder) -> None:
        """CORE documents no keyword or affiliation field for works."""
        assert builder.validate_query(_parse("key[catalysis]")).is_valid is False
        assert builder.validate_query(_parse("aff[Toronto]")).is_valid is False

    def test_empty_term_is_rejected(self, builder) -> None:
        """The parser forbids this; the builder is the second line of defence."""
        blank = Query(
            raw_query="ti[]",
            root=QueryNode(
                node_type=NodeType.ROOT,
                children=[
                    QueryNode(node_type=NodeType.TERM, value=" ", filter_code=FilterCode.TITLE)
                ],
            ),
        )
        result = builder.validate_query(blank)
        assert result.is_valid is False

    def test_rejected_query_reaches_the_ledger_as_query_unsupported(self, connector) -> None:
        """A refused query must not spend a request, and must say why.

        CORE's budget is a daily token allowance, so a wasted request is
        spent for the whole day rather than for a second.
        """
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(connector, "_get") as mocked,
        ):
            result = connector.search(_parse("key[catalysis]"))
        assert result.status is CoverageStatus.QUERY_UNSUPPORTED
        assert result.papers == ()
        mocked.assert_not_called()

    def test_a_rejected_query_outranks_a_missing_key(self, connector) -> None:
        """Both are determinate, and the query one is the actionable fact.

        Recorded because the ordering is a design choice: query validation runs
        in ``search`` before ``_fetch_papers``, so a keyless connector given an
        unsupported query reports QUERY_UNSUPPORTED rather than
        SKIPPED_NO_CREDENTIAL.  Supplying a key would not make that query work.
        """
        keyless = CoreConnector()
        with patch.object(keyless, "_rate_limit"), patch.object(keyless, "_get"):
            result = keyless.search(_parse("key[catalysis]"))
        assert result.status is CoverageStatus.QUERY_UNSUPPORTED


# ---------------------------------------------------------------------------
# The base class checks COMPLETE rather than accepting it
# ---------------------------------------------------------------------------


class TestCompleteIsVerifiedNotAccepted:
    """A connector that swallows a failure cannot claim COMPLETE.

    This proves the connector's HTTP goes through ``_get`` and therefore
    through ``ConnectorBase._request_with_retry``, which records every failed
    attempt on the retrieval audit *before* the connector's own ``except``
    clause runs.  It is the test that would catch a reintroduction of the
    donor's ``if response is None: return papers``.
    """

    def test_a_swallowed_page_failure_is_downgraded_to_partial(self, connector) -> None:
        """The audit contradicts a COMPLETE the connector still reported."""

        class _Liar(CoreConnector):
            """A connector with the donor's swallow and a COMPLETE claim."""

            def _build_coverage(self, run, *, retrieved, max_papers):
                """Claim COMPLETE unconditionally."""
                from lit_search.core.coverage import SourceCoverage

                return SourceCoverage.complete(self.name, retrieved, reported_total=retrieved)

            @staticmethod
            def _record_page_failure(run, exc):
                """Swallow the failure, as ``if response is None`` does."""
                return None

        liar = _Liar(api_key=_API_KEY)
        outcomes: list[Any] = [
            _response(_page([_record(0)], total=90)),
            requests.ConnectionError("dropped mid-run"),
        ]
        with (
            patch.object(liar, "_rate_limit"),
            patch.object(liar, "_request_with_retry_uncounted", side_effect=outcomes),
        ):
            result = liar.search(_parse("[machine learning]"))

        assert result.status is CoverageStatus.PARTIAL
        assert "ConnectionError" in (result.coverage.detail or "")
