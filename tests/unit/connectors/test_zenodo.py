"""Unit tests for ZenodoConnector.

These run under ``pytest-socket`` with ``--disable-socket`` (see
``pyproject.toml``), so every HTTP call is mocked and no test can reach the
network.  All fixture payloads mirror shapes observed against the live API on
2026-08-31.

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

from lit_search.connectors.zenodo import ZenodoBodyError, ZenodoConnector
from lit_search.core.coverage import CoverageStatus, CoveredPapers
from lit_search.core.paper import Database, PaperType
from lit_search.core.query import FilterCode, NodeType, Query, QueryNode
from lit_search.core.source import SourceType
from lit_search.query.builders.zenodo import ZenodoQueryBuilder

from .conftest import _parse

# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def _record(index: int, **overrides: Any) -> dict[str, Any]:
    """Build one Zenodo record.

    Parameters
    ----------
    index : int
        Used to make the title and identifiers distinct.
    **overrides : Any
        Fields to replace or add at the top level of the record.

    Returns
    -------
    dict[str, Any]
        A record shaped like a live ``hits.hits`` entry.
    """
    record: dict[str, Any] = {
        "id": 22000000 + index,
        "doi": f"10.5281/zenodo.{22000000 + index}",
        "links": {
            "self_html": f"https://zenodo.org/records/{22000000 + index}",
        },
        "files": [
            {
                "key": f"paper_{index}.pdf",
                "links": {"self": f"https://zenodo.org/api/records/{index}/files/x/content"},
            }
        ],
        "metadata": {
            "title": f"Synthetic Record {index}.",
            "description": f"<p>Synthetic abstract {index}.</p>",
            "publication_date": "2023-05-04",
            "access_right": "open",
            "creators": [{"name": f"Author {index}", "affiliation": "University of Toronto"}],
            "resource_type": {
                "title": "Journal article",
                "type": "publication",
                "subtype": "article",
            },
            "keywords": ["catalysis", "kinetics"],
        },
    }
    metadata_overrides = overrides.pop("metadata", None)
    record.update(overrides)
    if metadata_overrides is not None:
        record["metadata"] = {**record["metadata"], **metadata_overrides}
    return record


def _page(
    records: list[dict[str, Any]],
    total: object,
    next_link: str | None = None,
) -> dict[str, Any]:
    """Build one search-response body.

    Parameters
    ----------
    records : list[dict[str, Any]]
        Records for this page.
    total : object
        Value of ``hits.total``.  Typed as ``object`` rather than ``int``
        because one test supplies a value the connector cannot read, which
        is the case that must never become a ``COMPLETE``.
    next_link : str | None
        ``links.next`` to include.  ``None`` omits the key, which is how the
        live API signals the last page and a zero-hit result.

    Returns
    -------
    dict[str, Any]
        Response body.
    """
    body: dict[str, Any] = {
        "hits": {"hits": records, "total": total},
        "links": {"self": "https://zenodo.org/api/records?page=1"},
    }
    if next_link is not None:
        body["links"]["next"] = next_link
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
def connector() -> ZenodoConnector:
    """Return an anonymous Zenodo connector."""
    return ZenodoConnector()


def _search(connector: ZenodoConnector, responses: list[Any], **kwargs: Any) -> CoveredPapers:
    """Run a search against a scripted sequence of ``_get`` outcomes.

    Parameters
    ----------
    connector : ZenodoConnector
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
        assert connector.name == Database.ZENODO.value

    def test_min_request_interval_matches_the_documented_search_cap(self, connector) -> None:
        """Zenodo caps search endpoints at 30/min; 2.0s is exactly that."""
        assert connector.min_request_interval == 2.0

    def test_registered_in_the_search_registry(self) -> None:
        """A connector nobody can select is not ported."""
        from lit_search.connectors import SEARCH_REGISTRY

        assert SEARCH_REGISTRY[Database.ZENODO] is ZenodoConnector

    def test_quota_is_declared_because_the_limit_is_windowed(self, connector) -> None:
        """Interval pacing alone does not stop a burst after an idle period."""
        quota = connector.request_quota
        assert (quota.limit, quota.window_seconds) == (30, 60.0)

    def test_rate_limit_service_is_shared_across_zenodo_endpoints(self, connector) -> None:
        """The scope names the quota authority, not this one connector."""
        assert connector.rate_limit_service == "zenodo"

    def test_token_changes_the_rate_limit_bucket(self) -> None:
        """The token is the quota's identity, so it must reach the scope.

        Declaring ``_access_token`` rather than inheriting ``("_api_key",)`` is
        what stops two tokened processes sharing the anonymous clock.
        """
        anonymous = ZenodoConnector()
        tokened = ZenodoConnector(access_token="secret-token")
        assert anonymous.rate_limit_scope.fingerprint == "anon"
        assert tokened.rate_limit_scope.fingerprint != "anon"
        assert tokened.rate_limit_scope.service == anonymous.rate_limit_scope.service


# ---------------------------------------------------------------------------
# The four required cases, plus the one that matters
# ---------------------------------------------------------------------------


class TestCoverageIsHonest:
    """Each way a search can end reports a different, actionable status."""

    def test_normal_result_is_complete(self, connector) -> None:
        """Retrieving as many records as hits.total reports is COMPLETE."""
        result = _search(connector, [_response(_page([_record(0), _record(1)], total=2))])
        assert result.status is CoverageStatus.COMPLETE
        assert len(result.papers) == 2
        assert result.coverage.reported_total == 2

    def test_upstream_failure_is_failed(self, connector) -> None:
        """A dead first request is FAILED, and the error is carried."""
        result = _search(connector, [requests.ConnectionError("zenodo unreachable")])
        assert result.status is CoverageStatus.FAILED
        assert result.papers == ()
        assert "unreachable" in (result.coverage.error or "")
        assert result.coverage.error_type == "ConnectionError"

    def test_empty_but_successful_search_is_complete(self, connector) -> None:
        """A genuine zero-hit query is COMPLETE with no papers.

        This is the honest empty: Zenodo answered, said zero records match, and
        zero came back.
        """
        result = _search(connector, [_response(_page([], total=0))])
        assert result.status is CoverageStatus.COMPLETE
        assert result.papers == ()
        assert result.coverage.reported_total == 0

    def test_capped_result_is_capped(self, connector) -> None:
        """Stopping at max_papers is CAPPED and names the cap."""
        result = _search(
            connector,
            [_response(_page([_record(i) for i in range(3)], total=900, next_link="p2"))],
            max_papers=3,
        )
        assert result.status is CoverageStatus.CAPPED
        assert len(result.papers) == 3
        assert result.coverage.cap == 3
        assert result.coverage.reported_total == 900

    def test_failure_and_emptiness_are_different_statuses(self, connector) -> None:
        """The regression test for the whole port.

        In the donor implementation both of these paths end at the same
        ``return []`` -- ``zenodo.py:101-103`` catches bare ``Exception`` and
        returns an empty list -- so a caller cannot tell an outage from a
        repository that holds nothing.  Here they must differ, and the failure
        must not be readable as a complete answer.
        """
        failure = _search(connector, [requests.ConnectionError("boom")])
        empty = _search(connector, [_response(_page([], total=0))])

        assert failure.papers == empty.papers == ()
        assert failure.status is not empty.status
        assert failure.status is CoverageStatus.FAILED
        assert empty.status is CoverageStatus.COMPLETE
        assert not failure.coverage.is_complete
        assert empty.coverage.is_complete

    def test_missing_credential_is_not_a_status_zenodo_can_reach(self, connector) -> None:
        """Zenodo search works anonymously, so a token is never required.

        Recorded as a test because the opposite mistake -- marking an
        anonymously-usable source SKIPPED_NO_CREDENTIAL -- would drop a whole
        database from a search while looking like careful bookkeeping.
        """
        result = _search(connector, [_response(_page([_record(0)], total=1))])
        assert result.status is CoverageStatus.COMPLETE


# ---------------------------------------------------------------------------
# The 200-with-an-unusable-body family
# ---------------------------------------------------------------------------


class TestTwoHundredWithAnUnusableBody:
    """A 200 that is not a search result must not read as zero records."""

    def test_missing_hits_envelope_is_failed_not_empty(self, connector) -> None:
        """Without ``hits`` the body is not a search result at all."""
        result = _search(connector, [_response({"links": {}})])
        assert result.status is CoverageStatus.FAILED
        assert "hits" in (result.coverage.error or "")

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

        assert isinstance(ZenodoBodyError("x"), ConnectorError)

    def test_an_unreadable_total_cannot_yield_complete(self, connector) -> None:
        """A total the connector cannot read must not become a COMPLETE.

        InvenioRDM also serialises ``hits.total`` as an object, so a build that
        assumed one shape and got the other would compute no total.  Without a
        source-reported count, completeness is unverifiable by construction.
        """
        result = _search(connector, [_response(_page([_record(0)], total="many"))])
        assert result.status is CoverageStatus.PARTIAL
        assert result.coverage.reported_total is None
        assert "no result total" in (result.coverage.detail or "")

    def test_object_shaped_total_is_read(self, connector) -> None:
        """The ``{"value": n}`` form is accepted, so it does not lose the total."""
        body = _page([_record(0)], total=0)
        body["hits"]["total"] = {"value": 1, "relation": "eq"}
        result = _search(connector, [_response(body)])
        assert result.status is CoverageStatus.COMPLETE
        assert result.coverage.reported_total == 1


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class TestPagination:
    """Every abnormal stop reaches the coverage status instead of the log."""

    def test_absent_next_link_ends_a_single_page_result(self, connector) -> None:
        """Zenodo omits links.next on the last page."""
        result = _search(connector, [_response(_page([_record(0)], total=1))])
        assert result.status is CoverageStatus.COMPLETE
        assert len(result.papers) == 1

    def test_multi_page_follows_the_next_link(self, connector) -> None:
        """Pages are consumed while links.next is offered."""
        result = _search(
            connector,
            [
                _response(_page([_record(0), _record(1)], total=3, next_link="p2")),
                _response(_page([_record(2)], total=3)),
            ],
        )
        assert result.status is CoverageStatus.COMPLETE
        assert len(result.papers) == 3

    def test_short_page_alone_does_not_end_pagination(self, connector) -> None:
        """A page smaller than the page size keeps going while a link is offered.

        The short-page heuristic the donor uses on other sources was proved
        wrong on a sibling API by the reference port; ``links.next`` is the
        signal here and a short page is not consulted.
        """
        result = _search(
            connector,
            [
                _response(_page([_record(0)], total=2, next_link="p2")),
                _response(_page([_record(1)], total=2)),
            ],
        )
        assert len(result.papers) == 2
        assert result.status is CoverageStatus.COMPLETE

    def test_next_link_withdrawn_before_the_total_is_partial(self, connector) -> None:
        """Fewer records than hits.total with no further page is PARTIAL."""
        result = _search(connector, [_response(_page([_record(0)], total=50))])
        assert result.status is CoverageStatus.PARTIAL
        assert result.coverage.reported_total == 50
        assert "50" in (result.coverage.detail or "")

    def test_failure_after_some_pages_is_partial_not_failed(self, connector) -> None:
        """Records that did arrive are real, so the run is PARTIAL.

        FAILED would discard true records; COMPLETE would hide the shortfall.
        """
        result = _search(
            connector,
            [
                _response(_page([_record(0), _record(1)], total=90, next_link="p2")),
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
                _response(_page([_record(0), _record(1)], total=3, next_link="p2")),
                _response(_page([_record(2)], total=7)),
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert "changed mid-retrieval" in (result.coverage.detail or "")

    def test_unparsable_records_are_counted_as_a_shortfall(self, connector) -> None:
        """A record seen but not parsed is a coverage loss, not an absence."""
        result = _search(
            connector,
            [_response(_page([_record(0), _record(1, metadata={"title": "  "})], total=2))],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 1
        assert "could not be parsed" in (result.coverage.detail or "")

    def test_the_ten_thousand_record_window_is_a_reported_shortfall(self, connector) -> None:
        """Zenodo refuses page*size beyond 10,000 with HTTP 400.

        Reaching that ceiling is a real limit on recall.  Stopping there
        silently would report a truncated set as an exhausted one.
        """
        pages = [
            _response(_page([_record(index)], total=500_000, next_link="next"))
            for index in range(500)
        ]
        result = _search(connector, pages)
        assert result.status is CoverageStatus.PARTIAL
        assert "10000" in (result.coverage.detail or "")

    def test_page_size_is_constant_so_the_window_never_rewinds(self, connector) -> None:
        """Every request in one run carries the same ``size``.

        Zenodo's window is ``records[(page - 1) * size : page * size]``, so
        ``page`` and ``size`` are read together.  A ``size`` that tracked the
        remaining cap sent ``page=1&size=25`` then ``page=2&size=5``, and the
        second request returns records 6-10 rather than 26-30 -- verified live
        2026-08-31.  The run then reported ``capped (30 papers)`` while holding
        25 distinct records and 5 duplicates.
        """
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(
                connector,
                "_get",
                side_effect=[
                    _response(_page([_record(i) for i in range(25)], total=3328, next_link="p2")),
                    _response(
                        _page([_record(i) for i in range(25, 50)], total=3328, next_link="p3")
                    ),
                ],
            ) as mocked,
        ):
            result = connector.search(_parse("[machine learning]"), max_papers=30)

        sizes = [call.args[1]["size"] for call in mocked.call_args_list]
        pages = [call.args[1]["page"] for call in mocked.call_args_list]
        assert len(set(sizes)) == 1, (
            f"page size changed mid-run: {list(zip(pages, sizes, strict=True))}"
        )
        assert pages == [1, 2]
        assert len({paper.title for paper in result.papers}) == len(result.papers) == 30

    def test_empty_page_offering_a_next_link_stops_and_is_reported(self, connector) -> None:
        """A page carrying nothing while still linking on cannot advance the run.

        Following it re-asks the same question until the 10,000-record window
        closes -- 400 requests at two seconds each -- and stopping without
        saying so would under-report.  OpenAIRE's ``_advance`` already refuses
        this shape; the two siblings agree.
        """
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(
                connector,
                "_get",
                side_effect=[
                    _response(_page([_record(0)], total=500, next_link="p2")),
                    *[_response(_page([], total=500, next_link="p3")) for _ in range(10)],
                ],
            ) as mocked,
        ):
            result = connector.search(_parse("[machine learning]"))

        assert mocked.call_count == 2
        assert result.status is CoverageStatus.PARTIAL
        assert "empty page" in (result.coverage.detail or "")
        assert len(result.papers) == 1

    def test_coverage_count_always_matches_the_papers_returned(self, connector) -> None:
        """CoveredPapers rejects a mismatch, so this asserts the accounting."""
        result = _search(
            connector,
            [_response(_page([_record(i) for i in range(5)], total=99, next_link="p2"))],
            max_papers=2,
        )
        assert result.coverage.retrieved == len(result.papers) == 2


# ---------------------------------------------------------------------------
# Request shaping
# ---------------------------------------------------------------------------


class TestRequestShaping:
    """Parameters sent upstream match what the live API accepts."""

    def _capture(self, connector, **kwargs) -> dict[str, Any]:
        """Run one search and return the parameters of the first request."""
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(connector, "_get", return_value=_response(_page([], total=0))) as mocked,
        ):
            connector.search(_parse("[machine learning]"), **kwargs)
        params: dict[str, Any] = mocked.call_args.args[1]
        return params

    def test_anonymous_page_size_never_exceeds_twenty_five(self, connector) -> None:
        """size=26 without a token is refused with HTTP 400.

        The donor sends ``min(max_results, 200)``, which the live API rejects
        for any cap above 25 -- and its ``except Exception: return []`` then
        reports that rejection as an empty repository.
        """
        params = self._capture(connector, max_papers=100_000)
        assert 1 <= params["size"] <= 25

    def test_authenticated_page_size_may_reach_one_hundred(self) -> None:
        """A token raises the documented ceiling from 25 to 100."""
        tokened = ZenodoConnector(access_token="secret-token")
        with (
            patch.object(tokened, "_rate_limit"),
            patch.object(tokened, "_get", return_value=_response(_page([], total=0))) as mocked,
        ):
            tokened.search(_parse("[machine learning]"), max_papers=100_000)
        assert mocked.call_args.args[1]["size"] == 100

    def test_page_size_is_at_least_one_when_the_cap_is_one(self, connector) -> None:
        """A cap of 1 must not produce size=0, which the API rejects."""
        assert self._capture(connector, max_papers=1)["size"] >= 1

    def test_results_are_restricted_to_publications(self, connector) -> None:
        """Datasets, software and images are not papers."""
        assert self._capture(connector)["type"] == "publication"

    def test_ordering_is_by_relevance_not_recency(self, connector) -> None:
        """The donor's ``mostrecent`` returns the newest N under a cap.

        For a recall tool that is a materially different set from the
        best-matching N, and the difference is invisible in the output.
        """
        assert self._capture(connector)["sort"] == "bestmatch"

    def test_date_bounds_become_a_publication_date_filter(self, connector) -> None:
        """Both bounds render as an inclusive publication_date range."""
        params = self._capture(
            connector,
            since=datetime.date(2020, 1, 1),
            until=datetime.date(2021, 12, 31),
        )
        assert "publication_date:[2020-01-01 TO 2021-12-31]" in params["q"]

    def test_no_date_filter_when_no_bounds_are_given(self, connector) -> None:
        """An unbounded search must not silently acquire a date restriction."""
        assert "publication_date" not in self._capture(connector)["q"]

    def test_the_token_is_sent_as_a_bearer_header(self) -> None:
        """A configured token must actually reach the request."""
        headers = ZenodoConnector(access_token="secret-token")._prepare_headers({})
        assert headers["Authorization"] == "Bearer secret-token"

    def test_no_authorization_header_without_a_token(self, connector) -> None:
        """An anonymous client must not send an empty credential."""
        assert "Authorization" not in connector._prepare_headers({})


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    """Field mapping from a Zenodo record to a Paper."""

    def test_core_fields(self, connector) -> None:
        """Title, abstract, DOI, date, URLs and keywords are mapped."""
        paper = connector._parse_record(_record(7))
        assert paper is not None
        assert paper.title == "Synthetic Record 7."
        assert paper.doi == "10.5281/zenodo.22000007"
        assert paper.publication_date == datetime.date(2023, 5, 4)
        assert paper.url == "https://zenodo.org/records/22000007"
        assert paper.pdf_url is not None
        assert paper.keywords == {"catalysis", "kinetics"}
        assert paper.is_open_access is True
        assert Database.ZENODO in paper.found_in

    def test_html_is_stripped_from_the_description(self, connector) -> None:
        """Zenodo descriptions are HTML; an abstract is not."""
        paper = connector._parse_record(_record(3))
        assert paper is not None
        assert paper.abstract == "Synthetic abstract 3."

    def test_creators_carry_affiliations(self, connector) -> None:
        """Affiliation lives beside the creator name."""
        paper = connector._parse_record(_record(1))
        assert paper is not None
        assert paper.authors[0].name == "Author 1"
        assert paper.authors[0].affiliation == "University of Toronto"

    def test_split_creator_names_are_joined(self, connector) -> None:
        """Some records carry given_name/family_name instead of name."""
        record = _record(
            2, metadata={"creators": [{"given_name": "Ada", "family_name": "Lovelace"}]}
        )
        paper = connector._parse_record(record)
        assert paper is not None
        assert paper.authors[0].name == "Ada Lovelace"

    def test_source_is_the_repository(self, connector) -> None:
        """Zenodo is a repository, and the record names no journal."""
        paper = connector._parse_record(_record(0))
        assert paper is not None
        assert paper.source is not None
        assert paper.source.source_type is SourceType.REPOSITORY

    def test_preprint_subtype_becomes_unpublished(self, connector) -> None:
        """A preprint is not a published article."""
        record = _record(
            4, metadata={"resource_type": {"type": "publication", "subtype": "preprint"}}
        )
        paper = connector._parse_record(record)
        assert paper is not None
        assert paper.paper_type is PaperType.UNPUBLISHED

    def test_closed_access_is_not_asserted_as_closed(self, connector) -> None:
        """Only the positive open-access case is claimed."""
        record = _record(5, metadata={"access_right": "restricted"})
        paper = connector._parse_record(record)
        assert paper is not None
        assert paper.is_open_access is None

    def test_untitled_record_is_rejected(self, connector) -> None:
        """A record with no title cannot become a Paper."""
        assert connector._parse_record(_record(0, metadata={"title": ""})) is None

    def test_non_dict_record_is_rejected(self, connector) -> None:
        """A malformed entry is counted as unparsed, not crashed on."""
        assert connector._parse_record("not a record") is None

    def test_url_falls_back_to_the_record_id(self, connector) -> None:
        """A record with no links still gets a resolvable URL."""
        paper = connector._parse_record(_record(9, links={}))
        assert paper is not None
        assert paper.url == "https://zenodo.org/records/22000009"


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------


class TestQueryBuilder:
    """Only verified field names are emitted, and unsafe terms are refused."""

    @pytest.fixture
    def builder(self) -> ZenodoQueryBuilder:
        """Return a Zenodo query builder."""
        return ZenodoQueryBuilder()

    def test_default_filter_expands_to_the_verified_disjunction(self, builder) -> None:
        """Zenodo has no combined field, so the disjunction is written out."""
        converted = builder.convert_query(_parse("[machine learning]"))
        assert 'title:"machine learning"' in converted
        assert 'description:"machine learning"' in converted
        assert 'keywords:"machine learning"' in converted

    @pytest.mark.parametrize(
        ("query_string", "expected"),
        [
            ("ti[catalysis]", 'title:"catalysis"'),
            ("abs[catalysis]", 'description:"catalysis"'),
            ("key[catalysis]", 'keywords:"catalysis"'),
            ("au[Zhang]", 'creators.name:"Zhang"'),
        ],
    )
    def test_each_filter_maps_to_a_verified_field(self, builder, query_string, expected) -> None:
        """``abstract:`` and ``authors:`` both return zero live and are not used."""
        assert builder.convert_query(_parse(query_string)) == expected

    def test_connectors_map_to_zenodo_operators(self, builder) -> None:
        """AND, OR and AND NOT render as Lucene operators."""
        converted = builder.convert_query(_parse("ti[a] AND NOT ti[b]"))
        assert "NOT" in converted

    def test_wildcard_terms_are_left_unquoted(self, builder) -> None:
        """Zenodo expands wildcards, but not inside a quoted phrase."""
        assert builder.convert_query(_parse("ti[catalys*]")) == "title:catalys*"

    def test_empty_term_is_rejected(self, builder) -> None:
        """An empty query returned 7.2 million records live.

        The query *parser* already forbids an empty term, so this query is
        built directly.  The check is a second line of defence, and it is worth
        having precisely because the first line is somewhere else: any future
        path that constructs a ``Query`` without the parser -- an MCP tool
        schema, a stored handle, a programmatic caller -- would otherwise reach
        Zenodo with a match-all.
        """
        blank = Query(
            raw_query="ti[]",
            root=QueryNode(
                node_type=NodeType.ROOT,
                children=[
                    QueryNode(
                        node_type=NodeType.TERM,
                        value="   ",
                        filter_code=FilterCode.TITLE,
                    )
                ],
            ),
        )
        result = builder.validate_query(blank)
        assert result.is_valid is False
        assert "match-all" in (result.error_message or "")

    def test_unbalanced_brackets_are_rejected(self, builder) -> None:
        """'title:(((' returned 47,131 records with HTTP 200."""
        result = builder.validate_query(_parse("ti[catalysis (((]"))
        assert result.is_valid is False
        assert "unbalanced" in (result.error_message or "")

    def test_wildcard_with_a_space_is_rejected(self, builder) -> None:
        """Such a term would search a different scope than requested."""
        result = builder.validate_query(_parse("ti[machine learn*]"))
        assert result.is_valid is False

    def test_unindexed_filter_is_rejected(self, builder) -> None:
        """Zenodo indexes no journal title, so SOURCE has no faithful field."""
        result = builder.validate_query(_parse("src[Nature]"))
        assert result.is_valid is False

    def test_rejected_query_reaches_the_ledger_as_query_unsupported(self, connector) -> None:
        """A refused query must not spend a request, and must say why."""
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(connector, "_get") as mocked,
        ):
            result = connector.search(_parse("src[Nature]"))
        assert result.status is CoverageStatus.QUERY_UNSUPPORTED
        assert result.papers == ()
        mocked.assert_not_called()


# ---------------------------------------------------------------------------
# The base class checks COMPLETE rather than accepting it
# ---------------------------------------------------------------------------


class TestCompleteIsVerifiedNotAccepted:
    """A connector that swallows a failure cannot claim COMPLETE.

    This proves the connector's HTTP goes through ``_get`` and therefore
    through ``ConnectorBase._request_with_retry``, which records every failed
    attempt on the retrieval audit *before* the connector's own ``except``
    clause runs.  Patching ``_get`` -- as every other test here does -- would
    bypass that channel, so this one patches the retry loop's inner half and
    leaves the counting half in place.
    """

    def test_a_swallowed_page_failure_is_downgraded_to_partial(self, connector) -> None:
        """The audit contradicts a COMPLETE the connector still reported."""

        class _Liar(ZenodoConnector):
            """A connector that reports COMPLETE whatever happened."""

            def _build_coverage(self, run, *, retrieved, max_papers):
                """Claim COMPLETE unconditionally."""
                from lit_search.core.coverage import SourceCoverage

                return SourceCoverage.complete(self.name, retrieved, reported_total=retrieved)

            @staticmethod
            def _record_page_failure(run, exc):
                """Swallow the failure and keep whatever was retrieved."""
                return None

        liar = _Liar()
        outcomes: list[Any] = [
            _response(_page([_record(0)], total=90, next_link="p2")),
            requests.ConnectionError("dropped mid-run"),
        ]
        with (
            patch.object(liar, "_rate_limit"),
            patch.object(liar, "_request_with_retry_uncounted", side_effect=outcomes),
        ):
            result = liar.search(_parse("[machine learning]"))

        assert result.status is CoverageStatus.PARTIAL
        assert "ConnectionError" in (result.coverage.detail or "")
