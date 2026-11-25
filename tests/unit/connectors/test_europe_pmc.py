"""Unit tests for EuropePMCConnector.

These run under ``pytest-socket`` with ``--disable-socket`` (see
``pyproject.toml``), so every HTTP call is mocked and no test can reach the
network.  All fixture payloads mirror shapes observed against the live API on
2026-08-30 (service version 6.9).

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

from lit_search.connectors.europe_pmc import (
    EuropePMCBodyError,
    EuropePMCConnector,
)
from lit_search.core.coverage import CoverageStatus, CoveredPapers
from lit_search.core.paper import Database, PaperType
from lit_search.core.source import SourceType
from lit_search.query.builders.europe_pmc import EuropePMCQueryBuilder

from .conftest import _parse

# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def _record(index: int, **overrides: Any) -> dict[str, Any]:
    """Build one Europe PMC ``core`` record.

    Parameters
    ----------
    index : int
        Used to make the title and identifiers distinct.
    **overrides : Any
        Fields to replace or add.

    Returns
    -------
    dict[str, Any]
        A record shaped like a live ``resultList.result`` entry.
    """
    record: dict[str, Any] = {
        "id": f"{30000000 + index}",
        "source": "MED",
        "pmid": f"{30000000 + index}",
        "pmcid": f"PMC{4000000 + index}",
        "doi": f"10.1000/test.{index}",
        "title": f"Synthetic Paper {index}.",
        "abstractText": f"Synthetic abstract {index}.",
        "pubYear": "2023",
        "firstPublicationDate": "2023-05-04",
        "language": "eng",
        "isOpenAccess": "Y",
        "citedByCount": index,
        "authorList": {
            "author": [
                {
                    "fullName": f"Author {index}",
                    "authorAffiliationDetailsList": {
                        "authorAffiliation": [{"affiliation": "University of Toronto"}]
                    },
                }
            ]
        },
        "journalInfo": {"journal": {"title": "Nature", "issn": "0028-0836"}},
        "pubTypeList": {"pubType": ["Journal Article"]},
        "fullTextUrlList": {
            "fullTextUrl": [
                {"documentStyle": "html", "url": f"https://europepmc.org/article/MED/{index}"},
                {"documentStyle": "pdf", "url": f"https://europepmc.org/pdf/{index}"},
            ]
        },
    }
    record.update(overrides)
    return record


def _page(
    records: list[dict[str, Any]],
    hit_count: int,
    next_cursor: str | None = None,
) -> dict[str, Any]:
    """Build one search-response body.

    Parameters
    ----------
    records : list[dict[str, Any]]
        Records for this page.
    hit_count : int
        Value of the ``hitCount`` field.
    next_cursor : str | None
        ``nextCursorMark`` to include.  ``None`` omits the key, which is how
        the live API signals a single-page result set.

    Returns
    -------
    dict[str, Any]
        Response body.
    """
    body: dict[str, Any] = {
        "version": "6.9",
        "hitCount": hit_count,
        "resultList": {"result": records},
    }
    if next_cursor is not None:
        body["nextCursorMark"] = next_cursor
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
def connector() -> EuropePMCConnector:
    """Return a connector with rate limiting disabled for the test."""
    searcher = EuropePMCConnector()
    return searcher


def _search(connector: EuropePMCConnector, responses: list[Any], **kwargs: Any) -> CoveredPapers:
    """Run a search against a scripted sequence of ``_get`` outcomes.

    Parameters
    ----------
    connector : EuropePMCConnector
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
        assert connector.name == Database.EUROPE_PMC.value

    def test_min_request_interval_is_declared(self, connector) -> None:
        """The declared interval is the inferred 1.0s from the rate table."""
        assert connector.min_request_interval == 1.0

    def test_registered_in_the_search_registry(self) -> None:
        """A connector nobody can select is not ported."""
        from lit_search.connectors import SEARCH_REGISTRY

        assert SEARCH_REGISTRY[Database.EUROPE_PMC] is EuropePMCConnector

    def test_no_credential_is_declared_explicitly(self, connector) -> None:
        """Europe PMC has no key, so the anonymous host bucket is correct."""
        assert connector._credential_attrs == ()
        assert connector.rate_limit_scope.service == "europepmc"
        assert connector.rate_limit_scope.fingerprint == "anon"

    def test_rate_limit_service_is_shared_across_europe_pmc_endpoints(self, connector) -> None:
        """The scope names the quota authority, not this one connector."""
        assert connector.rate_limit_service == "europepmc"


# ---------------------------------------------------------------------------
# The four required cases, plus the one that matters
# ---------------------------------------------------------------------------


class TestCoverageIsHonest:
    """Each way a search can end reports a different, actionable status."""

    def test_normal_result_is_complete(self, connector) -> None:
        """Retrieving as many records as hitCount reports is COMPLETE."""
        result = _search(connector, [_response(_page([_record(0), _record(1)], hit_count=2))])
        assert result.status is CoverageStatus.COMPLETE
        assert len(result.papers) == 2
        assert result.coverage.retrieved == 2
        assert result.coverage.reported_total == 2

    def test_upstream_failure_is_failed(self, connector) -> None:
        """A dead first request is FAILED, and the error is carried."""
        result = _search(connector, [requests.ConnectionError("europepmc unreachable")])
        assert result.status is CoverageStatus.FAILED
        assert result.papers == ()
        assert "unreachable" in (result.coverage.error or "")
        assert result.coverage.error_type == "ConnectionError"

    def test_empty_but_successful_search_is_complete(self, connector) -> None:
        """A genuine zero-hit query is COMPLETE with no papers.

        This is the honest empty: Europe PMC answered, said zero records match,
        and zero came back.
        """
        result = _search(connector, [_response(_page([], hit_count=0))])
        assert result.status is CoverageStatus.COMPLETE
        assert result.papers == ()
        assert result.coverage.reported_total == 0

    def test_capped_result_is_capped(self, connector) -> None:
        """Stopping at max_papers is CAPPED and names the cap."""
        result = _search(
            connector,
            [_response(_page([_record(i) for i in range(3)], hit_count=900, next_cursor="c1"))],
            max_papers=3,
        )
        assert result.status is CoverageStatus.CAPPED
        assert len(result.papers) == 3
        assert result.coverage.cap == 3
        assert result.coverage.reported_total == 900

    def test_failure_and_emptiness_are_different_statuses(self, connector) -> None:
        """The regression test for the whole port.

        In the donor implementation both of these paths end at the same
        ``return papers`` with ``papers == []``, so a caller cannot tell an
        outage from a database that holds nothing.  Here they must differ, and
        the failure must not be readable as a complete answer.
        """
        failure = _search(connector, [requests.ConnectionError("boom")])
        empty = _search(connector, [_response(_page([], hit_count=0))])

        assert failure.papers == empty.papers == ()
        assert failure.status is not empty.status
        assert failure.status is CoverageStatus.FAILED
        assert empty.status is CoverageStatus.COMPLETE
        assert not failure.coverage.is_complete
        assert empty.coverage.is_complete


# ---------------------------------------------------------------------------
# The 200-with-an-error-body family
# ---------------------------------------------------------------------------


class TestTwoHundredWithAnErrorBody:
    """Europe PMC reports several errors with an HTTP 200 status.

    Each of these would satisfy ``raise_for_status`` and then yield ``[]`` from
    ``data.get("resultList", {}).get("result", [])``.
    """

    def test_err_code_body_about_the_request_is_failed(self, connector) -> None:
        """``pageSize=1001`` shape: HTTP 200, errCode, no resultList."""
        body = {"errCode": 404, "errMsg": "Invalid page size provided. Valid size is 1 to 1000"}
        result = _search(connector, [_response(body)])
        assert result.status is CoverageStatus.FAILED
        assert "Invalid page size" in (result.coverage.error or "")

    def test_err_code_body_about_the_criteria_is_query_unsupported(self, connector) -> None:
        """A body blaming the search criteria is a query problem, not an outage.

        The distinction is actionable: FAILED invites a retry, and retrying a
        query the source cannot express only wastes the rate-limit budget.
        """
        body = {
            "errCode": 404,
            "errMsg": "No search criteria provided. Please provide a search criteria.",
        }
        result = _search(connector, [_response(body)])
        assert result.status is CoverageStatus.QUERY_UNSUPPORTED
        assert result.papers == ()

    def test_missing_hit_count_is_failed_not_empty(self, connector) -> None:
        """A malformed cursor returns ``{"version": "6.9"}`` with HTTP 200.

        No hitCount and no resultList. Reading that as zero results is exactly
        the failure this contract exists to prevent.
        """
        result = _search(connector, [_response({"version": "6.9"})])
        assert result.status is CoverageStatus.FAILED
        assert "hitCount" in (result.coverage.error or "")

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

    def test_body_error_classification(self) -> None:
        """The exception carries the query-versus-request distinction."""
        assert EuropePMCBodyError("x", is_query_error=True).is_query_error is True
        assert EuropePMCBodyError("x").is_query_error is False


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class TestCursorPagination:
    """Both live exhaustion signals, and every abnormal stop, are handled."""

    def test_single_page_result_omits_next_cursor(self, connector) -> None:
        """A result set that fits one page has no nextCursorMark."""
        result = _search(connector, [_response(_page([_record(0)], hit_count=1))])
        assert result.status is CoverageStatus.COMPLETE
        assert len(result.papers) == 1

    def test_multi_page_ends_when_the_cursor_repeats(self, connector) -> None:
        """A multi-page set ends with an empty page repeating the cursor.

        A short page is deliberately not treated as the end: an observed 9-hit
        query returned pages of 2, 2, 2, 2, 1, 0.
        """
        result = _search(
            connector,
            [
                _response(_page([_record(0), _record(1)], hit_count=3, next_cursor="c1")),
                _response(_page([_record(2)], hit_count=3, next_cursor="c2")),
                _response(_page([], hit_count=3, next_cursor="c2")),
            ],
        )
        assert result.status is CoverageStatus.COMPLETE
        assert len(result.papers) == 3

    def test_short_page_alone_does_not_end_pagination(self, connector) -> None:
        """A page smaller than the page size keeps going while a cursor is offered."""
        result = _search(
            connector,
            [
                _response(_page([_record(0)], hit_count=2, next_cursor="c1")),
                _response(_page([_record(1)], hit_count=2, next_cursor="c1")),
            ],
        )
        assert len(result.papers) == 2
        assert result.status is CoverageStatus.COMPLETE

    def test_cursor_exhausted_before_hit_count_is_partial(self, connector) -> None:
        """Fewer records than hitCount, with the cursor spent, is PARTIAL."""
        result = _search(
            connector,
            [_response(_page([_record(0)], hit_count=50))],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert result.coverage.reported_total == 50
        assert "50" in (result.coverage.detail or "")

    def test_empty_page_with_a_fresh_cursor_is_partial(self, connector) -> None:
        """The API offered more and delivered nothing: stop and say so.

        Continuing would spin; stopping without a note would under-report.
        """
        result = _search(
            connector,
            [
                _response(_page([_record(0)], hit_count=9, next_cursor="c1")),
                _response(_page([], hit_count=9, next_cursor="c2")),
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert "empty page" in (result.coverage.detail or "")

    def test_failure_after_some_pages_is_partial_not_failed(self, connector) -> None:
        """Records that did arrive are real, so the run is PARTIAL.

        FAILED would discard true records; COMPLETE would hide the shortfall.
        """
        result = _search(
            connector,
            [
                _response(_page([_record(0), _record(1)], hit_count=90, next_cursor="c1")),
                requests.ConnectionError("dropped mid-run"),
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 2
        assert result.coverage.error_type == "ConnectionError"
        assert "dropped mid-run" in (result.coverage.detail or "")

    def test_body_error_after_some_pages_is_partial(self, connector) -> None:
        """A 200-with-errCode partway through degrades rather than discards."""
        result = _search(
            connector,
            [
                _response(_page([_record(0)], hit_count=90, next_cursor="c1")),
                _response({"errCode": 500, "errMsg": "internal"}),
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 1

    def test_hit_count_drift_between_pages_is_partial(self, connector) -> None:
        """A moving index means neither total can be checked.

        Retrieving 3 of a first-page hitCount of 3 would otherwise read as
        COMPLETE even though the index changed under the cursor.
        """
        result = _search(
            connector,
            [
                _response(_page([_record(0), _record(1)], hit_count=3, next_cursor="c1")),
                _response(_page([_record(2)], hit_count=7, next_cursor="c1")),
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert "changed mid-retrieval" in (result.coverage.detail or "")

    def test_unparsable_records_are_counted_as_a_shortfall(self, connector) -> None:
        """A record seen but not parsed is a coverage loss, not an absence.

        Silently returning fewer papers is how a schema drift becomes invisible.
        """
        result = _search(
            connector,
            [_response(_page([_record(0), _record(1, title="   ")], hit_count=2))],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 1
        assert "could not be parsed" in (result.coverage.detail or "")

    def test_coverage_count_always_matches_the_papers_returned(self, connector) -> None:
        """CoveredPapers rejects a mismatch, so this asserts the accounting."""
        result = _search(
            connector,
            [_response(_page([_record(i) for i in range(5)], hit_count=99, next_cursor="c1"))],
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
            patch.object(
                connector, "_get", return_value=_response(_page([], hit_count=0))
            ) as mocked,
        ):
            connector.search(_parse("[machine learning]"), **kwargs)
        params: dict[str, Any] = mocked.call_args.args[1]
        return params

    def test_core_result_type_and_cursor_start(self, connector) -> None:
        """resultType=core is required for abstracts; the cursor starts at '*'."""
        params = self._capture(connector)
        assert params["resultType"] == "core"
        assert params["cursorMark"] == "*"
        assert params["format"] == "json"

    def test_page_size_never_exceeds_the_api_maximum(self, connector) -> None:
        """pageSize > 1000 is refused upstream with an errCode body."""
        params = self._capture(connector, max_papers=100_000)
        assert 1 <= params["pageSize"] <= 1000

    def test_page_size_is_at_least_one_when_the_cap_is_one(self, connector) -> None:
        """A cap of 1 must not produce pageSize=0, which the API rejects."""
        params = self._capture(connector, max_papers=1)
        assert params["pageSize"] >= 1

    def test_date_bounds_become_a_first_pdate_filter(self, connector) -> None:
        """Both bounds render as an inclusive FIRST_PDATE range."""
        params = self._capture(
            connector,
            since=datetime.date(2020, 1, 1),
            until=datetime.date(2021, 12, 31),
        )
        assert "FIRST_PDATE:[2020-01-01 TO 2021-12-31]" in params["query"]

    def test_open_upper_bound_uses_the_wildcard(self, connector) -> None:
        """An open end uses '*', verified equivalent to a wide explicit range."""
        params = self._capture(connector, since=datetime.date(2020, 1, 1))
        assert "FIRST_PDATE:[2020-01-01 TO *]" in params["query"]

    def test_no_date_filter_when_no_bounds_are_given(self, connector) -> None:
        """An unbounded search must not silently acquire a date restriction."""
        params = self._capture(connector)
        assert "FIRST_PDATE" not in params["query"]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    """Field mapping from a Europe PMC ``core`` record to a Paper."""

    def test_core_fields(self, connector) -> None:
        """Title, abstract, DOI, date, URLs and citations are mapped."""
        paper = connector._parse_item(_record(7))
        assert paper is not None
        assert paper.title == "Synthetic Paper 7."
        assert paper.abstract == "Synthetic abstract 7."
        assert paper.doi == "10.1000/test.7"
        assert paper.publication_date == datetime.date(2023, 5, 4)
        assert paper.url == "https://europepmc.org/article/MED/7"
        assert paper.pdf_url == "https://europepmc.org/pdf/7"
        assert paper.citations == 7
        assert paper.language == "en"
        assert paper.is_open_access is True
        assert Database.EUROPE_PMC in paper.found_in

    def test_authors_carry_affiliations(self, connector) -> None:
        """Affiliation lives under authorAffiliationDetailsList."""
        paper = connector._parse_item(_record(1))
        assert paper is not None
        assert paper.authors[0].name == "Author 1"
        assert paper.authors[0].affiliation == "University of Toronto"

    def test_journal_becomes_a_journal_source(self, connector) -> None:
        """journalInfo.journal maps to a JOURNAL source with its ISSN."""
        paper = connector._parse_item(_record(1))
        assert paper is not None
        assert paper.source is not None
        assert paper.source.title == "Nature"
        assert paper.source.issn == "0028-0836"
        assert paper.source.source_type is SourceType.JOURNAL
        assert paper.paper_type is PaperType.ARTICLE

    def test_preprint_source_becomes_a_repository(self, connector) -> None:
        """A PPR record with no journal is an unpublished repository record."""
        paper = connector._parse_item(_record(2, source="PPR", journalInfo={}))
        assert paper is not None
        assert paper.source is not None
        assert paper.source.source_type is SourceType.REPOSITORY
        assert paper.paper_type is PaperType.UNPUBLISHED

    def test_retracted_publication_is_flagged(self, connector) -> None:
        """The MEDLINE 'Retracted Publication' type sets is_retracted."""
        paper = connector._parse_item(
            _record(3, pubTypeList={"pubType": ["Retracted Publication", "Editorial"]})
        )
        assert paper is not None
        assert paper.is_retracted is True

    def test_absence_of_the_marker_is_not_a_claim_of_no_retraction(self, connector) -> None:
        """No marker means unknown, not False."""
        paper = connector._parse_item(_record(4))
        assert paper is not None
        assert paper.is_retracted is None

    def test_keywords_merge_author_terms_and_mesh_descriptors(self, connector) -> None:
        """Both keyword sources are kept."""
        paper = connector._parse_item(
            _record(
                5,
                keywordList={"keyword": ["genome editing"]},
                meshHeadingList={"meshHeading": [{"descriptorName": "Fibroblasts"}]},
            )
        )
        assert paper is not None
        assert paper.keywords == {"genome editing", "Fibroblasts"}

    def test_url_falls_back_to_the_doi(self, connector) -> None:
        """A record with no HTML link still gets a resolvable URL."""
        paper = connector._parse_item(_record(6, fullTextUrlList={}))
        assert paper is not None
        assert paper.url == "https://doi.org/10.1000/test.6"

    def test_url_falls_back_to_europe_pmc_when_there_is_no_doi(self, connector) -> None:
        """PMCID is the next-best identifier for a landing page."""
        paper = connector._parse_item(_record(6, fullTextUrlList={}, doi=""))
        assert paper is not None
        assert paper.url == "https://europepmc.org/article/PMC/PMC4000006"

    def test_year_only_records_fall_back_to_january(self, connector) -> None:
        """pubYear alone still yields a date rather than dropping the record."""
        paper = connector._parse_item(_record(8, firstPublicationDate=""))
        assert paper is not None
        assert paper.publication_date == datetime.date(2023, 1, 1)

    def test_untitled_record_is_rejected(self, connector) -> None:
        """A record with no title cannot become a Paper."""
        assert connector._parse_item(_record(9, title="")) is None

    def test_non_dict_record_is_rejected(self, connector) -> None:
        """A malformed entry does not raise out of the pagination loop."""
        not_a_record: Any = "not a record"
        assert connector._parse_item(not_a_record) is None

    def test_string_author_entries_are_tolerated(self, connector) -> None:
        """Some sources hand back plain strings for authors."""
        paper = connector._parse_item(_record(10, authorList={"author": ["Plain Name"]}))
        assert paper is not None
        assert paper.authors[0].name == "Plain Name"


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------


class TestQueryBuilder:
    """Only whitelisted field names reach the API, and unsafe terms are refused."""

    @pytest.fixture
    def builder(self) -> EuropePMCQueryBuilder:
        """Return a query builder."""
        return EuropePMCQueryBuilder()

    def test_default_filter_uses_title_abstract_keywords(self, builder) -> None:
        """The default scope expands to the TITLE_ABS / KW disjunction."""
        converted = builder.convert_query(_parse("[machine learning]"))
        assert converted == '(TITLE_ABS:"machine learning" OR KW:"machine learning")'

    @pytest.mark.parametrize(
        ("query_string", "expected"),
        [
            ("ti[crispr]", 'TITLE:"crispr"'),
            ("abs[crispr]", 'ABSTRACT:"crispr"'),
            ("au[Doudna]", 'AUTH:"Doudna"'),
            ("key[crispr]", 'KW:"crispr"'),
            ("src[Nature]", 'JOURNAL:"Nature"'),
            ("aff[Toronto]", 'AFF:"Toronto"'),
            ("tiabs[crispr]", 'TITLE_ABS:"crispr"'),
        ],
    )
    def test_each_filter_maps_to_an_indexed_field(self, builder, query_string, expected) -> None:
        """Every emitted field name is one Europe PMC actually indexes.

        An unindexed field name returns hitCount 0 with HTTP 200, so a wrong
        mapping here is invisible at runtime.
        """
        assert builder.convert_query(_parse(query_string)) == expected

    def test_connectors_map_to_europe_pmc_operators(self, builder) -> None:
        """AND NOT becomes NOT, which is Europe PMC's exclusion operator."""
        converted = builder.convert_query(_parse("ti[a] AND NOT ti[b]"))
        assert converted == 'TITLE:"a" NOT TITLE:"b"'

    def test_wildcard_terms_are_left_unquoted(self, builder) -> None:
        """Europe PMC does not expand a wildcard inside a quoted phrase."""
        assert builder.convert_query(_parse("ti[neural*]")) == "TITLE:neural*"

    def test_wildcard_with_a_space_is_rejected(self, builder) -> None:
        """The field prefix would apply only to the first token.

        Verified live: TITLE_ABS:machine learn* returned 249,661 hits against
        378,639 for TITLE_ABS:machine, so the second token escaped the field.
        The query runs and looks fine, which is why it must be refused here.
        """
        result = builder.validate_query(_parse("ti[machine learn*]"))
        assert result.is_valid is False
        assert "wildcard" in (result.error_message or "")

    def test_valid_query_passes(self, builder) -> None:
        """A normal query is accepted."""
        assert builder.validate_query(_parse("[machine learning] AND [deep learning]")).is_valid

    def test_rejected_query_reaches_the_ledger_as_query_unsupported(self, connector) -> None:
        """A refused query is a coverage status, never a raised error or a zero."""
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(connector, "_get") as mocked,
        ):
            result = connector.search(_parse("ti[machine learn*]"))
        assert result.status is CoverageStatus.QUERY_UNSUPPORTED
        assert result.papers == ()
        mocked.assert_not_called()
