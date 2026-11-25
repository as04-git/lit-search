"""Unit tests for DOAJConnector.

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

from lit_search.connectors.doaj import DOAJBodyError, DOAJConnector
from lit_search.core.coverage import CoverageStatus, CoveredPapers, SourceCoverage
from lit_search.core.paper import Database, PaperType
from lit_search.core.source import SourceType
from lit_search.query.builders.doaj import DOAJQueryBuilder

from .conftest import _parse

# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def _record(index: int, **overrides: Any) -> dict[str, Any]:
    """Build one DOAJ ``results`` entry.

    Parameters
    ----------
    index : int
        Used to make the title and identifiers distinct.
    **overrides : Any
        ``bibjson`` fields to replace or add.

    Returns
    -------
    dict[str, Any]
        A record shaped like a live ``results`` entry.
    """
    bibjson: dict[str, Any] = {
        "title": f"Synthetic Paper {index}",
        "abstract": f"Synthetic abstract {index}.",
        "year": "2023",
        "month": "February",
        "keywords": ["open access", "testing"],
        "subject": [{"code": "QC1-999", "scheme": "LCC", "term": "Physics"}],
        "identifier": [
            {"id": f"10.1000/doaj.{index}", "type": "doi"},
            {"id": "2714-2817", "type": "pissn"},
        ],
        "author": [{"name": f"Author {index}", "affiliation": "University of Toronto"}],
        "journal": {
            "title": "Journal of Synthetic Physics",
            "issns": ["2714-2817", "2714-4704"],
            "publisher": "Synthetic Society",
            "country": "CA",
            "language": ["EN"],
        },
        "link": [
            {
                "content_type": "HTML",
                "type": "fulltext",
                "url": f"https://example.org/article/{index}",
            },
            {
                "content_type": "PDF",
                "type": "fulltext",
                "url": f"https://example.org/article/{index}.pdf",
            },
        ],
    }
    bibjson.update(overrides)
    return {"id": f"doajid{index}", "last_updated": "2025-08-20T01:49:50Z", "bibjson": bibjson}


def _page(
    records: list[dict[str, Any]],
    total: int,
    page: int = 1,
    page_size: int = 100,
) -> dict[str, Any]:
    """Build one search-response body.

    Parameters
    ----------
    records : list[dict[str, Any]]
        Records for this page.
    total : int
        Value of the ``total`` field.
    page : int
        Page number echoed back.
    page_size : int
        Page size echoed back.

    Returns
    -------
    dict[str, Any]
        Response body.
    """
    return {
        "total": total,
        "page": page,
        "pageSize": page_size,
        "timestamp": "2026-08-31T09:00:00.000000Z",
        "query": "synthetic",
        "results": records,
    }


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


def _http_error(body: Any, status_code: int = 400) -> requests.HTTPError:
    """Build the ``HTTPError`` ``raise_for_status`` would raise for *body*.

    DOAJ answers a refused query and the record ceiling with HTTP 400 and a
    JSON body naming the reason, so the body has to travel with the error.

    Parameters
    ----------
    body : Any
        Object returned by the response's ``.json()``.
    status_code : int
        HTTP status to report.

    Returns
    -------
    requests.HTTPError
        Error carrying the mock response.
    """
    return requests.HTTPError(f"{status_code} Client Error", response=_response(body, status_code))


_CEILING_BODY = {
    "status": "bad_request",
    "error": (
        "You cannot access results beyond 1000 records via this API.\n"
        "    If you would like to see more results, you can download all of our data from\n"
        "    https://doaj.org/docs/public-data-dump/."
    ),
}
_WILDCARD_BODY = {
    "status": "bad_request",
    "error": "Query contains disallowed Lucene features (ref: f6dcca78)",
}
_BAD_LUCENE_BODY = {
    "status": "bad_request",
    "error": "There was an error executing your query (ref: 8fc23d8c)",
}


@pytest.fixture
def connector() -> DOAJConnector:
    """Return a DOAJ connector."""
    return DOAJConnector()


def _search(connector: DOAJConnector, responses: list[Any], **kwargs: Any) -> CoveredPapers:
    """Run a search against a scripted sequence of ``_get`` outcomes.

    Parameters
    ----------
    connector : DOAJConnector
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
        assert connector.name == Database.DOAJ.value

    def test_min_request_interval_is_the_documented_two_per_second(self, connector) -> None:
        """DOAJ publishes 2 req/s on all API routes; 0.5s is that, not a guess."""
        assert connector.min_request_interval == 0.5

    def test_registered_in_the_search_registry(self) -> None:
        """A connector nobody can select is not ported."""
        from lit_search.connectors import SEARCH_REGISTRY

        assert SEARCH_REGISTRY[Database.DOAJ] is DOAJConnector

    def test_not_registered_for_lookup_roles_it_does_not_implement(self) -> None:
        """A connector registered for an unimplemented role returns silent Nones."""
        from lit_search.connectors import DOI_LOOKUP_REGISTRY, URL_LOOKUP_REGISTRY

        assert Database.DOAJ not in DOI_LOOKUP_REGISTRY
        assert Database.DOAJ not in URL_LOOKUP_REGISTRY

    def test_no_credential_is_declared_explicitly(self, connector) -> None:
        """DOAJ's keys are publisher-only, so search runs in the anonymous bucket."""
        assert connector._credential_attrs == ()
        assert connector.rate_limit_scope.service == "doaj"
        assert connector.rate_limit_scope.fingerprint == "anon"

    def test_rate_limit_service_is_shared_across_doaj_routes(self, connector) -> None:
        """DOAJ's limit is per API route family, not per connector."""
        assert connector.rate_limit_service == "doaj"


# ---------------------------------------------------------------------------
# The four required cases, plus the one that matters
# ---------------------------------------------------------------------------


class TestCoverageIsHonest:
    """Each way a search can end reports a different, actionable status."""

    def test_normal_result_is_complete(self, connector) -> None:
        """Retrieving as many records as ``total`` reports is COMPLETE."""
        result = _search(connector, [_response(_page([_record(0), _record(1)], total=2))])
        assert result.status is CoverageStatus.COMPLETE
        assert len(result.papers) == 2
        assert result.coverage.retrieved == 2
        assert result.coverage.reported_total == 2

    def test_upstream_failure_is_failed(self, connector) -> None:
        """A dead first request is FAILED, and the error is carried."""
        result = _search(connector, [requests.ConnectionError("doaj unreachable")])
        assert result.status is CoverageStatus.FAILED
        assert result.papers == ()
        assert "unreachable" in (result.coverage.error or "")
        assert result.coverage.error_type == "ConnectionError"

    def test_empty_but_successful_search_is_complete(self, connector) -> None:
        """A genuine zero-hit query is COMPLETE with no papers.

        This is the honest empty: DOAJ answered, said zero records match, and
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
            [_response(_page([_record(i) for i in range(3)], total=900, page_size=3))],
            max_papers=3,
        )
        assert result.status is CoverageStatus.CAPPED
        assert len(result.papers) == 3
        assert result.coverage.cap == 3
        assert result.coverage.reported_total == 900

    def test_failure_and_emptiness_are_different_statuses(self, connector) -> None:
        """The regression test for the whole port.

        In the donor implementation both of these paths end at the same
        ``return papers[:max_results]`` with ``papers == []``, so a caller
        cannot tell an outage from a database that holds nothing.  Here they
        must differ, and the failure must not be readable as a complete answer.
        """
        failure = _search(connector, [requests.ConnectionError("boom")])
        empty = _search(connector, [_response(_page([], total=0))])

        assert failure.papers == empty.papers == ()
        assert failure.status is not empty.status
        assert failure.status is CoverageStatus.FAILED
        assert empty.status is CoverageStatus.COMPLETE
        assert not failure.coverage.is_complete
        assert empty.coverage.is_complete


# ---------------------------------------------------------------------------
# Bodies that report an error rather than results
# ---------------------------------------------------------------------------


class TestErrorBodies:
    """DOAJ names its own failure reasons; none of them may become a zero.

    The HTTP 400 family is the dangerous one.  ``raise_for_status`` reduces the
    body to "400 Client Error", and the donor's ``except
    requests.exceptions.RequestException`` then converts it into the empty list
    it had accumulated.
    """

    def test_record_ceiling_on_the_first_page_is_failed(self, connector) -> None:
        """The 1,000-record ceiling arriving with no records is a loud failure."""
        result = _search(connector, [_http_error(_CEILING_BODY)])
        assert result.status is CoverageStatus.FAILED
        assert "beyond 1000 records" in (result.coverage.error or "")

    def test_wildcard_rejection_is_query_unsupported(self, connector) -> None:
        """A body blaming the query is a query problem, not an outage.

        The distinction is actionable: FAILED invites a retry, and retrying a
        query the source cannot express only wastes the rate-limit budget.
        """
        result = _search(connector, [_http_error(_WILDCARD_BODY)])
        assert result.status is CoverageStatus.QUERY_UNSUPPORTED
        assert result.papers == ()

    def test_unexecutable_query_is_query_unsupported(self, connector) -> None:
        """DOAJ's other query-side 400 is classified the same way."""
        result = _search(connector, [_http_error(_BAD_LUCENE_BODY)])
        assert result.status is CoverageStatus.QUERY_UNSUPPORTED

    def test_a_bodyless_http_error_stays_a_failure(self, connector) -> None:
        """A 502 from a proxy has no DOAJ explanation to recover."""
        broken = _response("<html>gateway</html>", 502)
        broken.json.side_effect = ValueError("Expecting value")
        result = _search(connector, [requests.HTTPError("502", response=broken)])
        assert result.status is CoverageStatus.FAILED

    def test_error_key_at_http_200_is_failed(self, connector) -> None:
        """The donor's ``if 'error' in data: return papers`` case."""
        result = _search(connector, [_response({"error": "something went wrong upstream"})])
        assert result.status is CoverageStatus.FAILED
        assert "something went wrong" in (result.coverage.error or "")

    def test_missing_total_is_failed_not_empty(self, connector) -> None:
        """A body with no ``total`` is not a search result."""
        result = _search(connector, [_response({"results": []})])
        assert result.status is CoverageStatus.FAILED
        assert "total" in (result.coverage.error or "")

    def test_non_json_body_is_failed(self, connector) -> None:
        """An HTML error page from a CDN is a failure, not an empty result."""
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
        assert DOAJBodyError("x", is_query_error=True).is_query_error is True
        assert DOAJBodyError("x").is_query_error is False


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class TestPagination:
    """Page-number paging, and every abnormal stop."""

    def test_multi_page_run_accumulates_until_the_total_is_reached(self, connector) -> None:
        """Two full pages and a short last page is the observed live shape."""
        result = _search(
            connector,
            [
                _response(_page([_record(i) for i in range(100)], total=201, page=1)),
                _response(_page([_record(100 + i) for i in range(100)], total=201, page=2)),
                _response(_page([_record(200)], total=201, page=3)),
            ],
        )
        assert result.status is CoverageStatus.COMPLETE
        assert len(result.papers) == 201

    def test_short_page_alone_does_not_end_pagination(self, connector) -> None:
        """The end signal is the record count, not a page shorter than the size."""
        result = _search(
            connector,
            [
                _response(_page([_record(0)], total=2, page=1)),
                _response(_page([_record(1)], total=2, page=2)),
            ],
        )
        assert len(result.papers) == 2
        assert result.status is CoverageStatus.COMPLETE

    def test_empty_page_before_the_total_is_partial(self, connector) -> None:
        """DOAJ still claims more records than arrived: stop and say so."""
        result = _search(
            connector,
            [
                _response(_page([_record(0)], total=9, page=1)),
                _response(_page([], total=9, page=2)),
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
                _response(_page([_record(0), _record(1)], total=900, page=1)),
                requests.ConnectionError("dropped mid-run"),
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 2
        assert result.coverage.error_type == "ConnectionError"
        assert "dropped mid-run" in (result.coverage.detail or "")

    def test_record_ceiling_after_some_pages_is_partial(self, connector) -> None:
        """A ceiling 400 partway through degrades rather than discards."""
        result = _search(
            connector,
            [
                _response(_page([_record(0)], total=5000, page=1)),
                _http_error(_CEILING_BODY),
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 1
        assert "beyond 1000 records" in (result.coverage.detail or "")

    def test_the_thousand_record_ceiling_is_reached_before_it_is_requested(self, connector) -> None:
        """The connector stops at record 1,000 rather than asking for page 11.

        DOAJ refuses ``page * pageSize > 1000`` outright, so a run that asked
        would trade a real page for an HTTP 400.
        """
        pages = [
            _response(_page([_record(i) for i in range(100)], total=5000, page=p))
            for p in range(1, 11)
        ]
        result = _search(connector, pages)
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 1000
        assert "beyond record 1000" in (result.coverage.detail or "")
        assert result.coverage.reported_total == 5000

    def test_total_drift_between_pages_is_partial(self, connector) -> None:
        """A moving index means neither total can be checked."""
        result = _search(
            connector,
            [
                _response(_page([_record(0), _record(1)], total=3, page=1)),
                _response(_page([_record(2)], total=7, page=2)),
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert "changed mid-retrieval" in (result.coverage.detail or "")

    def test_total_drift_still_degrades_when_the_cap_binds(self, connector) -> None:
        """Drift must outrank CAPPED, not be absorbed by it.

        ``_build_coverage`` short-circuits on ``truncated_early or total_drift``
        rather than letting ``from_pagination`` decide, and the drift half of
        that disjunct changes the answer in exactly one case: a run that also
        hit the caller's cap.  ``from_pagination`` tests the cap first, so
        without the short-circuit a drifted run comes back CAPPED -- a *better*
        status than PARTIAL -- while the index was demonstrably moving
        underneath it.  Removing ``or total_drift`` leaves every other test in
        this file green, so this is the one that holds it in place.
        """
        result = _search(
            connector,
            [
                _response(_page([_record(i) for i in range(2)], total=50, page=1, page_size=2)),
                _response(_page([_record(i) for i in range(2, 4)], total=90, page=2, page_size=2)),
            ],
            max_papers=4,
        )
        assert result.status is CoverageStatus.PARTIAL
        assert "changed mid-retrieval" in (result.coverage.detail or "")

    def test_unparsable_records_are_counted_as_a_shortfall(self, connector) -> None:
        """A record seen but not parsed is a coverage loss, not an absence."""
        result = _search(
            connector,
            [_response(_page([_record(0), _record(1, title="   ")], total=2))],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 1
        assert "could not be parsed" in (result.coverage.detail or "")

    def test_coverage_count_always_matches_the_papers_returned(self, connector) -> None:
        """CoveredPapers rejects a mismatch, so this asserts the accounting."""
        result = _search(
            connector,
            [_response(_page([_record(i) for i in range(5)], total=99, page_size=2))],
            max_papers=2,
        )
        assert result.coverage.retrieved == len(result.papers) == 2


# ---------------------------------------------------------------------------
# Request shaping
# ---------------------------------------------------------------------------


class TestRequestShaping:
    """Parameters and URLs sent upstream match what the live API accepts."""

    def _capture(self, connector, **kwargs) -> tuple[str, dict[str, Any]]:
        """Run one search and return the URL and parameters of the first request."""
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(connector, "_get", return_value=_response(_page([], total=0))) as mocked,
        ):
            connector.search(_parse("[machine learning]"), **kwargs)
        return mocked.call_args.args[0], mocked.call_args.args[1]

    def test_the_query_is_percent_encoded_into_the_path(self, connector) -> None:
        """DOAJ takes the query as a path segment, so nothing may stay raw.

        An unencoded ``/`` -- which a DOI term carries -- would become a path
        separator and 404 instead of searching.
        """
        url, _ = self._capture(connector)
        assert url.startswith("https://doaj.org/api/search/articles/")
        tail = url.rsplit("/", 1)[1]
        assert "/" not in tail
        assert " " not in tail
        assert '"' not in tail

    def test_page_size_never_exceeds_the_api_maximum(self, connector) -> None:
        """pageSize above 100 is silently clamped upstream, so ask for what is real."""
        _, params = self._capture(connector, max_papers=100_000)
        assert 1 <= params["pageSize"] <= 100

    def test_page_size_is_at_least_one_when_the_cap_is_one(self, connector) -> None:
        """A cap of 1 must not produce pageSize=0, which the API rewrites to 10."""
        _, params = self._capture(connector, max_papers=1)
        assert params["pageSize"] >= 1

    def test_page_size_is_constant_across_a_run(self, connector) -> None:
        """DOAJ pages by ``(page - 1) * pageSize``.

        Shrinking the size to fit a remaining cap would move the window
        backwards and re-read records already seen instead of continuing.
        """
        pages = [
            _response(_page([_record(i) for i in range(100)], total=5000, page=p))
            for p in range(1, 4)
        ]
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(connector, "_get", side_effect=pages) as mocked,
        ):
            connector.search(_parse("[machine learning]"), max_papers=250)
        sizes = {call.args[1]["pageSize"] for call in mocked.call_args_list}
        numbers = [call.args[1]["page"] for call in mocked.call_args_list]
        assert len(sizes) == 1
        assert numbers == [1, 2, 3]

    def test_date_bounds_become_a_year_range(self, connector) -> None:
        """DOAJ indexes bibjson.year, so bounds render as a year range."""
        url, _ = self._capture(
            connector,
            since=datetime.date(2020, 1, 1),
            until=datetime.date(2021, 12, 31),
        )
        assert "bibjson.year%3A%5B2020%20TO%202021%5D" in url

    @pytest.mark.parametrize(
        ("bounds", "expected"),
        [
            ({"since": datetime.date(2020, 1, 1)}, "bibjson.year%3A%3E%3D2020"),
            ({"until": datetime.date(2021, 12, 31)}, "bibjson.year%3A%3C%3D2021"),
        ],
    )
    def test_a_one_sided_bound_never_uses_an_open_range(self, connector, bounds, expected) -> None:
        """DOAJ refuses ``*`` inside a range, and a sentinel year loses records.

        Verified live 2026-08-31: ``bibjson.year:[2020 TO *]`` returns HTTP 400
        ``"Query contains disallowed Lucene features"``, which would be reported
        as an outage.  And substituting a wide range is not equivalent --
        ``bibjson.title:"protein"`` matched 96,725 records against 96,527 for
        the same query plus ``bibjson.year:[1000 TO 3000]``, so 198 records
        carry a year no sentinel range holds.
        """
        url, _ = self._capture(connector, **bounds)
        assert expected in url
        assert "TO%20%2A" not in url
        assert "1000" not in url
        assert "3000" not in url

    def test_no_date_filter_when_no_bounds_are_given(self, connector) -> None:
        """An unbounded search must not silently acquire a date restriction."""
        url, _ = self._capture(connector)
        assert "bibjson.year" not in url

    def test_year_aligned_bounds_do_not_degrade_coverage(self, connector) -> None:
        """A whole-year window is exactly what DOAJ can express, so nothing is lost."""
        result = _search(
            connector,
            [_response(_page([_record(0)], total=1))],
            since=datetime.date(2020, 1, 1),
            until=datetime.date(2021, 12, 31),
        )
        assert result.status is CoverageStatus.COMPLETE

    def test_sub_year_bounds_are_reported_as_a_shortfall(self, connector) -> None:
        """DOAJ has no publication-date index finer than the year.

        Widening the window cannot lose a matching record, but it does mean
        DOAJ's own ``total`` counts a different query -- so a COMPLETE checked
        against it would be checked against the wrong denominator.
        """
        result = _search(
            connector,
            [_response(_page([_record(0)], total=1))],
            since=datetime.date(2020, 6, 15),
        )
        assert result.status is CoverageStatus.PARTIAL
        assert "publication year only" in (result.coverage.detail or "")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    """Field mapping from a DOAJ ``bibjson`` record to a Paper."""

    def test_core_fields(self, connector) -> None:
        """Title, abstract, DOI, date and URLs are mapped."""
        paper = connector._parse_item(_record(7))
        assert paper is not None
        assert paper.title == "Synthetic Paper 7"
        assert paper.abstract == "Synthetic abstract 7."
        assert paper.doi == "10.1000/doaj.7"
        assert paper.publication_date == datetime.date(2023, 2, 1)
        assert paper.url == "https://example.org/article/7"
        assert paper.pdf_url == "https://example.org/article/7.pdf"
        assert paper.language == "en"
        assert paper.is_open_access is True
        assert paper.paper_type is PaperType.ARTICLE
        assert Database.DOAJ in paper.found_in

    def test_authors_carry_affiliations(self, connector) -> None:
        """DOAJ puts the affiliation on the author entry itself."""
        paper = connector._parse_item(_record(1))
        assert paper is not None
        assert paper.authors[0].name == "Author 1"
        assert paper.authors[0].affiliation == "University of Toronto"

    def test_journal_becomes_a_journal_source(self, connector) -> None:
        """The first ISSN and the publisher are carried onto the Source."""
        paper = connector._parse_item(_record(1))
        assert paper is not None
        assert paper.source is not None
        assert paper.source.title == "Journal of Synthetic Physics"
        assert paper.source.issn == "2714-2817"
        assert paper.source.publisher == "Synthetic Society"
        assert paper.source.source_type is SourceType.JOURNAL

    def test_keywords_merge_author_terms_and_subject_terms(self, connector) -> None:
        """Both keyword sources are kept."""
        paper = connector._parse_item(_record(5))
        assert paper is not None
        assert paper.keywords == {"open access", "testing", "Physics"}

    def test_numeric_month_is_parsed(self, connector) -> None:
        """DOAJ stores the month as a name or a number; both must resolve."""
        paper = connector._parse_item(_record(2, month="3"))
        assert paper is not None
        assert paper.publication_date == datetime.date(2023, 3, 1)

    def test_year_only_records_fall_back_to_january(self, connector) -> None:
        """A record with no month still yields a date rather than being dropped."""
        paper = connector._parse_item(_record(3, month=""))
        assert paper is not None
        assert paper.publication_date == datetime.date(2023, 1, 1)

    def test_record_without_a_year_has_no_date(self, connector) -> None:
        """An absent year is unknown, not year zero."""
        paper = connector._parse_item(_record(4, year=""))
        assert paper is not None
        assert paper.publication_date is None

    def test_url_falls_back_to_the_doi(self, connector) -> None:
        """A record with no HTML link still gets a resolvable URL."""
        paper = connector._parse_item(_record(6, link=[]))
        assert paper is not None
        assert paper.url == "https://doi.org/10.1000/doaj.6"

    def test_url_falls_back_to_the_doaj_article_page(self, connector) -> None:
        """With no link and no DOI, the DOAJ article id is the landing page."""
        paper = connector._parse_item(_record(6, link=[], identifier=[]))
        assert paper is not None
        assert paper.url == "https://doaj.org/article/doajid6"

    def test_pdf_link_detected_by_extension(self, connector) -> None:
        """Not every fulltext link declares content_type."""
        paper = connector._parse_item(
            _record(8, link=[{"type": "fulltext", "url": "https://example.org/x.pdf"}])
        )
        assert paper is not None
        assert paper.pdf_url == "https://example.org/x.pdf"

    def test_untitled_record_is_rejected(self, connector) -> None:
        """A record with no title cannot become a Paper."""
        assert connector._parse_item(_record(9, title="")) is None

    def test_record_without_bibjson_is_rejected(self, connector) -> None:
        """A malformed entry does not raise out of the pagination loop."""
        assert connector._parse_item({"id": "x"}) is None

    def test_non_dict_record_is_rejected(self, connector) -> None:
        """A malformed entry does not raise out of the pagination loop."""
        not_a_record: Any = "not a record"
        assert connector._parse_item(not_a_record) is None


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------


class TestQueryBuilder:
    """Only whitelisted field paths reach the API, and unrunnable terms are refused."""

    @pytest.fixture
    def builder(self) -> DOAJQueryBuilder:
        """Return a query builder."""
        return DOAJQueryBuilder()

    def test_default_filter_expands_to_title_abstract_keywords(self, builder) -> None:
        """The default scope has no single DOAJ field, so it is written out."""
        converted = builder.convert_query(_parse("[machine learning]"))
        assert converted == (
            '(bibjson.title:"machine learning" OR bibjson.abstract:"machine learning" '
            'OR bibjson.keywords:"machine learning")'
        )

    @pytest.mark.parametrize(
        ("query_string", "expected"),
        [
            ("ti[crispr]", 'bibjson.title:"crispr"'),
            ("abs[crispr]", 'bibjson.abstract:"crispr"'),
            ("au[Doudna]", 'bibjson.author.name:"Doudna"'),
            ("key[crispr]", 'bibjson.keywords:"crispr"'),
            ("src[Nature]", 'bibjson.journal.title:"Nature"'),
            ("aff[Toronto]", 'bibjson.author.affiliation:"Toronto"'),
        ],
    )
    def test_each_filter_maps_to_an_indexed_field(self, builder, query_string, expected) -> None:
        """Every emitted field path is one DOAJ actually indexes.

        This is the test that guards the donor's live defect: DOAJ's plausible
        short-hand names ``author:``, ``keywords:``, ``year:`` and
        ``journal.title:`` all return HTTP 200 with ``total: 0``, so a wrong
        mapping here is invisible at runtime.
        """
        assert builder.convert_query(_parse(query_string)) == expected

    def test_title_abstract_is_a_disjunction(self, builder) -> None:
        """DOAJ has no combined title/abstract field, so both are searched."""
        assert builder.convert_query(_parse("tiabs[crispr]")) == (
            '(bibjson.title:"crispr" OR bibjson.abstract:"crispr")'
        )

    def test_no_emitted_field_uses_a_short_hand_name(self, builder) -> None:
        """The short forms that silently match nothing must never be emitted."""
        from lit_search.query.builders.doaj import COMPOSITE_FIELDS, FIELD_MAP

        emitted = set(FIELD_MAP.values()) | {
            f for fields in COMPOSITE_FIELDS.values() for f in fields
        }
        assert all(field.startswith("bibjson.") for field in emitted)

    def test_connectors_map_to_lucene_operators(self, builder) -> None:
        """AND NOT becomes NOT, which is DOAJ's exclusion operator."""
        converted = builder.convert_query(_parse("ti[a] AND NOT ti[b]"))
        assert converted == 'bibjson.title:"a" NOT bibjson.title:"b"'

    def test_quotes_inside_a_term_are_escaped(self, builder) -> None:
        """An unescaped quote would end the phrase and unbalance the query."""
        from lit_search.query.builders.doaj import quote_value

        assert quote_value('say "hi"') == '"say \\"hi\\""'

    def test_wildcard_terms_are_rejected(self, builder) -> None:
        """DOAJ answers a wildcard with HTTP 400, not with results."""
        result = builder.validate_query(_parse("ti[neural*]"))
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
            result = connector.search(_parse("ti[neural*]"))
        assert result.status is CoverageStatus.QUERY_UNSUPPORTED
        assert result.papers == ()
        mocked.assert_not_called()


# ---------------------------------------------------------------------------
# The base class's second opinion
# ---------------------------------------------------------------------------


class _LyingDOAJConnector(DOAJConnector):
    """A DOAJ connector that reports every run as COMPLETE.

    Used to prove that this connector's HTTP goes through ``_get`` -- and
    therefore through the base class's request audit.  If it did not, the audit
    would see no failed request and the lie would survive.
    """

    def _build_coverage(self, **kwargs: Any) -> SourceCoverage:
        """Claim COMPLETE regardless of what happened.

        Parameters
        ----------
        **kwargs : Any
            The real signature's arguments, ignored except for the count.

        Returns
        -------
        SourceCoverage
            An unjustified COMPLETE.
        """
        retrieved = int(kwargs["retrieved"])
        return SourceCoverage.complete(self.name, retrieved, reported_total=retrieved)


class TestTheBaseClassChecksTheClaim:
    """A swallowed page failure cannot be talked into a COMPLETE."""

    def test_a_swallowed_page_failure_is_downgraded_to_partial(self) -> None:
        """The audit records the failure before the connector's except clause runs.

        This test also proves the connector routes its HTTP through ``_get``: a
        bare ``requests.get`` would escape ``_request_with_retry`` and leave the
        audit with nothing to contradict the claim.
        """
        outcomes: list[Any] = [
            _response(_page([_record(0)], total=900, page=1)),
            requests.ConnectionError("page 2 refused"),
        ]

        def _fake_request(*_args: Any, **_kwargs: Any) -> Any:
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        connector = _LyingDOAJConnector()
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(connector, "_request_with_retry_uncounted", side_effect=_fake_request),
        ):
            result = connector.search(_parse("[machine learning]"))

        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 1
        assert "request(s) failed" in (result.coverage.detail or "")
        assert "ConnectionError" in (result.coverage.detail or "")
