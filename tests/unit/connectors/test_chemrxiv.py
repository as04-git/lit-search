"""Unit tests for ChemRxivConnector.

These run under ``pytest-socket`` with ``--disable-socket`` (see
``pyproject.toml``), so every HTTP call is mocked and no test can reach the
network.  All fixture payloads mirror shapes observed against
``api.crossref.org`` on 2026-08-31.

The test that carries the most weight is
:meth:`TestCoverageIsHonest.test_failure_and_emptiness_are_different_statuses`.
Everything else here can pass while the connector still reports an upstream
outage as a confident zero; that one cannot.

Two of these tests exist because the *donor* connector fails them.  The donor
sends ``filter=type:posted-content,from-publisher:chemrxiv``, which CrossRef
refuses with HTTP 400, and the donor turns that refusal into ``[]``.  So the
donor returns zero results for every query ever asked of it and calls that an
answer.  :class:`TestTheDonorBug` pins both halves.
"""

from __future__ import annotations

import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from lit_search.connectors.chemrxiv import (
    CHEMRXIV_DOI_PREFIX,
    ChemRxivBodyError,
    ChemRxivConnector,
)
from lit_search.core.coverage import CoverageStatus, CoveredPapers
from lit_search.core.paper import Database, PaperType
from lit_search.query.builders.crossref import CrossRefQueryBuilder, UnexpressibleQueryError

from .conftest import _parse

# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def _work(index: int, **overrides: Any) -> dict[str, Any]:
    """Build one CrossRef ``posted-content`` work record.

    Parameters
    ----------
    index : int
        Used to make the title and identifiers distinct.
    **overrides : Any
        Fields to replace or add.

    Returns
    -------
    dict[str, Any]
        A record shaped like a live ``message.items`` entry.
    """
    record: dict[str, Any] = {
        "DOI": f"{CHEMRXIV_DOI_PREFIX}/chemrxiv-2024-{index:05d}",
        "type": "posted-content",
        "title": [f"Synthetic Preprint {index}"],
        "abstract": f"<jats:p>Synthetic abstract {index}.</jats:p>",
        "publisher": "American Chemical Society (ACS)",
        "author": [{"given": "Ada", "family": f"Lovelace{index}"}],
        "posted": {"date-parts": [[2024, 5, 4]]},
        "is-referenced-by-count": index,
        "URL": f"https://doi.org/{CHEMRXIV_DOI_PREFIX}/chemrxiv-2024-{index:05d}",
    }
    record.update(overrides)
    return record


def _page(
    works: list[dict[str, Any]],
    total_results: int,
    next_cursor: str | None = None,
) -> dict[str, Any]:
    """Build one CrossRef ``/works`` response body.

    Parameters
    ----------
    works : list[dict[str, Any]]
        Records for this page.
    total_results : int
        Value of ``message.total-results``.
    next_cursor : str | None
        ``message.next-cursor`` to include.  ``None`` omits the key, which is
        how the live API signals an exhausted cursor.

    Returns
    -------
    dict[str, Any]
        Response body.
    """
    message: dict[str, Any] = {
        "total-results": total_results,
        "items": works,
        "items-per-page": len(works),
    }
    if next_cursor is not None:
        message["next-cursor"] = next_cursor
    return {"status": "ok", "message-type": "work-list", "message": message}


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


def _http_error(status_code: int, body: Any) -> requests.HTTPError:
    """Build an :class:`requests.HTTPError` carrying a CrossRef error body.

    Parameters
    ----------
    status_code : int
        HTTP status of the rejected request.
    body : Any
        JSON body CrossRef returned with it.

    Returns
    -------
    requests.HTTPError
        Exception shaped like the one ``_get`` raises.
    """
    error = requests.HTTPError(f"{status_code} Client Error")
    error.response = _response(body, status_code)
    return error


@pytest.fixture
def connector() -> ChemRxivConnector:
    """Return a chemRxiv connector with no polite-pool mailto."""
    return ChemRxivConnector()


def _search(connector: ChemRxivConnector, responses: list[Any], **kwargs: Any) -> CoveredPapers:
    """Run a search against a scripted sequence of ``_get`` outcomes.

    Parameters
    ----------
    connector : ChemRxivConnector
        Connector under test.
    responses : list[Any]
        Values or exceptions for successive ``_get`` calls.
    **kwargs : Any
        Forwarded to ``search`` (``max_papers``, ``since``, ``until``), plus an
        optional ``query``.

    Returns
    -------
    CoveredPapers
        The search outcome.
    """
    query = kwargs.pop("query", None) or _parse("ti[catalysis]")
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
        assert connector.name == Database.CHEMRXIV.value

    def test_registered_in_the_search_registry(self) -> None:
        """A connector nobody can select is not ported."""
        from lit_search.connectors import SEARCH_REGISTRY

        assert SEARCH_REGISTRY[Database.CHEMRXIV] is ChemRxivConnector

    def test_anonymous_interval_is_crossrefs_own_published_rate(self, connector) -> None:
        """Live headers gave x-rate-limit-limit 1 per 1s without a mailto."""
        assert connector.min_request_interval == 1.0

    def test_polite_pool_interval_is_three_per_second(self) -> None:
        """Live headers gave x-rate-limit-limit 3 per 1s with a mailto."""
        assert ChemRxivConnector(email="someone@example.org").min_request_interval == 0.34

    def test_rate_limit_service_is_crossref_not_chemrxiv(self, connector) -> None:
        """Every request goes to api.crossref.org, so the bucket is CrossRef's.

        A bucket of its own would let this connector and CrossRef's DOI-lookup
        connector each spend a full allowance against one host.
        """
        assert connector.rate_limit_service == Database.CROSSREF.value
        assert connector.rate_limit_scope.service == Database.CROSSREF.value

    def test_the_credential_is_the_polite_pool_mailto(self, connector) -> None:
        """CrossRef keys its quota on the mailto, so that is the fingerprint."""
        assert connector._credential_attrs == ("_email",)
        assert connector.rate_limit_scope.fingerprint == "anon"
        assert ChemRxivConnector(email="a@b.c").rate_limit_scope.fingerprint != "anon"

    def test_not_registered_for_lookup_roles(self) -> None:
        """A role that was not implemented is not claimed."""
        from lit_search.connectors import DOI_LOOKUP_REGISTRY, URL_LOOKUP_REGISTRY

        assert Database.CHEMRXIV not in DOI_LOOKUP_REGISTRY
        assert Database.CHEMRXIV not in URL_LOOKUP_REGISTRY


# ---------------------------------------------------------------------------
# The donor bug, pinned
# ---------------------------------------------------------------------------


class TestTheDonorBug:
    """The two halves of the failure this port exists to remove."""

    def test_the_profile_filter_does_not_use_from_publisher(self, connector) -> None:
        """``from-publisher`` is not a CrossRef filter; the DOI prefix is.

        Verified live: the donor's filter returns HTTP 400
        ``filter-not-available``.
        """
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(connector, "_get", return_value=_response(_page([], 0))) as mocked,
        ):
            connector.search(_parse("ti[catalysis]"))
        sent = mocked.call_args.args[1]["filter"]
        assert "from-publisher" not in sent
        assert sent.startswith(f"type:posted-content,prefix:{CHEMRXIV_DOI_PREFIX}")

    def test_a_rejected_filter_is_reported_not_swallowed(self, connector) -> None:
        """The donor turns CrossRef's HTTP 400 into an empty result.

        Here the rejection reaches the ledger as FAILED and carries CrossRef's
        own explanation, so the reason is readable rather than inferable.
        """
        body = {
            "status": "failed",
            "message-type": "validation-failure",
            "message": [
                {
                    "type": "filter-not-available",
                    "value": "from-publisher",
                    "message": "Filter 'from-publisher' specified but there is no such filter",
                }
            ],
        }
        result = _search(connector, [_http_error(400, body)])
        assert result.status is CoverageStatus.FAILED
        assert result.papers == ()
        assert "from-publisher" in (result.coverage.error or "")


# ---------------------------------------------------------------------------
# The four required cases, plus the one that matters
# ---------------------------------------------------------------------------


class TestCoverageIsHonest:
    """Each way a search can end reports a different, actionable status."""

    def test_normal_result_is_complete(self, connector) -> None:
        """Retrieving as many records as total-results reports is COMPLETE."""
        result = _search(connector, [_response(_page([_work(0), _work(1)], 2))])
        assert result.status is CoverageStatus.COMPLETE
        assert len(result.papers) == 2
        assert result.coverage.reported_total == 2

    def test_upstream_failure_is_failed(self, connector) -> None:
        """A dead first request is FAILED, and the error is carried."""
        result = _search(connector, [requests.ConnectionError("crossref unreachable")])
        assert result.status is CoverageStatus.FAILED
        assert result.papers == ()
        assert "unreachable" in (result.coverage.error or "")
        assert result.coverage.error_type == "ConnectionError"

    def test_empty_but_successful_search_is_complete(self, connector) -> None:
        """A genuine zero-hit query is COMPLETE with no papers."""
        result = _search(connector, [_response(_page([], 0))])
        assert result.status is CoverageStatus.COMPLETE
        assert result.papers == ()
        assert result.coverage.reported_total == 0

    def test_capped_result_is_capped(self, connector) -> None:
        """Stopping at max_papers is CAPPED and names the cap."""
        result = _search(
            connector,
            [_response(_page([_work(i) for i in range(3)], 900, next_cursor="c1"))],
            max_papers=3,
        )
        assert result.status is CoverageStatus.CAPPED
        assert len(result.papers) == 3
        assert result.coverage.cap == 3
        assert result.coverage.reported_total == 900

    def test_failure_and_emptiness_are_different_statuses(self, connector) -> None:
        """The regression test for the whole port.

        In the donor implementation both of these paths end at the same
        ``return []``, so a caller cannot tell an outage from a database that
        holds nothing.  Here they must differ, and the failure must not be
        readable as a complete answer.
        """
        failure = _search(connector, [requests.ConnectionError("boom")])
        empty = _search(connector, [_response(_page([], 0))])

        assert failure.papers == empty.papers == ()
        assert failure.status is not empty.status
        assert failure.status is CoverageStatus.FAILED
        assert empty.status is CoverageStatus.COMPLETE
        assert not failure.coverage.is_complete
        assert empty.coverage.is_complete


# ---------------------------------------------------------------------------
# Bodies that are not results
# ---------------------------------------------------------------------------


class TestUnreadableBodies:
    """Anything that is not a work list is raised, never defaulted to zero."""

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

    def test_status_other_than_ok_is_failed(self, connector) -> None:
        """A 200 body reporting failure is not an empty result set."""
        result = _search(connector, [_response({"status": "failed", "message": []})])
        assert result.status is CoverageStatus.FAILED
        assert "failed" in (result.coverage.error or "")

    def test_missing_total_results_is_failed_not_empty(self, connector) -> None:
        """No ``total-results`` means the body is not a work list."""
        result = _search(connector, [_response({"status": "ok", "message": {"items": []}})])
        assert result.status is CoverageStatus.FAILED
        assert "total-results" in (result.coverage.error or "")

    def test_http_error_without_a_readable_body_still_fails(self, connector) -> None:
        """A 500 with no JSON body propagates as the original HTTPError."""
        broken = MagicMock()
        broken.status_code = 500
        broken.json.side_effect = ValueError("Expecting value")
        error = requests.HTTPError("500 Server Error")
        error.response = broken
        result = _search(connector, [error])
        assert result.status is CoverageStatus.FAILED
        assert result.coverage.error_type == "HTTPError"

    def test_body_error_is_a_connector_error(self) -> None:
        """The exception type is one ``search`` knows how to name."""
        from lit_search.exceptions import ConnectorError

        assert issubclass(ChemRxivBodyError, ConnectorError)


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class TestCursorPagination:
    """The live exhaustion signal, and every abnormal stop, are handled."""

    def test_multi_page_ends_on_an_empty_page_with_no_cursor(self, connector) -> None:
        """Verified live: the page past the end has no ``next-cursor``."""
        result = _search(
            connector,
            [
                _response(_page([_work(0), _work(1)], 3, next_cursor="c1")),
                _response(_page([_work(2)], 3, next_cursor="c2")),
                _response(_page([], 3)),
            ],
        )
        assert result.status is CoverageStatus.COMPLETE
        assert len(result.papers) == 3

    def test_short_page_alone_does_not_end_pagination(self, connector) -> None:
        """A page smaller than the request keeps going while a cursor is offered."""
        result = _search(
            connector,
            [
                _response(_page([_work(0)], 2, next_cursor="c1")),
                _response(_page([_work(1)], 2, next_cursor="c2")),
                _response(_page([], 2)),
            ],
        )
        assert result.status is CoverageStatus.COMPLETE
        assert len(result.papers) == 2

    def test_a_repeated_cursor_stops_the_run_and_says_so(self, connector) -> None:
        """CrossRef restarts pagination for a cursor it cannot parse.

        Verified live: ``cursor=GARBAGE!!`` returned HTTP 200 and page one
        again.  Trusting the cursor would re-read page one forever and report a
        large duplicate-filled result, so a repeat stops the run as PARTIAL.
        """
        result = _search(
            connector,
            [
                _response(_page([_work(0)], 50, next_cursor="c1")),
                _response(_page([_work(1)], 50, next_cursor="c1")),
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 2
        assert "cursor already used" in (result.coverage.detail or "")

    def test_a_restart_under_a_fresh_cursor_is_partial_not_complete(self, connector) -> None:
        """The seen-cursor guard catches only a *repeated* token.

        Verified live 2026-08-31: ``cursor=GARBAGE!!`` answers HTTP 200 with
        page one and a ``next-cursor`` of
        ``MTUwODg0NjYwNzAwMCwxMC4yNjQzNCUyRmNoZW1yeGl2LjU1MjU1NzIudjE=``
        (base64 for ``1508846607000,10.26434%2Fchemrxiv.5525572.v1``) -- a
        **new** token pointing just past the first record.  So a mid-walk
        restart never repeats a cursor and
        :meth:`ChemRxivConnector._advance` cannot see it.

        What does see it is arithmetic: four records came back where CrossRef
        said three existed.  A walk that re-serves records has also skipped
        the ones past the point it restarted from, so this must never be
        COMPLETE.  Before ``from_pagination`` grew its over-retrieval branch
        this returned ``complete`` with four papers against a total of three.
        """
        result = _search(
            connector,
            [
                _response(_page([_work(0), _work(1)], 3, next_cursor="cA")),
                _response(_page([_work(0), _work(1)], 3, next_cursor="cB")),
                _response(_page([], 3)),
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 4
        assert result.coverage.reported_total == 3
        assert "more distinct records than the source says exist" in (result.coverage.detail or "")

    def test_cursor_exhausted_before_the_total_is_partial(self, connector) -> None:
        """Fewer records than total-results, with the cursor spent, is PARTIAL."""
        result = _search(connector, [_response(_page([_work(0)], 50))])
        assert result.status is CoverageStatus.PARTIAL
        assert result.coverage.reported_total == 50
        assert "50" in (result.coverage.detail or "")

    def test_failure_after_some_pages_is_partial_not_failed(self, connector) -> None:
        """Records that did arrive are real, so the run is PARTIAL."""
        result = _search(
            connector,
            [
                _response(_page([_work(0), _work(1)], 90, next_cursor="c1")),
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
                _response(_page([_work(0), _work(1)], 3, next_cursor="c1")),
                _response(_page([_work(2)], 7, next_cursor="c2")),
                _response(_page([], 7)),
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert "changed mid-retrieval" in (result.coverage.detail or "")

    def test_unparsable_records_are_counted_as_a_shortfall(self, connector) -> None:
        """A record seen but not parsed is a coverage loss, not an absence."""
        result = _search(connector, [_response(_page([_work(0), _work(1, title=[])], 2))])
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 1
        assert "could not be parsed" in (result.coverage.detail or "")

    def test_coverage_count_always_matches_the_papers_returned(self, connector) -> None:
        """CoveredPapers rejects a mismatch, so this asserts the accounting."""
        result = _search(
            connector,
            [_response(_page([_work(i) for i in range(5)], 99, next_cursor="c1"))],
            max_papers=2,
        )
        assert result.coverage.retrieved == len(result.papers) == 2


# ---------------------------------------------------------------------------
# Scope degradation
# ---------------------------------------------------------------------------


class TestScopeDegradation:
    """A narrowed scope is a coverage loss the pagination counters cannot see."""

    def test_a_default_scoped_query_is_partial_even_when_the_total_is_met(self, connector) -> None:
        """CrossRef never searched the abstracts, so this is not COMPLETE.

        ``total-results`` counts only what CrossRef did search, so the
        arithmetic agrees with itself and the shortfall is invisible to it.
        """
        result = _search(
            connector,
            [_response(_page([_work(0), _work(1)], 2))],
            query=_parse("[catalysis]"),
        )
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 2
        assert "abstract" in (result.coverage.detail or "")

    def test_a_degraded_query_that_also_hits_the_cap_stays_partial(self, connector) -> None:
        """CAPPED would understate what is missing, so PARTIAL wins.

        ``CAPPED`` means "a known amount is missing, on purpose".  Here an
        *unknown* amount is missing as well, because CrossRef never searched
        the abstracts.  This assertion is what makes the explicit PARTIAL
        branch in ``_build_coverage`` load-bearing: without it, routing this
        case through ``from_pagination`` would return CAPPED and every test
        would still pass, because the degradation note alone is enough to keep
        the *uncapped* case PARTIAL.
        """
        result = _search(
            connector,
            [_response(_page([_work(i) for i in range(3)], 900, next_cursor="c1"))],
            max_papers=3,
            query=_parse("[catalysis]"),
        )
        assert result.status is CoverageStatus.PARTIAL
        assert "abstract" in (result.coverage.detail or "")

    def test_a_title_scoped_query_is_not_degraded(self, connector) -> None:
        """``query.title`` really does scope to titles, so nothing is lost.

        Verified live: ``query.title=smith`` gave 5 hits against 567 for
        ``query.bibliographic=smith``, so it is not an alias.
        """
        result = _search(
            connector, [_response(_page([_work(0)], 1))], query=_parse("ti[catalysis]")
        )
        assert result.status is CoverageStatus.COMPLETE


# ---------------------------------------------------------------------------
# Request shaping
# ---------------------------------------------------------------------------


class TestRequestShaping:
    """Parameters sent upstream match what the live API accepts."""

    def _capture(self, connector, **kwargs) -> dict[str, Any]:
        """Run one search and return the parameters of the first request."""
        query = kwargs.pop("query", None) or _parse("ti[catalysis]")
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(connector, "_get", return_value=_response(_page([], 0))) as mocked,
        ):
            connector.search(query, **kwargs)
        params: dict[str, Any] = mocked.call_args.args[1]
        return params

    def test_cursor_starts_at_the_wildcard(self, connector) -> None:
        """Deep paging needs the cursor; offset is capped at 9999 upstream."""
        assert self._capture(connector)["cursor"] == "*"

    def test_rows_never_exceeds_the_api_maximum(self, connector) -> None:
        """rows > 1000 is refused upstream with an HTTP 400 body."""
        assert 1 <= self._capture(connector, max_papers=100_000)["rows"] <= 1000

    def test_rows_is_at_least_one_when_the_cap_is_one(self, connector) -> None:
        """A cap of 1 must not produce rows=0."""
        assert self._capture(connector, max_papers=1)["rows"] >= 1

    def test_field_queries_come_from_the_builder(self, connector) -> None:
        """The term reaches CrossRef under a verified field-query name."""
        params = self._capture(connector, query=_parse("au[curie]"))
        assert params["query.author"] == "curie"

    def test_date_bounds_become_posted_date_filters(self, connector) -> None:
        """Preprints carry a posted date rather than a publication date."""
        params = self._capture(
            connector,
            since=datetime.date(2020, 1, 1),
            until=datetime.date(2021, 12, 31),
        )
        assert "from-posted-date:2020-01-01" in params["filter"]
        assert "until-posted-date:2021-12-31" in params["filter"]

    def test_no_date_filter_when_no_bounds_are_given(self, connector) -> None:
        """An unbounded search must not silently acquire a date restriction."""
        assert "posted-date" not in self._capture(connector)["filter"]

    def test_the_polite_pool_mailto_reaches_the_user_agent(self) -> None:
        """CrossRef reads the mailto from the User-Agent, not a parameter."""
        headers = ChemRxivConnector(email="a@b.c")._prepare_headers({})
        assert "mailto:a@b.c" in headers["User-Agent"]

    def test_no_mailto_means_no_mailto_clause(self, connector) -> None:
        """An anonymous client must not claim polite-pool membership."""
        assert "mailto" not in connector._prepare_headers({})["User-Agent"]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    """Field mapping from a CrossRef work record to a Paper."""

    def test_core_fields(self, connector) -> None:
        """Title, abstract and DOI come through the CrossRef parser."""
        paper = connector._parse_item(_work(7))
        assert paper is not None
        assert paper.title == "Synthetic Preprint 7"
        assert paper.abstract == "Synthetic abstract 7."
        assert paper.doi == f"{CHEMRXIV_DOI_PREFIX}/chemrxiv-2024-00007"

    def test_the_date_is_the_posted_date(self, connector) -> None:
        """A preprint's date is when it was posted, which is what we filter on.

        The inherited CrossRef parser never looks at ``posted``, so a record
        carrying only that field would come back dateless while still matching
        a ``from-posted-date`` bound.
        """
        paper = connector._parse_item(_work(7))
        assert paper is not None
        assert paper.publication_date == datetime.date(2024, 5, 4)

    def test_posted_wins_over_the_inherited_date_fields(self, connector) -> None:
        """When both are present the posted date is the one reported."""
        paper = connector._parse_item(
            _work(7, issued={"date-parts": [[2019, 1, 1]]}),
        )
        assert paper is not None
        assert paper.publication_date == datetime.date(2024, 5, 4)

    def test_a_record_without_a_posted_date_falls_back(self, connector) -> None:
        """The inherited parser still supplies a date when ``posted`` is absent."""
        record = _work(7)
        del record["posted"]
        record["issued"] = {"date-parts": [[2019, 3, 2]]}
        paper = connector._parse_item(record)
        assert paper is not None
        assert paper.publication_date == datetime.date(2019, 3, 2)

    def test_provenance_is_chemrxiv_not_crossref(self, connector) -> None:
        """The record was reached as chemRxiv, so that is what found_in says."""
        paper = connector._parse_item(_work(1))
        assert paper is not None
        assert paper.found_in == {Database.CHEMRXIV.value}

    def test_every_record_in_the_profile_is_a_preprint(self, connector) -> None:
        """The filter is ``type:posted-content``, so this holds by construction."""
        paper = connector._parse_item(_work(1))
        assert paper is not None
        assert paper.paper_type is PaperType.UNPUBLISHED

    def test_untitled_record_is_rejected(self, connector) -> None:
        """A record with no title cannot become a Paper."""
        assert connector._parse_item(_work(1, title=[])) is None

    def test_non_dict_record_is_rejected(self, connector) -> None:
        """A malformed entry is counted as unparsed, not crashed on."""
        assert connector._parse_item("not a record") is None


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------


class TestQueryBuilder:
    """CrossRef's expressible query shape, and what falls outside it."""

    @pytest.fixture
    def builder(self) -> CrossRefQueryBuilder:
        """Return a CrossRef query builder."""
        return CrossRefQueryBuilder()

    @pytest.mark.parametrize(
        ("query_string", "expected"),
        [
            ("ti[catalysis]", {"query.title": "catalysis"}),
            ("au[curie]", {"query.author": "curie"}),
            ("src[nature]", {"query.container-title": "nature"}),
            ("aff[sorbonne]", {"query.affiliation": "sorbonne"}),
            ("[catalysis]", {"query.bibliographic": "catalysis"}),
        ],
    )
    def test_each_filter_maps_to_a_verified_field_query(
        self, builder, query_string, expected
    ) -> None:
        """Every emitted name is in CrossRef's own valid-field-query list."""
        assert builder.convert_query(_parse(query_string)) == expected

    def test_or_within_one_field_becomes_one_parameter(self, builder) -> None:
        """Tokens inside one ``query.*`` are combined disjunctively upstream."""
        assert builder.convert_query(_parse("ti[a] OR ti[b]")) == {"query.title": "a b"}

    def test_and_across_fields_becomes_separate_parameters(self, builder) -> None:
        """Separate ``query.*`` parameters are combined conjunctively upstream."""
        converted = builder.convert_query(_parse("ti[a] AND au[b]"))
        assert converted == {"query.title": "a", "query.author": "b"}

    def test_and_within_one_field_is_rejected(self, builder) -> None:
        """It would be sent as an OR and return a superset."""
        result = builder.validate_query(_parse("ti[a] AND ti[b]"))
        assert not result.is_valid
        assert "disjunctively" in (result.error_message or "")

    def test_or_across_fields_is_rejected(self, builder) -> None:
        """It would be sent as an AND and return a subset."""
        result = builder.validate_query(_parse("ti[a] OR au[b]"))
        assert not result.is_valid
        assert "share one field query" in (result.error_message or "")

    def test_negation_is_rejected(self, builder) -> None:
        """``NOT`` is a stopword upstream: it would silently disappear.

        Measured: ``catalysis NOT photocatalysis`` and
        ``catalysis AND photocatalysis`` both returned 1079 hits.
        """
        result = builder.validate_query(_parse("ti[a] AND NOT ti[b]"))
        assert not result.is_valid
        assert "no negation" in (result.error_message or "")

    def test_abstract_scope_is_rejected(self, builder) -> None:
        """CrossRef has no abstract field query and does not index abstracts."""
        result = builder.validate_query(_parse("abs[catalysis]"))
        assert not result.is_valid
        assert "abstract" in (result.error_message or "")

    def test_keyword_scope_is_rejected(self, builder) -> None:
        """CrossRef has no keyword field query."""
        assert not builder.validate_query(_parse("key[catalysis]")).is_valid

    def test_wildcards_are_rejected(self, builder) -> None:
        """A wildcard character would be searched as an ordinary token."""
        result = builder.validate_query(_parse("ti[catalys*]"))
        assert not result.is_valid
        assert "wildcard" in (result.error_message or "")

    def test_mixed_connectors_at_one_level_are_rejected(self, builder) -> None:
        """AND is only across fields and OR only within one; the mix has no form."""
        result = builder.validate_query(_parse("ti[a] AND au[b] OR src[c]"))
        assert not result.is_valid

    def test_degradation_is_reported_for_composite_scopes(self, builder) -> None:
        """The connector turns this note into PARTIAL coverage."""
        assert builder.scope_degradation(_parse("[catalysis]")) is not None
        assert builder.scope_degradation(_parse("ti[catalysis]")) is None

    def test_convert_raises_rather_than_guessing(self, builder) -> None:
        """A caller that skipped validation gets an error, not a wrong query."""
        with pytest.raises(UnexpressibleQueryError):
            builder.convert_query(_parse("ti[a] AND NOT ti[b]"))

    def test_rejected_query_reaches_the_ledger_as_query_unsupported(self, connector) -> None:
        """And issues no request, because retrying it would change nothing."""
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(connector, "_get") as mocked,
        ):
            result = connector.search(_parse("ti[a] AND NOT ti[b]"))
        assert result.status is CoverageStatus.QUERY_UNSUPPORTED
        assert result.papers == ()
        mocked.assert_not_called()


# ---------------------------------------------------------------------------
# The base class checks the claim
# ---------------------------------------------------------------------------


class TestTheBaseClassChecksTheClaim:
    """A COMPLETE the evidence contradicts is downgraded, not accepted."""

    def test_a_swallowed_page_failure_cannot_claim_complete(self) -> None:
        """Proves this connector's HTTP goes through ``_get``.

        ``ConnectorBase._request_with_retry`` records every failed request on
        the audit *before* the connector's own ``except`` clause runs, so a
        connector that catches a mid-pagination failure and still reports
        COMPLETE is downgraded.  A connector reaching for a bare
        ``requests.get`` would escape that channel entirely, so this test
        patches the retry loop's inner half and leaves the counting half in
        place -- patching ``_get``, as every other test here does, would bypass
        exactly the mechanism under test.
        """
        from lit_search.core.coverage import SourceCoverage

        class _Liar(ChemRxivConnector):
            """A connector that reports COMPLETE whatever happened."""

            def _build_coverage(self, *, retrieved, run, max_papers, degradation):
                """Claim COMPLETE unconditionally."""
                return SourceCoverage.complete(self.name, retrieved, reported_total=retrieved)

            @staticmethod
            def _record_page_failure(run, exc, cursor):
                """Swallow the failure and keep whatever was retrieved."""
                return None

        liar = _Liar()
        outcomes: list[Any] = [
            _response(_page([_work(0)], 90, next_cursor="c1")),
            requests.ConnectionError("dropped mid-run"),
        ]
        with (
            patch.object(liar, "_rate_limit"),
            patch.object(liar, "_request_with_retry_uncounted", side_effect=outcomes),
        ):
            result = liar.search(_parse("ti[catalysis]"))

        assert result.status is CoverageStatus.PARTIAL
        assert "ConnectionError" in (result.coverage.detail or "")
