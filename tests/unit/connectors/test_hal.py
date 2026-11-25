"""Unit tests for HALConnector.

These run under ``pytest-socket`` with ``--disable-socket`` (see
``pyproject.toml``), so every HTTP call is mocked and no test can reach the
network.  All fixture payloads mirror shapes observed against
``https://api.archives-ouvertes.fr/search/`` on 2026-08-31.

The test that carries the most weight is
:meth:`TestCoverageIsHonest.test_failure_and_emptiness_are_different_statuses`.
Everything else here can pass while the connector still reports an upstream
outage as a confident zero; that one cannot.

The donor connector's entire error contract is::

    except Exception as exc:
        logger.error("HAL search failed: %s", exc)
        return []

so :class:`TestTwoHundredWithAnErrorBody` and the failure cases below are all
tests the donor fails.
"""

from __future__ import annotations

import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from lit_search.connectors.hal import HALBodyError, HALConnector
from lit_search.core.coverage import CoverageStatus, CoveredPapers
from lit_search.core.paper import Database, PaperType
from lit_search.core.source import SourceType
from lit_search.query.builders.hal import HALQueryBuilder

from .conftest import _parse

# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def _doc(index: int, **overrides: Any) -> dict[str, Any]:
    """Build one HAL Solr document.

    Parameters
    ----------
    index : int
        Used to make the title and identifiers distinct.
    **overrides : Any
        Fields to replace or add.

    Returns
    -------
    dict[str, Any]
        A record shaped like a live ``response.docs`` entry, including HAL's
        habit of returning single values inside one-element lists.
    """
    record: dict[str, Any] = {
        "halId_s": f"hal-0204248{index}",
        "title_s": [f"Synthetic Deposit {index}"],
        "abstract_s": [f"Synthetic abstract {index}."],
        "authFullName_s": ["Egon Heuson", "Franck Dumeignil"],
        "doiId_s": f"10.1039/D0CY0069{index}C",
        "publicationDate_s": "2020-11-05",
        "journalTitle_s": "Catalysis Science & Technology",
        "keyword_s": ["catalysis", "hybrid"],
        "docType_s": "ART",
        "language_s": ["en"],
        "openAccess_bool": True,
        "uri_s": f"https://hal.science/hal-0204248{index}v1",
        "fileMain_s": f"https://hal.science/hal-0204248{index}/file/main.pdf",
    }
    record.update(overrides)
    return record


def _page(
    docs: list[dict[str, Any]],
    num_found: int,
    next_cursor: str | None = None,
    num_found_exact: bool = True,
) -> dict[str, Any]:
    """Build one HAL search-response body.

    Parameters
    ----------
    docs : list[dict[str, Any]]
        Documents for this page.
    num_found : int
        Value of ``response.numFound``.
    next_cursor : str | None
        ``nextCursorMark`` to include.  Solr signals exhaustion by repeating
        the cursor that was sent, so pass the sent value to end the walk.
    num_found_exact : bool
        Value of ``response.numFoundExact``.

    Returns
    -------
    dict[str, Any]
        Response body.
    """
    body: dict[str, Any] = {
        "response": {
            "numFound": num_found,
            "numFoundExact": num_found_exact,
            "start": 0,
            "docs": docs,
        }
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
def connector() -> HALConnector:
    """Return a HAL connector."""
    return HALConnector()


def _search(connector: HALConnector, responses: list[Any], **kwargs: Any) -> CoveredPapers:
    """Run a search against a scripted sequence of ``_get`` outcomes.

    Parameters
    ----------
    connector : HALConnector
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
    query = kwargs.pop("query", None) or _parse("[catalysis]")
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
        assert connector.name == Database.HAL.value

    def test_registered_in_the_search_registry(self) -> None:
        """A connector nobody can select is not ported."""
        from lit_search.connectors import SEARCH_REGISTRY

        assert SEARCH_REGISTRY[Database.HAL] is HALConnector

    def test_min_request_interval_is_the_inferred_one_second(self, connector) -> None:
        """HAL publishes no rate limit; the value is inferred and says so."""
        assert connector.min_request_interval == 1.0

    def test_no_credential_is_declared_explicitly(self, connector) -> None:
        """HAL issues no key, so the anonymous host bucket is correct."""
        assert connector._credential_attrs == ()
        assert connector.rate_limit_scope.fingerprint == "anon"

    def test_rate_limit_service_covers_the_whole_host(self, connector) -> None:
        """A future OAI-PMH or reference connector must share this clock."""
        assert connector.rate_limit_service == "archives-ouvertes"
        assert connector.rate_limit_scope.service == "archives-ouvertes"

    def test_not_registered_for_lookup_roles(self) -> None:
        """A role that was not implemented is not claimed."""
        from lit_search.connectors import DOI_LOOKUP_REGISTRY, URL_LOOKUP_REGISTRY

        assert Database.HAL not in DOI_LOOKUP_REGISTRY
        assert Database.HAL not in URL_LOOKUP_REGISTRY


# ---------------------------------------------------------------------------
# The four required cases, plus the one that matters
# ---------------------------------------------------------------------------


class TestCoverageIsHonest:
    """Each way a search can end reports a different, actionable status."""

    def test_normal_result_is_complete(self, connector) -> None:
        """Retrieving as many records as numFound reports is COMPLETE."""
        result = _search(connector, [_response(_page([_doc(0), _doc(1)], 2, next_cursor="*"))])
        assert result.status is CoverageStatus.COMPLETE
        assert len(result.papers) == 2
        assert result.coverage.reported_total == 2

    def test_upstream_failure_is_failed(self, connector) -> None:
        """A dead first request is FAILED, and the error is carried.

        The donor returns ``[]`` here, from a bare ``except Exception``.
        """
        result = _search(connector, [requests.ConnectionError("hal unreachable")])
        assert result.status is CoverageStatus.FAILED
        assert result.papers == ()
        assert "unreachable" in (result.coverage.error or "")
        assert result.coverage.error_type == "ConnectionError"

    def test_empty_but_successful_search_is_complete(self, connector) -> None:
        """A genuine zero-hit query is COMPLETE with no papers."""
        result = _search(connector, [_response(_page([], 0, next_cursor="*"))])
        assert result.status is CoverageStatus.COMPLETE
        assert result.papers == ()
        assert result.coverage.reported_total == 0

    def test_capped_result_is_capped(self, connector) -> None:
        """Stopping at max_papers is CAPPED and names the cap."""
        result = _search(
            connector,
            [_response(_page([_doc(i) for i in range(3)], 900, next_cursor="c1"))],
            max_papers=3,
        )
        assert result.status is CoverageStatus.CAPPED
        assert len(result.papers) == 3
        assert result.coverage.cap == 3
        assert result.coverage.reported_total == 900

    def test_failure_and_emptiness_are_different_statuses(self, connector) -> None:
        """The regression test for the whole port.

        Both of these leave the donor's loop at the same ``return []``.
        """
        failure = _search(connector, [requests.ConnectionError("boom")])
        empty = _search(connector, [_response(_page([], 0, next_cursor="*"))])

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
    """HAL reports its errors with an HTTP 200 status.

    Each of these satisfies ``raise_for_status`` and then yields ``[]`` from
    ``data.get("response", {}).get("docs", [])``.
    """

    def test_error_body_is_failed_not_empty(self, connector) -> None:
        """Observed for an unknown field and for an unbalanced parenthesis."""
        result = _search(connector, [_response({"error": {"msg": "Error. See help : /docs"}})])
        assert result.status is CoverageStatus.FAILED
        assert result.papers == ()
        assert "See help" in (result.coverage.error or "")

    def test_the_error_body_is_never_reported_as_query_unsupported(self, connector) -> None:
        """HAL uses one message for every failure, so nothing can be inferred.

        Reporting QUERY_UNSUPPORTED here would be inventing a distinction the
        response does not carry.  The query shapes that *can* be recognised are
        rejected by the builder before a request is made.
        """
        result = _search(connector, [_response({"error": {"msg": "Error. See help : /docs"}})])
        assert result.status is not CoverageStatus.QUERY_UNSUPPORTED
        assert "cannot be told from the response" in (result.coverage.error or "")

    def test_missing_num_found_is_failed_not_empty(self, connector) -> None:
        """A body with docs but no numFound is not a search result."""
        result = _search(connector, [_response({"response": {"docs": []}})])
        assert result.status is CoverageStatus.FAILED
        assert "numFound" in (result.coverage.error or "")

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
        """The exception type is one ``search`` knows how to name."""
        from lit_search.exceptions import ConnectorError

        assert issubclass(HALBodyError, ConnectorError)


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class TestCursorPagination:
    """Solr's exhaustion signal, and every abnormal stop, are handled."""

    def test_multi_page_ends_when_the_cursor_repeats(self, connector) -> None:
        """Verified live: a one-record query paged 1, 0, 0 with a fixed cursor."""
        result = _search(
            connector,
            [
                _response(_page([_doc(0), _doc(1)], 3, next_cursor="c1")),
                _response(_page([_doc(2)], 3, next_cursor="c2")),
                _response(_page([], 3, next_cursor="c2")),
            ],
        )
        assert result.status is CoverageStatus.COMPLETE
        assert len(result.papers) == 3

    def test_a_restart_under_a_fresh_cursor_is_partial_not_complete(self, connector) -> None:
        """More records than ``numFound`` is never COMPLETE.

        ``nextCursorMark`` is a sort-position token -- one observed live on
        2026-08-31 was ``AoEnMTAxNTcyNw==``, base64 for a ``docid`` -- so a
        document inserted before the current position mid-walk causes records
        to be served twice, under a cursor that never repeats.  The
        cursor-repeat check therefore cannot see it, and ``numFound`` need not
        move, so the drift check cannot either.

        The arithmetic can: four records came back where HAL said three
        existed.  Re-serving means the walk also skipped what lay ahead, so
        completeness is unestablished.  Before ``from_pagination`` grew its
        over-retrieval branch this returned ``complete``.
        """
        result = _search(
            connector,
            [
                _response(_page([_doc(0), _doc(1)], 3, next_cursor="cA")),
                _response(_page([_doc(0), _doc(1)], 3, next_cursor="cB")),
                _response(_page([], 3, next_cursor="cB")),
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 4
        assert result.coverage.reported_total == 3
        assert "more distinct records than the source says exist" in (result.coverage.detail or "")

    def test_short_page_alone_does_not_end_pagination(self, connector) -> None:
        """A page smaller than the row count keeps going while a cursor advances."""
        result = _search(
            connector,
            [
                _response(_page([_doc(0)], 2, next_cursor="c1")),
                _response(_page([_doc(1)], 2, next_cursor="c1")),
            ],
        )
        assert result.status is CoverageStatus.COMPLETE
        assert len(result.papers) == 2

    def test_cursor_exhausted_before_num_found_is_partial(self, connector) -> None:
        """Fewer records than numFound, with the cursor spent, is PARTIAL."""
        result = _search(connector, [_response(_page([_doc(0)], 50, next_cursor="*"))])
        assert result.status is CoverageStatus.PARTIAL
        assert result.coverage.reported_total == 50
        assert "50" in (result.coverage.detail or "")

    def test_empty_page_with_a_fresh_cursor_is_partial(self, connector) -> None:
        """The API offered more and delivered nothing: stop and say so."""
        result = _search(
            connector,
            [
                _response(_page([_doc(0)], 9, next_cursor="c1")),
                _response(_page([], 9, next_cursor="c2")),
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert "empty page" in (result.coverage.detail or "")

    def test_missing_next_cursor_stops_the_run(self, connector) -> None:
        """A response with no cursor at all cannot be paged further."""
        result = _search(connector, [_response(_page([_doc(0)], 1))])
        assert result.status is CoverageStatus.COMPLETE
        assert len(result.papers) == 1

    def test_failure_after_some_pages_is_partial_not_failed(self, connector) -> None:
        """Records that did arrive are real, so the run is PARTIAL.

        FAILED would discard true records; the donor discards them too, and
        then reports the whole run as an empty success.
        """
        result = _search(
            connector,
            [
                _response(_page([_doc(0), _doc(1)], 90, next_cursor="c1")),
                requests.ConnectionError("dropped mid-run"),
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 2
        assert result.coverage.error_type == "ConnectionError"
        assert "dropped mid-run" in (result.coverage.detail or "")

    def test_error_body_after_some_pages_is_partial(self, connector) -> None:
        """A 200-with-error partway through degrades rather than discards."""
        result = _search(
            connector,
            [
                _response(_page([_doc(0)], 90, next_cursor="c1")),
                _response({"error": {"msg": "Error. See help : /docs"}}),
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 1

    def test_num_found_drift_between_pages_is_partial(self, connector) -> None:
        """A moving index means neither total can be checked."""
        result = _search(
            connector,
            [
                _response(_page([_doc(0), _doc(1)], 3, next_cursor="c1")),
                _response(_page([_doc(2)], 7, next_cursor="c1")),
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert "changed mid-retrieval" in (result.coverage.detail or "")

    def test_an_inexact_num_found_cannot_justify_complete(self, connector) -> None:
        """Solr may report numFound as a lower bound rather than a count.

        Claiming COMPLETE against a lower bound is exactly the unfalsifiable
        completeness the reviews found four ways to fake.
        """
        result = _search(
            connector,
            [_response(_page([_doc(0), _doc(1)], 2, next_cursor="*", num_found_exact=False))],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert "lower bound" in (result.coverage.detail or "")

    def test_an_inexact_total_that_also_hits_the_cap_stays_partial(self, connector) -> None:
        """CAPPED would understate what is missing, so PARTIAL wins.

        ``CAPPED`` means "a known amount is missing, on purpose", and with an
        inexact ``numFound`` the amount is not known.  This assertion is what
        makes the explicit PARTIAL branch in ``_build_coverage`` load-bearing:
        without it, routing this case through ``from_pagination`` would return
        CAPPED and every other test would still pass, because the lower-bound
        note alone keeps the *uncapped* case PARTIAL.
        """
        result = _search(
            connector,
            [
                _response(
                    _page([_doc(i) for i in range(3)], 900, next_cursor="c1", num_found_exact=False)
                )
            ],
            max_papers=3,
        )
        assert result.status is CoverageStatus.PARTIAL
        assert "lower bound" in (result.coverage.detail or "")

    def test_unparsable_records_are_counted_as_a_shortfall(self, connector) -> None:
        """A record seen but not parsed is a coverage loss, not an absence."""
        result = _search(
            connector,
            [_response(_page([_doc(0), _doc(1, title_s=[])], 2, next_cursor="*"))],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 1
        assert "could not be parsed" in (result.coverage.detail or "")

    def test_coverage_count_always_matches_the_papers_returned(self, connector) -> None:
        """CoveredPapers rejects a mismatch, so this asserts the accounting."""
        result = _search(
            connector,
            [_response(_page([_doc(i) for i in range(5)], 99, next_cursor="c1"))],
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
        query = kwargs.pop("query", None) or _parse("[catalysis]")
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(
                connector, "_get", return_value=_response(_page([], 0, next_cursor="*"))
            ) as mocked,
        ):
            connector.search(query, **kwargs)
        params: dict[str, Any] = mocked.call_args.args[1]
        return params

    def test_cursor_paging_always_sends_a_sort(self, connector) -> None:
        """A cursorMark without a sort returns HAL's opaque error body."""
        params = self._capture(connector)
        assert params["cursorMark"] == "*"
        assert params["sort"] == "docid asc"
        assert params["wt"] == "json"

    def test_requested_fields_are_all_verified_names(self, connector) -> None:
        """An unknown ``fl`` name yields fieldless documents, not an error."""
        requested = set(self._capture(connector)["fl"].split(","))
        assert {"halId_s", "title_s", "abstract_s", "doiId_s", "uri_s"} <= requested

    def test_rows_is_at_least_one_when_the_cap_is_one(self, connector) -> None:
        """A cap of 1 must not produce rows=0."""
        assert self._capture(connector, max_papers=1)["rows"] >= 1

    def test_date_bounds_use_the_solr_date_field(self, connector) -> None:
        """A range on the string date field would compare lexically."""
        params = self._capture(
            connector,
            since=datetime.date(2020, 1, 1),
            until=datetime.date(2021, 12, 31),
        )
        assert "publicationDate_tdate:[2020-01-01T00:00:00Z TO 2021-12-31T23:59:59Z]" in params["q"]

    def test_open_upper_bound_uses_the_wildcard(self, connector) -> None:
        """An open end is Solr's ``*``."""
        params = self._capture(connector, since=datetime.date(2020, 1, 1))
        assert "TO *]" in params["q"]

    def test_no_date_filter_when_no_bounds_are_given(self, connector) -> None:
        """An unbounded search must not silently acquire a date restriction."""
        assert "publicationDate_tdate" not in self._capture(connector)["q"]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    """Field mapping from a HAL Solr document to a Paper."""

    def test_core_fields(self, connector) -> None:
        """Title, abstract, DOI, date, URLs and language are mapped."""
        paper = connector._parse_doc(_doc(7))
        assert paper is not None
        assert paper.title == "Synthetic Deposit 7"
        assert paper.abstract == "Synthetic abstract 7."
        assert paper.doi == "10.1039/D0CY00697C"
        assert paper.publication_date == datetime.date(2020, 11, 5)
        assert paper.url == "https://hal.science/hal-02042487v1"
        assert paper.pdf_url == "https://hal.science/hal-02042487/file/main.pdf"
        assert paper.language == "en"
        assert paper.is_open_access is True
        assert paper.found_in == {Database.HAL.value}
        assert paper.keywords == {"catalysis", "hybrid"}

    def test_single_values_arrive_inside_lists(self, connector) -> None:
        """HAL wraps most text fields in a one-element list."""
        paper = connector._parse_doc(_doc(1, title_s=["Wrapped"], abstract_s=["Also wrapped"]))
        assert paper is not None
        assert paper.title == "Wrapped"
        assert paper.abstract == "Also wrapped"

    def test_a_subtitle_is_appended_to_the_title(self, connector) -> None:
        """Dropping it would lose the half that distinguishes similar deposits."""
        paper = connector._parse_doc(_doc(1, subTitle_s=["a second look"]))
        assert paper is not None
        assert paper.title.endswith(": a second look")

    def test_journal_deposits_become_a_journal_source(self, connector) -> None:
        """``docType_s: ART`` with a journal title is a journal article."""
        paper = connector._parse_doc(_doc(1))
        assert paper is not None
        assert paper.source is not None
        assert paper.source.source_type is SourceType.JOURNAL
        assert paper.paper_type is PaperType.ARTICLE

    def test_conference_doc_types_become_a_conference_source(self, connector) -> None:
        """``COMM`` is HAL's code for a conference paper."""
        paper = connector._parse_doc(_doc(1, docType_s="COMM"))
        assert paper is not None
        assert paper.source is not None
        assert paper.source.source_type is SourceType.CONFERENCE

    def test_a_venueless_deposit_becomes_a_repository_source(self, connector) -> None:
        """A deposit with no journal is still from somewhere: HAL itself."""
        paper = connector._parse_doc(_doc(1, journalTitle_s="", docType_s="UNDEFINED"))
        assert paper is not None
        assert paper.source is not None
        assert paper.source.source_type is SourceType.REPOSITORY
        assert paper.paper_type is PaperType.UNPUBLISHED

    def test_a_year_only_date_falls_back_to_january(self, connector) -> None:
        """HAL records a produced date as a bare year for older deposits."""
        paper = connector._parse_doc(_doc(1, publicationDate_s="2016"))
        assert paper is not None
        assert paper.publication_date == datetime.date(2016, 1, 1)

    def test_the_date_falls_back_through_produced_and_submitted(self, connector) -> None:
        """Not every deposit carries a publication date."""
        record = _doc(1)
        del record["publicationDate_s"]
        record["submittedDate_s"] = "2019-02-20 14:22:57"
        paper = connector._parse_doc(record)
        assert paper is not None
        assert paper.publication_date == datetime.date(2019, 2, 20)

    def test_the_url_falls_back_to_a_hal_landing_page(self, connector) -> None:
        """A record with no ``uri_s`` is still addressable by its HAL id."""
        paper = connector._parse_doc(_doc(1, uri_s=""))
        assert paper is not None
        assert paper.url == "https://hal.science/hal-02042481"

    def test_untitled_record_is_rejected(self, connector) -> None:
        """A record with no title cannot become a Paper."""
        assert connector._parse_doc(_doc(1, title_s=[])) is None

    def test_non_dict_record_is_rejected(self, connector) -> None:
        """A malformed entry is counted as unparsed, not crashed on."""
        assert connector._parse_doc("not a document") is None


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------


class TestQueryBuilder:
    """HAL indexes an equivalent of every filter code, with two traps."""

    @pytest.fixture
    def builder(self) -> HALQueryBuilder:
        """Return a HAL query builder."""
        return HALQueryBuilder()

    @pytest.mark.parametrize(
        ("query_string", "expected"),
        [
            ("ti[catalysis]", 'title_t:"catalysis"'),
            ("abs[catalysis]", 'abstract_t:"catalysis"'),
            ("key[catalysis]", 'keyword_t:"catalysis"'),
            ("au[dupont]", 'authFullName_t:"dupont"'),
            ("src[nature]", 'journalTitle_t:"nature"'),
            ("aff[sorbonne]", 'structName_t:"sorbonne"'),
        ],
    )
    def test_each_filter_maps_to_a_verified_field(self, builder, query_string, expected) -> None:
        """Every emitted name returned a plausible hit count on the live API."""
        assert builder.convert_query(_parse(query_string)) == expected

    def test_the_default_scope_expands_to_a_disjunction(self, builder) -> None:
        """HAL has no single title+abstract+keyword field, so it is written out."""
        converted = builder.convert_query(_parse("[catalysis]"))
        assert converted == (
            '(title_t:"catalysis" OR abstract_t:"catalysis" OR keyword_t:"catalysis")'
        )

    def test_title_abstract_scope_omits_keywords(self, builder) -> None:
        """``tiabs`` must not silently widen to include keywords."""
        converted = builder.convert_query(_parse("tiabs[catalysis]"))
        assert "keyword_t" not in converted
        assert converted == '(title_t:"catalysis" OR abstract_t:"catalysis")'

    def test_connectors_map_to_solr_operators(self, builder) -> None:
        """HAL is Solr, so it has real AND, OR and NOT."""
        converted = builder.convert_query(_parse("ti[a] AND ti[b] AND NOT ti[c]"))
        assert " AND " in converted
        assert " NOT " in converted

    def test_wildcard_terms_are_left_unquoted(self, builder) -> None:
        """Solr does not expand wildcards inside a quoted phrase."""
        assert builder.convert_query(_parse("ti[catalys*]")) == "title_t:catalys*"

    def test_wildcard_with_a_space_is_rejected(self, builder) -> None:
        """Measured: the second token escapes the field prefix.

        ``title_t:machine`` gave 14,537 and ``title_t:learn*`` 45,173, while
        ``title_t:machine learn*`` gave 8,131 and the grouped form 7,699 -- so
        the ungrouped form is neither, and it runs without complaint.
        """
        result = builder.validate_query(_parse("ti[machine learn*]"))
        assert not result.is_valid
        assert "wildcard" in (result.error_message or "")

    def test_a_quote_inside_a_term_is_rejected(self, builder) -> None:
        """It would unbalance the phrase and produce HAL's opaque error body."""
        from lit_search.core.query import NodeType, Query, QueryNode

        term = QueryNode(node_type=NodeType.TERM, value='cata"lysis')
        query = Query(raw_query="unused", root=QueryNode(node_type=NodeType.ROOT, children=[term]))
        result = builder.validate_query(query)
        assert not result.is_valid
        assert "quote" in (result.error_message or "")

    def test_an_empty_term_is_rejected(self, builder) -> None:
        """An empty clause returns HAL's error body, which reads as zero."""
        from lit_search.core.query import NodeType, Query, QueryNode

        term = QueryNode(node_type=NodeType.TERM, value="   ")
        query = Query(raw_query="unused", root=QueryNode(node_type=NodeType.ROOT, children=[term]))
        assert not builder.validate_query(query).is_valid

    def test_a_valid_query_passes(self, builder) -> None:
        """The common case is not accidentally rejected."""
        assert builder.validate_query(_parse("[machine learning] AND ti[deep]")).is_valid

    def test_rejected_query_reaches_the_ledger_as_query_unsupported(self, connector) -> None:
        """And issues no request, because retrying it would change nothing."""
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(connector, "_get") as mocked,
        ):
            result = connector.search(_parse("[machine learn*]"))
        assert result.status is CoverageStatus.QUERY_UNSUPPORTED
        assert result.papers == ()
        mocked.assert_not_called()


# ---------------------------------------------------------------------------
# The base class checks the claim
# ---------------------------------------------------------------------------


class TestCompleteIsVerifiedNotAccepted:
    """A connector that swallows a failure cannot claim COMPLETE.

    This proves the connector's HTTP goes through ``_get`` and therefore
    through ``ConnectorBase._request_with_retry``, which records every failed
    attempt on the retrieval audit *before* the connector's own ``except``
    clause runs.  Patching ``_get`` -- as every other test here does -- would
    bypass that channel, so this one patches the retry loop's inner half.
    """

    def test_a_swallowed_page_failure_is_downgraded_to_partial(self) -> None:
        """The audit contradicts a COMPLETE the connector still reported."""
        from lit_search.core.coverage import SourceCoverage

        class _Liar(HALConnector):
            """A connector that reports COMPLETE whatever happened."""

            def _build_coverage(self, *, retrieved, run, max_papers):
                """Claim COMPLETE unconditionally."""
                return SourceCoverage.complete(self.name, retrieved, reported_total=retrieved)

            @staticmethod
            def _record_page_failure(run, exc, cursor):
                """Swallow the failure and keep whatever was retrieved."""
                return None

        liar = _Liar()
        outcomes: list[Any] = [
            _response(_page([_doc(0)], 90, next_cursor="c1")),
            requests.ConnectionError("dropped mid-run"),
        ]
        with (
            patch.object(liar, "_rate_limit"),
            patch.object(liar, "_request_with_retry_uncounted", side_effect=outcomes),
        ):
            result = liar.search(_parse("[catalysis]"))

        assert result.status is CoverageStatus.PARTIAL
        assert "ConnectionError" in (result.coverage.detail or "")
