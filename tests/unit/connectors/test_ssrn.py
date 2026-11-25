"""Unit tests for SSRNConnector.

These run under ``pytest-socket`` with ``--disable-socket`` (see
``pyproject.toml``), so every HTTP call is mocked and no test can reach the
network.  The record shape used below was observed live on 2026-08-31 against
``api.ssrn.com/content/v1/bindings/{id}/papers`` -- SSRN's *browse* route,
which answers without a credential.  The *search* route answers HTTP 401, so
its envelope has never been seen, and that fact is what
:class:`TestSSRNIsNeverComplete` pins.

The test that carries the most weight is
:meth:`TestCoverageIsHonest.test_failure_and_emptiness_are_different_statuses`.
Everything else here can pass while the connector still reports an outage as a
confident zero; that one cannot.

The donor's loop ends the same way for a fetch failure, a markup change and a
genuinely empty search::

    html, err = self._fetch_page(query, page)
    if err or not html:
        logger.warning(...)
        break
    page_papers = self._parse_results(html)
    if not page_papers:
        break

It even recognises the Cloudflare challenge by name and then discards the
recognition.  :class:`TestTheChallengeIsNotZero` is the test that separates
those outcomes.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from lit_search.connectors.ssrn import (
    CREDENTIAL_NAME,
    SSRNAccessBlockedError,
    SSRNBodyError,
    SSRNConnector,
)
from lit_search.core.coverage import CoverageStatus, CoveredPapers
from lit_search.core.paper import Database, PaperType
from lit_search.core.source import SourceType
from lit_search.query.builders.ssrn import SSRNQueryBuilder

from .conftest import _parse

_CHALLENGE_HTML = (
    '<!DOCTYPE html><html lang="en-US"><head><title>Just a moment...</title>'
    '<meta http-equiv="content-security-policy" content="script-src '
    'https://challenges.cloudflare.com"></head></html>'
)


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def _record(index: int, **overrides: Any) -> dict[str, Any]:
    """Build one SSRN paper record.

    Field names and shapes come from a live response on 2026-08-31.

    Parameters
    ----------
    index : int
        Used to make the title and identifiers distinct.
    **overrides : Any
        Fields to replace or add.

    Returns
    -------
    dict[str, Any]
        A record shaped like a live ``papers`` entry.
    """
    record: dict[str, Any] = {
        "abstract_type": "Working Paper Series",
        "publication_status": "UNDER REVIEW",
        "is_paid": False,
        "page_count": 39 + index,
        "title": f"Synthetic Working Paper {index}",
        "authors": [
            {"id": 12959751 + index, "last_name": "Jian", "first_name": "Zhi-Hong"},
            {"id": 6922742 + index, "last_name": "Chen", "first_name": "Wei James"},
        ],
        "affiliations": "Pennsylvania State University - Department of Economics",
        "id": 7369158 + index,
        "is_approved": True,
        "approved_date": "31 Jul 2026",
        "downloads": index,
        "url": f"https://papers.ssrn.com/sol3/papers.cfm?abstract_id={7369158 + index}",
    }
    record.update(overrides)
    return record


def _page(records: list[dict[str, Any]], total: int) -> dict[str, Any]:
    """Build one SSRN response envelope.

    Parameters
    ----------
    records : list[dict[str, Any]]
        Records for this page.
    total : int
        Value of the ``total`` field.

    Returns
    -------
    dict[str, Any]
        Response body.
    """
    return {"total": total, "papers": records}


def _response(body: Any, status_code: int = 200, text: str = "") -> MagicMock:
    """Wrap *body* in a mock :class:`requests.Response`.

    Parameters
    ----------
    body : Any
        Object returned by ``.json()``.
    status_code : int
        HTTP status to report.
    text : str
        Raw body text, used by the challenge detector.  When not given it is
        the JSON serialization of *body*, because on a real response ``.text``
        and ``.json()`` are the same bytes.  Defaulting it to ``""`` decoupled
        the two and hid a defect: the challenge detector reads ``.text``, so a
        mock with empty text can never reproduce a marker matching *record
        content*, which is exactly what happened in production.

    Returns
    -------
    MagicMock
        Mock response.
    """
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = body
    if not text:
        try:
            text = json.dumps(body)
        except (TypeError, ValueError):
            text = str(body)
    mock.text = text
    return mock


def _html_response(text: str, status_code: int = 200) -> MagicMock:
    """Wrap an HTML body in a mock response whose ``.json()`` raises.

    Cloudflare's interstitial is an HTML page, so a faithful mock of one must
    fail JSON decoding the way a real response does.  The earlier fixture
    returned ``{}`` from ``.json()`` while carrying HTML text, which is a
    combination no real response produces, and it let the connector's challenge
    check pass a test it would fail on a body shaped like the live one.

    Parameters
    ----------
    text : str
        Raw HTML body.
    status_code : int
        HTTP status to report.

    Returns
    -------
    MagicMock
        Mock response.
    """
    mock = MagicMock()
    mock.status_code = status_code
    mock.text = text
    mock.json.side_effect = ValueError("Expecting value: line 1 column 1 (char 0)")
    return mock


def _http_error(status_code: int, text: str) -> requests.HTTPError:
    """Build an :class:`requests.HTTPError` carrying an HTML body.

    Parameters
    ----------
    status_code : int
        HTTP status of the rejected request.
    text : str
        Raw body text.

    Returns
    -------
    requests.HTTPError
        Exception shaped like the one ``_get`` raises.
    """
    error = requests.HTTPError(f"{status_code} Client Error")
    error.response = _response({}, status_code, text=text)
    return error


@pytest.fixture
def connector() -> SSRNConnector:
    """Return an SSRN connector holding a credential."""
    return SSRNConnector(api_key="test-key")


def _search(connector: SSRNConnector, responses: list[Any], **kwargs: Any) -> CoveredPapers:
    """Run a search against a scripted sequence of ``_get`` outcomes.

    Parameters
    ----------
    connector : SSRNConnector
        Connector under test.
    responses : list[Any]
        Values or exceptions for successive ``_get`` calls.
    **kwargs : Any
        Forwarded to ``search``, plus an optional ``query``.

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
        assert connector.name == Database.SSRN.value

    def test_registered_in_the_search_registry(self) -> None:
        """SSRN is registered even though it cannot run by default.

        An unregistered source is invisible, and "SSRN was not consulted" has
        to be sayable.  That is the registry-level version of the silence this
        whole contract exists to remove.
        """
        from lit_search.connectors import SEARCH_REGISTRY

        assert SEARCH_REGISTRY[Database.SSRN] is SSRNConnector

    def test_min_request_interval_is_ssrns_own_published_crawl_delay(self, connector) -> None:
        """robots.txt publishes ``Crawl-Delay: 5``, used verbatim."""
        assert connector.min_request_interval == 5.0

    def test_the_credential_keys_the_rate_limit_scope(self, connector) -> None:
        """Two keys against SSRN must not share one clock."""
        assert connector._credential_attrs == ("_api_key",)
        assert connector.rate_limit_scope.service == Database.SSRN.value
        assert connector.rate_limit_scope.fingerprint != "anon"
        assert SSRNConnector().rate_limit_scope.fingerprint == "anon"

    def test_constructing_without_a_key_does_not_raise(self) -> None:
        """The connector object must exist so it can report why it did nothing."""
        assert SSRNConnector().name == Database.SSRN.value

    def test_not_registered_for_lookup_roles(self) -> None:
        """A role that was not implemented is not claimed."""
        from lit_search.connectors import DOI_LOOKUP_REGISTRY, URL_LOOKUP_REGISTRY

        assert Database.SSRN not in DOI_LOOKUP_REGISTRY
        assert Database.SSRN not in URL_LOOKUP_REGISTRY


# ---------------------------------------------------------------------------
# The required cases, plus the one that matters
# ---------------------------------------------------------------------------


class TestCoverageIsHonest:
    """Each way a search can end reports a different, actionable status."""

    def test_a_missing_credential_is_skipped_not_empty(self) -> None:
        """SSRN's search route answers 401, so it cannot be asked anonymously.

        A default run therefore reports "skipped, supply ssrn_api_key" rather
        than reporting zero papers or omitting SSRN from the ledger.
        """
        anonymous = SSRNConnector()
        with (
            patch.object(anonymous, "_rate_limit"),
            patch.object(anonymous, "_get") as mocked,
        ):
            result = anonymous.search(_parse("[catalysis]"))
        assert result.status is CoverageStatus.SKIPPED_NO_CREDENTIAL
        assert result.coverage.missing_credential == CREDENTIAL_NAME
        assert result.papers == ()
        mocked.assert_not_called()

    def test_a_normal_result_carries_the_records(self, connector) -> None:
        """A successful search returns papers and a total."""
        result = _search(connector, [_response(_page([_record(0), _record(1)], 2))])
        assert len(result.papers) == 2
        assert result.coverage.reported_total == 2
        assert result.coverage.retrieved == 2

    def test_upstream_failure_is_failed(self, connector) -> None:
        """A dead first request is FAILED, and the error is carried."""
        result = _search(connector, [requests.ConnectionError("ssrn unreachable")])
        assert result.status is CoverageStatus.FAILED
        assert result.papers == ()
        assert "unreachable" in (result.coverage.error or "")
        assert result.coverage.error_type == "ConnectionError"

    def test_an_empty_search_is_not_a_failure(self, connector) -> None:
        """SSRN answered and matched nothing; that is a different fact."""
        result = _search(connector, [_response(_page([], 0))])
        assert result.status is CoverageStatus.PARTIAL
        assert result.papers == ()
        assert result.coverage.reported_total == 0
        assert result.coverage.error is None

    def test_a_capped_result_names_the_cap_but_stays_partial(self, connector) -> None:
        """The cap is recorded; the status is still the worse of the two.

        ``CAPPED`` means "a known amount is missing, on purpose" and ranks
        milder than ``PARTIAL``.  Here the cap sits on top of a degradation
        whose size is unknown, so reporting CAPPED would understate what is
        missing.  This assertion is also what stops
        :meth:`SSRNConnector._build_coverage` being quietly rewritten to route
        through ``from_pagination``, which would return CAPPED here.
        """
        result = _search(
            connector, [_response(_page([_record(i) for i in range(3)], 900))], max_papers=3
        )
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 3
        assert result.coverage.cap == 3
        assert result.coverage.reported_total == 900
        assert "cap of 3" in (result.coverage.detail or "")

    def test_failure_and_emptiness_are_different_statuses(self, connector) -> None:
        """The regression test for the whole port.

        In the donor implementation both of these leave the loop by the same
        ``break`` and arrive at the same ``return papers[:max_results]`` with
        ``papers == []``.
        """
        failure = _search(connector, [requests.ConnectionError("boom")])
        empty = _search(connector, [_response(_page([], 0))])

        assert failure.papers == empty.papers == ()
        assert failure.status is not empty.status
        assert failure.status is CoverageStatus.FAILED
        assert empty.status is CoverageStatus.PARTIAL
        assert not failure.coverage.is_complete
        assert not empty.coverage.is_complete

    def test_all_four_no_paper_outcomes_are_distinguishable(self) -> None:
        """Skipped, unsupported, failed and empty are four different answers.

        Under the donor contract all four are ``[]``.
        """
        blocked = SSRNConnector(api_key="k")
        outcomes = {
            "skipped": SSRNConnector().search(_parse("[a]")).status,
            "unsupported": blocked.search(_parse("[a] AND [b]")).status,
            "failed": _search(blocked, [requests.ConnectionError("x")]).status,
            "empty": _search(blocked, [_response(_page([], 0))]).status,
        }
        assert outcomes == {
            "skipped": CoverageStatus.SKIPPED_NO_CREDENTIAL,
            "unsupported": CoverageStatus.QUERY_UNSUPPORTED,
            "failed": CoverageStatus.FAILED,
            "empty": CoverageStatus.PARTIAL,
        }


# ---------------------------------------------------------------------------
# SSRN is never COMPLETE
# ---------------------------------------------------------------------------


class TestSSRNIsNeverComplete:
    """The degraded label the specification requires, enforced rather than written."""

    @pytest.mark.parametrize("total", [0, 1, 2])
    def test_no_response_shape_produces_complete(self, connector, total) -> None:
        """Even ``retrieved == total`` stays PARTIAL.

        The envelope this connector parses was observed on SSRN's browse route
        and never on its search route, so nothing establishes that ``total``
        counts matches rather than a capped figure.
        """
        result = _search(connector, [_response(_page([_record(i) for i in range(total)], total))])
        assert result.status is CoverageStatus.PARTIAL
        assert not result.coverage.is_complete

    def test_the_degraded_reason_is_in_every_result(self, connector) -> None:
        """A label nobody can read is not a label."""
        result = _search(connector, [_response(_page([_record(0)], 1))])
        assert "degraded source" in (result.coverage.detail or "")
        assert "never observed answering" in (result.coverage.detail or "")

    def test_a_date_bound_is_recorded_as_a_client_side_restriction(self, connector) -> None:
        """SSRN cannot apply one, so the ledger must say where it was applied."""
        import datetime

        result = _search(
            connector,
            [_response(_page([_record(0)], 1))],
            since=datetime.date(2020, 1, 1),
        )
        assert "not sent to SSRN" in (result.coverage.detail or "")

    def test_a_cap_under_a_date_bound_names_the_compound_loss(self, connector) -> None:
        """Capping the undated corpus and then filtering is a real recall loss.

        Retrieving 3 records of 900 and only then narrowing to a date range can
        leave nothing, while thousands of in-range records existed.  Nothing in
        the paper count shows that, so it is stated.
        """
        import datetime

        result = _search(
            connector,
            [_response(_page([_record(i) for i in range(3)], 900))],
            max_papers=3,
            since=datetime.date(2020, 1, 1),
            until=datetime.date(2021, 12, 31),
        )
        assert "truncated the undated corpus" in (result.coverage.detail or "")

    def test_no_date_note_when_no_bound_was_asked_for(self, connector) -> None:
        """An unbounded search must not acquire a warning about a bound."""
        result = _search(connector, [_response(_page([_record(0)], 1))])
        assert "not sent to SSRN" not in (result.coverage.detail or "")


# ---------------------------------------------------------------------------
# The Cloudflare challenge
# ---------------------------------------------------------------------------


class TestTheChallengeIsNotZero:
    """The donor detects the challenge by name and then reports zero papers."""

    def test_a_challenge_on_an_http_error_is_failed_and_named(self, connector) -> None:
        """Verified live: every SSRN path answers 403 with this interstitial."""
        result = _search(connector, [_http_error(403, _CHALLENGE_HTML)])
        assert result.status is CoverageStatus.FAILED
        assert result.papers == ()
        assert "anti-bot challenge" in (result.coverage.error or "")
        assert result.coverage.error_type == SSRNAccessBlockedError.__name__

    def test_a_challenge_served_with_http_200_is_also_failed(self, connector) -> None:
        """A challenge behind a 200 would otherwise reach the JSON decoder."""
        result = _search(connector, [_html_response(_CHALLENGE_HTML)])
        assert result.status is CoverageStatus.FAILED
        assert "anti-bot challenge" in (result.coverage.error or "")

    def test_a_result_whose_record_text_matches_a_marker_is_not_a_challenge(
        self, connector
    ) -> None:
        """A paper titled "Just a Moment" is a result, not an access block.

        The marker scan used to run over the whole raw body before JSON
        decoding, so it matched record content as readily as an interstitial.
        A genuine page was then reported ``FAILED`` with "not transient and
        will not clear on a retry" -- a false diagnosis that tells the user to
        stop retrying a search that worked, and drops the records.
        """
        record = _record(1, title="Just a Moment: Timing in Securities Regulation")
        result = _search(connector, [_response(_page([record], 1))])
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 1
        assert result.coverage.error_type is None
        assert "anti-bot challenge" not in (result.coverage.detail or "")

    def test_an_ordinary_http_error_is_not_reported_as_a_challenge(self, connector) -> None:
        """A 500 is transient; a challenge is not.  They must not be conflated."""
        result = _search(connector, [_http_error(500, "<html>Server Error</html>")])
        assert result.status is CoverageStatus.FAILED
        assert result.coverage.error_type == "HTTPError"

    def test_the_challenge_error_is_a_connector_error(self) -> None:
        """The exception type is one ``search`` knows how to name."""
        from lit_search.exceptions import ConnectorError

        assert issubclass(SSRNAccessBlockedError, ConnectorError)
        assert issubclass(SSRNBodyError, ConnectorError)


# ---------------------------------------------------------------------------
# Bodies that are not results
# ---------------------------------------------------------------------------


class TestUnreadableBodies:
    """Anything that is not a result set is raised, never defaulted to zero."""

    def test_non_json_body_is_failed(self, connector) -> None:
        """The XML the same endpoint serves by default is not silently zero.

        Observed live: without ``Accept: application/json`` this endpoint
        returns an XML ``PaperResultSet``.
        """
        broken = MagicMock()
        broken.status_code = 200
        broken.text = "<PaperResultSet><total>59666</total></PaperResultSet>"
        broken.json.side_effect = ValueError("Expecting value")
        result = _search(connector, [broken])
        assert result.status is CoverageStatus.FAILED

    def test_json_array_body_is_failed(self, connector) -> None:
        """A JSON body of the wrong shape is a failure."""
        result = _search(connector, [_response([1, 2, 3])])
        assert result.status is CoverageStatus.FAILED

    def test_a_body_without_papers_is_failed_not_empty(self, connector) -> None:
        """No ``papers`` list means the body is not a result set."""
        result = _search(connector, [_response({"total": 5})])
        assert result.status is CoverageStatus.FAILED
        assert "papers" in (result.coverage.error or "")


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class TestPagination:
    """Index-based paging, with every abnormal stop recorded."""

    def test_pages_until_the_total_is_reached(self, connector) -> None:
        """The index advances by the number of records actually seen."""
        result = _search(
            connector,
            [
                _response(_page([_record(0), _record(1)], 3)),
                _response(_page([_record(2)], 3)),
            ],
        )
        assert len(result.papers) == 3

    def test_an_empty_page_short_of_the_total_is_recorded(self, connector) -> None:
        """SSRN said more existed and then stopped supplying them."""
        result = _search(
            connector,
            [
                _response(_page([_record(0)], 50)),
                _response(_page([], 50)),
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert "stopped returning records" in (result.coverage.detail or "")

    def test_failure_after_some_pages_keeps_the_records(self, connector) -> None:
        """Records that did arrive are real; the donor discards them."""
        result = _search(
            connector,
            [
                _response(_page([_record(0), _record(1)], 90)),
                requests.ConnectionError("dropped mid-run"),
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 2
        assert result.coverage.error_type == "ConnectionError"
        assert "dropped mid-run" in (result.coverage.detail or "")

    def test_unparsable_records_are_counted_as_a_shortfall(self, connector) -> None:
        """A record seen but not parsed is a coverage loss, not an absence."""
        result = _search(connector, [_response(_page([_record(0), _record(1, title="")], 2))])
        assert len(result.papers) == 1
        assert "could not be parsed" in (result.coverage.detail or "")

    def test_coverage_count_always_matches_the_papers_returned(self, connector) -> None:
        """CoveredPapers rejects a mismatch, so this asserts the accounting."""
        result = _search(
            connector, [_response(_page([_record(i) for i in range(5)], 99))], max_papers=2
        )
        assert result.coverage.retrieved == len(result.papers) == 2


# ---------------------------------------------------------------------------
# Request shaping
# ---------------------------------------------------------------------------


class TestRequestShaping:
    """Parameters and headers match what the live endpoint accepts."""

    def _capture(self, connector, **kwargs) -> dict[str, Any]:
        """Run one search and return the parameters of the first request."""
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(connector, "_get", return_value=_response(_page([], 0))) as mocked,
        ):
            connector.search(_parse("[catalysis]"), **kwargs)
        params: dict[str, Any] = mocked.call_args.args[1]
        return params

    def test_the_term_and_the_window_are_sent(self, connector) -> None:
        """Both parameter names were observed on the reachable route."""
        params = self._capture(connector)
        assert params["query"] == "catalysis"
        assert params["index"] == 0
        assert params["count"] >= 1

    def test_count_is_at_least_one_when_the_cap_is_one(self, connector) -> None:
        """A cap of 1 must not produce count=0."""
        assert self._capture(connector, max_papers=1)["count"] >= 1

    def test_json_is_requested_explicitly(self, connector) -> None:
        """Without it the same endpoint returns XML, which parses as nothing."""
        headers = connector._prepare_headers({})
        assert headers["Accept"] == "application/json"

    def test_the_credential_is_sent_when_present(self, connector) -> None:
        """The header name is unverified but its absence would be worse."""
        assert connector._prepare_headers({})["X-API-Key"] == "test-key"

    def test_no_credential_header_when_there_is_no_credential(self) -> None:
        """An anonymous client must not send an empty credential."""
        assert "X-API-Key" not in SSRNConnector()._prepare_headers({})


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    """Field mapping from an SSRN record to a Paper."""

    def test_core_fields(self, connector) -> None:
        """Title, authors, date, URL and page count are mapped."""
        paper = connector._parse_record(_record(7))
        assert paper is not None
        assert paper.title == "Synthetic Working Paper 7"
        assert [author.name for author in paper.authors] == [
            "Zhi-Hong Jian",
            "Wei James Chen",
        ]
        assert paper.url.endswith("abstract_id=7369165")
        assert paper.page_count == 46
        assert paper.found_in == {Database.SSRN.value}
        assert paper.paper_type is PaperType.UNPUBLISHED

    def test_the_series_becomes_a_repository_source(self, connector) -> None:
        """SSRN is a preprint server, so every record is a repository deposit."""
        paper = connector._parse_record(_record(0))
        assert paper is not None
        assert paper.source is not None
        assert paper.source.title == "Working Paper Series"
        assert paper.source.source_type is SourceType.REPOSITORY

    def test_ssrn_records_carry_no_abstract_and_no_doi(self, connector) -> None:
        """Recorded because it degrades re-ranking, not because it is wanted.

        SSRN's list records carry neither field, so every SSRN paper reaches a
        re-ranker on its title alone.  Asserting it here stops the absence
        being mistaken for a parsing bug later.
        """
        paper = connector._parse_record(_record(0))
        assert paper is not None
        assert paper.abstract == ""
        assert paper.doi is None

    def test_the_approved_date_is_parsed(self, connector) -> None:
        """SSRN formats it as ``31 Jul 2026``, not ISO."""
        import datetime

        paper = connector._parse_record(_record(0))
        assert paper is not None
        assert paper.publication_date == datetime.date(2026, 7, 31)

    def test_the_url_falls_back_to_the_abstract_id(self, connector) -> None:
        """A record with no ``url`` is still addressable."""
        paper = connector._parse_record(_record(0, url=""))
        assert paper is not None
        assert paper.url == "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7369158"

    def test_untitled_record_is_rejected(self, connector) -> None:
        """A record with no title cannot become a Paper."""
        assert connector._parse_record(_record(0, title="")) is None

    def test_non_dict_record_is_rejected(self, connector) -> None:
        """A malformed entry is counted as unparsed, not crashed on."""
        assert connector._parse_record("not a record") is None

    def test_malformed_author_entries_are_skipped(self, connector) -> None:
        """A record whose author list is junk still becomes a paper."""
        paper = connector._parse_record(_record(0, authors=["not a dict", {}]))
        assert paper is not None
        assert paper.authors == []


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------


class TestQueryBuilder:
    """The builder refuses what SSRN's semantics were never observed to support."""

    @pytest.fixture
    def builder(self) -> SSRNQueryBuilder:
        """Return an SSRN query builder."""
        return SSRNQueryBuilder()

    def test_a_single_default_scoped_term_is_accepted(self, builder) -> None:
        """The one shape whose meaning does not depend on an operator."""
        assert builder.validate_query(_parse("[catalysis]")).is_valid
        assert builder.convert_query(_parse("[catalysis]")) == "catalysis"

    @pytest.mark.parametrize(
        "query_string",
        ["[a] AND [b]", "[a] OR [b]", "[a] AND NOT [b]"],
    )
    def test_boolean_queries_are_rejected(self, builder, query_string) -> None:
        """Sending them would return a plausible list for a different question."""
        result = builder.validate_query(_parse(query_string))
        assert not result.is_valid
        assert "401" in (result.error_message or "")

    @pytest.mark.parametrize("query_string", ["ti[a]", "au[a]", "abs[a]", "key[a]"])
    def test_field_scoped_terms_are_rejected(self, builder, query_string) -> None:
        """No field-scoped syntax could be observed, so none is invented."""
        result = builder.validate_query(_parse(query_string))
        assert not result.is_valid
        assert "scoped to" in (result.error_message or "")

    def test_an_empty_term_is_rejected(self, builder) -> None:
        """An empty query cannot be sent."""
        from lit_search.core.query import NodeType, Query, QueryNode

        term = QueryNode(node_type=NodeType.TERM, value="   ")
        query = Query(raw_query="unused", root=QueryNode(node_type=NodeType.ROOT, children=[term]))
        assert not builder.validate_query(query).is_valid

    def test_a_query_with_no_terms_is_rejected(self, builder) -> None:
        """Nothing to search for is not a search."""
        from lit_search.core.query import NodeType, Query, QueryNode

        query = Query(raw_query="unused", root=QueryNode(node_type=NodeType.ROOT, children=[]))
        assert not builder.validate_query(query).is_valid

    def test_a_rejected_query_issues_no_request(self, connector) -> None:
        """Retrying a query the source cannot express only spends budget."""
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(connector, "_get") as mocked,
        ):
            result = connector.search(_parse("[a] AND [b]"))
        assert result.status is CoverageStatus.QUERY_UNSUPPORTED
        assert result.papers == ()
        mocked.assert_not_called()
