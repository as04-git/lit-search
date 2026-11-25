"""Unit tests for OpenAIREConnector.

These run under ``pytest-socket`` with ``--disable-socket`` (see
``pyproject.toml``), so every HTTP call is mocked and no test can reach the
network.  All fixture payloads mirror shapes observed against the live Graph
API v3 on 2026-08-31.

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

from lit_search.connectors.openaire import OpenAIREBodyError, OpenAIREConnector
from lit_search.core.coverage import CoverageStatus, CoveredPapers
from lit_search.core.paper import Database, PaperType
from lit_search.core.query import FilterCode, NodeType, Query, QueryNode
from lit_search.core.source import SourceType
from lit_search.query.builders.openaire import OpenAIREQueryBuilder

from .conftest import _parse

# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def _record(index: int, **overrides: Any) -> dict[str, Any]:
    """Build one Graph API research product.

    Parameters
    ----------
    index : int
        Used to make the title and identifiers distinct.
    **overrides : Any
        Fields to replace or add.

    Returns
    -------
    dict[str, Any]
        A record shaped like a live ``results`` entry.
    """
    record: dict[str, Any] = {
        "id": f"doi_dedup___::{index:032x}",
        "mainTitle": f"Synthetic Product {index}.",
        "descriptions": [f"Synthetic abstract {index}."],
        "publicationDate": "2023-05-04",
        "publisher": "Elsevier",
        "type": "publication",
        "language": {"code": "eng", "label": "English"},
        "bestAccessRight": {"code": "c_abf2", "label": "OPEN"},
        "authors": [
            {"fullName": f"Author {index}", "name": "Author", "surname": str(index), "rank": 1}
        ],
        "pids": [{"scheme": "doi", "value": f"10.1000/test.{index}"}],
        "subjects": [{"subject": {"scheme": "FOS", "value": "01 natural sciences"}}],
        "container": {"name": "Nature", "issnPrinted": "0028-0836"},
        "instances": [
            {
                "type": "Article",
                "urls": [
                    f"https://example.org/article/{index}",
                    f"https://example.org/pdf/{index}.pdf",
                ],
            }
        ],
    }
    record.update(overrides)
    return record


def _page(
    records: list[dict[str, Any]],
    num_found: object,
    next_cursor: str | None = None,
) -> dict[str, Any]:
    """Build one search-response body.

    Parameters
    ----------
    records : list[dict[str, Any]]
        Records for this page.
    num_found : object
        Value of ``header.numFound``.  Typed as ``object`` rather than
        ``int`` because one test supplies a value the connector cannot read,
        which is the case that must never become a ``COMPLETE``.
    next_cursor : str | None
        ``header.nextCursor`` to include.  ``None`` omits the key, which is how
        the live API signals the end of a cursor run.

    Returns
    -------
    dict[str, Any]
        Response body.
    """
    header: dict[str, Any] = {"numFound": num_found, "maxScore": 6.58, "pageSize": len(records)}
    if next_cursor is not None:
        header["nextCursor"] = next_cursor
    return {"header": header, "results": records}


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
def connector() -> OpenAIREConnector:
    """Return an anonymous OpenAIRE connector."""
    return OpenAIREConnector()


def _search(connector: OpenAIREConnector, responses: list[Any], **kwargs: Any) -> CoveredPapers:
    """Run a search against a scripted sequence of ``_get`` outcomes.

    Parameters
    ----------
    connector : OpenAIREConnector
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
        assert connector.name == Database.OPENAIRE.value

    def test_anonymous_interval_is_the_documented_sixty_per_hour(self, connector) -> None:
        """OpenAIRE documents 60 requests per hour without a token."""
        assert connector.min_request_interval == 60.0

    def test_authenticated_interval_is_the_documented_seven_thousand_two_hundred(self) -> None:
        """A token raises the documented allowance 120-fold."""
        assert OpenAIREConnector(access_token="t").min_request_interval == 0.5

    def test_registered_in_the_search_registry(self) -> None:
        """A connector nobody can select is not ported."""
        from lit_search.connectors import SEARCH_REGISTRY

        assert SEARCH_REGISTRY[Database.OPENAIRE] is OpenAIREConnector

    def test_quota_is_declared_because_the_limit_is_hourly(self, connector) -> None:
        """Interval pacing alone cannot hold an hourly budget over a burst."""
        assert (connector.request_quota.limit, connector.request_quota.window_seconds) == (
            60,
            3600.0,
        )
        tokened = OpenAIREConnector(access_token="t").request_quota
        assert (tokened.limit, tokened.window_seconds) == (7200, 3600.0)

    def test_rate_limit_service_is_shared_across_openaire_endpoints(self, connector) -> None:
        """The hourly allowance belongs to the account, not to the endpoint."""
        assert connector.rate_limit_service == "openaire"

    def test_token_changes_the_rate_limit_bucket(self) -> None:
        """The token is the quota's identity, so it must reach the scope."""
        anonymous = OpenAIREConnector()
        tokened = OpenAIREConnector(access_token="secret-token")
        assert anonymous.rate_limit_scope.fingerprint == "anon"
        assert tokened.rate_limit_scope.fingerprint != "anon"
        assert tokened.rate_limit_scope.service == anonymous.rate_limit_scope.service

    def test_static_access_tokens_retain_distinct_quota_buckets(self) -> None:
        """Access-token-only callers remain scoped by their supplied token."""
        first = OpenAIREConnector(access_token="first-hour-token")
        same = OpenAIREConnector(access_token="first-hour-token")
        other = OpenAIREConnector(access_token="other-hour-token")

        assert first.rate_limit_scope == same.rate_limit_scope
        assert first.rate_limit_scope != other.rate_limit_scope


# ---------------------------------------------------------------------------
# The four required cases, plus the one that matters
# ---------------------------------------------------------------------------


class TestCoverageIsHonest:
    """Each way a search can end reports a different, actionable status."""

    def test_normal_result_is_complete(self, connector) -> None:
        """Retrieving as many records as numFound reports is COMPLETE."""
        result = _search(connector, [_response(_page([_record(0), _record(1)], num_found=2))])
        assert result.status is CoverageStatus.COMPLETE
        assert len(result.papers) == 2
        assert result.coverage.reported_total == 2

    def test_upstream_failure_is_failed(self, connector) -> None:
        """A dead first request is FAILED, and the error is carried."""
        result = _search(connector, [requests.ConnectionError("openaire unreachable")])
        assert result.status is CoverageStatus.FAILED
        assert result.papers == ()
        assert "unreachable" in (result.coverage.error or "")
        assert result.coverage.error_type == "ConnectionError"

    def test_empty_but_successful_search_is_complete(self, connector) -> None:
        """A genuine zero-hit query is COMPLETE with no papers.

        Verified live 2026-08-31: an unmatched string returns
        ``{"header": {"numFound": 0, ...}, "results": []}`` with HTTP 200.
        """
        result = _search(connector, [_response(_page([], num_found=0))])
        assert result.status is CoverageStatus.COMPLETE
        assert result.papers == ()
        assert result.coverage.reported_total == 0

    def test_capped_result_is_capped(self, connector) -> None:
        """Stopping at max_papers is CAPPED and names the cap."""
        result = _search(
            connector,
            [_response(_page([_record(i) for i in range(3)], num_found=900, next_cursor="c1"))],
            max_papers=3,
        )
        assert result.status is CoverageStatus.CAPPED
        assert len(result.papers) == 3
        assert result.coverage.cap == 3
        assert result.coverage.reported_total == 900

    def test_failure_and_emptiness_are_different_statuses(self, connector) -> None:
        """The regression test for the whole port.

        The donor reaches both of these through the same ``return papers``:
        ``_search_with_retry`` collects the exception into ``last_error``, the
        caller logs it as a warning and falls through to a legacy path whose
        own ``except Exception`` ends at ``return papers``.  Six routes, one
        indistinguishable value.  Here they must differ.
        """
        failure = _search(connector, [requests.ConnectionError("boom")])
        empty = _search(connector, [_response(_page([], num_found=0))])

        assert failure.papers == empty.papers == ()
        assert failure.status is not empty.status
        assert failure.status is CoverageStatus.FAILED
        assert empty.status is CoverageStatus.COMPLETE
        assert not failure.coverage.is_complete
        assert empty.coverage.is_complete

    def test_absence_of_a_token_is_not_a_missing_credential(self, connector) -> None:
        """OpenAIRE answers anonymously, so a token is an optimisation.

        Reporting SKIPPED_NO_CREDENTIAL here would drop a working database from
        every search while looking like careful bookkeeping -- the opposite
        error from the one this project was built to catch, and just as silent.
        """
        result = _search(connector, [_response(_page([_record(0)], num_found=1))])
        assert result.status is CoverageStatus.COMPLETE


# ---------------------------------------------------------------------------
# The 200-with-an-unusable-body family
# ---------------------------------------------------------------------------


class TestTwoHundredWithAnUnusableBody:
    """A 200 that is not a search result must not read as zero records."""

    def test_missing_envelope_is_failed_not_empty(self, connector) -> None:
        """Without header/results the body is not a search result at all."""
        result = _search(connector, [_response({"something": "else"})])
        assert result.status is CoverageStatus.FAILED
        assert "envelope" in (result.coverage.error or "")

    def test_error_message_body_is_failed(self, connector) -> None:
        """OpenAIRE's error bodies carry ``message``; it must not be dropped.

        Verified live: an unknown parameter returns ``{"message": "Unknown
        parameter: notafield; valid parameters are: [...]", "code": 400}``.
        That normally arrives with an HTTP 400 status, but a body carrying a
        message and no results is treated as an error whatever the status, so
        that a gateway rewriting the status cannot turn it into zero records.
        """
        result = _search(connector, [_response({"message": "Unknown parameter: notafield"})])
        assert result.status is CoverageStatus.FAILED
        assert "Unknown parameter" in (result.coverage.error or "")

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

        assert isinstance(OpenAIREBodyError("x"), ConnectorError)

    def test_an_unreadable_total_cannot_yield_complete(self, connector) -> None:
        """A total the connector cannot read must not become a COMPLETE."""
        result = _search(connector, [_response(_page([_record(0)], num_found="lots"))])
        assert result.status is CoverageStatus.PARTIAL
        assert result.coverage.reported_total is None


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class TestCursorPagination:
    """Every abnormal stop reaches the coverage status instead of the log."""

    def test_single_page_result_omits_next_cursor(self, connector) -> None:
        """A result set that fits one page has no nextCursor."""
        result = _search(connector, [_response(_page([_record(0)], num_found=1))])
        assert result.status is CoverageStatus.COMPLETE
        assert len(result.papers) == 1

    def test_multi_page_follows_the_cursor(self, connector) -> None:
        """Pages are consumed while a fresh cursor is offered."""
        result = _search(
            connector,
            [
                _response(_page([_record(0), _record(1)], num_found=3, next_cursor="c1")),
                _response(_page([_record(2)], num_found=3)),
            ],
        )
        assert result.status is CoverageStatus.COMPLETE
        assert len(result.papers) == 3

    def test_repeated_cursor_ends_the_run(self, connector) -> None:
        """A cursor that does not advance would otherwise loop forever."""
        result = _search(
            connector,
            [
                _response(_page([_record(0)], num_found=2, next_cursor="c1")),
                _response(_page([_record(1)], num_found=2, next_cursor="c1")),
            ],
        )
        assert len(result.papers) == 2
        assert result.status is CoverageStatus.COMPLETE

    def test_short_page_alone_does_not_end_pagination(self, connector) -> None:
        """A page smaller than the page size keeps going while a cursor is offered."""
        result = _search(
            connector,
            [
                _response(_page([_record(0)], num_found=2, next_cursor="c1")),
                _response(_page([_record(1)], num_found=2, next_cursor="c2")),
                _response(_page([], num_found=2, next_cursor="c2")),
            ],
        )
        assert len(result.papers) == 2
        assert result.status is CoverageStatus.COMPLETE

    def test_cursor_exhausted_before_num_found_is_partial(self, connector) -> None:
        """Fewer records than numFound with the cursor spent is PARTIAL."""
        result = _search(connector, [_response(_page([_record(0)], num_found=50))])
        assert result.status is CoverageStatus.PARTIAL
        assert result.coverage.reported_total == 50
        assert "50" in (result.coverage.detail or "")

    def test_empty_page_with_a_fresh_cursor_is_partial(self, connector) -> None:
        """The API offered more and delivered nothing: stop and say so."""
        result = _search(
            connector,
            [
                _response(_page([_record(0)], num_found=9, next_cursor="c1")),
                _response(_page([], num_found=9, next_cursor="c2")),
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert "empty page" in (result.coverage.detail or "")

    def test_failure_after_some_pages_is_partial_not_failed(self, connector) -> None:
        """Records that did arrive are real, so the run is PARTIAL."""
        result = _search(
            connector,
            [
                _response(_page([_record(0), _record(1)], num_found=90, next_cursor="c1")),
                requests.ConnectionError("dropped mid-run"),
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 2
        assert result.coverage.error_type == "ConnectionError"
        assert "dropped mid-run" in (result.coverage.detail or "")

    def test_body_error_after_some_pages_is_partial(self, connector) -> None:
        """An unusable body partway through degrades rather than discards."""
        result = _search(
            connector,
            [
                _response(_page([_record(0)], num_found=90, next_cursor="c1")),
                _response({"message": "internal error"}),
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 1

    def test_num_found_drift_between_pages_is_partial(self, connector) -> None:
        """A moving index means neither total can be checked."""
        result = _search(
            connector,
            [
                _response(_page([_record(0), _record(1)], num_found=3, next_cursor="c1")),
                _response(_page([_record(2)], num_found=7)),
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert "changed mid-retrieval" in (result.coverage.detail or "")

    def test_unparsable_records_are_counted_as_a_shortfall(self, connector) -> None:
        """A record seen but not parsed is a coverage loss, not an absence."""
        result = _search(
            connector,
            [_response(_page([_record(0), _record(1, mainTitle="  ")], num_found=2))],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 1
        assert "could not be parsed" in (result.coverage.detail or "")

    def test_coverage_count_always_matches_the_papers_returned(self, connector) -> None:
        """CoveredPapers rejects a mismatch, so this asserts the accounting."""
        result = _search(
            connector,
            [_response(_page([_record(i) for i in range(5)], num_found=99, next_cursor="c1"))],
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
        query = kwargs.pop("query", None) or _parse("[machine learning]")
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(
                connector, "_get", return_value=_response(_page([], num_found=0))
            ) as mocked,
        ):
            connector.search(query, **kwargs)
        params: dict[str, Any] = mocked.call_args.args[1]
        return params

    def test_cursor_pagination_is_used_not_offset(self, connector) -> None:
        """Offset paging is refused past 10,000 records with HTTP 400.

        Using it would make the ten-thousandth record a silent ceiling on
        recall for any broad query, which is precisely the shape of failure
        this fork exists to prevent.
        """
        params = self._capture(connector)
        assert params["cursor"] == "*"
        assert "page" not in params

    def test_page_size_never_exceeds_one_hundred(self, connector) -> None:
        """A page size beyond the API's own maximum is refused upstream."""
        assert 1 <= self._capture(connector, max_papers=100_000)["pageSize"] <= 100

    def test_page_size_is_at_least_one_when_the_cap_is_one(self, connector) -> None:
        """A cap of 1 must not produce pageSize=0."""
        assert self._capture(connector, max_papers=1)["pageSize"] >= 1

    def test_default_query_lands_on_the_search_parameter(self, connector) -> None:
        """An unfiltered query means title and abstract, which is ``search``."""
        params = self._capture(connector)
        assert params["search"] == '"machine learning"'

    def test_title_query_lands_on_main_title(self, connector) -> None:
        """An explicit title filter must not silently widen to ``search``."""
        params = self._capture(connector, query=_parse("ti[catalysis]"))
        assert params["mainTitle"] == '"catalysis"'
        assert "search" not in params

    def test_date_bounds_become_publication_date_parameters(self, connector) -> None:
        """Both bounds render as the API's own date filters."""
        params = self._capture(
            connector,
            since=datetime.date(2020, 1, 1),
            until=datetime.date(2021, 12, 31),
        )
        assert params["fromPublicationDate"] == "2020-01-01"
        assert params["toPublicationDate"] == "2021-12-31"

    def test_no_date_filter_when_no_bounds_are_given(self, connector) -> None:
        """An unbounded search must not silently acquire a date restriction."""
        params = self._capture(connector)
        assert "fromPublicationDate" not in params
        assert "toPublicationDate" not in params

    def test_the_token_is_sent_as_a_bearer_header(self) -> None:
        """A configured token must actually reach the request."""
        headers = OpenAIREConnector(access_token="secret-token")._prepare_headers({})
        assert headers["Authorization"] == "Bearer secret-token"

    def test_no_authorization_header_without_a_token(self, connector) -> None:
        """An anonymous client must not send an empty credential."""
        assert "Authorization" not in connector._prepare_headers({})


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    """Field mapping from a Graph API research product to a Paper."""

    def test_core_fields(self, connector) -> None:
        """Title, abstract, DOI, date, URLs and language are mapped."""
        paper = connector._parse_item(_record(7))
        assert paper is not None
        assert paper.title == "Synthetic Product 7."
        assert paper.abstract == "Synthetic abstract 7."
        assert paper.doi == "10.1000/test.7"
        assert paper.publication_date == datetime.date(2023, 5, 4)
        assert paper.url == "https://example.org/article/7"
        assert paper.pdf_url == "https://example.org/pdf/7.pdf"
        assert paper.language == "en"
        assert paper.is_open_access is True
        assert Database.OPENAIRE in paper.found_in

    def test_container_becomes_a_journal_source(self, connector) -> None:
        """The container names the venue; the publisher does not."""
        paper = connector._parse_item(_record(1))
        assert paper is not None
        assert paper.source is not None
        assert paper.source.title == "Nature"
        assert paper.source.source_type is SourceType.JOURNAL

    def test_publisher_is_the_fallback_venue(self, connector) -> None:
        """Without a container the publisher is used, and labelled as such."""
        paper = connector._parse_item(_record(2, container=None))
        assert paper is not None
        assert paper.source is not None
        assert paper.source.title == "Elsevier"

    def test_instance_type_maps_to_a_paper_type(self, connector) -> None:
        """An 'Article' instance is a journal article."""
        paper = connector._parse_item(_record(3))
        assert paper is not None
        assert paper.paper_type is PaperType.ARTICLE

    def test_preprint_instance_becomes_unpublished(self, connector) -> None:
        """A preprint is not a published article."""
        record = _record(4, instances=[{"type": "preprint", "urls": []}])
        paper = connector._parse_item(record)
        assert paper is not None
        assert paper.paper_type is PaperType.UNPUBLISHED

    def test_split_author_names_are_joined(self, connector) -> None:
        """Some records carry name/surname without a fullName."""
        record = _record(5, authors=[{"name": "Ada", "surname": "Lovelace"}])
        paper = connector._parse_item(record)
        assert paper is not None
        assert paper.authors[0].name == "Ada Lovelace"

    def test_instance_level_doi_is_found(self, connector) -> None:
        """A DOI can live on an instance rather than on the product."""
        record = _record(
            6,
            pids=[],
            instances=[
                {"type": "Article", "urls": [], "pids": [{"scheme": "doi", "value": "10.1/x"}]}
            ],
        )
        paper = connector._parse_item(record)
        assert paper is not None
        assert paper.doi == "10.1/x"

    def test_subjects_become_keywords(self, connector) -> None:
        """Subjects arrive nested under a ``subject`` object."""
        paper = connector._parse_item(_record(8))
        assert paper is not None
        assert paper.keywords == {"01 natural sciences"}

    def test_closed_access_is_not_asserted_as_closed(self, connector) -> None:
        """Only the positive open-access case is claimed."""
        record = _record(9, bestAccessRight={"label": "CLOSED"})
        paper = connector._parse_item(record)
        assert paper is not None
        assert paper.is_open_access is None

    def test_untitled_record_is_rejected(self, connector) -> None:
        """A record with no mainTitle cannot become a Paper."""
        assert connector._parse_item(_record(0, mainTitle="")) is None

    def test_non_dict_record_is_rejected(self, connector) -> None:
        """A malformed entry is counted as unparsed, not crashed on."""
        assert connector._parse_item("not a record") is None

    def test_url_falls_back_to_the_doi(self, connector) -> None:
        """A record with no instance URL still gets a resolvable link."""
        paper = connector._parse_item(_record(3, instances=[]))
        assert paper is not None
        assert paper.url == "https://doi.org/10.1000/test.3"


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------


class TestQueryBuilder:
    """Only verified parameters are emitted, and unsafe terms are refused."""

    @pytest.fixture
    def builder(self) -> OpenAIREQueryBuilder:
        """Return an OpenAIRE query builder."""
        return OpenAIREQueryBuilder()

    def test_default_filter_is_title_abstract_not_keywords(self, builder) -> None:
        """``search`` covers title and abstract, and nothing covers all three.

        Declaring tiabskey unsupported is what makes the default fall back to
        a parameter that does what it says, rather than one quietly omitting
        the keyword field it claims to search.
        """
        assert builder.supports_filter(FilterCode.TITLE_ABSTRACT_KEYWORDS) is False
        assert builder.convert_query(_parse("[machine learning]")) == {
            "search": '"machine learning"'
        }

    @pytest.mark.parametrize(
        ("query_string", "expected"),
        [
            ("ti[catalysis]", {"mainTitle": '"catalysis"'}),
            ("abs[catalysis]", {"description": '"catalysis"'}),
            ("key[catalysis]", {"subjects": '"catalysis"'}),
            ("au[Bengio]", {"authorFullName": '"Bengio"'}),
            ("tiabs[catalysis]", {"search": '"catalysis"'}),
        ],
    )
    def test_each_filter_maps_to_a_verified_parameter(
        self, builder, query_string, expected
    ) -> None:
        """Every parameter returned a plausible non-zero count live."""
        assert builder.convert_query(_parse(query_string)) == expected

    def test_boolean_structure_is_kept_inside_one_parameter(self, builder) -> None:
        """OpenAIRE runs AND/OR/NOT inside a single field parameter."""
        converted = builder.convert_query(_parse("ti[a catalysis] OR ti[b photolysis]"))
        assert converted == {"mainTitle": '"a catalysis" OR "b photolysis"'}

    def test_every_term_is_quoted(self, builder) -> None:
        """An unquoted multi-word term is AND-ed, not phrase-matched.

        Live: ``mainTitle=machine learning`` and ``mainTitle=machine AND
        learning`` both returned 528,357, against 509,041 quoted.
        """
        assert builder.convert_query(_parse("ti[machine learning]")) == {
            "mainTitle": '"machine learning"'
        }

    def test_mixed_field_query_is_rejected(self, builder) -> None:
        """One global logicalOperator cannot carry a mixed operator tree.

        Sending an approximation would return a plausible set for a different
        question, which is exactly the failure this fork exists to prevent.
        """
        result = builder.validate_query(_parse("ti[catalysis] AND abs[enzyme]"))
        assert result.is_valid is False
        assert "logicalOperator" in (result.error_message or "")

    def test_wildcard_terms_are_rejected(self, builder) -> None:
        """OpenAIRE narrows rather than expands: 103 hits against 86,158."""
        result = builder.validate_query(_parse("ti[catalys*]"))
        assert result.is_valid is False
        assert "wildcard" in (result.error_message or "")

    def test_unsupported_filter_is_rejected(self, builder) -> None:
        """The Graph API exposes no venue title and no affiliation."""
        assert builder.validate_query(_parse("src[Nature]")).is_valid is False
        assert builder.validate_query(_parse("aff[Toronto]")).is_valid is False

    def test_empty_term_is_rejected(self, builder) -> None:
        """An empty query returned 386,695,835 records live.

        The parser already forbids an empty term, so this query is built
        directly.  The check is a second line of defence for any future path
        that constructs a ``Query`` without the parser.
        """
        blank = Query(
            raw_query="ti[]",
            root=QueryNode(
                node_type=NodeType.ROOT,
                children=[
                    QueryNode(node_type=NodeType.TERM, value="  ", filter_code=FilterCode.TITLE)
                ],
            ),
        )
        result = builder.validate_query(blank)
        assert result.is_valid is False
        assert "match-all" in (result.error_message or "")

    def test_rejected_query_reaches_the_ledger_as_query_unsupported(self, connector) -> None:
        """A refused query must not spend a request, and must say why.

        This matters more here than elsewhere: an anonymous OpenAIRE client has
        sixty requests an hour, so a wasted one is a real cost.
        """
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(connector, "_get") as mocked,
        ):
            result = connector.search(_parse("ti[catalysis] AND abs[enzyme]"))
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
    clause runs.
    """

    def test_a_swallowed_page_failure_is_downgraded_to_partial(self, connector) -> None:
        """The audit contradicts a COMPLETE the connector still reported."""

        class _Liar(OpenAIREConnector):
            """A connector with the donor's swallow and a COMPLETE claim."""

            def _build_coverage(self, run, *, retrieved, max_papers):
                """Claim COMPLETE unconditionally."""
                from lit_search.core.coverage import SourceCoverage

                return SourceCoverage.complete(self.name, retrieved, reported_total=retrieved)

            @staticmethod
            def _record_page_failure(run, exc, cursor):
                """Swallow the failure, as ``_search_with_retry`` does."""
                return None

        liar = _Liar()
        outcomes: list[Any] = [
            _response(_page([_record(0)], num_found=90, next_cursor="c1")),
            requests.ConnectionError("dropped mid-run"),
        ]
        with (
            patch.object(liar, "_rate_limit"),
            patch.object(liar, "_request_with_retry_uncounted", side_effect=outcomes),
        ):
            result = liar.search(_parse("[machine learning]"))

        assert result.status is CoverageStatus.PARTIAL
        assert "ConnectionError" in (result.coverage.detail or "")
