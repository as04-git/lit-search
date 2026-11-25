"""Unit tests for DBLPConnector.

These run under ``pytest-socket`` with ``--disable-socket`` (see
``pyproject.toml``), so every HTTP call is mocked and no test can reach the
network.  Fixture payloads mirror shapes observed against the live API on
2026-08-31, including its habit of returning every numeric attribute as a
string.

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

from lit_search.connectors.dblp import DBLPBodyError, DBLPConnector
from lit_search.core.coverage import CoverageStatus, CoveredPapers, SourceCoverage
from lit_search.core.paper import Database, PaperType
from lit_search.core.source import SourceType
from lit_search.query.builders.dblp import DBLPQueryBuilder

from .conftest import _parse

# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def _hit(index: int, **overrides: Any) -> dict[str, Any]:
    """Build one dblp ``hits.hit[]`` entry.

    Parameters
    ----------
    index : int
        Used to make titles and identifiers distinct.
    **overrides : Any
        Fields to replace or add inside ``info``.

    Returns
    -------
    dict[str, Any]
        Entry shaped like a live hit.
    """
    info: dict[str, Any] = {
        "authors": {"author": [{"@pid": f"1/{index}", "text": f"Author {index}"}]},
        "title": f"Synthetic Paper {index}.",
        "venue": "CRYPTO",
        "year": "2023",
        "type": "Conference and Workshop Papers",
        "key": f"conf/crypto/Author{index}",
        "doi": f"10.1000/test.{index}",
        "ee": f"https://doi.org/10.1000/test.{index}",
        "url": f"https://dblp.org/rec/conf/crypto/Author{index}",
    }
    info.update(overrides)
    return {"@score": "1", "@id": str(index), "info": info}


def _page(
    hits: list[dict[str, Any]],
    total: int,
    first: int = 0,
) -> dict[str, Any]:
    """Build one dblp search-response body.

    Every numeric attribute is rendered as a string, which is what the live API
    does and what a port that assumes ``int`` gets wrong.

    Parameters
    ----------
    hits : list[dict[str, Any]]
        Entries for this page.
    total : int
        Value of ``hits.@total``.
    first : int
        Value of ``hits.@first``.

    Returns
    -------
    dict[str, Any]
        Response body.
    """
    block: dict[str, Any] = {
        "@total": str(total),
        "@computed": str(first + len(hits)),
        "@sent": str(len(hits)),
        "@first": str(first),
    }
    if hits:
        block["hit"] = hits
    return {
        "result": {
            "query": "Synthetic*",
            "status": {"@code": "200", "text": "OK"},
            "time": {"@unit": "msecs", "text": "2.01"},
            "hits": block,
        }
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


def _html_response(status_code: int = 500) -> MagicMock:
    """Return a mock response whose body is dblp's HTML error page.

    dblp serves HTML for overload (429/500/503) and for out-of-range
    parameters, so ``.json()`` raises :class:`ValueError`.

    Parameters
    ----------
    status_code : int
        HTTP status to report.

    Returns
    -------
    MagicMock
        Mock response.
    """
    mock = MagicMock()
    mock.status_code = status_code
    mock.text = "<!DOCTYPE html><html><title>dblp: error 500</title></html>"
    mock.json.side_effect = ValueError("Expecting value: line 1 column 1 (char 0)")
    return mock


@pytest.fixture
def connector() -> DBLPConnector:
    """Return a dblp connector."""
    return DBLPConnector()


def _search(connector: DBLPConnector, responses: list[Any], **kwargs: Any) -> CoveredPapers:
    """Run a search against a scripted sequence of ``_get`` outcomes.

    Parameters
    ----------
    connector : DBLPConnector
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
    query = kwargs.pop("query", None) or _parse("ti[machine learning]")
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
        assert connector.name == Database.DBLP.value

    def test_min_request_interval_is_dblps_published_crawl_delay(self, connector) -> None:
        """4.0s is ``Crawl-delay: 4`` from dblp's own robots.txt.

        Deliberately **not** the 2.0s in the project's rate-limit table: that
        row's stated evidence ("429 carries Retry-After") was falsified live on
        2026-08-31, so it is not a safe source for the number.
        """
        assert connector.min_request_interval == 4.0

    def test_registered_in_the_search_registry(self) -> None:
        """A connector nobody can select is not ported."""
        from lit_search.connectors import SEARCH_REGISTRY

        assert SEARCH_REGISTRY[Database.DBLP] is DBLPConnector

    def test_no_credential_is_declared_explicitly(self, connector) -> None:
        """dblp issues no key, so the anonymous host bucket is correct."""
        assert connector._credential_attrs == ()
        assert connector.rate_limit_scope.service == "dblp"
        assert connector.rate_limit_scope.fingerprint == "anon"

    def test_rate_limit_service_is_shared_across_dblp_endpoints(self, connector) -> None:
        """The scope names the quota authority, not this one connector."""
        assert connector.rate_limit_service == "dblp"

    def test_http_500_is_retryable_for_this_connector(self, connector) -> None:
        """dblp signals throttling with 500 as well as 429 and 503.

        Treating its 500 as a permanent fault would report dblp's own pacing
        penalty as a database outage.
        """
        assert 500 in connector._retryable_status_codes
        assert {429, 503}.issubset(connector._retryable_status_codes)


# ---------------------------------------------------------------------------
# The four required cases, plus the one that matters
# ---------------------------------------------------------------------------


class TestCoverageIsHonest:
    """Each way a search can end reports a different, actionable status."""

    def test_normal_result_is_complete(self, connector) -> None:
        """Retrieving as many records as ``@total`` reports is COMPLETE."""
        result = _search(connector, [_response(_page([_hit(0), _hit(1)], total=2))])
        assert result.status is CoverageStatus.COMPLETE
        assert len(result.papers) == 2
        assert result.coverage.retrieved == 2
        assert result.coverage.reported_total == 2

    def test_upstream_failure_is_failed(self, connector) -> None:
        """A dead first request is FAILED, and the error is carried."""
        result = _search(connector, [requests.ConnectionError("dblp unreachable")])
        assert result.status is CoverageStatus.FAILED
        assert result.papers == ()
        assert "unreachable" in (result.coverage.error or "")
        assert result.coverage.error_type == "ConnectionError"

    def test_empty_but_successful_search_is_complete(self, connector) -> None:
        """A genuine zero-hit query is COMPLETE with no papers.

        Verified live 2026-08-31: a nonsense query returns HTTP 200 with
        ``hits.@total == "0"`` and **no** ``hit`` key.  dblp answered, said zero
        records match, and zero came back.
        """
        result = _search(connector, [_response(_page([], total=0))])
        assert result.status is CoverageStatus.COMPLETE
        assert result.papers == ()
        assert result.coverage.reported_total == 0

    def test_capped_result_is_capped(self, connector) -> None:
        """Stopping at max_papers is CAPPED and names the cap."""
        result = _search(
            connector,
            [_response(_page([_hit(i) for i in range(3)], total=900))],
            max_papers=3,
        )
        assert result.status is CoverageStatus.CAPPED
        assert len(result.papers) == 3
        assert result.coverage.cap == 3
        assert result.coverage.reported_total == 900

    def test_failure_and_emptiness_are_different_statuses(self, connector) -> None:
        """The regression test for the whole port.

        In the donor implementation an outage falls through to
        ``_search_html_fallback``, whose own ``except Exception`` returns the
        empty list it started with -- the same value a genuinely empty query
        produces.  Here they must differ, and the failure must not be readable
        as a complete answer.
        """
        failure = _search(connector, [requests.ConnectionError("boom")])
        empty = _search(connector, [_response(_page([], total=0))])

        assert failure.papers == empty.papers == ()
        assert failure.status is not empty.status
        assert failure.status is CoverageStatus.FAILED
        assert empty.status is CoverageStatus.COMPLETE
        assert not failure.coverage.is_complete
        assert empty.coverage.is_complete

    def test_html_error_page_is_failed_not_empty(self, connector) -> None:
        """dblp's overload page is HTML, so ``.json()`` raises.

        A port that caught that and returned its accumulator would report a
        throttled request as a database holding nothing.
        """
        result = _search(connector, [_html_response()])
        assert result.status is CoverageStatus.FAILED
        assert result.papers == ()
        assert "non-JSON" in (result.coverage.error or "")

    def test_missing_hits_object_is_failed_not_empty(self, connector) -> None:
        """A body with no ``hits`` is malformed, not a zero-hit answer.

        An empty result carries ``hits`` with ``@total: "0"``, so the field's
        absence cannot be read as emptiness.
        """
        body = {"result": {"query": "x*", "status": {"@code": "200", "text": "OK"}}}
        result = _search(connector, [_response(body)])
        assert result.status is CoverageStatus.FAILED
        assert "hits" in (result.coverage.error or "")

    def test_non_200_status_in_the_body_is_failed(self, connector) -> None:
        """dblp reports a request-level complaint inside a 200 body."""
        body = {
            "result": {
                "query": "x*",
                "status": {"@code": "500", "text": "internal error"},
                "hits": {"@total": "0", "@sent": "0"},
            }
        }
        result = _search(connector, [_response(body)])
        assert result.status is CoverageStatus.FAILED
        assert "500" in (result.coverage.error or "")


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class TestPagination:
    """Retrieval stops for reasons it can name."""

    def test_pages_until_the_reported_total_is_reached(self, connector) -> None:
        """Two full pages reaching ``@total`` is COMPLETE."""
        page_one = _page([_hit(i) for i in range(100)], total=150)
        page_two = _page([_hit(100 + i) for i in range(50)], total=150, first=100)
        result = _search(connector, [_response(page_one), _response(page_two)])
        assert result.status is CoverageStatus.COMPLETE
        assert len(result.papers) == 150

    def test_a_short_page_is_not_treated_as_the_end(self, connector) -> None:
        """dblp caps ``@sent`` at 100 however large ``h`` is.

        The donor comments that "dblp supports up to 1000 results" and requests
        ``h=1000``; a page of 100 against that request looks short, and a
        ``len(page) < page_size: break`` would stop at 100 of 150 while
        reporting success.
        """
        page_one = _page([_hit(i) for i in range(100)], total=150)
        page_two = _page([_hit(100 + i) for i in range(50)], total=150, first=100)
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(connector, "_get", side_effect=[_response(page_one), _response(page_two)]),
        ):
            result = connector.search(_parse("ti[machine learning]"))
        assert len(result.papers) == 150
        assert result.status is CoverageStatus.COMPLETE

    def test_mid_pagination_failure_keeps_records_and_reports_partial(self, connector) -> None:
        """A failed second page is PARTIAL, with the first page's records kept.

        This is also how dblp's ``f`` ceiling arrives: ``f=9900`` returns HTTP
        500 while an ``f=0`` control succeeds.
        """
        page_one = _page([_hit(i) for i in range(100)], total=50_000)
        result = _search(
            connector,
            [_response(page_one), requests.HTTPError("500 Server Error")],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 100
        assert result.coverage.error_type == "HTTPError"
        assert "result-window ceiling" in (result.coverage.detail or "")

    def test_an_early_empty_page_is_partial_not_complete(self, connector) -> None:
        """dblp stopping short of its own ``@total`` is a shortfall."""
        page_one = _page([_hit(i) for i in range(100)], total=500)
        page_two = _page([], total=500, first=100)
        result = _search(connector, [_response(page_one), _response(page_two)])
        assert result.status is CoverageStatus.PARTIAL
        assert "100 of 500" in (result.coverage.detail or "")

    def test_total_drift_forces_partial(self, connector) -> None:
        """A ``@total`` that moves mid-run makes completeness uncheckable."""
        page_one = _page([_hit(i) for i in range(100)], total=150)
        page_two = _page([_hit(100 + i) for i in range(50)], total=140, first=100)
        result = _search(connector, [_response(page_one), _response(page_two)])
        assert result.status is CoverageStatus.PARTIAL
        assert "changed mid-retrieval" in (result.coverage.detail or "")

    def test_unparsed_records_are_counted_not_dropped(self, connector) -> None:
        """A record dblp returned and the connector could not read is a loss.

        The donor ``continue``s past it with only a log line, so a schema
        change degrades recall invisibly.
        """
        good = _hit(0)
        broken = {"@id": "1", "info": {"authors": {}, "year": "2023"}}  # no title
        result = _search(connector, [_response(_page([good, broken], total=2))])
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 1
        assert "1 of 2 records could not be parsed" in (result.coverage.detail or "")


# ---------------------------------------------------------------------------
# Query handling
# ---------------------------------------------------------------------------


class TestQueryHandling:
    """The queries dblp would silently rewrite never leave this process."""

    def test_and_not_reaches_the_ledger_as_query_unsupported(self, connector) -> None:
        """dblp turns ``NOT`` into ``AND``, returning the inverse of the intent.

        Verified live 2026-08-31: ``blockchain -consensus`` and
        ``blockchain consensus`` produce an identical parse echo and an
        identical total of 1,173.
        """
        get = MagicMock()
        with patch.object(connector, "_rate_limit"), patch.object(connector, "_get", get):
            result = connector.search(_parse("ti[machine learning] AND NOT ti[reinforcement]"))
        assert result.status is CoverageStatus.QUERY_UNSUPPORTED
        assert result.papers == ()
        assert "AND NOT" in (result.coverage.detail or "")

    def test_a_rejected_query_issues_no_request(self, connector) -> None:
        """Refusal happens before the rate-limit budget is spent."""
        get = MagicMock()
        with patch.object(connector, "_rate_limit"), patch.object(connector, "_get", get):
            connector.search(_parse("ti[machine learning] AND NOT ti[reinforcement]"))
        get.assert_not_called()

    def test_an_abstract_scoped_query_is_partial_not_complete(self, connector) -> None:
        """dblp holds no abstracts, so a ``tiabs`` query is under-answered.

        Retrieval can be arithmetically perfect and the answer still be a
        subset of what was asked, which is a coverage fact rather than a
        pagination one.
        """
        result = _search(
            connector,
            [_response(_page([_hit(0), _hit(1)], total=2))],
            query=_parse("tiabs[machine learning]"),
        )
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 2
        assert "no abstracts" in (result.coverage.detail or "")

    def test_a_title_scoped_query_can_be_complete(self, connector) -> None:
        """The degradation note is specific, not a blanket downgrade.

        A ledger that says the same thing whatever happened is a ledger nobody
        reads, so this pins that ``PARTIAL`` above is caused by the filter and
        not by the connector giving up on ``COMPLETE`` altogether.
        """
        result = _search(
            connector,
            [_response(_page([_hit(0)], total=1))],
            query=_parse("ti[machine learning]"),
        )
        assert result.status is CoverageStatus.COMPLETE

    def test_local_date_filtering_is_reported(self, connector) -> None:
        """dblp's ``year:`` takes one year, so a range is applied locally."""
        hits = [_hit(0, year="2023"), _hit(1, year="2010")]
        result = _search(
            connector,
            [_response(_page(hits, total=2))],
            since=datetime.date(2020, 1, 1),
        )
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 1
        assert "removed by the requested date range" in (result.coverage.detail or "")

    def test_a_paper_with_no_year_survives_date_filtering(self, connector) -> None:
        """Absence of a date is not evidence of falling outside the range."""
        hits = [_hit(0, year=""), _hit(1, year="2023")]
        result = _search(
            connector,
            [_response(_page(hits, total=2))],
            since=datetime.date(2020, 1, 1),
        )
        assert len(result.papers) == 2


# ---------------------------------------------------------------------------
# The base class checks the claim
# ---------------------------------------------------------------------------


class TestTheBaseClassAuditsThisConnector:
    """Evidence the connector did not author still reaches the audit."""

    def test_all_http_goes_through_the_instrumented_helper(self, connector) -> None:
        """Every request this connector makes passes the evidence channel.

        The patch is applied at ``_request_with_retry_uncounted``, the seam
        *inside* ``_request_with_retry`` -- so ``record_request_attempt`` and
        ``record_request_failure`` still run for real.  A connector that
        reached the network with a bare ``requests.get`` would not be
        intercepted here at all, ``pytest-socket`` would refuse the connection,
        and this test would error rather than pass.

        That is the property the porting guide asks for: HTTP goes through
        ``_get``/``_post`` and therefore through the audit.
        """
        page = _page([_hit(0), _hit(1)], total=2)
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(
                connector, "_request_with_retry_uncounted", return_value=_response(page)
            ) as transport,
        ):
            result = connector.search(_parse("ti[machine learning]"))

        assert transport.called
        assert result.status is CoverageStatus.COMPLETE
        assert len(result.papers) == 2

    def test_a_swallowed_page_failure_cannot_claim_complete(self, connector) -> None:
        """A forged ``COMPLETE`` over a truncated run is refused.

        The connector is made to lie: a second page fails and
        ``_build_coverage`` is replaced so the run claims ``COMPLETE`` anyway.
        The base class refuses it.

        Which guard fires is worth recording, because it is not the one this
        test was first written to exercise.  ``SourceCoverage.complete()`` now
        requires ``reported_total`` and requires ``retrieved`` to have reached
        it, so a forgery over a *truncated* run cannot even be constructed with
        the real total -- 100 retrieved against 50,000 reported raises
        ``ValueError`` in the dataclass.  Lowering the forged total to 100 to
        get past that arithmetic then contradicts the totals the audit actually
        observed coming back from dblp, and
        :func:`~lit_search.core.retrieval_audit.verify_coverage_claim` refuses
        it on that ground instead.

        Both guards fire here, and the request-failure one is the load-bearing
        half: it can only have seen that failure because
        ``_request_with_retry`` recorded it *before* the connector's own
        ``except`` clause ran.  Patching ``_get`` instead of the transport seam
        below would silently skip that recording and leave only the weaker
        total check -- which is exactly how this test first passed for the
        wrong reason.
        """
        page_one = _page([_hit(i) for i in range(100)], total=50_000)
        forged = SourceCoverage.complete("dblp", 100, reported_total=100)

        with (
            patch.object(connector, "_rate_limit"),
            patch.object(connector, "_build_coverage", return_value=forged),
            patch.object(
                connector,
                "_request_with_retry_uncounted",
                side_effect=[_response(page_one), requests.ConnectionError("second page died")],
            ),
        ):
            result = connector.search(_parse("ti[machine learning]"))

        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 100
        detail = result.coverage.detail or ""
        assert "request(s) failed during this retrieval" in detail
        assert "ConnectionError: second page died" in detail
        assert "no such total was ever observed" in detail

    def test_the_forged_complete_would_otherwise_have_stood(self, connector) -> None:
        """Control for the test above: the forgery is only caught by evidence.

        With nothing for the audit to contradict, the same forged ``COMPLETE``
        survives.  Without this control the test above would pass just as well
        against a connector that never claims ``COMPLETE`` at all, and would
        therefore be testing nothing.
        """
        forged = SourceCoverage.complete("dblp", 1, reported_total=1)
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(connector, "_build_coverage", return_value=forged),
            patch.object(
                connector,
                "_request_with_retry_uncounted",
                side_effect=[_response(_page([_hit(0)], total=1))],
            ),
        ):
            result = connector.search(_parse("ti[machine learning]"))
        assert result.status is CoverageStatus.COMPLETE


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    """Field mapping, including the shapes dblp collapses."""

    def test_fields_are_mapped(self, connector) -> None:
        """A complete hit becomes a fully populated paper."""
        result = _search(connector, [_response(_page([_hit(7)], total=1))])
        paper = result.papers[0]
        assert paper.title == "Synthetic Paper 7."
        assert [a.name for a in paper.authors] == ["Author 7"]
        assert paper.doi == "10.1000/test.7"
        assert paper.publication_date == datetime.date(2023, 1, 1)
        assert paper.paper_type is PaperType.INPROCEEDINGS
        assert paper.source is not None
        assert paper.source.title == "CRYPTO"
        assert paper.source.source_type is SourceType.CONFERENCE
        assert paper.found_in == {"dblp"}

    def test_abstract_is_empty_because_dblp_holds_none(self, connector) -> None:
        """The empty abstract is the truth about dblp, not a parse failure."""
        result = _search(connector, [_response(_page([_hit(0)], total=1))])
        assert result.papers[0].abstract == ""

    def test_a_single_author_arrives_as_a_bare_object(self, connector) -> None:
        """dblp collapses one-element arrays to the object itself."""
        hit = _hit(0, authors={"author": {"@pid": "1/0", "text": "Solo Author"}})
        result = _search(connector, [_response(_page([hit], total=1))])
        assert [a.name for a in result.papers[0].authors] == ["Solo Author"]

    def test_a_non_doi_electronic_edition_yields_no_doi(self, connector) -> None:
        """A publisher link that is not a DOI must not be stored as one."""
        hit = _hit(0, ee="https://example.org/paper.pdf")
        hit["info"].pop("doi")
        result = _search(connector, [_response(_page([hit], total=1))])
        assert result.papers[0].doi is None


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------


class TestQueryBuilder:
    """Only verified operators and field prefixes are emitted."""

    @pytest.fixture
    def builder(self) -> DBLPQueryBuilder:
        """Return a dblp query builder."""
        return DBLPQueryBuilder()

    def test_and_is_whitespace(self, builder) -> None:
        """dblp conjoins space-separated terms."""
        assert builder.convert_query(_parse("ti[alpha] AND ti[beta]")) == "title:alpha title:beta"

    def test_or_is_a_pipe(self, builder) -> None:
        """Verified live: ``blockchain|consensus`` returns the union."""
        assert builder.convert_query(_parse("ti[alpha] OR ti[beta]")) == "title:alpha | title:beta"

    def test_field_prefixes_are_the_verified_ones(self, builder) -> None:
        """Each prefix was confirmed by its ``:filter:`` parse echo."""
        assert builder.convert_query(_parse("au[knuth]")) == "author:knuth"
        assert builder.convert_query(_parse("src[crypto]")) == "venue:crypto"

    def test_a_tiabs_term_is_unscoped_free_text(self, builder) -> None:
        """dblp has no combined title-abstract field, and no abstracts.

        Emitting an invented prefix would return zero hits at HTTP 200, so the
        term is sent unscoped and the shortfall is reported as coverage.
        """
        assert builder.convert_query(_parse("tiabs[alpha]")) == "alpha"

    def test_and_not_is_rejected(self, builder) -> None:
        """The silent inversion never leaves this process."""
        result = builder.validate_query(_parse("ti[alpha] AND NOT ti[beta]"))
        assert not result.is_valid
        assert "AND NOT" in (result.error_message or "")

    def test_abstract_only_filters_are_rejected(self, builder) -> None:
        """Answering an abstract query from titles is a different question."""
        assert not builder.validate_query(_parse("abs[alpha]")).is_valid
        assert not builder.validate_query(_parse("key[alpha]")).is_valid

    def test_structural_characters_in_a_term_are_rejected(self, builder) -> None:
        """A term carrying dblp's operators would be reparsed as structure."""
        for term in ("al|pha", "al:pha", "al(pha"):
            result = builder.validate_query(_parse(f"ti[{term}]"))
            assert not result.is_valid, term

    def test_an_unverified_wildcard_is_rejected(self, builder) -> None:
        """``?`` was never confirmed against the live parser.

        An unrecognised operator on this API does not error; it becomes a
        search token and returns zero hits at HTTP 200.
        """
        assert not builder.validate_query(_parse("ti[al?ha]")).is_valid

    def test_the_star_wildcard_is_accepted(self, builder) -> None:
        """``*`` is dblp's own prefix operator and widens recall."""
        assert builder.validate_query(_parse("ti[alpha*]")).is_valid
        assert builder.convert_query(_parse("ti[alpha*]")) == "title:alpha*"

    def test_the_dollar_anchor_is_rejected(self, builder) -> None:
        """``$`` suppresses prefix expansion, so it would silently narrow recall.

        dblp documents the anchor; an attempt to confirm its effect against the
        live parse echo on 2026-08-31 drew an HTTP 503 throttle instead.  An
        operator whose parse is unconfirmed is refused, on the same rule already
        applied to ``?`` -- and this one narrows, which is the direction that
        hides.
        """
        assert not builder.validate_query(_parse("ti[alpha$]")).is_valid


class TestGroupingIsNotExpressible:
    """dblp discards parentheses, so a grouped query is answered as a different one.

    Verified live 2026-08-31 against ``dblp.org/search/publ/api``:

    - ``(blockchain hash)|quantum`` echoes ``blockchain* hash*|quantum*`` and
      returns **660**; ``blockchain hash`` alone returns 213 and ``quantum``
      alone returns 55,489, so dblp answered ``blockchain AND (hash OR
      quantum)`` rather than the ``(blockchain AND hash) OR quantum`` that was
      sent -- roughly a hundredth of the intended result set, at HTTP 200.
    - ``(title:blockchain title:hash) | title:quantum`` echoes
      ``:filter:title:blockchain* :filter:title:hash)*|:filter:title:quantum*``:
      the ``)`` is absorbed into the field value, so the clause searches titles
      beginning ``hash)``.

    Before this gate existed the connector reported that run as
    ``dblp: complete``.
    """

    @pytest.fixture
    def builder(self) -> DBLPQueryBuilder:
        """Return a dblp query builder."""
        return DBLPQueryBuilder()

    def test_an_and_inside_an_or_is_rejected(self, builder) -> None:
        """The shape the discarded parentheses were carrying."""
        result = builder.validate_query(_parse("(ti[a] AND ti[b]) OR ti[c]"))
        assert not result.is_valid
        assert "parenthes" in (result.error_message or "")

    def test_mixed_connectors_without_grouping_are_rejected(self, builder) -> None:
        """lit-search folds left to right; dblp binds ``|`` tighter.

        ``[a] AND [b] OR [c]`` means ``(a AND b) OR c`` here and
        ``a AND (b OR c)`` there, with nothing in either result saying so.
        """
        assert not builder.validate_query(_parse("ti[a] AND ti[b] OR ti[c]")).is_valid

    def test_conjunctive_normal_form_survives(self, builder) -> None:
        """An ``OR`` inside an ``AND`` is dblp's own precedence, so it is kept."""
        assert builder.validate_query(_parse("ti[a] AND (ti[b] OR ti[c])")).is_valid
        assert builder.validate_query(_parse("(ti[a] OR ti[b]) AND (ti[c] OR ti[d])")).is_valid
        assert builder.validate_query(_parse("ti[a] AND (ti[b] AND ti[c])")).is_valid
        assert builder.validate_query(_parse("(ti[a] OR ti[b]) OR ti[c]")).is_valid

    def test_no_parenthesis_is_ever_emitted(self, builder) -> None:
        """Emitting a character dblp deletes is how the query changes meaning."""
        for source in (
            "ti[a] AND (ti[b] OR ti[c])",
            "(ti[a] OR ti[b]) AND (ti[c] OR ti[d])",
            "ti[a] AND (ti[b] AND ti[c])",
        ):
            emitted = builder.convert_query(_parse(source))
            assert "(" not in emitted, source
            assert ")" not in emitted, source

    def test_the_flat_rendering_preserves_the_intended_grouping(self, builder) -> None:
        """CNF renders correctly without grouping because ``|`` binds tighter."""
        assert (
            builder.convert_query(_parse("(ti[a] OR ti[b]) AND (ti[c] OR ti[d])"))
            == "title:a | title:b title:c | title:d"
        )
        assert (
            builder.convert_query(_parse("ti[a] AND (ti[b] OR ti[c])"))
            == "title:a title:b | title:c"
        )

    def test_the_rejection_reaches_the_caller_as_query_unsupported(self, builder) -> None:
        """A refused shape must be a loud status, never a quiet empty result."""
        connector = DBLPConnector()
        result = connector.search(_parse("(ti[a] AND ti[b]) OR ti[c]"))
        assert result.status is CoverageStatus.QUERY_UNSUPPORTED
        assert result.papers == ()


class TestBodyError:
    """The body error is a ConnectorError, which decides how it is reported."""

    def test_it_is_a_connector_error(self) -> None:
        """``search`` catches ``ConnectorError`` and reports ``FAILED``.

        If this stopped being a :class:`ConnectorError` the exception would
        escape ``search`` uncaught instead, so the inheritance is what routes
        dblp's HTML error pages to a coverage status rather than a traceback.
        """
        from lit_search.exceptions import ConnectorError

        assert issubclass(DBLPBodyError, ConnectorError)
