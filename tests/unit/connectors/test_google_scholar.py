"""Unit tests for GoogleScholarConnector.

These run under ``pytest-socket`` with ``--disable-socket``; every HTTP call is
mocked, and no test opts the connector in against the live site.

The three tests that carry the weight are all in
:class:`TestABlockIsNotAnEmptyResult`.  Between them they cover the donor's
whole defect surface: a block on page one, a block after some records, and a
markup change.  The donor returns an empty or short list for all three, and the
runner records each as a successful search of a database that holds no matching
article.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests
from lxml import html

from lit_search.connectors.google_scholar import (
    _OPT_IN_ENV_VAR,
    GoogleScholarConnector,
    ScholarMarkupError,
)
from lit_search.core.coverage import CoverageStatus, CoveredPapers
from lit_search.core.paper import Database

from .conftest import _parse

# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------

_PAGE = """<html><body><div id="gs_res_ccl_mid">{body}</div></body></html>"""

_RESULT = """
<div class="gs_r gs_or gs_scl">
  <h3 class="gs_rt"><a href="https://example.org/{n}">{title}</a></h3>
  <div class="gs_a">A Author, B Author - Journal of Things, {year} - example.org</div>
  <div class="gs_rs">A short snippet describing {title} and its findings ...</div>
</div>
"""

_BLOCK_PAGE = """<html><body><h1>Our systems have detected unusual traffic from your
computer network.</h1><div id="captcha"></div></body></html>"""

_NO_RESULTS_PAGE = """<html><body><div class="gs_med">Your search - flurbleglax -
did not match any articles.</div></body></html>"""

_CHANGED_MARKUP_PAGE = """<html><body><div class="totally-new-container">
<span>Attention Is All You Need</span></div></body></html>"""


def _results_page(count: int, *, offset: int = 0) -> str:
    """Build a Scholar results page with *count* result blocks.

    Parameters
    ----------
    count : int
        Number of result containers.
    offset : int
        Index offset, to keep titles distinct across pages.

    Returns
    -------
    str
        HTML document.
    """
    body = "".join(
        _RESULT.format(n=offset + i, title=f"Paper Number {offset + i}", year=2020 + i % 5)
        for i in range(count)
    )
    return _PAGE.format(body=body)


def _response(body: str, status: int = 200) -> MagicMock:
    """Wrap an HTML body in a mock response.

    Parameters
    ----------
    body : str
        HTML document.
    status : int
        HTTP status code.

    Returns
    -------
    MagicMock
        Mock response.
    """
    response = MagicMock(spec=requests.Response)
    response.status_code = status
    response.text = body
    return response


def _http_error(status: int, body: str = "") -> requests.HTTPError:
    """Build an ``HTTPError`` carrying a response.

    Parameters
    ----------
    status : int
        HTTP status code.
    body : str
        Response body.

    Returns
    -------
    requests.HTTPError
        Error with a response attached.
    """
    return requests.HTTPError(f"{status} error", response=_response(body, status))


def _search(connector, responses: list, **kwargs) -> CoveredPapers:
    """Run a search against a scripted sequence of ``_get`` outcomes.

    Parameters
    ----------
    connector : GoogleScholarConnector
        Connector under test.
    responses : list
        Values (or exceptions) ``_get`` yields in order.
    **kwargs
        Forwarded to ``search``.

    Returns
    -------
    CoveredPapers
        The result.
    """
    query = kwargs.pop("query", None) or _parse("ti[attention]")
    with patch.object(connector, "_get", side_effect=responses):
        result: CoveredPapers = connector.search(query, **kwargs)
    return result


@pytest.fixture
def opted_in(monkeypatch: pytest.MonkeyPatch) -> GoogleScholarConnector:
    """Return a connector with the policy acknowledgement set."""
    monkeypatch.setenv(_OPT_IN_ENV_VAR, "1")
    return GoogleScholarConnector()


@pytest.fixture(autouse=True)
def _default_opted_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure no ambient opt-in leaks into the default-state tests."""
    monkeypatch.delenv(_OPT_IN_ENV_VAR, raising=False)


# ---------------------------------------------------------------------------
# Default state
# ---------------------------------------------------------------------------


class TestShippedDisabled:
    """robots.txt disallows the path, so the default is not to fetch it."""

    def test_a_search_reports_skipped_no_credential(self) -> None:
        """Scholar stays visible in the ledger as deliberately not consulted."""
        result = GoogleScholarConnector().search(_parse("ti[attention]"))
        assert result.status is CoverageStatus.SKIPPED_NO_CREDENTIAL
        assert result.coverage.missing_credential == _OPT_IN_ENV_VAR

    def test_no_request_is_issued_when_opted_out(self) -> None:
        """The policy decision is made before any traffic, not after."""
        connector = GoogleScholarConnector()
        with patch.object(connector, "_get") as mocked:
            connector.search(_parse("ti[attention]"))
        mocked.assert_not_called()

    def test_it_is_registered_so_the_absence_is_visible(self) -> None:
        """An unregistered source cannot be reported as not consulted."""
        from lit_search.connectors import SEARCH_REGISTRY

        assert SEARCH_REGISTRY[Database.GOOGLE_SCHOLAR] is GoogleScholarConnector


# ---------------------------------------------------------------------------
# The donor's defect
# ---------------------------------------------------------------------------


class TestABlockIsNotAnEmptyResult:
    """Every shape of refusal becomes an explicit status."""

    def test_failure_and_emptiness_are_different_statuses(
        self, opted_in: GoogleScholarConnector
    ) -> None:
        """The regression test for this connector.

        A CAPTCHA on page one and a genuinely empty search must not produce
        the same status.  The donor returns ``[]`` for both.
        """
        blocked = _search(opted_in, [_response(_BLOCK_PAGE)])
        empty = _search(GoogleScholarConnector(), [_response(_NO_RESULTS_PAGE)])

        assert blocked.status is CoverageStatus.FAILED
        assert empty.status is CoverageStatus.PARTIAL
        assert empty.papers == ()
        assert blocked.status is not empty.status

    def test_a_captcha_served_with_http_200_is_a_failure(
        self, opted_in: GoogleScholarConnector
    ) -> None:
        """``raise_for_status`` is silent here; the body has to be read."""
        result = _search(opted_in, [_response(_BLOCK_PAGE)])
        assert result.status is CoverageStatus.FAILED
        assert "block or CAPTCHA" in (result.coverage.error or "")

    def test_an_http_429_is_a_failure_naming_the_block(
        self, opted_in: GoogleScholarConnector
    ) -> None:
        """A rate-limit refusal is not an empty index either."""
        result = _search(opted_in, [_http_error(429)])
        assert result.status is CoverageStatus.FAILED
        assert "refused the request" in (result.coverage.error or "")

    def test_a_block_after_some_records_keeps_them_and_reports_partial(
        self, opted_in: GoogleScholarConnector
    ) -> None:
        """Records already retrieved are real; discarding them is its own lie."""
        result = _search(opted_in, [_response(_results_page(20)), _response(_BLOCK_PAGE)])
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 20
        assert result.coverage.error_type == "ScholarBlockedError"

    def test_a_markup_change_is_a_failure_not_zero_results(
        self, opted_in: GoogleScholarConnector
    ) -> None:
        """A served page matching no selector is not an empty index.

        This is the case the donor cannot express at all: its parser finds no
        results, the loop breaks, and an empty list is returned.
        """
        result = _search(opted_in, [_response(_CHANGED_MARKUP_PAGE)])
        assert result.status is CoverageStatus.FAILED
        assert "markup change" in (result.coverage.error or "")

    def test_only_scholars_own_no_results_notice_justifies_zero(
        self, opted_in: GoogleScholarConnector
    ) -> None:
        """Zero is claimed on Scholar's own evidence, never on our silence."""
        result = _search(opted_in, [_response(_NO_RESULTS_PAGE)])
        assert result.status is CoverageStatus.PARTIAL
        assert result.papers == ()
        assert result.coverage.error is None

    def test_a_markup_change_mid_pagination_truncates_rather_than_raising(
        self, opted_in: GoogleScholarConnector
    ) -> None:
        """With records in hand the shortfall is reported, not thrown away."""
        result = _search(opted_in, [_response(_results_page(20)), _response(_CHANGED_MARKUP_PAGE)])
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 20
        assert "matched no result markup" in (result.coverage.detail or "")

    def test_the_markup_error_is_raised_by_the_handler_directly(self) -> None:
        """Pin the decision procedure itself, independent of the search loop."""
        run = GoogleScholarConnector()
        with pytest.raises(ScholarMarkupError):
            GoogleScholarConnector._handle_empty_page(
                type(run)._handle_empty_page.__self__ if False else _fresh_run(),
                html.fromstring(_CHANGED_MARKUP_PAGE),
            )


def _fresh_run():
    """Return an empty run record for the handler test.

    Returns
    -------
    _Run
        Bookkeeping with no papers retrieved.
    """
    from lit_search.connectors.google_scholar import _Run

    return _Run()


# ---------------------------------------------------------------------------
# Coverage and records
# ---------------------------------------------------------------------------


class TestCoverageAndRecords:
    """What a successful Scholar search reports and returns."""

    def test_a_normal_result_is_partial_never_complete(
        self, opted_in: GoogleScholarConnector
    ) -> None:
        """No reliable total exists, so completeness is never claimed."""
        result = _search(opted_in, [_response(_results_page(3)), _response(_NO_RESULTS_PAGE)])
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 3
        assert "no reliable result total" in (result.coverage.detail or "")

    def test_the_missing_abstract_and_doi_are_reported(
        self, opted_in: GoogleScholarConnector
    ) -> None:
        """A field-level degradation that follows the records downstream."""
        result = _search(opted_in, [_response(_results_page(1)), _response(_NO_RESULTS_PAGE)])
        assert "truncated snippet rather than an abstract" in (result.coverage.detail or "")
        assert result.papers[0].doi is None

    def test_the_byline_is_split_into_authors_venue_and_year(
        self, opted_in: GoogleScholarConnector
    ) -> None:
        """Scholar's one metadata line carries three different things."""
        result = _search(opted_in, [_response(_results_page(1)), _response(_NO_RESULTS_PAGE)])
        (paper,) = result.papers
        assert [a.name for a in paper.authors] == ["A Author", "B Author"]
        assert paper.source is not None
        assert paper.source.title == "Journal of Things"
        assert paper.publication_date is not None
        assert paper.publication_date.year == 2020

    def test_the_cap_is_honoured_and_reported(self, opted_in: GoogleScholarConnector) -> None:
        """A caller's limit appears in the detail."""
        result = _search(opted_in, [_response(_results_page(20))], max_papers=5)
        assert len(result.papers) == 5
        assert "cap of 5" in (result.coverage.detail or "")

    def test_search_returns_the_contract_type(self, opted_in: GoogleScholarConnector) -> None:
        """The connector cannot hand back a bare list."""
        assert isinstance(_search(opted_in, [_response(_NO_RESULTS_PAGE)]), CoveredPapers)


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------


class TestQueryBuilder:
    """Scopes Scholar has, and the ones it silently does not."""

    def test_the_project_default_scope_is_rejected(self) -> None:
        """A bare term means title-and-abstract here and full text there."""
        result = GoogleScholarConnector().search(_parse("[attention]"))
        assert result.status is CoverageStatus.QUERY_UNSUPPORTED

    def test_a_title_term_uses_scholars_own_prefix(self, opted_in: GoogleScholarConnector) -> None:
        """``[ti]`` becomes ``intitle:``."""
        with patch.object(opted_in, "_get", side_effect=[_response(_NO_RESULTS_PAGE)]) as mocked:
            opted_in.search(_parse("ti[attention]"))
        params = mocked.call_args.args[1]
        assert params["q"] == "intitle:attention"

    def test_and_not_becomes_scholars_minus(self, opted_in: GoogleScholarConnector) -> None:
        """Scholar excludes with ``-``, not with ``NOT``."""
        with patch.object(opted_in, "_get", side_effect=[_response(_NO_RESULTS_PAGE)]) as mocked:
            opted_in.search(_parse("ti[attention] AND NOT ti[recurrent]"))
        params = mocked.call_args.args[1]
        assert params["q"] == "intitle:attention - intitle:recurrent"

    def test_date_bounds_become_year_filters(self, opted_in: GoogleScholarConnector) -> None:
        """Scholar's date granularity is a year, and that is all it is sent."""
        import datetime

        with patch.object(opted_in, "_get", side_effect=[_response(_NO_RESULTS_PAGE)]) as mocked:
            opted_in.search(_parse("ti[attention]"), since=datetime.date(2020, 5, 1))
        params = mocked.call_args.args[1]
        assert params["as_ylo"] == 2020
