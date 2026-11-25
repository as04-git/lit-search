"""Unit tests for PMCConnector.

These run under ``pytest-socket`` with ``--disable-socket`` (see
``pyproject.toml``), so every HTTP call is mocked and no test can reach the
network.  All fixture payloads mirror shapes observed against the live NCBI
E-utilities on 2026-08-31.

The test that carries the most weight is
:meth:`TestCoverageIsHonest.test_failure_and_emptiness_are_different_statuses`.
Everything else here can pass while the connector still reports an upstream
outage as a confident zero; that one cannot.

Running it a close second is
:meth:`TestTwoHundredWithAnErrorBody.test_a_dropped_field_tag_is_query_unsupported`:
Entrez answers an unknown field tag with a plausible four-million-record count
rather than an error, which is the one failure here that is not merely silent
but actively convincing.
"""

from __future__ import annotations

import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from lit_search.connectors import pmc as pmc_module
from lit_search.connectors.pmc import PMCBodyError, PMCConnector
from lit_search.core.coverage import CoverageStatus, CoveredPapers, SourceCoverage
from lit_search.core.paper import Database, PaperType
from lit_search.core.source import SourceType
from lit_search.query.builders.pmc import PMCQueryBuilder

from .conftest import _parse

# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------

_ESEARCH_HEAD = (
    '<?xml version="1.0" encoding="UTF-8" ?>'
    '<!DOCTYPE eSearchResult PUBLIC "-//NLM//DTD esearch 20060628//EN" '
    '"https://eutils.ncbi.nlm.nih.gov/eutils/dtd/20060628/esearch.dtd">'
)


def _esearch(
    count: int,
    *,
    web_env: str = "MCID_test",
    query_key: str = "1",
    extra: str = "",
    omit_history: bool = False,
) -> str:
    """Build an ``esearch`` response body.

    Parameters
    ----------
    count : int
        Value of ``<Count>``.
    web_env : str
        Value of ``<WebEnv>``.
    query_key : str
        Value of ``<QueryKey>``.
    extra : str
        Raw XML appended inside ``<eSearchResult>`` (an ``ErrorList`` or a
        ``WarningList``).
    omit_history : bool
        Drop ``<WebEnv>`` and ``<QueryKey>`` entirely, as a malformed response
        would.

    Returns
    -------
    str
        XML body.
    """
    history = "" if omit_history else f"<QueryKey>{query_key}</QueryKey><WebEnv>{web_env}</WebEnv>"
    return (
        f"{_ESEARCH_HEAD}<eSearchResult><Count>{count}</Count><RetMax>0</RetMax>"
        f"<RetStart>0</RetStart>{history}<IdList/>{extra}</eSearchResult>"
    )


def _esearch_error(message: str) -> str:
    """Build an ``esearch`` response whose only content is an ``<ERROR>``.

    Parameters
    ----------
    message : str
        Error text.

    Returns
    -------
    str
        XML body.
    """
    return f"{_ESEARCH_HEAD}<eSearchResult><ERROR>{message}</ERROR></eSearchResult>"


def _docsum(index: int, **overrides: str) -> str:
    """Build one ``<DocSum>`` element.

    Parameters
    ----------
    index : int
        Used to make the title and identifiers distinct.
    **overrides : str
        Item values to replace (``Title``, ``PubDate``, ``EPubDate``, ``DOI``,
        ``FullJournalName``).

    Returns
    -------
    str
        XML fragment.
    """
    uid = 13500000 + index
    values = {
        "Title": f"Synthetic Paper {index}.",
        "PubDate": "2026 Sep",
        "EPubDate": "2026-08-29",
        "DOI": f"10.1000/pmc.{index}",
        "FullJournalName": "Journal of Synthetic Biology",
    }
    values.update(overrides)
    return (
        f"<DocSum><Id>{uid}</Id>"
        f'<Item Name="PubDate" Type="Date">{values["PubDate"]}</Item>'
        f'<Item Name="EPubDate" Type="Date">{values["EPubDate"]}</Item>'
        f'<Item Name="Source" Type="String">J Synth Biol</Item>'
        f'<Item Name="AuthorList" Type="List">'
        f'<Item Name="Author" Type="String">Saleh NA</Item>'
        f'<Item Name="Author" Type="String">Shafiq R</Item>'
        f"</Item>"
        f'<Item Name="Title" Type="String">{values["Title"]}</Item>'
        f'<Item Name="ArticleIds" Type="List">'
        f'<Item Name="pmid" Type="String">4266900{index}</Item>'
        f'<Item Name="pmcid" Type="String">PMC{uid}</Item>'
        f'<Item Name="doi" Type="String">{values["DOI"]}</Item>'
        f"</Item>"
        f'<Item Name="DOI" Type="String">{values["DOI"]}</Item>'
        f'<Item Name="FullJournalName" Type="String">{values["FullJournalName"]}</Item>'
        f"</DocSum>"
    )


def _esummary(docsums: str = "", errors: tuple[str, ...] = ()) -> str:
    """Build an ``esummary`` response body.

    Parameters
    ----------
    docsums : str
        Concatenated ``<DocSum>`` fragments.
    errors : tuple[str, ...]
        ``<ERROR>`` texts to include alongside them.

    Returns
    -------
    str
        XML body.
    """
    error_xml = "".join(f"<ERROR>{text}</ERROR>" for text in errors)
    return (
        '<?xml version="1.0" encoding="UTF-8" ?>'
        f"<eSummaryResult>{docsums}{error_xml}</eSummaryResult>"
    )


def _summary_page(indices: range | list[int], errors: tuple[str, ...] = ()) -> str:
    """Build an ``esummary`` page holding one ``DocSum`` per index.

    Parameters
    ----------
    indices : range | list[int]
        Record indices.
    errors : tuple[str, ...]
        ``<ERROR>`` texts to include.

    Returns
    -------
    str
        XML body.
    """
    return _esummary("".join(_docsum(i) for i in indices), errors)


def _response(text: str, status_code: int = 200) -> MagicMock:
    """Wrap *text* in a mock :class:`requests.Response`.

    Parameters
    ----------
    text : str
        Body returned by ``.text``.
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
    return mock


@pytest.fixture
def connector() -> PMCConnector:
    """Return a PMC connector with no credential."""
    return PMCConnector()


def _search(connector: PMCConnector, bodies: list[Any], **kwargs: Any) -> CoveredPapers:
    """Run a search against a scripted sequence of ``_get`` outcomes.

    Parameters
    ----------
    connector : PMCConnector
        Connector under test.
    bodies : list[Any]
        XML strings, mock responses, or exceptions for successive ``_get``
        calls.  The first is always the ``esearch`` response.
    **kwargs : Any
        Forwarded to ``search`` (``max_papers``, ``since``, ``until``).

    Returns
    -------
    CoveredPapers
        The search outcome.
    """
    query = kwargs.pop("query", None) or _parse("[machine learning]")
    outcomes = [_response(b) if isinstance(b, str) else b for b in bodies]
    with (
        patch.object(connector, "_rate_limit"),
        patch.object(connector, "_get", side_effect=outcomes),
    ):
        return connector.search(query, **kwargs)


# ---------------------------------------------------------------------------
# Declared contract
# ---------------------------------------------------------------------------


class TestDeclaredContract:
    """The connector declares what the base classes require of it."""

    def test_name_is_the_registered_database_identifier(self, connector) -> None:
        """The ledger key matches the Database member."""
        assert connector.name == Database.PMC.value

    def test_min_request_interval_matches_the_documented_ncbi_rate(self, connector) -> None:
        """3 req/s without a key, 10 req/s with one; both are documented."""
        assert connector.min_request_interval == 0.34
        assert PMCConnector(api_key="k").min_request_interval == 0.11

    def test_registered_in_the_search_registry(self) -> None:
        """A connector nobody can select is not ported."""
        from lit_search.connectors import SEARCH_REGISTRY

        assert SEARCH_REGISTRY[Database.PMC] is PMCConnector

    def test_not_registered_for_lookup_roles_it_does_not_implement(self) -> None:
        """A connector registered for an unimplemented role returns silent Nones."""
        from lit_search.connectors import DOI_LOOKUP_REGISTRY, URL_LOOKUP_REGISTRY

        assert Database.PMC not in DOI_LOOKUP_REGISTRY
        assert Database.PMC not in URL_LOOKUP_REGISTRY

    def test_the_rate_limit_bucket_is_shared_with_pubmed(self, connector) -> None:
        """One NCBI key, one allowance, across every E-utilities endpoint.

        Two scopes against the same key would double the real request rate, and
        the symptom -- intermittent 429s -- is exactly the one that has to be
        disambiguated from genuine emptiness.
        """
        from lit_search.connectors.pubmed import PubmedConnector

        assert connector.rate_limit_service == "ncbi"
        assert connector.rate_limit_scope == PubmedConnector().rate_limit_scope

    def test_a_shared_key_lands_in_one_bucket(self) -> None:
        """The scope is keyed by credential, so the same key is the same clock."""
        from lit_search.connectors.pubmed import PubmedConnector

        assert (
            PMCConnector(api_key="shared").rate_limit_scope
            == PubmedConnector(api_key="shared").rate_limit_scope
        )

    def test_credential_attribute_is_the_default(self, connector) -> None:
        """The key lives on ``_api_key``, so the default declaration is correct."""
        assert connector._credential_attrs == ("_api_key",)

    def test_the_runner_actually_hands_pmc_the_ncbi_key(self) -> None:
        """The shared bucket is a property of the *system*, not of the class.

        The two assertions above both hold while a real run still splits the
        NCBI quota in two, because they construct the connectors themselves.
        The limiter's bucket is ``(service, credential fingerprint)``, so a
        runner that gives the key to ``pubmed`` and not to ``pmc`` produces
        ``('ncbi', <key>)`` at 0.11s alongside ``('ncbi', anon)`` at 0.34s,
        both firing at one IP against one allowance.  That was the state of
        ``SearchRunner._build_searchers`` when this test was written, and the
        symptom -- intermittent 429s on the unkeyed half -- reads as thin
        recall rather than as a misconfiguration.
        """
        from lit_search.runners.search_runner import SearchRunner

        runner = SearchRunner(
            query="[machine learning]",
            databases=["pmc", "pubmed"],
            pubmed_api_key="shared-ncbi-key",
        )
        scopes = {s.name: s.rate_limit_scope for s in runner._searchers}
        assert scopes[Database.PMC.value] == scopes[Database.PUBMED.value]
        intervals = {s.name: s.min_request_interval for s in runner._searchers}
        assert intervals[Database.PMC.value] == intervals[Database.PUBMED.value]


# ---------------------------------------------------------------------------
# The four required cases, plus the one that matters
# ---------------------------------------------------------------------------


class TestCoverageIsHonest:
    """Each way a search can end reports a different, actionable status."""

    def test_normal_result_is_complete(self, connector) -> None:
        """Retrieving as many records as ``<Count>`` reports is COMPLETE."""
        result = _search(connector, [_esearch(2), _summary_page(range(2))])
        assert result.status is CoverageStatus.COMPLETE
        assert len(result.papers) == 2
        assert result.coverage.retrieved == 2
        assert result.coverage.reported_total == 2

    def test_upstream_failure_is_failed(self, connector) -> None:
        """A dead esearch is FAILED, and the error is carried."""
        result = _search(connector, [requests.ConnectionError("eutils unreachable")])
        assert result.status is CoverageStatus.FAILED
        assert result.papers == ()
        assert "unreachable" in (result.coverage.error or "")
        assert result.coverage.error_type == "ConnectionError"

    def test_empty_but_successful_search_is_complete(self, connector) -> None:
        """A genuine zero-hit query is COMPLETE with no papers.

        This is the honest empty: Entrez answered, said zero records match, and
        no summary request was needed.
        """
        result = _search(connector, [_esearch(0)])
        assert result.status is CoverageStatus.COMPLETE
        assert result.papers == ()
        assert result.coverage.reported_total == 0

    def test_capped_result_is_capped(self, connector) -> None:
        """Stopping at max_papers is CAPPED and names the cap."""
        result = _search(
            connector,
            [_esearch(900), _summary_page(range(3))],
            max_papers=3,
        )
        assert result.status is CoverageStatus.CAPPED
        assert len(result.papers) == 3
        assert result.coverage.cap == 3
        assert result.coverage.reported_total == 900

    def test_failure_and_emptiness_are_different_statuses(self, connector) -> None:
        """The regression test for the whole port.

        In the donor implementation both of these paths end at the same
        ``return papers`` with ``papers == []`` -- one through ``except
        requests.RequestException``, the other through ``if not pmcids: return
        papers`` -- so a caller cannot tell an outage from a database that holds
        nothing.  Here they must differ.
        """
        failure = _search(connector, [requests.ConnectionError("boom")])
        empty = _search(connector, [_esearch(0)])

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
    """NCBI reports everything in the body with an HTTP 200 status.

    Each of these satisfies ``raise_for_status`` and then yields ``[]`` from
    ``findall('.//Id')``.
    """

    def test_a_dropped_field_tag_is_query_unsupported(self, connector) -> None:
        """The most convincing failure in this connector, and the least visible.

        ``cancer[ab]`` returns HTTP 200 with ``Count`` 4,013,901 -- the same as
        an unscoped search -- because Entrez drops the unknown tag and re-runs
        the term against every field.  The only trace is ``<ErrorList>``.  A
        port that reused PubMed's tags would hand back four million full-text
        hits as the answer to an abstract-scoped query.
        """
        body = _esearch(
            4013901,
            extra="<ErrorList><FieldNotFound>ab</FieldNotFound></ErrorList>",
        )
        result = _search(connector, [body])
        assert result.status is CoverageStatus.QUERY_UNSUPPORTED
        assert result.papers == ()
        assert "does not index the field tag" in (result.coverage.detail or "")

    def test_empty_term_error_is_query_unsupported(self, connector) -> None:
        """A body blaming the term is a query problem, not an outage.

        The distinction is actionable: FAILED invites a retry, and retrying a
        query the source cannot express only wastes the rate-limit budget.
        """
        result = _search(connector, [_esearch_error("Empty term and query_key - nothing todo")])
        assert result.status is CoverageStatus.QUERY_UNSUPPORTED
        assert result.papers == ()

    def test_invalid_db_error_is_failed_not_query_unsupported(self, connector) -> None:
        """A bad database name is this connector's bug and must be loud."""
        result = _search(connector, [_esearch_error("Invalid db name specified: notadb")])
        assert result.status is CoverageStatus.FAILED
        assert "Invalid db name" in (result.coverage.error or "")

    def test_retstart_ceiling_error_is_failed(self, connector) -> None:
        """Asking past record 9,998 is a request-shape bug, not a query problem."""
        result = _search(
            connector,
            [_esearch_error("Search Backend failed: 'retstart' cannot be larger than 9998.")],
        )
        assert result.status is CoverageStatus.FAILED

    def test_a_requoted_phrase_is_partial_not_complete(self, connector) -> None:
        """Entrez re-tokenised the phrase, so the query that ran is not the one asked.

        The zero it returned is a real zero for a *different* query, and saying
        so is the difference between "PMC holds nothing on this" and "PMC never
        looked for this".
        """
        body = _esearch(
            0,
            extra=(
                "<WarningList><QuotedPhraseNotFound>&quot;machine learning&quot;[ti]"
                "</QuotedPhraseNotFound><OutputMessage>No items found.</OutputMessage>"
                "</WarningList>"
            ),
        )
        result = _search(connector, [body])
        assert result.status is CoverageStatus.PARTIAL
        assert "no phrase index entry" in (result.coverage.detail or "")

    def test_a_single_word_phrase_warning_is_not_a_shortfall(self, connector) -> None:
        """Entrez raises the same warning for one word, where nothing changed.

        Verified live 2026-08-31: ``"zzqqwibbleflurb"[ti]`` reports
        ``QuotedPhraseNotFound`` and echoes a ``QueryTranslation`` identical to
        what was sent.  Treating that as a shortfall would make an ordinary
        unmatched single-word term ``PARTIAL``, which is the opposite error to
        the one this fork exists to prevent: the honest empty would become
        unreachable and the label would stop meaning anything.
        """
        body = _esearch(
            0,
            extra=(
                "<WarningList><QuotedPhraseNotFound>&quot;zzqqwibbleflurb&quot;[ti]"
                "</QuotedPhraseNotFound><OutputMessage>No items found.</OutputMessage>"
                "</WarningList>"
            ),
        )
        result = _search(connector, [body])
        assert result.status is CoverageStatus.COMPLETE
        assert result.papers == ()

    def test_an_ignored_term_is_partial(self, connector) -> None:
        """A term Entrez dropped narrows the query without saying so upstream."""
        body = _esearch(
            5,
            extra="<WarningList><PhraseIgnored>the</PhraseIgnored></WarningList>",
        )
        result = _search(connector, [body, _summary_page(range(5))])
        assert result.status is CoverageStatus.PARTIAL
        assert "ignored the term" in (result.coverage.detail or "")

    def test_non_xml_body_is_failed(self, connector) -> None:
        """NCBI serves HTML maintenance pages with a 200 status."""
        result = _search(connector, ["<html><body>Service unavailable</body>"])
        assert result.status is CoverageStatus.FAILED
        assert "not parsable XML" in (result.coverage.error or "")

    def test_missing_count_is_failed_not_empty(self, connector) -> None:
        """A body with no usable ``<Count>`` is not a search result."""
        result = _search(connector, [f"{_ESEARCH_HEAD}<eSearchResult><IdList/></eSearchResult>"])
        assert result.status is CoverageStatus.FAILED
        assert "Count" in (result.coverage.error or "")

    def test_records_without_a_history_handle_are_failed(self, connector) -> None:
        """Reporting zero here would hide a result set that exists."""
        result = _search(connector, [_esearch(42, omit_history=True)])
        assert result.status is CoverageStatus.FAILED
        assert "no history handle" in (result.coverage.error or "")

    def test_body_error_classification(self) -> None:
        """The exception carries the query-versus-request distinction."""
        assert PMCBodyError("x", is_query_error=True).is_query_error is True
        assert PMCBodyError("x").is_query_error is False


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class TestPagination:
    """Offset paging over the history server, and every abnormal stop."""

    def test_multi_page_run_accumulates_until_the_count_is_reached(self, connector) -> None:
        """Two full pages and a short last page."""
        result = _search(
            connector,
            [
                _esearch(201),
                _summary_page(range(100)),
                _summary_page(range(100, 200)),
                _summary_page([200]),
            ],
        )
        assert result.status is CoverageStatus.COMPLETE
        assert len(result.papers) == 201

    def test_summary_failure_after_some_pages_is_partial_not_failed(self, connector) -> None:
        """Records that did arrive are real, so the run is PARTIAL.

        FAILED would discard true records; COMPLETE would hide the shortfall.
        """
        result = _search(
            connector,
            [
                _esearch(300),
                _summary_page(range(100)),
                requests.ConnectionError("dropped mid-run"),
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 100
        assert result.coverage.error_type == "ConnectionError"
        assert "dropped mid-run" in (result.coverage.detail or "")

    def test_page_level_error_after_some_pages_is_partial(self, connector) -> None:
        """A whole-page ``<ERROR>`` partway through degrades rather than discards."""
        result = _search(
            connector,
            [
                _esearch(300),
                _summary_page(range(100)),
                _esummary(errors=("Search backend can not retrieve history data.",)),
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 100

    def test_a_page_level_error_on_the_first_page_is_failed(self, connector) -> None:
        """With nothing in hand the failure is the whole story."""
        result = _search(
            connector,
            [_esearch(5), _esummary(errors=("Search backend can not retrieve history data.",))],
        )
        assert result.status is CoverageStatus.FAILED

    def test_per_record_errors_are_counted_as_a_coverage_loss(self, connector) -> None:
        """A UID Entrez acknowledged and could not summarise is a lost record.

        Verified live: a batch containing an unresolvable UID returns the good
        ``DocSum`` elements plus ``<ERROR>UID=999999999: cannot get document
        summary</ERROR>``, at HTTP 200.  Skipping those would shrink the result
        with nothing recording that it shrank.
        """
        result = _search(
            connector,
            [
                _esearch(3),
                _summary_page(range(2), errors=("UID=999999999: cannot get document summary",)),
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 2
        assert "could not return 1 record" in (result.coverage.detail or "")
        assert "could not be parsed" in (result.coverage.detail or "")

    def test_an_empty_page_before_the_count_is_partial(self, connector) -> None:
        """Neither records nor errors means the offset is not advancing."""
        result = _search(
            connector,
            [_esearch(300), _summary_page(range(100)), _esummary()],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert "neither records nor errors" in (result.coverage.detail or "")

    def test_unparsable_records_are_counted_as_a_shortfall(self, connector) -> None:
        """A record seen but not parsed is a coverage loss, not an absence."""
        page = _esummary(_docsum(0) + _docsum(1, Title=""))
        result = _search(connector, [_esearch(2), page])
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 1
        assert "could not be parsed" in (result.coverage.detail or "")

    def test_coverage_count_always_matches_the_papers_returned(self, connector) -> None:
        """CoveredPapers rejects a mismatch, so this asserts the accounting."""
        result = _search(
            connector,
            [_esearch(99), _summary_page(range(2))],
            max_papers=2,
        )
        assert result.coverage.retrieved == len(result.papers) == 2


# ---------------------------------------------------------------------------
# NCBI's own 9,999-record ceiling
# ---------------------------------------------------------------------------


class TestTheEntrezRecordCeiling:
    """No Entrez query can yield more than 9,999 records, by any route."""

    def test_the_ceiling_constant_matches_what_ncbi_enforces(self) -> None:
        """Verified live: retstart above 9998 returns an ``<ERROR>``."""
        assert pmc_module._MAX_RETRIEVABLE == 9999

    def test_target_is_bounded_by_the_ceiling_and_the_note_names_it(self) -> None:
        """A run of 20,000 matches can only ever reach 9,999 of them."""
        target, note = PMCConnector._retrieval_target(20000, None)
        assert target == 9999
        assert note is not None
        assert "9999" in note
        assert "EDirect" in note

    def test_no_note_when_the_callers_cap_is_the_binding_limit(self) -> None:
        """Attributing the stop to NCBI when the caller asked for 5 would mislead."""
        target, note = PMCConnector._retrieval_target(20000, 5)
        assert target == 5
        assert note is None

    def test_no_note_when_the_result_set_fits(self) -> None:
        """A ceiling that did not bind is not a shortfall."""
        target, note = PMCConnector._retrieval_target(12, None)
        assert target == 12
        assert note is None

    def test_a_ceiling_bounded_run_reports_partial(self, connector, monkeypatch) -> None:
        """End to end, with the ceiling lowered so the run stays small.

        The mechanism is the same at 5 as at 9,999; the constant itself is
        asserted separately above.
        """
        monkeypatch.setattr(pmc_module, "_MAX_RETRIEVABLE", 5)
        result = _search(connector, [_esearch(20), _summary_page(range(5))])
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 5
        assert result.coverage.reported_total == 20
        assert "refuse to return more than 5 records" in (result.coverage.detail or "")


# ---------------------------------------------------------------------------
# Request shaping
# ---------------------------------------------------------------------------


class TestRequestShaping:
    """Parameters sent upstream match what the live API accepts."""

    def _capture(self, connector, **kwargs) -> list[Any]:
        """Run one search and return the parameter dicts of every request."""
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(
                connector,
                "_get",
                side_effect=[_response(_esearch(1)), _response(_summary_page(range(1)))],
            ) as mocked,
        ):
            connector.search(_parse("[machine learning]"), **kwargs)
        return [call.args[1] for call in mocked.call_args_list]

    def test_esearch_uses_the_history_server(self, connector) -> None:
        """Paging a history snapshot stops a live index shifting the window."""
        params = self._capture(connector)[0]
        assert params["db"] == "pmc"
        assert params["usehistory"] == "y"
        assert params["retmode"] == "xml"

    def test_esummary_pages_the_history_handle(self, connector) -> None:
        """The summary request carries WebEnv and query_key, not a UID list."""
        params = self._capture(connector)[1]
        assert params["WebEnv"] == "MCID_test"
        assert params["query_key"] == "1"
        assert params["retstart"] == 0
        assert params["retmax"] == 1

    def test_date_bounds_use_full_entrez_precision(self, connector) -> None:
        """Entrez accepts YYYY/MM/DD on db=pmc, so no bound has to be widened."""
        params = self._capture(
            connector,
            since=datetime.date(2020, 3, 4),
            until=datetime.date(2021, 11, 30),
        )[0]
        assert params["datetype"] == "pdat"
        assert params["mindate"] == "2020/03/04"
        assert params["maxdate"] == "2021/11/30"

    def test_no_date_filter_when_no_bounds_are_given(self, connector) -> None:
        """An unbounded search must not silently acquire a date restriction."""
        params = self._capture(connector)[0]
        assert "datetype" not in params

    @staticmethod
    def _capture_prepared(connector) -> list[Any]:
        """Return the parameters of every request *after* credential injection.

        Patching ``_get`` would capture the parameters the connector composed,
        which is one layer above ``_prepare_params`` and therefore says nothing
        about what actually goes on the wire.
        """
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(
                connector,
                "_request_with_retry_uncounted",
                side_effect=[_response(_esearch(1)), _response(_summary_page(range(1)))],
            ) as mocked,
        ):
            connector.search(_parse("[machine learning]"))
        return [call.kwargs["params"] for call in mocked.call_args_list]

    def test_identification_parameters_are_omitted_when_unset(self, connector) -> None:
        """A placeholder address misidentifies every caller that ports it.

        The donor connector hardcodes ``openags@example.com`` on every request.
        """
        for params in self._capture_prepared(connector):
            assert "tool" not in params
            assert "email" not in params
            assert "api_key" not in params

    def test_identification_parameters_are_sent_when_set(self) -> None:
        """NCBI asks that tool and email be set and registered.

        Checked on both requests: an ``api_key`` that reaches ``esearch`` and
        not ``esummary`` would silently halve the effective rate limit.
        """
        connector = PMCConnector(api_key="k", tool="lit-search", email="a@example.edu")
        captured = self._capture_prepared(connector)
        assert len(captured) == 2
        for params in captured:
            assert params["tool"] == "lit-search"
            assert params["email"] == "a@example.edu"
            assert params["api_key"] == "k"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    """Field mapping from an esummary ``DocSum`` to a Paper."""

    @staticmethod
    def _first(connector, page: str):
        """Parse the first DocSum in *page*."""
        from defusedxml import ElementTree as ET

        root = ET.fromstring(page)
        return connector._parse_docsum(root.findall("DocSum")[0])

    def test_core_fields(self, connector) -> None:
        """Title, DOI, date and URLs are mapped."""
        paper = self._first(connector, _summary_page([7]))
        assert paper is not None
        assert paper.title == "Synthetic Paper 7."
        assert paper.doi == "10.1000/pmc.7"
        assert paper.publication_date == datetime.date(2026, 8, 29)
        assert paper.url == "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC13500007/"
        assert paper.pdf_url == "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC13500007/pdf/"
        assert paper.is_open_access is True
        assert paper.paper_type is PaperType.ARTICLE
        assert Database.PMC in paper.found_in

    def test_the_abstract_is_empty_and_that_is_a_known_gap(self, connector) -> None:
        """``esummary`` carries no abstract, and ``efetch`` is unusable here.

        Measured: ``efetch`` for one ``db=pmc`` record returned 143,966 bytes of
        JATS, so fetching abstracts for a page of 100 would move megabytes per
        page.  This is a *field* gap, not a *record* gap, so it is not a
        coverage status -- but it is not silent either.
        """
        paper = self._first(connector, _summary_page([1]))
        assert paper is not None
        assert paper.abstract == ""

    def test_authors_are_parsed_without_affiliations(self, connector) -> None:
        """The summary record has names only; affiliations are not in it."""
        paper = self._first(connector, _summary_page([1]))
        assert paper is not None
        assert [a.name for a in paper.authors] == ["Saleh NA", "Shafiq R"]
        assert paper.authors[0].affiliation is None

    def test_journal_becomes_a_journal_source(self, connector) -> None:
        """FullJournalName is preferred over the abbreviated Source."""
        paper = self._first(connector, _summary_page([1]))
        assert paper is not None
        assert paper.source is not None
        assert paper.source.title == "Journal of Synthetic Biology"
        assert paper.source.source_type is SourceType.JOURNAL

    def test_pub_date_is_used_when_there_is_no_epub_date(self, connector) -> None:
        """``2026 Sep`` is a real PubDate shape and must not be dropped."""
        paper = self._first(connector, _esummary(_docsum(2, EPubDate="")))
        assert paper is not None
        assert paper.publication_date == datetime.date(2026, 9, 1)

    def test_untitled_record_is_rejected(self, connector) -> None:
        """A record with no title cannot become a Paper."""
        assert self._first(connector, _esummary(_docsum(3, Title=""))) is None

    def test_pmcid_is_derived_from_the_uid_when_the_id_list_lacks_one(self, connector) -> None:
        """The Id element is the fallback identifier."""
        page = _esummary(
            '<DocSum><Id>13500009</Id><Item Name="Title" Type="String">Fallback</Item></DocSum>'
        )
        paper = self._first(connector, page)
        assert paper is not None
        assert paper.url.endswith("/PMC13500009/")


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------


class TestQueryBuilder:
    """Only tags PMC indexes reach the API, and unexpressible scopes are refused."""

    @pytest.fixture
    def builder(self) -> PMCQueryBuilder:
        """Return a query builder."""
        return PMCQueryBuilder()

    def test_default_filter_is_title_abstract(self, builder) -> None:
        """PMC has no author-keyword field, so tiabskey is not the default."""
        assert builder.convert_query(_parse("[machine learning]")) == '"machine learning"[tiab]'

    @pytest.mark.parametrize(
        ("query_string", "expected"),
        [
            ("ti[crispr]", '"crispr"[ti]'),
            ("au[Doudna]", '"Doudna"[au]'),
            ("src[Nature]", '"Nature"[ta]'),
            ("aff[Toronto]", '"Toronto"[ad]'),
            ("tiabs[crispr]", '"crispr"[tiab]'),
        ],
    )
    def test_each_supported_filter_maps_to_an_indexed_tag(
        self, builder, query_string, expected
    ) -> None:
        """Every emitted tag was confirmed present in PMC's live einfo field list."""
        assert builder.convert_query(_parse(query_string)) == expected

    def test_no_pubmed_only_tag_is_ever_emitted(self, builder) -> None:
        """``[ab]`` and ``[ot]`` are PubMed tags that PMC silently drops.

        Emitting either turns a scoped query into a four-million-record
        full-text search that looks like a successful result.
        """
        from lit_search.query.builders.pmc import FIELD_MAP

        assert "ab" not in FIELD_MAP.values()
        assert "ot" not in FIELD_MAP.values()

    @pytest.mark.parametrize("query_string", ["abs[crispr]", "key[crispr]"])
    def test_scopes_pmc_cannot_express_are_refused(self, builder, query_string) -> None:
        """Approximating them would answer a different question silently."""
        result = builder.validate_query(_parse(query_string))
        assert result.is_valid is False
        assert "not supported by PMC" in (result.error_message or "")

    def test_the_abstract_rejection_says_what_to_do_instead(self, builder) -> None:
        """A refusal a user cannot act on is only half a refusal."""
        message = builder.validate_query(_parse("abs[crispr]")).error_message or ""
        assert "tiabs" in message
        assert "Europe PMC" in message

    def test_connectors_map_to_entrez_operators(self, builder) -> None:
        """AND NOT becomes NOT, which is Entrez's exclusion operator."""
        assert builder.convert_query(_parse("ti[a] AND NOT ti[b]")) == '"a"[ti] NOT "b"[ti]'

    def test_long_enough_wildcards_are_left_unquoted(self, builder) -> None:
        """Entrez does not expand a truncation inside a quoted phrase."""
        assert builder.convert_query(_parse("ti[neural*]")) == "neural*[ti]"

    def test_short_wildcard_prefixes_are_rejected(self, builder) -> None:
        """Measured: 'ne*[ti]' returned 579 against 369,990 for 'neur*[ti]'."""
        result = builder.validate_query(_parse("ti[ne*]"))
        assert result.is_valid is False
        assert "requires at least 4 before" in (result.error_message or "")

    def test_question_mark_is_rejected(self, builder) -> None:
        """Entrez treats '?' as a literal, so the query silently changes meaning."""
        result = builder.validate_query(_parse("ti[neu?al]"))
        assert result.is_valid is False
        assert "'?'" in (result.error_message or "")

    def test_valid_query_passes(self, builder) -> None:
        """A normal query is accepted."""
        assert builder.validate_query(_parse("[machine learning] AND [deep learning]")).is_valid

    def test_rejected_query_reaches_the_ledger_as_query_unsupported(self, connector) -> None:
        """A refused query is a coverage status, never a raised error or a zero."""
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(connector, "_get") as mocked,
        ):
            result = connector.search(_parse("abs[crispr]"))
        assert result.status is CoverageStatus.QUERY_UNSUPPORTED
        assert result.papers == ()
        mocked.assert_not_called()


# ---------------------------------------------------------------------------
# The base class's second opinion
# ---------------------------------------------------------------------------


class _LyingPMCConnector(PMCConnector):
    """A PMC connector that reports every run as COMPLETE.

    Used to prove that this connector's HTTP goes through ``_get`` -- and
    therefore through the base class's request audit.  If it did not, the audit
    would see no failed request and the lie would survive.
    """

    def _build_coverage(self, run: Any, **kwargs: Any) -> SourceCoverage:
        """Claim COMPLETE regardless of what happened.

        Parameters
        ----------
        run : Any
            The retrieval bookkeeper, ignored.
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
            _response(_esearch(300)),
            _response(_summary_page(range(100))),
            requests.ConnectionError("page 2 refused"),
        ]

        def _fake_request(*_args: Any, **_kwargs: Any) -> Any:
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        connector = _LyingPMCConnector()
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(connector, "_request_with_retry_uncounted", side_effect=_fake_request),
        ):
            result = connector.search(_parse("[machine learning]"))

        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 100
        assert "request(s) failed" in (result.coverage.detail or "")
        assert "ConnectionError" in (result.coverage.detail or "")
