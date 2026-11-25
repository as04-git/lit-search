"""Unit tests for IACRConnector.

These run under ``pytest-socket`` with ``--disable-socket`` (see
``pyproject.toml``), so every HTTP call is mocked and no test can reach the
network.  Fixture payloads mirror responses observed against the live OAI-PMH
endpoint on 2026-08-31.

Two tests carry the most weight:

- :meth:`TestCoverageIsHonest.test_failure_and_emptiness_are_different_statuses`,
  the regression test for the port as a whole; and
- :meth:`TestCoverageIsHonest.test_no_records_match_is_an_empty_result_not_a_failure`,
  which pins the one OAI-PMH error code that means "genuinely nothing here".
  Getting that backwards turns an empty window into a fake outage, or every
  malformed request into a fake empty archive.
"""

from __future__ import annotations

import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from lit_search.connectors.iacr import (
    _OPT_IN_ENV_VAR,
    IACRConnector,
    IACROAIError,
)
from lit_search.core.coverage import CoverageStatus, CoveredPapers
from lit_search.core.paper import Database, PaperType
from lit_search.core.source import SourceType
from lit_search.query.builders.iacr import IACRQueryBuilder

from .conftest import _parse

# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------

_ENVELOPE = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">'
    "<responseDate>2026-08-31T08:51:04Z</responseDate>"
    '<request verb="ListRecords">https://eprint.iacr.org/oai</request>'
    "{body}"
    "</OAI-PMH>"
)


def _record(
    index: int,
    *,
    title: str = "Threshold Signatures from Lattices",
    description: str = "We present a lattice-based threshold signature scheme.",
    subject: str = "Public-key cryptography",
    creators: tuple[str, ...] = ("Alice Cryptographer",),
    date: str = "2026-08-05T18:59:15Z",
) -> str:
    """Build one ``<record>`` in ``oai_dc``.

    Parameters
    ----------
    index : int
        Used to make identifiers distinct.
    title : str
        ``dc:title`` value.
    description : str
        ``dc:description`` value (the abstract).
    subject : str
        ``dc:subject`` value (the archive's category).
    creators : tuple[str, ...]
        ``dc:creator`` values.
    date : str
        ``dc:date`` value.

    Returns
    -------
    str
        Record XML.
    """
    people = "".join(f"<dc:creator>{name}</dc:creator>" for name in creators)
    return (
        "<record>"
        f"<header><identifier>oai:eprint.iacr.org:2026/{index}</identifier>"
        f"<datestamp>{date}</datestamp></header>"
        "<metadata>"
        '<oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"'
        ' xmlns:dc="http://purl.org/dc/elements/1.1/">'
        f"<dc:identifier>https://eprint.iacr.org/2026/{index}</dc:identifier>"
        f"<dc:title>{title}</dc:title>"
        f"{people}"
        f"<dc:subject>{subject}</dc:subject>"
        f"<dc:description>{description}</dc:description>"
        f"<dc:date>{date}</dc:date>"
        "<dc:publisher>International Association for Cryptologic Research (IACR)</dc:publisher>"
        "<dc:type>e-print</dc:type>"
        "<dc:language>en</dc:language>"
        "</oai_dc:dc>"
        "</metadata>"
        "</record>"
    )


def _list_records(records: str, token: str | None = None, complete_size: int | None = None) -> str:
    """Build a ``ListRecords`` response.

    Parameters
    ----------
    records : str
        Concatenated record XML.
    token : str | None
        ``resumptionToken`` text.  ``None`` omits the element, which is the
        protocol's end-of-list signal.
    complete_size : int | None
        ``completeListSize`` attribute on the token.

    Returns
    -------
    str
        Response XML.
    """
    resumption = ""
    if token is not None:
        attrs = f' completeListSize="{complete_size}"' if complete_size is not None else ""
        resumption = f'<resumptionToken{attrs} cursor="500">{token}</resumptionToken>'
    return _ENVELOPE.format(body=f"<ListRecords>{records}{resumption}</ListRecords>")


def _error(code: str, message: str = "message") -> str:
    """Build an OAI-PMH error response.

    Every OAI-PMH fault arrives with HTTP 200, which is the trap this connector
    is written around.

    Parameters
    ----------
    code : str
        ``error code`` attribute.
    message : str
        Element text.

    Returns
    -------
    str
        Response XML.
    """
    return _ENVELOPE.format(body=f'<error code="{code}">{message}</error>')


def _response(text: str, status_code: int = 200) -> MagicMock:
    """Wrap XML *text* in a mock :class:`requests.Response`.

    Parameters
    ----------
    text : str
        Response body.
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
def connector(monkeypatch) -> IACRConnector:
    """Return an opted-in IACR connector.

    Opting in is the normal condition for these tests; the gate itself is
    exercised in :class:`TestTheWhitelistGate`.
    """
    monkeypatch.setenv(_OPT_IN_ENV_VAR, "1")
    return IACRConnector()


def _search(connector: IACRConnector, responses: list[Any], **kwargs: Any) -> CoveredPapers:
    """Run a search against a scripted sequence of transport outcomes.

    The patch is applied at ``_request_with_retry_uncounted`` -- the seam
    *inside* ``_request_with_retry`` -- so the retrieval audit still records
    every attempt and failure for real.

    Parameters
    ----------
    connector : IACRConnector
        Connector under test.
    responses : list[Any]
        Values or exceptions for successive transport calls.
    **kwargs : Any
        Forwarded to ``search``.

    Returns
    -------
    CoveredPapers
        The search outcome.
    """
    query = kwargs.pop("query", None) or _parse("[threshold signatures]")
    with (
        patch.object(connector, "_rate_limit"),
        patch.object(connector, "_request_with_retry_uncounted", side_effect=responses),
    ):
        return connector.search(query, **kwargs)


# ---------------------------------------------------------------------------
# Declared contract
# ---------------------------------------------------------------------------


class TestDeclaredContract:
    """The connector declares what the base classes require of it."""

    def test_name_is_the_registered_database_identifier(self, connector) -> None:
        """The ledger key matches the Database member."""
        assert connector.name == Database.IACR.value

    def test_min_request_interval_is_iacrs_published_crawl_delay(self, connector) -> None:
        """5.0s is ``Crawl-delay: 5`` from the archive's own robots.txt."""
        assert connector.min_request_interval == 5.0

    def test_registered_in_the_search_registry(self) -> None:
        """A connector nobody can select is not ported."""
        from lit_search.connectors import SEARCH_REGISTRY

        assert SEARCH_REGISTRY[Database.IACR] is IACRConnector

    def test_rate_limit_service_is_shared_across_iacr_routes(self, connector) -> None:
        """The scope names the host, not this one connector."""
        assert connector.rate_limit_service == "iacr"

    def test_the_opt_in_flag_is_not_part_of_the_rate_limit_identity(self, connector) -> None:
        """The acknowledgement is not a credential and must not split buckets.

        Including it would give an opted-in process a different clock from an
        opted-out one against the same host, which is the opposite of what the
        crawl delay is for.
        """
        assert connector._credential_attrs == ()
        assert connector.rate_limit_scope.fingerprint == "anon"


# ---------------------------------------------------------------------------
# The whitelist gate
# ---------------------------------------------------------------------------


class TestTheWhitelistGate:
    """The archive is not contacted until the operator says it may be."""

    def test_without_opt_in_the_status_is_skipped_no_credential(self, monkeypatch) -> None:
        """A source that was never asked must say so, not report zero."""
        monkeypatch.delenv(_OPT_IN_ENV_VAR, raising=False)
        conn = IACRConnector()
        result = _search(conn, [])
        assert result.status is CoverageStatus.SKIPPED_NO_CREDENTIAL
        assert result.papers == ()
        assert result.coverage.missing_credential == _OPT_IN_ENV_VAR

    def test_without_opt_in_no_request_is_issued(self, monkeypatch) -> None:
        """The refusal happens before any traffic reaches the archive."""
        monkeypatch.delenv(_OPT_IN_ENV_VAR, raising=False)
        conn = IACRConnector()
        transport = MagicMock()
        with (
            patch.object(conn, "_rate_limit"),
            patch.object(conn, "_request_with_retry_uncounted", transport),
        ):
            conn.search(_parse("[threshold signatures]"))
        transport.assert_not_called()

    def test_a_non_truthy_value_does_not_opt_in(self, monkeypatch) -> None:
        """An explicit "no" is still a no."""
        monkeypatch.setenv(_OPT_IN_ENV_VAR, "false")
        conn = IACRConnector()
        assert _search(conn, []).status is CoverageStatus.SKIPPED_NO_CREDENTIAL

    def test_skipped_is_distinct_from_both_empty_and_failed(self, connector, monkeypatch) -> None:
        """Three zero-paper outcomes, three different statuses.

        "We never asked", "we asked and it holds nothing" and "we asked and it
        broke" are different facts, and a ledger that collapses them cannot
        tell a reader which one happened.
        """
        monkeypatch.delenv(_OPT_IN_ENV_VAR, raising=False)
        skipped = _search(IACRConnector(), [])

        monkeypatch.setenv(_OPT_IN_ENV_VAR, "1")
        empty = _search(IACRConnector(), [_response(_error("noRecordsMatch", "No matching"))])
        failed = _search(IACRConnector(), [requests.ConnectionError("iacr unreachable")])

        statuses = {skipped.status, empty.status, failed.status}
        assert len(statuses) == 3
        assert skipped.papers == empty.papers == failed.papers == ()


# ---------------------------------------------------------------------------
# The required cases, plus the ones that matter
# ---------------------------------------------------------------------------


class TestCoverageIsHonest:
    """Each way a harvest can end reports a different, actionable status."""

    def test_a_normal_harvest_is_partial_never_complete(self, connector) -> None:
        """The archive cannot be queried, so completeness is never claimable.

        Retrieval here can be flawless and the answer still be a slice of the
        archive chosen by date.  ``COMPLETE`` would assert something the
        connector never asked the source about.
        """
        result = _search(connector, [_response(_list_records(_record(1) + _record(2)))])
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 2
        assert "no queryable endpoint" in (result.coverage.detail or "")

    def test_upstream_failure_is_failed(self, connector) -> None:
        """A dead first request is FAILED, and the error is carried."""
        result = _search(connector, [requests.ConnectionError("iacr unreachable")])
        assert result.status is CoverageStatus.FAILED
        assert result.papers == ()
        assert "unreachable" in (result.coverage.error or "")
        assert result.coverage.error_type == "ConnectionError"

    def test_no_records_match_is_an_empty_result_not_a_failure(self, connector) -> None:
        """OAI-PMH reports "nothing in this window" as an ``<error>``.

        Verified live 2026-08-31: a window in 2050 returns HTTP 200 with
        ``<error code="noRecordsMatch">No matching records</error>``.  Treating
        that as a failure would invent an outage; treating the *other* error
        codes as emptiness would invent an empty archive.  Only this code is an
        empty result.
        """
        result = _search(connector, [_response(_error("noRecordsMatch", "No matching records"))])
        assert result.status is CoverageStatus.PARTIAL
        assert result.papers == ()
        assert result.coverage.error is None

    @pytest.mark.parametrize(
        "code",
        ["badArgument", "cannotDisseminateFormat", "badVerb", "idDoesNotExist"],
    )
    def test_every_other_oai_error_is_failed_not_empty(self, connector, code) -> None:
        """Each arrives with HTTP 200 and no ``<record>`` elements.

        A naive ``findall(".//record")`` returns ``[]`` for all of them, which
        is how a malformed request comes to be reported as an empty archive.
        """
        result = _search(connector, [_response(_error(code, "broken"))])
        assert result.status is CoverageStatus.FAILED
        assert result.papers == ()
        assert code in (result.coverage.error or "")

    def test_failure_and_emptiness_are_different_statuses(self, connector) -> None:
        """The regression test for the whole port.

        In the donor implementation ``if response.status_code != 200: return
        papers``, ``if not results: return papers`` and a whole-body
        ``except Exception`` all reach the same empty list.  Here an outage and
        an empty window must differ.
        """
        failure = _search(connector, [requests.ConnectionError("boom")])
        empty = _search(connector, [_response(_error("noRecordsMatch", "No matching records"))])

        assert failure.papers == empty.papers == ()
        assert failure.status is not empty.status
        assert failure.status is CoverageStatus.FAILED
        assert empty.status is CoverageStatus.PARTIAL
        assert failure.coverage.error is not None
        assert empty.coverage.error is None

    def test_malformed_xml_is_failed_not_empty(self, connector) -> None:
        """A truncated or non-XML body is an outage, not an empty archive."""
        result = _search(connector, [_response("<OAI-PMH><unclosed>")])
        assert result.status is CoverageStatus.FAILED
        assert "not well-formed XML" in (result.coverage.error or "")

    def test_a_capped_harvest_stays_partial(self, connector) -> None:
        """``CAPPED`` ranks milder than ``PARTIAL`` and must not apply here.

        Whatever the cap did, everything outside the harvest window was never
        consulted, so an unknown amount is missing.  Reporting ``CAPPED`` would
        make a truncated harvest read as healthier than an untruncated one.
        """
        records = "".join(_record(i) for i in range(5))
        result = _search(connector, [_response(_list_records(records))], max_papers=2)
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 2
        assert "caller's limit of 2" in (result.coverage.detail or "")


# ---------------------------------------------------------------------------
# Harvesting
# ---------------------------------------------------------------------------


class TestHarvest:
    """Resumption, windows, and what the counters mean."""

    def test_a_resumption_token_is_followed(self, connector) -> None:
        """Two rounds are harvested and both contribute papers."""
        first = _list_records(_record(1), token="tok-1", complete_size=2)
        second = _list_records(_record(2))
        result = _search(connector, [_response(first), _response(second)])
        assert len(result.papers) == 2
        assert "the 2 the archive reports for that window" in (result.coverage.detail or "")

    def test_complete_list_size_is_reported_as_window_context_only(self, connector) -> None:
        """``completeListSize`` counts the window, never the query.

        It must not become a ``reported_total``: matching 1 of 500 harvested
        records is a correct answer, not a 499-record shortfall, and feeding it
        to the completeness arithmetic would say the opposite.
        """
        records = _record(1) + _record(2, title="Unrelated work on error codes")
        result = _search(
            connector,
            [
                _response(_list_records(records, token="tok", complete_size=500)),
                _response(_list_records("")),
            ],
        )
        assert result.coverage.reported_total is None
        assert len(result.papers) == 1
        assert "matched 1 against the query locally" in (result.coverage.detail or "")

    def test_mid_harvest_failure_keeps_records_and_reports_partial(self, connector) -> None:
        """A failed second round keeps the first round's papers."""
        first = _list_records(_record(1), token="tok-1", complete_size=900)
        result = _search(connector, [_response(first), requests.ConnectionError("died")])
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 1
        assert result.coverage.error_type == "ConnectionError"
        assert "harvest failed at resumption round" in (result.coverage.detail or "")

    def test_the_default_window_is_named_when_no_start_date_is_given(self, connector) -> None:
        """An unbounded search consults a recent window and says which."""
        result = _search(connector, [_response(_list_records(_record(1)))])
        detail = result.coverage.detail or ""
        assert "default 365-day window" in detail
        assert "the archive runs from 1996" in detail

    def test_a_caller_supplied_window_is_reported_verbatim(self, connector) -> None:
        """The window actually consulted is stated, not implied."""
        result = _search(
            connector,
            [_response(_list_records(_record(1)))],
            since=datetime.date(2024, 1, 1),
            until=datetime.date(2024, 6, 30),
        )
        detail = result.coverage.detail or ""
        assert "between 2024-01-01 and 2024-06-30" in detail
        assert "default" not in detail

    def test_the_datestamp_caveat_is_always_stated(self, connector) -> None:
        """OAI-PMH filters on last-modified, not on first publication.

        A 2019 paper revised last month falls in last month's window and a 2019
        paper never revised does not, so the window is not the date range a
        reader would assume.
        """
        result = _search(connector, [_response(_list_records(_record(1)))])
        assert "last-modified date" in (result.coverage.detail or "")

    def test_a_deleted_record_is_counted_but_is_not_a_paper(self, connector) -> None:
        """Deleted records carry a header and no metadata."""
        deleted = (
            '<record><header status="deleted">'
            "<identifier>oai:eprint.iacr.org:2020/1</identifier>"
            "<datestamp>2026-01-01T00:00:00Z</datestamp></header></record>"
        )
        result = _search(connector, [_response(_list_records(_record(1) + deleted))])
        assert len(result.papers) == 1
        assert "1 of 2 harvested records" in (result.coverage.detail or "")

    def test_the_window_start_is_sent_at_the_declared_granularity(self, connector) -> None:
        """``Identify`` declares ``YYYY-MM-DDThh:mm:ssZ`` granularity."""
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(connector, "_get", return_value=_response(_list_records(""))) as get,
        ):
            connector.search(_parse("[threshold signatures]"), since=datetime.date(2024, 3, 1))
        params = get.call_args.args[1]
        assert params["from"] == "2024-03-01T00:00:00Z"
        assert params["metadataPrefix"] == "oai_dc"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    """``oai_dc`` field mapping."""

    def test_fields_are_mapped(self, connector) -> None:
        """A complete record becomes a populated paper."""
        result = _search(connector, [_response(_list_records(_record(1)))])
        paper = result.papers[0]
        assert paper.title == "Threshold Signatures from Lattices"
        assert paper.abstract.startswith("We present a lattice-based")
        assert [a.name for a in paper.authors] == ["Alice Cryptographer"]
        assert paper.url == "https://eprint.iacr.org/2026/1"
        assert paper.pdf_url == "https://eprint.iacr.org/2026/1.pdf"
        assert paper.keywords == {"Public-key cryptography"}
        assert paper.publication_date == datetime.date(2026, 8, 5)
        assert paper.paper_type is PaperType.UNPUBLISHED
        assert paper.is_open_access is True
        assert paper.source is not None
        assert paper.source.source_type is SourceType.REPOSITORY
        assert paper.found_in == {"iacr"}

    def test_a_record_without_a_title_is_counted_as_unparsed(self, connector) -> None:
        """An unreadable record is a coverage loss, not an absent paper."""
        broken = (
            "<record><header><identifier>x</identifier></header><metadata>"
            '<oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"'
            ' xmlns:dc="http://purl.org/dc/elements/1.1/">'
            "<dc:description>No title here.</dc:description>"
            "</oai_dc:dc></metadata></record>"
        )
        result = _search(connector, [_response(_list_records(_record(1) + broken))])
        assert len(result.papers) == 1
        assert "1 of 2 harvested records" in (result.coverage.detail or "")


# ---------------------------------------------------------------------------
# The local predicate
# ---------------------------------------------------------------------------


class TestLocalMatching:
    """The query is applied here, so it must be applied exactly."""

    def test_a_non_matching_record_is_harvested_but_not_returned(self, connector) -> None:
        """Harvest counts records; the result counts matches."""
        records = _record(1) + _record(
            2, title="Side channels", description="Power analysis.", subject="Implementation"
        )
        result = _search(connector, [_response(_list_records(records))])
        assert len(result.papers) == 1
        assert "harvested 2 records" in (result.coverage.detail or "")
        assert "matched 1 against the query locally" in (result.coverage.detail or "")

    def test_and_not_is_supported_because_matching_is_local(self, connector) -> None:
        """Nothing is sent upstream, so the full grammar is answerable.

        Worth pinning as a contrast with dblp, where ``AND NOT`` is refused
        because the remote parser silently turns it into ``AND``.
        """
        records = _record(1) + _record(
            2,
            title="Threshold signatures from isogenies",
            description="An isogeny construction.",
        )
        result = _search(
            connector,
            [_response(_list_records(records))],
            query=_parse("[threshold] AND NOT [isogenies]"),
        )
        assert len(result.papers) == 1
        assert result.papers[0].title == "Threshold Signatures from Lattices"

    def test_or_is_supported(self, connector) -> None:
        """Either branch matching is enough."""
        records = _record(1) + _record(
            2, title="Side channels", description="Power analysis.", subject="Implementation"
        )
        result = _search(
            connector,
            [_response(_list_records(records))],
            query=_parse("[threshold] OR [power analysis]"),
        )
        assert len(result.papers) == 2

    def test_matching_is_case_insensitive(self, connector) -> None:
        """Query casing must not change recall."""
        result = _search(
            connector,
            [_response(_list_records(_record(1)))],
            query=_parse("[THRESHOLD SIGNATURES]"),
        )
        assert len(result.papers) == 1

    def test_a_term_does_not_match_inside_a_longer_word(self, connector) -> None:
        """Word boundaries keep ``[key]`` from matching "monkey".

        The subject is overridden here because the default fixture category is
        "Public-key cryptography", where ``key`` *does* sit on a word boundary
        after the hyphen -- a real match, and a reminder that the default
        scope covers ``dc:subject`` as well as title and abstract.
        """
        record = _record(
            1,
            title="Monkey testing",
            description="Nothing cryptographic.",
            subject="Implementation",
        )
        result = _search(connector, [_response(_list_records(record))], query=_parse("[key]"))
        assert result.papers == ()

    def test_a_wildcard_crosses_the_boundary_deliberately(self, connector) -> None:
        """``*`` is what a caller uses to ask for the wider match."""
        record = _record(
            1,
            title="Monkey testing",
            description="Nothing cryptographic.",
            subject="Implementation",
        )
        result = _search(connector, [_response(_list_records(record))], query=_parse("[monk*]"))
        assert len(result.papers) == 1

    def test_a_title_filter_does_not_match_the_abstract(self, connector) -> None:
        """Field scoping is honoured rather than approximated."""
        record = _record(1, title="Side channels", description="A threshold scheme.")
        result = _search(
            connector, [_response(_list_records(record))], query=_parse("ti[threshold]")
        )
        assert result.papers == ()

    def test_an_abstract_filter_matches_the_description(self, connector) -> None:
        """``dc:description`` is the abstract for this repository."""
        record = _record(1, title="Side channels", description="A threshold scheme.")
        result = _search(
            connector, [_response(_list_records(record))], query=_parse("abs[threshold]")
        )
        assert len(result.papers) == 1


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------


class TestQueryBuilder:
    """Only filters ``oai_dc`` can answer are accepted."""

    @pytest.fixture
    def builder(self) -> IACRQueryBuilder:
        """Return an IACR query builder."""
        return IACRQueryBuilder()

    def test_venue_and_affiliation_filters_are_rejected(self, builder) -> None:
        """``oai_dc`` carries neither for this repository."""
        assert not builder.validate_query(_parse("src[crypto]")).is_valid
        assert not builder.validate_query(_parse("aff[mit]")).is_valid

    def test_supported_filters_validate(self, builder) -> None:
        """Title, abstract, keywords and author all map to a dc element."""
        for query in ("ti[a]", "abs[a]", "key[a]", "au[a]", "tiabs[a]", "[a]"):
            assert builder.validate_query(_parse(query)).is_valid, query

    def test_convert_query_renders_the_local_expression(self, builder) -> None:
        """The rendering records what the predicate tested, for provenance."""
        rendered = builder.convert_query(_parse("ti[alpha] AND NOT abs[beta]"))
        assert rendered == 'title:"alpha" AND NOT description:"beta"'

    def test_a_query_unsupported_result_issues_no_request(self, connector) -> None:
        """Refusal happens before the archive is contacted."""
        transport = MagicMock()
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(connector, "_request_with_retry_uncounted", transport),
        ):
            result = connector.search(_parse("src[crypto]"))
        assert result.status is CoverageStatus.QUERY_UNSUPPORTED
        transport.assert_not_called()


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class TestOAIError:
    """The error carries the code so callers can branch on it."""

    def test_the_oai_code_is_preserved(self) -> None:
        """``IACROAIError.code`` is what distinguishes our bug from theirs."""
        error = IACROAIError("boom", code="badArgument")
        assert error.code == "badArgument"
