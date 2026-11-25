"""Unit tests for ZoteroConnector.

These run under ``pytest-socket`` with ``--disable-socket`` (see
``pyproject.toml``), so every HTTP call is mocked and no test can reach the
network.  Payload shapes follow the Zotero Web API v3 specification, which the
local API implements; Zotero desktop was not reachable from this machine when
the connector was written, so no fixture here was captured from a live
response.  That is stated in the connector's module docstring too, because it
bounds what these tests prove.

Three tests carry the most weight:

- :meth:`TestCoverageIsHonest.test_failure_and_emptiness_are_different_statuses`,
  the regression test for the connector as a whole;
- :meth:`TestCoverageIsHonest.test_permission_off_and_app_closed_are_different_statuses`,
  which pins the split the module docstring promises -- the preference being
  off is fixable configuration, the app being closed may be transient, and
  collapsing them loses the only actionable part of the message; and
- :meth:`TestExcludeLibraryRead.test_absent_total_refuses_to_claim_complete`,
  because a false ``COMPLETE`` on the subtract-set is what silently marks
  already-held papers as new.
"""

from __future__ import annotations

import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from lit_search.connectors.zotero import ZoteroConnector, ZoteroLocalApiError
from lit_search.core.coverage import CoverageStatus, CoveredPapers
from lit_search.core.paper import Database, PaperType
from lit_search.core.source import SourceType
from lit_search.exceptions import MissingApiKeyError
from lit_search.query.builders.zotero import ZoteroQueryBuilder

from .conftest import _parse

# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def _item(
    key: str = "ABCD1234",
    *,
    title: str = "Attention Is All You Need",
    abstract: str = "We propose the Transformer, a model based solely on attention.",
    item_type: str = "conferencePaper",
    creators: tuple[str, ...] = ("Ashish Vaswani",),
    tags: tuple[str, ...] = ("transformers",),
    date: str = "2017-06-12",
    doi: str = "10.5555/3295222.3295349",
    venue: str = "Advances in Neural Information Processing Systems",
) -> dict[str, Any]:
    """Build one Zotero item envelope.

    Parameters
    ----------
    key : str
        Item key.
    title : str
        Item title.
    abstract : str
        ``abstractNote`` value.
    item_type : str
        Zotero item type.
    creators : tuple[str, ...]
        Author names, split into first and last on the last space.
    tags : tuple[str, ...]
        Tag names.
    date : str
        ``date`` field value.
    doi : str
        ``DOI`` field value.
    venue : str
        ``proceedingsTitle`` value.

    Returns
    -------
    dict[str, Any]
        Item in the shape ``include=data`` returns.
    """
    people = []
    for name in creators:
        first, _, last = name.rpartition(" ")
        people.append({"creatorType": "author", "firstName": first, "lastName": last})
    return {
        "key": key,
        "version": 1,
        "library": {"type": "user", "id": 0},
        "data": {
            "key": key,
            "itemType": item_type,
            "title": title,
            "abstractNote": abstract,
            "creators": people,
            "proceedingsTitle": venue,
            "date": date,
            "DOI": doi,
            "url": f"https://example.org/{key}",
            "language": "en",
            "tags": [{"tag": tag} for tag in tags],
        },
    }


def _note_item(key: str = "NOTE0001") -> dict[str, Any]:
    """Build a standalone note, which is not a bibliographic record.

    Parameters
    ----------
    key : str
        Item key.

    Returns
    -------
    dict[str, Any]
        Item whose ``itemType`` is ``note``.
    """
    return {"key": key, "data": {"key": key, "itemType": "note", "note": "<p>a thought</p>"}}


def _response(
    items: list[dict[str, Any]] | dict[str, Any] | str,
    *,
    total: int | None = None,
    status_code: int = 200,
) -> MagicMock:
    """Build a mock local-API response.

    Parameters
    ----------
    items : list[dict[str, Any]] | dict[str, Any] | str
        Decoded body, or a string to make ``json()`` raise.
    total : int | None
        ``Total-Results`` header value; omitted when ``None``.
    status_code : int
        HTTP status.

    Returns
    -------
    MagicMock
        Response double.
    """
    mock = MagicMock(spec=requests.Response)
    mock.status_code = status_code
    mock.headers = {} if total is None else {"Total-Results": str(total)}
    if isinstance(items, str):
        mock.json.side_effect = ValueError(items)
    else:
        mock.json.return_value = items
    return mock


def _http_error(status: int) -> requests.HTTPError:
    """Build an ``HTTPError`` carrying a response with *status*.

    Parameters
    ----------
    status : int
        HTTP status code.

    Returns
    -------
    requests.HTTPError
        Error the base class would raise after exhausting retries.
    """
    response = MagicMock(spec=requests.Response)
    response.status_code = status
    return requests.HTTPError(f"{status} Client Error", response=response)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def connector() -> ZoteroConnector:
    """Return a connector pointed at the default loopback address."""
    return ZoteroConnector()


def _run(
    connector: ZoteroConnector,
    responses: list[Any],
    *,
    query: str = "[transformer]",
    max_papers: int | None = None,
    since: datetime.date | None = None,
    until: datetime.date | None = None,
) -> CoveredPapers:
    """Run a search against a scripted sequence of ``_get`` outcomes.

    Parameters
    ----------
    connector : ZoteroConnector
        Connector under test.
    responses : list[Any]
        Responses or exceptions, in order.
    query : str
        Query string to parse.
    max_papers : int | None
        Caller-imposed cap.
    since : datetime.date | None
        Lower publication-date bound.
    until : datetime.date | None
        Upper publication-date bound.

    Returns
    -------
    CoveredPapers
        What the connector returned.
    """
    with (
        patch.object(connector, "_rate_limit"),
        patch.object(connector, "_get", side_effect=responses),
    ):
        return connector.search(_parse(query), max_papers, None, since, until)


def _library(connector: ZoteroConnector, responses: list[Any], **kwargs: Any) -> CoveredPapers:
    """Run :meth:`ZoteroConnector.fetch_library` against scripted responses.

    Parameters
    ----------
    connector : ZoteroConnector
        Connector under test.
    responses : list[Any]
        Responses or exceptions, in order.
    **kwargs : Any
        Forwarded to ``fetch_library``.

    Returns
    -------
    CoveredPapers
        What the connector returned.
    """
    with (
        patch.object(connector, "_rate_limit"),
        patch.object(connector, "_get", side_effect=responses),
    ):
        return connector.fetch_library(**kwargs)


# ---------------------------------------------------------------------------
# The tests that matter most
# ---------------------------------------------------------------------------


class TestCoverageIsHonest:
    """An empty Zotero result must never be how a failure is reported."""

    def test_failure_and_emptiness_are_different_statuses(self, connector) -> None:
        """The regression test for the connector as a whole.

        A connector written the way the donor connectors in this project's port
        list are written would return ``[]`` for both of these, and a caller
        could not tell "Zotero is closed" from "your library holds nothing
        matching".  On the subtract-set that difference is the whole ballgame:
        an empty library marks every already-held paper as new.
        """
        failure = _run(connector, [requests.ConnectionError("connection refused")])
        empty = _run(connector, [_response([], total=0)])

        assert failure.papers == empty.papers == ()
        assert failure.status is not empty.status
        assert failure.status is CoverageStatus.FAILED
        assert empty.status is CoverageStatus.PARTIAL
        assert "could not connect" in (failure.coverage.error or "")

    def test_permission_off_and_app_closed_are_different_statuses(self, connector) -> None:
        """403 is configuration; a refused connection may be transient.

        Both mean no records, and a connector that collapsed them would throw
        away the only actionable half of the message -- one is fixed by a
        checkbox, the other by launching an app.
        """
        forbidden = _run(connector, [_http_error(403)])
        closed = _run(connector, [requests.ConnectionError("connection refused")])

        assert forbidden.status is CoverageStatus.SKIPPED_NO_CREDENTIAL
        assert closed.status is CoverageStatus.FAILED
        assert "Allow other applications" in (forbidden.coverage.missing_credential or "")

    def test_a_search_is_never_complete(self, connector) -> None:
        """Local matching cannot justify ``COMPLETE``, however well it went.

        Zotero reported three items, three arrived, one matched.  The counters
        balance, and the claim is still unavailable: Zotero was never asked the
        query, so its item count says nothing about how many records match it.
        """
        result = _run(
            connector,
            [
                _response(
                    [_item("A"), _item("B", title="Vision"), _item("C", title="Speech")], total=3
                )
            ],
        )
        assert result.status is CoverageStatus.PARTIAL
        assert "matched the query locally" in (result.coverage.detail or "")

    def test_cap_is_reported_as_partial_not_capped(self, connector) -> None:
        """A truncated local match must not read as milder than an untruncated one.

        ``CAPPED`` means a known amount is missing on purpose and ranks below
        ``PARTIAL`` in severity.  Here an unknown amount is missing whatever the
        cap did, so reporting ``CAPPED`` would make the worse run look better.
        """
        items = [_item("A"), _item("B"), _item("C")]
        result = _run(connector, [_response(items, total=3)], max_papers=1)

        assert len(result.papers) == 1
        assert result.status is CoverageStatus.PARTIAL
        assert "limit of 1" in (result.coverage.detail or "")


# ---------------------------------------------------------------------------
# HTTP outcomes that are not exceptions
# ---------------------------------------------------------------------------


class TestUnusableBodies:
    """A 200 with the wrong shape must not decode to an empty library."""

    def test_non_json_body_is_failed(self, connector) -> None:
        """A body that is not JSON is a failure, not an empty library."""
        result = _run(connector, [_response("not json")])
        assert result.status is CoverageStatus.FAILED
        assert "non-JSON body" in (result.coverage.error or "")

    def test_json_object_instead_of_array_is_failed(self, connector) -> None:
        """The local API answers a mistyped path with an object, not an array."""
        result = _run(connector, [_response({"message": "No endpoint found"})])
        assert result.status is CoverageStatus.FAILED
        assert "array of items was expected" in (result.coverage.error or "")

    def test_unknown_library_prefix_is_failed(self, connector) -> None:
        """A 404 names a configuration problem rather than an empty library."""
        result = _run(connector, [_http_error(404)])
        assert result.status is CoverageStatus.FAILED
        assert "HTTP 404" in (result.coverage.error or "")

    def test_wrong_user_id_is_failed(self, connector) -> None:
        """The docs say any user id but the logged-in one returns 400."""
        result = _run(connector, [_http_error(400)])
        assert result.status is CoverageStatus.FAILED
        assert "HTTP 400" in (result.coverage.error or "")


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class TestPagination:
    """Termination is decided from the reported total, never from a short page."""

    def test_pages_until_the_reported_total_is_reached(self, connector) -> None:
        """Two pages of two are walked because the total says four exist."""
        page_one = _response([_item("A"), _item("B")], total=4)
        page_two = _response([_item("C"), _item("D")], total=4)
        result = _run(connector, [page_one, page_two])

        assert len(result.papers) == 4
        assert "saw 4 top-level items of the 4 Zotero reported" in (result.coverage.detail or "")

    def test_mid_pagination_failure_keeps_the_records_and_reports_partial(self, connector) -> None:
        """Records that did arrive are real; discarding them is its own dishonesty."""
        page_one = _response([_item("A"), _item("B")], total=6)
        result = _run(connector, [page_one, requests.ConnectionError("dropped")])

        assert len(result.papers) == 2
        assert result.status is CoverageStatus.PARTIAL
        assert "start=2" in (result.coverage.detail or "")

    def test_short_page_is_not_treated_as_the_end_of_the_set(self, connector) -> None:
        """A short page must not end the walk while the total says more exist.

        The connector asks for 100 items a page.  A page of one, with a total of
        two, is exactly the shape that a ``len(page) < page_size: break``
        heuristic would stop on, losing the second record silently.
        """
        result = _run(
            connector, [_response([_item("A")], total=2), _response([_item("B")], total=2)]
        )
        assert len(result.papers) == 2

    def test_early_exhaustion_is_reported(self, connector) -> None:
        """Running out of items before the total is a shortfall, not closure."""
        result = _run(connector, [_response([_item("A")], total=9), _response([], total=9)])
        assert result.status is CoverageStatus.PARTIAL
        assert "stopped returning items after 1 of the 9" in (result.coverage.detail or "")


# ---------------------------------------------------------------------------
# The unfiltered read the exclude-my-library operation needs
# ---------------------------------------------------------------------------


class TestExcludeLibraryRead:
    """``fetch_library`` is the one path that may claim ``COMPLETE``."""

    def test_full_read_is_complete(self, connector) -> None:
        """Every item Zotero said existed arrived, and no query was answered."""
        result = _library(connector, [_response([_item("A"), _item("B")], total=2)])

        assert result.status is CoverageStatus.COMPLETE
        assert result.coverage.reported_total == 2
        assert len(result.papers) == 2

    def test_absent_total_refuses_to_claim_complete(self, connector) -> None:
        """No ``Total-Results`` means completeness cannot be checked.

        The local API documentation does not mention the header, so its absence
        is a live possibility rather than a hypothetical.  Assuming completeness
        from the connector's own count is exactly the "fed its own count back as
        the source total" attack the coverage contract was written against.
        """
        result = _library(connector, [_response([_item("A"), _item("B")]), _response([])])

        assert result.status is CoverageStatus.PARTIAL
        assert "reported no result total" in (result.coverage.detail or "")

    def test_skipped_notes_are_a_coverage_loss_not_a_smaller_library(self, connector) -> None:
        """An item seen and not returned is reported, never quietly dropped."""
        result = _library(connector, [_response([_item("A"), _note_item()], total=2)])

        assert len(result.papers) == 1
        assert result.status is CoverageStatus.PARTIAL
        assert "standalone notes or annotations" in (result.coverage.detail or "")

    def test_total_drift_refuses_to_claim_complete(self, connector) -> None:
        """A library edited mid-read cannot be checked against either total."""
        result = _library(
            connector, [_response([_item("A")], total=2), _response([_item("B")], total=3)]
        )
        assert result.status is CoverageStatus.PARTIAL
        assert "changed mid-read" in (result.coverage.detail or "")

    def test_empty_library_is_complete_not_failed(self, connector) -> None:
        """An empty library is a real answer when Zotero says the total is zero."""
        result = _library(connector, [_response([], total=0)])
        assert result.status is CoverageStatus.COMPLETE
        assert result.papers == ()

    def test_closed_app_is_failed_not_an_empty_library(self, connector) -> None:
        """The subtract-set must never silently become empty."""
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(connector, "_get", side_effect=requests.ConnectionError("refused")),
            pytest.raises(ZoteroLocalApiError, match="could not connect"),
        ):
            connector.fetch_library()

    def test_permission_off_raises_for_the_caller_to_classify(self, connector) -> None:
        """``fetch_library`` has no base class to translate for it, so it raises."""
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(connector, "_get", side_effect=_http_error(403)),
            pytest.raises(MissingApiKeyError),
        ):
            connector.fetch_library()


# ---------------------------------------------------------------------------
# Query handling
# ---------------------------------------------------------------------------


class TestQueryIsAnsweredExactly:
    """The reason the connector does not use Zotero's own ``q`` parameter."""

    def test_and_not_is_honoured(self, connector) -> None:
        """Quicksearch has no negation; the local predicate does."""
        items = [
            _item("A", title="Transformer models", abstract="text"),
            _item("B", title="Transformer for vision", abstract="vision transformer"),
        ]
        result = _run(
            connector, [_response(items, total=2)], query="[transformer] AND NOT [vision]"
        )

        assert [paper.title for paper in result.papers] == ["Transformer models"]

    def test_source_filter_is_answerable(self, connector) -> None:
        """Zotero stores the venue, so ``src`` is a real filter here."""
        items = [
            _item("A", venue="Advances in Neural Information Processing Systems"),
            _item("B", title="Other", venue="Journal of Irreproducible Results"),
        ]
        result = _run(connector, [_response(items, total=2)], query="src[Neural Information]")

        assert [paper.title for paper in result.papers] == ["Attention Is All You Need"]

    def test_affiliation_query_is_rejected_without_a_request(self, connector) -> None:
        """A filter the records cannot answer is refused, not silently widened."""
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(connector, "_get") as mocked,
        ):
            result = connector.search(_parse("aff[mit]"))

        assert result.status is CoverageStatus.QUERY_UNSUPPORTED
        assert "affiliation" in (result.coverage.detail or "")
        mocked.assert_not_called()

    def test_date_bounds_keep_undated_records(self, connector) -> None:
        """Zotero's date field is free text and often blank or unparseable."""
        items = [_item("A", date="2017-06-12"), _item("B", date="")]
        result = _run(
            connector,
            [_response(items, total=2)],
            since=datetime.date(2020, 1, 1),
        )
        assert [paper.title for paper in result.papers] == ["Attention Is All You Need"]


# ---------------------------------------------------------------------------
# Field mapping and configuration
# ---------------------------------------------------------------------------


class TestParsing:
    """Mapping a Zotero item onto the project's record shape."""

    def test_fields_are_mapped(self, connector) -> None:
        """Title, authors, venue, DOI, tags and type all survive the trip."""
        result = _run(connector, [_response([_item()], total=1)])
        paper = result.papers[0]

        assert paper.title == "Attention Is All You Need"
        assert [author.name for author in paper.authors] == ["Ashish Vaswani"]
        assert paper.source is not None
        assert paper.source.title == "Advances in Neural Information Processing Systems"
        assert paper.source.source_type is SourceType.CONFERENCE
        assert paper.doi == "10.5555/3295222.3295349"
        assert paper.keywords == {"transformers"}
        assert paper.paper_type is PaperType.INPROCEEDINGS
        assert paper.publication_date == datetime.date(2017, 6, 12)
        assert paper.found_in == {Database.ZOTERO.value}

    def test_untitled_item_is_counted_as_a_loss(self, connector) -> None:
        """An item that cannot be parsed is reported, not quietly skipped."""
        result = _run(connector, [_response([_item("A"), _item("B", title="")], total=2)])

        assert len(result.papers) == 1
        assert "carried no title or failed model validation" in (result.coverage.detail or "")


class TestConfiguration:
    """Refusals that stop a misconfiguration becoming a plausible answer."""

    def test_non_loopback_base_url_is_refused(self) -> None:
        """Pointed at the public API this would read a 403 as a local problem."""
        with pytest.raises(ValueError, match="loopback"):
            ZoteroConnector(base_url="https://api.zotero.org")

    def test_rate_limit_is_declared_positive(self, connector) -> None:
        """``0.0`` silently disables the limiter and its announcement.

        There is no remote quota here, so ``0.0`` is the natural thing to write
        -- and it is the value that makes ``reserve_slot`` return before it logs
        the active limit, which the specification's watchlist calls out as a
        silent failure.  This test is the guard on that.
        """
        assert connector.min_request_interval > 0
        assert connector.rate_limit_service == "zotero_local"

    def test_builder_is_the_zotero_one(self, connector) -> None:
        """The connector must not inherit another source's field mapping."""
        assert isinstance(connector.query_builder, ZoteroQueryBuilder)
