"""Unit tests for OpenReviewConnector.

These run under ``pytest-socket`` with ``--disable-socket`` (see
``pyproject.toml``), so no test reaches the network: the ``openreview-py``
client class is replaced by a double, which also keeps the suite from depending
on the package being installed.

Payload shapes and venue ids mirror what was observed live on 2026-08-31
against ``https://api2.openreview.net`` -- the ``venues`` group listing, the
``ratelimit-policy: 180;w=60`` headers, and the anonymous ``403
ChallengeRequiredError`` on ``/notes``.

Three tests carry the most weight:

- :meth:`TestCoverageIsHonest.test_failure_and_emptiness_are_different_statuses`,
  the regression test for the connector as a whole;
- :meth:`TestAcceptedPapersOnly.test_papers_are_requested_by_venueid`, which
  pins the one thing this connector exists to get right.  Retrieving by
  invitation instead mixes rejected submissions into the result and nothing in
  the output reveals it; and
- :meth:`TestUnknownVenues.test_a_venue_openreview_does_not_host_is_not_queried`,
  because a mistyped or non-existent venue id returns zero notes and no error,
  which reads as a conference that accepted nothing.
"""

from __future__ import annotations

import datetime
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import pytest
import requests

from lit_search.connectors import openreview as openreview_module
from lit_search.connectors.openreview import (
    _BASE_URL,
    _PASSWORD_ENV,
    _USERNAME_ENV,
    OpenReviewApiError,
    OpenReviewConnector,
    _PacedAdapter,
)
from lit_search.core.coverage import CoverageStatus, CoveredPapers
from lit_search.core.paper import Database, PaperType
from lit_search.core.source import SourceType
from lit_search.query.builders.openreview import OpenReviewQueryBuilder

from .conftest import _parse

_VENUE = "ICLR.cc/2025/Conference"
_OTHER_VENUE = "NeurIPS.cc/2024/Conference"

# Venue ids read from the live ``venues`` group on 2026-08-31.  ICML has no
# 2021 or 2022 Conference venue, which is why the unknown-venue path is not a
# hypothetical.
_KNOWN_VENUES = (
    _VENUE,
    _OTHER_VENUE,
    "ICLR.cc/2024/Conference",
    "ICML.cc/2025/Conference",
    "NeurIPS.cc/2025/Conference",
)


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


def _note(
    note_id: str = "abc123",
    *,
    title: str = "Scaling Laws for Neural Language Models",
    abstract: str = "We study empirical scaling laws for language model performance.",
    keywords: tuple[str, ...] = ("scaling", "language models"),
    authors: tuple[str, ...] = ("Jared Kaplan",),
    venue: str = "ICLR 2025 poster",
    pdate: int | None = 1_735_689_600_000,
) -> dict[str, Any]:
    """Build one API v2 note payload in ``Note.to_json`` shape.

    Parameters
    ----------
    note_id : str
        Note id, which is also the forum id in the paper URL.
    title : str
        ``content.title``.
    abstract : str
        ``content.abstract``.
    keywords : tuple[str, ...]
        ``content.keywords``.
    authors : tuple[str, ...]
        ``content.authors``.
    venue : str
        ``content.venue``, the human-readable decision label.
    pdate : int | None
        Publication timestamp in epoch milliseconds.

    Returns
    -------
    dict[str, Any]
        Note payload with API v2's ``{"value": ...}`` wrapping.
    """
    payload: dict[str, Any] = {
        "id": note_id,
        "content": {
            "title": {"value": title},
            "abstract": {"value": abstract},
            "keywords": {"value": list(keywords)},
            "authors": {"value": list(authors)},
            "venue": {"value": venue},
            "pdf": {"value": f"/pdf/{note_id}.pdf"},
        },
        "cdate": 1_700_000_000_000,
    }
    if pdate is not None:
        payload["pdate"] = pdate
    return payload


class _FakeNote:
    """Stand-in for ``openreview.api.Note`` carrying only ``to_json``.

    Parameters
    ----------
    payload : dict[str, Any]
        The dictionary ``to_json`` should return.
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        """Store the payload."""
        self._payload = payload

    def to_json(self) -> dict[str, Any]:
        """Return the note payload.

        Returns
        -------
        dict[str, Any]
            The payload this note was built from.
        """
        return self._payload


class _FakeSession:
    """Minimal :class:`requests.Session` stand-in that records mounts."""

    def __init__(self) -> None:
        """Start with the adapters ``openreview-py`` mounts itself."""
        self.adapters: dict[str, Any] = {
            "https://": MagicMock(max_retries="original-retry-policy"),
            "http://": MagicMock(max_retries="original-retry-policy"),
        }

    def mount(self, prefix: str, adapter: Any) -> None:
        """Record an adapter mount.

        Parameters
        ----------
        prefix : str
            URL prefix.
        adapter : Any
            Adapter mounted for it.
        """
        self.adapters[prefix] = adapter


class _FakeClient:
    """Stand-in for ``openreview.api.OpenReviewClient``.

    Parameters
    ----------
    baseurl : str | None
        Base URL the connector passed.
    username : str | None
        Username the connector passed.
    password : str | None
        Password the connector passed.
    """

    #: Notes returned per venue id.
    notes: ClassVar[dict[str, list[dict[str, Any]]]] = {}
    #: Counts returned per venue id; a venue absent here reports no count.
    counts: ClassVar[dict[str, int]] = {}
    #: Exceptions raised per venue id instead of returning notes.
    errors: ClassVar[dict[str, Exception]] = {}
    #: Exceptions raised by the count probe alone, leaving the notes reachable.
    count_errors: ClassVar[dict[str, Exception]] = {}
    #: Venue ids the ``venues`` group lists, or ``None`` to fail the lookup.
    venues: ClassVar[tuple[str, ...] | None] = _KNOWN_VENUES
    #: Every constructed instance, so tests can assert on the arguments.
    instances: ClassVar[list[_FakeClient]] = []

    def __init__(
        self,
        baseurl: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        """Record the constructor arguments and start a call log."""
        self.baseurl = baseurl
        self.username = username
        self.password = password
        self.session = _FakeSession()
        self.calls: list[tuple[str, Any]] = []
        type(self).instances.append(self)

    def get_group(self, group_id: str) -> Any:
        """Return the venues group, or fail when the listing is unavailable.

        Parameters
        ----------
        group_id : str
            Group id requested.

        Returns
        -------
        Any
            Object carrying a ``members`` list.

        Raises
        ------
        RuntimeError
            When :attr:`venues` is ``None``.
        """
        self.calls.append(("get_group", group_id))
        venues = type(self).venues
        if venues is None:
            raise RuntimeError("group listing unavailable")
        return MagicMock(members=list(venues))

    def get_notes(
        self,
        content: dict[str, str] | None = None,
        limit: int | None = None,
        with_count: bool | None = None,
    ) -> Any:
        """Return a count probe for one venue.

        Parameters
        ----------
        content : dict[str, str] | None
            Content filter.
        limit : int | None
            Page size.
        with_count : bool | None
            Whether the count is wanted.

        Returns
        -------
        Any
            ``(notes, count)`` when a count is configured, else a bare list.

        Raises
        ------
        Exception
            The exception configured for this venue.
        """
        venue = (content or {}).get("venueid", "")
        self.calls.append(("get_notes", venue))
        if venue in type(self).count_errors:
            raise type(self).count_errors[venue]
        if venue in type(self).errors:
            raise type(self).errors[venue]
        notes = [_FakeNote(payload) for payload in type(self).notes.get(venue, [])]
        if with_count and venue in type(self).counts:
            return notes[:limit], type(self).counts[venue]
        return notes[:limit]

    def get_all_notes(self, content: dict[str, str] | None = None) -> list[_FakeNote]:
        """Return every accepted note for one venue.

        Parameters
        ----------
        content : dict[str, str] | None
            Content filter; the connector must pass ``{"venueid": ...}``.

        Returns
        -------
        list[_FakeNote]
            Notes for the venue.

        Raises
        ------
        Exception
            The exception configured for this venue.
        """
        venue = (content or {}).get("venueid", "")
        self.calls.append(("get_all_notes", dict(content or {})))
        if venue in type(self).errors:
            raise type(self).errors[venue]
        return [_FakeNote(payload) for payload in type(self).notes.get(venue, [])]


@pytest.fixture(autouse=True)
def _reset_fake_client() -> Any:
    """Reset the client double's class-level configuration between tests."""
    _FakeClient.notes = {}
    _FakeClient.counts = {}
    _FakeClient.errors = {}
    _FakeClient.count_errors = {}
    _FakeClient.venues = _KNOWN_VENUES
    _FakeClient.instances = []
    yield
    _FakeClient.instances = []


@pytest.fixture
def credentials(monkeypatch) -> None:
    """Set the OpenReview credentials the connector requires."""
    monkeypatch.setenv(_USERNAME_ENV, "researcher@example.edu")
    monkeypatch.setenv(_PASSWORD_ENV, "hunter2")


@pytest.fixture
def connector(tmp_path) -> OpenReviewConnector:
    """Return a connector with a private cache directory and one venue."""
    return OpenReviewConnector(venue_ids=[_VENUE], cache_dir=tmp_path / "cache")


def _run(
    connector: OpenReviewConnector,
    *,
    query: str = "[scaling laws]",
    max_papers: int | None = None,
    since: datetime.date | None = None,
    until: datetime.date | None = None,
) -> CoveredPapers:
    """Run a search with the client double installed.

    Parameters
    ----------
    connector : OpenReviewConnector
        Connector under test.
    query : str
        Query string to parse.
    max_papers : int | None
        Caller-imposed cap.
    since : datetime.date | None
        Lower bound.
    until : datetime.date | None
        Upper bound.

    Returns
    -------
    CoveredPapers
        What the connector returned.
    """
    with (
        patch.object(connector, "_rate_limit"),
        patch.object(openreview_module, "_load_client_class", return_value=_FakeClient),
    ):
        return connector.search(_parse(query), max_papers, None, since, until)


# ---------------------------------------------------------------------------
# The one thing this connector exists to get right
# ---------------------------------------------------------------------------


class TestAcceptedPapersOnly:
    """Rejections must not be reachable from this connector at all."""

    def test_papers_are_requested_by_venueid(self, connector, credentials) -> None:
        """``content={'venueid': ...}`` is what excludes rejected submissions.

        An accepted note carries the venue's own id in ``venueid``; withdrawn
        and desk-rejected submissions carry a different one.  Retrieving by
        invitation or by forum instead returns them mixed together, and nothing
        downstream can tell which is which -- a reading list silently seeded
        with work the field rejected.
        """
        _FakeClient.notes = {_VENUE: [_note()]}
        _FakeClient.counts = {_VENUE: 1}

        result = _run(connector)

        client = _FakeClient.instances[0]
        assert ("get_all_notes", {"venueid": _VENUE}) in client.calls
        assert len(result.papers) == 1

    def test_the_production_base_url_is_passed_explicitly(self, connector, credentials) -> None:
        """The client's own default ``baseurl`` is ``http://localhost:3001``.

        Verified in ``openreview/api/client.py:64`` on 2026-08-31.  Constructing
        the client without an explicit base URL sends every request to a port on
        this machine, so the connector must never rely on the default.
        """
        _FakeClient.notes = {_VENUE: []}
        _FakeClient.counts = {_VENUE: 0}
        _run(connector)

        assert _FakeClient.instances[0].baseurl == _BASE_URL


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


class TestCredentials:
    """An anonymous read of ``/notes`` is refused outright by the service."""

    @pytest.mark.parametrize(
        ("username", "password", "expected"),
        [
            ("", "pw", _USERNAME_ENV),
            ("user", "", _PASSWORD_ENV),
            ("", "", f"{_USERNAME_ENV} and {_PASSWORD_ENV}"),
        ],
    )
    def test_absent_credentials_are_skipped_not_empty(
        self, connector, monkeypatch, username, password, expected
    ) -> None:
        """Missing credentials must never look like a venue that accepted nothing.

        The message names exactly which variable is absent, because "set the
        OpenReview credentials" is not actionable when one of the two is
        already set.
        """
        monkeypatch.setenv(_USERNAME_ENV, username)
        monkeypatch.setenv(_PASSWORD_ENV, password)

        with (
            patch.object(connector, "_rate_limit"),
            patch.object(openreview_module, "_load_client_class") as loader,
        ):
            result = connector.search(_parse("[scaling laws]"))

        assert result.status is CoverageStatus.SKIPPED_NO_CREDENTIAL
        assert result.papers == ()
        assert result.coverage.missing_credential == expected
        loader.assert_not_called()

    def test_rejected_credentials_are_failed(self, connector, credentials) -> None:
        """A refused login is a failure, not an empty proceedings archive."""

        def _refuse(**_kwargs: Any) -> Any:
            """Raise the way the client does on a bad password."""
            raise RuntimeError("Invalid username or password")

        with (
            patch.object(connector, "_rate_limit"),
            patch.object(openreview_module, "_load_client_class", return_value=_refuse),
        ):
            result = connector.search(_parse("[scaling laws]"))

        assert result.status is CoverageStatus.FAILED
        assert "could not authenticate" in (result.coverage.error or "")

    def test_missing_package_is_failed_and_names_the_fix(self, connector, credentials) -> None:
        """A library that is not installed is a failure with an actionable message.

        The coverage enum has no member for "a local dependency is missing", so
        this takes the loudest available status rather than inventing one, and
        the message carries the remedy.
        """
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(
                openreview_module,
                "_load_client_class",
                side_effect=OpenReviewApiError("openreview-py is not installed"),
            ),
        ):
            result = connector.search(_parse("[scaling laws]"))

        assert result.status is CoverageStatus.FAILED
        assert "openreview-py is not installed" in (result.coverage.error or "")


# ---------------------------------------------------------------------------
# The tests that matter most
# ---------------------------------------------------------------------------


class TestCoverageIsHonest:
    """A failure must never be reported as a venue that accepted nothing."""

    def test_failure_and_emptiness_are_different_statuses(self, tmp_path, credentials) -> None:
        """The regression test for the connector as a whole.

        The two connectors use separate cache directories on purpose.  Sharing
        one would let the first run's cached answer satisfy the second, and the
        test would pass without the failing path ever executing -- which is how
        a green suite comes to prove nothing.
        """
        _FakeClient.notes = {_VENUE: []}
        _FakeClient.counts = {_VENUE: 0}
        empty = _run(OpenReviewConnector(venue_ids=[_VENUE], cache_dir=tmp_path / "empty"))

        _FakeClient.errors = {_VENUE: RuntimeError("upstream exploded")}
        failure = _run(OpenReviewConnector(venue_ids=[_VENUE], cache_dir=tmp_path / "failure"))

        assert failure.papers == empty.papers == ()
        assert failure.status is not empty.status
        assert failure.status is CoverageStatus.FAILED
        assert empty.status is CoverageStatus.PARTIAL

    def test_client_exceptions_do_not_escape_the_search(self, connector, credentials) -> None:
        """``OpenReviewException`` is not a class the base class catches.

        It derives from :class:`Exception` and from nothing this project knows,
        so an unconverted one would travel straight out of ``search`` and abort
        the whole multi-database run rather than degrading one source.
        """
        _FakeClient.errors = {_VENUE: RuntimeError("OpenReviewException-like")}
        result = _run(connector)

        assert result.status is CoverageStatus.FAILED
        assert result.coverage.error_type == OpenReviewApiError.__name__

    def test_a_result_is_never_complete(self, connector, credentials) -> None:
        """Every note the venue holds arrived, and ``COMPLETE`` is still unavailable.

        Two independent reasons, either sufficient: OpenReview was never asked
        the query, so its venue count is not a count of matching records; and a
        chosen handful of venues says nothing about the thousands not consulted.
        """
        _FakeClient.notes = {_VENUE: [_note("a"), _note("b", title="Other work")]}
        _FakeClient.counts = {_VENUE: 2}

        result = _run(connector)

        assert result.status is CoverageStatus.PARTIAL
        assert "matched the query locally" in (result.coverage.detail or "")
        assert "thousands of other venues" in (result.coverage.detail or "")

    def test_cap_is_reported_as_partial_not_capped(self, connector, credentials) -> None:
        """A truncated local match must not read as milder than an untruncated one."""
        _FakeClient.notes = {_VENUE: [_note("a"), _note("b"), _note("c")]}
        _FakeClient.counts = {_VENUE: 3}

        result = _run(connector, max_papers=1)

        assert len(result.papers) == 1
        assert result.status is CoverageStatus.PARTIAL
        assert "limit of 1" in (result.coverage.detail or "")

    def test_a_short_venue_is_reported_as_a_shortfall(self, connector, credentials) -> None:
        """Fewer notes than the venue's own count is a loss, not a smaller venue.

        ``get_all_notes`` obtains this count internally and discards it, so
        without the separate count request a pagination walk that returned half
        a venue would be invisible.
        """
        _FakeClient.notes = {_VENUE: [_note("a")]}
        _FakeClient.counts = {_VENUE: 400}

        result = _run(connector)

        assert result.status is CoverageStatus.PARTIAL
        assert "1 notes returned against the 400" in (result.coverage.detail or "")

    def test_a_failed_count_probe_does_not_cost_the_venue(self, connector, credentials) -> None:
        """The cross-check is worth less than the papers it checks.

        Dropping three thousand accepted papers because an auxiliary count
        request failed would trade recall for a check that only sharpens the
        detail.  The venue is kept and the missing cross-check is reported, so
        the loss is not mistaken for a venue that publishes no count.
        """
        _FakeClient.notes = {_VENUE: [_note("a"), _note("b")]}
        _FakeClient.count_errors = {_VENUE: RuntimeError("count endpoint 500")}

        result = _run(connector)

        assert len(result.papers) == 2
        assert result.status is CoverageStatus.PARTIAL
        assert "count could not be read" in (result.coverage.detail or "")

    def test_transport_failures_are_reported_too(self, connector, credentials) -> None:
        """A network error reaching OpenReview is a failure like any other."""
        _FakeClient.errors = {_VENUE: requests.ConnectionError("dns")}
        result = _run(connector)

        assert result.status is CoverageStatus.FAILED


# ---------------------------------------------------------------------------
# Venue selection
# ---------------------------------------------------------------------------


class TestUnknownVenues:
    """A venue id that does not exist returns zero notes and no error."""

    def test_a_venue_openreview_does_not_host_is_not_queried(self, tmp_path, credentials) -> None:
        """The gap is real: ICML has no 2021 or 2022 Conference venue.

        Without this check, asking for one would read as "ICML 2022 accepted no
        papers" -- a confident, well-formed, entirely false statement.
        """
        missing = "ICML.cc/2022/Conference"
        connector = OpenReviewConnector(venue_ids=[_VENUE, missing], cache_dir=tmp_path / "cache")
        _FakeClient.notes = {_VENUE: [_note()]}
        _FakeClient.counts = {_VENUE: 1}

        result = _run(connector)

        client = _FakeClient.instances[0]
        assert ("get_all_notes", {"venueid": missing}) not in client.calls
        assert "not hosted by OpenReview" in (result.coverage.detail or "")
        assert missing in (result.coverage.detail or "")

    def test_an_unavailable_venue_listing_does_not_reject_everything(
        self, tmp_path, credentials
    ) -> None:
        """Validation is best effort; failing it must not empty the search."""
        connector = OpenReviewConnector(venue_ids=[_VENUE], cache_dir=tmp_path / "cache")
        _FakeClient.venues = None
        _FakeClient.notes = {_VENUE: [_note()]}
        _FakeClient.counts = {_VENUE: 1}

        result = _run(connector)

        assert len(result.papers) == 1

    def test_default_venues_follow_the_date_bounds(self, tmp_path, credentials) -> None:
        """With no venue list, the years come from the search's own bounds."""
        connector = OpenReviewConnector(cache_dir=tmp_path / "cache")
        _FakeClient.notes = {_VENUE: [_note()], _OTHER_VENUE: []}
        _FakeClient.counts = {_VENUE: 1, _OTHER_VENUE: 0}

        result = _run(
            connector,
            since=datetime.date(2024, 1, 1),
            until=datetime.date(2025, 12, 31),
        )

        detail = result.coverage.detail or ""
        assert _VENUE in detail
        assert _OTHER_VENUE in detail

    def test_every_consulted_venue_is_named(self, connector, credentials) -> None:
        """ "We searched OpenReview" would be a false description of the run."""
        _FakeClient.notes = {_VENUE: []}
        _FakeClient.counts = {_VENUE: 0}

        result = _run(connector)

        assert _VENUE in (result.coverage.detail or "")


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class TestCache:
    """Responses are cached locally, and a cached answer says how old it is."""

    def test_a_second_search_reuses_the_cache(self, connector, credentials) -> None:
        """A repeat run makes no further requests to OpenReview."""
        _FakeClient.notes = {_VENUE: [_note()]}
        _FakeClient.counts = {_VENUE: 1}

        _run(connector)
        first_calls = len(_FakeClient.instances[0].calls)
        result = _run(connector)

        assert len(_FakeClient.instances[0].calls) == first_calls
        assert len(result.papers) == 1

    def test_a_cached_venue_says_so(self, connector, credentials) -> None:
        """A stale cache makes a recent paper's absence look like a valid zero.

        The remedy is not to distrust the cache but to say when it was filled,
        so that absence can be read correctly.
        """
        _FakeClient.notes = {_VENUE: [_note()]}
        _FakeClient.counts = {_VENUE: 1}

        _run(connector)
        result = _run(connector)

        assert "served from the local cache" in (result.coverage.detail or "")

    def test_an_expired_cache_is_refetched(self, tmp_path, credentials) -> None:
        """A zero time-to-live means every run goes to the source."""
        connector = OpenReviewConnector(
            venue_ids=[_VENUE], cache_dir=tmp_path / "cache", cache_ttl_seconds=0
        )
        _FakeClient.notes = {_VENUE: [_note()]}
        _FakeClient.counts = {_VENUE: 1}

        _run(connector)
        before = len(_FakeClient.instances[0].calls)
        _run(connector)

        assert len(_FakeClient.instances[0].calls) > before

    def test_a_corrupt_cache_file_is_a_miss_not_an_outage(self, connector, credentials) -> None:
        """Refusing to run because a cache is unreadable would be worse."""
        _FakeClient.notes = {_VENUE: [_note()]}
        _FakeClient.counts = {_VENUE: 1}
        _run(connector)
        connector._cache_path(_VENUE).write_text("{not json", encoding="utf-8")

        result = _run(connector)

        assert len(result.papers) == 1


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """The client's own HTTP has to be paced and audited like everything else."""

    def test_the_clients_session_is_remounted(self, connector, credentials) -> None:
        """Left alone, ``openreview-py``'s session bypasses the limiter entirely."""
        _FakeClient.notes = {_VENUE: []}
        _FakeClient.counts = {_VENUE: 0}
        _run(connector)

        session = _FakeClient.instances[0].session
        assert isinstance(session.adapters["https://"], _PacedAdapter)
        assert isinstance(session.adapters["http://"], _PacedAdapter)

    def test_the_adapter_reserves_a_slot_and_records_the_attempt(self) -> None:
        """Each request reserves a slot before it goes out, and is audited."""
        reserved: list[int] = []
        adapter = _PacedAdapter(lambda: reserved.append(1))

        with (
            patch.object(requests.adapters.HTTPAdapter, "send", return_value="response") as sent,
            patch.object(openreview_module, "record_request_attempt") as noted,
        ):
            assert adapter.send(MagicMock()) == "response"

        assert reserved == [1]
        noted.assert_called_once()
        sent.assert_called_once()

    def test_the_adapter_records_a_failure_before_reraising(self) -> None:
        """A failure inside the client's HTTP must reach the retrieval audit."""
        adapter = _PacedAdapter(lambda: None)
        boom = requests.ConnectionError("boom")

        with (
            patch.object(requests.adapters.HTTPAdapter, "send", side_effect=boom),
            patch.object(openreview_module, "record_request_failure") as recorded,
            pytest.raises(requests.ConnectionError),
        ):
            adapter.send(MagicMock())

        recorded.assert_called_once_with(boom)

    def test_the_published_limit_is_declared(self, connector) -> None:
        """OpenReview publishes ``ratelimit-policy: 180;w=60`` in every response.

        The project specification records that it publishes none.  That was
        wrong on 2026-08-31, and the number is declared here rather than
        guessed.  The interval is deliberately slower than the budget; see the
        connector module.
        """
        assert connector.request_quota.limit == 180
        assert connector.request_quota.window_seconds == 60.0
        assert connector.min_request_interval > 0
        assert connector.rate_limit_service == "openreview"

    def test_the_bucket_is_host_wide_not_per_account(self, connector) -> None:
        """The allowance is enforced per source IP, so accounts share one clock.

        Scoping by credential here would give two signed-in processes separate
        clocks against one budget, which is the failure the credential-keyed
        scope exists to prevent in the other direction.
        """
        assert connector._credential_attrs == ()
        assert connector.rate_limit_scope.is_anonymous


# ---------------------------------------------------------------------------
# Query handling and parsing
# ---------------------------------------------------------------------------


class TestQueryIsAnsweredExactly:
    """OpenReview cannot be asked the query, so the query is answered here."""

    def test_and_not_is_honoured(self, connector, credentials) -> None:
        """Neither the notes endpoint nor ``search_notes`` can express negation."""
        _FakeClient.notes = {
            _VENUE: [
                _note("a", title="Scaling laws for language models", abstract="text only"),
                _note("b", title="Scaling laws for vision models", abstract="vision transformer"),
            ]
        }
        _FakeClient.counts = {_VENUE: 2}

        result = _run(connector, query="[scaling laws] AND NOT [vision]")

        assert [paper.title for paper in result.papers] == ["Scaling laws for language models"]

    def test_source_query_is_rejected_without_contacting_openreview(
        self, connector, credentials
    ) -> None:
        """A filter a note cannot answer is refused, not silently widened."""
        with (
            patch.object(connector, "_rate_limit"),
            patch.object(openreview_module, "_load_client_class") as loader,
        ):
            result = connector.search(_parse("src[ICLR]"))

        assert result.status is CoverageStatus.QUERY_UNSUPPORTED
        loader.assert_not_called()


class TestParsing:
    """Mapping an API v2 note onto the project's record shape."""

    def test_fields_are_mapped(self, connector, credentials) -> None:
        """Title, authors, venue, keywords, URLs and date survive the trip."""
        _FakeClient.notes = {_VENUE: [_note()]}
        _FakeClient.counts = {_VENUE: 1}

        paper = _run(connector).papers[0]

        assert paper.title == "Scaling Laws for Neural Language Models"
        assert [author.name for author in paper.authors] == ["Jared Kaplan"]
        assert paper.source is not None
        assert paper.source.title == "ICLR 2025 poster"
        assert paper.source.source_type is SourceType.CONFERENCE
        assert paper.paper_type is PaperType.INPROCEEDINGS
        assert paper.url == "https://openreview.net/forum?id=abc123"
        assert paper.pdf_url == "https://openreview.net/pdf/abc123.pdf"
        assert paper.keywords == {"scaling", "language models"}
        assert paper.publication_date == datetime.date(2025, 1, 1)
        assert paper.found_in == {Database.OPENREVIEW.value}

    def test_unwrapped_api_v1_content_still_parses(self, connector, credentials) -> None:
        """Notes migrated from API v1 store content values unwrapped.

        Rejecting that shape would drop records for a reason nobody would ever
        see in the output.
        """
        _FakeClient.notes = {
            _VENUE: [
                {
                    "id": "v1note",
                    "content": {
                        "title": "An older submission",
                        "abstract": "scaling laws revisited",
                        "authors": ["A Researcher"],
                    },
                    "cdate": 1_600_000_000_000,
                }
            ]
        }
        _FakeClient.counts = {_VENUE: 1}

        result = _run(connector)

        assert [paper.title for paper in result.papers] == ["An older submission"]

    def test_untitled_note_is_counted_as_a_loss(self, connector, credentials) -> None:
        """A note that cannot be parsed is reported, not quietly skipped."""
        _FakeClient.notes = {_VENUE: [_note("a"), _note("b", title="")]}
        _FakeClient.counts = {_VENUE: 2}

        result = _run(connector)

        assert len(result.papers) == 1
        assert "carried no readable title" in (result.coverage.detail or "")

    def test_builder_is_the_openreview_one(self, connector) -> None:
        """The connector must not inherit another source's field mapping."""
        assert isinstance(connector.query_builder, OpenReviewQueryBuilder)
