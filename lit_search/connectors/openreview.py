"""OpenReview proceedings connector, driven by the official ``openreview-py`` client.

**This is new code, not a port.**  ``monk1337/resp``, which the project
specification originally listed as the proceedings source, turned out to
implement its OpenReview method as ``self.engine.google_search('site:...')``
through a paid SerpAPI key -- web search wearing the costume of proceedings
coverage -- and was dropped.  ``D22`` replaces it with the official client.

Everything below was verified live on 2026-08-31 unless marked otherwise, and
three of the findings contradict what the specification assumed.

The one thing this connector exists to get right
------------------------------------------------

Accepted papers are retrieved with ``get_all_notes(content={'venueid': ...})``.
The ``venueid`` field on an accepted note is the venue's own id; submissions,
withdrawals and desk rejections carry a *different* ``venueid``
(``.../Withdrawn_Submission`` and similar).  A crawler that enumerates by
invitation, or by forum, gets rejected papers mixed in with accepted ones **and
nothing in the output reveals it** -- a reading list silently seeded with work
the field rejected.  The pattern is taken from ``danmackinlay/openreview_finder``
as a design source, not as code.

Findings that contradict the specification
------------------------------------------

1. **OpenReview does publish a rate limit.**  The specification says it
   publishes none and that a limiter must therefore be guessed.  A live
   response carries ``ratelimit-policy: 180;w=60`` together with
   ``x-ratelimit-limit: 180``, ``x-ratelimit-remaining`` and
   ``ratelimit-reset``.  See :data:`_REQUEST_QUOTA`.
2. **Anonymous reads of ``/notes`` are refused.**  ``GET
   https://api2.openreview.net/notes?content.venueid=ICLR.cc/2025/Conference``
   returns HTTP 403 ``{"name": "ChallengeRequiredError", "message": "Challenge
   verification required"}``.  ``/groups`` still answers anonymously.  So the
   credential requirement is not a policy choice this connector is making: an
   unauthenticated note search cannot work at all, and a connector that
   swallowed that 403 would report every venue as holding no accepted papers.
3. **``OpenReviewClient``'s default ``baseurl`` is ``http://localhost:3001``**
   (``api/client.py:64``), not the production API.  Constructing the client
   without an explicit ``baseurl`` and without ``OPENREVIEW_API_BASEURL_V2``
   set points it at a port on the local machine.  This connector always passes
   :data:`_BASE_URL` explicitly.

Two more facts read from ``openreview-py`` 2.5.1's own source, both of which
shape the code here:

- ``get_all_notes`` asks the API for the result count (``efficient_iterget``
  sets ``with_count=True`` on its first page) and then **uses it only to size a
  progress bar and throws it away**.  The count is exactly the source-published
  total the coverage contract wants, so this connector asks for it separately
  with ``get_notes(..., limit=1, with_count=True)`` and cross-checks the number
  of notes actually returned against it.  A pagination walk that quietly
  returns fewer notes than the venue holds is otherwise invisible.
- The client does its own HTTP through a private :class:`requests.Session` with
  a urllib3 retry policy of its own, which would bypass this project's
  cross-process rate limiter and its retrieval audit entirely.  The session is
  therefore re-mounted with :class:`_PacedAdapter` before any request is made.

Why a result is never ``COMPLETE``
----------------------------------

Two independent reasons, and each alone is sufficient:

- OpenReview cannot be asked the query, so the query is answered locally (see
  :mod:`lit_search.query.local_match`).  The venue's own count is a count of
  its accepted papers, not of the records matching the query.
- The connector consults a **chosen list of venues**.  The live ``venues``
  group listed 4,607 venue ids on 2026-08-31; the default here is three
  conference families over a few years.  "We searched OpenReview" would be a
  false description of that, so :meth:`OpenReviewConnector._build_coverage`
  names every venue consulted and never claims more.

Venue ids are checked against the live ``venues`` group before use, because a
venue id that does not exist returns zero notes and no error.  ``ICML.cc`` has
no ``2021`` or ``2022`` Conference venue (verified: its Conference venue ids are
2020 and 2023-2026), so asking for one is not hypothetical -- and without the
check it would read as "ICML 2022 accepted no papers".
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import requests

if TYPE_CHECKING:
    from lit_search.query.builder import QueryBuilder

from lit_search.connectors.search_base import SearchConnectorBase
from lit_search.core.author import Author
from lit_search.core.coverage import CoveredPapers, SourceCoverage
from lit_search.core.paper import Database, Paper, PaperType
from lit_search.core.query import Query
from lit_search.core.retrieval_audit import record_request_attempt, record_request_failure
from lit_search.core.source import Source, SourceType
from lit_search.exceptions import ConnectorError, MissingApiKeyError
from lit_search.query.builders.openreview import OpenReviewQueryBuilder
from lit_search.query.local_match import Predicate, Record
from lit_search.store.store import default_store_root
from lit_search.utils.persistence import atomic_write_json
from lit_search.utils.rate_limit import RateLimitQuota

logger = logging.getLogger(__name__)

_BASE_URL = "https://api2.openreview.net"
"""Production API v2 base URL, always passed explicitly.

``OpenReviewClient.__init__`` defaults ``baseurl`` to ``http://localhost:3001``
when neither the argument nor ``OPENREVIEW_API_BASEURL_V2`` is supplied
(``openreview/api/client.py:64``, read 2026-08-31).  Relying on that default
sends every request to a local port.
"""

_WEB_URL = "https://openreview.net"

_USERNAME_ENV = "OPENREVIEW_USERNAME"
_PASSWORD_ENV = "OPENREVIEW_PASSWORD"

_VENUES_GROUP_ID = "venues"
"""Group whose members are every venue id OpenReview hosts.

``openreview.tools.get_all_venues`` reads ``host`` instead; ``venues`` is what
the live service answers with the 4,607-entry list used here (checked
2026-08-31).
"""

_MIN_REQUEST_INTERVAL = 1.0
"""Seconds between requests.

**Deliberately about three times slower than the published allowance**, and the
asymmetry is the reason.  The live headers give 180 requests per 60 seconds
(:data:`_REQUEST_QUOTA`), which is one every 0.34s.  Pacing at exactly the
budget leaves no room for anything else on this host that talks to OpenReview
-- a browser session, a second agent, a notebook -- because the allowance is
enforced per source IP rather than per account (see :attr:`_credential_attrs`).

Being too slow costs seconds over a handful of venue fetches.  Being too fast
costs an HTTP 429, and a 429 storm reaches the ledger as ``FAILED`` against the
*database*, which reads as an OpenReview outage rather than as our own pacing
collapsing.  The quota below is the documented ceiling; this interval is the
pacing.
"""

_REQUEST_QUOTA = RateLimitQuota(limit=180, window_seconds=60.0)
"""Budget-shaped limit, read from live response headers on 2026-08-31.

``ratelimit-policy: 180;w=60``, alongside ``x-ratelimit-limit: 180``,
``x-ratelimit-remaining`` and ``ratelimit-reset``.  The project specification
and build plan both record that OpenReview "publishes no rate limit"; that is
wrong as of this date, and the number is published in every response.

The quota is declared as well as the interval because the two constrain
different things: the interval cannot stop a burst that arrives from several
processes at once, and this service's allowance is a fixed window.
"""

_DEFAULT_YEARS_BACK = 2
"""How many years before the current one the default venue list reaches back.

Kept small on purpose.  Every additional year is another venue fetch per
family, and an unbounded default would harvest tens of thousands of notes for a
query that probably wanted the last two conferences.  Whatever it resolves to
is named in the coverage detail, so a caller can see the window they got rather
than assuming they got everything.
"""

_VENUE_TEMPLATES: dict[str, str] = {
    "ICLR": "ICLR.cc/{year}/Conference",
    "NeurIPS": "NeurIPS.cc/{year}/Conference",
    "ICML": "ICML.cc/{year}/Conference",
}
"""Venue-id templates for the machine-learning conferences ``D22`` names.

Verified against the live ``venues`` group on 2026-08-31: ICLR has Conference
venues for 2018-2027, NeurIPS for 2020-2026, and ICML for 2020 and 2023-2026.
The gaps are real, which is why every generated id is checked before use.
"""

_CACHE_SCHEMA_VERSION = 1
_CACHE_SUBDIRECTORY = ("cache", "openreview")
_DEFAULT_CACHE_TTL_SECONDS = 24 * 60 * 60
_VENUES_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60


class OpenReviewApiError(ConnectorError):
    """An OpenReview request that could not be turned into records.

    ``openreview.OpenReviewException`` derives from :class:`Exception` and from
    nothing this project catches, so it would travel straight through
    :meth:`~lit_search.connectors.search_base.SearchConnectorBase.search`,
    which deliberately catches only
    :class:`~requests.RequestException` and
    :class:`~lit_search.exceptions.ConnectorError`.  Every failure raised by the
    client is re-raised as this instead, so it reaches the ledger as ``FAILED``
    rather than escaping the search.

    Parameters
    ----------
    message : str
        What went wrong, phrased so the reader can act on it.
    """


@dataclass(slots=True)
class _VenueResult:
    """What one venue fetch produced.

    Attributes
    ----------
    venue_id : str
        The venue whose accepted papers were requested.
    notes : list[dict[str, Any]]
        Note payloads, in the shape ``openreview.api.Note.to_json`` produces.
    reported_total : int | None
        Count the API published for this venue, or ``None`` when the count
        request itself failed.
    from_cache : bool
        Whether the payload came from the local cache rather than the API.
    age_seconds : float
        Age of the cached payload; ``0.0`` for a fresh fetch.
    count_error : str | None
        Set when the count probe failed while the notes still arrived.  The
        venue is kept, and the fact that its arrival could not be cross-checked
        is reported rather than left to look like a venue that publishes no
        count.
    """

    venue_id: str
    notes: list[dict[str, Any]]
    reported_total: int | None
    from_cache: bool = False
    age_seconds: float = 0.0
    count_error: str | None = None


@dataclass(slots=True)
class _Harvest:
    """Mutable bookkeeping for one multi-venue retrieval.

    Attributes
    ----------
    papers : list[Paper]
        Records that parsed **and** matched the query, in retrieval order.
    processed : int
        Notes seen, whether or not they parsed or matched.
    parsed : int
        Notes that became a :class:`~lit_search.core.paper.Paper`.
    matched : int
        Parsed notes that satisfied the query predicate.
    reported_total : int
        Sum of the per-venue counts OpenReview published.
    consulted : list[str]
        Venue ids actually fetched.
    stale : list[str]
        Descriptions of venues served from cache, with their age.
    shortfalls : list[str]
        Descriptions of venues where fewer notes arrived than were published.
    failures : list[str]
        Descriptions of venues that could not be fetched.
    first_error : Exception | None
        The first failure, re-raised when nothing at all was enumerated.
    truncated_by_cap : bool
        Whether the caller's ``max_papers`` stopped the harvest.
    """

    papers: list[Paper] = field(default_factory=list)
    processed: int = 0
    parsed: int = 0
    matched: int = 0
    reported_total: int = 0
    consulted: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    shortfalls: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    first_error: Exception | None = None
    truncated_by_cap: bool = False

    def note_failure(self, venue_id: str, exc: Exception) -> None:
        """Record a venue that could not be fetched, keeping the first cause.

        Parameters
        ----------
        venue_id : str
            Venue whose fetch failed.
        exc : Exception
            The failure.
        """
        self.failures.append(f"{venue_id}: {type(exc).__name__}: {exc}")
        if self.first_error is None:
            self.first_error = exc


class _PacedAdapter(requests.adapters.HTTPAdapter):
    """Transport adapter that paces and records ``openreview-py``'s own HTTP.

    ``openreview-py`` talks to the API through a :class:`requests.Session` it
    owns.  Left alone, none of that traffic passes through
    :meth:`~lit_search.connectors.connector_base.ConnectorBase._get`, so it is
    neither rate limited nor visible to the retrieval audit -- the two things
    ``PORTING-GUIDE.md`` section 5 says a connector must never let happen.

    Mounting this adapter on the client's session puts every request back
    inside both mechanisms without reimplementing the client.  The one caveat,
    stated because it is invisible otherwise: urllib3's own retries happen
    *inside* one :meth:`send` call, so a retried request is paced by the
    client's backoff rather than by this project's limiter.  The client's
    policy honours ``Retry-After``, so that is a delay rather than a burst.

    Parameters
    ----------
    reserve : Callable[[], None]
        Called before each request; blocks until a slot is available.
    **kwargs : Any
        Forwarded to :class:`requests.adapters.HTTPAdapter`, so the client's
        own retry policy can be preserved.
    """

    def __init__(self, reserve: Callable[[], None], **kwargs: Any) -> None:
        """Store the reservation callable and build the underlying adapter."""
        self._reserve = reserve
        super().__init__(**kwargs)

    def send(
        self,
        request: requests.PreparedRequest,
        stream: bool = False,
        timeout: float | tuple[float, float] | tuple[float, None] | None = None,
        verify: bool | str = True,
        cert: bytes | str | tuple[bytes | str, bytes | str] | None = None,
        proxies: Mapping[str, str] | None = None,
    ) -> requests.Response:
        """Reserve a slot, record the attempt, then send.

        The signature mirrors :meth:`requests.adapters.HTTPAdapter.send`
        exactly rather than collapsing into ``**kwargs``, so that a change to
        the transport contract is a type error here rather than a silently
        dropped argument -- ``verify`` and ``cert`` in particular.

        Parameters
        ----------
        request : requests.PreparedRequest
            Prepared request.
        stream : bool
            Whether to stream the response body.
        timeout : float | tuple[float, float] | tuple[float, None] | None
            Connect and read timeouts.
        verify : bool | str
            TLS verification setting, or a CA bundle path.
        cert : bytes | str | tuple[bytes | str, bytes | str] | None
            Client certificate.
        proxies : Mapping[str, str] | None
            Proxy configuration.

        Returns
        -------
        requests.Response
            The HTTP response.

        Raises
        ------
        Exception
            Whatever the underlying adapter raises, after recording it on the
            active retrieval audit.
        """
        self._reserve()
        record_request_attempt()
        try:
            return super().send(
                request,
                stream=stream,
                timeout=timeout,
                verify=verify,
                cert=cert,
                proxies=proxies,
            )
        except Exception as exc:
            record_request_failure(exc)
            raise


class OpenReviewConnector(SearchConnectorBase):
    """Connector for OpenReview conference proceedings.

    Enumerates the accepted papers of a chosen set of venues and applies the
    query locally.  See the module docstring for why there is no remote query,
    why credentials are mandatory, and why a result is never ``COMPLETE``.
    """

    _credential_attrs: ClassVar[tuple[str, ...]] = ()
    """The rate-limit bucket is host-wide, not per account, and that is deliberate.

    OpenReview's allowance is enforced at its edge by source IP -- the response
    carries ``ratelimit-policy: 180;w=60`` with no account in sight -- so two
    processes signed in as different users still draw on one budget.  Declaring
    the username here would give them separate clocks and let them together
    exceed the limit they share, which is the precise failure ``R8``'s
    credential-keyed scope exists to prevent in the other direction.
    """

    def __init__(
        self,
        query_builder: OpenReviewQueryBuilder | None = None,
        *,
        username: str | None = None,
        password: str | None = None,
        venue_ids: Sequence[str] | None = None,
        cache_dir: Path | str | None = None,
        cache_ttl_seconds: float = _DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        """Create an OpenReview searcher.

        Parameters
        ----------
        query_builder : OpenReviewQueryBuilder | None
            Builder used to validate queries and compile the local predicate.
        venue_ids : Sequence[str] | None
            Venue ids to consult.  When ``None`` they are generated from
            :data:`_VENUE_TEMPLATES` and the search's date bounds, and the
            resolved list is named in the coverage detail either way.
        cache_dir : Path | str | None
            Where fetched venues are cached.  Defaults to a subdirectory of the
            handle store's root, so the whole tool keeps one cache root.
        cache_ttl_seconds : float
            How long a cached venue stays usable.  A cached venue's age is
            reported in the coverage detail, because a stale cache makes a
            recent paper's absence look like a valid zero.
        """
        super().__init__()
        self._username = (username or "").strip() or None
        self._password = (password or "").strip() or None
        self._query_builder: OpenReviewQueryBuilder = query_builder or OpenReviewQueryBuilder()
        self._venue_ids = tuple(venue_ids) if venue_ids is not None else None
        self._cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else default_store_root().joinpath(*_CACHE_SUBDIRECTORY)
        )
        self._cache_ttl_seconds = cache_ttl_seconds
        self._client: Any | None = None

    @property
    def name(self) -> str:
        """Return the database identifier.

        Returns
        -------
        str
            Database name.
        """
        return Database.OPENREVIEW.value

    @property
    def query_builder(self) -> QueryBuilder:
        """Return the OpenReview query builder.

        Returns
        -------
        QueryBuilder
            The underlying builder instance.
        """
        return self._query_builder

    @property
    def min_request_interval(self) -> float:
        """Return the minimum seconds between HTTP requests.

        Returns
        -------
        float
            Interval in seconds.  See :data:`_MIN_REQUEST_INTERVAL`.
        """
        return _MIN_REQUEST_INTERVAL

    @property
    def request_quota(self) -> RateLimitQuota:
        """Return OpenReview's published per-window request budget.

        Returns
        -------
        RateLimitQuota
            180 requests per 60 seconds.  See :data:`_REQUEST_QUOTA`.
        """
        return _REQUEST_QUOTA

    @property
    def rate_limit_service(self) -> str:
        """Return the quota authority these requests are paced against.

        Stated explicitly so that any future connector against another
        OpenReview route -- the PDF endpoint, the reviews an
        ``ideaforge``-style crawler would read -- shares this clock rather than
        doubling the rate against one allowance.

        Returns
        -------
        str
            Quota-authority identifier.
        """
        return "openreview"

    # ------------------------------------------------------------------
    # Credentials and client
    # ------------------------------------------------------------------

    def _require_credentials(self) -> tuple[str, str]:
        """Return the configured credentials, or refuse to run without them.

        Returns
        -------
        tuple[str, str]
            Username and password.

        Raises
        ------
        MissingApiKeyError
            When either environment variable is absent or blank.  The base
            class turns this into ``SKIPPED_NO_CREDENTIAL``, so OpenReview
            stays visible in the ledger as a source that was deliberately not
            consulted rather than one that held nothing.
        """
        username = self._username or os.environ.get(_USERNAME_ENV, "").strip()
        password = self._password or os.environ.get(_PASSWORD_ENV, "").strip()
        if not username or not password:
            missing = " and ".join(
                name
                for name, value in ((_USERNAME_ENV, username), (_PASSWORD_ENV, password))
                if not value
            )
            raise MissingApiKeyError(missing)
        return username, password

    def _open_client(self) -> Any:
        """Build and cache an authenticated API v2 client.

        Returns
        -------
        Any
            An ``openreview.api.OpenReviewClient`` whose session has been
            re-mounted on :class:`_PacedAdapter`.

        Raises
        ------
        OpenReviewApiError
            When ``openreview-py`` is not installed, or when authentication
            failed.  Both reach the ledger as ``FAILED``: a missing library and
            a rejected password are different problems, but both mean the
            source was not read, and neither may become an empty result.
        """
        if self._client is not None:
            return self._client
        username, password = self._require_credentials()
        client_class = _load_client_class()
        try:
            client = client_class(baseurl=_BASE_URL, username=username, password=password)
        except Exception as exc:
            raise OpenReviewApiError(
                f"could not authenticate to {_BASE_URL} as '{username}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self._pace_session(client)
        self._client = client
        return client

    def _pace_session(self, client: Any) -> None:
        """Re-mount the client's session so its HTTP is paced and audited.

        Parameters
        ----------
        client : Any
            Client whose :class:`requests.Session` should be wrapped.  The
            existing adapter's retry policy is preserved, so only the pacing
            and the audit hooks are added.
        """
        session = getattr(client, "session", None)
        if session is None:  # pragma: no cover - the client always builds one
            return
        self._apply_transport_configuration(session)
        for prefix in ("https://", "http://"):
            existing = session.adapters.get(prefix)
            retries = getattr(existing, "max_retries", None)
            adapter = (
                _PacedAdapter(self._rate_limit, max_retries=retries)
                if retries is not None
                else _PacedAdapter(self._rate_limit)
            )
            session.mount(prefix, adapter)

    # ------------------------------------------------------------------
    # Venue selection
    # ------------------------------------------------------------------

    def _resolve_requested_venues(
        self,
        since: datetime.date | None,
        until: datetime.date | None,
    ) -> tuple[str, ...]:
        """Return the venue ids this search should consult.

        Parameters
        ----------
        since : datetime.date | None
            Lower publication-date bound, used only to pick years.
        until : datetime.date | None
            Upper publication-date bound, used only to pick years.

        Returns
        -------
        tuple[str, ...]
            Venue ids, either the caller's or generated from the templates.
        """
        if self._venue_ids is not None:
            return self._venue_ids
        last = (until or datetime.date.today()).year
        first = since.year if since is not None else last - _DEFAULT_YEARS_BACK
        years = range(min(first, last), last + 1)
        return tuple(
            template.format(year=year) for year in years for template in _VENUE_TEMPLATES.values()
        )

    def _known_venue_ids(self, client: Any) -> frozenset[str]:
        """Return every venue id OpenReview hosts, from cache when fresh.

        Parameters
        ----------
        client : Any
            Authenticated client.

        Returns
        -------
        frozenset[str]
            Venue ids.  Empty when the listing could not be read, which makes
            the caller skip validation rather than reject every venue.
        """
        cached = self._read_cache("venues", _VENUES_CACHE_TTL_SECONDS)
        if cached is not None:
            members = cached.get("members")
            if isinstance(members, list):
                return frozenset(str(member) for member in members)
        try:
            group = client.get_group(_VENUES_GROUP_ID)
            members = [str(member) for member in (group.members or [])]
        except Exception as exc:
            logger.warning(
                "OpenReview: could not read the '%s' group, so venue ids cannot be "
                "checked for existence this run: %s",
                _VENUES_GROUP_ID,
                exc,
            )
            return frozenset()
        self._write_cache("venues", {"members": members})
        return frozenset(members)

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        """Return the file a cache *key* is stored in.

        Parameters
        ----------
        key : str
            Cache key, typically a venue id.

        Returns
        -------
        Path
            File path.  The name is a hash because venue ids contain ``/``.
        """
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return self._cache_dir / f"{digest}.json"

    def _read_cache(self, key: str, ttl_seconds: float) -> dict[str, Any] | None:
        """Return a cached payload when one exists and is fresh enough.

        A cache file that cannot be read, or that carries a schema version this
        code does not know, is treated as a miss rather than as an error: the
        remedy is to fetch again, and refusing to run because a cache is stale
        would convert a convenience into an outage.

        Parameters
        ----------
        key : str
            Cache key.
        ttl_seconds : float
            Maximum acceptable age.

        Returns
        -------
        dict[str, Any] | None
            The payload, with ``age_seconds`` added, or ``None`` on a miss.
        """
        path = self._cache_path(key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict) or payload.get("schema") != _CACHE_SCHEMA_VERSION:
            return None
        if payload.get("key") != key:
            return None
        age = time.time() - float(payload.get("fetched_at", 0.0))
        if age < 0 or age > ttl_seconds:
            return None
        payload["age_seconds"] = age
        return payload

    def _write_cache(self, key: str, payload: dict[str, Any]) -> None:
        """Store a payload under *key*, ignoring a cache that cannot be written.

        Parameters
        ----------
        key : str
            Cache key.
        payload : dict[str, Any]
            Payload to store.
        """
        record = {
            "schema": _CACHE_SCHEMA_VERSION,
            "key": key,
            "fetched_at": time.time(),
            **payload,
        }
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self._cache_path(key), record)
        except OSError as exc:
            logger.debug("OpenReview: could not write the cache for '%s': %s", key, exc)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _fetch_venue(self, client: Any, venue_id: str) -> _VenueResult:
        """Retrieve one venue's accepted papers, from cache when fresh.

        Parameters
        ----------
        client : Any
            Authenticated client.
        venue_id : str
            Venue id, used as the ``content.venueid`` filter.

        Returns
        -------
        _VenueResult
            Notes and the count OpenReview published for the venue.

        Raises
        ------
        OpenReviewApiError
            When the venue could not be retrieved.
        """
        cached = self._read_cache(venue_id, self._cache_ttl_seconds)
        if cached is not None:
            notes = cached.get("notes")
            if isinstance(notes, list):
                total = cached.get("reported_total")
                return _VenueResult(
                    venue_id=venue_id,
                    notes=[note for note in notes if isinstance(note, dict)],
                    reported_total=total if isinstance(total, int) else None,
                    from_cache=True,
                    age_seconds=float(cached.get("age_seconds", 0.0)),
                )

        content = {"venueid": venue_id}
        # The count probe is a cross-check, not the retrieval, so its failure
        # must not cost the venue.  Losing three thousand accepted papers
        # because an auxiliary request 500'd would trade recall -- the thing
        # this project exists to protect -- for a check that only sharpens the
        # coverage detail.  The loss of the check is itself recorded.
        count_error: str | None = None
        try:
            reported_total = _venue_count(client, content)
        except Exception as exc:
            reported_total = None
            count_error = f"{type(exc).__name__}: {exc}"
            logger.debug("OpenReview: count probe for '%s' failed: %s", venue_id, count_error)

        try:
            notes = [note.to_json() for note in client.get_all_notes(content=content)]
        except (requests.RequestException, ConnectorError):
            raise
        except Exception as exc:
            raise OpenReviewApiError(
                f"OpenReview refused the accepted-paper request for venue '{venue_id}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        self._write_cache(venue_id, {"notes": notes, "reported_total": reported_total})
        return _VenueResult(
            venue_id=venue_id,
            notes=notes,
            reported_total=reported_total,
            count_error=count_error,
        )

    def _absorb_venue(
        self,
        run: _Harvest,
        result: _VenueResult,
        matches: Predicate,
        max_papers: int | None,
    ) -> None:
        """Fold one venue's notes into *run*, keeping the records that match.

        Parameters
        ----------
        run : _Harvest
            Bookkeeping for the current retrieval, updated in place.
        result : _VenueResult
            One venue's notes.
        matches : Predicate
            Compiled query predicate.
        max_papers : int | None
            Caller-imposed cap.
        """
        run.consulted.append(result.venue_id)
        if result.from_cache:
            run.stale.append(f"{result.venue_id} (cached {result.age_seconds / 3600:.1f}h ago)")
        if result.count_error is not None:
            run.shortfalls.append(
                f"{result.venue_id}: the notes arrived but the venue's own count could not "
                f"be read ({result.count_error}), so nothing corroborates that all of them "
                f"did"
            )
        if result.reported_total is not None:
            run.reported_total += result.reported_total
            if len(result.notes) != result.reported_total:
                run.shortfalls.append(
                    f"{result.venue_id}: {len(result.notes)} notes returned against the "
                    f"{result.reported_total} OpenReview reported for that venue"
                )

        for note in result.notes:
            run.processed += 1
            record = _note_record(note)
            paper = self._parse_note(note, record, result.venue_id)
            if paper is None:
                continue
            run.parsed += 1
            if not matches(record):
                continue
            run.matched += 1
            if max_papers is None or len(run.papers) < max_papers:
                run.papers.append(paper)
            else:
                run.truncated_by_cap = True

    def _fetch_papers(
        self,
        query: Query,
        max_papers: int | None,
        progress_callback: Callable[[int, int | None], None] | None,
        since: datetime.date | None = None,
        until: datetime.date | None = None,
    ) -> CoveredPapers:
        """Enumerate the chosen venues' accepted papers and match locally.

        Parameters
        ----------
        query : Query
            Validated query object.
        max_papers : int | None
            Maximum papers to retrieve.  ``None`` means unlimited.
        progress_callback : Callable[[int, int | None], None] | None
            Progress callback called as ``callback(notes_seen, reported_total)``.
        since : datetime.date | None
            Lower bound, used to choose venue years when no venue list was given.
        until : datetime.date | None
            Upper bound, used the same way.

        Returns
        -------
        CoveredPapers
            Matching papers and the coverage status describing them.

        Raises
        ------
        MissingApiKeyError
            When the OpenReview credentials are absent.
        OpenReviewApiError
            When no venue could be enumerated at all.
        """
        client = self._open_client()
        matches = self._query_builder.compile_predicate(query)
        requested = self._resolve_requested_venues(since, until)
        known = self._known_venue_ids(client)
        venues, unknown = _partition_venues(requested, known)

        run = _Harvest()
        for venue_id in venues:
            try:
                result = self._fetch_venue(client, venue_id)
            except (requests.RequestException, ConnectorError) as exc:
                logger.warning("OpenReview: venue '%s' could not be read: %s", venue_id, exc)
                run.note_failure(venue_id, exc)
                continue
            self._absorb_venue(run, result, matches, max_papers)
            if progress_callback is not None:
                progress_callback(run.processed, run.reported_total or None)

        if progress_callback is not None:
            progress_callback(run.processed, run.reported_total or None)

        if run.processed == 0 and run.first_error is not None:
            # Nothing was enumerated, so there is no partial result to report
            # and the failure is the whole story.  Re-raising is what stops it
            # becoming "these venues accepted no papers".
            raise run.first_error

        return CoveredPapers(
            papers=tuple(run.papers),
            coverage=self._build_coverage(
                run=run, requested=requested, unknown=unknown, max_papers=max_papers
            ),
        )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_note(
        self,
        note: dict[str, Any],
        record: Record,
        venue_id: str,
    ) -> Paper | None:
        """Build a :class:`Paper` from one accepted-paper note.

        Parameters
        ----------
        note : dict[str, Any]
            Note payload in ``Note.to_json`` shape.
        record : Record
            The same note's text, already normalised for the predicate.
        venue_id : str
            Venue the note was retrieved for.

        Returns
        -------
        Paper | None
            Parsed paper, or ``None`` when the note has no title or fails model
            validation.  Callers **must** count the ``None`` returns: an
            unparsed note is a coverage loss, not an absent paper.
        """
        title = next(iter(record.get("title", ())), "").strip()
        if not title:
            return None
        content = note.get("content") or {}
        note_id = str(note.get("id") or "").strip()
        pdf = _content_value(content, "pdf")
        try:
            return Paper(
                title=title,
                abstract=next(iter(record.get("abstract", ())), "").strip(),
                authors=[Author(name=name) for name in record.get("authors", ()) if name],
                source=Source(
                    title=str(_content_value(content, "venue") or venue_id),
                    source_type=SourceType.CONFERENCE,
                ),
                publication_date=_note_date(note),
                url=f"{_WEB_URL}/forum?id={note_id}" if note_id else None,
                pdf_url=f"{_WEB_URL}{pdf}"
                if isinstance(pdf, str) and pdf.startswith("/")
                else None,
                keywords=set(record.get("keywords", ())) or None,
                found_in={self.name},
                paper_type=PaperType.INPROCEEDINGS,
                # Every note reachable under an accepted venueid is publicly
                # readable on openreview.net.
                is_open_access=True,
            )
        except ValueError:
            logger.debug("OpenReview: note %s failed model validation.", note.get("id"))
            return None

    # ------------------------------------------------------------------
    # Coverage
    # ------------------------------------------------------------------

    def _build_coverage(
        self,
        *,
        run: _Harvest,
        requested: Sequence[str],
        unknown: Sequence[str],
        max_papers: int | None,
    ) -> SourceCoverage:
        """Describe what the retrieval achieved.

        The status is always ``PARTIAL``, for two reasons that hold
        independently of how well the run went.

        OpenReview cannot be asked the query, so what it published -- a count of
        each venue's accepted papers -- is not a count of records matching the
        query, and ``COMPLETE`` needs the latter.  And the connector consults a
        chosen list of venues out of the thousands OpenReview hosts, so even a
        flawless run has said nothing about the rest.

        It is not ``CAPPED`` either when the caller's limit bound, for the same
        reason the IACR connector is not: ``CAPPED`` means a known amount is
        missing on purpose and ranks milder than ``PARTIAL``, whereas here an
        unknown amount is missing whatever the cap did.  The cap is named in the
        detail instead.

        Parameters
        ----------
        run : _Harvest
            Bookkeeping for the finished retrieval.
        requested : Sequence[str]
            Venue ids the search asked for.
        unknown : Sequence[str]
            Requested venue ids that OpenReview does not host.
        max_papers : int | None
            Caller-imposed cap.

        Returns
        -------
        SourceCoverage
            A ``PARTIAL`` record naming every venue consulted and every gap.
        """
        notes: list[str] = [
            "OpenReview has no boolean search over accepted papers, so this enumerated "
            f"{len(run.consulted)} of the {len(requested)} requested venue(s) "
            f"({', '.join(run.consulted) or 'none'}) and matched the query locally",
            f"saw {run.processed} accepted-paper notes"
            + (
                f" against the {run.reported_total} OpenReview reported for those venues"
                if run.reported_total
                else ""
            )
            + f" and matched {run.matched}",
            "OpenReview hosts thousands of other venues that were not consulted, so this "
            "says nothing about proceedings outside the list above",
        ]
        if unknown:
            notes.append(
                f"{len(unknown)} requested venue(s) are not hosted by OpenReview and were "
                f"not queried ({', '.join(unknown)}); a venue id that does not exist returns "
                f"zero notes and no error, which would read as a venue that accepted nothing"
            )
        if run.stale:
            notes.append("served from the local cache: " + "; ".join(run.stale))
        if run.shortfalls:
            notes.append("fewer notes arrived than were published: " + "; ".join(run.shortfalls))
        if run.failures:
            notes.append("venues that could not be read: " + "; ".join(run.failures))
        if run.truncated_by_cap and max_papers is not None:
            notes.append(
                f"the caller's limit of {max_papers} stopped collection, so {run.matched} "
                f"records had already matched when it bound"
            )
        unparsed = run.processed - run.parsed
        if unparsed > 0:
            notes.append(
                f"{unparsed} of {run.processed} notes carried no readable title or failed "
                f"model validation"
            )

        error = run.failures[0] if run.failures else None
        return SourceCoverage.partial(
            self.name,
            len(run.papers),
            detail="; ".join(notes),
            error=error,
            error_type=type(run.first_error).__name__ if run.first_error is not None else None,
        )


def _load_client_class() -> Any:
    """Import ``openreview.api.OpenReviewClient``, or say why it is unavailable.

    The import is deferred to call time rather than done at module scope so
    that registering the connector does not pull ``openreview-py`` and its
    dependency tree into every process that imports the registry.

    Returns
    -------
    Any
        The client class.

    Raises
    ------
    OpenReviewApiError
        When the package is not installed.  This reaches the ledger as
        ``FAILED``, which is the loudest status available and the only honest
        one: the source was not read, and the enum has no member for "a local
        dependency is missing".  The message names the fix.
    """
    try:
        from openreview.api import OpenReviewClient  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - exercised only without the package
        raise OpenReviewApiError(
            "openreview-py is not installed, so OpenReview could not be consulted. "
            "Install it with 'pip install openreview-py' (>=2.5.1). Reporting this as a "
            "failure rather than as an empty result is deliberate: no venue was read."
        ) from exc
    return OpenReviewClient


def _venue_count(client: Any, content: dict[str, str]) -> int | None:
    """Ask OpenReview how many accepted papers a venue holds.

    ``get_all_notes`` obtains this number internally -- ``efficient_iterget``
    sets ``with_count=True`` on its first page -- and then uses it only to size
    a progress bar.  It is asked for separately here because it is the only
    independent statement of how many records should have arrived, and without
    it a pagination walk that quietly returns half a venue is invisible.

    Parameters
    ----------
    client : Any
        Authenticated client.
    content : dict[str, str]
        Content filter, normally ``{"venueid": ...}``.

    Returns
    -------
    int | None
        The count, or ``None`` when the response did not carry one.
    """
    response = client.get_notes(content=content, limit=1, with_count=True)
    if isinstance(response, tuple) and len(response) == 2:
        count = response[1]
        if isinstance(count, int) and not isinstance(count, bool):
            return count
    return None


def _partition_venues(
    requested: Sequence[str],
    known: frozenset[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split requested venue ids into ones OpenReview hosts and ones it does not.

    Parameters
    ----------
    requested : Sequence[str]
        Venue ids the search asked for.
    known : frozenset[str]
        Venue ids OpenReview hosts.  Empty means the listing was unavailable,
        in which case every requested venue is attempted rather than rejected.

    Returns
    -------
    tuple[tuple[str, ...], tuple[str, ...]]
        ``(to_query, unknown)``.
    """
    if not known:
        return tuple(requested), ()
    to_query = tuple(venue for venue in requested if venue in known)
    unknown = tuple(venue for venue in requested if venue not in known)
    return to_query, unknown


def _content_value(content: dict[str, Any], field_name: str) -> Any:
    """Read one field from an API v2 note's ``content`` object.

    API v2 wraps every content value as ``{"value": ...}``; API v1 stored the
    value directly, and notes migrated from v1 can still be seen in that shape.
    Both are accepted, because rejecting one of them would drop records for a
    reason nobody would ever see in the output.

    Parameters
    ----------
    content : dict[str, Any]
        The note's ``content`` object.
    field_name : str
        Field to read.

    Returns
    -------
    Any
        The unwrapped value, or ``None`` when the field is absent.
    """
    raw = content.get(field_name)
    if isinstance(raw, dict) and "value" in raw:
        return raw["value"]
    return raw


def _as_text_list(value: Any) -> list[str]:
    """Normalise a note content value into a list of non-empty strings.

    Parameters
    ----------
    value : Any
        Raw value: a string, a list of strings, or something else.

    Returns
    -------
    list[str]
        Text values, with blanks dropped.
    """
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Iterable) and not isinstance(value, bytes | dict):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _note_record(note: dict[str, Any]) -> Record:
    """Normalise one note into the field bag the local predicate reads.

    Parameters
    ----------
    note : dict[str, Any]
        Note payload in ``Note.to_json`` shape.

    Returns
    -------
    Record
        Text keyed by the field names in
        :data:`lit_search.query.builders.openreview.FIELD_MAP`.
    """
    content = note.get("content") or {}
    if not isinstance(content, dict):
        content = {}
    return {
        "title": _as_text_list(_content_value(content, "title")),
        "abstract": _as_text_list(_content_value(content, "abstract")),
        "keywords": _as_text_list(_content_value(content, "keywords")),
        "authors": _as_text_list(_content_value(content, "authors")),
    }


def _note_date(note: dict[str, Any]) -> datetime.date | None:
    """Resolve a publication date from a note's timestamps.

    Prefers ``pdate`` -- the publication date OpenReview stamps when a paper is
    released -- and falls back to ``cdate``, the note's creation date.  Both are
    epoch milliseconds.

    Parameters
    ----------
    note : dict[str, Any]
        Note payload.

    Returns
    -------
    datetime.date | None
        Parsed date, or ``None`` when neither timestamp is usable.
    """
    for key in ("pdate", "cdate"):
        value = note.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            try:
                return datetime.datetime.fromtimestamp(value / 1000, tz=datetime.UTC).date()
            except (OverflowError, OSError, ValueError):
                continue
    return None
