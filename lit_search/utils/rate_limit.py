"""Cross-process, credential-scoped rate limiting.

Why this module exists
----------------------
``ConnectorBase`` used to enforce ``min_request_interval`` with a
``threading.Lock`` and a per-instance ``float``.  That is correct for two
threads sharing one connector object and wrong for every other case this
project cares about:

* Twenty agent processes, several CLI invocations and an MCP server each got
  their own clock, so a 1 request/second cumulative API key was violated by a
  factor of the process count.
* Even inside a single process the snowball path builds a fresh ``GetRunner``
  (and therefore fresh connectors) per DOI, which reset the in-memory clock to
  zero on every DOI.  The limiter did not fire at all on that path.

Both failures are silent in the direction that matters: over-fast requests earn
HTTP 429s, the retry budget is spent, and the connector layer historically
turned the resulting error into an empty result list, which is indistinguishable
from "this database genuinely has nothing".

Design
------
**File lock, no daemon.**  The shared clock lives in a small state file under a
machine-local directory.  Mutual exclusion is ``fcntl.flock`` taken on that same
file descriptor, so the lock and the data it protects cannot drift apart.  The
kernel releases an ``flock`` when the owning file descriptor is closed, which
includes abnormal process death, so a holder that is SIGKILLed mid-request
cannot wedge the lock.  See :func:`_acquire` for the live-but-hung case.

**Ticket reservation, not sleep-under-lock.**  A caller takes the lock, reads
the next free slot, writes ``slot + interval`` back as the new next free slot,
and releases.  The sleep happens *outside* the critical section.  The lock is
therefore held for microseconds regardless of how long the wait is, and N
concurrent callers queue up deterministically at exactly ``interval`` spacing.
A process that reserves a slot and then dies costs one slot of throughput, which
errs toward being too slow.  Every failure mode in this module is deliberately
biased that way.

**Wall clock, not monotonic.**  ``time.monotonic`` has a per-process epoch and
cannot be compared across processes.  The shared clock is therefore
``time.time``, which is comparable but can be stepped by NTP.  A stored slot
further in the future than any legitimate reservation is treated as corrupt and
reset; see :data:`_CLOCK_SKEW_GRACE`.

**Spacing is measured between request starts.**  The replaced implementation
measured from response completion, which spaced requests by
``interval + latency``.  Every published limit this project verified is phrased
per request start ("one request every three seconds", "1 request per second"),
so start-based spacing is the correct reading.  It is marginally more aggressive
than the old behaviour; the declared intervals already carry margin (Semantic
Scholar declares 1.1s against a 1.0s limit).

Scope identity
--------------
The limit belongs to the API key, not to the connector class and not to the
process, so the lock is keyed on ``(service, credential fingerprint)``:

* ``service`` is the *quota authority*, which is not always the connector name.
  PubMed and PMC are two connectors sharing one NCBI key and one bucket, so both
  declare ``rate_limit_service = "ncbi"``.  Semantic Scholar's search, citation
  and recommendation endpoints share one 1 RPS budget and are already one
  connector, but they would share a scope even if they were not.
* The fingerprint is ``sha256`` of the credential material, truncated.  The raw
  credential never reaches a path name, a log line or an error message.

**Services with no credential** are scoped to ``"<service>:anon"``, and the
choice is deliberate: an unauthenticated quota is enforced against the source IP
address the service sees, which is identical for every process on this host.  So
"this host, this service, unauthenticated" is the correct shared identity, and a
host-scoped file lock is the strongest thing a lock file can express.  arXiv's
terms are stricter still -- they aggregate "all of the machines under your
control" -- and no local mechanism can honour that; a multi-host deployment
would need a network broker, which R8 rejected for good reasons.

If a ported connector forgets to declare where it keeps its credential it falls
into the anonymous bucket for its service.  That shares a clock with every other
unkeyed user of the same service on this host, so the consequence is
over-throttling, never over-sending.  :func:`describe_limit` is logged once per
scope per process precisely so that a mis-declaration is visible in the output.

Surfacing waits
---------------
Waiting is recorded three ways, and none of them is an input a model can act on:

* Every wait increments a per-scope :class:`RateLimitStats` accumulator, which
  a runner or CLI renders with :func:`format_rate_limit_report` so that a slow
  run is diagnosable after the fact.
* A wait at or above :data:`_WAIT_NOTICE_SECONDS` logs at ``INFO`` naming the
  scope, the interval and the cause.  Shorter waits log at ``DEBUG``.
* The first use of a scope logs the active limit at ``INFO``.

There is no parameter, environment variable or API that shortens or skips a
wait.  ``LIT_SEARCH_RATELIMIT_DIR`` moves *where* the state lives, never
*whether* the limiter runs.
"""

from __future__ import annotations

import contextlib
import contextvars
import errno
import hashlib
import json
import logging
import os
import random
import re
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from lit_search.exceptions import (
    RateLimitBusyError,
    RateLimitLockTimeoutError,
    RateLimitStoreUnavailableError,
)

try:  # pragma: no cover - exercised implicitly on POSIX; see module docstring.
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl.
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

#: Environment variable that relocates the lock directory.  It changes *where*
#: the shared clock lives, never whether the limiter runs.
ENV_LOCK_DIR = "LIT_SEARCH_RATELIMIT_DIR"

#: Seconds to keep retrying a contended lock before giving up loudly.  The
#: critical section is microseconds long, so exhausting this budget means the
#: environment is broken rather than busy.
DEFAULT_LOCK_TIMEOUT = 30.0

#: Single waits at or above this many seconds are logged at INFO instead of
#: DEBUG, so that an unusually slow run leaves a trace without flooding logs
#: with one line per 1.1s Semantic Scholar wait.
_WAIT_NOTICE_SECONDS = 2.0

#: Slack allowed before a stored future slot is treated as clock skew or a
#: corrupt state file rather than a legitimate reservation.
_CLOCK_SKEW_GRACE = 60.0

#: On-disk state schema version.  A record with an unknown version is discarded
#: rather than misread.
_STATE_VERSION = 1
SLOWDOWN_SECONDS = 600.0
_MAX_SLOW_FACTOR = 8.0
_MAX_PAUSE_SECONDS = 300.0

#: Sentinel fingerprint used when a connector supplies no credential material.
ANONYMOUS_FINGERPRINT = "anon"

_FINGERPRINT_CHARS = 16
_SERVICE_MAX_CHARS = 64
_UNSAFE_SERVICE_CHARS = re.compile(r"[^a-z0-9._-]")

# Backoff bounds for the non-blocking lock retry loop.  Randomised so that many
# contending processes do not resynchronise on a fixed period.
_LOCK_RETRY_MIN = 0.001
_LOCK_RETRY_MAX = 0.010

_STATE_READ_BYTES = 4096


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RateLimitQuota:
    """A budget-shaped limit: at most *limit* requests per *window_seconds*.

    Several services in this project's port list are budget-shaped rather than
    interval-shaped (OpenAIRE allows 60 requests per hour, Zenodo 30 per minute
    on search endpoints, CORE a daily token allowance).  Pacing perfectly at an
    interval still exhausts those budgets and then returns zeros for the rest of
    the window, which reads exactly like empty recall.  Declaring a quota is
    optional; connectors that declare none are paced by interval alone.

    Attributes
    ----------
    limit : int
        Maximum number of requests permitted inside one window.
    window_seconds : float
        Length of the window in seconds.
    """

    limit: int
    window_seconds: float

    def __post_init__(self) -> None:
        """Reject nonsensical quotas at construction time.

        Raises
        ------
        ValueError
            When *limit* or *window_seconds* is not positive.
        """
        if self.limit <= 0:
            raise ValueError("RateLimitQuota.limit must be positive.")
        if self.window_seconds <= 0:
            raise ValueError("RateLimitQuota.window_seconds must be positive.")


@dataclass(frozen=True, slots=True)
class RateLimitScope:
    """Identity of a shared rate-limit bucket.

    Attributes
    ----------
    service : str
        The quota authority (for example ``"ncbi"``, ``"semantic_scholar"``).
        Normalised to lowercase and to path-safe characters.
    fingerprint : str
        Truncated SHA-256 of the credential material, or
        :data:`ANONYMOUS_FINGERPRINT` when no credential was supplied.
    """

    service: str
    fingerprint: str

    @classmethod
    def build(cls, service: str, credentials: Sequence[str | None] = ()) -> RateLimitScope:
        """Derive a scope from a service name and its credential material.

        Blank and ``None`` entries are dropped, so a connector that was
        constructed without its optional API key lands in the anonymous bucket
        for its service rather than in a bucket of its own.

        Parameters
        ----------
        service : str
            Quota authority name.
        credentials : Sequence[str | None]
            Credential strings identifying the quota.  Order is significant.

        Returns
        -------
        RateLimitScope
            The derived scope.

        Raises
        ------
        ValueError
            When *service* normalises to an empty string.
        """
        normalised = _UNSAFE_SERVICE_CHARS.sub("_", service.strip().lower())[:_SERVICE_MAX_CHARS]
        if not normalised:
            raise ValueError("RateLimitScope.service must not be empty.")
        material = [c.strip() for c in credentials if c is not None and c.strip()]
        if not material:
            return cls(service=normalised, fingerprint=ANONYMOUS_FINGERPRINT)
        digest = hashlib.sha256("\x00".join(material).encode("utf-8")).hexdigest()
        return cls(service=normalised, fingerprint=digest[:_FINGERPRINT_CHARS])

    @property
    def key(self) -> str:
        """Return a stable human-readable identifier for this bucket.

        Returns
        -------
        str
            ``"<service>:<fingerprint>"``.  Safe to log: the fingerprint is a
            one-way digest, never the credential.
        """
        return f"{self.service}:{self.fingerprint}"

    @property
    def is_anonymous(self) -> bool:
        """Return whether this scope carries no credential.

        Returns
        -------
        bool
            ``True`` when the scope is the shared unauthenticated bucket.
        """
        return self.fingerprint == ANONYMOUS_FINGERPRINT

    @property
    def filename(self) -> str:
        """Return the lock/state file name for this scope.

        Returns
        -------
        str
            File name, containing only path-safe characters.
        """
        return f"{self.service}.{self.fingerprint}.lock"


#: Why a caller waited.  ``"none"`` means the slot was already available.
WaitReason = Literal["none", "interval", "quota"]


@dataclass(frozen=True, slots=True)
class RateLimitWait:
    """Outcome of a single slot reservation.

    Attributes
    ----------
    scope_key : str
        The bucket that was reserved against.
    interval : float
        The declared minimum interval in seconds.
    waited : float
        Seconds actually slept.  Zero when the slot was already free.
    reason : WaitReason
        Whether the wait came from the interval, from a quota window, or not at
        all.
    """

    scope_key: str
    interval: float
    waited: float
    reason: WaitReason


@dataclass(frozen=True, slots=True)
class RateLimitStats:
    """Immutable per-scope wait totals for one process.

    This is a read-only diagnostic record.  Nothing consumes it to decide
    whether to wait; it exists so a slow run can be explained afterwards.

    Attributes
    ----------
    scope_key : str
        The bucket these totals belong to.
    interval : float
        Most recently declared interval for the bucket, in seconds.
    requests : int
        Slots reserved.
    waits : int
        Reservations that actually slept.
    total_wait_seconds : float
        Sum of all sleeps.
    max_wait_seconds : float
        Longest single sleep.
    quota_waits : int
        Reservations that slept because a quota window was exhausted.
    """

    scope_key: str
    interval: float
    requests: int
    waits: int
    total_wait_seconds: float
    max_wait_seconds: float
    quota_waits: int


@dataclass(slots=True)
class _State:
    """Mutable on-disk record for one bucket.

    Attributes
    ----------
    next_slot : float
        Wall-clock time of the next unreserved request slot.
    window_start : float
        Wall-clock start of the current quota window.
    count : int
        Requests reserved inside the current quota window.
    slow_until : float
        Wall-clock end of a slowdown imposed after a refusal (HTTP 429).
    slow_factor : float
        Interval multiplier while the slowdown lasts.
    """

    next_slot: float = 0.0
    window_start: float = 0.0
    count: int = 0
    slow_until: float = 0.0
    slow_factor: float = 1.0


# ---------------------------------------------------------------------------
# Clock seam (monkeypatched by tests; not a caller-facing parameter)
# ---------------------------------------------------------------------------


def _now() -> float:
    """Return the shared wall-clock time.

    Returns
    -------
    float
        Seconds since the epoch.  Wall clock rather than monotonic because the
        value must be comparable across processes.
    """
    return time.time()


def _sleep(seconds: float) -> None:
    """Sleep for *seconds*.

    Parameters
    ----------
    seconds : float
        Duration to sleep.  Non-positive values are ignored.
    """
    if seconds > 0:
        time.sleep(seconds)


# ---------------------------------------------------------------------------
# Lock directory
# ---------------------------------------------------------------------------


def rate_limit_dir() -> Path:
    """Return the directory holding shared rate-limit state, creating it.

    Resolution order: ``LIT_SEARCH_RATELIMIT_DIR``, then
    ``$XDG_CACHE_HOME/lit-search/ratelimit``, then
    ``~/.cache/lit-search/ratelimit``.  The cache directory is preferred over a
    runtime directory because every process that must share the clock (agents,
    CLI invocations, the MCP server) reliably resolves it to the same path for
    the same user, and credentials are per-user.

    Returns
    -------
    Path
        Existing directory.

    Raises
    ------
    RateLimitStoreUnavailableError
        When the directory cannot be created.  This is raised rather than
        silently falling back to unsynchronised limiting: an un-shared clock is
        exactly the defect this module exists to remove, and its symptom
        (HTTP 429 storms turning into empty result sets) is invisible.
    """
    override = os.environ.get(ENV_LOCK_DIR)
    if override:
        base = Path(override)
    else:
        cache_home = os.environ.get("XDG_CACHE_HOME")
        root = Path(cache_home) if cache_home else Path.home() / ".cache"
        base = root / "lit-search" / "ratelimit"
    try:
        base.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        raise RateLimitStoreUnavailableError(
            f"Cannot create the rate-limit state directory '{base}': {exc}. "
            f"Set {ENV_LOCK_DIR} to a writable path."
        ) from exc
    return base


# ---------------------------------------------------------------------------
# Locked state file
# ---------------------------------------------------------------------------


def _open_state_file(scope: RateLimitScope) -> int:
    """Open (creating if needed) the state file for *scope*.

    Parameters
    ----------
    scope : RateLimitScope
        Bucket whose file is wanted.

    Returns
    -------
    int
        Open file descriptor in read/write mode.

    Raises
    ------
    RateLimitStoreUnavailableError
        When the file cannot be opened.
    """
    path = rate_limit_dir() / scope.filename
    try:
        return os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    except OSError as exc:
        raise RateLimitStoreUnavailableError(
            f"Cannot open the rate-limit state file '{path}': {exc}."
        ) from exc


def _acquire(fd: int, scope: RateLimitScope, timeout: float) -> None:
    """Take an exclusive ``flock`` on *fd*, or fail loudly.

    A non-blocking retry loop is used rather than a blocking ``flock`` so that a
    holder which is alive but wedged cannot block every other process forever.
    A holder which *dies* needs no handling at all: the kernel drops its
    ``flock`` when its file descriptor is closed, including on SIGKILL, so the
    lock can never be left stale by process death.

    Parameters
    ----------
    fd : int
        Open file descriptor for the state file.
    scope : RateLimitScope
        Bucket being locked, for the error message.
    timeout : float
        Seconds to keep retrying.

    Raises
    ------
    RateLimitStoreUnavailableError
        When this platform provides no ``fcntl``.
    RateLimitLockTimeoutError
        When the lock could not be taken within *timeout*.
    """
    if fcntl is None:  # pragma: no cover - POSIX-only project; see docstring.
        raise RateLimitStoreUnavailableError(
            "Cross-process rate limiting requires fcntl.flock, which this "
            "platform does not provide. Supported on POSIX only."
        )
    deadline = _now() + timeout
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise RateLimitStoreUnavailableError(
                    f"Cannot lock the rate-limit state for '{scope.key}': {exc}."
                ) from exc
            if _now() >= deadline:
                raise RateLimitLockTimeoutError(
                    f"Timed out after {timeout:.1f}s waiting for the rate-limit lock on "
                    f"'{scope.key}'. The critical section is sub-millisecond, so this "
                    f"indicates a wedged holder or a broken filesystem, not contention."
                ) from exc
            # nosec B311: jitter only, not security-sensitive.
            _sleep(random.uniform(_LOCK_RETRY_MIN, _LOCK_RETRY_MAX))  # nosec B311


def _read_state(fd: int) -> _State:
    """Read the state record from *fd*, tolerating absence and corruption.

    Parameters
    ----------
    fd : int
        Locked file descriptor positioned anywhere.

    Returns
    -------
    _State
        The stored record, or a zeroed record when the file is empty, truncated,
        not valid JSON, or written by an unknown schema version.
    """
    os.lseek(fd, 0, os.SEEK_SET)
    raw = os.read(fd, _STATE_READ_BYTES)
    if not raw:
        return _State()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return _State()
    if not isinstance(payload, dict) or payload.get("v") != _STATE_VERSION:
        return _State()
    try:
        return _State(
            next_slot=float(payload.get("next_slot", 0.0)),
            window_start=float(payload.get("window_start", 0.0)),
            count=int(payload.get("count", 0)),
            slow_until=float(payload.get("slow_until", 0.0)),
            slow_factor=float(payload.get("slow_factor", 1.0)),
        )
    except (TypeError, ValueError):
        return _State()


def _write_state(fd: int, state: _State) -> None:
    """Overwrite the state record on *fd*.

    No ``fsync`` is issued.  The record is visible to other processes through
    the page cache immediately, which is what matters, and losing it requires a
    machine crash -- at which point there are no in-flight requests to pace.
    Paying a synchronous write on every HTTP request is not worth that.

    Parameters
    ----------
    fd : int
        Locked file descriptor.
    state : _State
        Record to persist.
    """
    payload = json.dumps(
        {
            "v": _STATE_VERSION,
            "next_slot": state.next_slot,
            "window_start": state.window_start,
            "count": state.count,
            "slow_until": state.slow_until,
            "slow_factor": state.slow_factor,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, payload)
    os.ftruncate(fd, len(payload))


@contextlib.contextmanager
def _locked_state(scope: RateLimitScope, timeout: float) -> Iterator[tuple[int, _State]]:
    """Yield the locked file descriptor and current state for *scope*.

    Parameters
    ----------
    scope : RateLimitScope
        Bucket to lock.
    timeout : float
        Seconds to keep retrying the lock.

    Yields
    ------
    tuple[int, _State]
        The locked descriptor and the record read under that lock.  Callers
        write back with :func:`_write_state` before leaving the block.
    """
    fd = _open_state_file(scope)
    try:
        _acquire(fd, scope, timeout)
        try:
            yield fd, _read_state(fd)
        finally:
            if fcntl is not None:  # pragma: no branch - POSIX-only project.
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Reservation planning (pure)
# ---------------------------------------------------------------------------


def _is_stale(state: _State, now: float, interval: float, quota: RateLimitQuota | None) -> bool:
    """Return whether *state* points implausibly far into the future.

    A record can only legitimately reserve up to one interval (plus one quota
    window) ahead of the caller that wrote it.  Anything beyond that plus
    :data:`_CLOCK_SKEW_GRACE` means the wall clock was stepped backwards, the
    file came from a different machine, or the record is corrupt.  Trusting it
    would stall every process for an unbounded time.

    Parameters
    ----------
    state : _State
        Record read from disk.
    now : float
        Current wall-clock time.
    interval : float
        Declared minimum interval in seconds.
    quota : RateLimitQuota | None
        Declared quota, if any.

    Returns
    -------
    bool
        ``True`` when the record must be discarded.
    """
    window = quota.window_seconds if quota is not None else 0.0
    slowed = interval * _MAX_SLOW_FACTOR + _MAX_PAUSE_SECONDS
    horizon = now + slowed + window + _CLOCK_SKEW_GRACE
    return state.next_slot > horizon or state.window_start > horizon


def _plan_quota(
    slot: float, state: _State, quota: RateLimitQuota
) -> tuple[float, float, int, bool]:
    """Apply the quota window to a candidate *slot*.

    Parameters
    ----------
    slot : float
        Candidate slot time from interval pacing.
    state : _State
        Record read from disk.
    quota : RateLimitQuota
        Declared quota.

    Returns
    -------
    tuple[float, float, int, bool]
        Adjusted slot, new window start, new in-window count, and whether the
        quota (rather than the interval) forced the wait.
    """
    window_start = state.window_start
    count = state.count
    if slot >= window_start + quota.window_seconds:
        window_start, count = slot, 0
    forced = False
    if count >= quota.limit:
        slot = max(slot, window_start + quota.window_seconds)
        window_start, count = slot, 0
        forced = True
    return slot, window_start, count + 1, forced


def _plan(
    state: _State,
    now: float,
    interval: float,
    quota: RateLimitQuota | None,
    scope_key: str,
) -> tuple[float, _State, WaitReason]:
    """Compute the slot to take and the record to write back.

    Parameters
    ----------
    state : _State
        Record read from disk.
    now : float
        Current wall-clock time.
    interval : float
        Declared minimum interval in seconds.
    quota : RateLimitQuota | None
        Declared quota, if any.
    scope_key : str
        Bucket identifier, used only for the stale-state warning.

    Returns
    -------
    tuple[float, _State, WaitReason]
        The reserved slot time, the record to persist, and why the caller waits.
    """
    if _is_stale(state, now, interval, quota):
        logger.warning(
            "[rate-limit] Discarding implausible state for '%s' (next slot %.3fs in the "
            "future). Cause is usually a wall-clock step; proceeding from now.",
            scope_key,
            state.next_slot - now,
        )
        state = _State()

    slot = max(now, state.next_slot)
    reason: WaitReason = "interval" if slot > now else "none"
    window_start, count = state.window_start, state.count
    if quota is not None:
        slot, window_start, count, forced = _plan_quota(slot, state, quota)
        if forced:
            reason = "quota"
    slowed = now < state.slow_until
    factor = state.slow_factor if slowed else 1.0
    return (
        slot,
        _State(
            next_slot=slot + interval * factor,
            window_start=window_start,
            count=count,
            slow_until=state.slow_until if slowed else 0.0,
            slow_factor=factor,
        ),
        reason,
    )


# ---------------------------------------------------------------------------
# Statistics (read-only diagnostics)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Accumulator:
    """Mutable per-scope wait totals.

    Attributes
    ----------
    interval : float
        Most recently declared interval.
    requests : int
        Slots reserved.
    waits : int
        Reservations that slept.
    total : float
        Sum of sleeps.
    longest : float
        Longest single sleep.
    quota_waits : int
        Reservations blocked by a quota window.
    """

    interval: float = 0.0
    requests: int = 0
    waits: int = 0
    total: float = 0.0
    longest: float = 0.0
    quota_waits: int = 0


_stats_lock = threading.Lock()
_stats: dict[str, _Accumulator] = {}
_announced: set[str] = set()


def _record(wait: RateLimitWait) -> None:
    """Fold *wait* into this process's statistics.

    Parameters
    ----------
    wait : RateLimitWait
        Outcome of one reservation.
    """
    with _stats_lock:
        acc = _stats.setdefault(wait.scope_key, _Accumulator())
        acc.interval = wait.interval
        acc.requests += 1
        if wait.waited > 0:
            acc.waits += 1
            acc.total += wait.waited
            acc.longest = max(acc.longest, wait.waited)
        if wait.reason == "quota":
            acc.quota_waits += 1


def rate_limit_stats() -> tuple[RateLimitStats, ...]:
    """Return per-scope wait totals for this process, worst first.

    Returns
    -------
    tuple[RateLimitStats, ...]
        Immutable snapshot ordered by total wait, descending.
    """
    with _stats_lock:
        snapshot = [
            RateLimitStats(
                scope_key=key,
                interval=acc.interval,
                requests=acc.requests,
                waits=acc.waits,
                total_wait_seconds=acc.total,
                max_wait_seconds=acc.longest,
                quota_waits=acc.quota_waits,
            )
            for key, acc in _stats.items()
        ]
    return tuple(sorted(snapshot, key=lambda s: s.total_wait_seconds, reverse=True))


def reset_rate_limit_stats() -> None:
    """Clear this process's wait statistics and first-use announcements."""
    with _stats_lock:
        _stats.clear()
        _announced.clear()


def format_rate_limit_report(stats: Sequence[RateLimitStats] | None = None) -> str:
    """Render wait statistics as human-readable lines.

    Parameters
    ----------
    stats : Sequence[RateLimitStats] | None
        Snapshot to render.  Defaults to :func:`rate_limit_stats`.

    Returns
    -------
    str
        One line per scope that actually waited, or ``""`` when nothing waited.
    """
    rows = list(stats if stats is not None else rate_limit_stats())
    lines = [
        f"[rate-limit] {row.scope_key}: waited {row.total_wait_seconds:.1f}s across "
        f"{row.waits}/{row.requests} requests (interval {row.interval:.2f}s, "
        f"longest {row.max_wait_seconds:.1f}s"
        + (f", {row.quota_waits} quota stalls" if row.quota_waits else "")
        + ")"
        for row in rows
        if row.waits
    ]
    return "\n".join(lines)


def describe_limit(scope: RateLimitScope, interval: float, quota: RateLimitQuota | None) -> str:
    """Render the active limit for a scope as one line.

    Parameters
    ----------
    scope : RateLimitScope
        Bucket being described.
    interval : float
        Declared minimum interval in seconds.
    quota : RateLimitQuota | None
        Declared quota, if any.

    Returns
    -------
    str
        Description naming the interval, any quota, and whether a credential
        was detected.
    """
    credential = "no credential (shared host bucket)" if scope.is_anonymous else "credential-scoped"
    parts = [f"min interval {interval:.2f}s", credential]
    if quota is not None:
        parts.insert(1, f"quota {quota.limit}/{quota.window_seconds:.0f}s")
    return f"[rate-limit] {scope.key}: " + ", ".join(parts)


def _announce(scope: RateLimitScope, interval: float, quota: RateLimitQuota | None) -> None:
    """Log the active limit the first time a scope is used in this process.

    Parameters
    ----------
    scope : RateLimitScope
        Bucket being used.
    interval : float
        Declared minimum interval in seconds.
    quota : RateLimitQuota | None
        Declared quota, if any.
    """
    with _stats_lock:
        if scope.key in _announced:
            return
        _announced.add(scope.key)
    logger.info("%s", describe_limit(scope, interval, quota))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


_DEADLINE: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "lit_search_request_deadline", default=None
)


@contextlib.contextmanager
def request_deadline(at: float | None) -> Iterator[None]:
    """Refuse slots that lie past wall-clock time *at* for calls made inside.

    Context variables do not cross into new threads, so a worker thread must
    enter this itself.

    Parameters
    ----------
    at : float | None
        Wall-clock deadline (``time.time()`` scale); ``None`` lifts it.

    Yields
    ------
    None
    """
    token = _DEADLINE.set(at)
    try:
        yield
    finally:
        _DEADLINE.reset(token)


def reserve_slot(
    scope: RateLimitScope,
    min_interval: float,
    *,
    quota: RateLimitQuota | None = None,
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
) -> RateLimitWait:
    """Reserve the next request slot for *scope*, sleeping until it arrives.

    Reservation is atomic across processes; the sleep is not held under the
    lock.  Callers must issue their HTTP request immediately after this returns.

    Parameters
    ----------
    scope : RateLimitScope
        Bucket to reserve against.
    min_interval : float
        Declared minimum seconds between request starts.
    quota : RateLimitQuota | None
        Optional budget-shaped limit layered on top of the interval.
    lock_timeout : float
        Seconds to keep retrying a contended lock before failing loudly.

    Returns
    -------
    RateLimitWait
        What was reserved and how long the caller slept.

    Raises
    ------
    RateLimitLockTimeoutError
        When the lock could not be taken.  Deliberately *not* a
        :class:`~lit_search.exceptions.ConnectorError`, because connector code
        converts those into empty result lists and an unpaced run must never be
        mistaken for a database with nothing to say.
    RateLimitStoreUnavailableError
        When shared state cannot be reached at all.
    RateLimitBusyError
        When a :func:`request_deadline` is active and the next free slot lies
        past it.  Nothing is reserved.
    """
    if min_interval <= 0 and quota is None:
        # Nothing to enforce (for example the web-scraping connector, whose
        # real limit is per-host and is not expressible in this scope). Skip
        # the file entirely rather than paying I/O per request.
        wait = RateLimitWait(scope.key, 0.0, 0.0, "none")
        _record(wait)
        return wait

    _announce(scope, min_interval, quota)
    deadline = _DEADLINE.get()
    with _locked_state(scope, lock_timeout) as (fd, state):
        now = _now()
        slot, new_state, reason = _plan(state, now, min_interval, quota, scope.key)
        if deadline is not None and slot > deadline:
            raise RateLimitBusyError(scope.key, slot - now, deadline - now)
        _write_state(fd, new_state)

    delay = max(0.0, slot - now)
    wait = RateLimitWait(scope.key, min_interval, delay, reason)
    _log_wait(wait)
    _record(wait)
    _sleep(delay)
    return wait


def penalize(
    scope: RateLimitScope,
    pause: float,
    *,
    slow_seconds: float = SLOWDOWN_SECONDS,
    max_factor: float = _MAX_SLOW_FACTOR,
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
) -> float:
    """Back every process off a bucket after the service refused a request.

    The next free slot moves at least ``pause`` seconds ahead, so peers that
    share the credential stop instead of each discovering the refusal on its
    own, and the interval is multiplied (doubling per refusal, up to
    :data:`_MAX_SLOW_FACTOR`) for ``slow_seconds`` after the latest refusal.

    Parameters
    ----------
    scope : RateLimitScope
        Bucket that was refused.
    pause : float
        Seconds no process may send a request to this bucket.
    slow_seconds : float
        How long the widened interval lasts.
    max_factor : float
        Ceiling on the interval multiplier.  ``1.0`` disables widening, for a
        service whose refusals do not depend on the request rate.
    lock_timeout : float
        Seconds to keep retrying a contended lock.

    Returns
    -------
    float
        The interval multiplier now in force.
    """
    with _locked_state(scope, lock_timeout) as (fd, state):
        now = _now()
        factor = max(
            1.0, min(max_factor, (state.slow_factor * 2) if now < state.slow_until else 2.0)
        )
        state.next_slot = max(state.next_slot, now + min(pause, _MAX_PAUSE_SECONDS))
        state.slow_until = now + slow_seconds
        state.slow_factor = factor
        _write_state(fd, state)
    logger.warning(
        "[rate-limit] '%s' refused a request: all processes pause %.1fs, then space requests "
        "%.0fx wider for %.0f s.",
        scope.key,
        pause,
        factor,
        slow_seconds,
    )
    return factor


def _log_wait(wait: RateLimitWait) -> None:
    """Surface a wait in the log, loudly enough to notice and quietly enough to read.

    Parameters
    ----------
    wait : RateLimitWait
        Outcome of one reservation.
    """
    if wait.waited <= 0:
        return
    message = "[rate-limit] %s: waiting %.2fs (%s, interval %.2fs)"
    args = (wait.scope_key, wait.waited, wait.reason, wait.interval)
    if wait.waited >= _WAIT_NOTICE_SECONDS:
        logger.info(message, *args)
    else:
        logger.debug(message, *args)
