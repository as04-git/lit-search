"""Abstract base class for external API connectors.

Provides shared HTTP infrastructure (rate limiting, credential injection,
request/response logging) so that every module that talks to an external
service inherits a consistent, production-ready networking layer.
"""

from __future__ import annotations

import contextlib
import logging
import random
import time
from abc import ABC, abstractmethod
from types import TracebackType
from typing import ClassVar
from urllib.parse import urlencode

import requests

from lit_search.core.retrieval_audit import record_request_attempt, record_request_failure
from lit_search.utils import request_log
from lit_search.utils.logging_config import SENSITIVE_PARAM_NAMES
from lit_search.utils.rate_limit import RateLimitQuota, RateLimitScope, penalize, reserve_slot
from lit_search.utils.version import package_version

logger = logging.getLogger(__name__)

# Alias kept for internal use; the canonical definition lives in
# lit_search.utils.logging_config so the SensitiveDataFilter can share it.
_SENSITIVE_PARAM_NAMES: frozenset[str] = SENSITIVE_PARAM_NAMES

# Header names (compared case-insensitively) that carry API credentials
# and must be redacted before logging.
_SENSITIVE_HEADER_NAMES: frozenset[str] = frozenset(
    {
        "x-els-apikey",
        "x-api-key",
        "x-apikey",
        "authorization",
    }
)


def user_agent() -> str:
    """Return the User-Agent every connector sends.

    Some services throttle the generic ``python-requests`` agent shared by
    countless scrapers, and several ask for a contact address.  The address
    comes from ``LIT_SEARCH_EMAIL``; without it the agent names only the tool.

    Returns
    -------
    str
        ``lit-search/<version> (+<repo>; mailto:<address>)``.
    """
    from lit_search import credentials

    contact = credentials.get("LIT_SEARCH_EMAIL")
    suffix = f"; mailto:{contact}" if contact else ""
    return f"lit-search/{package_version()} (+https://github.com/as04-git/lit-search{suffix})"


class ConnectorBase(ABC):
    """Abstract base class for external API connectors.

    Provides rate-limited HTTP helpers (``_get`` / ``_post``), automatic
    credential injection via ``_prepare_params`` / ``_prepare_headers``,
    and debug-level request/response logging with sensitive-parameter
    redaction.

    A :class:`requests.Session` is created lazily on the first HTTP call
    so that TCP+TLS connections are pooled and reused across requests to
    the same host.

    Subclasses must implement :attr:`name` and :attr:`min_request_interval`.

    Rate limiting is **cross-process** and keyed by credential; see
    :mod:`lit_search.utils.rate_limit` and :attr:`rate_limit_scope`.  A
    per-instance clock would be defeated by the concurrency this project
    targets (many agent processes, CLI invocations and an MCP server sharing one
    API key) and is additionally defeated inside a single process wherever
    connectors are rebuilt per item, as the snowball path does.
    """

    # Default HTTP timeout in seconds for all requests.  Subclasses or callers
    # can override by setting ``self._timeout`` in ``__init__``.
    _timeout: float = 30.0

    #: Instance attribute names holding the credential material that identifies
    #: this connector's quota.  Missing, ``None`` and blank attributes are
    #: dropped, which lands the connector in the shared anonymous bucket for its
    #: service.  The default matches the six existing keyed connectors; override
    #: it when the credential lives elsewhere (CrossRef's polite pool is keyed
    #: by ``_email``) or when there is none (arXiv).
    #:
    #: A ported connector that stores its key under another name and forgets to
    #: declare it here shares the anonymous clock for its service, so the
    #: consequence is over-throttling rather than over-sending.  The active
    #: scope is logged once per process to make that visible.
    _credential_attrs: ClassVar[tuple[str, ...]] = ("_api_key",)

    def __init__(self) -> None:  # noqa: B027
        """Initialise the connector.

        Subclasses **must** call ``super().__init__()``.  The body is
        deliberately empty and deliberately kept: there is no per-instance
        rate-limiter state to set up any more, because the clock is shared on
        disk and keyed by credential so that it survives object churn.  Keeping
        the method preserves the co-operative ``super().__init__()`` chain that
        every connector already calls.

        ``B027`` is silenced rather than obeyed: this is not an optional hook
        for subclasses to override, it is a base initialiser they invoke.
        """

    # ------------------------------------------------------------------
    # Retry configuration
    # ------------------------------------------------------------------

    # HTTP status codes that trigger an automatic retry.
    _retryable_status_codes: frozenset[int] = frozenset({429, 502, 503, 504})

    # Maximum number of retry attempts for connection errors / non-429 retryable
    # status codes (excluding the initial request).
    _max_retries: int = 3

    # Maximum number of retry attempts exclusively for HTTP 429 responses.
    # This budget is independent of _max_retries so that transient timeouts
    # do not consume the rate-limit retry allowance.
    _max_rate_limit_retries: int = 3

    # Base delay in seconds for exponential backoff (delay = base * 2^attempt).
    _retry_base_delay: float = 1.0

    # Base delay used when retrying HTTP 429 (rate-limited) responses.
    # Rate-limit back-off needs longer waits than generic transient errors.
    _rate_limit_base_delay: float = 5.0

    # Maximum delay cap in seconds to prevent excessively long waits.
    _retry_max_delay: float = 60.0

    # How every process sharing the credential backs off after a 429: the
    # interval is widened up to this factor for this many seconds.  Right for a
    # service that refuses when the shared rate is too high; wrong for one that
    # refuses at any rate (see SemanticScholarConnector).
    _refusal_max_factor: float = 8.0
    _refusal_slow_seconds: float = 600.0

    # ------------------------------------------------------------------
    # HTTP Session (connection pooling)
    # ------------------------------------------------------------------

    def _get_session(self) -> requests.Session:
        """Return the shared :class:`requests.Session`, creating it lazily.

        Using a session allows ``urllib3`` to keep TCP+TLS connections alive
        across consecutive requests to the same host, which significantly
        reduces latency for connectors that issue many paginated calls.

        Returns
        -------
        requests.Session
            The reusable HTTP session bound to this connector instance.
        """
        # Instance attribute created on first access; avoids requiring
        # subclasses to call super().__init__().
        if not hasattr(self, "_http_session"):
            self._http_session = requests.Session()
            self._http_session.headers["User-Agent"] = user_agent()
            self._apply_transport_configuration(self._http_session)
        return self._http_session

    def configure_transport(self, *, proxy: str | None, ssl_verify: bool) -> None:
        """Configure HTTP transport used by this connector.

        Search connectors are normally created by :class:`SearchRunner`, so
        transport policy is applied after their source-specific constructor
        arguments.  Keeping this operation on the shared base prevents a new
        connector from being added to the registry while silently ignoring the
        facade's proxy or TLS policy.

        Parameters
        ----------
        proxy : str | None
            Proxy URL used for both HTTP and HTTPS requests.
        ssl_verify : bool
            Whether TLS certificates are verified.
        """
        self._proxy = proxy
        self._ssl_verify = ssl_verify
        if hasattr(self, "_http_session"):
            self._apply_transport_configuration(self._http_session)

    def _apply_transport_configuration(self, session: requests.Session) -> None:
        """Apply the configured proxy and TLS policy to *session*."""
        proxy = getattr(self, "_proxy", None)
        proxies = getattr(session, "proxies", None)
        if proxies is None:
            proxies = {}
            session.proxies = proxies
        else:
            proxies.clear()
        if proxy:
            proxies.update({"http": proxy, "https": proxy})
        session.verify = getattr(self, "_ssl_verify", True)

    def close(self) -> None:
        """Close the underlying HTTP session, releasing pooled connections.

        Safe to call multiple times or even if no request was ever made.
        """
        if hasattr(self, "_http_session"):
            self._http_session.close()
            del self._http_session

    # Context-manager protocol so connectors can be used with ``with``.

    def __enter__(self) -> ConnectorBase:
        """Enter the runtime context and return the connector itself.

        Returns
        -------
        ConnectorBase
            ``self``, allowing ``with Connector() as c: ...`` usage.
        """
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        """Exit the runtime context, closing the HTTP session."""
        self.close()

    def __del__(self) -> None:
        """Best-effort cleanup: close the session when garbage-collected.

        This is a safety-net only; callers should prefer :meth:`close` or
        a ``with`` block for deterministic resource release.
        """
        self.close()

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable connector identifier used in log messages.

        Returns
        -------
        str
            Connector name (e.g. ``"arxiv"``, ``"crossref"``).
        """

    @property
    @abstractmethod
    def min_request_interval(self) -> float:
        """Minimum number of seconds that must elapse between consecutive HTTP requests.

        This stays an abstract property because it is a real contract: a
        connector that does not declare its rate cannot be built.  Note the
        precise strength of that guarantee -- an unimplemented abstract property
        fails at **instantiation**, not at class definition.  The class body
        will import and subclass cleanly; the error surfaces the first time
        somebody constructs it.  That is enough for a registry that builds every
        connector it registers, and it is weaker than "cannot exist".

        The value is the declared interval only.  Where it is *enforced* is
        :attr:`rate_limit_scope`, which is shared across processes.

        Returns
        -------
        float
            Interval in seconds.  Zero or less disables interval pacing for
            this connector (see :class:`WebScrapingConnector`, whose real limit
            is per-host and not expressible as one scope).
        """

    @property
    def rate_limit_service(self) -> str:
        """Return the name of the quota authority this connector draws on.

        Defaults to :attr:`name`, which is right whenever one connector maps to
        one service.  Override when several connectors share one budget: PubMed
        and PMC are distinct databases that share a single NCBI key and a single
        request bucket, so both return ``"ncbi"``.

        Returns
        -------
        str
            Quota-authority identifier.
        """
        return self.name

    @property
    def request_quota(self) -> RateLimitQuota | None:
        """Return a budget-shaped limit layered on top of the interval, if any.

        Several services cap requests per hour or per day rather than per
        second.  Pacing perfectly at an interval still exhausts such a budget
        and then returns errors for the rest of the window, which reads as empty
        recall.  Connectors whose service publishes only an interval return
        ``None``.

        Returns
        -------
        RateLimitQuota | None
            Declared quota, or ``None``.
        """
        return None

    def _credential_material(self) -> tuple[str, ...]:
        """Return the credential strings that identify this connector's quota.

        Reads the attributes named by :attr:`_credential_attrs`.  Values are
        never logged or written to a path; :class:`RateLimitScope` hashes them.

        Returns
        -------
        tuple[str, ...]
            Present, non-blank credential values, in declaration order.
        """
        values: list[str] = []
        for attr in self._credential_attrs:
            value = getattr(self, attr, None)
            if isinstance(value, str) and value.strip():
                values.append(value)
        return tuple(values)

    @property
    def rate_limit_scope(self) -> RateLimitScope:
        """Return the shared bucket this connector's requests are paced against.

        The bucket is ``(quota authority, credential fingerprint)``.  It is
        deliberately independent of the connector object and of the process, so
        that two instances built from the same key -- in one process, in twenty
        agent processes, in the CLI, or in the MCP server -- share one clock.

        Returns
        -------
        RateLimitScope
            Cached scope for this instance.
        """
        # Cached on first access rather than in __init__ so that subclasses
        # which assign their credential after super().__init__() still get the
        # right scope, matching how _get_session defers session creation.
        if not hasattr(self, "_rate_limit_scope"):
            self._rate_limit_scope = RateLimitScope.build(
                self.rate_limit_service, self._credential_material()
            )
        return self._rate_limit_scope

    def _rate_limit(self) -> None:
        """Reserve the next request slot for this connector's credential.

        Blocks until the slot arrives.  Enforcement is cross-process: the shared
        clock is a small file locked with ``fcntl.flock``, so it survives
        process death, needs no daemon, and cannot be bypassed by rebuilding the
        connector.  The wait is surfaced in the log and in
        :func:`~lit_search.utils.rate_limit.rate_limit_stats`; it is never an
        input any caller can shorten.
        """
        reserve_slot(
            self.rate_limit_scope,
            self.min_request_interval,
            quota=self.request_quota,
        )

    def _retry_delay(
        self,
        attempt: int,
        response: requests.Response | None = None,
        *,
        base_delay: float | None = None,
    ) -> float:
        """Compute the delay before the next retry attempt.

        Uses exponential backoff with jitter.  If the response contains a
        ``Retry-After`` header (common with 429 responses), that value is
        used as the minimum delay.

        Parameters
        ----------
        attempt : int
            Zero-based retry attempt number (0 = first retry).
        response : requests.Response | None
            The HTTP response that triggered the retry, if available.
        base_delay : float | None
            Override the exponential-backoff base delay.  When ``None`` the
            instance attribute :attr:`_retry_base_delay` is used.

        Returns
        -------
        float
            Delay in seconds before the next retry.
        """
        effective_base = base_delay if base_delay is not None else self._retry_base_delay
        # Exponential backoff: base * 2^attempt, capped at max_delay.
        backoff: float = min(effective_base * (2**attempt), self._retry_max_delay)

        # Honour Retry-After header when present (429 responses).
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                # Non-numeric Retry-After (e.g. HTTP-date); ignore.
                with contextlib.suppress(ValueError, TypeError):
                    backoff = max(backoff, float(retry_after))

        # Add jitter (0-25 %) to avoid thundering-herd effects.
        # nosec B311: random is intentionally used here for timing jitter only, not for
        # cryptographic or security-sensitive purposes.
        jitter: float = backoff * 0.25 * random.random()  # nosec B311
        return backoff + jitter

    def _request_with_retry(
        self,
        method: str,
        url: str,
        params: dict | None = None,
        headers: dict | None = None,
        json_body: dict | list | None = None,
    ) -> requests.Response:
        """Execute an HTTP request with automatic retry on transient failures.

        Retries are triggered for status codes in :attr:`_retryable_status_codes`
        and for :class:`requests.ConnectionError` /
        :class:`requests.Timeout` exceptions.  Uses exponential backoff with
        jitter between attempts.

        HTTP 429 (rate-limited) responses use a **separate** retry counter
        (:attr:`_max_rate_limit_retries`) so that transient timeouts or
        connection errors do not consume the rate-limit retry allowance.
        The back-off base for 429 retries is :attr:`_rate_limit_base_delay`
        (typically larger than :attr:`_retry_base_delay`).

        Parameters
        ----------
        method : str
            HTTP method (``"GET"`` or ``"POST"``).
        url : str
            Target URL.
        params : dict | None
            Query parameters (already prepared).
        headers : dict | None
            HTTP headers (already prepared).
        json_body : dict | list | None
            JSON payload for POST requests.

        Returns
        -------
        requests.Response
            Successful HTTP response.

        Raises
        ------
        requests.HTTPError
            On non-2xx status codes after all retries are exhausted.
        requests.ConnectionError
            On persistent connection failures.
        requests.Timeout
            On persistent timeout failures.
        """
        # Every request that leaves this method is recorded on the active
        # retrieval audit *before* the caller's own except clause can see it.
        # A connector that catches its own page failure and keeps paginating
        # has still lost records; the audit remembers even when the connector
        # does not mention it, and that is what stops the swallowed-exception
        # route to a false COMPLETE.
        record_request_attempt()
        try:
            return self._request_with_retry_uncounted(
                method=method,
                url=url,
                params=params,
                headers=headers,
                json_body=json_body,
            )
        except Exception as exc:
            record_request_failure(exc)
            raise

    def _request_with_retry_uncounted(
        self,
        method: str,
        url: str,
        params: dict | None = None,
        headers: dict | None = None,
        json_body: dict | list | None = None,
    ) -> requests.Response:
        """Execute the retry loop itself, without touching the retrieval audit.

        Split out of :meth:`_request_with_retry` so that the audit bookkeeping
        wraps every exit path -- including the ``raise`` at the end of the
        retry budget -- without another level of indentation in the loop.

        Parameters
        ----------
        method : str
            HTTP method (``"GET"`` or ``"POST"``).
        url : str
            Target URL.
        params : dict | None
            Query parameters (already prepared).
        headers : dict | None
            HTTP headers (already prepared).
        json_body : dict | list | None
            JSON payload for POST requests.

        Returns
        -------
        requests.Response
            Successful HTTP response.

        Raises
        ------
        requests.HTTPError
            On non-2xx status codes after all retries are exhausted.
        requests.ConnectionError
            On persistent connection failures.
        requests.Timeout
            On persistent timeout failures.
        """
        # general_attempt counts retries for connection errors, timeouts, and
        # non-429 retryable status codes.
        # rate_limit_attempt counts retries exclusively for HTTP 429 so that
        # transient failures do not consume the rate-limit retry allowance.
        general_attempt: int = 0
        rate_limit_attempt: int = 0

        while True:
            sent: float | None = None
            try:
                self._rate_limit()
                self._log_request(url, params or None, method=method, headers=headers)

                sent = request_log.now()
                if method == "POST":
                    response = self._get_session().post(
                        url,
                        json=json_body,
                        params=params or None,
                        headers=headers or None,
                        timeout=self._timeout,
                    )
                else:
                    response = self._get_session().get(
                        url,
                        params=params or None,
                        headers=headers or None,
                        timeout=self._timeout,
                    )

                request_log.record(
                    scope=self.rate_limit_scope.key,
                    method=method,
                    url=url,
                    sent=sent,
                    status=response.status_code,
                    latency=request_log.now() - sent,
                )

                # No post-response clock update: the slot was already reserved
                # before the request went out.  Stamping completion time here
                # would double-count the request's own latency, and it could
                # not hold back a peer process that had already reserved its
                # own slot.
                self._log_response(response)

                if response.status_code == 429:
                    # 429 uses its own retry budget so general retries (e.g.
                    # timeouts on earlier attempts) do not exhaust it.
                    if rate_limit_attempt < self._max_rate_limit_retries:
                        delay = self._retry_delay(
                            rate_limit_attempt,
                            response,
                            base_delay=self._rate_limit_base_delay,
                        )
                        logger.debug(
                            "[%s] HTTP 429 from %s: retrying in %.1fs (attempt %d/%d).",
                            self.name,
                            url,
                            delay,
                            rate_limit_attempt + 1,
                            self._max_rate_limit_retries,
                        )
                        penalize(
                            self.rate_limit_scope,
                            delay,
                            slow_seconds=self._refusal_slow_seconds,
                            max_factor=self._refusal_max_factor,
                        )
                        rate_limit_attempt += 1
                        continue
                    # Rate-limit retries exhausted: peers still back off, then raise.
                    penalize(
                        self.rate_limit_scope,
                        self._retry_delay(
                            rate_limit_attempt, response, base_delay=self._rate_limit_base_delay
                        ),
                        slow_seconds=self._refusal_slow_seconds,
                        max_factor=self._refusal_max_factor,
                    )

                elif (
                    response.status_code in self._retryable_status_codes
                    and general_attempt < self._max_retries
                ):
                    delay = self._retry_delay(general_attempt, response)
                    logger.warning(
                        "[%s] HTTP %d from %s: retrying in %.1fs (attempt %d/%d).",
                        self.name,
                        response.status_code,
                        url,
                        delay,
                        general_attempt + 1,
                        self._max_retries,
                    )
                    time.sleep(delay)
                    general_attempt += 1
                    continue
                    # General retries exhausted: fall through to raise.

                if not response.ok:
                    try:
                        body = response.json()
                    except ValueError:
                        body = response.text
                    logger.debug(
                        "[%s] HTTP %d response body from %s: %s",
                        self.name,
                        response.status_code,
                        url,
                        body,
                    )
                response.raise_for_status()
                return response

            except (requests.ConnectionError, requests.Timeout) as exc:
                if sent is not None:
                    request_log.record(
                        scope=self.rate_limit_scope.key,
                        method=method,
                        url=url,
                        sent=sent,
                        status=type(exc).__name__,
                        latency=request_log.now() - sent,
                    )
                if general_attempt < self._max_retries:
                    delay = self._retry_delay(general_attempt)
                    logger.warning(
                        "[%s] %s for %s: retrying in %.1fs (attempt %d/%d).",
                        self.name,
                        type(exc).__name__,
                        url,
                        delay,
                        general_attempt + 1,
                        self._max_retries,
                    )
                    time.sleep(delay)
                    general_attempt += 1
                    continue
                raise

    def _prepare_params(self, params: dict) -> dict:
        """Augment query parameters before the request is sent.

        The default implementation returns *params* unchanged.  Subclasses
        override this to inject API keys or other credentials into the query
        string.

        Parameters
        ----------
        params : dict
            Raw query parameters supplied by the caller.

        Returns
        -------
        dict
            Augmented parameters.
        """
        return params

    def _prepare_headers(self, headers: dict) -> dict:
        """Return a copy of the provided headers without modifications.

        Subclasses should call ``super()._prepare_headers(headers)`` and then
        add their own keys (API tokens, Accept types, etc.).

        Parameters
        ----------
        headers : dict
            Raw HTTP headers supplied by the caller.

        Returns
        -------
        dict
            Copy of the input headers.
        """
        return dict(headers)

    def _get(
        self,
        url: str,
        params: dict | None = None,
        headers: dict | None = None,
    ) -> requests.Response:
        """Perform a rate-limited, logged GET request with automatic retry.

        Calls :meth:`_prepare_params` and :meth:`_prepare_headers` so
        subclasses can inject credentials without duplicating the
        rate-limiting and logging boilerplate.  Transient failures
        (429, 502, 503, 504 and connection/timeout errors) are retried
        with exponential backoff.

        Parameters
        ----------
        url : str
            Target URL.
        params : dict | None
            Query parameters (before credential injection).
        headers : dict | None
            HTTP headers (before credential injection).

        Returns
        -------
        requests.Response
            HTTP response.

        Raises
        ------
        requests.HTTPError
            On non-2xx status codes after all retries are exhausted.
        """
        prepared_params = self._prepare_params(dict(params) if params else {})
        prepared_headers = self._prepare_headers(dict(headers) if headers else {})
        return self._request_with_retry(
            method="GET",
            url=url,
            params=prepared_params,
            headers=prepared_headers,
        )

    def _post(
        self,
        url: str,
        json_body: dict | list | None = None,
        params: dict | None = None,
        headers: dict | None = None,
    ) -> requests.Response:
        """Perform a rate-limited, logged POST request with automatic retry.

        Mirrors :meth:`_get` but sends a JSON body via ``requests.post``.
        Calls :meth:`_prepare_params` and :meth:`_prepare_headers` for
        credential injection and browser-header defaults.

        Parameters
        ----------
        url : str
            Target URL.
        json_body : dict | list | None
            JSON-serialisable payload for the request body.
        params : dict | None
            Query parameters (before credential injection).
        headers : dict | None
            HTTP headers (before credential injection).

        Returns
        -------
        requests.Response
            HTTP response.

        Raises
        ------
        requests.HTTPError
            On non-2xx status codes after all retries are exhausted.
        """
        prepared_params = self._prepare_params(dict(params) if params else {})
        prepared_headers = self._prepare_headers(dict(headers) if headers else {})
        return self._request_with_retry(
            method="POST",
            url=url,
            params=prepared_params,
            headers=prepared_headers,
            json_body=json_body,
        )

    def _log_request(
        self,
        url: str,
        params: dict | None = None,
        method: str = "GET",
        headers: dict | None = None,
    ) -> None:
        """Log an outgoing HTTP request at ``DEBUG`` level.

        API keys and other sensitive parameters and headers are replaced with
        ``"***"`` so that credentials are never written to logs.

        Parameters
        ----------
        url : str
            Base request URL (without query string).
        params : dict | None
            Query parameters to be sent with the request.
        method : str
            HTTP method (e.g. ``"GET"`` or ``"POST"``).
        headers : dict | None
            HTTP headers to be sent with the request.
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return
        if params:
            safe_params = {
                k: "***" if k.lower() in _SENSITIVE_PARAM_NAMES else v for k, v in params.items()
            }
            full_url = f"{url}?{urlencode(safe_params)}"
        else:
            full_url = url
        logger.debug("[%s] %s %s", self.name, method, full_url)
        if headers:
            safe_headers = {
                k: "***" if k.lower() in _SENSITIVE_HEADER_NAMES else v for k, v in headers.items()
            }
            logger.debug("[%s] headers: %s", self.name, safe_headers)

    def _log_response(self, response: requests.Response) -> None:
        """Log a summary of an HTTP response at ``DEBUG`` level.

        Logs the HTTP status code, content-type header, and body size so that
        verbose sessions can trace what each request returned without printing
        the full body.

        Parameters
        ----------
        response : requests.Response
            The completed HTTP response to summarise.
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return
        status = f"{response.status_code} {response.reason}"
        content_type = response.headers.get("Content-Type", "unknown").split(";")[0].strip()
        size = len(response.content)
        logger.debug(
            "[%s] <- %s | content-type: %s | %d bytes",
            self.name,
            status,
            content_type,
            size,
        )
