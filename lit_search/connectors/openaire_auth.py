"""OpenAIRE personal-token refresh support.

OpenAIRE personal access tokens last one hour.  A separately issued refresh
token lasts one month and can be exchanged for another access token at the
user-management endpoint.  This module keeps that exchange small, strict, and
independent of search pagination:

* refresh tokens are never logged or interpolated into exception messages;
* a failed refresh raises instead of silently turning the request anonymous;
* concurrent callers share one refresh operation; and
* the access token is refreshed shortly before its reported expiry.

The refresh token itself is deliberately kept in memory.  Persisting secrets
belongs to the caller's credential store, not to a literature-search cache or
result handle.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from math import isfinite
from typing import Any

import requests

from lit_search.exceptions import ConnectorError

OPENAIRE_REFRESH_URL = "https://services.openaire.eu/uoa-user-management/api/users/getAccessToken"
"""Documented endpoint for exchanging a personal refresh token."""

_DEFAULT_TIMEOUT_SECONDS = 30.0
_MAX_EXPIRY_SKEW_SECONDS = 60.0


class OpenAIREAuthenticationError(ConnectorError):
    """OpenAIRE could not produce a usable authenticated request.

    Messages intentionally omit response bodies, request URLs, and underlying
    exception text: any of those can contain the refresh token because the
    upstream protocol puts it in a query parameter.
    """


class OpenAIRETokenManager:
    """Resolve and refresh an OpenAIRE personal access token.

    Parameters
    ----------
    access_token : str | None
        An existing one-hour access token.  When a refresh token is also
        supplied, ``access_token_expires_in`` is required to trust this token;
        otherwise the manager refreshes it on first use rather than guessing
        how much of its lifetime remains.
    refresh_token : str | None
        The month-lived personal refresh token issued by OpenAIRE.
    access_token_expires_in : float | None
        Remaining lifetime of ``access_token`` in seconds, measured when this
        manager is constructed.
    timeout : float
        Timeout for the token exchange.
    clock : Callable[[], float]
        Monotonic clock, injectable for deterministic tests.

    Notes
    -----
    OpenAIRE's response contains another ``refresh_token`` field.  The manager
    adopts a non-empty returned value in memory so rotating-token deployments
    keep working for the lifetime of this object.
    """

    def __init__(
        self,
        *,
        access_token: str | None = None,
        refresh_token: str | None = None,
        access_token_expires_in: float | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._access_token = _clean_secret(access_token)
        self._refresh_token = _clean_secret(refresh_token)
        if self._access_token is None and self._refresh_token is None:
            raise ValueError("OpenAIRE authentication requires an access token or refresh token.")
        if not isfinite(timeout) or timeout <= 0:
            raise ValueError("OpenAIRE token refresh timeout must be positive.")

        self._timeout = float(timeout)
        self._clock = clock
        self._lock = threading.Lock()
        self._access_token_expires_at: float | None = None
        if access_token_expires_in is not None:
            lifetime = _parse_lifetime(access_token_expires_in)
            self._access_token_expires_at = self._expiry_deadline(lifetime)

    @property
    def refresh_token(self) -> str | None:
        """Return the current in-memory refresh token.

        This is intentionally not included in ``repr`` and is never logged.
        It exists so a higher-level credential store can explicitly persist a
        rotated token in a future integration.
        """
        return self._refresh_token

    @property
    def can_refresh(self) -> bool:
        """Return whether a failed bearer request can obtain a new token."""
        return self._refresh_token is not None

    def get_access_token(self, session: requests.Session | None = None) -> str:
        """Return a usable access token, refreshing it when necessary.

        A static access token without a refresh token is returned as supplied;
        its upstream rejection remains a loud HTTP failure.  When refresh
        credentials are configured, any refresh failure raises
        :class:`OpenAIREAuthenticationError`; anonymous fallback is never
        attempted.
        """
        with self._lock:
            if self._access_token_is_usable():
                assert self._access_token is not None
                return self._access_token

            if self._refresh_token is None:
                if self._access_token is not None:
                    return self._access_token
                raise OpenAIREAuthenticationError(
                    "OpenAIRE authentication has no usable token material."
                )

            owned_session = session is None
            auth_session = session or requests.Session()
            try:
                return self._refresh(auth_session)
            finally:
                if owned_session:
                    auth_session.close()

    def force_refresh(self, session: requests.Session | None = None) -> str:
        """Exchange the refresh token even when the cached token looks fresh.

        This is used exactly once after an authenticated Graph request rejects
        a bearer token.  It never downgrades to anonymous access.
        """
        with self._lock:
            if self._refresh_token is None:
                raise OpenAIREAuthenticationError(
                    "OpenAIRE authentication cannot refresh the rejected access token."
                )
            owned_session = session is None
            auth_session = session or requests.Session()
            try:
                return self._refresh(auth_session)
            finally:
                if owned_session:
                    auth_session.close()

    def _access_token_is_usable(self) -> bool:
        """Return whether the cached token can be used without a refresh."""
        if self._access_token is None:
            return False
        if self._refresh_token is None:
            # A caller that supplies only an access token has chosen the static
            # one-hour-token path.  Upstream rejection is still loud.
            return True
        if self._access_token_expires_at is None:
            # With a refresh token available, do not guess the age of an
            # independently acquired access token.
            return False
        return self._clock() < self._access_token_expires_at

    def _refresh(self, session: requests.Session) -> str:
        """Exchange the refresh token, validating the complete response."""
        try:
            response = session.get(
                OPENAIRE_REFRESH_URL,
                params={"refreshToken": self._refresh_token},
                timeout=self._timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            suffix = f" with HTTP {status}" if isinstance(status, int) else ""
            raise OpenAIREAuthenticationError(
                f"OpenAIRE access-token refresh failed{suffix} ({type(exc).__name__})."
            ) from None

        try:
            payload: Any = response.json()
        except ValueError:
            raise OpenAIREAuthenticationError(
                "OpenAIRE access-token refresh returned a non-JSON response."
            ) from None

        if not isinstance(payload, dict):
            raise OpenAIREAuthenticationError(
                "OpenAIRE access-token refresh returned a non-object JSON response."
            )

        access_token = _clean_secret(payload.get("access_token"))
        token_type = str(payload.get("token_type") or "Bearer").strip()
        if access_token is None or token_type.casefold() != "bearer":
            raise OpenAIREAuthenticationError(
                "OpenAIRE access-token refresh returned no usable bearer token."
            )

        try:
            lifetime = _parse_lifetime(payload.get("expires_in"))
        except (TypeError, ValueError):
            raise OpenAIREAuthenticationError(
                "OpenAIRE access-token refresh returned an invalid expiry."
            ) from None

        rotated_refresh_token = _clean_secret(payload.get("refresh_token"))
        if rotated_refresh_token is not None:
            self._refresh_token = rotated_refresh_token
        self._access_token = access_token
        self._access_token_expires_at = self._expiry_deadline(lifetime)
        return access_token

    def _expiry_deadline(self, lifetime: float) -> float:
        """Return a proactive-refresh deadline for a reported lifetime."""
        skew = min(_MAX_EXPIRY_SKEW_SECONDS, lifetime * 0.1)
        return self._clock() + max(0.0, lifetime - skew)


def _clean_secret(value: object) -> str | None:
    """Normalize a credential without ever rendering non-strings."""
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _parse_lifetime(value: object) -> float:
    """Parse a strictly positive token lifetime in seconds."""
    if isinstance(value, bool):
        raise TypeError("boolean is not a token lifetime")
    lifetime = float(value)  # type: ignore[arg-type]
    if not isfinite(lifetime) or lifetime <= 0:
        raise ValueError("token lifetime must be positive")
    return lifetime


__all__ = [
    "OPENAIRE_REFRESH_URL",
    "OpenAIREAuthenticationError",
    "OpenAIRETokenManager",
]
