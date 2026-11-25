"""A minimal HTTP GET seam for the fetcher.

The fetcher talks to the network only through an :class:`HttpGet` callable,
so tests substitute a fake and production uses :class:`CurlTransport`
(curl_cffi with Chrome TLS impersonation, which many publisher sites require).

A network failure raises :class:`TransportError`; it is never turned into
``None``.  TLS certificates are always verified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from curl_cffi.requests import Session as _CurlSession
from curl_cffi.requests.errors import RequestsError as _CurlError

_CURL_MESSAGE = re.compile(r"curl: \((\d+)\) ([^\n]*?)(?:\.? See https?://.*)?$")


class TransportError(Exception):
    """A request produced no HTTP response (DNS, TLS, timeout, reset, ...)."""


@dataclass(frozen=True)
class HttpResponse:
    """The parts of an HTTP response the fetcher uses.

    Attributes
    ----------
    status : int
        HTTP status code.
    url : str
        Final URL after redirects.
    content_type : str
        Media type without parameters, lower-cased (``""`` when absent).
    content : bytes
        Body.
    """

    status: int
    url: str
    content_type: str
    content: bytes


class HttpGet(Protocol):
    """Callable performing one GET request."""

    def __call__(self, url: str, *, timeout: float) -> HttpResponse:
        """Fetch *url*.

        Parameters
        ----------
        url : str
            Absolute URL.
        timeout : float
            Seconds.

        Returns
        -------
        HttpResponse
            The response, whatever its status.

        Raises
        ------
        TransportError
            If no response was received.
        """
        ...


def short_curl_message(error: BaseException) -> str:
    """Condense a curl_cffi error into one short line.

    Parameters
    ----------
    error : BaseException
        Error raised by curl_cffi.

    Returns
    -------
    str
        E.g. ``"curl 6: Could not resolve host: sci-hub.se"``.
    """
    text = str(error).strip()
    match = _CURL_MESSAGE.search(text)
    if match:
        return f"curl {match.group(1)}: {match.group(2).rstrip('.')}"
    first_line = text.splitlines()[0] if text else type(error).__name__
    return first_line[:160]


class CurlTransport:
    """:class:`HttpGet` backed by one reusable curl_cffi session.

    Parameters
    ----------
    impersonate : str
        Browser fingerprint to present.  Defaults to ``"chrome"``.
    proxy : str | None
        Proxy URL applied to HTTP and HTTPS.
    """

    def __init__(self, impersonate: str = "chrome", proxy: str | None = None) -> None:
        self._session: _CurlSession = _CurlSession(impersonate=impersonate)
        self._proxies = {"http": proxy, "https": proxy} if proxy else None

    def __call__(self, url: str, *, timeout: float) -> HttpResponse:
        """Fetch *url*; see :class:`HttpGet`.

        Parameters
        ----------
        url : str
            Absolute URL.
        timeout : float
            Seconds.

        Returns
        -------
        HttpResponse
            The response, whatever its status.

        Raises
        ------
        TransportError
            If curl produced no response.
        """
        try:
            response = self._session.get(
                url,
                timeout=timeout,
                allow_redirects=True,
                verify=True,
                proxies=self._proxies,  # type: ignore[arg-type]
            )
        except _CurlError as error:
            raise TransportError(short_curl_message(error)) from error
        content_type = response.headers.get("content-type", "") or ""
        return HttpResponse(
            status=int(response.status_code),
            url=str(response.url),
            content_type=content_type.split(";")[0].strip().lower(),
            content=bytes(response.content),
        )

    def close(self) -> None:
        """Close the underlying session.

        Returns
        -------
        None
        """
        self._session.close()


__all__ = ["CurlTransport", "HttpGet", "HttpResponse", "TransportError", "short_curl_message"]
