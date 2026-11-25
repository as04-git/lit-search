"""Sci-Hub as a PDF *fetcher*: mirror configuration and page parsing.

Sci-Hub is not a search source and is never registered in
:data:`lit_search.connectors.SEARCH_REGISTRY`.  It is consulted only by
:class:`lit_search.fetch.fetcher.PdfFetcher`, only when explicitly enabled
(``LIT_SEARCH_SCIHUB=1`` or ``lit-search fetch --scihub``), and only for a DOI.

The PDF-locating heuristics (``embed``/``iframe``/button ``onclick``/links,
with ``//`` and ``/`` relative forms) follow openags/paper-search-mcp
@234678ab ``academic_platforms/sci_hub.py`` (MIT).  Its error handling was not
ported: nothing here returns ``None`` for a failure, and TLS is verified.
Two differences found live on 2026-10-10: one mirror injects a hidden
advertising ``<iframe>`` before any content, so an iframe is accepted only
when it is the ``#pdf`` viewer or points at a ``.pdf``; and several mirrors
answer with a bot-challenge page, which :func:`is_challenge_page` recognises so
it is reported as *blocked* rather than as *not in Sci-Hub*.
"""

from __future__ import annotations

import os
import re
import urllib.parse
from collections.abc import Mapping
from typing import cast

import lxml.etree as _lxml_etree
import lxml.html as _lxml_html

#: Environment variable enabling the Sci-Hub step (``1``/``true``/``yes``).
SCIHUB_ENV_VAR = "LIT_SEARCH_SCIHUB"
#: Environment variable overriding the mirror list (comma-separated base URLs).
SCIHUB_MIRRORS_ENV_VAR = "LIT_SEARCH_SCIHUB_MIRRORS"

#: Default mirrors, in order.  Verified live 2026-10-10 against
#: ``10.1038/nature14539``: each of these served the article page with a PDF
#: ``<embed>``/``<iframe>``.  Probed and excluded that day: sci-hub.se (DNS
#: failure), sci-hub.st and sci-hub.box (HTTP 403), sci-hub.ru and sci-hub.wf
#: (bot-challenge page), sci-hub.red (HTTP 504), sci-hub.do (no PDF; ad
#: iframe only), sci-hub.shop (no PDF link).  Mirrors churn; override with
#: :data:`SCIHUB_MIRRORS_ENV_VAR`.
DEFAULT_MIRRORS: tuple[str, ...] = (
    "https://sci-hub.al",
    "https://sci-hub.mk",
    "https://sci-hub.ee",
    "https://sci-hub.in",
    "https://www.sci-hub.ren",
)

_TRUTHY = {"1", "true", "yes", "on"}
_ONCLICK_URL = re.compile(r"""location\.href\s*=\s*['"]([^'"]+)['"]""")
_CHALLENGE_MARKERS = (
    "are you a robot",
    "are you are robot",
    "checking your browser",
    "altcha-widget",
    "captcha",
    "just a moment...",
)


def scihub_enabled_from_env(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether the environment enables the Sci-Hub step.

    Parameters
    ----------
    environ : Mapping[str, str] | None
        Environment to read; defaults to ``os.environ``.

    Returns
    -------
    bool
        ``True`` when :data:`SCIHUB_ENV_VAR` is ``1``, ``true``, ``yes`` or ``on``.
    """
    env = os.environ if environ is None else environ
    return env.get(SCIHUB_ENV_VAR, "").strip().lower() in _TRUTHY


def mirrors_from_env(environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Return the mirror list: the environment override, else the default.

    Parameters
    ----------
    environ : Mapping[str, str] | None
        Environment to read; defaults to ``os.environ``.

    Returns
    -------
    tuple[str, ...]
        Base URLs without a trailing slash.

    Raises
    ------
    ValueError
        If the override is set but names no mirror, or a mirror is not an
        ``http(s)://`` URL.  A malformed override is not silently replaced by
        the default list.
    """
    env = os.environ if environ is None else environ
    raw = env.get(SCIHUB_MIRRORS_ENV_VAR)
    if raw is None:
        return DEFAULT_MIRRORS
    mirrors = tuple(part.strip().rstrip("/") for part in raw.split(",") if part.strip())
    if not mirrors:
        raise ValueError(f"{SCIHUB_MIRRORS_ENV_VAR} is set but names no mirror")
    for mirror in mirrors:
        if not mirror.startswith(("http://", "https://")):
            raise ValueError(f"{SCIHUB_MIRRORS_ENV_VAR}: {mirror!r} is not an http(s):// base URL")
    return mirrors


def article_url(mirror: str, doi: str) -> str:
    """Return the mirror page URL for *doi*.

    Parameters
    ----------
    mirror : str
        Mirror base URL.
    doi : str
        Bare DOI.

    Returns
    -------
    str
        ``<mirror>/<doi>``.
    """
    return f"{mirror.rstrip('/')}/{doi}"


def _absolute(raw: str, page_url: str) -> str:
    """Resolve a (possibly ``//``- or ``/``-relative) link and drop its fragment."""
    url = urllib.parse.urljoin(page_url, raw.strip())
    return urllib.parse.urldefrag(url).url


def _looks_like_pdf_link(raw: str) -> bool:
    """Return whether a link plausibly points at a PDF file."""
    path = urllib.parse.urlsplit(raw.strip()).path.lower()
    return path.endswith(".pdf") or "/pdf/" in path


def find_pdf_link(html: bytes, page_url: str) -> str | None:
    """Locate the PDF URL on a Sci-Hub article page.

    Checked in order: a PDF ``<embed>``, the ``#pdf`` viewer ``<iframe>`` (or
    any iframe whose ``src`` is a ``.pdf``), an ``<object>`` with PDF data, a
    ``<button onclick="location.href='...'">`` mentioning ``pdf``, then an
    ``<a href>`` to a ``.pdf`` or ``/pdf/`` path.

    Parameters
    ----------
    html : bytes
        Page body.
    page_url : str
        Final URL of the page, for resolving relative links.

    Returns
    -------
    str | None
        Absolute PDF URL, or ``None`` when the page carries no PDF link (the
        caller records that as a miss; it is an answer, not an error).

    Raises
    ------
    ValueError
        If the body cannot be parsed as HTML at all.
    """
    try:
        tree = _lxml_html.fromstring(html)
    except (_lxml_etree.ParserError, ValueError) as error:
        raise ValueError(f"unparseable HTML: {error}") from error

    def _elements(xpath: str) -> list[_lxml_etree._Element]:
        return cast(list[_lxml_etree._Element], tree.xpath(xpath))

    for embed in _elements("//embed[@src]"):
        src = embed.get("src") or ""
        if (embed.get("type") or "").lower() == "application/pdf" or embed.get("id") == "pdf":
            return _absolute(src, page_url)
    for iframe in _elements("//iframe[@src]"):
        src = iframe.get("src") or ""
        if iframe.get("id") == "pdf" or _looks_like_pdf_link(src):
            return _absolute(src, page_url)
    for obj in _elements("//object[@data]"):
        data = obj.get("data") or ""
        if (obj.get("type") or "").lower() == "application/pdf" or _looks_like_pdf_link(data):
            return _absolute(data, page_url)
    for button in _elements("//button[@onclick]"):
        onclick = button.get("onclick") or ""
        match = _ONCLICK_URL.search(onclick)
        if match and "pdf" in onclick.lower():
            return _absolute(match.group(1), page_url)
    for anchor in _elements("//a[@href]"):
        href = anchor.get("href") or ""
        if _looks_like_pdf_link(href):
            return _absolute(href, page_url)
    return None


def is_challenge_page(html: bytes) -> bool:
    """Return whether a page is a bot challenge rather than content.

    Parameters
    ----------
    html : bytes
        Page body.

    Returns
    -------
    bool
        ``True`` for captcha / "checking your browser" interstitials.
    """
    text = html[:20000].decode("utf-8", "replace").lower()
    return any(marker in text for marker in _CHALLENGE_MARKERS)


__all__ = [
    "DEFAULT_MIRRORS",
    "SCIHUB_ENV_VAR",
    "SCIHUB_MIRRORS_ENV_VAR",
    "article_url",
    "find_pdf_link",
    "is_challenge_page",
    "mirrors_from_env",
    "scihub_enabled_from_env",
]
