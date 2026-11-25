"""Every credential and machine setting lit-search reads, and where it reads them.

One table, :data:`SETTINGS`, names each value once.  API keys use the name the
service's own documentation and most key files use (``SEMANTIC_SCHOLAR_API_KEY``);
lit-search's own settings use the ``LIT_SEARCH_`` prefix.

A value comes from the environment if set there, otherwise from the credentials
file (``$XDG_CONFIG_HOME/lit-search/credentials.env``, or ``LIT_SEARCH_CREDENTIALS``),
a plain ``NAME=value`` file that ``lit-search config`` writes with mode 600.  The
file is parsed, never executed.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

ENV_FILE = "LIT_SEARCH_CREDENTIALS"


@dataclass(frozen=True)
class Setting:
    """One value lit-search can use.

    Parameters
    ----------
    name : str
        Environment / file name.
    source : str
        What it enables (a source name, or ``"general"``).
    secret : bool
        Whether the value must never be printed.
    help : str
        Where to get it and what happens without it.
    legacy : str | None
        The pre-2026-10-12 ``FINDPAPERS_*`` name, accepted only by
        ``config import`` and reported by ``doctor``.
    """

    name: str
    source: str
    secret: bool
    help: str
    legacy: str | None = None


SETTINGS: tuple[Setting, ...] = (
    Setting(
        "SEMANTIC_SCHOLAR_API_KEY",
        "semantic_scholar",
        True,
        "free key: semanticscholar.org/product/api#api-key-form; without it requests share a "
        "public pool and are refused often",
        "FINDPAPERS_SEMANTIC_SCHOLAR_API_TOKEN",
    ),
    Setting(
        "OPENALEX_API_KEY",
        "openalex",
        True,
        "free key: openalex.org/settings/api; without it the daily budget is about 10 requests",
        "FINDPAPERS_OPENALEX_API_TOKEN",
    ),
    Setting(
        "OPENAIRE_REFRESH_TOKEN",
        "openaire",
        True,
        "personal refresh token (lasts a month): develop.openaire.eu/personal-token; "
        "without it OpenAIRE runs anonymously at a low quota",
        "FINDPAPERS_OPENAIRE_REFRESH_TOKEN",
    ),
    Setting(
        "OPENAIRE_ACCESS_TOKEN",
        "openaire",
        True,
        "optional one-hour access token; the refresh token is enough",
        "FINDPAPERS_OPENAIRE_ACCESS_TOKEN",
    ),
    Setting(
        "CORE_API_KEY",
        "core",
        True,
        "free key: core.ac.uk/services/api; CORE is skipped without it",
        "FINDPAPERS_CORE_API_TOKEN",
    ),
    Setting(
        "NCBI_API_KEY",
        "pubmed",
        True,
        "free key from an NCBI account (settings > API key); raises PubMed's limit to 10/s",
        "FINDPAPERS_PUBMED_API_TOKEN",
    ),
    Setting(
        "IEEE_API_KEY",
        "ieee",
        True,
        "developer.ieee.org; IEEE Xplore is skipped without it",
        "FINDPAPERS_IEEE_API_TOKEN",
    ),
    Setting(
        "SCOPUS_API_KEY",
        "scopus",
        True,
        "dev.elsevier.com (institutional); Scopus is skipped without it",
        "FINDPAPERS_SCOPUS_API_TOKEN",
    ),
    Setting(
        "WOS_API_KEY",
        "wos",
        True,
        "developer.clarivate.com (institutional); Web of Science is skipped without it",
        "FINDPAPERS_WOS_API_TOKEN",
    ),
    Setting(
        "SSRN_API_KEY", "ssrn", True, "SSRN is skipped without it", "FINDPAPERS_SSRN_API_TOKEN"
    ),
    Setting(
        "ZENODO_ACCESS_TOKEN",
        "zenodo",
        True,
        "optional: zenodo.org/account/settings/applications",
        "FINDPAPERS_ZENODO_ACCESS_TOKEN",
    ),
    Setting(
        "OPENREVIEW_USERNAME",
        "openreview",
        False,
        "an OpenReview account; OpenReview is skipped without it",
        "FINDPAPERS_OPENREVIEW_USERNAME",
    ),
    Setting(
        "OPENREVIEW_PASSWORD",
        "openreview",
        True,
        "password for OPENREVIEW_USERNAME",
        "FINDPAPERS_OPENREVIEW_PASSWORD",
    ),
    Setting(
        "HF_TOKEN",
        "huggingface",
        True,
        "optional; Hugging Face Papers search works without it",
    ),
    Setting(
        "LIT_SEARCH_EMAIL",
        "general",
        False,
        "contact address some services ask for (Crossref polite pool, NCBI, ChemRxiv)",
        "FINDPAPERS_EMAIL",
    ),
    Setting(
        "LIT_SEARCH_NCBI_EMAIL",
        "pubmed",
        False,
        "contact for NCBI if different from LIT_SEARCH_EMAIL",
        "FINDPAPERS_NCBI_EMAIL",
    ),
    Setting(
        "LIT_SEARCH_NCBI_TOOL",
        "pubmed",
        False,
        "tool name reported to NCBI",
        "FINDPAPERS_NCBI_TOOL",
    ),
    Setting(
        "LIT_SEARCH_CHEMRXIV_EMAIL",
        "chemrxiv",
        False,
        "contact for ChemRxiv if different from LIT_SEARCH_EMAIL",
        "FINDPAPERS_CHEMRXIV_EMAIL",
    ),
    Setting(
        "LIT_SEARCH_RERANK_URL",
        "general",
        False,
        "local relevance service (tools/rerank_server.py), e.g. http://127.0.0.1:8797; hunt moves "
        "then list new candidates by relevance to the question; unset or down = own order",
    ),
    Setting(
        "LIT_SEARCH_PROXY",
        "general",
        False,
        "HTTP(S) proxy URL for all requests",
        "FINDPAPERS_PROXY",
    ),
    Setting(
        "LIT_SEARCH_SSL_VERIFY",
        "general",
        False,
        "set to 0/false/no to disable TLS verification (prefer installing the CA instead)",
        "FINDPAPERS_SSL_VERIFY",
    ),
    Setting(
        "LIT_SEARCH_OPENREVIEW_CACHE_DIR",
        "openreview",
        False,
        "where OpenReview downloads are cached",
        "FINDPAPERS_OPENREVIEW_CACHE_DIR",
    ),
    Setting(
        "LIT_SEARCH_ARXIV_SNAPSHOT",
        "arxiv",
        False,
        "path to a local arXiv metadata snapshot for bulk search",
        "FINDPAPERS_ARXIV_SNAPSHOT",
    ),
    Setting(
        "LIT_SEARCH_ZOTERO_BASE_URL",
        "zotero",
        False,
        "Zotero API base URL",
        "FINDPAPERS_ZOTERO_BASE_URL",
    ),
    Setting(
        "LIT_SEARCH_ZOTERO_LIBRARY",
        "zotero",
        False,
        "Zotero library id",
        "FINDPAPERS_ZOTERO_LIBRARY",
    ),
)
BY_NAME: dict[str, Setting] = {s.name: s for s in SETTINGS}
BY_LEGACY: dict[str, Setting] = {s.legacy: s for s in SETTINGS if s.legacy}

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def credentials_path() -> Path:
    """Path of the credentials file.

    Returns
    -------
    Path
        ``$LIT_SEARCH_CREDENTIALS``, else ``$XDG_CONFIG_HOME/lit-search/credentials.env``
        (``~/.config`` when ``XDG_CONFIG_HOME`` is unset).
    """
    explicit = os.environ.get(ENV_FILE)
    if explicit:
        return Path(explicit).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "lit-search" / "credentials.env"


def parse_env_file(path: Path) -> dict[str, str]:
    """Read ``NAME=value`` lines without executing anything.

    Blank lines, comments, ``export`` prefixes and matching surrounding quotes are
    handled; lines that are not plain assignments (``source``, ``set -a``) and values
    that need the shell (``$VAR``, backticks, outside single quotes) are ignored.

    Parameters
    ----------
    path : Path
        File to read.

    Returns
    -------
    dict[str, str]
        Assignments in file order (later ones win).
    """
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.lstrip().startswith("#"):
            continue
        match = _LINE.match(raw)
        if not match:
            continue
        name, value = match.groups()
        if len(value) >= 2 and value[0] == value[-1] == "'":
            value = value[1:-1]
        else:
            if len(value) >= 2 and value[0] == value[-1] == '"':
                value = value[1:-1]
            if "$" in value or "`" in value:
                continue
        values[name] = value
    return values


_file_cache: tuple[Path, float, dict[str, str]] | None = None


def _file_values() -> dict[str, str]:
    global _file_cache
    path = credentials_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    if _file_cache is None or _file_cache[0] != path or _file_cache[1] != mtime:
        _file_cache = (path, mtime, parse_env_file(path))
    return _file_cache[2]


def get(name: str) -> str | None:
    """Return a setting's value: the environment first, then the credentials file.

    Parameters
    ----------
    name : str
        A name from :data:`SETTINGS`.

    Returns
    -------
    str | None
        The value, or ``None`` when unset or blank.
    """
    if name not in BY_NAME:
        raise KeyError(f"{name} is not a lit-search setting")
    value = (os.environ.get(name) or "").strip() or (_file_values().get(name) or "").strip()
    return value or None


def origin(name: str) -> str | None:
    """Where a setting's value comes from.

    Parameters
    ----------
    name : str
        A name from :data:`SETTINGS`.

    Returns
    -------
    str | None
        ``"environment"``, ``"file"`` or ``None`` when unset.
    """
    if (os.environ.get(name) or "").strip():
        return "environment"
    if (_file_values().get(name) or "").strip():
        return "file"
    return None


def write(updates: dict[str, str | None]) -> Path:
    """Set or remove values in the credentials file, keeping its other lines.

    Parameters
    ----------
    updates : dict[str, str | None]
        New values; ``None`` removes the name.

    Returns
    -------
    Path
        The file written (mode 600).
    """
    unknown = [n for n in updates if n not in BY_NAME]
    if unknown:
        raise KeyError(f"not lit-search settings: {', '.join(unknown)}")
    for name, value in updates.items():
        if value is not None and ("\n" in value or "\r" in value):
            raise ValueError(f"{name}: value contains a line break")
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (
        path.read_text(encoding="utf-8").splitlines()
        if path.exists()
        else [
            "# lit-search credentials. Written by `lit-search config`; plain NAME=value lines.",
        ]
    )
    pending = dict(updates)
    kept: list[str] = []
    for line in lines:
        match = _LINE.match(line)
        if match and match.group(1) in pending:
            value = pending.pop(match.group(1))
            if value is not None:
                kept.append(f"{match.group(1)}={value}")
            continue
        kept.append(line)
    kept += [f"{name}={value}" for name, value in pending.items() if value is not None]
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".credentials-")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write("\n".join(kept) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return path


def legacy_in_environment() -> list[str]:
    """Old ``FINDPAPERS_*`` names still set in the environment (now ignored).

    Returns
    -------
    list[str]
        The names found.
    """
    return sorted(n for n in os.environ if n.startswith("FINDPAPERS_"))


__all__ = [
    "BY_LEGACY",
    "BY_NAME",
    "ENV_FILE",
    "SETTINGS",
    "Setting",
    "credentials_path",
    "get",
    "legacy_in_environment",
    "origin",
    "parse_env_file",
    "write",
]
