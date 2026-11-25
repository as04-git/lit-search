"""``config`` and ``doctor``: set lit-search up on a machine and check that it works.

Designed so an agent can do the whole setup unattended::

    lit-search config import ~/.config/api-keys.env
    lit-search doctor

``doctor`` exits 0 when every source the instant and hunt modes rely on is
usable, and 3 otherwise, printing the command that fixes each problem.
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from lit_search import credentials
from lit_search.cli.output import EXIT_ERROR, EXIT_INCOMPLETE, EXIT_OK, emit_json, emit_text
from lit_search.store.store import default_store_root

#: Sources the instant (``find``) and ``hunt`` modes use; problems here fail ``doctor``.
CORE_SOURCES = ("semantic_scholar", "openalex", "arxiv", "openaire", "huggingface")
_TIMEOUT = 20.0


def add_setup_parsers(subparsers: argparse._SubParsersAction, parents: list) -> None:
    """Register ``config`` and ``doctor``.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparser registry.
    parents : list
        Shared parent parsers.

    Returns
    -------
    None
    """
    config = subparsers.add_parser(
        "config",
        parents=parents,
        help="show or set credentials (stored in ~/.config/lit-search/credentials.env)",
        description=(
            "Credentials come from the environment, else from the credentials file this "
            "command manages (plain NAME=value lines, mode 600). Values are never printed."
        ),
    )
    actions = config.add_subparsers(dest="config_action", metavar="ACTION", required=True)
    actions.add_parser("path", parents=parents, help="print the credentials file path")
    actions.add_parser(
        "list", parents=parents, help="every setting: set or not, from where, what it is for"
    )
    setter = actions.add_parser("set", parents=parents, help="set one value")
    setter.add_argument("name", choices=sorted(credentials.BY_NAME), metavar="NAME")
    setter.add_argument("value", nargs="?", help="the value (or use --stdin)")
    setter.add_argument(
        "--stdin", action="store_true", help="read the value from stdin (keeps it out of history)"
    )
    unset = actions.add_parser("unset", parents=parents, help="remove a value from the file")
    unset.add_argument("name", choices=sorted(credentials.BY_NAME), metavar="NAME")
    importer = actions.add_parser(
        "import",
        parents=parents,
        help="copy every recognised NAME=value from a file (also converts FINDPAPERS_* names)",
    )
    importer.add_argument("file", type=Path)
    importer.add_argument(
        "--overwrite", action="store_true", help="replace values already in the credentials file"
    )

    doctor = subparsers.add_parser(
        "doctor",
        parents=parents,
        help="check credentials and sources with one small request each; prints fixes",
    )
    doctor.add_argument(
        "--offline", action="store_true", help="only check what is configured; no requests"
    )


# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------


def cmd_config(args: argparse.Namespace) -> int:
    """Run a ``config`` action.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code.
    """
    action = args.config_action
    if action == "path":
        emit_text(str(credentials.credentials_path()))
        return EXIT_OK
    if action == "list":
        return _list(args)
    if action == "set":
        value = sys.stdin.read().strip() if args.stdin else (args.value or "").strip()
        if not value:
            emit_text("no value given (pass it as an argument or with --stdin)")
            return EXIT_ERROR
        path = credentials.write({args.name: value})
        emit_text(f"set {args.name} in {path}")
        return EXIT_OK
    if action == "unset":
        path = credentials.write({args.name: None})
        emit_text(f"removed {args.name} from {path}")
        return EXIT_OK
    return _import(args)


def _list(args: argparse.Namespace) -> int:
    rows = []
    for setting in credentials.SETTINGS:
        where = credentials.origin(setting.name)
        rows.append(
            {
                "name": setting.name,
                "source": setting.source,
                "set": where is not None,
                "from": where,
                "help": setting.help,
            }
        )
    if args.json:
        emit_json({"file": str(credentials.credentials_path()), "settings": rows})
        return EXIT_OK
    lines = [f"credentials file: {credentials.credentials_path()}"]
    for row in rows:
        state = f"set ({row['from']})" if row["set"] else "unset"
        lines.append(f"  {row['name']:<32} {row['source']:<17} {state}")
        if not row["set"]:
            lines.append(f"      {row['help']}")
    emit_text("\n".join(lines))
    return EXIT_OK


def _import(args: argparse.Namespace) -> int:
    if not args.file.exists():
        emit_text(f"no such file: {args.file}")
        return EXIT_ERROR
    found = credentials.parse_env_file(args.file)
    existing = (
        credentials.parse_env_file(credentials.credentials_path())
        if (credentials.credentials_path().exists())
        else {}
    )
    updates: dict[str, str | None] = {}
    converted: list[str] = []
    for name, value in found.items():
        setting = credentials.BY_NAME.get(name) or credentials.BY_LEGACY.get(name)
        if setting is None or not value.strip():
            continue
        if setting.name in existing and not args.overwrite:
            continue
        if setting.name in updates and name.startswith("FINDPAPERS_"):
            continue
        updates[setting.name] = value.strip()
        if name != setting.name:
            converted.append(f"{name} -> {setting.name}")
    skipped = sorted(
        n for n in found if n not in credentials.BY_NAME and n not in credentials.BY_LEGACY
    )
    if updates:
        path = credentials.write(updates)
        emit_text(f"imported into {path}: {', '.join(sorted(updates))}")
    else:
        emit_text(
            "nothing new to import" + ("" if args.overwrite else " (use --overwrite to replace)")
        )
    if converted:
        emit_text("converted old names: " + "; ".join(converted))
    if skipped:
        emit_text(f"not lit-search settings, ignored: {', '.join(skipped)}")
    emit_text("next: lit-search doctor")
    return EXIT_OK


# ----------------------------------------------------------------------
# doctor
# ----------------------------------------------------------------------


@dataclass
class Check:
    """One line of the doctor's report."""

    name: str
    status: str  # ok | warn | missing | fail | info
    detail: str
    fix: str = ""


def _status_of(response: requests.Response, accepted: str) -> tuple[str, str]:
    code = response.status_code
    if 200 <= code < 300:
        return "ok", accepted
    if code in (401, 403):
        return "fail", f"credential rejected (HTTP {code})"
    if code == 429:
        return "warn", "rate-limited right now (HTTP 429); the key could not be verified"
    return "warn", f"unexpected HTTP {code}"


def _paced_get(service_connector: object, url: str, **kwargs: Any) -> requests.Response:
    rate_limit = getattr(service_connector, "_rate_limit", None)
    if callable(rate_limit):
        rate_limit()
    return requests.get(url, timeout=_TIMEOUT, **kwargs)


def _check_semantic_scholar(key: str | None) -> tuple[str, str]:
    from lit_search.connectors.semantic_scholar import SemanticScholarConnector

    headers = {"x-api-key": key} if key else {}
    response = _paced_get(
        SemanticScholarConnector(api_key=key),
        "https://api.semanticscholar.org/graph/v1/paper/arXiv:1706.03762",
        params={"fields": "title"},
        headers=headers,
    )
    return _status_of(response, "key accepted" if key else "reachable without a key")


def _check_openalex(key: str | None) -> tuple[str, str]:
    from lit_search.connectors.openalex import OpenAlexConnector

    params: dict[str, str | int] = {"per_page": 1, "search": "test"}
    if key:
        params["api_key"] = key
    response = _paced_get(
        OpenAlexConnector(api_key=key), "https://api.openalex.org/works", params=params
    )
    return _status_of(response, "key accepted" if key else "reachable without a key")


def _check_openaire(access: str | None, refresh: str | None) -> tuple[str, str]:
    from lit_search.connectors.openaire_auth import OpenAIRETokenManager

    headers = {}
    if access or refresh:
        try:
            token = OpenAIRETokenManager(
                access_token=access, refresh_token=refresh
            ).get_access_token()
        except Exception as exc:
            return (
                "fail",
                f"token refresh failed ({type(exc).__name__}); the refresh token has "
                "probably expired (they last a month)",
            )
        headers["Authorization"] = f"Bearer {token}"
    response = requests.get(
        "https://api.openaire.eu/graph/v3/research-products",
        params={"search": "test", "pageSize": "1"},
        headers=headers,
        timeout=_TIMEOUT,
    )
    return _status_of(response, "token accepted" if headers else "reachable anonymously")


def _check_arxiv() -> tuple[str, str]:
    from lit_search.connectors.arxiv import ArxivConnector

    response = _paced_get(
        ArxivConnector(),
        "https://export.arxiv.org/api/query",
        params={"search_query": "all:test", "max_results": 1},
    )
    return _status_of(response, "reachable (no key needed)")


def _check_huggingface(token: str | None) -> tuple[str, str]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = requests.get(
        "https://huggingface.co/api/papers/search",
        params={"q": "test"},
        headers=headers,
        timeout=_TIMEOUT,
    )
    return _status_of(response, "reachable" + (" with token" if token else " (no key needed)"))


def _check_core(key: str) -> tuple[str, str]:
    response = requests.get(
        "https://api.core.ac.uk/v3/search/works",
        params={"q": "test", "limit": "1"},
        headers={"Authorization": f"Bearer {key}"},
        timeout=_TIMEOUT,
    )
    return _status_of(response, "key accepted")


def _check_pubmed(key: str) -> tuple[str, str]:
    response = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"db": "pubmed", "term": "test", "retmax": "1", "retmode": "json", "api_key": key},
        timeout=_TIMEOUT,
    )
    status, detail = _status_of(response, "key accepted")
    if status == "ok" and "error" in (response.json() or {}):
        return "fail", f"key rejected ({response.json()['error'][:80]})"
    return status, detail


def _live(name: str, run: Callable[[], tuple[str, str]], fix: str) -> Check:
    try:
        status, detail = run()
    except requests.RequestException as exc:
        return Check(name, "fail", f"unreachable ({type(exc).__name__})", "check the network")
    return Check(name, status, detail, fix if status in ("fail", "warn") else "")


def _set_fix(name: str) -> str:
    return f"lit-search config set {name} --stdin  ({credentials.BY_NAME[name].help})"


def run_doctor(offline: bool) -> list[Check]:
    """Collect every check.

    Parameters
    ----------
    offline : bool
        Skip network requests.

    Returns
    -------
    list[Check]
        Report lines, core sources first.
    """
    get = credentials.get
    checks: list[Check] = []
    path = credentials.credentials_path()
    if path.exists():
        mode = stat.S_IMODE(path.stat().st_mode)
        checks.append(
            Check(
                "credentials file",
                "ok" if not mode & 0o077 else "warn",
                str(path),
                "" if not mode & 0o077 else f"chmod 600 {path}",
            )
        )
    else:
        checks.append(
            Check(
                "credentials file",
                "info",
                f"{path} (not created; environment only)",
                "lit-search config import <file with your keys>",
            )
        )
    legacy = credentials.legacy_in_environment()
    if legacy:
        checks.append(
            Check(
                "old variable names",
                "warn",
                f"ignored since 2026-10-12: {', '.join(legacy)}",
                "lit-search config import <the file that sets them>, then stop exporting them",
            )
        )

    s2 = get("SEMANTIC_SCHOLAR_API_KEY")
    oa = get("OPENALEX_API_KEY")
    access, refresh = get("OPENAIRE_ACCESS_TOKEN"), get("OPENAIRE_REFRESH_TOKEN")
    hf = get("HF_TOKEN")
    if offline:
        for name, value in (
            ("semantic_scholar", s2),
            ("openalex", oa),
            ("openaire", refresh or access),
        ):
            wanted = {
                "semantic_scholar": "SEMANTIC_SCHOLAR_API_KEY",
                "openalex": "OPENALEX_API_KEY",
                "openaire": "OPENAIRE_REFRESH_TOKEN",
            }[name]
            checks.append(
                Check(
                    name,
                    "ok" if value else "missing",
                    "configured (not verified)" if value else f"{wanted} not set",
                    "" if value else _set_fix(wanted),
                )
            )
    else:
        checks.append(
            _live(
                "semantic_scholar",
                lambda: _check_semantic_scholar(s2),
                _set_fix("SEMANTIC_SCHOLAR_API_KEY"),
            )
        )
        checks.append(_live("openalex", lambda: _check_openalex(oa), _set_fix("OPENALEX_API_KEY")))
        checks.append(
            _live("arxiv", _check_arxiv, "arXiv may be refusing this address; retry later")
        )
        checks.append(
            _live(
                "openaire",
                lambda: _check_openaire(access, refresh),
                _set_fix("OPENAIRE_REFRESH_TOKEN"),
            )
        )
        checks.append(_live("huggingface", lambda: _check_huggingface(hf), "retry later"))
        for check, key_name in (
            (checks[-5], "SEMANTIC_SCHOLAR_API_KEY"),
            (checks[-4], "OPENALEX_API_KEY"),
            (checks[-2], "OPENAIRE_REFRESH_TOKEN"),
        ):
            if check.status == "ok" and not get(key_name):
                check.status, check.fix = "warn", _set_fix(key_name)
                check.detail += f"; {key_name} not set, so limits are much lower"

    core_key, ncbi = get("CORE_API_KEY"), get("NCBI_API_KEY")
    if core_key and not offline:
        checks.append(_live("core", lambda: _check_core(core_key), _set_fix("CORE_API_KEY")))
    if ncbi and not offline:
        checks.append(_live("pubmed", lambda: _check_pubmed(ncbi), _set_fix("NCBI_API_KEY")))
    for setting in credentials.SETTINGS:
        if setting.source in CORE_SOURCES or setting.source in ("general",):
            continue
        if setting.source in ("core", "pubmed") and not offline and get(setting.name):
            continue
        if setting.name.startswith("LIT_SEARCH_") or setting.name in ("OPENREVIEW_PASSWORD",):
            continue
        state = "configured (not verified)" if get(setting.name) else f"{setting.name} not set"
        checks.append(
            Check(
                setting.source, "info", state, "" if get(setting.name) else _set_fix(setting.name)
            )
        )
    if not get("LIT_SEARCH_EMAIL"):
        checks.append(
            Check("contact email", "info", "LIT_SEARCH_EMAIL not set", _set_fix("LIT_SEARCH_EMAIL"))
        )

    from lit_search.utils.relevance import relevance_backend, relevance_scores

    backend = relevance_backend()
    if backend in ("voyage", "local") and not offline:
        scored = relevance_scores("state tracking", ["parity in linear RNNs", "coffee"], timeout=8)
        checks.append(
            Check(
                "relevance order",
                "ok" if scored else "warn",
                f"{scored[1]} answered"
                if scored
                else f"{backend} did not answer; moves keep the hunt's own order",
                "" if scored else "check VOYAGE_API_KEY / LIT_SEARCH_RERANK_URL",
            )
        )
    else:
        checks.append(
            Check(
                "relevance order",
                "info" if backend == "off" else "warn",
                f"{backend}; new candidates are listed in the hunt's own order",
                "" if backend == "off" else _set_fix("VOYAGE_API_KEY"),
            )
        )

    checks.append(
        Check(
            "pdftotext",
            "ok" if shutil.which("pdftotext") else "warn",
            "found"
            if shutil.which("pdftotext")
            else "missing: `hunt read` cannot extract PDF text",
            "" if shutil.which("pdftotext") else "install poppler-utils (apt) or poppler (brew)",
        )
    )
    store = default_store_root()
    try:
        store.mkdir(parents=True, exist_ok=True)
        writable = os.access(store, os.W_OK)
    except OSError:
        writable = False
    checks.append(
        Check(
            "store",
            "ok" if writable else "fail",
            str(store),
            "" if writable else "set LIT_SEARCH_HOME to a writable directory",
        )
    )
    return checks


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check this machine's setup.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        0 when the core sources are usable, 3 otherwise.
    """
    checks = run_doctor(args.offline)
    bad = [
        c for c in checks if c.name in (*CORE_SOURCES, "store") and c.status in ("fail", "missing")
    ]
    if args.json:
        emit_json({"ok": not bad, "checks": [c.__dict__ for c in checks]})
    else:
        mark = {"ok": "ok  ", "warn": "WARN", "missing": "MISS", "fail": "FAIL", "info": "  - "}
        lines = []
        for c in checks:
            lines.append(f"{mark[c.status]} {c.name:<18} {c.detail}")
            if c.fix:
                lines.append(f"     fix: {c.fix}")
        lines.append("ready" if not bad else f"not ready: {', '.join(c.name for c in bad)}")
        emit_text("\n".join(lines))
    return EXIT_OK if not bad else EXIT_INCOMPLETE


__all__ = ["CORE_SOURCES", "Check", "add_setup_parsers", "cmd_config", "cmd_doctor", "run_doctor"]
