"""Append-only log of every HTTP request a connector sends.

One JSON line per attempt: when it was sent, which quota bucket it drew on,
the endpoint path (never the query string, which can carry keys), the status
or transport failure, and the latency.  Every process appends to the same
file, so the log shows the request rate a service actually saw from this
machine and which of those requests it refused.  ``lit-search eval
requests`` reads it.

Logging must never fail a request: every error here is swallowed.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.parse import urlsplit

ENV_DISABLE = "LIT_SEARCH_REQUEST_LOG"
MAX_BYTES = 20 * 1024 * 1024


def log_path() -> Path:
    """Where the request log lives.

    Beside the rate limiter's directory, which every process shares whatever
    its store root (replays use a separate store but the same quotas).
    """
    from lit_search.utils.rate_limit import rate_limit_dir

    return rate_limit_dir().parent / "requests.jsonl"


def record(
    *,
    scope: str,
    method: str,
    url: str,
    sent: float,
    status: int | str,
    latency: float,
) -> None:
    """Append one request outcome; never raises."""
    if os.environ.get(ENV_DISABLE, "1") == "0":
        return
    try:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if path.stat().st_size > MAX_BYTES:
                path.replace(path.with_suffix(".jsonl.1"))
        except FileNotFoundError:
            pass
        line = json.dumps(
            {
                "t": round(sent, 3),
                "scope": scope,
                "method": method,
                "path": urlsplit(url).netloc + urlsplit(url).path,
                "status": status,
                "latency": round(latency, 3),
                "pid": os.getpid(),
            },
            separators=(",", ":"),
        )
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, (line + "\n").encode("utf-8"))
        finally:
            os.close(fd)
    except Exception:
        return


def now() -> float:
    """Wall-clock time used for ``sent``."""
    return time.time()
