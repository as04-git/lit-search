"""Run a bounded cloud search and export only a credential-safe review artifact."""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, quote_plus, urlsplit, urlunsplit

from lit_search.core.search_result import SearchResult
from lit_search.engine import Engine
from lit_search.query.validator import QueryValidator

SOURCES = ("openalex", "semantic_scholar")
MAX_PAPERS = 25
CREDENTIALS = (
    "OPENALEX_API_KEY",
    "SEMANTIC_SCHOLAR_API_KEY",
)
_URL = re.compile(r"https?://[^\s<>\"']+")


@dataclass(frozen=True)
class SearchInputs:
    """Validated, deliberately small query and publication window."""

    query: str
    since: dt.date
    until: dt.date


def read_inputs(env: Mapping[str, str], today: dt.date | None = None) -> SearchInputs:
    """Validate a query and an inclusive publication window of at most 31 days."""
    today = today or dt.datetime.now(dt.UTC).date()
    query = env.get("LIT_SEARCH_QUERY", "").strip()
    if not 1 <= len(query) <= 512 or any(ord(char) < 32 for char in query):
        raise ValueError("invalid query length or control character")
    QueryValidator().validate(query)
    dates = []
    for name, fallback in (
        ("LIT_SEARCH_UNTIL", today),
        ("LIT_SEARCH_SINCE", None),
    ):
        raw = env.get(name, "").strip()
        if raw and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
            raise ValueError("dates must use YYYY-MM-DD")
        value = dt.date.fromisoformat(raw) if raw else fallback
        dates.append(value)
    until = dates[0]
    assert until is not None
    since = dates[1] or until - dt.timedelta(days=13)
    if until > today or not 0 <= (until - since).days <= 30:
        raise ValueError("publication window must be past/present and at most 31 days")
    return SearchInputs(query=query, since=since, until=until)


def _strip_url_query(match: re.Match[str]) -> str:
    """Remove URL user information, query parameters and fragments from exports."""
    try:
        parts = urlsplit(match.group())
        return urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, "", ""))
    except ValueError:
        return "[invalid-url]"


def safe_text(value: str | None, secrets: tuple[str, ...]) -> str | None:
    """Redact active credentials and strip credential-bearing URL components."""
    if value is None:
        return None
    variants = set()
    for secret in secrets:
        if secret:
            variants.update((secret, quote(secret, safe=""), quote_plus(secret)))
            variants.add(quote(quote(secret, safe=""), safe=""))
    for variant in sorted(variants, key=len, reverse=True):
        value = value.replace(variant, "[REDACTED]")
    return _URL.sub(_strip_url_query, value)


def export_result(
    result: SearchResult, inputs: SearchInputs, secrets: tuple[str, ...]
) -> dict[str, Any]:
    """Allowlist bibliographic data and typed coverage, never raw provider errors.

    This is a review artifact, not a lossless SearchResult or handle export.
    Provider detail/error strings can contain authenticated request URLs and are
    intentionally excluded along with arbitrary metadata and request history.
    """
    papers = []
    for paper in result.papers:
        papers.append(
            {
                "title": safe_text(paper.title, secrets),
                "abstract": safe_text(paper.abstract, secrets),
                "authors": [safe_text(author.name, secrets) for author in paper.authors],
                "publication_date": (
                    paper.publication_date.isoformat() if paper.publication_date else None
                ),
                "doi": safe_text(paper.doi, secrets),
                "url": safe_text(paper.url, secrets),
                "found_in": sorted(set(paper.found_in) & set(SOURCES)),
            }
        )
    return {
        "schema": "lit-search/cloud-radar/v1",
        "query": safe_text(inputs.query, secrets),
        "since": inputs.since.isoformat(),
        "until": inputs.until.isoformat(),
        "sources": list(SOURCES),
        "max_papers_per_source": MAX_PAPERS,
        "enrichment": False,
        "execution_status": result.coverage.execution_status.value,
        "coverage_status": result.coverage.status.value,
        "coverage": [
            {
                "source": entry.source,
                "status": entry.status.value,
                "execution_status": (
                    entry.execution_status.value if entry.execution_status else "unknown"
                ),
                "retrieved": entry.retrieved,
                "cap": entry.cap,
                "reported_total": entry.reported_total,
            }
            for entry in result.coverage.entries
            if entry.source in SOURCES
        ],
        "paper_count": len(papers),
        "papers": papers,
        "omitted": ["provider_error_detail", "request_metadata", "raw_handles"],
    }


def main(env: Mapping[str, str] | None = None) -> int:
    """Write one safe artifact; return 0, 1, 2 or 3 without raw exception text."""
    env = os.environ if env is None else env
    output = Path(env.get("LIT_SEARCH_OUTPUT_DIR", "/tmp/lit-search-radar")) / "result.json"
    payload: dict[str, Any] = {"schema": "lit-search/cloud-radar/v1", "status": "failed"}
    code = 1
    try:
        inputs = read_inputs(env)
    except Exception:
        payload["reason"] = "invalid_query_or_date_window"
        code = 2
    else:
        secrets = tuple(env.get(name, "") for name in CREDENTIALS)
        if not all(value.strip() for value in secrets):
            payload["reason"] = "missing_openalex_or_semantic_scholar_credential"
        else:
            try:
                engine = Engine(openalex_api_key=secrets[0], semantic_scholar_api_key=secrets[1])
                result = engine.search(
                    inputs.query,
                    databases=list(SOURCES),
                    since=inputs.since,
                    until=inputs.until,
                    max_papers_per_database=MAX_PAPERS,
                    enrichment_databases=[],
                    num_workers=1,
                    verbose=False,
                    show_progress=False,
                )
                payload = export_result(result, inputs, secrets)
                code = 0 if result.coverage.execution_is_complete else 3
            except Exception:
                # Avoid str(error), repr(error), traceback and raw result dumps.
                payload["reason"] = "search_or_safe_export_failed"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
