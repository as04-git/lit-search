"""Question-relevance ordering for a hunt move's new candidates.

Two backends, tried in order: Voyage AI's hosted reranker (when ``VOYAGE_API_KEY``
is set) and a local cross-encoder service (``tools/rerank_server.py`` at
``LIT_SEARCH_RERANK_URL``).  ``LIT_SEARCH_RELEVANCE`` picks one (``voyage``,
``local``) or turns ordering ``off``; the default, ``auto``, uses whichever is
configured, Voyage first, and falls through to the next on any failure.

Every failure (unset, down, slow, refused, malformed) ends in ``None`` and the
caller keeps its own order: relevance ordering is an aid to reading, never a
condition of a move succeeding.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from lit_search import credentials

logger = logging.getLogger(__name__)
TIMEOUT_SECONDS = 10.0
VOYAGE_URL = "https://api.voyageai.com/v1/rerank"
DEFAULT_VOYAGE_MODEL = "rerank-3"
#: Characters of each document sent to Voyage.  Title plus abstract fits; the cap
#: keeps a 60-candidate move near 20k tokens, which matters on a free-tier quota.
DOCUMENT_CHARS = 1500


def _post(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
        return json.loads(response.read())


def _voyage(query: str, documents: list[str], timeout: float) -> tuple[list[float], str] | None:
    key = credentials.get("VOYAGE_API_KEY")
    if not key:
        return None
    model = credentials.get("LIT_SEARCH_VOYAGE_MODEL") or DEFAULT_VOYAGE_MODEL
    body = {
        "query": query,
        "documents": [d[:DOCUMENT_CHARS] or "(no text)" for d in documents],
        "model": model,
        "truncation": True,
    }
    try:
        payload = _post(VOYAGE_URL, body, {"Authorization": f"Bearer {key}"}, timeout)
        scores = [0.0] * len(documents)
        seen = 0
        for item in payload["data"]:
            scores[int(item["index"])] = float(item["relevance_score"])
            seen += 1
    except urllib.error.HTTPError as exc:
        logger.warning("Voyage reranker refused (HTTP %s); falling back", exc.code)
        return None
    except Exception as exc:
        logger.info("Voyage reranker unavailable (%s); falling back", exc)
        return None
    if seen != len(documents):
        return None
    return scores, f"voyage/{payload.get('model') or model}"


def _local(query: str, documents: list[str], timeout: float) -> tuple[list[float], str] | None:
    url = credentials.get("LIT_SEARCH_RERANK_URL")
    if not url:
        return None
    try:
        payload = _post(
            url.rstrip("/") + "/rerank", {"query": query, "documents": documents}, {}, timeout
        )
        scores = [float(s) for s in payload["scores"]]
    except Exception as exc:
        logger.info("relevance service unavailable (%s); keeping the tool's order", exc)
        return None
    if len(scores) != len(documents):
        return None
    return scores, str(payload.get("model") or "reranker")


def relevance_scores(
    query: str, documents: list[str], *, timeout: float = TIMEOUT_SECONDS
) -> tuple[list[float], str] | None:
    """Score *documents* against *query*, or ``None`` when no backend can.

    Parameters
    ----------
    query : str
        The hunt's question.
    documents : list[str]
        One text per candidate (title and abstract).
    timeout : float
        Seconds to wait for each backend.

    Returns
    -------
    tuple[list[float], str] | None
        Scores aligned with *documents* and the backend/model name, or ``None``.
    """
    if not documents or not query.strip():
        return None
    mode = (credentials.get("LIT_SEARCH_RELEVANCE") or "auto").strip().lower()
    backends = {"auto": (_voyage, _local), "voyage": (_voyage,), "local": (_local,)}.get(mode, ())
    for backend in backends:
        scored = backend(query, documents, timeout)
        if scored is not None:
            return scored
    return None


def relevance_backend() -> str:
    """Name the backend :func:`relevance_scores` would try first, for ``doctor``.

    Returns
    -------
    str
        ``"voyage"``, ``"local"``, ``"off"``, or ``"none configured"``.
    """
    mode = (credentials.get("LIT_SEARCH_RELEVANCE") or "auto").strip().lower()
    if mode == "off":
        return "off"
    if mode in ("auto", "voyage") and credentials.get("VOYAGE_API_KEY"):
        return "voyage"
    if mode in ("auto", "local") and credentials.get("LIT_SEARCH_RERANK_URL"):
        return "local"
    return "none configured"
