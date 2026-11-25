"""Optional question-relevance ordering from a local cross-encoder service.

The service (``tools/rerank_server.py``) runs in a separate environment, so this
client has no model dependencies.  Every failure (unset, down, slow, malformed)
returns ``None`` and the caller keeps its own order: relevance ordering is an
aid to reading, never a condition of a move succeeding.
"""

from __future__ import annotations

import json
import logging
import urllib.request

from lit_search import credentials

logger = logging.getLogger(__name__)
TIMEOUT_SECONDS = 10.0


def relevance_scores(
    query: str, documents: list[str], *, timeout: float = TIMEOUT_SECONDS
) -> tuple[list[float], str] | None:
    """Score *documents* against *query*, or ``None`` when the service cannot.

    Parameters
    ----------
    query : str
        The hunt's question.
    documents : list[str]
        One text per candidate (title and abstract).
    timeout : float
        Seconds to wait for the service.

    Returns
    -------
    tuple[list[float], str] | None
        Scores aligned with *documents* and the model name, or ``None``.
    """
    url = credentials.get("LIT_SEARCH_RERANK_URL")
    if not url or not documents:
        return None
    body = json.dumps({"query": query, "documents": documents}).encode("utf-8")
    request = urllib.request.Request(
        url.rstrip("/") + "/rerank",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
            payload = json.loads(response.read())
        scores = [float(s) for s in payload["scores"]]
    except Exception as exc:
        logger.info("relevance service unavailable (%s); keeping the tool's order", exc)
        return None
    if len(scores) != len(documents):
        return None
    return scores, str(payload.get("model") or "reranker")
