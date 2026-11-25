"""PDF fetching with every attempt recorded.

Sci-Hub lives here as a *fetcher* step (:mod:`lit_search.fetch.sci_hub`),
never as a search connector.
"""

from lit_search.fetch.fetcher import STEPS, PdfFetcher, is_pdf
from lit_search.fetch.http import CurlTransport, HttpGet, HttpResponse, TransportError
from lit_search.fetch.model import (
    Attempt,
    AttemptResult,
    FetchOutcome,
    FetchStatus,
    FetchTarget,
)
from lit_search.fetch.targets import parse_identifier, target_from_paper

__all__ = [
    "STEPS",
    "Attempt",
    "AttemptResult",
    "CurlTransport",
    "FetchOutcome",
    "FetchStatus",
    "FetchTarget",
    "HttpGet",
    "HttpResponse",
    "PdfFetcher",
    "TransportError",
    "is_pdf",
    "parse_identifier",
    "target_from_paper",
]
