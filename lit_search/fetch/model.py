"""Typed records for PDF fetching: targets, attempts and per-paper outcomes.

Every request the fetcher makes becomes an :class:`Attempt`, and every paper
ends in exactly one :class:`FetchOutcome`.  Nothing is reported as a bare
``bool`` or ``None``: a miss says which steps were tried and what each one
answered, and a step that was not tried says why (``skipped: disabled``).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class AttemptResult(StrEnum):
    """What one request (or one decision not to make a request) produced."""

    #: A valid PDF (``%PDF-`` magic bytes) was received.
    PDF = "pdf"
    #: A metadata or mirror page answered and pointed at further candidates.
    LOOKUP = "lookup"
    #: A definitive negative: 404, no open-access location, no PDF link, HTML instead of PDF.
    MISS = "miss"
    #: The source refused: 401/403/418/429 or a bot-challenge page.
    BLOCKED = "blocked"
    #: No answer was obtained: network failure, 5xx, unparseable payload, write failure.
    ERROR = "error"
    #: The step was not tried, with the reason (``disabled``, ``no DOI``, ...).
    SKIPPED = "skipped"


class FetchStatus(StrEnum):
    """Final verdict for one paper."""

    DOWNLOADED = "downloaded"
    ALREADY_PRESENT = "already_present"
    #: Every step that ran reached a verdict and none produced a PDF.
    NOT_FOUND = "not_found"
    #: An error prevented a verdict: some step produced only errors, or the file could not be written.
    FAILED = "failed"
    #: No PDF, but absence is not established: a step was refused (403/429/bot check)
    #: or answered without reaching a definite yes or no.
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class Attempt:
    """One step of the fetch chain, as it happened.

    Attributes
    ----------
    step : str
        Chain step (``arxiv``, ``pdf_url``, ``openalex``, ``publisher``,
        ``scihub``, ``existing`` or ``write``).
    url : str | None
        URL requested (never carries credentials), or ``None`` for a skip.
    result : AttemptResult
        What happened.
    reason : str
        Short human-readable explanation, e.g. ``HTTP 403``.
    final_url : str | None
        URL after redirects, when it differs from *url*.
    """

    step: str
    url: str | None
    result: AttemptResult
    reason: str
    final_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping.

        Returns
        -------
        dict[str, Any]
            The attempt's fields.
        """
        payload: dict[str, Any] = {
            "step": self.step,
            "url": self.url,
            "result": self.result.value,
            "reason": self.reason,
        }
        if self.final_url is not None:
            payload["final_url"] = self.final_url
        return payload


_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_STEM = 150


def _safe_stem(text: str) -> str:
    """Return *text* reduced to a filesystem-safe stem.

    When the reduction loses information (characters replaced, or the stem
    truncated), a short hash of the original text is appended so that two
    distinct identifiers -- ``10.1/a/b`` and ``10.1/a_b`` -- never share a file.
    """
    stem = _UNSAFE_FILENAME.sub("_", text).strip("._")
    if stem == text and len(stem) <= _MAX_STEM:
        return stem or "paper"
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{stem[:_MAX_STEM] or 'paper'}-{digest}"


@dataclass(frozen=True)
class FetchTarget:
    """What to fetch, and every identifier known for it.

    Attributes
    ----------
    label : str
        How the paper is named in output (the identifier the user gave, or
        the paper's DOI / arXiv id / title).
    doi : str | None
        Bare DOI.
    arxiv_id : str | None
        arXiv identifier, e.g. ``2310.06825`` or ``hep-th/9901001``.
    pdf_url : str | None
        Direct PDF URL already known for the paper.
    landing_url : str | None
        Landing page, used by the publisher step only when there is no DOI.
    title : str | None
        Title, used for the filename only when no identifier exists.
    year : int | None
        Publication year, used with *title* for the filename.
    """

    label: str
    doi: str | None = None
    arxiv_id: str | None = None
    pdf_url: str | None = None
    landing_url: str | None = None
    title: str | None = None
    year: int | None = None

    def filename(self) -> str:
        """Return the deterministic PDF filename for this target.

        The arXiv id wins over the DOI so that ``2310.06825`` and
        ``10.48550/arXiv.2310.06825`` land on the same file.

        Returns
        -------
        str
            File name ending in ``.pdf``.
        """
        if self.arxiv_id:
            return f"arxiv-{_safe_stem(self.arxiv_id)}.pdf"
        if self.doi:
            return f"doi-{_safe_stem(self.doi.lower())}.pdf"
        if self.title:
            prefix = f"{self.year}-" if self.year else ""
            return f"{prefix}{_safe_stem(self.title.lower())}.pdf"
        return f"{_safe_stem(self.label)}.pdf"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping.

        Returns
        -------
        dict[str, Any]
            The non-empty identifiers.
        """
        fields = {
            "label": self.label,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "pdf_url": self.pdf_url,
            "landing_url": self.landing_url,
            "title": self.title,
            "year": self.year,
        }
        return {key: value for key, value in fields.items() if value is not None}


@dataclass(frozen=True)
class FetchOutcome:
    """The verdict for one paper, with every attempt that led to it.

    Attributes
    ----------
    target : FetchTarget
        What was fetched.
    status : FetchStatus
        Verdict.
    attempts : tuple[Attempt, ...]
        Every step, in order, including skipped ones.
    path : str | None
        File on disk for ``downloaded`` and ``already_present``.
    via : str | None
        Step that produced the PDF, for ``downloaded``.
    size : int | None
        File size in bytes, for ``downloaded`` and ``already_present``.
    error : str | None
        What prevented a verdict, for ``failed``.
    """

    target: FetchTarget
    status: FetchStatus
    attempts: tuple[Attempt, ...] = field(default_factory=tuple)
    path: str | None = None
    via: str | None = None
    size: int | None = None
    error: str | None = None

    @property
    def has_pdf(self) -> bool:
        """Whether a PDF is on disk for this paper after the run.

        Returns
        -------
        bool
            ``True`` for ``downloaded`` and ``already_present``.
        """
        return self.status in (FetchStatus.DOWNLOADED, FetchStatus.ALREADY_PRESENT)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping.

        Returns
        -------
        dict[str, Any]
            Status first, then location, then the full attempt list.
        """
        return {
            "status": self.status.value,
            "target": self.target.to_dict(),
            "path": self.path,
            "via": self.via,
            "bytes": self.size,
            "error": self.error,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }


__all__ = [
    "Attempt",
    "AttemptResult",
    "FetchOutcome",
    "FetchStatus",
    "FetchTarget",
]
