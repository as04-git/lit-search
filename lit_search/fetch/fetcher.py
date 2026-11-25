"""PdfFetcher: an ordered, fully recorded chain of places a PDF might be.

For each :class:`~lit_search.fetch.model.FetchTarget` the chain is

1. ``arxiv``     -- ``https://arxiv.org/pdf/<id>`` for arXiv ids and arXiv DOIs;
2. ``pdf_url``   -- the paper's own known PDF URL;
3. ``openalex``  -- open-access locations OpenAlex lists for the DOI
   (``best_oa_location`` then ``locations[]``; ``pdf_url`` first, then OA
   landing pages);
4. ``publisher`` -- the doi.org landing page, then its ``citation_pdf_url``
   meta tag or a known publisher URL pattern
   (:class:`~lit_search.runners.download_runner.DownloadRunner` helpers);
5. ``scihub``    -- only when enabled; each mirror in order.

The chain stops at the first valid PDF.  Every request is an
:class:`~lit_search.fetch.model.Attempt`; steps that cannot or may not run are
recorded as ``skipped`` with the reason.  A body is accepted as a PDF only on
its ``%PDF-`` magic bytes, never on its content-type, and files are written
atomically.

Verdicts: ``downloaded`` / ``already_present`` when a PDF is on disk;
``failed`` when some step produced *only* errors (so its silence is not a
verdict) or the file could not be written; ``not_found`` otherwise.
"""

from __future__ import annotations

import json
import os
import tempfile
import urllib.parse
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lit_search.fetch import sci_hub
from lit_search.fetch.http import CurlTransport, HttpGet, HttpResponse, TransportError
from lit_search.fetch.model import (
    Attempt,
    AttemptResult,
    FetchOutcome,
    FetchStatus,
    FetchTarget,
)
from lit_search.runners.download_runner import DownloadRunner

#: Steps of the chain, in order.
STEPS: tuple[str, ...] = ("arxiv", "pdf_url", "openalex", "publisher", "scihub")

#: The magic bytes must appear within this many leading bytes (the PDF
#: specification's own tolerance, which Acrobat applies).
PDF_MAGIC_WINDOW = 1024
PDF_MAGIC = b"%PDF-"
PDF_EOF = b"%%EOF"
PDF_EOF_WINDOW = 2048
PDF_MIN_BYTES = 1024

#: At most this many OpenAlex-listed candidate URLs are requested per paper.
MAX_OPENALEX_CANDIDATES = 4

OPENALEX_WORKS_URL = "https://api.openalex.org/works/doi:"
_OPENALEX_SELECT = "id,doi,open_access,best_oa_location,locations"

_BLOCKED_STATUSES = frozenset({401, 403, 418, 429})
_MISSING_STATUSES = frozenset({404, 410})
_HTML_TYPES = ("text/html", "application/xhtml+xml")


def is_pdf(content: bytes) -> bool:
    """Return whether *content* is plausibly a whole PDF.

    Parameters
    ----------
    content : bytes
        Response body or file contents.

    Returns
    -------
    bool
        ``True`` when ``%PDF-`` occurs within the first
        :data:`PDF_MAGIC_WINDOW` bytes, the body is at least
        :data:`PDF_MIN_BYTES` long, and an ``%%EOF`` marker occurs in its last
        :data:`PDF_EOF_WINDOW` bytes.  A bare header, or a download cut off
        mid-transfer, is not a PDF.
    """
    return (
        len(content) >= PDF_MIN_BYTES
        and PDF_MAGIC in content[:PDF_MAGIC_WINDOW]
        and PDF_EOF in content[-PDF_EOF_WINDOW:]
    )


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Write *content* to *path* via a same-directory temp file and rename.

    Parameters
    ----------
    path : Path
        Destination.
    content : bytes
        Data.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If the write or rename fails; the temporary file is removed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".part"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        # mkstemp creates 0600; give the PDF the mode a plain open() would.
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(temporary, 0o666 & ~umask)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _host(url: str) -> str:
    """Return the host of *url* (for compact labels)."""
    return urllib.parse.urlsplit(url).netloc or url


def _is_absolute(url: str | None) -> bool:
    """Return whether *url* is an absolute http(s) URL."""
    return bool(url) and str(url).startswith(("http://", "https://"))


@dataclass
class _Run:
    """Mutable state for fetching one target."""

    target: FetchTarget
    attempts: list[Attempt] = field(default_factory=list)
    tried: set[str] = field(default_factory=set)

    def record(
        self,
        step: str,
        url: str | None,
        result: AttemptResult,
        reason: str,
        final_url: str | None = None,
    ) -> None:
        """Append one attempt."""
        if final_url is not None and final_url == url:
            final_url = None
        self.attempts.append(Attempt(step, url, result, reason, final_url))


@dataclass(frozen=True)
class _Found:
    """A PDF body and the step that produced it."""

    step: str
    content: bytes


class PdfFetcher:
    """Fetch PDFs through the recorded chain described in the module docstring.

    Parameters
    ----------
    output_dir : str | Path
        Directory PDFs are written to (created on first write).
    transport : HttpGet | None
        GET implementation.  Defaults to a :class:`CurlTransport`, which this
        fetcher then owns and closes in :meth:`close`.
    scihub : bool
        Whether the Sci-Hub step may run.  When ``False`` it is recorded as
        ``skipped: disabled``.
    scihub_mirrors : Sequence[str] | None
        Mirror base URLs, tried in order.  Defaults to
        :func:`lit_search.fetch.sci_hub.mirrors_from_env`.
    openalex_api_key : str | None
        OpenAlex key, sent as ``api_key`` and never recorded in an attempt URL.
    timeout : float
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        transport: HttpGet | None = None,
        scihub: bool = False,
        scihub_mirrors: Sequence[str] | None = None,
        openalex_api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._output_dir = Path(output_dir).expanduser()
        self._owned_transport: CurlTransport | None = None
        if transport is None:
            self._owned_transport = CurlTransport()
            transport = self._owned_transport
        self._get: HttpGet = transport
        self._scihub = scihub
        self._mirrors = (
            tuple(scihub_mirrors) if scihub_mirrors is not None else sci_hub.mirrors_from_env()
        )
        self._openalex_api_key = openalex_api_key
        self._timeout = timeout

    @property
    def output_dir(self) -> Path:
        """Directory PDFs are written to.

        Returns
        -------
        Path
            Output directory.
        """
        return self._output_dir

    def close(self) -> None:
        """Release the transport this fetcher created, if any.

        Returns
        -------
        None
        """
        if self._owned_transport is not None:
            self._owned_transport.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_all(
        self,
        targets: Iterable[FetchTarget],
        on_outcome: Callable[[FetchOutcome], None] | None = None,
    ) -> list[FetchOutcome]:
        """Fetch every target in turn; one outcome each, in input order.

        Parameters
        ----------
        targets : Iterable[FetchTarget]
            What to fetch.
        on_outcome : Callable[[FetchOutcome], None] | None
            Called as each outcome is decided (for streaming output).

        Returns
        -------
        list[FetchOutcome]
            Outcomes.
        """
        outcomes: list[FetchOutcome] = []
        for target in targets:
            outcome = self.fetch(target)
            outcomes.append(outcome)
            if on_outcome is not None:
                on_outcome(outcome)
        return outcomes

    def fetch(self, target: FetchTarget) -> FetchOutcome:
        """Fetch one target.

        Parameters
        ----------
        target : FetchTarget
            What to fetch.

        Returns
        -------
        FetchOutcome
            The verdict and every attempt.  Network and HTTP failures are
            recorded, never raised.
        """
        run = _Run(target)
        path = self._output_dir / target.filename()

        if path.exists():
            try:
                existing = path.read_bytes()
            except OSError as error:
                message = f"existing file unreadable: {type(error).__name__}: {error}"
                run.record("existing", str(path), AttemptResult.ERROR, message)
                return FetchOutcome(target, FetchStatus.FAILED, tuple(run.attempts), error=message)
            if is_pdf(existing):
                run.record("existing", str(path), AttemptResult.PDF, "already present")
                return FetchOutcome(
                    target,
                    FetchStatus.ALREADY_PRESENT,
                    tuple(run.attempts),
                    path=str(path),
                    size=path.stat().st_size,
                )
            run.record(
                "existing",
                str(path),
                AttemptResult.MISS,
                "existing file is not a PDF; fetching a replacement",
            )

        found = self._run_chain(run)
        if found is None:
            return FetchOutcome(target, self._verdict(run), tuple(run.attempts))

        try:
            atomic_write_bytes(path, found.content)
        except OSError as error:
            message = f"{type(error).__name__}: {error}"
            run.record("write", str(path), AttemptResult.ERROR, message)
            return FetchOutcome(
                target, FetchStatus.FAILED, tuple(run.attempts), via=found.step, error=message
            )
        return FetchOutcome(
            target,
            FetchStatus.DOWNLOADED,
            tuple(run.attempts),
            path=str(path),
            via=found.step,
            size=len(found.content),
        )

    # ------------------------------------------------------------------
    # Chain
    # ------------------------------------------------------------------

    def _run_chain(self, run: _Run) -> _Found | None:
        """Run each step until one yields a PDF."""
        steps: tuple[Callable[[_Run], bytes | None], ...] = (
            self._step_arxiv,
            self._step_pdf_url,
            self._step_openalex,
            self._step_publisher,
            self._step_scihub,
        )
        for name, step in zip(STEPS, steps, strict=True):
            content = step(run)
            if content is not None:
                return _Found(name, content)
        return None

    @staticmethod
    def _verdict(run: _Run) -> FetchStatus:
        """Classify a run that produced no PDF.

        ``not_found`` needs a definite negative from every step that ran.  A
        step that produced only errors makes the run ``failed``; a step that
        was refused, or answered without a definite yes or no, leaves absence
        unestablished: ``unresolved``.
        """
        by_step: dict[str, list[AttemptResult]] = {}
        for attempt in run.attempts:
            if attempt.result is not AttemptResult.SKIPPED:
                by_step.setdefault(attempt.step, []).append(attempt.result)
        if all(AttemptResult.MISS in results for results in by_step.values()):
            return FetchStatus.NOT_FOUND
        if any(all(r is AttemptResult.ERROR for r in results) for results in by_step.values()):
            return FetchStatus.FAILED
        return FetchStatus.UNRESOLVED

    def _step_arxiv(self, run: _Run) -> bytes | None:
        arxiv_id = run.target.arxiv_id
        if not arxiv_id:
            run.record("arxiv", None, AttemptResult.SKIPPED, "no arXiv id")
            return None
        return self._try_url(run, "arxiv", f"https://arxiv.org/pdf/{arxiv_id}")

    def _step_pdf_url(self, run: _Run) -> bytes | None:
        pdf_url = run.target.pdf_url
        if not pdf_url:
            run.record("pdf_url", None, AttemptResult.SKIPPED, "no known PDF URL")
            return None
        if not _is_absolute(pdf_url):
            run.record("pdf_url", pdf_url, AttemptResult.SKIPPED, "not an absolute URL")
            return None
        return self._try_url(run, "pdf_url", pdf_url, follow_landing=True)

    def _step_openalex(self, run: _Run) -> bytes | None:
        doi = run.target.doi
        if not doi:
            run.record("openalex", None, AttemptResult.SKIPPED, "no DOI")
            return None
        params: dict[str, str] = {"select": _OPENALEX_SELECT}
        public_url = (
            f"{OPENALEX_WORKS_URL}{urllib.parse.quote(doi, safe='/')}"
            f"?{urllib.parse.urlencode(params)}"
        )
        request_url = public_url
        if self._openalex_api_key:
            request_url += "&" + urllib.parse.urlencode({"api_key": self._openalex_api_key})

        response = self._request(run, "openalex", public_url, request_url)
        if response is None:
            return None
        if response.status in _MISSING_STATUSES:
            run.record("openalex", public_url, AttemptResult.MISS, "DOI not in OpenAlex")
            return None
        if not self._status_ok(run, "openalex", public_url, response):
            return None
        try:
            work = json.loads(response.content)
        except ValueError as error:
            run.record("openalex", public_url, AttemptResult.ERROR, f"unparseable JSON: {error}")
            return None
        if not isinstance(work, dict):
            run.record("openalex", public_url, AttemptResult.ERROR, "unexpected JSON shape")
            return None

        pdf_urls, landing_urls = self._openalex_candidates(work)
        candidates = [(url, False) for url in pdf_urls] + [(url, True) for url in landing_urls]
        candidates = [(url, follow) for url, follow in candidates if url not in run.tried]
        if not candidates:
            is_oa = (work.get("open_access") or {}).get("is_oa")
            reason = "not open access" if is_oa is False else "no OA location"
            run.record("openalex", public_url, AttemptResult.MISS, reason)
            return None
        shown = candidates[:MAX_OPENALEX_CANDIDATES]
        dropped = len(candidates) - len(shown)
        reason = f"{len(candidates)} OA candidate(s)"
        if dropped:
            reason += f"; trying the first {len(shown)}"
        run.record("openalex", public_url, AttemptResult.LOOKUP, reason)
        for url, follow in shown:
            content = self._try_url(run, "openalex", url, follow_landing=follow)
            if content is not None:
                return content
        return None

    @staticmethod
    def _openalex_candidates(work: dict[str, Any]) -> tuple[list[str], list[str]]:
        """Return (PDF URLs, non-doi.org OA landing pages), best location first."""
        locations: list[dict[str, Any]] = []
        best = work.get("best_oa_location")
        if isinstance(best, dict):
            locations.append(best)
        for location in work.get("locations") or []:
            if isinstance(location, dict) and location.get("is_oa"):
                locations.append(location)
        pdf_urls: list[str] = []
        landing_urls: list[str] = []
        for location in locations:
            pdf_url = location.get("pdf_url")
            if isinstance(pdf_url, str) and _is_absolute(pdf_url) and pdf_url not in pdf_urls:
                pdf_urls.append(pdf_url)
            landing = location.get("landing_page_url")
            if (
                isinstance(landing, str)
                and _is_absolute(landing)
                and _host(landing) not in {"doi.org", "dx.doi.org"}
                and landing not in landing_urls
            ):
                landing_urls.append(landing)
        return pdf_urls, [url for url in landing_urls if url not in pdf_urls]

    def _step_publisher(self, run: _Run) -> bytes | None:
        target = run.target
        if target.doi:
            url = f"https://doi.org/{target.doi}"
        elif _is_absolute(target.landing_url):
            url = str(target.landing_url)
        else:
            run.record("publisher", None, AttemptResult.SKIPPED, "no DOI or landing URL")
            return None
        return self._try_url(run, "publisher", url, follow_landing=True)

    def _step_scihub(self, run: _Run) -> bytes | None:
        if not self._scihub:
            run.record("scihub", None, AttemptResult.SKIPPED, "disabled")
            return None
        doi = run.target.doi
        if not doi:
            run.record("scihub", None, AttemptResult.SKIPPED, "no DOI")
            return None
        if not self._mirrors:
            run.record("scihub", None, AttemptResult.SKIPPED, "no mirrors configured")
            return None
        for mirror in self._mirrors:
            content = self._try_mirror(run, mirror, doi)
            if content is not None:
                return content
        return None

    def _try_mirror(self, run: _Run, mirror: str, doi: str) -> bytes | None:
        page_url = sci_hub.article_url(mirror, doi)
        response = self._request(run, "scihub", page_url)
        if response is None or not self._status_ok(run, "scihub", page_url, response):
            return None
        if is_pdf(response.content):
            run.record(
                "scihub", page_url, AttemptResult.PDF, self._pdf_reason(response), response.url
            )
            return response.content
        try:
            link = sci_hub.find_pdf_link(response.content, response.url)
        except ValueError as error:
            run.record("scihub", page_url, AttemptResult.ERROR, str(error), response.url)
            return None
        if link is None:
            if sci_hub.is_challenge_page(response.content):
                run.record("scihub", page_url, AttemptResult.BLOCKED, "bot challenge", response.url)
            else:
                run.record(
                    "scihub", page_url, AttemptResult.MISS, "no PDF link on page", response.url
                )
            return None
        if link in run.tried:
            run.record(
                "scihub", page_url, AttemptResult.LOOKUP, "PDF link already tried", response.url
            )
            return None
        run.record("scihub", page_url, AttemptResult.LOOKUP, "PDF link found", response.url)
        return self._try_url(run, "scihub", link)

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _request(
        self, run: _Run, step: str, public_url: str, request_url: str | None = None
    ) -> HttpResponse | None:
        """GET, recording a transport failure as an error attempt."""
        run.tried.add(public_url)
        try:
            return self._get(request_url or public_url, timeout=self._timeout)
        except TransportError as error:
            run.record(step, public_url, AttemptResult.ERROR, f"network: {error}")
            return None

    @staticmethod
    def _status_ok(run: _Run, step: str, url: str, response: HttpResponse) -> bool:
        """Record a non-2xx response; return whether the status is 2xx."""
        status = response.status
        if 200 <= status < 300:
            return True
        final = response.url
        if status in _BLOCKED_STATUSES:
            run.record(step, url, AttemptResult.BLOCKED, f"HTTP {status}", final)
        elif status in _MISSING_STATUSES:
            run.record(step, url, AttemptResult.MISS, f"HTTP {status}", final)
        else:
            run.record(step, url, AttemptResult.ERROR, f"HTTP {status}", final)
        return False

    @staticmethod
    def _pdf_reason(response: HttpResponse) -> str:
        return f"PDF, {len(response.content)} bytes"

    def _try_url(
        self, run: _Run, step: str, url: str, *, follow_landing: bool = False
    ) -> bytes | None:
        """Fetch *url*; accept it only if it is a PDF by magic bytes.

        With *follow_landing*, an HTML answer is searched for a
        ``citation_pdf_url`` meta tag and a known publisher PDF pattern, and
        each found URL is tried once (without further following).
        """
        if url in run.tried:
            return None
        response = self._request(run, step, url)
        if response is None or not self._status_ok(run, step, url, response):
            return None
        if is_pdf(response.content):
            run.record(step, url, AttemptResult.PDF, self._pdf_reason(response), response.url)
            return response.content
        content_type = response.content_type or "no content-type"
        if not (follow_landing and response.content_type in _HTML_TYPES):
            if response.content_type == "application/pdf":
                reason = "content-type says PDF but body is not one"
            else:
                reason = f"not a PDF ({content_type})"
            run.record(step, url, AttemptResult.MISS, reason, response.url)
            return None

        candidates: list[tuple[str, str]] = []
        meta = DownloadRunner._extract_meta_pdf_url(response.content, base_url=response.url)
        if meta:
            candidates.append(("citation_pdf_url", meta))
        pattern = DownloadRunner._resolve_pdf_url(response.url, doi=run.target.doi)
        if pattern and pattern != meta:
            candidates.append(("URL pattern", pattern))
        candidates = [(kind, link) for kind, link in candidates if link not in run.tried]
        if not candidates:
            if sci_hub.is_challenge_page(response.content):
                run.record(step, url, AttemptResult.BLOCKED, "bot challenge", response.url)
            else:
                run.record(step, url, AttemptResult.MISS, "landing page, no PDF link", response.url)
            return None
        kinds = ", ".join(kind for kind, _ in candidates)
        run.record(step, url, AttemptResult.LOOKUP, f"landing page -> {kinds}", response.url)
        for _, link in candidates:
            content = self._try_url(run, step, link)
            if content is not None:
                return content
        return None


__all__ = [
    "MAX_OPENALEX_CANDIDATES",
    "PDF_MAGIC",
    "STEPS",
    "PdfFetcher",
    "atomic_write_bytes",
    "is_pdf",
]
