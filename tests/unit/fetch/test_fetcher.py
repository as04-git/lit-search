"""Tests for the recorded PDF fetch chain, with a fake HTTP transport.

The suite runs under ``--disable-socket``; every request goes through
:class:`FakeTransport`, which also fails the test on any URL it was not told
about, so an unexpected request cannot pass silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lit_search.fetch import sci_hub
from lit_search.fetch.fetcher import PdfFetcher, atomic_write_bytes, is_pdf
from lit_search.fetch.http import HttpResponse, TransportError
from lit_search.fetch.model import AttemptResult, FetchStatus, FetchTarget
from lit_search.fetch.targets import parse_identifier, target_from_paper

PDF = b"%PDF-1.7\n" + b"x" * 2000 + b"\n%%EOF\n"
HTML = "text/html"
DOI = "10.1234/example.5"
DOI_URL = f"https://doi.org/{DOI}"
OPENALEX_PREFIX = f"https://api.openalex.org/works/doi:{DOI}?"


def ok(url: str, content: bytes, content_type: str = "application/pdf") -> HttpResponse:
    """Build a 200 response."""
    return HttpResponse(200, url, content_type, content)


def status(url: str, code: int) -> HttpResponse:
    """Build an empty response with *code*."""
    return HttpResponse(code, url, HTML, b"<html></html>")


def landing(url: str, pdf_link: str | None) -> HttpResponse:
    """Build an HTML landing page, optionally carrying citation_pdf_url."""
    meta = f'<meta name="citation_pdf_url" content="{pdf_link}">' if pdf_link else ""
    body = f"<html><head>{meta}<title>t</title></head><body>article</body></html>"
    return HttpResponse(200, url, HTML, body.encode())


def openalex(work: dict) -> HttpResponse:
    """Build an OpenAlex works response."""
    return HttpResponse(
        200, "https://api.openalex.org/works/x", "application/json", json.dumps(work).encode()
    )


class FakeTransport:
    """Answer GETs from a table; anything unlisted fails the test."""

    def __init__(self, routes: dict[str, HttpResponse | Exception]) -> None:
        self.routes = routes
        self.calls: list[str] = []

    def _lookup(self, url: str) -> HttpResponse | Exception:
        if url in self.routes:
            return self.routes[url]
        for key, value in self.routes.items():
            if key.endswith("?") and url.startswith(key):
                return value
        raise AssertionError(f"unexpected request: {url}")

    def __call__(self, url: str, *, timeout: float) -> HttpResponse:
        self.calls.append(url)
        answer = self._lookup(url)
        if isinstance(answer, Exception):
            raise answer
        return answer


def make_fetcher(tmp_path: Path, routes: dict, **kwargs) -> tuple[PdfFetcher, FakeTransport]:
    """Return a fetcher over a fake transport writing into *tmp_path*."""
    transport = FakeTransport(routes)
    kwargs.setdefault("scihub_mirrors", ("https://mirror-a.test", "https://mirror-b.test"))
    return PdfFetcher(tmp_path, transport=transport, **kwargs), transport


def results(outcome) -> list[tuple[str, AttemptResult, str]]:
    """Return (step, result, reason) triples."""
    return [(a.step, a.result, a.reason) for a in outcome.attempts]


NO_OA = openalex({"open_access": {"is_oa": False}, "best_oa_location": None, "locations": []})


class TestChainSteps:
    """Each step can produce the PDF, and the chain stops there."""

    def test_arxiv_step_downloads_and_stops(self, tmp_path: Path) -> None:
        target = parse_identifier("2310.06825")
        fetcher, transport = make_fetcher(
            tmp_path,
            {"https://arxiv.org/pdf/2310.06825": ok("https://arxiv.org/pdf/2310.06825", PDF)},
        )
        outcome = fetcher.fetch(target)
        assert outcome.status is FetchStatus.DOWNLOADED
        assert outcome.via == "arxiv"
        assert outcome.size == len(PDF)
        assert Path(outcome.path).read_bytes() == PDF
        assert Path(outcome.path).name == "arxiv-2310.06825.pdf"
        assert transport.calls == ["https://arxiv.org/pdf/2310.06825"]

    def test_arxiv_doi_uses_the_arxiv_step(self, tmp_path: Path) -> None:
        target = parse_identifier("10.48550/arXiv.2310.06825")
        assert target.arxiv_id == "2310.06825"
        fetcher, _ = make_fetcher(
            tmp_path,
            {"https://arxiv.org/pdf/2310.06825": ok("https://arxiv.org/pdf/2310.06825", PDF)},
        )
        assert fetcher.fetch(target).via == "arxiv"

    def test_known_pdf_url_step(self, tmp_path: Path) -> None:
        target = FetchTarget(label="x", doi=DOI, pdf_url="https://repo.test/x.pdf")
        fetcher, _ = make_fetcher(
            tmp_path, {"https://repo.test/x.pdf": ok("https://repo.test/x.pdf", PDF)}
        )
        outcome = fetcher.fetch(target)
        assert outcome.via == "pdf_url"
        assert results(outcome)[0] == ("arxiv", AttemptResult.SKIPPED, "no arXiv id")

    def test_openalex_oa_pdf_url_step(self, tmp_path: Path) -> None:
        work = {
            "open_access": {"is_oa": True},
            "best_oa_location": {
                "is_oa": True,
                "pdf_url": "https://oa.test/a.pdf",
                "landing_page_url": DOI_URL,
            },
            "locations": [],
        }
        fetcher, transport = make_fetcher(
            tmp_path,
            {
                OPENALEX_PREFIX: openalex(work),
                "https://oa.test/a.pdf": ok("https://oa.test/a.pdf", PDF),
            },
            openalex_api_key="SECRET-KEY",
        )
        outcome = fetcher.fetch(FetchTarget(label=DOI, doi=DOI))
        assert outcome.via == "openalex"
        # The key is sent but never recorded.
        assert "api_key=SECRET-KEY" in transport.calls[0]
        assert "mailto" not in transport.calls[0]
        assert all("SECRET-KEY" not in (a.url or "") for a in outcome.attempts)
        assert "SECRET-KEY" not in json.dumps(outcome.to_dict())

    def test_openalex_oa_landing_page_is_followed(self, tmp_path: Path) -> None:
        work = {
            "best_oa_location": {
                "is_oa": True,
                "pdf_url": None,
                "landing_page_url": "https://hal.test/hal-1",
            },
            "locations": [],
        }
        fetcher, _ = make_fetcher(
            tmp_path,
            {
                OPENALEX_PREFIX: openalex(work),
                "https://hal.test/hal-1": landing(
                    "https://hal.test/hal-1", "https://hal.test/hal-1/file.pdf"
                ),
                "https://hal.test/hal-1/file.pdf": ok("https://hal.test/hal-1/file.pdf", PDF),
            },
        )
        outcome = fetcher.fetch(FetchTarget(label=DOI, doi=DOI))
        assert outcome.via == "openalex"

    def test_publisher_citation_pdf_url_step(self, tmp_path: Path) -> None:
        fetcher, _ = make_fetcher(
            tmp_path,
            {
                OPENALEX_PREFIX: NO_OA,
                DOI_URL: landing("https://pub.test/article/5", "/article/5.pdf"),
                "https://pub.test/article/5.pdf": ok("https://pub.test/article/5.pdf", PDF),
            },
        )
        outcome = fetcher.fetch(FetchTarget(label=DOI, doi=DOI))
        assert outcome.via == "publisher"
        assert ("openalex", AttemptResult.MISS, "not open access") in results(outcome)

    def test_scihub_step_when_enabled(self, tmp_path: Path) -> None:
        page = b'<html><body><embed type="application/pdf" src="//store.test/p.pdf#view=FitH"></body></html>'
        fetcher, _ = make_fetcher(
            tmp_path,
            {
                OPENALEX_PREFIX: NO_OA,
                DOI_URL: status(DOI_URL, 403),
                f"https://mirror-a.test/{DOI}": ok(f"https://mirror-a.test/{DOI}", page, HTML),
                "https://store.test/p.pdf": ok("https://store.test/p.pdf", PDF),
            },
            scihub=True,
        )
        outcome = fetcher.fetch(FetchTarget(label=DOI, doi=DOI))
        assert outcome.status is FetchStatus.DOWNLOADED
        assert outcome.via == "scihub"
        assert ("publisher", AttemptResult.BLOCKED, "HTTP 403") in results(outcome)


class TestSciHubMirrors:
    """Mirrors are tried in order and each one is a recorded attempt."""

    def test_down_mirror_is_recorded_and_next_mirror_used(self, tmp_path: Path) -> None:
        page = b'<html><iframe src="https://ads.test/x.html"></iframe><iframe id="pdf" src="/d/p.pdf"></iframe></html>'
        fetcher, transport = make_fetcher(
            tmp_path,
            {
                OPENALEX_PREFIX: NO_OA,
                DOI_URL: status(DOI_URL, 404),
                f"https://mirror-a.test/{DOI}": TransportError(
                    "curl 6: Could not resolve host: mirror-a.test"
                ),
                f"https://mirror-b.test/{DOI}": ok(f"https://mirror-b.test/{DOI}", page, HTML),
                "https://mirror-b.test/d/p.pdf": ok("https://mirror-b.test/d/p.pdf", PDF),
            },
            scihub=True,
        )
        outcome = fetcher.fetch(FetchTarget(label=DOI, doi=DOI))
        assert outcome.status is FetchStatus.DOWNLOADED
        scihub = [a for a in outcome.attempts if a.step == "scihub"]
        assert scihub[0].url == f"https://mirror-a.test/{DOI}"
        assert scihub[0].result is AttemptResult.ERROR
        assert "Could not resolve host" in scihub[0].reason
        # The advertising iframe was not mistaken for the PDF.
        assert "https://ads.test/x.html" not in transport.calls

    def test_all_mirrors_answering_without_pdf_is_not_found(self, tmp_path: Path) -> None:
        challenge = b"<html><head><title>Are you a robot?</title></head><altcha-widget></altcha-widget></html>"
        fetcher, _ = make_fetcher(
            tmp_path,
            {
                OPENALEX_PREFIX: NO_OA,
                DOI_URL: status(DOI_URL, 404),
                f"https://mirror-a.test/{DOI}": ok(f"https://mirror-a.test/{DOI}", challenge, HTML),
                f"https://mirror-b.test/{DOI}": ok(
                    f"https://mirror-b.test/{DOI}", b"<html>article not found</html>", HTML
                ),
            },
            scihub=True,
        )
        outcome = fetcher.fetch(FetchTarget(label=DOI, doi=DOI))
        assert outcome.status is FetchStatus.NOT_FOUND
        assert ("scihub", AttemptResult.BLOCKED, "bot challenge") in results(outcome)
        assert ("scihub", AttemptResult.MISS, "no PDF link on page") in results(outcome)

    def test_every_mirror_down_is_failed_not_not_found(self, tmp_path: Path) -> None:
        fetcher, _ = make_fetcher(
            tmp_path,
            {
                OPENALEX_PREFIX: NO_OA,
                DOI_URL: status(DOI_URL, 404),
                f"https://mirror-a.test/{DOI}": status(f"https://mirror-a.test/{DOI}", 504),
                f"https://mirror-b.test/{DOI}": TransportError("curl 28: timed out"),
            },
            scihub=True,
        )
        outcome = fetcher.fetch(FetchTarget(label=DOI, doi=DOI))
        assert outcome.status is FetchStatus.FAILED
        assert ("scihub", AttemptResult.ERROR, "HTTP 504") in results(outcome)

    def test_disabled_scihub_is_recorded_as_skipped(self, tmp_path: Path) -> None:
        fetcher, transport = make_fetcher(
            tmp_path, {OPENALEX_PREFIX: NO_OA, DOI_URL: status(DOI_URL, 404)}, scihub=False
        )
        outcome = fetcher.fetch(FetchTarget(label=DOI, doi=DOI))
        assert outcome.status is FetchStatus.NOT_FOUND
        assert results(outcome)[-1] == ("scihub", AttemptResult.SKIPPED, "disabled")
        assert not any("mirror" in call for call in transport.calls)


class TestValidation:
    """Only real PDFs are accepted, and only by their bytes."""

    def test_pdf_content_type_with_html_body_is_rejected(self, tmp_path: Path) -> None:
        url = "https://arxiv.org/pdf/2310.06825"
        fetcher, _ = make_fetcher(
            tmp_path,
            {
                url: ok(url, b"<html>login required</html>", "application/pdf"),
                "https://api.openalex.org/works/doi:10.48550/arXiv.2310.06825?": NO_OA,
                "https://doi.org/10.48550/arXiv.2310.06825": status("https://doi.org/x", 404),
            },
        )
        outcome = fetcher.fetch(parse_identifier("2310.06825"))
        assert outcome.status is FetchStatus.NOT_FOUND
        assert (
            "arxiv",
            AttemptResult.MISS,
            "content-type says PDF but body is not one",
        ) in results(outcome)
        assert list(tmp_path.iterdir()) == []

    def test_pdf_body_with_wrong_content_type_is_accepted(self, tmp_path: Path) -> None:
        url = "https://arxiv.org/pdf/2310.06825"
        fetcher, _ = make_fetcher(tmp_path, {url: ok(url, PDF, "application/octet-stream")})
        assert fetcher.fetch(parse_identifier("2310.06825")).status is FetchStatus.DOWNLOADED

    def test_is_pdf_magic(self) -> None:
        body = b"x" * 2000 + b"\n%%EOF\n"
        assert is_pdf(b"%PDF-1.4 " + body)
        assert is_pdf(b"\n\n%PDF-1.4" + body)
        assert not is_pdf(b"<html>" + b" " * 2000 + b"%PDF-" + body)

    def test_bare_header_or_truncated_body_is_not_a_pdf(self) -> None:
        assert not is_pdf(b"%PDF-1.7\n")
        assert not is_pdf(b"%PDF-1.7\n" + b"x" * 5000)
        assert not is_pdf(b"")


class TestErrorsAreRecorded:
    """Failures are attempts with reasons, never swallowed."""

    def test_openalex_server_error_and_bad_json(self, tmp_path: Path) -> None:
        fetcher, _ = make_fetcher(
            tmp_path,
            {
                OPENALEX_PREFIX: status("https://api.openalex.org/x", 503),
                DOI_URL: status(DOI_URL, 404),
            },
        )
        outcome = fetcher.fetch(FetchTarget(label=DOI, doi=DOI))
        assert ("openalex", AttemptResult.ERROR, "HTTP 503") in results(outcome)
        assert outcome.status is FetchStatus.FAILED

        bad = HttpResponse(200, "https://api.openalex.org/x", "application/json", b"{not json")
        fetcher, _ = make_fetcher(tmp_path, {OPENALEX_PREFIX: bad, DOI_URL: status(DOI_URL, 404)})
        outcome = fetcher.fetch(FetchTarget(label=DOI, doi=DOI))
        step, result, reason = results(outcome)[2]
        assert (step, result) == ("openalex", AttemptResult.ERROR)
        assert reason.startswith("unparseable JSON")

    def test_network_error_on_publisher_is_recorded(self, tmp_path: Path) -> None:
        fetcher, _ = make_fetcher(
            tmp_path,
            {OPENALEX_PREFIX: NO_OA, DOI_URL: TransportError("curl 35: TLS handshake failed")},
        )
        outcome = fetcher.fetch(FetchTarget(label=DOI, doi=DOI))
        assert outcome.status is FetchStatus.FAILED
        assert (
            "publisher",
            AttemptResult.ERROR,
            "network: curl 35: TLS handshake failed",
        ) in results(outcome)

    def test_openalex_404_is_a_verdict_not_an_error(self, tmp_path: Path) -> None:
        fetcher, _ = make_fetcher(
            tmp_path,
            {
                OPENALEX_PREFIX: status("https://api.openalex.org/x", 404),
                DOI_URL: status(DOI_URL, 404),
            },
        )
        outcome = fetcher.fetch(FetchTarget(label=DOI, doi=DOI))
        assert outcome.status is FetchStatus.NOT_FOUND
        assert ("openalex", AttemptResult.MISS, "DOI not in OpenAlex") in results(outcome)

    def test_write_failure_is_failed_with_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url = "https://arxiv.org/pdf/2310.06825"
        fetcher, _ = make_fetcher(tmp_path, {url: ok(url, PDF)})

        def refuse(path: Path, content: bytes) -> None:
            raise OSError(28, "No space left on device")

        monkeypatch.setattr("lit_search.fetch.fetcher.atomic_write_bytes", refuse)
        outcome = fetcher.fetch(parse_identifier("2310.06825"))
        assert outcome.status is FetchStatus.FAILED
        assert "No space left" in (outcome.error or "")
        assert outcome.attempts[-1].step == "write"

    def test_nothing_to_try_is_not_found_with_reasons(self, tmp_path: Path) -> None:
        fetcher, transport = make_fetcher(tmp_path, {}, scihub=True)
        outcome = fetcher.fetch(FetchTarget(label="Untitled", title="Untitled"))
        assert outcome.status is FetchStatus.NOT_FOUND
        assert transport.calls == []
        assert {result for _, result, _ in results(outcome)} == {AttemptResult.SKIPPED}
        assert ("scihub", AttemptResult.SKIPPED, "no DOI") in results(outcome)


class TestFiles:
    """Existing files short-circuit; writes are atomic."""

    def test_already_present_skips_the_network(self, tmp_path: Path) -> None:
        (tmp_path / "arxiv-2310.06825.pdf").write_bytes(PDF)
        fetcher, transport = make_fetcher(tmp_path, {})
        outcome = fetcher.fetch(parse_identifier("arXiv:2310.06825"))
        assert outcome.status is FetchStatus.ALREADY_PRESENT
        assert transport.calls == []

    def test_existing_non_pdf_is_replaced(self, tmp_path: Path) -> None:
        (tmp_path / "arxiv-2310.06825.pdf").write_bytes(b"<html>")
        url = "https://arxiv.org/pdf/2310.06825"
        fetcher, _ = make_fetcher(tmp_path, {url: ok(url, PDF)})
        outcome = fetcher.fetch(parse_identifier("2310.06825"))
        assert outcome.status is FetchStatus.DOWNLOADED
        assert outcome.attempts[0].step == "existing"
        assert (tmp_path / "arxiv-2310.06825.pdf").read_bytes() == PDF

    def test_atomic_write_leaves_no_temp_file(self, tmp_path: Path) -> None:
        target = tmp_path / "sub" / "a.pdf"
        atomic_write_bytes(target, PDF)
        assert target.read_bytes() == PDF
        assert [p.name for p in target.parent.iterdir()] == ["a.pdf"]


class TestTargets:
    """Identifier parsing and handle papers."""

    @pytest.mark.parametrize(
        ("raw", "doi", "arxiv_id"),
        [
            ("2310.06825", "10.48550/arXiv.2310.06825", "2310.06825"),
            ("arXiv:2310.06825v2", "10.48550/arXiv.2310.06825", "2310.06825v2"),
            ("https://arxiv.org/abs/2310.06825", "10.48550/arXiv.2310.06825", "2310.06825"),
            ("hep-th/9901001", "10.48550/arXiv.hep-th/9901001", "hep-th/9901001"),
            ("10.1038/nature14539", "10.1038/nature14539", None),
            ("https://doi.org/10.1038/nature14539", "10.1038/nature14539", None),
            ("doi:10.1038/nature14539", "10.1038/nature14539", None),
        ],
    )
    def test_parse(self, raw: str, doi: str, arxiv_id: str | None) -> None:
        target = parse_identifier(raw)
        assert target.doi == doi
        assert target.arxiv_id == arxiv_id

    def test_unrecognised_identifier_raises(self) -> None:
        with pytest.raises(ValueError, match="not a DOI or arXiv id"):
            parse_identifier("nature deep learning")

    def test_paper_target_keeps_identifiers(self, make_paper) -> None:
        paper = make_paper(doi="10.48550/arXiv.2310.06825", pdf_url="https://x.test/p.pdf")
        target = target_from_paper(paper)
        assert target.arxiv_id == "2310.06825"
        assert target.pdf_url == "https://x.test/p.pdf"
        assert target.filename() == "arxiv-2310.06825.pdf"


class TestSciHubConfig:
    """Enabling and mirror configuration come from the environment, loudly."""

    def test_enabled_flag(self) -> None:
        assert sci_hub.scihub_enabled_from_env({"LIT_SEARCH_SCIHUB": "1"})
        assert not sci_hub.scihub_enabled_from_env({})

    def test_mirror_override_and_validation(self) -> None:
        env = {"LIT_SEARCH_SCIHUB_MIRRORS": "https://a.test/, https://b.test"}
        assert sci_hub.mirrors_from_env(env) == ("https://a.test", "https://b.test")
        assert sci_hub.mirrors_from_env({}) == sci_hub.DEFAULT_MIRRORS
        with pytest.raises(ValueError, match="not an http"):
            sci_hub.mirrors_from_env({"LIT_SEARCH_SCIHUB_MIRRORS": "sci-hub.example"})
        with pytest.raises(ValueError, match="names no mirror"):
            sci_hub.mirrors_from_env({"LIT_SEARCH_SCIHUB_MIRRORS": " , "})

    @pytest.mark.parametrize(
        ("html", "expected"),
        [
            (
                b'<embed type="application/pdf" src="//s.test/a.pdf#view=FitH">',
                "https://s.test/a.pdf",
            ),
            (b'<iframe id="pdf" src="/x/a.pdf">', "https://m.test/x/a.pdf"),
            (
                b"<button onclick=\"location.href='//s.test/b.pdf?download=true'\">pdf</button>",
                "https://s.test/b.pdf?download=true",
            ),
            (b'<a href="https://s.test/c.pdf">download</a>', "https://s.test/c.pdf"),
            (b'<iframe src="https://ads.test/x.html"></iframe><p>nothing</p>', None),
        ],
    )
    def test_find_pdf_link(self, html: bytes, expected: str | None) -> None:
        assert (
            sci_hub.find_pdf_link(
                b"<html><body>" + html + b"</body></html>", "https://m.test/10.1/x"
            )
            == expected
        )

    def test_scihub_is_not_a_search_source(self) -> None:
        from lit_search.connectors import SEARCH_REGISTRY

        assert not any(
            "sci" in str(name).lower() and "hub" in str(name).lower() for name in SEARCH_REGISTRY
        )


class TestCurlMessage:
    """Transport errors are condensed to one line, not dropped."""

    def test_short_curl_message(self) -> None:
        from lit_search.fetch.http import short_curl_message

        raw = (
            "Failed to perform, curl: (6) Could not resolve host: sci-hub.se. "
            "See https://curl.se/libcurl/c/libcurl-errors.html first for more details."
        )
        assert short_curl_message(Exception(raw)) == "curl 6: Could not resolve host: sci-hub.se"
        assert short_curl_message(Exception("boom\nmore")) == "boom"


class TestReviewRegressions:
    """Defects found in the 2026-10-10 Codex review."""

    def test_distinct_dois_never_share_a_filename(self) -> None:
        left = FetchTarget(label="a", doi="10.1/a/b")
        right = FetchTarget(label="b", doi="10.1/a_b")
        assert left.filename() != right.filename()

    def test_every_source_refusing_is_unresolved_not_absent(self, tmp_path: Path) -> None:
        target = FetchTarget(label="x", doi=DOI)
        fetcher, _ = make_fetcher(
            tmp_path,
            {
                OPENALEX_PREFIX: status("https://api.openalex.org/works/x", 429),
                DOI_URL: status(DOI_URL, 403),
            },
        )
        outcome = fetcher.fetch(target)
        assert outcome.status is FetchStatus.UNRESOLVED

    def test_definite_negatives_everywhere_is_not_found(self, tmp_path: Path) -> None:
        target = FetchTarget(label="x", doi=DOI)
        fetcher, _ = make_fetcher(
            tmp_path, {OPENALEX_PREFIX: NO_OA, DOI_URL: landing(DOI_URL, None)}
        )
        outcome = fetcher.fetch(target)
        assert outcome.status is FetchStatus.NOT_FOUND
