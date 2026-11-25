"""Citation edges from arXiv source and OpenAlex full text, with fake services."""

from __future__ import annotations

import datetime as dt
import gzip
import io
import tarfile
import time
from typing import Any

from lit_search.core.coverage import CoverageStatus
from lit_search.runners import open_graph

BBL = r"""
\begin{thebibliography}{2}
\bibitem{a} A. Author.
\newblock Negative eigenvalues let linear recurrent networks track state.
\newblock arXiv:2411.12537, 2024.
\bibitem{b} B. Author.
\newblock A paper only findable by its title here.
\newblock In NeurIPS, 2023.
\end{thebibliography}
"""
TEX = r"Prior work solves parity this way \cite{a}. We differ."


def _eprint() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, text in (("main.bbl", BBL), ("main.tex", TEX)):
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return gzip.compress(buffer.getvalue())


def _work(wid: str, title: str, doi: str, date: str = "2024-06-01") -> dict[str, Any]:
    return {
        "id": f"https://openalex.org/{wid}",
        "doi": f"https://doi.org/{doi}",
        "title": title,
        "display_name": title,
        "publication_date": date,
        "authorships": [],
        "locations": [],
    }


class _Response:
    def __init__(self, payload: Any = None, content: bytes = b"") -> None:
        self.payload = payload
        self.content = content

    def json(self) -> Any:
        return self.payload


class FakeOpenAlex:
    def __init__(self) -> None:
        self.filters: list[str] = []

    def _get(self, url: str, params: dict[str, Any]) -> _Response:
        flt = params.get("filter", "")
        self.filters.append(flt)
        if flt.startswith("doi:"):
            return _Response(
                {
                    "results": [
                        _work(
                            "W1",
                            "Negative eigenvalues let linear recurrent networks track state",
                            "10.48550/arxiv.2411.12537",
                        )
                    ]
                }
            )
        if flt.startswith("title.search:"):
            return _Response(
                {"results": [_work("W2", "A paper only findable by its title here", "10.1/b")]}
            )
        if flt.startswith("fulltext.search:"):
            return _Response(
                {
                    "results": [
                        _work("W3", "A later paper that cites the seed", "10.1/c", "2025-03-01"),
                        _work("W9", "The Seed Paper Title Itself", "10.48550/arxiv.2501.00001"),
                    ]
                }
            )
        raise AssertionError(flt)

    def _parse_paper(self, work: dict[str, Any]) -> Any:
        from lit_search.connectors.openalex import OpenAlexConnector

        return OpenAlexConnector._parse_paper(OpenAlexConnector(api_key="k"), work)


class FakeArxiv:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.urls: list[str] = []

    def _get(self, url: str) -> _Response:
        self.urls.append(url)
        return _Response(content=self.data)


SEED = open_graph.OpenSeed(
    "c1", "The Seed Paper Title Itself", "2501.00001", None, dt.date(2025, 1, 2)
)


def test_backward_reads_the_bibliography_and_citing_sentences() -> None:
    arxiv, openalex = FakeArxiv(_eprint()), FakeOpenAlex()
    found, step = open_graph.backward_from_source(
        arxiv,
        openalex,
        SEED,
        deadline=time.time() + 30,  # type: ignore[arg-type]
    )
    assert arxiv.urls == ["https://export.arxiv.org/e-print/2501.00001"]
    titles = {p.title: s for p, s in found}
    assert set(titles) == {
        "Negative eigenvalues let linear recurrent networks track state",
        "A paper only findable by its title here",
    }
    assert titles["Negative eigenvalues let linear recurrent networks track state"] == [
        "Prior work solves parity this way [cite]."
    ]
    assert step.status is CoverageStatus.COMPLETE


def test_backward_reports_a_pdf_only_upload() -> None:
    found, step = open_graph.backward_from_source(
        FakeArxiv(b"%PDF-1.5"),
        FakeOpenAlex(),
        SEED,
        deadline=time.time() + 30,  # type: ignore[arg-type]
    )
    assert found == []
    assert step.status is CoverageStatus.PARTIAL
    assert "PDF only" in (step.detail or "")


def test_backward_skips_a_seed_without_arxiv_id() -> None:
    seed = open_graph.OpenSeed("c2", "Journal paper", None, "10.1/j", None)
    _, step = open_graph.backward_from_source(
        FakeArxiv(b""),
        FakeOpenAlex(),
        seed,
        deadline=time.time() + 30,  # type: ignore[arg-type]
    )
    assert step.status is CoverageStatus.SKIPPED_BY_POLICY


def test_forward_searches_title_and_id_from_the_seed_date_and_drops_the_seed() -> None:
    openalex = FakeOpenAlex()
    papers, step = open_graph.forward_from_fulltext(
        openalex,
        SEED,
        deadline=time.time() + 30,  # type: ignore[arg-type]
    )
    assert [p.title for p in papers] == ["A later paper that cites the seed"]
    assert step.status is CoverageStatus.PARTIAL
    assert all("from_publication_date:2025-01-02" in f for f in openalex.filters)
    assert any('fulltext.search:"2501.00001"' in f for f in openalex.filters)
    assert any('fulltext.search:"the seed paper title itself"' in f for f in openalex.filters)


def test_title_phrases_include_colon_halves_and_apostrophe_free_forms() -> None:
    phrases = open_graph._title_phrases(
        "Surprise! Uniform Information Density Isn't the Whole Story"
    )
    assert "surprise uniform information density isnt the whole story" in phrases
    phrases = open_graph._title_phrases("DeltaProduct: Improving State-Tracking in Linear RNNs")
    assert "improving state tracking in linear rnns" in phrases


def test_run_open_snowball_labels_routes() -> None:
    outcome = open_graph.run_open_snowball(
        [SEED],
        forward=True,
        backward=True,
        openalex=FakeOpenAlex(),  # type: ignore[arg-type]
        arxiv=FakeArxiv(_eprint()),  # type: ignore[arg-type]
    )
    routes = {p.title: outcome.routes[id(p)] for p in outcome.papers}
    assert routes["A later paper that cites the seed"] == ["cites c1 [fulltext]"]
    assert routes["A paper only findable by its title here"] == ["cited by c1 [source]"]


class FakeRelatedOpenAlex(FakeOpenAlex):
    def _get(self, url: str, params: dict[str, Any]) -> _Response:
        self.filters.append(str(params.get("filter") or params.get("select")))
        if params.get("search"):
            since = params["filter"]
            if since.endswith("2026-08-27"):
                return _Response(
                    {"results": [_work("W5", "A very recent similar paper", "10.1/r")]}
                )
            return _Response(
                {
                    "results": [
                        _work("W9", "The Seed Paper Title Itself", "10.48550/arxiv.2501.00001"),
                        _work("W4", "An older similar paper", "10.1/s"),
                        _work("W6", "A paper the agent dropped", "10.1/d"),
                    ]
                }
            )
        if str(params.get("filter", "")).startswith("title.search:"):
            return _Response(
                {
                    "results": [
                        _work("W9", "The Seed Paper Title Itself", "10.48550/arxiv.2501.00001")
                    ]
                }
            )
        if params.get("select") == "related_works":
            return _Response({"related_works": ["https://openalex.org/W7"]})
        if str(params.get("filter", "")).startswith("openalex:"):
            return _Response({"results": [_work("W7", "OpenAlex's own neighbour", "10.1/n")]})
        raise AssertionError(params)


def test_open_related_combines_searches_and_related_works_without_seeds_or_negatives() -> None:
    outcome = open_graph.run_open_related(
        [SEED],
        negative=["A paper the agent dropped"],
        openalex=FakeRelatedOpenAlex(),  # type: ignore[arg-type]
        today=dt.date(2026, 11, 25),
    )
    routes = {p.title: outcome.routes[id(p)] for p in outcome.papers}
    assert routes == {
        "An older similar paper": ["similar#2 to c1"],
        "A very recent similar paper": ["recent#1 to c1"],
        "OpenAlex's own neighbour": ["related to c1 [openalex]"],
    }


class FakeLookupArxiv:
    def __init__(self) -> None:
        self.asked: list[list[str]] = []

    def fetch_papers_batch(self, ids: list[str]) -> Any:
        from types import SimpleNamespace

        from lit_search.core.citation import IdentifierKind, PaperIdentifier
        from lit_search.core.paper import Paper

        self.asked.append(ids)
        paper = Paper(
            title="Folded preprint", abstract="", authors=[], source=None, publication_date=None
        )
        paper.identifiers.append(PaperIdentifier(IdentifierKind.ARXIV, "2502.10297"))
        return SimpleNamespace(papers=(paper,))


class FakeLookupOpenAlex(FakeOpenAlex):
    def _get(self, url: str, params: dict[str, Any]) -> _Response:
        flt = params.get("filter", "")
        if flt.startswith("doi:"):
            return _Response({"results": [_work("W1", "Known by DOI", "10.1/known")]})
        if flt.startswith("title.search:"):
            return _Response({"results": [_work("W2", "An Exact Title Here", "10.1/t")]})
        raise AssertionError(flt)


def test_lookup_uses_openalex_then_arxiv_for_folded_preprints() -> None:
    arxiv = FakeLookupArxiv()
    found, steps = open_graph.lookup_papers(
        FakeLookupOpenAlex(),  # type: ignore[arg-type]
        ["10.1/known", "2502.10297", "An exact title here"],
        arxiv_connector=arxiv,  # type: ignore[arg-type]
    )
    assert {k: v.title for k, v in found.items()} == {
        "10.1/known": "Known by DOI",
        "2502.10297": "Folded preprint",
        "An exact title here": "An Exact Title Here",
    }
    assert arxiv.asked == [["2502.10297"]]
    assert [s.status for s in steps] == [
        CoverageStatus.PARTIAL,
        CoverageStatus.COMPLETE,
        CoverageStatus.COMPLETE,
    ]
