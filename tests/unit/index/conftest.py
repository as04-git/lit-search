"""Shared fixtures for the paper-index tests."""

from __future__ import annotations

import datetime
from collections.abc import Callable
from pathlib import Path

import pytest

from lit_search.core.author import Author
from lit_search.core.paper import Paper
from lit_search.core.source import Source, SourceType
from lit_search.index.embedding import DeterministicEmbedder
from lit_search.index.paper_index import PaperIndex


@pytest.fixture
def make_paper() -> Callable[..., Paper]:
    """Return a factory building papers with controllable text and identity."""

    def factory(
        title: str,
        doi: str | None = None,
        abstract: str = "",
        authors: tuple[str, ...] = (),
        keywords: tuple[str, ...] = (),
        year: int = 2026,
        preprint_doi: str | None = None,
        venue: str | None = "arXiv",
    ) -> Paper:
        source = (
            Source(title=venue, source_type=SourceType.REPOSITORY) if venue is not None else None
        )
        return Paper(
            title=title,
            abstract=abstract,
            authors=[Author(name=name) for name in authors],
            source=source,
            publication_date=datetime.date(year, 4, 13),
            doi=doi,
            preprint_doi=preprint_doi,
            keywords=set(keywords),
            found_in={"arxiv"},
        )

    return factory


@pytest.fixture
def embedder() -> DeterministicEmbedder:
    """Return the reproducible non-semantic embedder used across index tests."""
    return DeterministicEmbedder(dimension=32)


@pytest.fixture
def index(tmp_path: Path, embedder: DeterministicEmbedder) -> PaperIndex:
    """Return an empty index rooted in a temporary directory."""
    return PaperIndex(tmp_path / "index", embedder)


@pytest.fixture
def corpus(make_paper: Callable[..., Paper]) -> list[Paper]:
    """Return four papers spanning two research vocabularies plus one control."""
    return [
        make_paper(
            "FM-Agent: Scaling Formal Methods via LLM-Based Hoare-Style Reasoning",
            "10.48550/arXiv.2604.11556",
            abstract="Derive a function specification from its callers.",
            authors=("Wei Li",),
            keywords=("formal methods",),
        ),
        make_paper(
            "BMC-Agent: Agentic Model Checking",
            "10.48550/arXiv.2605.21434",
            abstract="Dual-source generation contrasts caller intent with implementation.",
            keywords=("model checking",),
        ),
        make_paper(
            "Socrates: Active Learning for Program Semantics",
            "10.48550/arXiv.2604.08792",
            abstract="Version spaces and distinguishing inputs for specification inference.",
            keywords=("program synthesis",),
        ),
        make_paper(
            "Preguss: Specification Synthesis",
            "10.0/preguss",
            keywords=("program synthesis",),
        ),
    ]
