"""Shared fixtures for handle-store tests."""

from __future__ import annotations

import datetime
from collections.abc import Callable
from pathlib import Path

import pytest

from lit_search.core.paper import Paper
from lit_search.store.lineage import ExternalReference, Operation, OperationName
from lit_search.store.order import RankingProvenance, RankMode
from lit_search.store.record import HandleRecord
from lit_search.store.store import HandleStore

#: Fixed timestamp so that content addresses are reproducible across runs.
FIXED_TIME = datetime.datetime(2026, 8, 30, 12, 0, 0, tzinfo=datetime.UTC)


@pytest.fixture
def store(tmp_path: Path) -> HandleStore:
    """Return a handle store rooted in a temporary directory."""
    return HandleStore(tmp_path / "cache")


@pytest.fixture
def make_paper() -> Callable[..., Paper]:
    """Return a factory building papers with distinct identities."""

    def factory(title: str, doi: str | None = None, abstract: str = "An abstract.") -> Paper:
        return Paper(
            title=title,
            abstract=abstract,
            authors=[],
            source=None,
            publication_date=None,
            doi=doi,
        )

    return factory


@pytest.fixture
def papers(make_paper: Callable[..., Paper]) -> list[Paper]:
    """Return four papers with distinct DOIs."""
    return [make_paper(f"Paper {index}", f"10.1000/{index}") for index in range(4)]


@pytest.fixture
def make_record() -> Callable[..., HandleRecord]:
    """Return a factory building source handles at a fixed timestamp."""

    def factory(
        members: list[Paper],
        *,
        query: str = "[a] AND [b]",
        label: str | None = None,
        coverage: object = None,
    ) -> HandleRecord:
        return HandleRecord(
            operation=Operation(
                name=OperationName.SEARCH,
                parameters={"query": query, "databases": ["arxiv"]},
            ),
            members=tuple(members),
            created_at=FIXED_TIME,
            created_by="test",
            label=label,
            coverage=coverage,
        )

    return factory


@pytest.fixture
def query_ranking() -> RankingProvenance:
    """Return a pool-independent query-mode ranking provenance."""
    return RankingProvenance(
        mode=RankMode.QUERY,
        query_text="how do agents verify generated programs",
        model_id="intfloat/e5-large-v2",
        model_revision="deadbeef",
        prompt_convention="e5:query:/passage:",
        scoring_convention="cosine",
        normalization="l2",
        text_policy="title+abstract/v1",
        pool_independent=True,
    )


@pytest.fixture
def zotero_reference() -> ExternalReference:
    """Return an external reference standing for a Zotero library snapshot."""
    return ExternalReference(
        kind="zotero_library",
        identity={"library_id": 7067953, "item_count": 412},
        captured_at=FIXED_TIME,
    )
