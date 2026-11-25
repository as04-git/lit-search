"""Adversarial tests for conservative record deduplication.

Only explicit record identity may collapse members, and only an explicit
``preprint_doi`` assertion may establish a version link.  Title and semantic
similarity remain unresolved evidence.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable, Sequence
from dataclasses import replace
from itertools import permutations
from pathlib import Path

import pytest

from lit_search.core.citation import IdentifierKind, PaperIdentifier
from lit_search.core.paper import Paper
from lit_search.core.source import Source, SourceType
from lit_search.exceptions import InvalidParameterError
from lit_search.index.embedding import DeterministicEmbedder, EmbeddingIdentity
from lit_search.store.dedup import (
    MatchMethod,
    deduplicate,
    normalise_title,
    record_deduplication,
)
from lit_search.store.operations import searched
from lit_search.store.store import HandleStore

ARXIV_DOI = "10.48550/arXiv.2604.11556"
PUBLISHED_DOI = "10.1145/3712345.3712999"
TITLE = "FM-Agent: Scaling Formal Methods via LLM-Based Hoare-Style Reasoning"


@pytest.fixture
def paper() -> Callable[..., Paper]:
    """Return a factory building papers with controllable identity."""

    def factory(
        title: str = TITLE,
        doi: str | None = None,
        abstract: str = "",
        year: int | None = 2026,
        preprint_doi: str | None = None,
        identifiers: list[PaperIdentifier] | None = None,
    ) -> Paper:
        return Paper(
            title=title,
            abstract=abstract,
            authors=[],
            source=Source(title="arXiv", source_type=SourceType.REPOSITORY),
            publication_date=None if year is None else datetime.date(year, 4, 13),
            doi=doi,
            preprint_doi=preprint_doi,
            found_in={"arxiv"},
            identifiers=identifiers,
        )

    return factory


class TestExactCollapse:
    """A duplicate is collapsed only when there is nothing to preserve."""

    def test_identical_dois_collapse(self, paper: Callable[..., Paper]) -> None:
        """The same record from two sources is one record."""
        plan = deduplicate([paper(doi=ARXIV_DOI), paper(doi=ARXIV_DOI, abstract="longer")])
        assert len(plan.members) == 1
        assert plan.collapsed_count == 1
        assert plan.duplicate_groups[0].method is MatchMethod.EXACT_RECORD

    def test_same_title_and_no_identifier_never_collapses(
        self, paper: Callable[..., Paper]
    ) -> None:
        """A generic title is not record identity, even in the same year."""
        plan = deduplicate([paper(title="Workshop Note"), paper(title="Workshop  note.")])
        assert len(plan.members) == 2
        assert plan.collapsed_count == 0
        assert plan.version_links == ()

    def test_same_provider_record_identity_collapses(self, paper: Callable[..., Paper]) -> None:
        """A provider-native ID is explicit record identity even without a DOI."""
        provider_id = PaperIdentifier(IdentifierKind.OPENALEX, "W123")
        plan = deduplicate(
            [
                paper(title="Sparse record", identifiers=[provider_id]),
                paper(title="Enriched record", identifiers=[provider_id]),
            ]
        )
        assert len(plan.members) == 1
        assert plan.collapsed_count == 1
        assert plan.duplicate_groups[0].method is MatchMethod.EXACT_RECORD
        assert "openalex:w123" in plan.duplicate_groups[0].kept_key
        assert "sparse record" not in plan.duplicate_groups[0].kept_key

    def test_provider_alias_cannot_override_conflicting_dois(
        self, paper: Callable[..., Paper]
    ) -> None:
        """An aggregator collision is preserved when publication identities conflict."""
        provider_id = PaperIdentifier(IdentifierKind.SEMANTIC_SCHOLAR, "shared")
        plan = deduplicate(
            [
                paper(doi="10.0/one", identifiers=[provider_id]),
                paper(doi="10.0/two", identifiers=[provider_id]),
            ]
        )
        assert len(plan.members) == 2
        assert plan.collapsed_count == 0
        assert plan.version_links == ()

    def test_unversioned_alias_cannot_bridge_conflicting_versions(
        self, paper: Callable[..., Paper]
    ) -> None:
        """Exact collapse is invariant to an unknown provider record arriving first."""
        provider_id = PaperIdentifier(IdentifierKind.SEMANTIC_SCHOLAR, "shared")
        plan = deduplicate(
            [
                paper(title="Unversioned", identifiers=[provider_id]),
                paper(doi="10.0/one", identifiers=[provider_id]),
                paper(doi="10.0/two", identifiers=[provider_id]),
            ]
        )
        assert len(plan.members) == 3
        assert plan.collapsed_count == 0
        assert plan.version_links == ()

    def test_published_record_cannot_bridge_doi_and_arxiv_versions(
        self, paper: Callable[..., Paper]
    ) -> None:
        """Dedup order cannot collapse a preprint through a publication alias set."""
        for ordering in permutations(("published", "doi", "arxiv")):
            published = paper(doi="10.0/published")
            published.identifiers.append(PaperIdentifier(IdentifierKind.ARXIV, "2601.00001"))
            records = {
                "published": published,
                "doi": paper(doi="10.0/published"),
                "arxiv": paper(
                    title="Preprint",
                    identifiers=[PaperIdentifier(IdentifierKind.ARXIV, "2601.00001")],
                ),
            }
            plan = deduplicate([records[name] for name in ordering])
            assert len(plan.members) == 2
            assert plan.collapsed_count == 1

    def test_incompatible_years_block_a_title_match(self, paper: Callable[..., Paper]) -> None:
        """Two different papers can share a title; a decade apart they are not one work."""
        plan = deduplicate(
            [paper(title="A Common Title", year=2005), paper(title="A Common Title", year=2026)]
        )
        assert len(plan.members) == 2
        assert plan.collapsed_count == 0
        assert plan.version_links == ()

    def test_a_missing_year_does_not_by_itself_force_a_match(
        self, paper: Callable[..., Paper]
    ) -> None:
        """The fork base merges whenever either year is None; that rule is not used here."""
        plan = deduplicate(
            [paper(title="Same Title", year=None), paper(title="Different Title", year=2026)]
        )
        assert len(plan.members) == 2


class TestVersionLinking:
    """Two identities means two records, plus an explicit relationship."""

    def test_a_declared_preprint_doi_links_rather_than_merges(
        self, paper: Callable[..., Paper]
    ) -> None:
        """Merging would keep one DOI and drop the other."""
        plan = deduplicate(
            [
                paper(doi=ARXIV_DOI, abstract="the long preprint abstract"),
                paper(doi=PUBLISHED_DOI, abstract="short", preprint_doi=ARXIV_DOI),
            ]
        )
        assert len(plan.members) == 2
        assert plan.collapsed_count == 0
        assert len(plan.version_links) == 1
        assert MatchMethod.PREPRINT_LINK in plan.version_links[0].methods

    def test_read_and_cite_targets_are_separate(self, paper: Callable[..., Paper]) -> None:
        """A more detailed preprint is the one to read; the journal record is the one to cite."""
        plan = deduplicate(
            [
                paper(doi=ARXIV_DOI, abstract="a much longer and more detailed abstract"),
                paper(doi=PUBLISHED_DOI, abstract="short", preprint_doi=ARXIV_DOI),
            ]
        )
        link = plan.version_links[0]
        assert link.read_target == ARXIV_DOI.lower()
        assert link.cite_target == PUBLISHED_DOI.lower()
        assert link.divergence is not None
        assert plan.divergent_links == (link,)

    def test_no_divergence_when_the_same_record_wins_both_roles(
        self, paper: Callable[..., Paper]
    ) -> None:
        """A divergence is reported only when there really is one."""
        plan = deduplicate(
            [
                paper(doi=ARXIV_DOI, abstract="short"),
                paper(
                    doi=PUBLISHED_DOI,
                    abstract="the longer published abstract",
                    preprint_doi=ARXIV_DOI,
                ),
            ]
        )
        link = plan.version_links[0]
        assert link.read_target == link.cite_target == PUBLISHED_DOI.lower()
        assert link.divergence is None

    def test_same_title_two_dois_remains_unresolved(self, paper: Callable[..., Paper]) -> None:
        """Two works may share a title; titles do not prove a version relationship."""
        plan = deduplicate([paper(doi=ARXIV_DOI), paper(doi=PUBLISHED_DOI)])
        assert len(plan.members) == 2
        assert plan.version_links == ()

    def test_title_does_not_add_evidence_to_an_explicit_link(
        self, paper: Callable[..., Paper]
    ) -> None:
        """The link records only the metadata assertion that proves it."""
        plan = deduplicate([paper(doi=ARXIV_DOI), paper(doi=PUBLISHED_DOI, preprint_doi=ARXIV_DOI)])
        assert len(plan.version_links) == 1
        assert plan.version_links[0].methods == (MatchMethod.PREPRINT_LINK,)

    def test_a_record_with_a_doi_and_one_without_remain_unresolved(
        self, paper: Callable[..., Paper]
    ) -> None:
        """A shared title does not reconcile an identified and unidentified record."""
        plan = deduplicate([paper(doi=ARXIV_DOI), paper(doi=None)])
        assert len(plan.members) == 2
        assert plan.version_links == ()


class TestSemanticStage:
    """Stage two runs, refuses, or reports that it did not run."""

    def test_it_says_so_when_no_model_was_supplied(self, paper: Callable[..., Paper]) -> None:
        """ "Nobody looked" is not "nothing found"."""
        plan = deduplicate([paper(title="One", doi="10.0/1"), paper(title="Two", doi="10.0/2")])
        assert plan.semantic_stage.ran is False
        assert "no embedding model" in plan.semantic_stage.reason
        assert "unexamined, not clean" in plan.semantic_stage.describe()
        assert "SKIPPED" in plan.strategy()

    def test_a_non_semantic_model_is_refused(self, paper: Callable[..., Paper]) -> None:
        """Meaningless cosine scores must not be emitted as if they were measurements."""
        plan = deduplicate(
            [paper(title="One", doi="10.0/1"), paper(title="Two", doi="10.0/2")],
            embedder=DeterministicEmbedder(16),
        )
        assert plan.semantic_stage.ran is False
        assert "is_semantic=False" in plan.semantic_stage.reason

    def test_semantic_false_positive_is_retained_as_unresolved_candidate(
        self, paper: Callable[..., Paper]
    ) -> None:
        """Even a score of one cannot establish bibliographic identity."""
        plan = deduplicate(
            [
                paper(title="Deriving Specs From Call Sites", doi="10.0/preprint"),
                paper(title="Specification Derivation From Callers", doi="10.0/published"),
            ],
            embedder=_AlwaysSimilar(),
            semantic_threshold=0.5,
        )
        assert plan.semantic_stage.ran is True
        assert plan.semantic_stage.pairs_matched == 1
        assert len(plan.members) == 2
        assert plan.collapsed_count == 0
        assert plan.version_links == ()
        assert "unresolved candidate" in plan.semantic_stage.describe()

    def test_semantic_match_without_identifiers_also_never_collapses(
        self, paper: Callable[..., Paper]
    ) -> None:
        """Missing identifiers reduce certainty; they do not license destruction."""
        plan = deduplicate(
            [paper(title="First Note", doi=None), paper(title="Second Note", doi=None)],
            embedder=_AlwaysSimilar(),
            semantic_threshold=0.5,
        )
        assert len(plan.members) == 2
        assert plan.collapsed_count == 0
        assert plan.version_links == ()
        assert plan.semantic_stage.pairs_matched == 1

    def test_the_residue_drops_collapsed_records_but_keeps_the_survivor(
        self, paper: Callable[..., Paper]
    ) -> None:
        """A collapsed record leaves the residue; the record it collapsed into does not.

        The survivor stays eligible as an unresolved candidate against a third
        record; a similarity score still cannot establish a version link.
        """
        plan = deduplicate(
            [paper(doi=ARXIV_DOI), paper(doi=ARXIV_DOI), paper(title="Elsewhere", doi="10.0/x")],
            embedder=_AlwaysSimilar(),
        )
        assert plan.semantic_stage.residue_size == 2
        assert len(plan.members) == 2
        assert plan.version_links == ()

    def test_similarity_chain_never_becomes_a_transitive_version_group(
        self, paper: Callable[..., Paper]
    ) -> None:
        """Three pairwise-high scores remain three records and zero proven links."""
        plan = deduplicate(
            [
                paper(title="Alpha", doi="10.0/a"),
                paper(title="Beta", doi="10.0/b"),
                paper(title="Gamma", doi="10.0/c"),
            ],
            embedder=_AlwaysSimilar(),
            semantic_threshold=0.5,
        )
        assert plan.semantic_stage.pairs_matched == 3
        assert len(plan.members) == 3
        assert plan.version_links == ()


class TestRecording:
    """The plan becomes a handle whose lineage carries the version links."""

    def test_links_survive_a_store_round_trip(
        self, tmp_path: Path, paper: Callable[..., Paper]
    ) -> None:
        """They live in operation parameters, which schema version 1 already allows."""
        store = HandleStore(tmp_path / "cache")
        papers = [
            paper(doi=ARXIV_DOI, abstract="long preprint abstract"),
            paper(doi=PUBLISHED_DOI, abstract="short", preprint_doi=ARXIV_DOI),
        ]
        base_id = store.put(searched(papers, query="[fm]", databases=["arxiv"]))
        plan = deduplicate(papers)
        handle_id = store.put(record_deduplication(store, base_id, plan, label="deduped"))

        reloaded = store.get(handle_id)
        links = reloaded.operation.parameters["version_links"]
        assert len(links) == 1
        assert links[0]["read_target"] == ARXIV_DOI.lower()
        assert links[0]["cite_target"] == PUBLISHED_DOI.lower()
        assert reloaded.operation.parameters["submitted"] == 2
        assert reloaded.operation.parameters["survived"] == 2

    def test_the_semantic_stage_report_is_stored(
        self, tmp_path: Path, paper: Callable[..., Paper]
    ) -> None:
        """A handle whose residue was never examined must say so a week later."""
        store = HandleStore(tmp_path / "cache")
        papers = [paper(title="One", doi="10.0/1"), paper(title="Two", doi="10.0/2")]
        base_id = store.put(searched(papers, query="[q]", databases=["arxiv"]))
        handle_id = store.put(record_deduplication(store, base_id, deduplicate(papers)))
        stage = store.get(handle_id).operation.parameters["semantic_stage"]
        assert stage["ran"] is False
        assert stage["residue_size"] == 2

    def test_a_plan_for_another_set_is_refused(
        self, tmp_path: Path, paper: Callable[..., Paper]
    ) -> None:
        """Recording a plan against the wrong base would write a false lineage."""
        store = HandleStore(tmp_path / "cache")
        base_papers = [paper(title="Base A", doi="10.0/a"), paper(title="Base B", doi="10.0/b")]
        base_id = store.put(searched(base_papers, query="[q]", databases=["arxiv"]))
        other_plan = deduplicate(
            [paper(title="Other A", doi="10.0/c"), paper(title="Other B", doi="10.0/d")]
        )
        with pytest.raises(InvalidParameterError, match="input snapshot does not match"):
            record_deduplication(store, base_id, other_plan)

    def test_a_tampered_survivor_list_is_refused(
        self, tmp_path: Path, paper: Callable[..., Paper]
    ) -> None:
        """An input commitment cannot authorize members that were never submitted."""
        store = HandleStore(tmp_path / "cache")
        papers = [paper(title="Base A"), paper(title="Base B")]
        base_id = store.put(searched(papers, query="[q]", databases=["arxiv"]))
        plan = deduplicate(papers)
        tampered = replace(plan, members=(paper(title="Invented"), paper(title="Base B")))
        with pytest.raises(InvalidParameterError, match="one-to-one"):
            record_deduplication(store, base_id, tampered)

    def test_coverage_is_propagated_from_the_base(
        self, tmp_path: Path, paper: Callable[..., Paper]
    ) -> None:
        """A derived set must never report better coverage than its operand."""
        from lit_search.core.coverage import CoverageLedger, SourceCoverage

        store = HandleStore(tmp_path / "cache")
        ledger = CoverageLedger.from_sources(
            [SourceCoverage.partial("arxiv", 2, detail="page 3 failed")]
        )
        papers = [paper(doi="10.0/1"), paper(doi="10.0/2")]
        base_id = store.put(
            searched(papers, query="[q]", databases=["arxiv"], coverage=ledger.to_dict())
        )
        record = record_deduplication(store, base_id, deduplicate(papers))
        assert CoverageLedger.from_dict(record.coverage).status.value == "partial"


class TestTitleNormalisation:
    """Comparison keys, not display strings."""

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("FM-Agent: Reasoning", "fm agent  reasoning"),
            ("A Study (Extended)", "a study extended"),
        ],
    )
    def test_equivalent_titles_normalise_alike(self, left: str, right: str) -> None:
        """Punctuation and case must not create false distinctions."""
        assert normalise_title(left) == normalise_title(right)


class _AlwaysSimilar:
    """A semantic model whose every pair of documents is identical.

    Used to reach the semantic branch deterministically without downloading a
    model.  It declares ``is_semantic=True`` because the point of the test is
    the branch that runs, not the refusal.
    """

    @property
    def identity(self) -> EmbeddingIdentity:
        """Return an identity claiming to be semantic.

        Returns
        -------
        EmbeddingIdentity
            Identity.
        """
        return EmbeddingIdentity(
            model_id="test/always-similar",
            dimension=2,
            normalization="l2",
            prompt_convention="none",
            is_semantic=True,
        )

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one identical vector per text.

        Parameters
        ----------
        texts : list[str]
            Ignored.

        Returns
        -------
        list[list[float]]
            Identical unit vectors.
        """
        return [[1.0, 0.0] for _ in texts]

    def encode_query(self, text: str) -> list[float]:
        """Return the same vector for any query.

        Parameters
        ----------
        text : str
            Ignored.

        Returns
        -------
        list[float]
            Unit vector.
        """
        return [1.0, 0.0]
