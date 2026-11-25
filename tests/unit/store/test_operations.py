"""Tests for set algebra, order transitions, and coverage propagation."""

from __future__ import annotations

import dataclasses
import datetime
from itertools import permutations

import pytest

from lit_search.core.citation import IdentifierKind, PaperIdentifier
from lit_search.core.coverage import CoverageLedger, SourceCoverage
from lit_search.core.paper import Paper
from lit_search.core.search_result import SearchResult
from lit_search.exceptions import CoveragePropagationError, InvalidParameterError
from lit_search.store import operations as ops
from lit_search.store.lineage import OperandKind, OperationName
from lit_search.store.order import OrderRecord, OrderState
from lit_search.store.record import HandleRecord

from .conftest import FIXED_TIME


class PassThroughPropagator:
    """A propagator that keeps the first operand's ledger, for testing only.

    First-operand-wins is the *wrong* rule for a difference, which is exactly
    why the protocol hands over :class:`~lit_search.store.propagation.CoverageOperand`
    values carrying a role rather than a bare list of payloads.  It is kept
    here only to prove the seam still accepts a caller-supplied propagator.
    """

    def combine(self, operation, operands):  # type: ignore[no-untyped-def]
        """Return the first operand's ledger."""
        return operands[0].coverage


def _ledger(*, complete=None, partial=None):
    """Build a serialized coverage ledger payload for a handle.

    Parameters
    ----------
    complete : tuple[str, int] | None
        ``(source, count)`` retrieved completely.
    partial : tuple[str, int] | None
        ``(source, count)`` retrieved partially.

    Returns
    -------
    dict
        Payload in the form a real search handle carries.
    """
    entries = []
    if complete is not None:
        entries.append(
            SourceCoverage.complete(complete[0], complete[1], reported_total=complete[1])
        )
    if partial is not None:
        entries.append(SourceCoverage.partial(partial[0], partial[1], detail="stubbed as partial"))
    return CoverageLedger.from_sources(entries).to_dict()


def _ranked(store, base_id, ranking, order):
    """Store a ranked handle derived from *base_id* in the given member order."""
    scores = [1.0 - index * 0.1 for index in range(len(order))]
    return store.put(ops.reranked(store, base_id, members=order, scores=scores, ranking=ranking))


class TestSourceHandles:
    """Handles built from a search, a snowball, or an import."""

    def test_search_handle_is_unordered(self, papers) -> None:
        """A Boolean search returns papers in no meaningful order."""
        record = ops.searched(papers, query="[a]", databases=["arxiv"])
        assert record.order.state is OrderState.UNORDERED
        assert record.operation.name is OperationName.SEARCH
        assert record.operation.parameters["query"] == "[a]"

    def test_from_search_result_keeps_failed_databases_as_a_parameter(self, papers) -> None:
        """The base's thrown-error list is carried, and is not a coverage ledger."""
        result = SearchResult(
            query="[a]",
            databases=["arxiv", "ieee"],
            papers=list(papers),
            failed_databases=["ieee"],
        )
        record = ops.from_search_result(result)
        assert record.operation.parameters["recorded_failed_databases"] == ["ieee"]

    def test_from_search_result_carries_the_coverage_ledger(self, papers) -> None:
        """A ledger the search recorded reaches the handle rather than vanishing.

        A search that recorded nothing yields a ledger reporting ``unknown``,
        which is honest.  Dropping the ledger here would have let the handle
        read as complete instead.
        """
        result = SearchResult(query="[a]", databases=["arxiv"], papers=list(papers))
        record = ops.from_search_result(result)
        assert record.coverage is not None
        assert record.coverage["status"] == "unknown"

    def test_an_explicit_ledger_overrides_the_result(self, papers) -> None:
        """A caller that already combined coverage can supply the result."""
        result = SearchResult(query="[a]", databases=["arxiv"], papers=list(papers))
        record = ops.from_search_result(result, coverage={"status": "capped"})
        assert record.coverage == {"status": "capped"}

    def test_import_records_its_external_operand(self, papers, zotero_reference) -> None:
        """A mutable source becomes an explicit, timestamped operand."""
        record = ops.imported(papers, source=zotero_reference)
        assert record.operation.name is OperationName.IMPORT
        (operand,) = record.operation.operands
        assert operand.kind is OperandKind.EXTERNAL
        assert operand.external is not None
        assert operand.external.identity["library_id"] == 7067953
        assert operand.external.captured_at == FIXED_TIME


class TestRerank:
    """A re-rank reorders a set; it does not select from one."""

    def test_rerank_produces_a_ranked_handle(self, store, make_record, papers, query_ranking):
        """The order state, provenance, and scores are all recorded."""
        base = store.put(make_record(papers))
        reordered = list(reversed(papers))
        record = ops.reranked(
            store, base, members=reordered, scores=[0.9, 0.8, 0.7, 0.6], ranking=query_ranking
        )
        assert record.order.state is OrderState.RANKED
        assert record.order.ranking == query_ranking
        assert record.order.scores == (0.9, 0.8, 0.7, 0.6)
        assert [p.doi for p in record.members] == [p.doi for p in reordered]

    def test_rerank_records_the_model_and_prompt_convention(
        self, store, make_record, papers, query_ranking
    ) -> None:
        """Every re-rank output states which model and prefixes produced it."""
        base = store.put(make_record(papers))
        record = ops.reranked(store, base, members=papers, scores=[1.0] * 4, ranking=query_ranking)
        recorded = record.operation.parameters["ranking"]
        assert recorded["model_id"] == "intfloat/e5-large-v2"
        assert recorded["prompt_convention"] == "e5:query:/passage:"

    def test_rerank_refuses_to_drop_members(self, store, make_record, papers, query_ranking):
        """A truncating re-ranker would hide the truncation inside an ordering."""
        base = store.put(make_record(papers))
        with pytest.raises(InvalidParameterError, match="does not choose a subset"):
            ops.reranked(store, base, members=papers[:2], scores=[0.9, 0.8], ranking=query_ranking)

    def test_rerank_refuses_a_different_set(
        self, store, make_record, papers, make_paper, query_ranking
    ) -> None:
        """A re-rank must be a permutation of its input."""
        base = store.put(make_record(papers))
        swapped = [*papers[:3], make_paper("Intruder", "10.1000/x")]
        with pytest.raises(InvalidParameterError, match="different set of papers"):
            ops.reranked(store, base, members=swapped, scores=[1.0] * 4, ranking=query_ranking)

    def test_rerank_accepts_separately_deserialized_same_doi_records(
        self, store, make_record, papers, query_ranking
    ) -> None:
        """A handle round-trip does not turn an exact DOI permutation into a new set."""
        base = store.put(make_record(papers[:2]))
        clones = [Paper.from_dict(paper.to_dict()) for paper in reversed(papers[:2])]

        record = ops.reranked(
            store,
            base,
            members=clones,
            scores=[0.9, 0.8],
            ranking=query_ranking,
        )

        assert [paper.doi for paper in record.members] == ["10.1000/1", "10.1000/0"]

    def test_rerank_identity_matching_preserves_member_multiplicity(
        self, store, make_record, papers, query_ranking
    ) -> None:
        """Two copies of one identity cannot impersonate two distinct base members."""
        base = store.put(make_record(papers[:2]))
        repeated = [Paper.from_dict(papers[0].to_dict()) for _ in range(2)]

        with pytest.raises(InvalidParameterError, match="different set of papers"):
            ops.reranked(
                store,
                base,
                members=repeated,
                scores=[1.0, 0.5],
                ranking=query_ranking,
            )

    def test_rerank_accepts_identifier_free_exact_snapshot_round_trip(
        self, store, make_record, make_paper, query_ranking
    ) -> None:
        """The same stored snapshot remains rerankable when it has no identifier."""
        original = make_paper("A deliberately generic title")
        exact_snapshot = Paper.from_dict(original.to_dict())
        base = store.put(make_record([original]))

        record = ops.reranked(
            store,
            base,
            members=[exact_snapshot],
            scores=[1.0],
            ranking=query_ranking,
        )

        assert record.members[0].to_dict() == original.to_dict()

    def test_rerank_rejects_identifier_free_title_only_match(
        self, store, make_record, make_paper, query_ranking
    ) -> None:
        """Shared title alone cannot substitute a different unknown record."""
        original = make_paper("A deliberately generic title")
        title_only_match = make_paper(
            "A deliberately generic title", abstract="A different source record."
        )
        base = store.put(make_record([original]))

        with pytest.raises(InvalidParameterError, match="different set of papers"):
            ops.reranked(
                store,
                base,
                members=[title_only_match],
                scores=[1.0],
                ranking=query_ranking,
            )

    def test_rerank_refuses_mismatched_score_count(
        self, store, make_record, papers, query_ranking
    ) -> None:
        """One score per member."""
        base = store.put(make_record(papers))
        with pytest.raises(InvalidParameterError, match="score"):
            ops.reranked(store, base, members=papers, scores=[1.0], ranking=query_ranking)

    def test_top_k_is_rerank_then_filter(self, store, make_record, papers, query_ranking):
        """Truncation appears in the lineage as its own operation."""
        base = store.put(make_record(papers))
        ranked = _ranked(store, base, query_ranking, list(papers))
        top_two_dois = {paper.doi for paper in store.get(ranked).members[:2]}
        top = store.put(
            ops.filtered(
                store,
                ranked,
                lambda paper: paper.doi in top_two_dois,
                description="top 2 by rerank score",
            )
        )
        record = store.get(top)
        assert len(record.members) == 2
        assert record.order.state is OrderState.RANKED
        assert record.operation.name is OperationName.FILTER
        assert record.operation.parameters["description"] == "top 2 by rerank score"


class TestUnion:
    """Union combines membership, and keeps an order only when it may."""

    def test_union_of_unordered_sets_is_unordered(self, store, make_record, papers) -> None:
        """Nothing to preserve, nothing to invalidate."""
        first = store.put(make_record(papers[:2]))
        second = store.put(make_record(papers[2:]))
        record = ops.union(store, [first, second])
        assert record.order.state is OrderState.UNORDERED
        assert len(record.members) == 4

    def test_union_deduplicates_by_identity_first_seen_wins(
        self, store, make_record, papers, make_paper
    ) -> None:
        """The earlier operand's record survives and is not merged into."""
        rich = make_paper("Paper 0 with a much longer title", "10.1000/0", abstract="Detailed.")
        first = store.put(make_record([papers[0]]))
        second = store.put(make_record([rich, papers[1]]))
        record = ops.union(store, [first, second])
        assert len(record.members) == 2
        assert record.members[0].title == "Paper 0"
        assert record.members[0].abstract == "An abstract."

    def test_union_preserves_distinct_identifier_free_title_matches(
        self, store, make_record, make_paper
    ) -> None:
        """The same title is not enough to collapse two unknown source records."""
        left = make_paper("Attention Mechanisms for Time Series")
        right = make_paper("Attention Mechanisms for Time Series", abstract="Other record.")
        first = store.put(make_record([left]))
        second = store.put(make_record([right]))

        record = ops.union(store, [first, second])

        assert len(record.members) == 2
        assert [paper.abstract for paper in record.members] == ["An abstract.", "Other record."]

    def test_union_matches_native_provider_ids_after_deserialization(
        self, store, make_record, make_paper
    ) -> None:
        """A provider-native ID is sufficient exact identity without a DOI."""
        left = make_paper("Sparse source record")
        left.identifiers = [PaperIdentifier(IdentifierKind.OPENALEX, "W123")]
        right = Paper.from_dict(left.to_dict())
        right.title = "Richer source record"
        first = store.put(make_record([left]))
        second = store.put(make_record([right]))

        record = ops.union(store, [first, second])

        assert len(record.members) == 1
        assert record.members[0].title == "Sparse source record"

    def test_union_preserves_conflicting_dois_despite_shared_provider_alias(
        self, store, make_record, make_paper
    ) -> None:
        """A provider alias conflict cannot authorize collapsing two DOI versions."""
        shared = PaperIdentifier(IdentifierKind.SEMANTIC_SCHOLAR, "shared-record")
        left = make_paper("Version", "10.1000/left")
        right = make_paper("Version", "10.1000/right")
        left.identifiers = [shared]
        right.identifiers = [shared]
        first = store.put(make_record([left]))
        second = store.put(make_record([right]))

        record = ops.union(store, [first, second])

        assert [paper.doi for paper in record.members] == ["10.1000/left", "10.1000/right"]

    def test_union_unversioned_alias_cannot_bridge_conflicting_versions(
        self, store, make_record, make_paper
    ) -> None:
        """First-seen unknown metadata cannot erase two later DOI versions."""
        shared = PaperIdentifier(IdentifierKind.SEMANTIC_SCHOLAR, "shared-record")
        unknown = make_paper("Unversioned")
        first = make_paper("First version", "10.1000/first")
        second = make_paper("Second version", "10.1000/second")
        for item in (unknown, first, second):
            item.identifiers = [shared]
        handles = [store.put(make_record([item])) for item in (unknown, first, second)]

        record = ops.union(store, handles)

        assert len(record.members) == 3
        assert [paper.doi for paper in record.members] == [None, "10.1000/first", "10.1000/second"]

    def test_union_published_record_cannot_bridge_doi_and_arxiv_versions(
        self, store, make_record, make_paper
    ) -> None:
        """Extra arXiv metadata on a publication is link evidence, not equality."""
        published = make_paper("Published", "10.1000/published")
        published.identifiers.append(PaperIdentifier(IdentifierKind.ARXIV, "2601.00001"))
        doi_only = make_paper("Published DOI", "10.1000/published")
        arxiv_only = make_paper("Preprint")
        arxiv_only.identifiers = [PaperIdentifier(IdentifierKind.ARXIV, "2601.00001")]
        handles = {
            name: store.put(make_record([paper]))
            for name, paper in (
                ("published", published),
                ("doi", doi_only),
                ("arxiv", arxiv_only),
            )
        }

        for ordering in permutations(handles):
            record = ops.union(store, [handles[name] for name in ordering])
            assert len(record.members) == 2
            assert {paper.doi for paper in record.members} == {"10.1000/published", None}

    def test_union_of_ranked_and_unordered_is_invalidated(
        self, store, make_record, papers, query_ranking
    ) -> None:
        """The unranked members were never scored, so no order can be claimed."""
        base = store.put(make_record(papers[:2]))
        ranked = _ranked(store, base, query_ranking, list(papers[:2]))
        plain = store.put(make_record(papers[2:]))
        record = ops.union(store, [ranked, plain])
        assert record.order.state is OrderState.INVALIDATED
        assert record.order.invalidated_by == "union"
        assert record.order.previous_ranking == query_ranking
        assert record.order.scores is None

    def test_union_of_compatible_rankings_merges_by_score(
        self, store, make_record, papers, query_ranking
    ) -> None:
        """Pool-independent scores under an identical claim are comparable."""
        left_base = store.put(make_record(papers[:2]))
        right_base = store.put(make_record(papers[2:]))
        left = store.put(
            ops.reranked(
                store, left_base, members=papers[:2], scores=[0.9, 0.4], ranking=query_ranking
            )
        )
        right = store.put(
            ops.reranked(
                store, right_base, members=papers[2:], scores=[0.7, 0.2], ranking=query_ranking
            )
        )
        record = ops.union(store, [left, right])
        assert record.order.state is OrderState.RANKED
        assert record.order.scores == (0.9, 0.7, 0.4, 0.2)
        assert [p.doi for p in record.members] == [
            "10.1000/0",
            "10.1000/2",
            "10.1000/1",
            "10.1000/3",
        ]

    def test_union_of_pool_dependent_rankings_is_invalidated(
        self, store, make_record, papers, query_ranking
    ) -> None:
        """A diversified order scores a paper by what else is in the pool."""
        mmr = dataclasses.replace(
            query_ranking, scoring_convention="mmr(lambda=0.5)", pool_independent=False
        )
        left_base = store.put(make_record(papers[:2]))
        right_base = store.put(make_record(papers[2:]))
        left = _ranked(store, left_base, mmr, list(papers[:2]))
        right = _ranked(store, right_base, mmr, list(papers[2:]))
        record = ops.union(store, [left, right])
        assert record.order.state is OrderState.INVALIDATED
        assert record.order.previous_ranking == mmr

    def test_union_of_different_queries_is_invalidated(
        self, store, make_record, papers, query_ranking
    ) -> None:
        """Scores against different questions are not comparable."""
        other = dataclasses.replace(query_ranking, query_text="a different question")
        left_base = store.put(make_record(papers[:2]))
        right_base = store.put(make_record(papers[2:]))
        left = _ranked(store, left_base, query_ranking, list(papers[:2]))
        right = _ranked(store, right_base, other, list(papers[2:]))
        record = ops.union(store, [left, right])
        assert record.order.state is OrderState.INVALIDATED

    def test_union_needs_two_operands(self, store, make_record, papers) -> None:
        """A one-operand union is not a union."""
        only = store.put(make_record(papers))
        with pytest.raises(InvalidParameterError, match="at least 2 operands"):
            ops.union(store, [only])

    def test_union_records_every_operand(self, store, make_record, papers) -> None:
        """Lineage is a graph; both operands appear as edges."""
        first = store.put(make_record(papers[:2]))
        second = store.put(make_record(papers[2:]))
        record = ops.union(store, [first, second])
        assert record.operation.handle_operand_ids() == [first, second]


class TestDifference:
    """Difference removes members and keeps the base's order."""

    def test_difference_removes_members(self, store, make_record, papers) -> None:
        """Members of the subtrahend are gone from the result."""
        base = store.put(make_record(papers))
        subtrahend = store.put(make_record(papers[:2]))
        record = ops.difference(store, base, [subtrahend])
        assert [p.doi for p in record.members] == ["10.1000/2", "10.1000/3"]

    def test_difference_removes_separately_deserialized_same_doi(
        self, store, make_record, papers
    ) -> None:
        """Subtraction is by explicit record identity, not Python object identity."""
        base = store.put(make_record(papers[:2]))
        clone = Paper.from_dict(papers[0].to_dict())
        subtrahend = store.put(make_record([clone]))

        record = ops.difference(store, base, [subtrahend])

        assert [paper.doi for paper in record.members] == ["10.1000/1"]

    def test_difference_preserves_a_ranking(
        self, store, make_record, papers, query_ranking
    ) -> None:
        """Removing members cannot falsify the survivors' relative sequence."""
        base = store.put(make_record(papers))
        ranked = store.put(
            ops.reranked(
                store,
                base,
                members=papers,
                scores=[0.9, 0.8, 0.7, 0.6],
                ranking=query_ranking,
            )
        )
        subtrahend = store.put(make_record([papers[1]]))
        record = ops.difference(store, ranked, [subtrahend])
        assert record.order.state is OrderState.RANKED
        assert record.order.ranking == query_ranking
        assert record.order.scores == (0.9, 0.7, 0.6)

    def test_difference_labels_operand_roles(self, store, make_record, papers) -> None:
        """Operand position is meaningful and is recorded as a role."""
        base = store.put(make_record(papers))
        subtrahend = store.put(make_record(papers[:1]))
        record = ops.difference(store, base, [subtrahend])
        roles = [operand.role for operand in record.operation.operands]
        assert roles == ["base", "subtrahend"]

    def test_difference_accepts_several_subtrahends(self, store, make_record, papers) -> None:
        """Difference is n-ary on the right."""
        base = store.put(make_record(papers))
        first = store.put(make_record(papers[:1]))
        second = store.put(make_record(papers[1:2]))
        record = ops.difference(store, base, [first, second])
        assert [p.doi for p in record.members] == ["10.1000/2", "10.1000/3"]

    def test_difference_needs_a_subtrahend(self, store, make_record, papers) -> None:
        """Subtracting nothing is not a difference."""
        base = store.put(make_record(papers))
        with pytest.raises(InvalidParameterError, match="at least one subtrahend"):
            ops.difference(store, base, [])

    def test_zotero_subtraction_goes_through_an_import_handle(
        self, store, make_record, papers, zotero_reference
    ) -> None:
        """The library snapshot is an operand in the graph, not an assumption."""
        found = store.put(make_record(papers))
        library = store.put(ops.imported(papers[:2], source=zotero_reference))
        new_work = store.put(ops.difference(store, found, [library], label="not yet held"))

        graph = store.lineage(new_work)
        assert library in graph.nodes
        assert [ref.kind for _, ref in graph.external] == ["zotero_library"]
        assert dict(graph.external)[library].identity["item_count"] == 412
        assert len(store.get(new_work).members) == 2


class TestIntersection:
    """Intersection keeps members common to every operand."""

    def test_intersection_keeps_common_members(self, store, make_record, papers) -> None:
        """Only papers present everywhere survive."""
        first = store.put(make_record(papers[:3]))
        second = store.put(make_record(papers[1:]))
        record = ops.intersection(store, [first, second])
        assert [p.doi for p in record.members] == ["10.1000/1", "10.1000/2"]

    def test_intersection_matches_native_provider_id_without_title_fallback(
        self, store, make_record, make_paper
    ) -> None:
        """Native record identity survives differing metadata in separate handles."""
        left = make_paper("Metadata from source A")
        left.identifiers = [PaperIdentifier(IdentifierKind.SEMANTIC_SCHOLAR, "S2-42")]
        right = Paper.from_dict(left.to_dict())
        right.title = "Metadata from source B"
        first = store.put(make_record([left]))
        second = store.put(make_record([right]))

        record = ops.intersection(store, [first, second])

        assert len(record.members) == 1
        assert record.members[0].title == "Metadata from source A"

    def test_intersection_inherits_the_first_operand_order(
        self, store, make_record, papers, query_ranking
    ) -> None:
        """The result is a subset of the first operand."""
        base = store.put(make_record(papers))
        ranked = store.put(
            ops.reranked(
                store, base, members=papers, scores=[0.9, 0.8, 0.7, 0.6], ranking=query_ranking
            )
        )
        other = store.put(make_record(papers[:2]))
        record = ops.intersection(store, [ranked, other])
        assert record.order.state is OrderState.RANKED
        assert record.order.scores == (0.9, 0.8)


class TestFilterAndDeduplicate:
    """Predicate filters and duplicate collapsing."""

    def test_filter_requires_a_description(self, store, make_record, papers) -> None:
        """An unexplained filter is a hole in the lineage."""
        base = store.put(make_record(papers))
        with pytest.raises(InvalidParameterError, match="non-empty description"):
            ops.filtered(store, base, lambda paper: True, description="   ")

    def test_filter_preserves_order(self, store, make_record, papers, query_ranking) -> None:
        """Filtering is a subsetting operation."""
        base = store.put(make_record(papers))
        ranked = store.put(
            ops.reranked(
                store, base, members=papers, scores=[0.9, 0.8, 0.7, 0.6], ranking=query_ranking
            )
        )
        record = ops.filtered(
            store,
            ranked,
            lambda paper: paper.doi in {"10.1000/0", "10.1000/3"},
            description="two chosen dois",
        )
        assert record.order.state is OrderState.RANKED
        assert record.order.scores == (0.9, 0.6)

    def test_deduplicate_invalidates_a_ranking(
        self, store, make_record, papers, query_ranking
    ) -> None:
        """A collapsed record is not the object that was scored."""
        base = store.put(make_record(papers))
        ranked = store.put(
            ops.reranked(
                store, base, members=papers, scores=[0.9, 0.8, 0.7, 0.6], ranking=query_ranking
            )
        )
        record = ops.deduplicated(
            store, ranked, members=papers[:3], strategy="exact-doi then model2vec cosine"
        )
        assert record.order.state is OrderState.INVALIDATED
        assert record.order.invalidated_by == "deduplicate"

    def test_deduplicate_on_an_unordered_set_stays_unordered(
        self, store, make_record, papers
    ) -> None:
        """The ordinary workflow deduplicates before ranking, so nothing is lost."""
        base = store.put(make_record(papers))
        record = ops.deduplicated(store, base, members=papers[:3], strategy="exact-doi")
        assert record.order.state is OrderState.UNORDERED

    def test_deduplicate_cannot_add_members(self, store, make_record, papers, make_paper):
        """Collapsing duplicates never grows a set."""
        base = store.put(make_record(papers[:2]))
        with pytest.raises(InvalidParameterError, match="cannot add members"):
            ops.deduplicated(store, base, members=papers, strategy="exact-doi")

    def test_deduplicate_refuses_unrelated_same_sized_substitution(
        self, store, make_record, papers, make_paper
    ) -> None:
        """A non-growing output is not necessarily a subset of the base."""
        base = store.put(make_record(papers[:2]))
        unrelated = [
            make_paper("Paper 0", "10.1000/unrelated-a"),
            make_paper("Paper 1", "10.1000/unrelated-b"),
        ]

        with pytest.raises(InvalidParameterError, match="cannot substitute unrelated papers"):
            ops.deduplicated(store, base, members=unrelated, strategy="purported dedup")

    def test_deduplicate_accepts_exact_id_subset_after_deserialization(
        self, store, make_record, papers
    ) -> None:
        """A separately loaded DOI record may truthfully survive deduplication."""
        base = store.put(make_record(papers[:2]))
        survivor = Paper.from_dict(papers[1].to_dict())

        record = ops.deduplicated(store, base, members=[survivor], strategy="exact-doi")

        assert [paper.doi for paper in record.members] == ["10.1000/1"]

    def test_deduplicate_accepts_identifier_free_exact_snapshot_subset(
        self, store, make_record, make_paper
    ) -> None:
        """Immutable identifier-free snapshots can survive a recorder round trip."""
        first = make_paper("First unknown record")
        second = make_paper("Second unknown record")
        base = store.put(make_record([first, second]))
        survivor = Paper.from_dict(second.to_dict())

        record = ops.deduplicated(store, base, members=[survivor], strategy="manual review")

        assert record.members[0].to_dict() == second.to_dict()

    def test_deduplicate_subset_matching_preserves_multiplicity(
        self, store, make_record, papers
    ) -> None:
        """One base identity cannot justify two surviving output positions."""
        base = store.put(make_record(papers[:2]))
        repeated = [Paper.from_dict(papers[0].to_dict()) for _ in range(2)]

        with pytest.raises(InvalidParameterError, match="cannot substitute unrelated papers"):
            ops.deduplicated(store, base, members=repeated, strategy="purported dedup")

    def test_deduplicate_requires_a_strategy(self, store, make_record, papers) -> None:
        """A merge with no stated rule cannot be audited."""
        base = store.put(make_record(papers))
        with pytest.raises(InvalidParameterError, match="non-empty strategy"):
            ops.deduplicated(store, base, members=papers, strategy="")


class TestCoveragePropagation:
    """A derived set never silently loses its operands' coverage."""

    def test_default_propagator_passes_through_absent_ledgers(
        self, store, make_record, papers
    ) -> None:
        """Nothing to propagate, nothing to refuse."""
        first = store.put(make_record(papers[:2]))
        second = store.put(make_record(papers[2:]))
        assert ops.union(store, [first, second]).coverage is None

    def test_default_propagator_refuses_an_unreadable_ledger(
        self, store, make_record, papers
    ) -> None:
        """Silently dropping coverage is the failure the ledger exists to prevent."""
        first = store.put(make_record(papers[:2], coverage={"arxiv": "capped"}))
        second = store.put(make_record(papers[2:]))
        with pytest.raises(CoveragePropagationError, match="not a serialized CoverageLedger"):
            ops.union(store, [first, second])

    def test_difference_against_an_incomplete_operand_is_recorded(
        self, store, make_record, papers
    ) -> None:
        """Subtracting an incomplete library labels held papers as new.

        This is the D8/D19 exclude-the-Zotero-library workflow.  It used to
        raise outright, because the algebra had no caller; it now completes and
        the result says why it cannot be trusted.
        """
        base = store.put(make_record(papers, coverage=_ledger(complete=("arxiv", 4))))
        library = store.put(make_record(papers[:2], coverage=_ledger(partial=("zotero", 2))))
        record = ops.difference(store, base, [library])
        assert record.coverage["status"] == "partial"
        assert record.coverage["derivations"][0]["role"] == "subtrahend"
        assert "may be present here in error" in record.coverage["derivations"][0]["detail"]

    def test_supplied_propagator_is_used(self, store, make_record, papers) -> None:
        """The seam works: a real propagator decides what the result carries."""
        first = store.put(make_record(papers[:2], coverage={"arxiv": "capped"}))
        second = store.put(make_record(papers[2:]))
        record = ops.union(store, [first, second], propagator=PassThroughPropagator())
        assert record.coverage == {"arxiv": "capped"}

    def test_refusing_propagator_names_the_carrying_operands(
        self, store, make_record, papers
    ) -> None:
        """The opt-in strict propagator points at which operand carried a ledger."""
        first = store.put(make_record(papers[:2]))
        second = store.put(make_record(papers[2:], coverage={"ieee": "failed"}))
        with pytest.raises(CoveragePropagationError, match=r"operand\(s\) \[1\]"):
            ops.union(store, [first, second], propagator=ops.RefusingCoveragePropagator())


class TestDerivedHandlesArePersistable:
    """Every derived record can be written and read back."""

    def test_full_pipeline_round_trips(
        self, store, make_record, papers, query_ranking, zotero_reference
    ) -> None:
        """Search, import, difference, rerank, filter all survive the store."""
        found = store.put(make_record(papers))
        library = store.put(ops.imported(papers[:1], source=zotero_reference))
        new_work = store.put(ops.difference(store, found, [library]))
        remaining = list(store.get(new_work).members)
        ranked = store.put(
            ops.reranked(
                store,
                new_work,
                members=list(reversed(remaining)),
                scores=[0.9, 0.5, 0.1],
                ranking=query_ranking,
            )
        )
        loaded = store.get(ranked)
        assert isinstance(loaded, HandleRecord)
        assert loaded.order == OrderRecord.ranked(query_ranking, [0.9, 0.5, 0.1])
        assert store.ancestors(ranked) == [new_work, found, library]


class TestSnowballBridge:
    """Handles built from the library's snowball result."""

    def test_snowball_handle_is_unordered_and_records_its_limits(self, papers) -> None:
        """A traversal returns papers in no ranked order; its caps are recorded."""
        from lit_search.core.snowball_result import SnowballResult

        result = SnowballResult(
            seed_paper=papers[0],
            max_depth=2,
            direction="backward",
            since=datetime.date(2020, 1, 2),
            until=datetime.date(2025, 3, 4),
            databases=["openalex"],
            enrichment_databases=["crossref", "web_scraping"],
            max_cited_by=75,
            max_papers_per_level=40,
            max_expansion_per_level=20,
            papers=list(papers[1:]),
        )
        record = ops.from_snowball_result(result, label="round one")
        assert record.order.state is OrderState.UNORDERED
        assert record.operation.name is OperationName.SNOWBALL
        assert record.operation.parameters["max_depth"] == 2
        assert record.operation.parameters["seed_doi"] == "10.1000/0"
        assert record.operation.parameters["databases"] == ["openalex"]
        assert record.operation.parameters["enrichment_databases"] == [
            "crossref",
            "web_scraping",
        ]
        assert record.operation.parameters["since"] == "2020-01-02"
        assert record.operation.parameters["until"] == "2025-03-04"
        assert record.operation.parameters["max_cited_by"] == 75
        assert record.operation.parameters["max_papers_per_level"] == 40
        assert record.operation.parameters["max_expansion_per_level"] == 20
        assert record.label == "round one"
        assert len(record.members) == 4
