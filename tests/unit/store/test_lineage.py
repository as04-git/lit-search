"""Tests for the lineage DAG: operands, external references, traversal."""

from __future__ import annotations

import datetime

import pytest

from lit_search.exceptions import HandleNotFoundError, HandleSchemaError
from lit_search.store import operations as ops
from lit_search.store.lineage import (
    ExternalReference,
    Operand,
    OperandKind,
    Operation,
    OperationName,
)

from .conftest import FIXED_TIME


class TestExternalReference:
    """An external operand must actually identify something."""

    def test_empty_identity_is_refused(self) -> None:
        """An unidentified operand is an implied dependency by another name."""
        with pytest.raises(HandleSchemaError, match="empty identity"):
            ExternalReference(kind="zotero_library", identity={}, captured_at=FIXED_TIME)

    def test_blank_kind_is_refused(self) -> None:
        """The reference must say what sort of thing it points at."""
        with pytest.raises(HandleSchemaError, match="what kind of thing"):
            ExternalReference(kind="", identity={"id": 1}, captured_at=FIXED_TIME)

    def test_non_scalar_identity_value_is_refused(self) -> None:
        """Identity fields must survive JSON without losing meaning."""
        with pytest.raises(HandleSchemaError, match="JSON scalar"):
            ExternalReference(
                kind="zotero_library", identity={"tags": ["a"]}, captured_at=FIXED_TIME
            )

    def test_naive_capture_time_is_normalized(self) -> None:
        """A naive timestamp is recorded as UTC rather than left ambiguous."""
        reference = ExternalReference(
            kind="kaggle_arxiv_metadata",
            identity={"snapshot": "2026-08-01"},
            captured_at=datetime.datetime(2026, 8, 1, 9, 0, 0),
        )
        assert reference.captured_at.tzinfo is datetime.UTC

    def test_round_trip(self, zotero_reference) -> None:
        """Serialization preserves kind, identity, and capture time."""
        restored = ExternalReference.from_dict(zotero_reference.to_dict(), "external")
        assert restored == zotero_reference

    def test_unknown_key_is_refused(self, zotero_reference) -> None:
        """Refusal reaches into external references too."""
        payload = zotero_reference.to_dict()
        payload["collection"] = "reading"
        with pytest.raises(HandleSchemaError, match="unrecognised key"):
            ExternalReference.from_dict(payload, "external")

    def test_bad_timestamp_is_refused(self, zotero_reference) -> None:
        """An unparsable capture time is refused, not defaulted to now."""
        payload = zotero_reference.to_dict()
        payload["captured_at"] = "last tuesday"
        with pytest.raises(HandleSchemaError, match="ISO-8601"):
            ExternalReference.from_dict(payload, "external")


class TestOperand:
    """An operand's payload must match its kind."""

    def test_handle_operand_needs_an_identifier(self) -> None:
        """A handle operand pointing at nothing is malformed."""
        with pytest.raises(HandleSchemaError, match="must carry a handle_id"):
            Operand(kind=OperandKind.HANDLE, role="base")

    def test_external_operand_needs_a_reference(self) -> None:
        """An external operand with no reference is malformed."""
        with pytest.raises(HandleSchemaError, match="must carry an external reference"):
            Operand(kind=OperandKind.EXTERNAL, role="source")

    def test_operand_needs_a_role(self, zotero_reference) -> None:
        """Roles are what make a subtraction's operand positions readable."""
        with pytest.raises(HandleSchemaError, match="must record the role"):
            Operand.of_external(zotero_reference, role="")

    def test_mixed_payload_is_refused(self, zotero_reference) -> None:
        """An operand is one kind or the other, never both."""
        with pytest.raises(HandleSchemaError, match="no external reference"):
            Operand(
                kind=OperandKind.HANDLE,
                role="base",
                handle_id="h_" + "0" * 32,
                external=zotero_reference,
            )

    def test_round_trip(self, zotero_reference) -> None:
        """Both operand kinds survive serialization."""
        for operand in (
            Operand.of_handle("h_" + "0" * 32, "base"),
            Operand.of_external(zotero_reference),
        ):
            assert Operand.from_dict(operand.to_dict(), "operand") == operand


class TestOperationArity:
    """Set operations refuse to be recorded with too few operands."""

    @pytest.mark.parametrize(
        "name", [OperationName.UNION, OperationName.DIFFERENCE, OperationName.INTERSECTION]
    )
    def test_multi_operand_operations_need_two(self, name) -> None:
        """A binary operation recorded with one operand hides its other input."""
        with pytest.raises(HandleSchemaError, match="at least two handle operands"):
            Operation(name=name, operands=(Operand.of_handle("h_" + "0" * 32, "base"),))

    @pytest.mark.parametrize(
        "name", [OperationName.RERANK, OperationName.FILTER, OperationName.DEDUPLICATE]
    )
    def test_single_operand_operations_need_exactly_one(self, name) -> None:
        """A transform of one set records exactly one base."""
        with pytest.raises(HandleSchemaError, match="exactly one handle operand"):
            Operation(name=name)

    def test_external_operands_do_not_count_toward_arity(self, zotero_reference) -> None:
        """An import has an external operand and no handle operands."""
        operation = Operation(
            name=OperationName.IMPORT, operands=(Operand.of_external(zotero_reference),)
        )
        assert operation.handle_operand_ids() == []

    def test_round_trip(self, zotero_reference) -> None:
        """An operation survives serialization with operands and parameters."""
        operation = Operation(
            name=OperationName.IMPORT,
            operands=(Operand.of_external(zotero_reference),),
            parameters={"collection": "reading"},
        )
        assert Operation.from_dict(operation.to_dict(), "operation") == operation

    def test_unknown_operation_name_is_refused(self) -> None:
        """The operation set is closed."""
        payload = Operation(name=OperationName.SEARCH).to_dict()
        payload["name"] = "vibe_search"
        with pytest.raises(HandleSchemaError, match="not one of"):
            Operation.from_dict(payload, "operation")


class TestLineageTraversal:
    """Ancestry is a graph walk, not a parent chain."""

    def test_source_handle_has_no_parents(self, store, make_record, papers) -> None:
        """A search handle is a root."""
        handle_id = store.put(make_record(papers))
        graph = store.lineage(handle_id)
        assert graph.nodes == (handle_id,)
        assert graph.edges == ()
        assert store.ancestors(handle_id) == []

    def test_difference_records_both_operands_with_roles(self, store, make_record, papers) -> None:
        """A subtraction's second operand is in the graph, not discarded."""
        base = store.put(make_record(papers))
        subtrahend = store.put(make_record(papers[:2]))
        result = store.put(ops.difference(store, base, [subtrahend]))

        graph = store.lineage(result)
        assert set(graph.parents_of(result)) == {base, subtrahend}
        roles = {edge.parent: edge.role for edge in graph.edges}
        assert roles == {base: "base", subtrahend: "subtrahend"}
        assert all(edge.operation is OperationName.DIFFERENCE for edge in graph.edges)

    def test_diamond_visits_a_shared_ancestor_once(self, store, make_record, papers) -> None:
        """A DAG with two paths to one ancestor is walked, not re-walked."""
        base = store.put(make_record(papers))
        left = store.put(
            ops.filtered(
                store, base, lambda p: p.doi in {"10.1000/0", "10.1000/1"}, description="first two"
            )
        )
        right = store.put(
            ops.filtered(
                store, base, lambda p: p.doi in {"10.1000/2", "10.1000/3"}, description="last two"
            )
        )
        merged = store.put(ops.union(store, [left, right]))

        graph = store.lineage(merged)
        assert sorted(graph.nodes) == sorted([merged, left, right, base])
        assert graph.nodes.count(base) == 1
        # Both branches keep their edge to the shared ancestor.
        assert sum(1 for edge in graph.edges if edge.parent == base) == 2
        assert graph.missing == ()

    def test_missing_ancestor_is_reported_not_skipped(self, store, make_record, papers) -> None:
        """A hole in the ancestry is a different fact from a complete one."""
        base = store.put(make_record(papers))
        derived = store.put(ops.filtered(store, base, lambda p: True, description="everything"))
        store.path_for(base).unlink()

        graph = store.lineage(derived)
        assert graph.missing == (base,)
        assert base not in graph.nodes
        assert graph.parents_of(derived) == [base]

    def test_external_snapshots_surface_in_the_graph(
        self, store, make_record, papers, zotero_reference
    ) -> None:
        """The chain back to a mutable source is walkable, not assumed."""
        found = store.put(make_record(papers))
        library = store.put(ops.imported(papers[:2], source=zotero_reference))
        result = store.put(ops.difference(store, found, [library]))

        graph = store.lineage(result)
        assert len(graph.external) == 1
        owner, reference = graph.external[0]
        assert owner == library
        assert reference.kind == "zotero_library"
        assert reference.captured_at == FIXED_TIME

    def test_lineage_of_a_missing_handle_raises(self, store) -> None:
        """You cannot explain a handle that is not there."""
        with pytest.raises(HandleNotFoundError):
            store.lineage("h_" + "0" * 32)

    def test_ancestors_are_breadth_first(self, store, make_record, papers) -> None:
        """Nearest ancestors come first, which is what a reader wants."""
        base = store.put(make_record(papers))
        middle = store.put(ops.filtered(store, base, lambda p: True, description="all"))
        top = store.put(ops.filtered(store, middle, lambda p: True, description="all again"))
        assert store.ancestors(top) == [middle, base]


def test_operand_payload_missing_external_key_is_refused(zotero_reference) -> None:
    """A nullable key that is absent entirely is still refused."""
    payload = Operand.of_external(zotero_reference).to_dict()
    del payload["external"]
    with pytest.raises(HandleSchemaError, match="external: required key is missing"):
        Operand.from_dict(payload, "operand")


def test_external_identity_with_non_string_key_is_refused() -> None:
    """Identity keys must be strings so the payload survives JSON."""
    with pytest.raises(HandleSchemaError, match="identity keys must be strings"):
        ExternalReference(kind="k", identity={1: "a"}, captured_at=FIXED_TIME)
