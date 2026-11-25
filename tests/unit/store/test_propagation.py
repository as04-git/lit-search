"""Tests that the coverage algebra is actually reached by set operations.

``lit_search/core/coverage.py`` shipped a correct asymmetric propagation
algebra and ``lit_search/store/operations.py`` shipped a ``CoveragePropagator``
seam, and nothing in the tree connected them: the algebra had **zero callers**.
That is worse than having no algebra, because from the outside it reads as
solved.

Every test in :class:`TestTheAlgebraIsReached` fails if the wiring is taken
out -- if the default propagator goes back to refusing, or returns ``None``, or
stops dispatching on the operation.
"""

from __future__ import annotations

import pytest

from lit_search.core.coverage import (
    CoverageLedger,
    CoverageScope,
    CoverageStatus,
    ExecutionStatus,
    SourceCoverage,
    difference_coverage,
    union_coverage,
)
from lit_search.exceptions import CoveragePropagationError
from lit_search.store import operations as ops
from lit_search.store.lineage import OperationName
from lit_search.store.propagation import (
    BASE_ROLE,
    MEMBER_ROLE,
    SUBTRAHEND_ROLE,
    CoverageOperand,
    LedgerCoveragePropagator,
    RefusingCoveragePropagator,
)


def _complete(source: str, count: int) -> CoverageLedger:
    """Build a ledger for one completely retrieved source.

    Parameters
    ----------
    source : str
        Source identifier.
    count : int
        Records retrieved.

    Returns
    -------
    CoverageLedger
        Ledger whose aggregate status is complete.
    """
    return CoverageLedger.from_sources(
        [SourceCoverage.complete(source, count, reported_total=count)]
    )


def _capped(source: str, count: int, cap: int) -> CoverageLedger:
    """Build a ledger for one source truncated at a cap.

    Parameters
    ----------
    source : str
        Source identifier.
    count : int
        Records retrieved.
    cap : int
        Limit retrieval stopped at.

    Returns
    -------
    CoverageLedger
        Ledger whose aggregate status is capped.
    """
    return CoverageLedger.from_sources([SourceCoverage.capped(source, count, cap=cap)])


def _partial(source: str, count: int) -> CoverageLedger:
    """Build a ledger for one partially retrieved source.

    Parameters
    ----------
    source : str
        Source identifier.
    count : int
        Records retrieved.

    Returns
    -------
    CoverageLedger
        Ledger whose aggregate status is partial.
    """
    return CoverageLedger.from_sources(
        [SourceCoverage.partial(source, count, detail="page 3 never arrived")]
    )


class TestTheAlgebraIsReached:
    """Set operations run the propagation algebra rather than refusing."""

    def test_union_produces_exactly_what_the_algebra_says(self, store, make_record, papers) -> None:
        """The wiring test: the handle carries ``union_coverage``'s own answer."""
        left = _complete("arxiv", 2)
        right = _capped("openalex", 2, 2)
        first = store.put(make_record(papers[:2], coverage=left.to_dict()))
        second = store.put(make_record(papers[2:], coverage=right.to_dict()))

        record = ops.union(store, [first, second])

        assert record.coverage == union_coverage(left, right).to_dict()
        assert record.coverage["status"] == CoverageStatus.CAPPED.value

    def test_difference_produces_exactly_what_the_algebra_says(
        self, store, make_record, papers
    ) -> None:
        """The asymmetric case reaches ``difference_coverage``, not a merge."""
        minuend = _complete("arxiv", 4)
        subtrahend = _partial("zotero", 2)
        base = store.put(make_record(papers, coverage=minuend.to_dict()))
        library = store.put(make_record(papers[:2], coverage=subtrahend.to_dict()))

        record = ops.difference(store, base, [library])

        assert record.coverage["status"] == CoverageStatus.PARTIAL.value
        assert [note["role"] for note in record.coverage["derivations"]] == ["subtrahend"]
        # The subtrahend's sources are not sources of the result: nothing was
        # searched from them, they were subtracted.
        assert [entry["source"] for entry in record.coverage["sources"]] == ["arxiv"]
        assert record.coverage["execution"]["status"] == ExecutionStatus.INCOMPLETE.value

    def test_scope_and_planned_partial_execution_survive_handle_propagation(
        self, store, make_record, papers
    ) -> None:
        """Opaque handle persistence does not drop either new verdict dimension."""
        bounded = CoverageLedger(
            entries=(
                SourceCoverage.partial(
                    "archive",
                    2,
                    detail="declared source window",
                    execution_status=ExecutionStatus.COMPLETE,
                ),
            ),
            scope=CoverageScope(
                profile="explicit",
                selected_sources=("archive",),
                registered_sources=("archive", "ieee"),
            ),
        )
        base = store.put(make_record(papers[:2], coverage=bounded.to_dict()))

        record = ops.filtered(store, base, lambda paper: True, description="keep")
        restored = CoverageLedger.from_dict(record.coverage)

        assert restored.status is CoverageStatus.PARTIAL
        assert restored.execution_status is ExecutionStatus.COMPLETE
        assert restored.scope == bounded.scope

    def test_a_complete_subtrahend_costs_nothing(self, store, make_record, papers) -> None:
        """The guard must not fire when the library really was fully synced."""
        base = store.put(make_record(papers, coverage=_complete("arxiv", 4).to_dict()))
        library = store.put(make_record(papers[:2], coverage=_complete("zotero", 2).to_dict()))

        record = ops.difference(store, base, [library])

        assert record.coverage["status"] == CoverageStatus.COMPLETE.value
        assert record.coverage["derivations"] == []

    def test_intersection_reaches_the_algebra(self, store, make_record, papers) -> None:
        """An intersection is not safer for being smaller."""
        first = store.put(make_record(papers[:3], coverage=_complete("arxiv", 3).to_dict()))
        second = store.put(make_record(papers[1:], coverage=_partial("ieee", 3).to_dict()))

        record = ops.intersection(store, [first, second])

        assert record.coverage["status"] == CoverageStatus.PARTIAL.value

    def test_filter_and_rerank_carry_the_base_ledger(self, store, make_record, papers) -> None:
        """Membership-reducing operations pass coverage through.

        This is deliberately **not** a claim that the behaviour is right.  A
        set truncated by ``filter`` keeps its ``complete`` ledger, and
        subtracting it then reports already-held papers as new.  Which of the
        two candidate fixes to apply is open item O10, so the current
        behaviour is pinned here rather than silently changed.
        """
        base = store.put(make_record(papers, coverage=_complete("arxiv", 4).to_dict()))

        record = ops.filtered(store, base, lambda paper: False, description="drop everything")

        assert record.members == ()
        assert record.coverage["status"] == CoverageStatus.COMPLETE.value

    def test_the_payload_survives_the_store(self, store, make_record, papers) -> None:
        """A propagated ledger must be JSON-serializable and hash-stable."""
        first = store.put(make_record(papers[:2], coverage=_complete("arxiv", 2).to_dict()))
        second = store.put(make_record(papers[2:], coverage=_partial("ieee", 2).to_dict()))

        handle = store.put(ops.union(store, [first, second]))
        reloaded = store.get(handle)

        assert CoverageLedger.from_dict(reloaded.coverage).status is CoverageStatus.PARTIAL


class TestAMissingLedgerIsUnknownNotNeutral:
    """An operand with no coverage must not disarm the guard."""

    def test_a_snowball_operand_floors_the_union(self, store, make_record, papers) -> None:
        """Every snowball handle carries ``coverage=None`` today."""
        searched = store.put(make_record(papers[:2], coverage=_complete("arxiv", 2).to_dict()))
        snowballed = store.put(make_record(papers[2:]))

        record = ops.union(store, [searched, snowballed])

        assert record.coverage["status"] == CoverageStatus.UNKNOWN.value
        assert any(
            "no coverage ledger" in note["detail"] for note in record.coverage["derivations"]
        )

    def test_a_snowball_subtrahend_makes_the_difference_unknown(
        self, store, make_record, papers
    ) -> None:
        """Subtracting a set nobody measured cannot yield a complete result."""
        base = store.put(make_record(papers, coverage=_complete("arxiv", 4).to_dict()))
        snowballed = store.put(make_record(papers[:2]))

        record = ops.difference(store, base, [snowballed])

        assert record.coverage["status"] == CoverageStatus.UNKNOWN.value

    def test_no_operand_carrying_anything_claims_nothing(self, store, make_record, papers) -> None:
        """Nothing observed, nothing claimed -- and nothing invented either."""
        first = store.put(make_record(papers[:2]))
        second = store.put(make_record(papers[2:]))

        assert ops.union(store, [first, second]).coverage is None


class TestRolesNotPositions:
    """``combine`` is told which operand is the subtrahend."""

    def test_the_propagator_receives_roles(self, store, make_record, papers) -> None:
        """Position alone is unsafe, so the role travels with the operand."""
        seen: list[list[str]] = []

        class _RoleRecorder:
            def combine(self, operation, operands):
                """Record the roles and return nothing.

                Returns
                -------
                None
                    No ledger.
                """
                seen.append([operand.role for operand in operands])
                return None

        base = store.put(make_record(papers))
        first = store.put(make_record(papers[:1]))
        second = store.put(make_record(papers[1:2]))
        ops.difference(store, base, [first, second], propagator=_RoleRecorder())

        assert seen == [[BASE_ROLE, SUBTRAHEND_ROLE, SUBTRAHEND_ROLE]]

    def test_first_operand_wins_would_give_a_different_answer(self) -> None:
        """The rule the only worked propagator used is unsound for difference."""
        minuend = _complete("arxiv", 4)
        subtrahend = _partial("zotero", 2)
        operands = [
            CoverageOperand(BASE_ROLE, minuend.to_dict(), 4),
            CoverageOperand(SUBTRAHEND_ROLE, subtrahend.to_dict(), 2),
        ]

        combined = LedgerCoveragePropagator().combine(OperationName.DIFFERENCE, operands)

        assert combined != minuend.to_dict()
        assert (
            combined
            == difference_coverage(
                minuend, subtrahend, subtrahend_label="subtrahend operand 1 (2 member(s))"
            ).to_dict()
        )

    def test_every_subtrahend_contributes_a_note(self) -> None:
        """Two incomplete subtrahends are two separate reasons to doubt."""
        operands = [
            CoverageOperand(BASE_ROLE, _complete("arxiv", 4).to_dict(), 4),
            CoverageOperand(SUBTRAHEND_ROLE, _partial("zotero", 1).to_dict(), 1),
            CoverageOperand(SUBTRAHEND_ROLE, _capped("mendeley", 1, 1).to_dict(), 1),
        ]

        combined = LedgerCoveragePropagator().combine(OperationName.DIFFERENCE, operands)

        assert len(combined["derivations"]) == 2


class TestUnreadablePayloadsAreRefused:
    """A payload that cannot be combined must not be dropped."""

    def test_an_opaque_dict_is_refused(self, store, make_record, papers) -> None:
        """Silently dropping it is the failure the ledger exists to prevent."""
        first = store.put(make_record(papers[:2], coverage={"arxiv": "capped"}))
        second = store.put(make_record(papers[2:]))

        with pytest.raises(CoveragePropagationError, match="not a serialized CoverageLedger"):
            ops.union(store, [first, second])

    def test_the_message_names_the_operand_and_its_role(self) -> None:
        """An error that does not say which operand is not actionable."""
        operands = [
            CoverageOperand(BASE_ROLE, _complete("arxiv", 1).to_dict(), 1),
            CoverageOperand(SUBTRAHEND_ROLE, "not a ledger at all", 1),
        ]

        with pytest.raises(CoveragePropagationError, match=r"operand 1 \(subtrahend\)"):
            LedgerCoveragePropagator().combine(OperationName.DIFFERENCE, operands)

    def test_a_malformed_ledger_shape_is_refused(self) -> None:
        """Ledger-shaped but unreadable is still unreadable."""
        operands = [
            CoverageOperand(MEMBER_ROLE, {"sources": [{"source": "x"}]}, 1),
            CoverageOperand(MEMBER_ROLE, None, 0),
        ]

        with pytest.raises(CoveragePropagationError, match="could not be read as a ledger"):
            LedgerCoveragePropagator().combine(OperationName.UNION, operands)


class TestRefusingPropagatorRemainsAvailable:
    """The strict propagator survives as an explicit opt-in."""

    def test_it_refuses_a_carried_ledger(self, store, make_record, papers) -> None:
        """A caller who wants an operation to fail rather than propagate can say so."""
        first = store.put(make_record(papers[:2], coverage=_complete("arxiv", 2).to_dict()))
        second = store.put(make_record(papers[2:]))

        with pytest.raises(CoveragePropagationError, match=r"operand\(s\) \[0\]"):
            ops.union(store, [first, second], propagator=RefusingCoveragePropagator())

    def test_it_allows_an_operation_with_no_ledgers(self, store, make_record, papers) -> None:
        """Nothing to propagate, nothing to refuse."""
        first = store.put(make_record(papers[:2]))
        second = store.put(make_record(papers[2:]))

        assert (
            ops.union(store, [first, second], propagator=RefusingCoveragePropagator()).coverage
            is None
        )
