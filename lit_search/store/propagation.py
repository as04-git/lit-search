"""How a derived set's coverage is computed from its operands' coverage.

This module is the connection between two halves that were shipped separately
and never wired together: the asymmetric propagation algebra in
:mod:`lit_search.core.coverage` (``union_coverage``, ``intersection_coverage``,
``difference_coverage``) and the :class:`CoveragePropagator` seam that
:mod:`lit_search.store.operations` calls.  Until they were connected, the
algebra had **zero callers**, and every set operation on a handle derived from
a real search raised :class:`~lit_search.exceptions.CoveragePropagationError`
-- which blocked the exclude-the-Zotero-library workflow outright while
reading, from the outside, as though coverage propagation was solved.

Three things this module gets right that the seam alone could not
--------------------------------------------------------------------

**Operands carry their role.**  ``combine`` receives
:class:`CoverageOperand` values, not a bare sequence of payloads, because
``difference`` is asymmetric and position alone is an unsafe way to say so.
The only worked propagator that existed before this returned ``operands[0]`` --
first-operand-wins -- which is precisely the wrong rule for a difference: it
keeps the minuend's clean ledger and discards the fact that the subtrahend was
incomplete, so records that should have been removed and were never seen come
back labelled new.

**A missing ledger is UNKNOWN, not neutral.**  An operand with no coverage --
every snowball handle today -- becomes an empty
:class:`~lit_search.core.coverage.CoverageLedger`, whose aggregate status is
:data:`~lit_search.core.coverage.CoverageStatus.UNKNOWN`, so it floors the
result instead of quietly disappearing from the merge.

**An unreadable payload is refused, not dropped.**  The ``coverage`` slot on a
handle is opaque to the store, so a payload that is present but is not a
serialized ledger cannot be combined.  Refusing is the only honest answer;
returning ``None`` would produce a derived set that looks complete while its
operands were not.

What this module deliberately does NOT do
-----------------------------------------

``filter``, ``deduplicate`` and ``rerank`` are **membership** changes, and
their coverage is passed through unchanged.  That is a known hole -- a set
truncated by ``filter`` keeps a ``complete`` ledger, and subtracting it then
reports already-held papers as new -- and it is open item **O10** in the
specification, which offers two competing fixes (floor the status on every
membership-reducing operation, versus carry a member count on the ledger and
cross-check it).  Choosing between them is not this module's call, and picking
one here by default would settle an open question silently.  The behaviour is
named here so that the hole is visible rather than implied.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from lit_search.core.coverage import (
    CoverageLedger,
    difference_coverage,
    intersection_coverage,
    union_coverage,
)
from lit_search.exceptions import CoveragePropagationError
from lit_search.store.lineage import OperationName

#: Operand role assigned to the set an asymmetric operation subtracts.
SUBTRAHEND_ROLE = "subtrahend"

#: Operand role assigned to the set an operation derives from.
BASE_ROLE = "base"

#: Operand role assigned to a symmetric operand of a union or intersection.
MEMBER_ROLE = "member"


@dataclass(frozen=True, slots=True)
class CoverageOperand:
    """One operand of a set operation, as the propagator sees it.

    Parameters
    ----------
    role : str
        The operand's role in the operation: :data:`BASE_ROLE`,
        :data:`SUBTRAHEND_ROLE` or :data:`MEMBER_ROLE`.  Supplied because
        ``difference`` treats its operands differently and reading that off
        positions is a mistake waiting to be made.
    coverage : Any
        The operand handle's opaque coverage payload, or ``None`` when the
        handle carries no ledger.
    member_count : int
        How many papers the operand holds.  Carried because a propagator that
        wants to cross-check membership against a ledger cannot get it any
        other way, and adding it later would be a protocol change.
    """

    role: str
    coverage: Any
    member_count: int


@runtime_checkable
class CoveragePropagator(Protocol):
    """Combines the coverage ledgers of an operation's operands.

    The handle store does not know what a coverage ledger contains; that is
    owned by the coverage contract.  This protocol is the seam through which
    the two meet, and :class:`LedgerCoveragePropagator` is the implementation
    on the coverage side of it.

    An implementation must be conservative: a derived set inherits the worst
    status of its operands, and a difference taken against a non-complete
    operand is itself non-complete.  A difference against an incomplete
    library would otherwise label already-held papers as new.
    """

    def combine(self, operation: OperationName, operands: Sequence[CoverageOperand]) -> Any:
        """Return the coverage ledger for a derived set.

        Parameters
        ----------
        operation : OperationName
            Operation being performed.
        operands : Sequence[CoverageOperand]
            Operands in operand order, each carrying its role, its coverage
            payload (``None`` when it has none) and its member count.

        Returns
        -------
        Any
            Coverage payload for the result.
        """
        ...


class RefusingCoveragePropagator:
    """A propagator that refuses rather than guessing.

    Kept as an explicit opt-in for a caller that would rather have an
    operation fail than accept a propagation rule it did not choose.  It is no
    longer the default, because a default that refuses every set operation on
    a handle derived from a real search does not protect anybody -- it just
    makes the coverage-aware workflow unreachable, which is how the algebra
    came to have no callers.
    """

    def combine(self, operation: OperationName, operands: Sequence[CoverageOperand]) -> Any:
        """Refuse when any operand carries a ledger.

        Parameters
        ----------
        operation : OperationName
            Operation being performed.
        operands : Sequence[CoverageOperand]
            Operands in operand order.

        Returns
        -------
        Any
            Always ``None``, when no operand carried a ledger.

        Raises
        ------
        CoveragePropagationError
            If any operand carries a coverage ledger.
        """
        carrying = [index for index, operand in enumerate(operands) if operand.coverage is not None]
        if carrying:
            raise CoveragePropagationError(
                f"Operation {operation.value!r} has operand(s) {carrying} carrying a "
                "coverage ledger, and RefusingCoveragePropagator was selected.  Refusing "
                "to produce a derived set that silently drops its operands' coverage; use "
                "LedgerCoveragePropagator, or supply a propagator implementing the "
                "worst-status rule."
            )
        return None


def _looks_like_ledger(payload: Any) -> bool:
    """Whether *payload* has the shape :meth:`CoverageLedger.to_dict` produces.

    Parameters
    ----------
    payload : Any
        A handle's opaque coverage payload.

    Returns
    -------
    bool
        ``True`` when the payload can be read back as a ledger.
    """
    return isinstance(payload, dict) and ("sources" in payload or "derivations" in payload)


class LedgerCoveragePropagator:
    """The propagator that actually applies the coverage algebra.

    This is the ~25-line adapter the seam was waiting for: it reads each
    operand's opaque payload back into a
    :class:`~lit_search.core.coverage.CoverageLedger`, dispatches to the right
    algebra function for the operation, and serializes the answer back into a
    payload the handle can store.

    Operand payloads are handled as follows:

    ``None``
        Treated as an **empty ledger**, whose aggregate status is
        :data:`~lit_search.core.coverage.CoverageStatus.UNKNOWN`.  A snowball
        handle carries no ledger today, and letting that operand contribute
        nothing would leave the result reading as well-covered as its one
        instrumented operand.
    a serialized ledger
        Read back with :meth:`CoverageLedger.from_dict`.
    anything else
        Refused with :class:`~lit_search.exceptions.CoveragePropagationError`.
        A payload that cannot be read cannot be combined, and dropping it
        would be the silent loss the ledger exists to prevent.

    When **no** operand carries any payload, the result carries ``None``: the
    operands made no coverage claim, so neither does the result, and inventing
    an UNKNOWN ledger there would add a warning with no observation behind it.
    """

    def combine(self, operation: OperationName, operands: Sequence[CoverageOperand]) -> Any:
        """Return the derived set's coverage payload.

        Parameters
        ----------
        operation : OperationName
            Operation being performed.
        operands : Sequence[CoverageOperand]
            Operands in operand order, each carrying its role.

        Returns
        -------
        Any
            Serialized :class:`~lit_search.core.coverage.CoverageLedger`, or
            ``None`` when no operand carried one.

        Raises
        ------
        CoveragePropagationError
            If an operand's payload is present but is not a readable ledger,
            or if the operation needs operands and was given none.
        """
        if not operands or all(operand.coverage is None for operand in operands):
            return None

        ledgers = [self._read(operation, index, operand) for index, operand in enumerate(operands)]

        if operation is OperationName.UNION:
            return union_coverage(*ledgers).to_dict()
        if operation is OperationName.INTERSECTION:
            return intersection_coverage(*ledgers).to_dict()
        if operation is OperationName.DIFFERENCE:
            return self._difference(operands, ledgers).to_dict()
        # RERANK, FILTER and DEDUPLICATE derive from a single base and change
        # only membership.  See the module docstring and O10: coverage is
        # passed through unchanged, which is a known and deliberately
        # unresolved hole rather than an oversight.
        return ledgers[0].to_dict()

    def _difference(
        self, operands: Sequence[CoverageOperand], ledgers: Sequence[CoverageLedger]
    ) -> CoverageLedger:
        """Fold every subtrahend's shortfall into the minuend's ledger.

        Subtrahends are identified by :data:`SUBTRAHEND_ROLE`, never by
        position, so an operation that reorders its operands cannot silently
        turn a subtrahend into a minuend.

        Parameters
        ----------
        operands : Sequence[CoverageOperand]
            Operands in operand order.
        ledgers : Sequence[CoverageLedger]
            Their ledgers, in the same order.

        Returns
        -------
        CoverageLedger
            Ledger for the difference.
        """
        result = ledgers[0]
        for index, operand in enumerate(operands):
            if operand.role != SUBTRAHEND_ROLE:
                continue
            result = difference_coverage(
                result,
                ledgers[index],
                subtrahend_label=f"subtrahend operand {index} ({operand.member_count} member(s))",
            )
        return result

    def _read(
        self, operation: OperationName, index: int, operand: CoverageOperand
    ) -> CoverageLedger:
        """Read one operand's payload back into a ledger.

        Parameters
        ----------
        operation : OperationName
            Operation being performed, for the error message.
        index : int
            Operand position, for the error message.
        operand : CoverageOperand
            Operand to read.

        Returns
        -------
        CoverageLedger
            The operand's ledger, empty when it carried none.

        Raises
        ------
        CoveragePropagationError
            If the payload is present but unreadable.
        """
        payload = operand.coverage
        if payload is None:
            return CoverageLedger()
        if isinstance(payload, CoverageLedger):
            return payload
        if _looks_like_ledger(payload):
            try:
                return CoverageLedger.from_dict(payload)
            except (KeyError, TypeError, ValueError) as exc:
                raise CoveragePropagationError(
                    f"Operation {operation.value!r} operand {index} ({operand.role}) carries "
                    f"a coverage payload that could not be read as a ledger: {exc}. Refusing "
                    "to drop it; a derived set that loses its operand's coverage is the "
                    "confident incomplete answer the ledger exists to prevent."
                ) from exc
        raise CoveragePropagationError(
            f"Operation {operation.value!r} operand {index} ({operand.role}) carries a "
            f"coverage payload of type {type(payload).__name__} that is not a serialized "
            "CoverageLedger, so it cannot be combined. Refusing to drop it; a derived set "
            "that loses its operand's coverage is the confident incomplete answer the "
            "ledger exists to prevent."
        )
