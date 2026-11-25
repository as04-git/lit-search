"""Renderable views over stored handles.

:func:`lit_search.core.rendering.coverage_view` requires ``result.coverage`` to
be a real :class:`~lit_search.core.coverage.CoverageLedger`, and refuses
anything else -- deliberately, because :class:`~lit_search.store.record.HandleRecord`
types its ``coverage`` slot as ``Any`` and a bare string ``'complete'`` would
otherwise pass as a claim.  But a handle's ledger on disk *is* a plain
dictionary, so a handle rendered directly always reports "coverage is a dict,
not a CoverageLedger".

:class:`HandleView` closes that gap in the one honest way: it re-reads the
payload back into a ledger and refuses when it cannot, so a handle whose
coverage slot holds something unreadable renders as UNKNOWN with the reason
stated rather than as a ledger it is not.

This is what makes ``D18``'s "surface the failed list in every rendering of
that set" true of handles as well as of live search results -- which matters,
because a handle is the object a person actually looks at a day later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lit_search.core.coverage import CoverageLedger
from lit_search.core.paper import Paper
from lit_search.store.record import HandleRecord

#: Reason attached when a handle simply carries no ledger.
NO_LEDGER_REASON = (
    "this handle carries no coverage ledger. Its operation recorded none, so "
    "nothing is known about which sources were reached or what they failed to "
    "return -- which is not the same as nothing having been missed"
)


@dataclass(frozen=True, slots=True)
class HandleView:
    """A handle presented so the standard renderers can read it.

    Parameters
    ----------
    handle_id : str
        Identifier of the handle being shown.
    record : HandleRecord
        The handle itself.
    coverage : CoverageLedger | None
        Its ledger, rehydrated.  ``None`` when it has none or the payload could
        not be read.
    coverage_unavailable_reason : str | None
        Why there is no usable ledger.
    """

    handle_id: str
    record: HandleRecord
    coverage: CoverageLedger | None = None
    coverage_unavailable_reason: str | None = None

    @property
    def papers(self) -> list[Paper]:
        """Return the handle's members.

        Returns
        -------
        list[Paper]
            Members in the handle's own order.
        """
        return list(self.record.members)

    def headline(self) -> str:
        """Return the one-line description of the set.

        Returns
        -------
        str
            Operation, order state and member count.
        """
        order = self.record.order.state.value
        label = f" {self.record.label!r}" if self.record.label else ""
        return (
            f"{self.handle_id}{label} -- {self.record.operation.name.value}, "
            f"{len(self.record.members)} member(s), order {order}"
        )


def handle_view(handle_id: str, record: HandleRecord) -> HandleView:
    """Build a renderable view of a stored handle.

    Parameters
    ----------
    handle_id : str
        Identifier of the handle.
    record : HandleRecord
        The handle.

    Returns
    -------
    HandleView
        View whose ``coverage`` is a real ledger when one could be read, and
        which states why not when one could not.
    """
    payload: Any = record.coverage
    if payload is None:
        return HandleView(
            handle_id=handle_id, record=record, coverage_unavailable_reason=NO_LEDGER_REASON
        )
    if isinstance(payload, CoverageLedger):
        return HandleView(handle_id=handle_id, record=record, coverage=payload)
    if isinstance(payload, dict):
        try:
            return HandleView(
                handle_id=handle_id, record=record, coverage=CoverageLedger.from_dict(payload)
            )
        except (KeyError, TypeError, ValueError) as error:
            return HandleView(
                handle_id=handle_id,
                record=record,
                coverage_unavailable_reason=(
                    f"this handle's coverage payload is a mapping that could not be read "
                    f"as a ledger ({error}), so its claim cannot be checked and must not "
                    "be believed"
                ),
            )
    return HandleView(
        handle_id=handle_id,
        record=record,
        coverage_unavailable_reason=(
            f"this handle's coverage payload is a {type(payload).__name__}, not a "
            "serialized ledger, so its claim cannot be checked and must not be believed"
        ),
    )


__all__ = ["NO_LEDGER_REASON", "HandleView", "handle_view"]
