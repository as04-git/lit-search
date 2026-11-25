"""Lineage: how a handle came to exist.

Lineage is a **directed acyclic graph, not a parent chain**.  A single-parent
model cannot express the operations that matter most here: union and
intersection take two or more operands, and difference takes a base and one or
more subtrahends whose *positions* are meaningful.  Recording only "the parent"
would silently discard the second operand of every subtraction, which is the
operand whose incompleteness would make the answer wrong.

Two kinds of operand
--------------------

:attr:`OperandKind.HANDLE`
    Another handle in the same store.  The store validates on write that the
    referenced handle exists, so a dangling lineage edge fails at the moment it
    is created rather than months later during an audit.

:attr:`OperandKind.EXTERNAL`
    Something that is not a handle, described by an
    :class:`ExternalReference`: a Zotero library at a moment in time, a Kaggle
    arXiv metadata snapshot, a checked-in index.

Why an external operand must be explicit
----------------------------------------

Subtracting "the papers I already have" from a search depends on a **mutable**
library.  If that dependency is implied rather than recorded, two runs a week
apart produce different answers with nothing in either result explaining why,
and a subtraction taken against a partially-synced library will confidently
label already-held papers as new.

The store therefore takes the stricter route: to subtract a library you must
first import it as its own handle, which pins its contents and makes it an
ordinary operand.  That import handle records the library's identity as an
:class:`ExternalReference`, so the chain from "this set is new work" back to
"as judged against this library at this timestamp with this many items" is a
walkable path rather than an assumption.  An :class:`ExternalReference` with an
empty identity is refused for the same reason.

Acyclicity
----------

Cycles are impossible by construction: a handle identifier is a content hash
of a record that already contains its operands' identifiers, so closing a cycle
would require knowing a hash before computing it.  Traversal is still
cycle-guarded, because a hand-written file is not bound by that argument.
"""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from lit_search.exceptions import HandleSchemaError
from lit_search.store.strict import (
    reject_unknown_keys,
    require,
    require_enum,
    require_mapping,
    require_optional,
)


class OperationName(str, Enum):
    """The closed set of operations that can produce a handle.

    Closed rather than free-form so that order-transition rules and coverage
    propagation rules can be exhaustive.  Adding an operation is a deliberate
    act that forces both rule tables to be revisited.
    """

    #: A database search produced this set.
    SEARCH = "search"
    #: A citation traversal produced this set.
    SNOWBALL = "snowball"
    #: A typed query-family plan combined its per-cell search handles.
    QUERY_FAMILY = "query_family"
    #: Papers were brought in from outside the store (file, Zotero, dataset).
    IMPORT = "import"
    #: Members were re-ordered against a query or a seed paper.
    RERANK = "rerank"
    #: Members of two or more sets were combined.
    UNION = "union"
    #: Members of later operands were removed from the first.
    DIFFERENCE = "difference"
    #: Only members present in every operand were kept.
    INTERSECTION = "intersection"
    #: Members were removed by a predicate.
    FILTER = "filter"
    #: Duplicate records within one set were collapsed.
    DEDUPLICATE = "deduplicate"


class OperandKind(str, Enum):
    """Whether an operand is a stored handle or something outside the store."""

    #: Another handle in this store.
    HANDLE = "handle"
    #: A snapshot of something the store does not own.
    EXTERNAL = "external"


_EXTERNAL_KEYS = ("kind", "identity", "captured_at")


@dataclass(frozen=True)
class ExternalReference:
    """Identity of something outside the store that an operation depended on.

    Attributes
    ----------
    kind : str
        What sort of thing this is, for example ``"zotero_library"`` or
        ``"kaggle_arxiv_metadata"``.  Deliberately a free string rather than a
        closed enum: new external sources appear, and refusing to record one
        because it has no enum member would push callers back toward leaving
        the dependency implicit.
    identity : Mapping[str, str | int | float | bool]
        Fields that pin *which* thing and *which version of it*, for example
        ``{"library_id": 7067953, "item_count": 412, "snapshot_digest": "..."}``.
        Must be non-empty.
    captured_at : datetime.datetime
        When the snapshot was taken, in UTC.
    """

    kind: str
    identity: Mapping[str, str | int | float | bool]
    captured_at: datetime.datetime

    def __post_init__(self) -> None:
        """Reject an external reference that does not actually identify anything.

        Returns
        -------
        None

        Raises
        ------
        HandleSchemaError
            If the kind is blank, the identity is empty, or the identity holds
            a value that is not a JSON scalar.
        """
        if not self.kind:
            raise HandleSchemaError("An external operand must record what kind of thing it is")
        if not self.identity:
            raise HandleSchemaError(
                f"External operand of kind {self.kind!r} has an empty identity; an "
                "operand that cannot be identified is an implied dependency, which "
                "is what recording it is meant to prevent"
            )
        for key, value in self.identity.items():
            if not isinstance(key, str):
                raise HandleSchemaError("External operand identity keys must be strings")
            if not isinstance(value, str | int | float | bool):
                raise HandleSchemaError(
                    f"External operand identity field {key!r} must be a JSON scalar, "
                    f"got {type(value).__name__}"
                )
        object.__setattr__(self, "identity", dict(self.identity))
        object.__setattr__(self, "captured_at", _as_utc(self.captured_at))

    def to_dict(self) -> dict[str, Any]:
        """Serialize this reference to a plain dictionary.

        Returns
        -------
        dict[str, Any]
            JSON-ready representation with identity keys sorted for a stable
            content hash.
        """
        return {
            "kind": self.kind,
            "identity": {key: self.identity[key] for key in sorted(self.identity)},
            "captured_at": self.captured_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: Any, context: str) -> ExternalReference:
        """Reconstruct an external reference, refusing anything unrecognised.

        Parameters
        ----------
        payload : Any
            Mapping produced by :meth:`to_dict`.
        context : str
            Dotted path of the payload, used in error messages.

        Returns
        -------
        ExternalReference
            Reconstructed reference.

        Raises
        ------
        HandleSchemaError
            If a key is missing, mistyped, or unrecognised.
        """
        data = require_mapping(payload, context)
        reject_unknown_keys(data, _EXTERNAL_KEYS, context)
        identity_raw = require_mapping(
            require(data, "identity", dict, context), f"{context}.identity"
        )
        return cls(
            kind=require(data, "kind", str, context),
            identity=dict(identity_raw),
            captured_at=_parse_timestamp(
                require(data, "captured_at", str, context), f"{context}.captured_at"
            ),
        )


_OPERAND_KEYS = ("kind", "role", "handle_id", "external")


@dataclass(frozen=True)
class Operand:
    """One input to the operation that produced a handle.

    Attributes
    ----------
    kind : OperandKind
        Whether this operand is a handle or an external snapshot.
    role : str
        What part this operand played, for example ``"base"``,
        ``"subtrahend"``, ``"member"``, or ``"source"``.  Roles make a
        difference operand's position self-describing instead of relying on
        list order alone.
    handle_id : str | None
        Set exactly when *kind* is :attr:`OperandKind.HANDLE`.
    external : ExternalReference | None
        Set exactly when *kind* is :attr:`OperandKind.EXTERNAL`.
    """

    kind: OperandKind
    role: str
    handle_id: str | None = None
    external: ExternalReference | None = None

    def __post_init__(self) -> None:
        """Reject an operand whose payload does not match its kind.

        Returns
        -------
        None

        Raises
        ------
        HandleSchemaError
            If the operand carries no role, or if the wrong payload field is
            populated for its kind.
        """
        if not self.role:
            raise HandleSchemaError("An operand must record the role it played")
        if self.kind is OperandKind.HANDLE:
            if not self.handle_id or self.external is not None:
                raise HandleSchemaError(
                    "A handle operand must carry a handle_id and no external reference"
                )
        elif self.external is None or self.handle_id is not None:
            raise HandleSchemaError(
                "An external operand must carry an external reference and no handle_id"
            )

    @classmethod
    def of_handle(cls, handle_id: str, role: str) -> Operand:
        """Build an operand referring to another handle.

        Parameters
        ----------
        handle_id : str
            Identifier of the referenced handle.
        role : str
            Part the operand played.

        Returns
        -------
        Operand
            A handle operand.
        """
        return cls(kind=OperandKind.HANDLE, role=role, handle_id=handle_id)

    @classmethod
    def of_external(cls, external: ExternalReference, role: str = "source") -> Operand:
        """Build an operand referring to a snapshot outside the store.

        Parameters
        ----------
        external : ExternalReference
            Identity of the external thing.
        role : str
            Part the operand played, by default ``"source"``.

        Returns
        -------
        Operand
            An external operand.
        """
        return cls(kind=OperandKind.EXTERNAL, role=role, external=external)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this operand to a plain dictionary.

        Returns
        -------
        dict[str, Any]
            JSON-ready representation.
        """
        return {
            "kind": self.kind.value,
            "role": self.role,
            "handle_id": self.handle_id,
            "external": self.external.to_dict() if self.external is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: Any, context: str) -> Operand:
        """Reconstruct an operand, refusing anything unrecognised.

        Parameters
        ----------
        payload : Any
            Mapping produced by :meth:`to_dict`.
        context : str
            Dotted path of the payload, used in error messages.

        Returns
        -------
        Operand
            Reconstructed operand.

        Raises
        ------
        HandleSchemaError
            If a key is missing, mistyped, or unrecognised.
        """
        data = require_mapping(payload, context)
        reject_unknown_keys(data, _OPERAND_KEYS, context)
        if "external" not in data:
            raise HandleSchemaError(f"{context}.external: required key is missing")
        external_raw = data["external"]
        external = (
            None
            if external_raw is None
            else ExternalReference.from_dict(external_raw, f"{context}.external")
        )
        return cls(
            kind=require_enum(data, "kind", OperandKind, context),
            role=require(data, "role", str, context),
            handle_id=require_optional(data, "handle_id", str, context),
            external=external,
        )


_OPERATION_KEYS = ("name", "operands", "parameters")


@dataclass(frozen=True)
class Operation:
    """The operation that produced a handle, with all of its inputs.

    Attributes
    ----------
    name : OperationName
        Which operation ran.
    operands : tuple[Operand, ...]
        Inputs in a meaningful order.  For a difference, the first operand is
        the base and the rest are subtrahends.
    parameters : Mapping[str, Any]
        Everything else needed to reproduce the operation that is not itself a
        set: the Boolean query, the date bounds, the caps, the databases asked.
    """

    name: OperationName
    operands: tuple[Operand, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize parameters and check operand arity for set operations.

        Returns
        -------
        None

        Raises
        ------
        HandleSchemaError
            If a set operation has too few operands to be well defined.
        """
        object.__setattr__(self, "operands", tuple(self.operands))
        object.__setattr__(self, "parameters", dict(self.parameters))
        handle_count = sum(1 for operand in self.operands if operand.kind is OperandKind.HANDLE)
        if self.name in _MULTI_OPERAND_OPERATIONS and handle_count < 2:
            raise HandleSchemaError(
                f"Operation {self.name.value!r} needs at least two handle operands, "
                f"got {handle_count}"
            )
        if self.name in _SINGLE_OPERAND_OPERATIONS and handle_count != 1:
            raise HandleSchemaError(
                f"Operation {self.name.value!r} needs exactly one handle operand, "
                f"got {handle_count}"
            )

    def handle_operand_ids(self) -> list[str]:
        """List the identifiers of every handle this operation consumed.

        Returns
        -------
        list[str]
            Handle identifiers, in operand order, with duplicates preserved.
        """
        return [
            operand.handle_id
            for operand in self.operands
            if operand.kind is OperandKind.HANDLE and operand.handle_id is not None
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize this operation to a plain dictionary.

        Returns
        -------
        dict[str, Any]
            JSON-ready representation with parameter keys sorted for a stable
            content hash.
        """
        return {
            "name": self.name.value,
            "operands": [operand.to_dict() for operand in self.operands],
            "parameters": {key: self.parameters[key] for key in sorted(self.parameters)},
        }

    @classmethod
    def from_dict(cls, payload: Any, context: str) -> Operation:
        """Reconstruct an operation, refusing anything unrecognised.

        Parameters
        ----------
        payload : Any
            Mapping produced by :meth:`to_dict`.
        context : str
            Dotted path of the payload, used in error messages.

        Returns
        -------
        Operation
            Reconstructed operation.

        Raises
        ------
        HandleSchemaError
            If a key is missing, mistyped, or unrecognised.
        """
        data = require_mapping(payload, context)
        reject_unknown_keys(data, _OPERATION_KEYS, context)
        raw_operands = require(data, "operands", list, context)
        operands = tuple(
            Operand.from_dict(item, f"{context}.operands[{index}]")
            for index, item in enumerate(raw_operands)
        )
        parameters = require_mapping(
            require(data, "parameters", dict, context), f"{context}.parameters"
        )
        return cls(
            name=require_enum(data, "name", OperationName, context),
            operands=operands,
            parameters=dict(parameters),
        )


#: Operations that combine two or more sets.
_MULTI_OPERAND_OPERATIONS = frozenset(
    {OperationName.UNION, OperationName.DIFFERENCE, OperationName.INTERSECTION}
)

#: Operations that transform exactly one set.
_SINGLE_OPERAND_OPERATIONS = frozenset(
    {OperationName.RERANK, OperationName.FILTER, OperationName.DEDUPLICATE}
)


@dataclass(frozen=True)
class LineageEdge:
    """One dependency of a handle on another handle.

    Attributes
    ----------
    child : str
        Handle produced by the operation.
    parent : str
        Handle consumed by the operation.
    role : str
        Part the parent played, for example ``"base"`` or ``"subtrahend"``.
    operation : OperationName
        Operation that created the dependency.
    """

    child: str
    parent: str
    role: str
    operation: OperationName


@dataclass(frozen=True)
class LineageGraph:
    """The ancestry of one handle, as nodes and edges.

    Attributes
    ----------
    root : str
        Handle the graph was built for.
    nodes : tuple[str, ...]
        Every handle reachable from *root*, including *root* itself, in
        breadth-first order.
    edges : tuple[LineageEdge, ...]
        Dependencies between those handles.
    external : tuple[tuple[str, ExternalReference], ...]
        External snapshots the ancestry depended on, paired with the handle
        that recorded each one.
    missing : tuple[str, ...]
        Referenced handles that are no longer in the store.  Reported rather
        than skipped, because a lineage with a hole in it is a different fact
        from a complete one.
    """

    root: str
    nodes: tuple[str, ...]
    edges: tuple[LineageEdge, ...]
    external: tuple[tuple[str, ExternalReference], ...]
    missing: tuple[str, ...] = ()

    def parents_of(self, handle_id: str) -> list[str]:
        """List the direct parents of a handle within this graph.

        Parameters
        ----------
        handle_id : str
            Handle to look up.

        Returns
        -------
        list[str]
            Parent handle identifiers, in operand order.
        """
        return [edge.parent for edge in self.edges if edge.child == handle_id]


def _as_utc(value: datetime.datetime) -> datetime.datetime:
    """Normalize a datetime to an aware UTC datetime.

    Parameters
    ----------
    value : datetime.datetime
        Timestamp, naive or aware.

    Returns
    -------
    datetime.datetime
        Equivalent timestamp in UTC.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=datetime.UTC)
    return value.astimezone(datetime.UTC)


def _parse_timestamp(raw: str, context: str) -> datetime.datetime:
    """Parse an ISO-8601 timestamp, refusing anything unparsable.

    Parameters
    ----------
    raw : str
        ISO-8601 string.
    context : str
        Dotted path of the value, used in the error message.

    Returns
    -------
    datetime.datetime
        Parsed timestamp in UTC.

    Raises
    ------
    HandleSchemaError
        If *raw* is not a valid ISO-8601 timestamp.
    """
    try:
        return _as_utc(datetime.datetime.fromisoformat(raw))
    except ValueError as exc:
        raise HandleSchemaError(f"{context}: {raw!r} is not an ISO-8601 timestamp") from exc
