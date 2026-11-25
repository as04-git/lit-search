"""The serialized form of a handle, and the rules for trusting one.

A handle is an immutable, on-disk, content-addressed set of papers together
with the operation that produced it, the order claim it carries, and a slot
for its coverage ledger.

Content addressing
------------------

A handle's identifier is a BLAKE2b digest of its own canonical JSON, with the
identifier field excluded.  This buys three things at once:

* **Uniqueness** without a counter, a lock, or a central allocator, so two
  processes writing at the same moment cannot collide on a name.
* **Idempotent retry.**  Re-writing the same record produces the same bytes at
  the same path, so a crashed-and-restarted operation is a no-op rather than a
  duplicate.
* **Tamper and truncation detection.**  A file whose content no longer hashes
  to its own name is corrupt, and saying so is better than loading it.

The digest is taken over the *raw payload* on load, never over re-serialized
objects, so it does not depend on any round-trip being byte-exact.

Schema refusal
--------------

The envelope carries ``schema`` and ``schema_version``, and loading refuses on
anything it does not recognise: an unknown version, a missing required key, a
mistyped value, or an unrecognised extra key.  Refusing on *extra* keys is the
part that matters most.  A version-1 reader that quietly ignored a version-2
field would keep working while no longer meaning what it says, which is
exactly the silent-drift failure this store exists to avoid.  The consequence
is deliberate: adding a field requires bumping the version.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any

from lit_search.core.paper import Paper
from lit_search.exceptions import HandleSchemaError
from lit_search.store.lineage import Operation, _as_utc, _parse_timestamp
from lit_search.store.order import OrderRecord
from lit_search.store.strict import (
    reject_unknown_keys,
    require,
    require_mapping,
    require_optional,
)
from lit_search.utils.version import package_version

#: Marker identifying a payload as a handle written by this store.
SCHEMA_NAME = "lit-search/handle"

#: Schema version this module writes and is the only version it accepts.
#:
#: Bump this, and add a migration, for **any** change to the envelope: a new
#: key, a removed key, a changed type, or a changed meaning.  There is no
#: forward compatibility by omission here, by design.
SCHEMA_VERSION = 1

#: Envelope keys defined by :data:`SCHEMA_VERSION`.  Anything else is refused.
_ENVELOPE_KEYS = (
    "schema",
    "schema_version",
    "handle_id",
    "created_at",
    "created_by",
    "label",
    "operation",
    "order",
    "coverage",
    "members",
)

#: Prefix on every handle identifier, so an identifier is recognisable on
#: sight in a log line or a command.
HANDLE_ID_PREFIX = "h_"

#: Digest length in bytes.  Sixteen bytes gives a 32-character hex identifier:
#: long enough that a collision is not a practical concern, short enough that
#: prefix resolution keeps it typeable at a terminal.
_DIGEST_SIZE = 16


@dataclass(frozen=True)
class HandleRecord:
    """One immutable set of papers, addressed by a content hash.

    Attributes
    ----------
    operation : Operation
        What produced this set, including every operand it consumed.
    order : OrderRecord
        Whether the member sequence carries a ranking claim, and what backs it.
    members : tuple[Paper, ...]
        The papers.  Sequence is significant when *order* is ranked, and a
        stable arbitrary artifact otherwise.
    created_at : datetime.datetime
        When the handle was produced, in UTC.
    created_by : str
        Version of the package that produced it.
    label : str | None
        Optional human-readable name fixed at creation time.  Handles are
        immutable, so this cannot be edited later; renaming would need a
        separate mutable alias namespace, which is deliberately not built here
        because it would reintroduce the write conflicts immutability removes.
    coverage : Any
        Reserved slot for the per-source coverage ledger, owned by the
        coverage contract rather than by this module.  Stored and hashed
        opaquely.  ``None`` means "this handle carries no ledger", which is
        distinguishable from the key being absent, which is refused.
    """

    operation: Operation
    order: OrderRecord = field(default_factory=OrderRecord.unordered)
    members: tuple[Paper, ...] = ()
    created_at: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC)
    )
    created_by: str = field(default_factory=package_version)
    label: str | None = None
    coverage: Any = None

    def __post_init__(self) -> None:
        """Normalize members and check the order claim against them.

        Returns
        -------
        None

        Raises
        ------
        OrderSemanticsError
            If a ranked handle's score count does not match its member count.
        """
        object.__setattr__(self, "members", tuple(self.members))
        object.__setattr__(self, "created_at", _as_utc(self.created_at))
        self.order.validate_member_count(len(self.members))

    @property
    def handle_id(self) -> str:
        """Return this record's content address.

        Returns
        -------
        str
            Identifier of the form ``h_<32 hex characters>``.
        """
        return compute_handle_id(self._payload_without_id())

    def with_coverage(self, coverage: Any) -> HandleRecord:
        """Return a copy carrying a coverage ledger.

        Because the identifier is a content hash, the returned record has a
        **different** identifier.  That is the intended behaviour: attaching a
        ledger changes what the handle asserts, and an immutable handle cannot
        change what it asserts in place.

        Parameters
        ----------
        coverage : Any
            Coverage ledger payload, opaque to this module.

        Returns
        -------
        HandleRecord
            A new record.
        """
        return replace(self, coverage=coverage)

    def _payload_without_id(self) -> dict[str, Any]:
        """Build the canonical payload used to compute the identifier.

        Returns
        -------
        dict[str, Any]
            Envelope with every key except ``handle_id``.
        """
        return {
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "label": self.label,
            "operation": self.operation.to_dict(),
            "order": self.order.to_dict(),
            "coverage": self.coverage,
            "members": [member.to_dict() for member in self.members],
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize this handle to its on-disk representation.

        Returns
        -------
        dict[str, Any]
            Full envelope, identifier included.
        """
        payload = self._payload_without_id()
        payload["handle_id"] = compute_handle_id(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> HandleRecord:
        """Reconstruct a handle, refusing anything it cannot fully understand.

        Parameters
        ----------
        payload : Any
            Envelope produced by :meth:`to_dict`.

        Returns
        -------
        HandleRecord
            Reconstructed handle.

        Raises
        ------
        HandleSchemaError
            If the schema marker, version, keys, or types are not exactly what
            this version defines.
        ModelValidationError
            If a member paper cannot be reconstructed.
        """
        data = require_mapping(payload, "handle")
        _check_schema(data)
        reject_unknown_keys(data, _ENVELOPE_KEYS, "handle")

        if "coverage" not in data:
            raise HandleSchemaError("handle.coverage: required key is missing")

        raw_members = require(data, "members", list, "handle")
        members: list[Paper] = []
        for index, item in enumerate(raw_members):
            members.append(Paper.from_dict(dict(require_mapping(item, f"handle.members[{index}]"))))

        return cls(
            operation=Operation.from_dict(
                require(data, "operation", dict, "handle"), "handle.operation"
            ),
            order=OrderRecord.from_dict(require(data, "order", dict, "handle"), "handle.order"),
            members=tuple(members),
            created_at=_parse_timestamp(
                require(data, "created_at", str, "handle"), "handle.created_at"
            ),
            created_by=require(data, "created_by", str, "handle"),
            label=require_optional(data, "label", str, "handle"),
            coverage=data["coverage"],
        )


def _check_schema(data: Any) -> None:
    """Refuse a payload this reader is not built for.

    Parameters
    ----------
    data : Any
        Parsed envelope.

    Returns
    -------
    None

    Raises
    ------
    HandleSchemaError
        If the schema marker is wrong or the version is not
        :data:`SCHEMA_VERSION`.
    """
    schema = require(data, "schema", str, "handle")
    if schema != SCHEMA_NAME:
        raise HandleSchemaError(
            f"handle.schema: expected {SCHEMA_NAME!r}, got {schema!r}; this file was "
            "not written by the handle store"
        )
    version = require(data, "schema_version", int, "handle")
    if version != SCHEMA_VERSION:
        direction = "newer" if version > SCHEMA_VERSION else "older"
        raise HandleSchemaError(
            f"handle.schema_version: this handle uses schema version {version}, and "
            f"this reader only supports version {SCHEMA_VERSION}.  Refusing to load a "
            f"{direction} schema rather than guess at the fields it does or does not "
            "define; no migration is registered for it."
        )


def canonical_json(payload: Any) -> str:
    """Render a payload to the exact text form the identifier is computed over.

    Keys are sorted and separators are tight so that two logically identical
    payloads always produce identical bytes.

    Parameters
    ----------
    payload : Any
        JSON-serializable object.

    Returns
    -------
    str
        Canonical JSON text.
    """
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def compute_handle_id(payload_without_id: Any) -> str:
    """Compute the content address of a handle payload.

    Parameters
    ----------
    payload_without_id : Any
        Envelope with the ``handle_id`` key absent.

    Returns
    -------
    str
        Identifier of the form ``h_<32 hex characters>``.
    """
    digest = hashlib.blake2b(
        canonical_json(payload_without_id).encode("utf-8"), digest_size=_DIGEST_SIZE
    )
    return f"{HANDLE_ID_PREFIX}{digest.hexdigest()}"


def verify_payload_id(payload: Any) -> str:
    """Recompute a stored payload's identifier and check it against the file.

    Parameters
    ----------
    payload : Any
        Envelope as read from disk, identifier included.

    Returns
    -------
    str
        The verified identifier.

    Raises
    ------
    HandleSchemaError
        If the payload is not a mapping or has no identifier.
    HandleIntegrityError
        If the recomputed identifier does not match the stored one.
    """
    from lit_search.exceptions import HandleIntegrityError

    data = require_mapping(payload, "handle")
    stored = require(data, "handle_id", str, "handle")
    without_id = {key: value for key, value in data.items() if key != "handle_id"}
    recomputed = compute_handle_id(without_id)
    if recomputed != stored:
        raise HandleIntegrityError(
            f"Handle {stored} does not match its own content (recomputed {recomputed}); "
            "the file was truncated, hand-edited, or corrupted after it was written"
        )
    return stored
