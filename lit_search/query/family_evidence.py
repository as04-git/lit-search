"""Verify query-family evidence against immutable local artifacts.

Structural validation proves that evidence identifiers and derivations form a
sound graph.  This module proves the separate claim that a handle-member
locator still resolves to the exact paper field from which a query object was
derived.  Verification is deliberately performed before any search request.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from lit_search.core.paper import Paper
from lit_search.core.query_family import EvidenceKind, QueryFamilyRevision, canonical_json
from lit_search.store.store import HandleStore

_DIGEST_SIZE = 16


class QueryFamilyEvidenceError(ValueError):
    """A pinned query-family evidence locator cannot be reproduced."""


def _digest(prefix: str, value: Any) -> str:
    raw = canonical_json(value).encode("utf-8")
    return f"{prefix}_{hashlib.blake2b(raw, digest_size=_DIGEST_SIZE).hexdigest()}"


def member_evidence_digest(paper: Paper) -> str:
    """Return the content digest used to pin one persisted handle member.

    Handles reconstruct :class:`Paper` instances on read.  Canonicalizing
    through that public round trip prevents a constructor-only ``None`` versus
    stored empty-string normalization from making a just-written member
    impossible to cite.
    """

    persisted = Paper.from_dict(paper.to_dict()).to_dict()
    return _digest("qfm", persisted)


def field_evidence_digest(value: Any) -> str:
    """Return the content digest used to pin one serialized field value."""

    return _digest("qfv", value)


@dataclass(frozen=True, slots=True)
class HandleEvidenceVerifier:
    """Resolve every handle-member evidence locator through a handle store."""

    store: HandleStore

    def verify(self, revision: QueryFamilyRevision) -> None:
        """Verify pinned member and field digests, failing before execution."""

        for evidence in revision.evidence:
            if evidence.kind is not EvidenceKind.HANDLE_MEMBER:
                continue
            # The semantic validator requires all four fields for this kind.
            # Keep the checks here too so direct use never turns malformed
            # evidence into an AttributeError or an accidental store lookup.
            if (
                evidence.handle_id is None
                or evidence.member_digest is None
                or evidence.field is None
                or evidence.field_value_digest is None
            ):
                raise QueryFamilyEvidenceError(
                    f"handle evidence {evidence.evidence_id!r} is missing its pinned locator"
                )
            try:
                handle_id = self.store.resolve(evidence.handle_id)
                record = self.store.get(handle_id)
            except Exception as error:
                raise QueryFamilyEvidenceError(
                    f"handle evidence {evidence.evidence_id!r} cannot resolve "
                    f"{evidence.handle_id!r}: {type(error).__name__}: {error}"
                ) from error
            if handle_id != evidence.handle_id:
                raise QueryFamilyEvidenceError(
                    f"handle evidence {evidence.evidence_id!r} pins prefix "
                    f"{evidence.handle_id!r}, not full immutable identifier {handle_id}"
                )

            matches = [
                member
                for member in record.members
                if member_evidence_digest(member) == evidence.member_digest
            ]
            if not matches:
                raise QueryFamilyEvidenceError(
                    f"handle evidence {evidence.evidence_id!r} member digest "
                    f"{evidence.member_digest!r} is absent from {handle_id}"
                )
            if len(matches) > 1:
                raise QueryFamilyEvidenceError(
                    f"handle evidence {evidence.evidence_id!r} member digest is ambiguous "
                    f"within {handle_id}; {len(matches)} byte-equivalent members exist"
                )

            payload = matches[0].to_dict()
            if evidence.field not in payload:
                raise QueryFamilyEvidenceError(
                    f"handle evidence {evidence.evidence_id!r} names unknown Paper field "
                    f"{evidence.field!r}"
                )
            actual = field_evidence_digest(payload[evidence.field])
            if actual != evidence.field_value_digest:
                raise QueryFamilyEvidenceError(
                    f"handle evidence {evidence.evidence_id!r} field digest changed: "
                    f"stored {evidence.field_value_digest}, observed {actual}"
                )


__all__ = [
    "HandleEvidenceVerifier",
    "QueryFamilyEvidenceError",
    "field_evidence_digest",
    "member_evidence_digest",
]
