"""Typed, immutable query-family specifications.

The ordinary :mod:`lit_search.core.query` tree is an execution syntax: its
terminal nodes contain literal strings.  A query family is the semantic layer
above that syntax.  Executable expressions may refer only to accepted term
identifiers, and every accepted object records the derivation that introduced
it.  Model- or extractor-produced proposals are represented separately and
cannot appear in an executable expression.

The serialized form is strict and versioned.  Unknown fields are refused, and
the revision identifier is a digest of canonical semantic JSON rather than a
mutable name or timestamp.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, TypeAlias

from lit_search.core.query import FilterCode

QUERY_FAMILY_SCHEMA = "lit-search/query-family"
QUERY_FAMILY_SCHEMA_VERSION = 1
_DIGEST_SIZE = 16


class QueryFamilySchemaError(ValueError):
    """A serialized query family cannot be understood exactly."""


class TermRole(StrEnum):
    """How an accepted term may participate in an executable expression."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    DISAMBIGUATOR = "disambiguator"


class AmbiguityStrategy(StrEnum):
    """How one variant explicitly treats a known ambiguous term."""

    ALLOW_WITH_RATIONALE = "allow_with_rationale"
    REQUIRE_CONTEXT = "require_context"
    EXCLUDE_CONTEXT = "exclude_context"


class TranslationMode(StrEnum):
    """Whether a source-specific expression preserves the variant semantics."""

    EQUIVALENT = "equivalent"
    DEGRADED = "degraded"


class EvidenceKind(StrEnum):
    """Kinds of evidence from which query objects can be derived."""

    USER = "user"
    MANUAL = "manual"
    HANDLE_MEMBER = "handle_member"
    CONTROLLED_VOCABULARY = "controlled_vocabulary"


def canonical_json(value: Any) -> str:
    """Serialize a semantic value deterministically."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _semantic_digest(prefix: str, value: Any) -> str:
    digest = hashlib.blake2b(
        canonical_json(value).encode("utf-8"), digest_size=_DIGEST_SIZE
    ).hexdigest()
    return f"{prefix}_{digest}"


def _strict_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise QueryFamilySchemaError(f"{context}: expected an object with string keys")
    return value


def _strict_keys(data: dict[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(data))
    unknown = sorted(set(data) - expected)
    if missing:
        raise QueryFamilySchemaError(f"{context}: missing required key(s) {missing}")
    if unknown:
        raise QueryFamilySchemaError(
            f"{context}: unrecognised key(s) {unknown}; schema drift is refused"
        )


def _string(data: dict[str, Any], key: str, context: str) -> str:
    value = data[key]
    if not isinstance(value, str):
        raise QueryFamilySchemaError(f"{context}.{key}: expected str, got {type(value).__name__}")
    return value


def _optional_string(data: dict[str, Any], key: str, context: str) -> str | None:
    value = data[key]
    if value is not None and not isinstance(value, str):
        raise QueryFamilySchemaError(
            f"{context}.{key}: expected str or null, got {type(value).__name__}"
        )
    return value


def _strings(data: dict[str, Any], key: str, context: str) -> tuple[str, ...]:
    value = data[key]
    if not isinstance(value, list):
        raise QueryFamilySchemaError(f"{context}.{key}: expected list, got {type(value).__name__}")
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise QueryFamilySchemaError(
                f"{context}.{key}[{index}]: expected str, got {type(item).__name__}"
            )
    return tuple(value)


def _optional_strings(data: dict[str, Any], key: str, context: str) -> tuple[str, ...] | None:
    if data[key] is None:
        return None
    return _strings(data, key, context)


def _enum(data: dict[str, Any], key: str, enum_type: type[StrEnum], context: str) -> Any:
    raw = _string(data, key, context)
    try:
        return enum_type(raw)
    except ValueError as error:
        allowed = sorted(item.value for item in enum_type)
        raise QueryFamilySchemaError(f"{context}.{key}: {raw!r} is not one of {allowed}") from error


@dataclass(frozen=True, slots=True)
class TermRef:
    """Reference an accepted term in an executable field."""

    term_id: str
    field: FilterCode = FilterCode.TITLE_ABSTRACT

    kind: ClassVar[str] = "term"

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-ready representation."""

        return {"kind": self.kind, "term_id": self.term_id, "field": self.field.value}


@dataclass(frozen=True, slots=True)
class AllOf:
    """Require every child expression."""

    children: tuple[FamilyExpression, ...]
    kind: ClassVar[str] = "all"

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-ready representation."""

        return {
            "kind": self.kind,
            "children": sorted((child.to_dict() for child in self.children), key=canonical_json),
        }


@dataclass(frozen=True, slots=True)
class AnyOf:
    """Accept any child expression."""

    children: tuple[FamilyExpression, ...]
    kind: ClassVar[str] = "any"

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-ready representation."""

        return {
            "kind": self.kind,
            "children": sorted((child.to_dict() for child in self.children), key=canonical_json),
        }


@dataclass(frozen=True, slots=True)
class AndNot:
    """Require *include* while explicitly excluding *exclude*."""

    include: FamilyExpression
    exclude: FamilyExpression
    kind: ClassVar[str] = "and_not"

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-ready representation."""

        return {
            "kind": self.kind,
            "include": self.include.to_dict(),
            "exclude": self.exclude.to_dict(),
        }


FamilyExpression: TypeAlias = TermRef | AllOf | AnyOf | AndNot


def expression_from_dict(value: Any, context: str = "expression") -> FamilyExpression:
    """Read one strict tagged expression."""

    data = _strict_mapping(value, context)
    kind = data.get("kind")
    if kind == TermRef.kind:
        _strict_keys(data, {"kind", "term_id", "field"}, context)
        raw_field = _string(data, "field", context)
        try:
            field = FilterCode(raw_field)
        except ValueError as error:
            raise QueryFamilySchemaError(
                f"{context}.field: unknown filter {raw_field!r}"
            ) from error
        return TermRef(term_id=_string(data, "term_id", context), field=field)
    if kind in {AllOf.kind, AnyOf.kind}:
        _strict_keys(data, {"kind", "children"}, context)
        raw_children = data["children"]
        if not isinstance(raw_children, list):
            raise QueryFamilySchemaError(f"{context}.children: expected list")
        children = tuple(
            expression_from_dict(child, f"{context}.children[{index}]")
            for index, child in enumerate(raw_children)
        )
        return AllOf(children) if kind == AllOf.kind else AnyOf(children)
    if kind == AndNot.kind:
        _strict_keys(data, {"kind", "include", "exclude"}, context)
        return AndNot(
            include=expression_from_dict(data["include"], f"{context}.include"),
            exclude=expression_from_dict(data["exclude"], f"{context}.exclude"),
        )
    raise QueryFamilySchemaError(f"{context}.kind: unknown expression kind {kind!r}")


@dataclass(frozen=True, slots=True)
class Concept:
    """A semantic component of the research question."""

    concept_id: str
    label: str
    definition: str
    derivation_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized concept."""

        return {
            "concept_id": self.concept_id,
            "label": self.label,
            "definition": self.definition,
            "derivation_ids": sorted(self.derivation_ids),
        }

    @classmethod
    def from_dict(cls, value: Any, context: str) -> Concept:
        """Read a strict concept payload."""

        data = _strict_mapping(value, context)
        _strict_keys(data, {"concept_id", "label", "definition", "derivation_ids"}, context)
        return cls(
            _string(data, "concept_id", context),
            _string(data, "label", context),
            _string(data, "definition", context),
            _strings(data, "derivation_ids", context),
        )


@dataclass(frozen=True, slots=True)
class TerminologyFamily:
    """A coherent community vocabulary linked to one or more concepts."""

    family_id: str
    label: str
    community: str | None
    concept_ids: tuple[str, ...]
    derivation_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized terminology family."""

        return {
            "family_id": self.family_id,
            "label": self.label,
            "community": self.community,
            "concept_ids": sorted(self.concept_ids),
            "derivation_ids": sorted(self.derivation_ids),
        }

    @classmethod
    def from_dict(cls, value: Any, context: str) -> TerminologyFamily:
        """Read a strict terminology-family payload."""

        data = _strict_mapping(value, context)
        expected = {"family_id", "label", "community", "concept_ids", "derivation_ids"}
        _strict_keys(data, expected, context)
        return cls(
            _string(data, "family_id", context),
            _string(data, "label", context),
            _optional_string(data, "community", context),
            _strings(data, "concept_ids", context),
            _strings(data, "derivation_ids", context),
        )


@dataclass(frozen=True, slots=True)
class AcceptedSearchTerm:
    """A reviewed term that may be referenced by executable expressions."""

    term_id: str
    text: str
    terminology_family_id: str
    concept_ids: tuple[str, ...]
    role: TermRole
    derivation_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized accepted term."""

        return {
            "term_id": self.term_id,
            "text": self.text,
            "terminology_family_id": self.terminology_family_id,
            "concept_ids": sorted(self.concept_ids),
            "role": self.role.value,
            "derivation_ids": sorted(self.derivation_ids),
        }

    @classmethod
    def from_dict(cls, value: Any, context: str) -> AcceptedSearchTerm:
        """Read a strict accepted-term payload."""

        data = _strict_mapping(value, context)
        expected = {
            "term_id",
            "text",
            "terminology_family_id",
            "concept_ids",
            "role",
            "derivation_ids",
        }
        _strict_keys(data, expected, context)
        return cls(
            _string(data, "term_id", context),
            _string(data, "text", context),
            _string(data, "terminology_family_id", context),
            _strings(data, "concept_ids", context),
            _enum(data, "role", TermRole, context),
            _strings(data, "derivation_ids", context),
        )


@dataclass(frozen=True, slots=True)
class TermProposal:
    """A non-executable candidate produced by a person, model, or extractor."""

    proposal_id: str
    text: str
    rationale: str
    evidence_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized proposal."""

        return {
            "proposal_id": self.proposal_id,
            "text": self.text,
            "rationale": self.rationale,
            "evidence_ids": sorted(self.evidence_ids),
        }

    @classmethod
    def from_dict(cls, value: Any, context: str) -> TermProposal:
        """Read a strict proposal payload."""

        data = _strict_mapping(value, context)
        _strict_keys(data, {"proposal_id", "text", "rationale", "evidence_ids"}, context)
        return cls(
            _string(data, "proposal_id", context),
            _string(data, "text", context),
            _string(data, "rationale", context),
            _strings(data, "evidence_ids", context),
        )


@dataclass(frozen=True, slots=True)
class Ambiguity:
    """A known competing sense of an accepted term."""

    ambiguity_id: str
    term_id: str
    competing_sense: str
    derivation_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized ambiguity."""

        return {
            "ambiguity_id": self.ambiguity_id,
            "term_id": self.term_id,
            "competing_sense": self.competing_sense,
            "derivation_ids": sorted(self.derivation_ids),
        }

    @classmethod
    def from_dict(cls, value: Any, context: str) -> Ambiguity:
        """Read a strict ambiguity payload."""

        data = _strict_mapping(value, context)
        expected = {"ambiguity_id", "term_id", "competing_sense", "derivation_ids"}
        _strict_keys(data, expected, context)
        return cls(
            _string(data, "ambiguity_id", context),
            _string(data, "term_id", context),
            _string(data, "competing_sense", context),
            _strings(data, "derivation_ids", context),
        )


@dataclass(frozen=True, slots=True)
class AmbiguityResolution:
    """One explicit, provenance-bearing resolution of an ambiguity."""

    resolution_id: str
    ambiguity_id: str
    strategy: AmbiguityStrategy
    rationale: str
    context_term_ids: tuple[str, ...]
    derivation_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized resolution."""

        return {
            "resolution_id": self.resolution_id,
            "ambiguity_id": self.ambiguity_id,
            "strategy": self.strategy.value,
            "rationale": self.rationale,
            "context_term_ids": sorted(self.context_term_ids),
            "derivation_ids": sorted(self.derivation_ids),
        }

    @classmethod
    def from_dict(cls, value: Any, context: str) -> AmbiguityResolution:
        """Read a strict resolution payload."""

        data = _strict_mapping(value, context)
        expected = {
            "resolution_id",
            "ambiguity_id",
            "strategy",
            "rationale",
            "context_term_ids",
            "derivation_ids",
        }
        _strict_keys(data, expected, context)
        return cls(
            _string(data, "resolution_id", context),
            _string(data, "ambiguity_id", context),
            _enum(data, "strategy", AmbiguityStrategy, context),
            _string(data, "rationale", context),
            _strings(data, "context_term_ids", context),
            _strings(data, "derivation_ids", context),
        )


@dataclass(frozen=True, slots=True)
class SourceTranslation:
    """A semantic expression explicitly translated for one source."""

    translation_id: str
    source: str
    expression: FamilyExpression
    mode: TranslationMode
    rationale: str
    derivation_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized source translation."""

        return {
            "translation_id": self.translation_id,
            "source": self.source,
            "expression": self.expression.to_dict(),
            "mode": self.mode.value,
            "rationale": self.rationale,
            "derivation_ids": sorted(self.derivation_ids),
        }

    @classmethod
    def from_dict(cls, value: Any, context: str) -> SourceTranslation:
        """Read a strict source-translation payload."""

        data = _strict_mapping(value, context)
        expected = {
            "translation_id",
            "source",
            "expression",
            "mode",
            "rationale",
            "derivation_ids",
        }
        _strict_keys(data, expected, context)
        return cls(
            _string(data, "translation_id", context),
            _string(data, "source", context),
            expression_from_dict(data["expression"], f"{context}.expression"),
            _enum(data, "mode", TranslationMode, context),
            _string(data, "rationale", context),
            _strings(data, "derivation_ids", context),
        )


@dataclass(frozen=True, slots=True)
class QueryVariant:
    """One explicitly reviewed executable lane in the query family."""

    variant_id: str
    label: str
    purpose: str
    expression: FamilyExpression
    ambiguity_resolution_ids: tuple[str, ...]
    source_translation_ids: tuple[str, ...]
    applies_to_sources: tuple[str, ...] | None
    derivation_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized query variant."""

        return {
            "variant_id": self.variant_id,
            "label": self.label,
            "purpose": self.purpose,
            "expression": self.expression.to_dict(),
            "ambiguity_resolution_ids": sorted(self.ambiguity_resolution_ids),
            "source_translation_ids": sorted(self.source_translation_ids),
            "applies_to_sources": (
                None if self.applies_to_sources is None else sorted(self.applies_to_sources)
            ),
            "derivation_ids": sorted(self.derivation_ids),
        }

    @classmethod
    def from_dict(cls, value: Any, context: str) -> QueryVariant:
        """Read a strict query-variant payload."""

        data = _strict_mapping(value, context)
        expected = {
            "variant_id",
            "label",
            "purpose",
            "expression",
            "ambiguity_resolution_ids",
            "source_translation_ids",
            "applies_to_sources",
            "derivation_ids",
        }
        _strict_keys(data, expected, context)
        return cls(
            _string(data, "variant_id", context),
            _string(data, "label", context),
            _string(data, "purpose", context),
            expression_from_dict(data["expression"], f"{context}.expression"),
            _strings(data, "ambiguity_resolution_ids", context),
            _strings(data, "source_translation_ids", context),
            _optional_strings(data, "applies_to_sources", context),
            _strings(data, "derivation_ids", context),
        )


@dataclass(frozen=True, slots=True)
class Evidence:
    """A typed evidence locator used by one or more derivations."""

    evidence_id: str
    kind: EvidenceKind
    statement: str
    handle_id: str | None = None
    member_digest: str | None = None
    field: str | None = None
    field_value_digest: str | None = None
    authority: str | None = None
    authority_version: str | None = None
    authority_identifier: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized evidence locator."""

        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind.value,
            "statement": self.statement,
            "handle_id": self.handle_id,
            "member_digest": self.member_digest,
            "field": self.field,
            "field_value_digest": self.field_value_digest,
            "authority": self.authority,
            "authority_version": self.authority_version,
            "authority_identifier": self.authority_identifier,
        }

    @classmethod
    def from_dict(cls, value: Any, context: str) -> Evidence:
        """Read a strict evidence payload."""

        data = _strict_mapping(value, context)
        expected = {
            "evidence_id",
            "kind",
            "statement",
            "handle_id",
            "member_digest",
            "field",
            "field_value_digest",
            "authority",
            "authority_version",
            "authority_identifier",
        }
        _strict_keys(data, expected, context)
        return cls(
            evidence_id=_string(data, "evidence_id", context),
            kind=_enum(data, "kind", EvidenceKind, context),
            statement=_string(data, "statement", context),
            handle_id=_optional_string(data, "handle_id", context),
            member_digest=_optional_string(data, "member_digest", context),
            field=_optional_string(data, "field", context),
            field_value_digest=_optional_string(data, "field_value_digest", context),
            authority=_optional_string(data, "authority", context),
            authority_version=_optional_string(data, "authority_version", context),
            authority_identifier=_optional_string(data, "authority_identifier", context),
        )


@dataclass(frozen=True, slots=True)
class Derivation:
    """An explicit transformation from evidence/objects to query objects."""

    derivation_id: str
    relation: str
    input_object_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    output_object_ids: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized derivation."""

        return {
            "derivation_id": self.derivation_id,
            "relation": self.relation,
            "input_object_ids": sorted(self.input_object_ids),
            "evidence_ids": sorted(self.evidence_ids),
            "output_object_ids": sorted(self.output_object_ids),
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, value: Any, context: str) -> Derivation:
        """Read a strict derivation payload."""

        data = _strict_mapping(value, context)
        expected = {
            "derivation_id",
            "relation",
            "input_object_ids",
            "evidence_ids",
            "output_object_ids",
            "rationale",
        }
        _strict_keys(data, expected, context)
        return cls(
            _string(data, "derivation_id", context),
            _string(data, "relation", context),
            _strings(data, "input_object_ids", context),
            _strings(data, "evidence_ids", context),
            _strings(data, "output_object_ids", context),
            _string(data, "rationale", context),
        )


@dataclass(frozen=True, slots=True)
class QueryFamilyRevision:
    """One immutable, content-addressed revision of a query family."""

    research_question: str
    parent_revision_ids: tuple[str, ...] = ()
    concepts: tuple[Concept, ...] = ()
    terminology_families: tuple[TerminologyFamily, ...] = ()
    accepted_terms: tuple[AcceptedSearchTerm, ...] = ()
    proposals: tuple[TermProposal, ...] = ()
    ambiguities: tuple[Ambiguity, ...] = ()
    ambiguity_resolutions: tuple[AmbiguityResolution, ...] = ()
    source_translations: tuple[SourceTranslation, ...] = ()
    variants: tuple[QueryVariant, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    derivations: tuple[Derivation, ...] = ()

    def semantic_dict(self) -> dict[str, Any]:
        """Return the canonical semantic payload hashed by :attr:`revision_id`."""

        def ordered(items: tuple[Any, ...], id_name: str) -> list[dict[str, Any]]:
            return [
                item.to_dict() for item in sorted(items, key=lambda item: getattr(item, id_name))
            ]

        return {
            "schema": QUERY_FAMILY_SCHEMA,
            "schema_version": QUERY_FAMILY_SCHEMA_VERSION,
            "research_question": self.research_question,
            "parent_revision_ids": sorted(self.parent_revision_ids),
            "concepts": ordered(self.concepts, "concept_id"),
            "terminology_families": ordered(self.terminology_families, "family_id"),
            "accepted_terms": ordered(self.accepted_terms, "term_id"),
            "proposals": ordered(self.proposals, "proposal_id"),
            "ambiguities": ordered(self.ambiguities, "ambiguity_id"),
            "ambiguity_resolutions": ordered(self.ambiguity_resolutions, "resolution_id"),
            "source_translations": ordered(self.source_translations, "translation_id"),
            "variants": ordered(self.variants, "variant_id"),
            "evidence": ordered(self.evidence, "evidence_id"),
            "derivations": ordered(self.derivations, "derivation_id"),
        }

    @property
    def revision_id(self) -> str:
        """Return the deterministic semantic content address."""

        return _semantic_digest("qf", self.semantic_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the strict serialized revision, identifier included."""

        return {**self.semantic_dict(), "revision_id": self.revision_id}

    @classmethod
    def from_dict(cls, value: Any) -> QueryFamilyRevision:
        """Read and integrity-check one strict version-1 revision."""

        context = "query_family"
        data = _strict_mapping(value, context)
        expected = {
            "schema",
            "schema_version",
            "revision_id",
            "research_question",
            "parent_revision_ids",
            "concepts",
            "terminology_families",
            "accepted_terms",
            "proposals",
            "ambiguities",
            "ambiguity_resolutions",
            "source_translations",
            "variants",
            "evidence",
            "derivations",
        }
        _strict_keys(data, expected, context)
        if _string(data, "schema", context) != QUERY_FAMILY_SCHEMA:
            raise QueryFamilySchemaError("query_family.schema: unknown schema")
        version = data["schema_version"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise QueryFamilySchemaError("query_family.schema_version: expected int")
        if version != QUERY_FAMILY_SCHEMA_VERSION:
            raise QueryFamilySchemaError(
                f"query_family.schema_version: unsupported version {version}"
            )

        def objects(key: str, reader: Any) -> tuple[Any, ...]:
            raw = data[key]
            if not isinstance(raw, list):
                raise QueryFamilySchemaError(f"query_family.{key}: expected list")
            return tuple(
                reader(item, f"query_family.{key}[{index}]") for index, item in enumerate(raw)
            )

        revision = cls(
            research_question=_string(data, "research_question", context),
            parent_revision_ids=_strings(data, "parent_revision_ids", context),
            concepts=objects("concepts", Concept.from_dict),
            terminology_families=objects("terminology_families", TerminologyFamily.from_dict),
            accepted_terms=objects("accepted_terms", AcceptedSearchTerm.from_dict),
            proposals=objects("proposals", TermProposal.from_dict),
            ambiguities=objects("ambiguities", Ambiguity.from_dict),
            ambiguity_resolutions=objects("ambiguity_resolutions", AmbiguityResolution.from_dict),
            source_translations=objects("source_translations", SourceTranslation.from_dict),
            variants=objects("variants", QueryVariant.from_dict),
            evidence=objects("evidence", Evidence.from_dict),
            derivations=objects("derivations", Derivation.from_dict),
        )
        stored_id = _string(data, "revision_id", context)
        if stored_id != revision.revision_id:
            raise QueryFamilySchemaError(
                f"query_family.revision_id: stored {stored_id}, recomputed {revision.revision_id}"
            )
        return revision


__all__ = [
    "QUERY_FAMILY_SCHEMA",
    "QUERY_FAMILY_SCHEMA_VERSION",
    "AcceptedSearchTerm",
    "AllOf",
    "Ambiguity",
    "AmbiguityResolution",
    "AmbiguityStrategy",
    "AndNot",
    "AnyOf",
    "Concept",
    "Derivation",
    "Evidence",
    "EvidenceKind",
    "FamilyExpression",
    "QueryFamilyRevision",
    "QueryFamilySchemaError",
    "QueryVariant",
    "SourceTranslation",
    "TermProposal",
    "TermRef",
    "TermRole",
    "TerminologyFamily",
    "TranslationMode",
    "canonical_json",
    "expression_from_dict",
]
