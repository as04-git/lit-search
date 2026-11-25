"""Whole-graph validation for immutable query-family revisions.

Validation happens before compilation or source execution.  It is deliberately
global: a locally well-formed term is not executable when its family is
missing, its derivation does not actually name it as an output, or a variant
uses its ambiguous sense without recording a resolution.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from lit_search.core.query_family import (
    AcceptedSearchTerm,
    AllOf,
    Ambiguity,
    AmbiguityResolution,
    AmbiguityStrategy,
    AnyOf,
    Derivation,
    Evidence,
    EvidenceKind,
    FamilyExpression,
    QueryFamilyRevision,
    QueryVariant,
    SourceTranslation,
    TermRef,
    TermRole,
    TranslationMode,
)


class QueryFamilyValidationError(ValueError):
    """A query family is structurally valid JSON but semantically unsound."""


_HANDLE_ID_RE = re.compile(r"h_[0-9a-f]{32}\Z")
_MEMBER_DIGEST_RE = re.compile(r"qfm_[0-9a-f]{32}\Z")
_FIELD_DIGEST_RE = re.compile(r"qfv_[0-9a-f]{32}\Z")


def _require_text(value: str, context: str) -> None:
    if not value.strip():
        raise QueryFamilyValidationError(f"{context} must not be blank")


def _unique(items: Iterable[tuple[str, object]], context: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for object_id, item in items:
        _require_text(object_id, f"{context} identifier")
        if object_id in result:
            raise QueryFamilyValidationError(f"duplicate {context} identifier {object_id!r}")
        result[object_id] = item
    return result


def _all_object_items(revision: QueryFamilyRevision) -> list[tuple[str, object]]:
    return [
        *((item.concept_id, item) for item in revision.concepts),
        *((item.family_id, item) for item in revision.terminology_families),
        *((item.term_id, item) for item in revision.accepted_terms),
        *((item.proposal_id, item) for item in revision.proposals),
        *((item.ambiguity_id, item) for item in revision.ambiguities),
        *((item.resolution_id, item) for item in revision.ambiguity_resolutions),
        *((item.translation_id, item) for item in revision.source_translations),
        *((item.variant_id, item) for item in revision.variants),
    ]


def _term_refs(expression: FamilyExpression) -> tuple[TermRef, ...]:
    if isinstance(expression, TermRef):
        return (expression,)
    if isinstance(expression, AllOf | AnyOf):
        return tuple(ref for child in expression.children for ref in _term_refs(child))
    return (*_term_refs(expression.include), *_term_refs(expression.exclude))


def _validate_expression_shape(expression: FamilyExpression, context: str) -> None:
    if isinstance(expression, TermRef):
        _require_text(expression.term_id, f"{context}.term_id")
        return
    if isinstance(expression, AllOf | AnyOf):
        if len(expression.children) < 2:
            raise QueryFamilyValidationError(
                f"{context} {expression.kind!r} needs at least two children"
            )
        for index, child in enumerate(expression.children):
            _validate_expression_shape(child, f"{context}.children[{index}]")
        return
    _validate_expression_shape(expression.include, f"{context}.include")
    _validate_expression_shape(expression.exclude, f"{context}.exclude")


def _validate_polarity(
    expression: FamilyExpression,
    terms: dict[str, AcceptedSearchTerm],
    context: str,
    *,
    exclusion: bool = False,
) -> None:
    if isinstance(expression, TermRef):
        term = terms.get(expression.term_id)
        if term is None:
            raise QueryFamilyValidationError(
                f"{context} references {expression.term_id!r}, which is not an accepted term; "
                "proposals and literal strings are not executable"
            )
        if term.role is TermRole.NEGATIVE and not exclusion:
            raise QueryFamilyValidationError(
                f"negative term {term.term_id!r} appears outside an explicit and_not exclusion"
            )
        if exclusion and term.role is not TermRole.NEGATIVE:
            raise QueryFamilyValidationError(
                f"{context} excludes {term.term_id!r}, but its role is {term.role.value!r}; "
                "exclusion terms must be accepted explicitly as negative"
            )
        return
    if isinstance(expression, AllOf | AnyOf):
        for index, child in enumerate(expression.children):
            _validate_polarity(child, terms, f"{context}.children[{index}]", exclusion=exclusion)
        return
    _validate_polarity(expression.include, terms, f"{context}.include", exclusion=exclusion)
    _validate_polarity(expression.exclude, terms, f"{context}.exclude", exclusion=True)


def _validate_evidence(evidence: Evidence) -> None:
    _require_text(evidence.evidence_id, "evidence_id")
    _require_text(evidence.statement, f"evidence {evidence.evidence_id!r} statement")
    handle_fields = (
        evidence.handle_id,
        evidence.member_digest,
        evidence.field,
        evidence.field_value_digest,
    )
    authority_fields = (
        evidence.authority,
        evidence.authority_version,
        evidence.authority_identifier,
    )
    if evidence.kind is EvidenceKind.HANDLE_MEMBER:
        if any(value is None or not value.strip() for value in handle_fields):
            raise QueryFamilyValidationError(
                f"handle-member evidence {evidence.evidence_id!r} must pin handle, member, "
                "field, and field-value digests"
            )
        if any(value is not None for value in authority_fields):
            raise QueryFamilyValidationError(
                f"handle-member evidence {evidence.evidence_id!r} carries authority fields"
            )
        if not _HANDLE_ID_RE.fullmatch(evidence.handle_id or ""):
            raise QueryFamilyValidationError(
                f"handle-member evidence {evidence.evidence_id!r} must pin a full immutable "
                "h_<32 hex> handle identifier, not a mutable or prefix locator"
            )
        if not _MEMBER_DIGEST_RE.fullmatch(evidence.member_digest or ""):
            raise QueryFamilyValidationError(
                f"handle-member evidence {evidence.evidence_id!r} has an invalid member digest"
            )
        if not _FIELD_DIGEST_RE.fullmatch(evidence.field_value_digest or ""):
            raise QueryFamilyValidationError(
                f"handle-member evidence {evidence.evidence_id!r} has an invalid field digest"
            )
    elif evidence.kind is EvidenceKind.CONTROLLED_VOCABULARY:
        if any(value is None or not value.strip() for value in authority_fields):
            raise QueryFamilyValidationError(
                f"controlled-vocabulary evidence {evidence.evidence_id!r} must pin authority, "
                "version, and identifier"
            )
        if any(value is not None for value in handle_fields):
            raise QueryFamilyValidationError(
                f"controlled-vocabulary evidence {evidence.evidence_id!r} carries handle fields"
            )
    elif any(value is not None for value in (*handle_fields, *authority_fields)):
        raise QueryFamilyValidationError(
            f"{evidence.kind.value} evidence {evidence.evidence_id!r} carries foreign locator fields"
        )


def _validate_derivation_graph(
    revision: QueryFamilyRevision,
    objects: dict[str, object],
    evidence: dict[str, Evidence],
) -> dict[str, Derivation]:
    raw_derivations = _unique(
        ((item.derivation_id, item) for item in revision.derivations), "derivation"
    )
    derivations = {
        object_id: item
        for object_id, item in raw_derivations.items()
        if isinstance(item, Derivation)
    }
    for derivation in revision.derivations:
        _require_text(derivation.relation, f"derivation {derivation.derivation_id!r} relation")
        _require_text(derivation.rationale, f"derivation {derivation.derivation_id!r} rationale")
        if not derivation.evidence_ids:
            raise QueryFamilyValidationError(
                f"derivation {derivation.derivation_id!r} has no evidence"
            )
        if not derivation.output_object_ids:
            raise QueryFamilyValidationError(
                f"derivation {derivation.derivation_id!r} has no outputs"
            )
        for evidence_id in derivation.evidence_ids:
            if evidence_id not in evidence:
                raise QueryFamilyValidationError(
                    f"derivation {derivation.derivation_id!r} references unknown evidence "
                    f"{evidence_id!r}"
                )
        for object_id in (*derivation.input_object_ids, *derivation.output_object_ids):
            if object_id not in objects:
                raise QueryFamilyValidationError(
                    f"derivation {derivation.derivation_id!r} references unknown object "
                    f"{object_id!r}"
                )

    dependencies: dict[str, set[str]] = {object_id: set() for object_id in objects}
    for derivation in revision.derivations:
        for output in derivation.output_object_ids:
            dependencies[output].update(derivation.input_object_ids)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(object_id: str) -> None:
        if object_id in visiting:
            raise QueryFamilyValidationError(
                f"derivation graph contains a cycle through {object_id!r}"
            )
        if object_id in visited:
            return
        visiting.add(object_id)
        for dependency in dependencies[object_id]:
            visit(dependency)
        visiting.remove(object_id)
        visited.add(object_id)

    for object_id in objects:
        visit(object_id)
    return derivations


def _require_provenance(
    object_id: str,
    derivation_ids: tuple[str, ...],
    derivations: dict[str, Derivation],
) -> None:
    if not derivation_ids:
        raise QueryFamilyValidationError(
            f"executable object {object_id!r} has no provenance derivation"
        )
    for derivation_id in derivation_ids:
        raw = derivations.get(derivation_id)
        if raw is None:
            raise QueryFamilyValidationError(
                f"object {object_id!r} references unknown derivation {derivation_id!r}"
            )
        if object_id not in raw.output_object_ids:
            raise QueryFamilyValidationError(
                f"derivation {derivation_id!r} does not name {object_id!r} as an output"
            )


def _validate_variant_ambiguities(
    variant: QueryVariant,
    expressions: tuple[FamilyExpression, ...],
    ambiguities: dict[str, Ambiguity],
    resolutions: dict[str, AmbiguityResolution],
) -> None:
    used_by_expression = [
        {ref.term_id for ref in _term_refs(expression)} for expression in expressions
    ]
    used_terms = set().union(*used_by_expression)
    required = {
        ambiguity.ambiguity_id
        for ambiguity in ambiguities.values()
        if ambiguity.term_id in used_terms
    }
    selected: dict[str, AmbiguityResolution] = {}
    for resolution_id in variant.ambiguity_resolution_ids:
        resolution = resolutions.get(resolution_id)
        if resolution is None:
            raise QueryFamilyValidationError(
                f"variant {variant.variant_id!r} references unknown ambiguity resolution "
                f"{resolution_id!r}"
            )
        if resolution.ambiguity_id in selected:
            raise QueryFamilyValidationError(
                f"variant {variant.variant_id!r} selects multiple resolutions for ambiguity "
                f"{resolution.ambiguity_id!r}"
            )
        selected[resolution.ambiguity_id] = resolution
    missing = sorted(required - set(selected))
    if missing:
        raise QueryFamilyValidationError(
            f"variant {variant.variant_id!r} uses ambiguous term(s) without explicit "
            f"resolution: {missing}"
        )
    unused = sorted(set(selected) - required)
    if unused:
        raise QueryFamilyValidationError(
            f"variant {variant.variant_id!r} records unused ambiguity resolution(s): {unused}"
        )
    for expression_terms in used_by_expression:
        expression_ambiguities = {
            ambiguity.ambiguity_id
            for ambiguity in ambiguities.values()
            if ambiguity.term_id in expression_terms
        }
        for ambiguity_id in expression_ambiguities:
            resolution = selected[ambiguity_id]
            if resolution.strategy is AmbiguityStrategy.ALLOW_WITH_RATIONALE:
                continue
            missing_context = sorted(set(resolution.context_term_ids) - expression_terms)
            if missing_context:
                raise QueryFamilyValidationError(
                    f"variant {variant.variant_id!r} resolution for {ambiguity_id!r} requires "
                    f"context term(s) absent from an executable source expression: "
                    f"{missing_context}"
                )


def validate_query_family(revision: QueryFamilyRevision) -> None:
    """Validate every reference, provenance edge, and executable invariant."""

    _require_text(revision.research_question, "research_question")
    if not revision.variants:
        raise QueryFamilyValidationError("a query family needs at least one executable variant")

    objects = _unique(_all_object_items(revision), "query-family object")
    concepts = {item.concept_id: item for item in revision.concepts}
    families = {item.family_id: item for item in revision.terminology_families}
    terms = {item.term_id: item for item in revision.accepted_terms}
    ambiguities = {item.ambiguity_id: item for item in revision.ambiguities}
    resolutions = {item.resolution_id: item for item in revision.ambiguity_resolutions}
    translations = {item.translation_id: item for item in revision.source_translations}
    evidence = {
        key: value
        for key, value in _unique(
            ((item.evidence_id, item) for item in revision.evidence), "evidence"
        ).items()
        if isinstance(value, Evidence)
    }
    for item in revision.evidence:
        _validate_evidence(item)
    derivations = _validate_derivation_graph(revision, objects, evidence)

    for concept in revision.concepts:
        _require_text(concept.label, f"concept {concept.concept_id!r} label")
        _require_text(concept.definition, f"concept {concept.concept_id!r} definition")
        _require_provenance(concept.concept_id, concept.derivation_ids, derivations)
    for family in revision.terminology_families:
        _require_text(family.label, f"terminology family {family.family_id!r} label")
        if not family.concept_ids:
            raise QueryFamilyValidationError(
                f"terminology family {family.family_id!r} is linked to no concept"
            )
        for concept_id in family.concept_ids:
            if concept_id not in concepts:
                raise QueryFamilyValidationError(
                    f"terminology family {family.family_id!r} references unknown concept "
                    f"{concept_id!r}"
                )
        _require_provenance(family.family_id, family.derivation_ids, derivations)
    for term in revision.accepted_terms:
        _require_text(term.text, f"term {term.term_id!r} text")
        if "[" in term.text or "]" in term.text:
            raise QueryFamilyValidationError(
                f"term {term.term_id!r} contains query syntax; executable literals are refused"
            )
        term_family = families.get(term.terminology_family_id)
        if term_family is None:
            raise QueryFamilyValidationError(
                f"term {term.term_id!r} references unknown terminology family "
                f"{term.terminology_family_id!r}"
            )
        if not term.concept_ids:
            raise QueryFamilyValidationError(f"term {term.term_id!r} is linked to no concept")
        for concept_id in term.concept_ids:
            if concept_id not in concepts or concept_id not in term_family.concept_ids:
                raise QueryFamilyValidationError(
                    f"term {term.term_id!r} has concept {concept_id!r} outside its family"
                )
        _require_provenance(term.term_id, term.derivation_ids, derivations)
    for proposal in revision.proposals:
        _require_text(proposal.text, f"proposal {proposal.proposal_id!r} text")
        _require_text(proposal.rationale, f"proposal {proposal.proposal_id!r} rationale")
        for evidence_id in proposal.evidence_ids:
            if evidence_id not in evidence:
                raise QueryFamilyValidationError(
                    f"proposal {proposal.proposal_id!r} references unknown evidence {evidence_id!r}"
                )
    for ambiguity in revision.ambiguities:
        if ambiguity.term_id not in terms:
            raise QueryFamilyValidationError(
                f"ambiguity {ambiguity.ambiguity_id!r} references unknown accepted term "
                f"{ambiguity.term_id!r}"
            )
        _require_text(
            ambiguity.competing_sense, f"ambiguity {ambiguity.ambiguity_id!r} competing_sense"
        )
        _require_provenance(ambiguity.ambiguity_id, ambiguity.derivation_ids, derivations)
    for resolution in revision.ambiguity_resolutions:
        if resolution.ambiguity_id not in ambiguities:
            raise QueryFamilyValidationError(
                f"resolution {resolution.resolution_id!r} references unknown ambiguity "
                f"{resolution.ambiguity_id!r}"
            )
        _require_text(resolution.rationale, f"resolution {resolution.resolution_id!r} rationale")
        if (
            resolution.strategy is not AmbiguityStrategy.ALLOW_WITH_RATIONALE
            and not resolution.context_term_ids
        ):
            raise QueryFamilyValidationError(
                f"resolution {resolution.resolution_id!r} needs explicit context terms"
            )
        for term_id in resolution.context_term_ids:
            if term_id not in terms:
                raise QueryFamilyValidationError(
                    f"resolution {resolution.resolution_id!r} references unknown context term "
                    f"{term_id!r}"
                )
        _require_provenance(resolution.resolution_id, resolution.derivation_ids, derivations)
    for translation in revision.source_translations:
        _require_text(translation.source, f"translation {translation.translation_id!r} source")
        if translation.source != translation.source.strip().lower():
            raise QueryFamilyValidationError(
                f"translation {translation.translation_id!r} source must be canonical lower-case"
            )
        _require_text(
            translation.rationale, f"translation {translation.translation_id!r} rationale"
        )
        if translation.mode is TranslationMode.DEGRADED and not translation.rationale.strip():
            raise QueryFamilyValidationError(
                f"degraded translation {translation.translation_id!r} must say what was lost"
            )
        _validate_expression_shape(
            translation.expression, f"translation {translation.translation_id!r}.expression"
        )
        _validate_polarity(
            translation.expression,
            terms,
            f"translation {translation.translation_id!r}.expression",
        )
        _require_provenance(translation.translation_id, translation.derivation_ids, derivations)

    used_translation_ids: set[str] = set()
    for variant in revision.variants:
        _require_text(variant.label, f"variant {variant.variant_id!r} label")
        _require_text(variant.purpose, f"variant {variant.variant_id!r} purpose")
        _validate_expression_shape(variant.expression, f"variant {variant.variant_id!r}.expression")
        _validate_polarity(variant.expression, terms, f"variant {variant.variant_id!r}.expression")
        if variant.applies_to_sources is not None:
            if not variant.applies_to_sources:
                raise QueryFamilyValidationError(
                    f"variant {variant.variant_id!r} applies to no sources"
                )
            if len(set(variant.applies_to_sources)) != len(variant.applies_to_sources):
                raise QueryFamilyValidationError(
                    f"variant {variant.variant_id!r} has duplicate source scope entries"
                )
            for source in variant.applies_to_sources:
                if source != source.strip().lower() or not source:
                    raise QueryFamilyValidationError(
                        f"variant {variant.variant_id!r} has non-canonical source {source!r}"
                    )
        selected_translations: list[SourceTranslation] = []
        seen_sources: set[str] = set()
        for translation_id in variant.source_translation_ids:
            selected_translation = translations.get(translation_id)
            if selected_translation is None:
                raise QueryFamilyValidationError(
                    f"variant {variant.variant_id!r} references unknown source translation "
                    f"{translation_id!r}"
                )
            if selected_translation.source in seen_sources:
                raise QueryFamilyValidationError(
                    f"variant {variant.variant_id!r} has multiple translations for source "
                    f"{selected_translation.source!r}"
                )
            seen_sources.add(selected_translation.source)
            selected_translations.append(selected_translation)
            used_translation_ids.add(translation_id)
            if (
                variant.applies_to_sources is not None
                and selected_translation.source not in variant.applies_to_sources
            ):
                raise QueryFamilyValidationError(
                    f"variant {variant.variant_id!r} translation {translation_id!r} targets a "
                    "source outside the variant scope"
                )
        _validate_variant_ambiguities(
            variant,
            (variant.expression, *(item.expression for item in selected_translations)),
            ambiguities,
            resolutions,
        )
        _require_provenance(variant.variant_id, variant.derivation_ids, derivations)

    unused_translations = sorted(set(translations) - used_translation_ids)
    if unused_translations:
        raise QueryFamilyValidationError(
            f"source translation(s) are not owned by any variant: {unused_translations}"
        )


__all__ = ["QueryFamilyValidationError", "validate_query_family"]
