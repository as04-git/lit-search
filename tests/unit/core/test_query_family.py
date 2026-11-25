"""Tests for the immutable query-family semantic schema."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from lit_search.core.query_family import (
    QUERY_FAMILY_SCHEMA_VERSION,
    AcceptedSearchTerm,
    AllOf,
    Ambiguity,
    AmbiguityResolution,
    AmbiguityStrategy,
    Concept,
    Derivation,
    Evidence,
    EvidenceKind,
    QueryFamilyRevision,
    QueryFamilySchemaError,
    QueryVariant,
    SourceTranslation,
    TerminologyFamily,
    TermProposal,
    TermRef,
    TermRole,
    TranslationMode,
)
from lit_search.query.family_validator import (
    QueryFamilyValidationError,
    validate_query_family,
)


def _revision(*, reverse: bool = False) -> QueryFamilyRevision:
    object_ids = (
        "c_contract",
        "f_synthesis",
        "t_contract",
        "t_repository",
        "a_contract",
        "r_contract",
        "st_openalex",
        "v_main",
    )
    concepts = (Concept("c_contract", "Contract intent", "Infer intended contracts.", ("d_seed",)),)
    families = (
        TerminologyFamily("f_synthesis", "Program synthesis", "PL", ("c_contract",), ("d_seed",)),
    )
    terms = (
        AcceptedSearchTerm(
            "t_contract",
            "contract inference",
            "f_synthesis",
            ("c_contract",),
            TermRole.POSITIVE,
            ("d_seed",),
        ),
        AcceptedSearchTerm(
            "t_repository",
            "repository",
            "f_synthesis",
            ("c_contract",),
            TermRole.DISAMBIGUATOR,
            ("d_seed",),
        ),
    )
    ambiguity = Ambiguity("a_contract", "t_contract", "blockchain contract", ("d_seed",))
    resolution = AmbiguityResolution(
        "r_contract",
        "a_contract",
        AmbiguityStrategy.REQUIRE_CONTEXT,
        "Require repository context.",
        ("t_repository",),
        ("d_seed",),
    )
    translation = SourceTranslation(
        "st_openalex",
        "openalex",
        AllOf((TermRef("t_repository"), TermRef("t_contract"))),
        TranslationMode.DEGRADED,
        "OpenAlex lacks the requested controlled-vocabulary facet.",
        ("d_seed",),
    )
    variant = QueryVariant(
        "v_main",
        "Synthesis lane",
        "Search the originating vocabulary.",
        AllOf((TermRef("t_contract"), TermRef("t_repository"))),
        ("r_contract",),
        ("st_openalex",),
        None,
        ("d_seed",),
    )
    evidence = Evidence("e_user", EvidenceKind.USER, "The user's research formulation.")
    derivation = Derivation(
        "d_seed",
        "decomposed_from_user_question",
        (),
        ("e_user",),
        object_ids,
        "Make every executable object traceable to the supplied formulation.",
    )
    revision = QueryFamilyRevision(
        research_question="How can repository evidence support specification inference?",
        concepts=concepts,
        terminology_families=families,
        accepted_terms=terms,
        proposals=(
            TermProposal("p_model_checking", "model checking", "Candidate only.", ("e_user",)),
        ),
        ambiguities=(ambiguity,),
        ambiguity_resolutions=(resolution,),
        source_translations=(translation,),
        variants=(variant,),
        evidence=(evidence,),
        derivations=(derivation,),
    )
    if not reverse:
        return revision
    translation_expression = revision.source_translations[0].expression
    variant_expression = revision.variants[0].expression
    assert isinstance(translation_expression, AllOf)
    assert isinstance(variant_expression, AllOf)
    reversed_translation = replace(
        revision.source_translations[0],
        expression=AllOf(tuple(reversed(translation_expression.children))),
    )
    reversed_variant = replace(
        revision.variants[0],
        expression=AllOf(tuple(reversed(variant_expression.children))),
    )
    return replace(
        revision,
        accepted_terms=tuple(reversed(revision.accepted_terms)),
        variants=(reversed_variant,),
        source_translations=(reversed_translation,),
    )


def test_revision_is_frozen_and_content_addressed() -> None:
    """A revision cannot be mutated and carries a query-family digest."""

    revision = _revision()
    with pytest.raises(FrozenInstanceError):
        revision.research_question = "changed"  # type: ignore[misc]
    assert revision.revision_id.startswith("qf_")


def test_canonical_digest_ignores_collection_and_expression_order() -> None:
    """Commutative input reordering does not create a different revision."""

    left = _revision()
    right = _revision(reverse=True)
    assert left.revision_id == right.revision_id
    assert left.to_dict() == right.to_dict()


def test_strict_round_trip_and_schema_version() -> None:
    """The serialized revision round-trips without dropping typed objects."""

    revision = _revision()
    restored = QueryFamilyRevision.from_dict(revision.to_dict())
    assert restored.to_dict() == revision.to_dict()
    assert restored.revision_id == revision.revision_id
    assert restored.to_dict()["schema_version"] == QUERY_FAMILY_SCHEMA_VERSION


def test_unknown_field_is_refused() -> None:
    """A newer/foreign field cannot be ignored by the version-1 reader."""

    payload = _revision().to_dict()
    payload["silent_new_meaning"] = True
    with pytest.raises(QueryFamilySchemaError, match="unrecognised key"):
        QueryFamilyRevision.from_dict(payload)


def test_unknown_nested_field_is_refused() -> None:
    """Strictness applies recursively rather than only to the envelope."""

    payload = _revision().to_dict()
    payload["concepts"][0]["new_field"] = "ignored by an old reader"
    with pytest.raises(QueryFamilySchemaError, match="unrecognised key"):
        QueryFamilyRevision.from_dict(payload)


def test_unknown_schema_version_is_refused() -> None:
    """No migration is guessed for an unsupported schema."""

    payload = _revision().to_dict()
    payload["schema_version"] = 2
    with pytest.raises(QueryFamilySchemaError, match="unsupported version 2"):
        QueryFamilyRevision.from_dict(payload)


def test_tampered_semantics_fail_digest_check() -> None:
    """A stored revision ID cannot survive changed semantic content."""

    payload = _revision().to_dict()
    payload["research_question"] = "tampered"
    with pytest.raises(QueryFamilySchemaError, match="recomputed"):
        QueryFamilyRevision.from_dict(payload)


def test_all_executable_objects_require_real_derivation_outputs() -> None:
    """Naming a derivation is insufficient when it did not derive the object."""

    revision = _revision()
    bad = replace(
        revision,
        derivations=(replace(revision.derivations[0], output_object_ids=("c_contract",)),),
    )
    with pytest.raises(QueryFamilyValidationError, match="does not name 'f_synthesis'"):
        validate_query_family(bad)


def test_handle_member_evidence_must_pin_field_digests() -> None:
    """Paper-derived vocabulary cannot point vaguely at an unpinned set."""

    revision = _revision()
    evidence = Evidence(
        "e_user",
        EvidenceKind.HANDLE_MEMBER,
        "Term observed in a retrieved paper.",
        handle_id="h_abc",
    )
    with pytest.raises(QueryFamilyValidationError, match="must pin handle"):
        validate_query_family(replace(revision, evidence=(evidence,)))


def test_handle_member_evidence_refuses_prefix_and_untyped_digests() -> None:
    """A revision pins immutable full locators rather than contextual prefixes."""
    revision = _revision()
    evidence = Evidence(
        "e_user",
        EvidenceKind.HANDLE_MEMBER,
        "Term observed in a retrieved paper.",
        handle_id="h_abc",
        member_digest="abc",
        field="title",
        field_value_digest="def",
    )
    with pytest.raises(QueryFamilyValidationError, match="full immutable"):
        validate_query_family(replace(revision, evidence=(evidence,)))


def test_proposal_is_not_an_accepted_term() -> None:
    """A model proposal cannot become executable merely by sharing an ID."""

    revision = _revision()
    variant = replace(
        revision.variants[0],
        expression=AllOf((TermRef("p_model_checking"), TermRef("t_repository"))),
        ambiguity_resolution_ids=(),
    )
    with pytest.raises(QueryFamilyValidationError, match="not an accepted term"):
        validate_query_family(replace(revision, variants=(variant,)))
