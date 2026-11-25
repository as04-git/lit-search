"""Falsification tests for query-family validation and compilation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from lit_search.core.query_family import (
    AcceptedSearchTerm,
    AllOf,
    AndNot,
    Derivation,
    QueryFamilyRevision,
    QueryVariant,
    TermRef,
    TermRole,
    TranslationMode,
)
from lit_search.query.family_compiler import compile_expression, plan_query_family
from lit_search.query.family_validator import (
    QueryFamilyValidationError,
    validate_query_family,
)
from tests.unit.core.test_query_family import _revision


def test_plan_compiles_one_cell_per_applicable_variant_source() -> None:
    """Generic and source-translated cells remain distinct work units."""

    plan = plan_query_family(_revision(), ["openalex", "arxiv"])
    assert [(cell.variant_id, cell.source) for cell in plan.cells] == [
        ("v_main", "arxiv"),
        ("v_main", "openalex"),
    ]
    arxiv, openalex = plan.cells
    assert arxiv.translation_id is None
    assert arxiv.translation_mode is TranslationMode.EQUIVALENT
    assert arxiv.degradation_reason is None
    assert openalex.translation_id == "st_openalex"
    assert openalex.translation_mode is TranslationMode.DEGRADED
    assert "controlled-vocabulary" in (openalex.degradation_reason or "")


def test_compilation_produces_existing_query_ast_with_no_literal_seam() -> None:
    """Accepted term text reaches the ordinary Query only through TermRef lookup."""

    revision = _revision()
    query = compile_expression(revision.variants[0].expression, revision.accepted_terms)
    assert query.raw_query == "(tiabs[contract inference] AND tiabs[repository])"
    assert query.get_all_terms() == ["contract inference", "repository"]


def test_source_override_fails_closed_instead_of_using_generic_expression() -> None:
    """A broken explicit translation is never replaced by the valid generic lane."""

    revision = _revision()
    translation = replace(
        revision.source_translations[0],
        expression=AllOf((TermRef("p_model_checking"), TermRef("t_repository"))),
    )
    with pytest.raises(QueryFamilyValidationError, match="not an accepted term"):
        plan_query_family(replace(revision, source_translations=(translation,)), ["openalex"])


def test_ambiguous_term_requires_resolution_in_every_variant_that_uses_it() -> None:
    """Known sense drift cannot disappear because a new lane omitted metadata."""

    revision = _revision()
    variant = replace(revision.variants[0], ambiguity_resolution_ids=())
    with pytest.raises(QueryFamilyValidationError, match="without explicit resolution"):
        validate_query_family(replace(revision, variants=(variant,)))


def _with_negative_term() -> tuple[QueryFamilyRevision, AcceptedSearchTerm]:
    revision = _revision()
    negative = AcceptedSearchTerm(
        "t_blockchain",
        "blockchain",
        "f_synthesis",
        ("c_contract",),
        TermRole.NEGATIVE,
        ("d_seed",),
    )
    derivation: Derivation = revision.derivations[0]
    updated = replace(
        revision,
        accepted_terms=(*revision.accepted_terms, negative),
        derivations=(
            replace(
                derivation,
                output_object_ids=(*derivation.output_object_ids, negative.term_id),
            ),
        ),
    )
    return updated, negative


def test_negative_term_compiles_only_in_explicit_exclusion_subtree() -> None:
    """Known negatives are visible as AND NOT rather than globally injected."""

    raw_revision, negative = _with_negative_term()
    revision = raw_revision
    variant: QueryVariant = revision.variants[0]
    expression = AndNot(include=variant.expression, exclude=TermRef(negative.term_id))
    valid_variant = replace(variant, expression=expression)
    valid_revision = replace(revision, variants=(valid_variant,))
    validate_query_family(valid_revision)
    query = compile_expression(expression, valid_revision.accepted_terms)
    assert "AND NOT tiabs[blockchain]" in query.raw_query


def test_negative_term_in_positive_branch_is_refused() -> None:
    """A negative role cannot masquerade as a positive synonym."""

    raw_revision, negative = _with_negative_term()
    revision = raw_revision
    variant: QueryVariant = revision.variants[0]
    invalid = replace(
        variant,
        expression=AllOf((variant.expression, TermRef(negative.term_id))),
    )
    with pytest.raises(QueryFamilyValidationError, match="outside an explicit"):
        validate_query_family(replace(revision, variants=(invalid,)))


def test_plan_ids_are_invariant_to_input_reordering_and_duplicates() -> None:
    """Equivalent revision/source orderings yield the same executable plan."""

    left = plan_query_family(_revision(), ["openalex", "arxiv", "openalex"])
    right = plan_query_family(_revision(reverse=True), ["arxiv", "openalex"])
    assert left.plan_id == right.plan_id
    assert [cell.cell_id for cell in left.cells] == [cell.cell_id for cell in right.cells]
    assert [cell.query.raw_query for cell in left.cells] == [
        cell.query.raw_query for cell in right.cells
    ]


def test_mutating_a_returned_query_ast_cannot_change_cell_or_plan_identity() -> None:
    """The mutable legacy Query model is reparsed, not embedded in a frozen cell."""

    plan = plan_query_family(_revision(), ["openalex"])
    cell = plan.cells[0]
    original_cell_id = cell.cell_id
    original_plan_id = plan.plan_id
    query = cell.query

    query.raw_query = "tiabs[mutated outside the cell]"

    assert cell.query.raw_query != query.raw_query
    assert cell.cell_id == original_cell_id
    assert plan.plan_id == original_plan_id


def test_variant_scope_is_recorded_separately_from_execution_cells() -> None:
    """A planned source exclusion is scope metadata, not a failed query cell."""

    revision = _revision()
    variant = replace(revision.variants[0], applies_to_sources=("openalex",))
    scoped = replace(revision, variants=(variant,))
    plan = plan_query_family(scoped, ["arxiv", "openalex"])
    assert [cell.source for cell in plan.cells] == ["openalex"]
    assert [(item.variant_id, item.source) for item in plan.scope_exclusions] == [
        ("v_main", "arxiv")
    ]


def test_every_ambiguity_context_term_must_be_present_in_source_translation() -> None:
    """A translation cannot drop the context that justified an ambiguous term."""

    revision = _revision()
    translation = replace(
        revision.source_translations[0],
        expression=AllOf((TermRef("t_contract"), TermRef("t_contract"))),
    )
    with pytest.raises(QueryFamilyValidationError, match=r"context term.*absent"):
        validate_query_family(replace(revision, source_translations=(translation,)))
