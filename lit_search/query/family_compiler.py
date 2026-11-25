"""Pure compilation and deterministic planning for query families.

This module performs no I/O and owns no connector.  It selects the exact
semantic expression declared for each ``(variant, source)`` cell, compiles it
through the existing validated query syntax, and content-addresses both cells
and the complete plan.  An explicit source translation is fail-closed: if its
expression cannot compile, the error escapes and the generic expression is
never substituted.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from lit_search.core.query import Query
from lit_search.core.query_family import (
    AcceptedSearchTerm,
    AllOf,
    AnyOf,
    FamilyExpression,
    QueryFamilyRevision,
    QueryVariant,
    SourceTranslation,
    TermRef,
    TranslationMode,
    canonical_json,
)
from lit_search.query.family_validator import validate_query_family
from lit_search.query.parser import QueryParser
from lit_search.query.propagator import FilterPropagator
from lit_search.query.validator import QueryValidator

_DIGEST_SIZE = 16


def _digest(prefix: str, value: Any) -> str:
    raw = canonical_json(value).encode("utf-8")
    return f"{prefix}_{hashlib.blake2b(raw, digest_size=_DIGEST_SIZE).hexdigest()}"


@dataclass(frozen=True, slots=True)
class ScopeExclusion:
    """A requested cell that the declared variant scope excludes."""

    variant_id: str
    source: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        """Return the serialized scope exclusion."""

        return {
            "variant_id": self.variant_id,
            "source": self.source,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class QueryCell:
    """One deterministic source-by-variant executable query unit."""

    revision_id: str
    variant_id: str
    source: str
    query_text: str
    translation_id: str | None
    translation_mode: TranslationMode
    degradation_reason: str | None

    @property
    def query(self) -> Query:
        """Return a fresh AST for the frozen canonical query text.

        The existing :class:`Query` model is mutable. Embedding one directly
        would let a caller mutate a frozen cell's semantic value and silently
        change its content address. Reparsing preserves the familiar
        execution interface without retaining mutable shared state.
        """

        return _parse_query_text(self.query_text)

    def semantic_dict(self) -> dict[str, Any]:
        """Return the canonical semantic cell payload."""

        return {
            "revision_id": self.revision_id,
            "variant_id": self.variant_id,
            "source": self.source,
            "query_text": self.query_text,
            "query_ast": self.query.to_dict(),
            "translation_id": self.translation_id,
            "translation_mode": self.translation_mode.value,
            "degradation_reason": self.degradation_reason,
        }

    @property
    def cell_id(self) -> str:
        """Return the deterministic cell content address."""

        return _digest("qc", self.semantic_dict())

    @property
    def query_digest(self) -> str:
        """Return a digest of only the canonical executable query."""

        return _digest("q", self.query.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized cell, identifiers included."""

        return {
            **self.semantic_dict(),
            "cell_id": self.cell_id,
            "query_digest": self.query_digest,
        }


@dataclass(frozen=True, slots=True)
class QueryFamilyPlan:
    """Canonical cells and explicit scope exclusions for one revision."""

    revision_id: str
    selected_sources: tuple[str, ...]
    cells: tuple[QueryCell, ...]
    scope_exclusions: tuple[ScopeExclusion, ...]

    def semantic_dict(self) -> dict[str, Any]:
        """Return the canonical plan payload."""

        return {
            "revision_id": self.revision_id,
            "selected_sources": sorted(self.selected_sources),
            "cells": [cell.to_dict() for cell in sorted(self.cells, key=lambda item: item.cell_id)],
            "scope_exclusions": [
                item.to_dict()
                for item in sorted(
                    self.scope_exclusions, key=lambda item: (item.variant_id, item.source)
                )
            ],
        }

    @property
    def plan_id(self) -> str:
        """Return the deterministic plan content address."""

        return _digest("qfp", self.semantic_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized plan, identifier included."""

        return {**self.semantic_dict(), "plan_id": self.plan_id}


def _render_expression(
    expression: FamilyExpression,
    terms: dict[str, AcceptedSearchTerm],
) -> str:
    if isinstance(expression, TermRef):
        term = terms[expression.term_id]
        return f"{expression.field.value}[{term.text}]"
    if isinstance(expression, AllOf | AnyOf):
        operator = " AND " if isinstance(expression, AllOf) else " OR "
        rendered = sorted(_render_expression(child, terms) for child in expression.children)
        return f"({operator.join(rendered)})"
    return (
        f"({_render_expression(expression.include, terms)} AND NOT "
        f"{_render_expression(expression.exclude, terms)})"
    )


def compile_expression(
    expression: FamilyExpression,
    accepted_terms: tuple[AcceptedSearchTerm, ...],
) -> Query:
    """Compile one validated symbolic expression into the existing Query AST."""

    terms = {term.term_id: term for term in accepted_terms}
    raw_query = _render_expression(expression, terms)
    return _parse_query_text(raw_query)


def _parse_query_text(raw_query: str) -> Query:
    """Parse one canonical query without retaining a mutable shared AST."""

    QueryValidator().validate(raw_query)
    query = QueryParser().parse(raw_query)
    FilterPropagator().propagate(query)
    return query


def _translations_for(
    variant: QueryVariant,
    translations: dict[str, SourceTranslation],
) -> dict[str, SourceTranslation]:
    return {
        translations[translation_id].source: translations[translation_id]
        for translation_id in variant.source_translation_ids
    }


def plan_query_family(
    revision: QueryFamilyRevision,
    sources: tuple[str, ...] | list[str],
) -> QueryFamilyPlan:
    """Validate and compile every declared source-by-variant query cell.

    Source input order and duplicates are semantically irrelevant.  Explicit
    variant scope exclusions remain visible in the plan but are not execution
    failures.  An explicit source translation, when present, is the only
    expression considered for that cell.
    """

    validate_query_family(revision)
    selected_sources = tuple(sorted({source.strip().lower() for source in sources}))
    if not selected_sources or any(not source for source in selected_sources):
        raise ValueError("query-family planning requires at least one non-blank source")

    translations = {
        translation.translation_id: translation for translation in revision.source_translations
    }
    cells: list[QueryCell] = []
    exclusions: list[ScopeExclusion] = []
    for variant in sorted(revision.variants, key=lambda item: item.variant_id):
        by_source = _translations_for(variant, translations)
        scope = None if variant.applies_to_sources is None else set(variant.applies_to_sources)
        for source in selected_sources:
            if scope is not None and source not in scope:
                exclusions.append(
                    ScopeExclusion(
                        variant_id=variant.variant_id,
                        source=source,
                        reason="source is outside the variant's declared applicability scope",
                    )
                )
                continue
            translation = by_source.get(source)
            expression = variant.expression if translation is None else translation.expression
            mode = TranslationMode.EQUIVALENT if translation is None else translation.mode
            reason = (
                translation.rationale
                if translation is not None and mode is TranslationMode.DEGRADED
                else None
            )
            # Fail closed.  In particular, do not catch a failure here and retry
            # with variant.expression when an explicit translation was selected.
            query = compile_expression(expression, revision.accepted_terms)
            cells.append(
                QueryCell(
                    revision_id=revision.revision_id,
                    variant_id=variant.variant_id,
                    source=source,
                    query_text=query.raw_query,
                    translation_id=None if translation is None else translation.translation_id,
                    translation_mode=mode,
                    degradation_reason=reason,
                )
            )

    return QueryFamilyPlan(
        revision_id=revision.revision_id,
        selected_sources=selected_sources,
        cells=tuple(sorted(cells, key=lambda item: (item.variant_id, item.source))),
        scope_exclusions=tuple(sorted(exclusions, key=lambda item: (item.variant_id, item.source))),
    )


__all__ = [
    "QueryCell",
    "QueryFamilyPlan",
    "ScopeExclusion",
    "compile_expression",
    "plan_query_family",
]
