"""Strict identity helpers for paper records and publication versions.

These helpers deliberately do not answer whether two records describe the
same scholarly work.  Work equivalence is an evidence-bearing inference, not
an object-equality rule.  This module only exposes identities that a source
record states explicitly enough to compare without title or semantic
guesswork.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lit_search.utils.normalization import normalize_doi

from .citation import IdentifierKind

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .paper import Paper

_ARXIV_DOI_PREFIX = "10.48550/arxiv."
_SOURCE_RECORD_KINDS = frozenset({IdentifierKind.SEMANTIC_SCHOLAR, IdentifierKind.OPENALEX})


def version_identity_key(paper: Paper) -> str | None:
    """Return the record's explicit primary publication-version identity.

    A primary DOI is preferred.  The synthetic arXiv DOI and a typed arXiv ID
    are canonicalised to the same key.  Provider-native Semantic Scholar and
    OpenAlex IDs are source-record identities, so they are intentionally not
    used as publication-version identities.

    ``None`` means that the record does not state a version identity.  A title
    is never a substitute: two different works can have the same title.
    """
    if paper.doi is not None:
        doi = (normalize_doi(paper.doi) or paper.doi.strip()).lower()
        if doi.startswith(_ARXIV_DOI_PREFIX):
            return f"arxiv:{doi[len(_ARXIV_DOI_PREFIX) :]}"
        if doi:
            return f"doi:{doi}"
    typed_dois: list[str] = []
    for identifier in paper.identifiers:
        if identifier.kind is not IdentifierKind.DOI:
            continue
        doi = (normalize_doi(identifier.value) or identifier.value.strip()).lower()
        if doi.startswith(_ARXIV_DOI_PREFIX):
            typed_dois.append(f"arxiv:{doi[len(_ARXIV_DOI_PREFIX) :]}")
        elif doi:
            typed_dois.append(f"doi:{doi}")
    if typed_dois:
        return min(typed_dois)
    arxiv_ids = sorted(
        identifier.value.strip().lower()
        for identifier in paper.identifiers
        if identifier.kind is IdentifierKind.ARXIV and identifier.value.strip()
    )
    return f"arxiv:{arxiv_ids[0]}" if arxiv_ids else None


def version_identity_keys(paper: Paper) -> frozenset[str]:
    """Return explicit version-like aliases as relationship evidence.

    A provider response may place a DOI only in the typed identifier list, so
    checking :attr:`Paper.doi` alone would silently discard real identity
    evidence.  Synthetic arXiv DOIs and typed arXiv IDs canonicalize to the
    same key. These aliases are *not* an exact-equivalence set: a publisher DOI
    and an arXiv ID on one provider record often name different versions of
    the same work. Exact record membership uses :func:`version_identity_key`,
    which prefers the primary DOI. ``preprint_doi`` is deliberately excluded:
    it explicitly links another version rather than naming this one.
    """
    raw_dois = [
        identifier.value
        for identifier in paper.identifiers
        if identifier.kind is IdentifierKind.DOI
    ]
    if paper.doi is not None:
        raw_dois.append(paper.doi)
    keys: set[str] = set()
    for raw in raw_dois:
        doi = (normalize_doi(raw) or raw.strip()).lower()
        if doi.startswith(_ARXIV_DOI_PREFIX):
            keys.add(f"arxiv:{doi[len(_ARXIV_DOI_PREFIX) :]}")
        elif doi:
            keys.add(f"doi:{doi}")
    keys.update(
        f"arxiv:{identifier.value.strip().lower()}"
        for identifier in paper.identifiers
        if identifier.kind is IdentifierKind.ARXIV and identifier.value.strip()
    )
    return frozenset(keys)


def source_record_keys(paper: Paper) -> frozenset[str]:
    """Return provider-qualified native source-record identities.

    DOI and arXiv aliases are excluded because they identify publication
    versions rather than records in a particular aggregator.  Multiple native
    keys are retained instead of choosing one and hiding an alias conflict.
    """
    return frozenset(
        identifier.key
        for identifier in paper.identifiers
        if identifier.kind in _SOURCE_RECORD_KINDS
    )


def has_version_identity_conflict(left: Paper, right: Paper) -> bool:
    """Return whether two records state different non-null version identities."""
    left_key = version_identity_key(left)
    right_key = version_identity_key(right)
    return left_key is not None and right_key is not None and left_key != right_key


def record_identity_keys(paper: Paper) -> frozenset[str]:
    """Return every explicit key that may establish source-record sameness.

    Publication-version identities and provider-native record identities are
    kept in one comparison set only for exact membership operations.  The
    function deliberately returns an empty set for an identifier-free record:
    title, year, authors and semantic similarity are evidence for a possible
    relationship, not a safe record identity.
    """
    keys = set(source_record_keys(paper))
    version_key = version_identity_key(paper)
    if version_key is not None:
        keys.add(version_key)
    return frozenset(keys)


def same_record_identity(left: Paper, right: Paper) -> bool:
    """Return whether two objects state the same explicit record identity.

    Exact object identity is preserved for identifier-free records within one
    in-memory operation. Publication-version identity matches only by an
    explicit shared version alias. Provider-record identity matches only while
    *both* records state no publication version.

    The latter restriction is essential for transitivity. An unversioned
    provider record must not become a wildcard bridge between two records that
    share its provider alias but state conflicting DOIs. Enrichment may later
    model that compatibility as evidence; exact membership cannot.
    """
    if left is right:
        return True
    left_version = version_identity_key(left)
    right_version = version_identity_key(right)
    if left_version is not None or right_version is not None:
        return left_version is not None and left_version == right_version
    left_sources = source_record_keys(left)
    return bool(left_sources and left_sources.intersection(source_record_keys(right)))


__all__ = [
    "has_version_identity_conflict",
    "record_identity_keys",
    "same_record_identity",
    "source_record_keys",
    "version_identity_key",
    "version_identity_keys",
]
