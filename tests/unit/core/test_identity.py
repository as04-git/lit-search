"""Focused tests for explicit paper-record and version identities."""

from __future__ import annotations

from lit_search.core.citation import IdentifierKind, PaperIdentifier
from lit_search.core.identity import (
    has_version_identity_conflict,
    record_identity_keys,
    same_record_identity,
    source_record_keys,
    version_identity_key,
    version_identity_keys,
)
from lit_search.core.paper import Paper


def _paper(
    *,
    doi: str | None = None,
    identifiers: list[PaperIdentifier] | None = None,
) -> Paper:
    return Paper(
        title="A Paper",
        abstract="",
        authors=[],
        source=None,
        publication_date=None,
        doi=doi,
        identifiers=identifiers,
    )


def test_version_identity_normalises_doi_forms_and_case() -> None:
    """Equivalent explicit DOI spellings produce the same version key."""
    bare = _paper(doi="10.1234/Example")
    url = _paper(doi="https://doi.org/10.1234/example")

    assert version_identity_key(bare) == version_identity_key(url) == "doi:10.1234/example"
    assert not has_version_identity_conflict(bare, url)


def test_arxiv_doi_and_typed_arxiv_id_share_one_version_key() -> None:
    """The two explicit spellings of an arXiv version are canonical aliases."""
    doi = _paper(doi="10.48550/arXiv.2601.00001")
    typed = _paper(identifiers=[PaperIdentifier(IdentifierKind.ARXIV, "2601.00001")])

    assert version_identity_key(doi) == version_identity_key(typed) == "arxiv:2601.00001"
    assert not has_version_identity_conflict(doi, typed)


def test_typed_doi_is_version_identity_even_without_primary_doi_field() -> None:
    """A connector's typed DOI evidence is not discarded by identity logic."""
    typed = _paper(identifiers=[PaperIdentifier(IdentifierKind.DOI, "10.1234/Typed")])
    primary = _paper(doi="https://doi.org/10.1234/typed")

    assert version_identity_keys(typed) == frozenset({"doi:10.1234/typed"})
    assert same_record_identity(typed, primary)


def test_primary_doi_prevents_cross_version_alias_bridge() -> None:
    """A provider's arXiv alias does not merge its publisher DOI with a preprint."""
    multi = _paper(
        identifiers=[
            PaperIdentifier(IdentifierKind.DOI, "10.1234/one"),
            PaperIdentifier(IdentifierKind.ARXIV, "2601.00001"),
        ]
    )
    shared = _paper(doi="10.48550/arXiv.2601.00001")
    disjoint = _paper(doi="10.1234/other")

    assert version_identity_key(multi) == "doi:10.1234/one"
    assert has_version_identity_conflict(multi, shared)
    assert not same_record_identity(multi, shared)
    assert has_version_identity_conflict(multi, disjoint)


def test_title_is_never_a_version_identity_fallback() -> None:
    """An unidentified record remains explicitly unidentified."""
    assert version_identity_key(_paper()) is None


def test_different_non_null_version_identities_conflict() -> None:
    """Two distinct DOI versions cannot be treated as one merge target."""
    assert has_version_identity_conflict(
        _paper(doi="10.1234/preprint"), _paper(doi="10.5678/published")
    )


def test_source_record_keys_are_provider_qualified_and_separate_from_versions() -> None:
    """Aggregator-native record IDs do not masquerade as publication versions."""
    paper = _paper(
        doi="10.1234/example",
        identifiers=[
            PaperIdentifier(IdentifierKind.SEMANTIC_SCHOLAR, "S2-ID"),
            PaperIdentifier(IdentifierKind.OPENALEX, "https://openalex.org/W123"),
            PaperIdentifier(IdentifierKind.ARXIV, "2601.00001"),
        ],
    )

    assert source_record_keys(paper) == frozenset({"semantic_scholar:s2-id", "openalex:w123"})
    assert version_identity_key(paper) == "doi:10.1234/example"


def test_record_identity_requires_explicit_overlap() -> None:
    """A matching title is not record identity, even for sparse records."""
    left = _paper()
    right = _paper()

    assert record_identity_keys(left) == frozenset()
    assert same_record_identity(left, left)
    assert not same_record_identity(left, right)


def test_record_identity_accepts_version_or_source_record_overlap() -> None:
    """Either kind of explicit record key can prove exact membership."""
    by_version = _paper(doi="10.1000/example")
    same_version = _paper(doi="https://doi.org/10.1000/EXAMPLE")
    by_source = _paper(identifiers=[PaperIdentifier(IdentifierKind.OPENALEX, "W123")])
    same_source = _paper(identifiers=[PaperIdentifier(IdentifierKind.OPENALEX, "W123")])

    assert same_record_identity(by_version, same_version)
    assert same_record_identity(by_source, same_source)


def test_record_identity_preserves_conflicting_versions_despite_source_alias() -> None:
    """An aggregator alias cannot launder two explicit DOI versions together."""
    shared = PaperIdentifier(IdentifierKind.SEMANTIC_SCHOLAR, "shared")
    left = _paper(doi="10.1000/left", identifiers=[shared])
    right = _paper(doi="10.1000/right", identifiers=[shared])

    assert not same_record_identity(left, right)


def test_unversioned_provider_record_is_not_a_wildcard_bridge() -> None:
    """Exact record identity stays transitive across unknown/versioned records."""
    shared = PaperIdentifier(IdentifierKind.SEMANTIC_SCHOLAR, "shared")
    unknown = _paper(identifiers=[shared])
    first = _paper(doi="10.1000/first", identifiers=[shared])
    second = _paper(doi="10.1000/second", identifiers=[shared])

    assert not same_record_identity(unknown, first)
    assert not same_record_identity(unknown, second)
    assert not same_record_identity(first, second)


def test_publisher_record_with_arxiv_alias_is_not_a_version_bridge() -> None:
    """Published DOI plus arXiv external ID cannot collapse both versions."""
    published = _paper(
        doi="10.1000/published",
        identifiers=[PaperIdentifier(IdentifierKind.ARXIV, "2601.00001")],
    )
    doi_only = _paper(doi="10.1000/published")
    arxiv_only = _paper(identifiers=[PaperIdentifier(IdentifierKind.ARXIV, "2601.00001")])

    assert same_record_identity(published, doi_only)
    assert not same_record_identity(published, arxiv_only)
    assert not same_record_identity(doi_only, arxiv_only)
