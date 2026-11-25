"""Tests for embedding identity and the query-embedding cache.

The point of the identity type is that a vector means nothing without it, and
the point of the cache is that a cache keyed on text alone hands back a vector
built under a prompt convention that no longer applies.
"""

from __future__ import annotations

import pytest

from lit_search.index.embedding import (
    DeterministicEmbedder,
    EmbeddingError,
    EmbeddingIdentity,
    QueryEmbeddingCache,
)


def _identity(**overrides: object) -> EmbeddingIdentity:
    """Build an identity with sensible defaults.

    Parameters
    ----------
    **overrides : object
        Fields to override.

    Returns
    -------
    EmbeddingIdentity
        Identity.
    """
    fields: dict = {
        "model_id": "acme/embed",
        "dimension": 4,
        "normalization": "l2",
        "prompt_convention": "acme:query/passage",
    }
    fields.update(overrides)
    return EmbeddingIdentity(**fields)


class TestIdentity:
    """An identity that cannot describe a vector space is refused."""

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"model_id": "  "}, "must name its model"),
            ({"dimension": 0}, "positive dimension"),
            ({"prompt_convention": ""}, "prompt convention"),
            ({"normalization": ""}, "normalization"),
        ],
    )
    def test_incomplete_identity_is_refused(self, overrides: dict, message: str) -> None:
        """Every field that changes what a vector means is required."""
        with pytest.raises(EmbeddingError, match=message):
            _identity(**overrides)

    def test_fingerprint_separates_models_at_the_same_dimension(self) -> None:
        """A dimension check alone cannot tell two vector spaces apart."""
        left = _identity(model_id="a/one")
        right = _identity(model_id="b/two")
        assert left.dimension == right.dimension
        assert left.fingerprint != right.fingerprint

    def test_prompt_convention_is_part_of_the_fingerprint(self) -> None:
        """The same model under two conventions is two vector spaces."""
        assert _identity().fingerprint != _identity(prompt_convention="none").fingerprint

    def test_round_trip(self) -> None:
        """Serializing and reading back preserves the identity exactly."""
        original = _identity(model_revision="abc123", is_semantic=False)
        assert EmbeddingIdentity.from_dict(original.to_dict()) == original

    def test_a_serialized_identity_missing_a_field_is_refused(self) -> None:
        """Defaulting a missing field would make it indistinguishable from a recorded one."""
        payload = _identity().to_dict()
        del payload["normalization"]
        with pytest.raises(EmbeddingError, match="missing"):
            EmbeddingIdentity.from_dict(payload)

    def test_a_non_semantic_identity_says_so_when_described(self) -> None:
        """The stand-in must be recognisable wherever it appears."""
        assert "NOT SEMANTIC" in _identity(is_semantic=False).describe()


class TestDeterministicEmbedder:
    """The stand-in is reproducible, asymmetric, and honest about itself."""

    def test_query_and_document_vectors_differ(self) -> None:
        """A test that swaps the two encode paths must be able to see it."""
        embedder = DeterministicEmbedder(dimension=16)
        assert embedder.encode_query("hoare logic") != embedder.encode_documents(["hoare logic"])[0]

    def test_documents_sharing_words_are_closer(self) -> None:
        """Token-level hashing keeps the residue and diversity tests meaningful."""
        from lit_search.index.mmr import cosine

        embedder = DeterministicEmbedder(dimension=64)
        near, far = (
            embedder.encode_documents(
                ["hoare logic model checking", "hoare logic model verification"]
            ),
            embedder.encode_documents(["hoare logic model checking", "unrelated marine biology"]),
        )
        assert cosine(near[0], near[1]) > cosine(far[0], far[1])

    def test_it_declares_itself_non_semantic(self) -> None:
        """Downstream refusals depend on this flag being set."""
        assert DeterministicEmbedder(4).identity.is_semantic is False


class TestQueryEmbeddingCache:
    """The key carries the identity, and the entry is re-validated on read."""

    def test_a_repeated_query_is_served_from_cache(self) -> None:
        """The whole point, and the only benefit being claimed."""
        cache = QueryEmbeddingCache()
        embedder = DeterministicEmbedder(8)
        first = cache.embed_query(embedder, "specification inference")
        second = cache.embed_query(embedder, "specification inference")
        assert first == second
        assert cache.statistics == {"hits": 1, "misses": 1, "entries": 1}

    def test_changing_the_model_changes_the_key(self) -> None:
        """A cache keyed on text alone would serve the previous model's vector."""
        cache = QueryEmbeddingCache()
        small, large = DeterministicEmbedder(8), DeterministicEmbedder(16)
        assert len(cache.embed_query(small, "q")) == 8
        assert len(cache.embed_query(large, "q")) == 16
        assert cache.statistics["hits"] == 0

    def test_key_includes_the_prompt_convention(self) -> None:
        """Changing only the convention must invalidate the entry."""
        left = QueryEmbeddingCache.key_for(_identity(), "q")
        right = QueryEmbeddingCache.key_for(_identity(prompt_convention="none"), "q")
        assert left != right
