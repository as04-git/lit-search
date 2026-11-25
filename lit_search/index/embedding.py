"""Embedding models, their identity, and the cache that must not outlive it.

Three things are load-bearing here, and each of them exists because getting it
wrong is silent.

**Identity travels with the vector, not with the code.**
:class:`EmbeddingIdentity` carries the model, its revision, its dimension and
its prompt convention.  It is stored on the index *and on every row*, because
an index rebuilt with a different model produces vectors that are still 1024
floats and still cosine-comparable to the old ones -- and completely
meaningless against them.  Dimension alone cannot detect that: two different
1024-dimension models agree on the only thing a dimension check looks at.

**A model without a query prompt is refused, not silently accepted.**
``sentence_transformers`` implements ``encode_query`` as "use the ``query``
prompt if the model ships one", and falls through to a plain symmetric encode
when it does not -- no error, no warning.  So the ``encode_query`` /
``encode_document`` discipline is a *no-op on exactly the models that need it
most*.  :class:`SentenceTransformerEmbedder` therefore asserts that the loaded
model ships a ``query`` prompt and raises when it does not.

**A query-embedding cache is keyed by the identity, not by the text.**
Lesson taken from ``ziwenhahaha/daily-paper-reader`` (MIT, 887 stars, pushed
2026-08-28), which composes its cache key from a version marker, the model
name and the *prefixed* query text, then re-validates the stored model and
prefixed text when reading the entry back.  A cache keyed on raw query text
alone returns a vector built under the previous prompt convention after the
convention changes, which is a silently wrong ranking.  :class:`QueryEmbeddingCache`
copies the key composition and the read-time re-validation, not the code.
"""

from __future__ import annotations

import hashlib
import logging
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from lit_search.exceptions import LitSearchError

logger = logging.getLogger(__name__)

#: Bumped whenever the cache key composition changes, so old entries cannot be
#: read under new rules.
_CACHE_KEY_VERSION = "v1"

#: Identity recorded by :class:`DeterministicEmbedder`.
DETERMINISTIC_MODEL_ID = "lit-search/deterministic-hash-embedder"


class EmbeddingError(LitSearchError):
    """Raised when an embedding model cannot be used as declared.

    Covers a missing optional dependency, a model that ships no ``query``
    prompt while being asked for asymmetric retrieval, and a mismatch between
    the dimension a model reports and the dimension it produces.  Raised rather
    than degraded in every case: a query embedded without its prefix, or scored
    against vectors from another model, produces a plausible ordering that is
    wrong, and nothing downstream can tell.
    """


@dataclass(frozen=True, slots=True)
class EmbeddingIdentity:
    """Everything needed to know what a stored vector means.

    Parameters
    ----------
    model_id : str
        Model identifier, for example ``"Qwen/Qwen3-Embedding-0.6B"``.
    dimension : int
        Number of components per vector.  Read from the loaded model, never
        hardcoded: the design reference for this index
        (``TKasperczyk/claude-memory``) fixes ``EMBEDDING_DIM = 4096`` in a
        constants module, builds its Arrow ``FixedSizeList`` from it, and then
        merely *warns* when the configured model reports something else.  That
        is the negative lesson this field exists to avoid.
    normalization : str
        What the model does to its vectors before they leave it, for example
        ``"l2"`` or ``"none"``.  No default, because whether cosine and dot
        product agree depends on it and a defaulted value cannot be told from
        an observed one.  Read from the loaded model rather than assumed.
    model_revision : str | None
        Pinned revision when the run pinned one.  ``None`` records that it did
        not, which is itself worth knowing when two indexes disagree.
    prompt_convention : str
        How query and document text are prefixed, for example
        ``"qwen3:query-instruct/document-plain"`` or ``"none"``.  Recorded
        because it cannot be recovered from the model object afterwards and
        getting it wrong degrades retrieval without any symptom.
    is_semantic : bool
        Whether the vectors carry semantic meaning.  ``False`` marks a
        deterministic stand-in used for tests and for structural checks; the
        semantic deduplication stage refuses such a model rather than
        producing meaningless similarity scores that look like real ones.
    """

    model_id: str
    dimension: int
    normalization: str
    model_revision: str | None = None
    prompt_convention: str = "none"
    is_semantic: bool = True

    def __post_init__(self) -> None:
        """Reject an identity that cannot describe a usable vector space.

        Returns
        -------
        None

        Raises
        ------
        EmbeddingError
            If the model identifier is blank or the dimension is not positive.
        """
        if not self.model_id.strip():
            raise EmbeddingError("An embedding identity must name its model")
        if self.dimension <= 0:
            raise EmbeddingError(
                f"An embedding identity must have a positive dimension, got {self.dimension}"
            )
        if not self.prompt_convention.strip():
            raise EmbeddingError(
                "An embedding identity must record its prompt convention; use 'none' "
                "when the model takes unprefixed text"
            )
        if not self.normalization.strip():
            raise EmbeddingError(
                "An embedding identity must record its normalization; use 'none' when the "
                "model returns unnormalized vectors"
            )

    @property
    def fingerprint(self) -> str:
        """Return a short stable string identifying this vector space.

        Returns
        -------
        str
            Compact identity string, safe to compare for equality and to show
            in a report.
        """
        revision = self.model_revision or "unpinned"
        return (
            f"{self.model_id}@{revision}/d{self.dimension}"
            f"/{self.prompt_convention}/norm={self.normalization}"
        )

    def describe(self) -> str:
        """Return a one-line human-readable description.

        Returns
        -------
        str
            Description including the non-semantic warning when it applies.
        """
        suffix = "" if self.is_semantic else " (NOT SEMANTIC: deterministic stand-in)"
        return f"{self.fingerprint}{suffix}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, Any]
            Serialized identity.
        """
        return {
            "model_id": self.model_id,
            "dimension": self.dimension,
            "normalization": self.normalization,
            "model_revision": self.model_revision,
            "prompt_convention": self.prompt_convention,
            "is_semantic": self.is_semantic,
        }

    @classmethod
    def from_dict(cls, data: Any) -> EmbeddingIdentity:
        """Reconstruct from the format produced by :meth:`to_dict`.

        Parameters
        ----------
        data : Any
            Serialized identity.

        Returns
        -------
        EmbeddingIdentity
            Reconstructed identity.

        Raises
        ------
        EmbeddingError
            If *data* is not a mapping carrying the required keys.
        """
        if not isinstance(data, dict):
            raise EmbeddingError(
                f"An embedding identity must deserialize from a mapping, got {type(data).__name__}"
            )
        missing = {"model_id", "dimension", "normalization"} - set(data)
        if missing:
            raise EmbeddingError(
                f"Serialized embedding identity is missing {sorted(missing)}; refusing to "
                "default it, because a defaulted identity cannot be told from a recorded one"
            )
        return cls(
            model_id=str(data["model_id"]),
            dimension=int(data["dimension"]),
            normalization=str(data["normalization"]),
            model_revision=data.get("model_revision"),
            prompt_convention=str(data.get("prompt_convention", "none")),
            is_semantic=bool(data.get("is_semantic", True)),
        )


@runtime_checkable
class EmbeddingModel(Protocol):
    """A source of vectors, with an identity that travels with them.

    Two encode methods rather than one, because query-to-document retrieval
    and document-to-document similarity are different operations and the models
    that do the first well require an asymmetric prefix.  A single ``encode``
    is how that distinction gets lost.
    """

    @property
    def identity(self) -> EmbeddingIdentity:
        """Return what these vectors mean.

        Returns
        -------
        EmbeddingIdentity
            Model, revision, dimension and prompt convention.
        """
        ...

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages for storage.

        Parameters
        ----------
        texts : Sequence[str]
            Document texts.

        Returns
        -------
        list[list[float]]
            One vector per text, in the same order.
        """
        ...

    def encode_query(self, text: str) -> list[float]:
        """Embed a question for retrieval.

        Parameters
        ----------
        text : str
            Query text.

        Returns
        -------
        list[float]
            One vector.
        """
        ...


class DeterministicEmbedder:
    """A dependency-free, reproducible embedder that is not semantic.

    Exists so that the index, the hybrid retrieval lanes and the diversifier
    can be tested without a model download, and so that a structural test can
    assert dimension handling without pinning a real model's dimension into the
    test suite.

    It declares ``is_semantic=False``, and that declaration is enforced where it
    matters: :func:`lit_search.store.dedup.deduplicate` refuses to run its
    semantic stage with a non-semantic model rather than emitting cosine scores
    that look like measurements.  Query and document vectors are produced from
    *differently salted* hashes, so a test that swaps the two sees a different
    answer -- which is what makes the prompt-convention tests meaningful.
    """

    def __init__(self, dimension: int = 16) -> None:
        """Create a deterministic embedder.

        Parameters
        ----------
        dimension : int
            Vector width.

        Raises
        ------
        EmbeddingError
            If *dimension* is not positive.
        """
        self._identity = EmbeddingIdentity(
            model_id=DETERMINISTIC_MODEL_ID,
            dimension=dimension,
            normalization="l2",
            model_revision=_CACHE_KEY_VERSION,
            prompt_convention="deterministic:query-salt/document-salt",
            is_semantic=False,
        )

    @property
    def identity(self) -> EmbeddingIdentity:
        """Return what these vectors mean.

        Returns
        -------
        EmbeddingIdentity
            Identity declaring ``is_semantic=False``.
        """
        return self._identity

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages deterministically.

        Parameters
        ----------
        texts : Sequence[str]
            Document texts.

        Returns
        -------
        list[list[float]]
            One unit-norm vector per text.
        """
        return [self._vector(text, salt="document") for text in texts]

    def encode_query(self, text: str) -> list[float]:
        """Embed a question deterministically.

        Parameters
        ----------
        text : str
            Query text.

        Returns
        -------
        list[float]
            One unit-norm vector, salted differently from a document vector.
        """
        return self._vector(text, salt="query")

    def _vector(self, text: str, *, salt: str) -> list[float]:
        """Build a unit-norm vector from a salted digest of *text*.

        Token-level hashing rather than whole-string hashing, so that texts
        sharing words are measurably closer than texts that do not.  That keeps
        the diversification and residue tests meaningful without claiming the
        result is semantic.

        Parameters
        ----------
        text : str
            Text to embed.
        salt : str
            Distinguishes the query space from the document space.

        Returns
        -------
        list[float]
            Unit-norm vector of the declared dimension.
        """
        width = self._identity.dimension
        components = [0.0] * width
        tokens = text.lower().split() or [""]
        for token in tokens:
            digest = hashlib.blake2b(f"{salt}|{token}".encode(), digest_size=8).digest()
            (raw,) = struct.unpack("<Q", digest)
            components[raw % width] += 1.0
        norm = math.sqrt(sum(value * value for value in components))
        if norm == 0.0:  # pragma: no cover - a zero vector needs an empty token set
            components[0] = 1.0
            return components
        return [value / norm for value in components]


class SentenceTransformerEmbedder:
    """A ``sentence_transformers`` model, loaded strictly.

    Two refusals rather than warnings:

    * **No ``query`` prompt, no query-mode embedding.**  ``encode_query`` in
      ``sentence_transformers`` silently falls back to a symmetric encode when
      the model ships no ``query`` prompt.  This class checks
      ``model.prompts`` at load time and raises, so a model without a genuine
      asymmetric query mode cannot be used for query-to-document retrieval by
      accident.  Pass ``require_query_prompt=False`` to use such a model
      deliberately, which records ``prompt_convention="none"`` and makes the
      choice visible in every ranking provenance downstream.
    * **The reported dimension is checked against a produced vector.**  A
      declared dimension that does not match the vectors written into the index
      is a mismatch nothing else detects until retrieval quietly stops working.
    """

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        device: str | None = None,
        require_query_prompt: bool = True,
    ) -> None:
        """Load a sentence-transformers model.

        Parameters
        ----------
        model_id : str
            HuggingFace model identifier.
        revision : str | None
            Revision to pin.  ``None`` loads the default and records that no
            revision was pinned.
        device : str | None
            Torch device, for example ``"cuda"``.  ``None`` lets the library
            choose.
        require_query_prompt : bool
            Whether to refuse a model that ships no ``query`` prompt.

        Raises
        ------
        EmbeddingError
            If ``sentence_transformers`` is not installed, if the model ships
            no ``query`` prompt while one is required, or if the model's
            reported dimension does not match a produced vector.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise EmbeddingError(
                "sentence-transformers is not installed, so no real embedding model can "
                "be loaded. Install it with 'pip install \"sentence-transformers>=6.0.0\"' "
                "(>=6.0.0 is required: 5.x cannot load a CrossEncoder-style reranker at all, "
                "and randomises half-precision sigmoid reranker output). Refusing to fall "
                "back to a non-semantic stand-in, which would return a plausible ranking "
                "that means nothing."
            ) from error

        kwargs: dict[str, Any] = {}
        if revision is not None:
            kwargs["revision"] = revision
        if device is not None:
            kwargs["device"] = device
        self._model = SentenceTransformer(model_id, **kwargs)
        prompts = dict(getattr(self._model, "prompts", None) or {})
        has_query_prompt = bool(prompts.get("query"))
        if require_query_prompt and not has_query_prompt:
            raise EmbeddingError(
                f"Model {model_id!r} ships no non-empty 'query' prompt "
                f"(prompts={sorted(prompts)}), so sentence-transformers' encode_query() "
                "would apply no prefix and fall through to a plain symmetric encode with "
                "no error and no warning. That is a silently degraded ranking. Either pick "
                "a model that ships one, or pass require_query_prompt=False to accept "
                "symmetric encoding deliberately -- which is then recorded as "
                "prompt_convention='none' in every ranking built from it."
            )
        dimension = int(self._model.get_sentence_embedding_dimension() or 0)
        convention = f"{model_id}:query-prompt/document-plain" if has_query_prompt else "none"
        self._identity = EmbeddingIdentity(
            model_id=model_id,
            dimension=dimension,
            normalization=self._observed_normalization(),
            model_revision=revision,
            prompt_convention=convention,
            is_semantic=True,
        )
        self._verify_dimension()

    def _observed_normalization(self) -> str:
        """Report whether the loaded pipeline normalizes its output.

        Read from the model's own module list rather than assumed.  Whether
        cosine and dot product agree depends on it, and a model with no
        ``Normalize`` module returns vectors whose dot products are not
        similarities -- which produces a plausible ordering that is wrong.

        Returns
        -------
        str
            ``"l2"`` when a ``Normalize`` module is in the pipeline, otherwise
            a string that says so and names the consequence.
        """
        names = [type(module).__name__ for module in self._model]
        if "Normalize" in names:
            return "l2"
        return f"none (no Normalize module in pipeline: {'+'.join(names)})"

    @property
    def identity(self) -> EmbeddingIdentity:
        """Return what these vectors mean.

        Returns
        -------
        EmbeddingIdentity
            Identity read from the loaded model.
        """
        return self._identity

    def _verify_dimension(self) -> None:
        """Check that a produced vector is as wide as the model claims.

        Returns
        -------
        None

        Raises
        ------
        EmbeddingError
            If the produced width differs from the reported dimension.
        """
        probe = self.encode_documents(["dimension probe"])[0]
        if len(probe) != self._identity.dimension:
            raise EmbeddingError(
                f"Model {self._identity.model_id!r} reports dimension "
                f"{self._identity.dimension} but produced a vector of length {len(probe)}. "
                "Refusing to build an index whose declared width is not its real width."
            )

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages with the document convention.

        Parameters
        ----------
        texts : Sequence[str]
            Document texts.

        Returns
        -------
        list[list[float]]
            One vector per text.
        """
        encoded = self._model.encode_document(list(texts), convert_to_numpy=True)
        return [[float(value) for value in row] for row in encoded]

    def encode_query(self, text: str) -> list[float]:
        """Embed a question with the query convention.

        Parameters
        ----------
        text : str
            Query text.

        Returns
        -------
        list[float]
            One vector.
        """
        encoded = self._model.encode_query([text], convert_to_numpy=True)
        return [float(value) for value in encoded[0]]


class QueryEmbeddingCache:
    """In-memory cache of query vectors, keyed by identity as well as text.

    The key composition is the transferable lesson from
    ``ziwenhahaha/daily-paper-reader``: a version marker, the model identity
    and the query text.  Two additions on top of it.

    * The **prompt convention and revision** are part of the key, not just the
      model name.  Two runs of the same model under different conventions
      produce different vectors, and a key that cannot tell them apart serves
      the wrong one.
    * The stored identity is **re-validated on read**, so a hash collision or a
      hand-edited cache cannot substitute a vector from another space.

    Deliberately not persisted to disk.  The value of caching a query
    embedding is one model call per repeated query in one session; the risk of
    a stale on-disk cache surviving a model change is a silently wrong ranking.
    The asymmetry decides it.
    """

    def __init__(self) -> None:
        """Create an empty cache."""
        self._entries: dict[str, tuple[EmbeddingIdentity, tuple[float, ...]]] = {}
        self._hits = 0
        self._misses = 0

    @staticmethod
    def key_for(identity: EmbeddingIdentity, text: str) -> str:
        """Compose the cache key for one query under one identity.

        Parameters
        ----------
        identity : EmbeddingIdentity
            Identity the vector would be produced under.
        text : str
            Query text.

        Returns
        -------
        str
            Hex digest key.
        """
        payload = f"{_CACHE_KEY_VERSION}|{identity.fingerprint}|{text.strip()}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def statistics(self) -> dict[str, int]:
        """Return hit and miss counts, so a cache can be shown to be working.

        Returns
        -------
        dict[str, int]
            Keys ``hits``, ``misses`` and ``entries``.
        """
        return {"hits": self._hits, "misses": self._misses, "entries": len(self._entries)}

    def embed_query(self, model: EmbeddingModel, text: str) -> list[float]:
        """Return the query vector, from cache when the identity still matches.

        Parameters
        ----------
        model : EmbeddingModel
            Model to embed with.
        text : str
            Query text.

        Returns
        -------
        list[float]
            Query vector.
        """
        identity = model.identity
        key = self.key_for(identity, text)
        cached = self._entries.get(key)
        if cached is not None and cached[0] == identity:
            self._hits += 1
            return list(cached[1])
        if cached is not None:  # pragma: no cover - requires a digest collision
            logger.warning(
                "[index] discarding a cached query vector whose identity (%s) does not "
                "match the active model (%s)",
                cached[0].fingerprint,
                identity.fingerprint,
            )
        self._misses += 1
        vector = model.encode_query(text)
        self._entries[key] = (identity, tuple(vector))
        return vector


__all__ = [
    "DETERMINISTIC_MODEL_ID",
    "DeterministicEmbedder",
    "EmbeddingError",
    "EmbeddingIdentity",
    "EmbeddingModel",
    "QueryEmbeddingCache",
    "SentenceTransformerEmbedder",
]
