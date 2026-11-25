"""Loading the two re-ranking models, and refusing the ones that cannot work.

Model choices are the most version-sensitive thing in this project, so the
identities below carry the date and the evidence rather than an assurance.
Everything in this module's docstring was read from a live source on
**2026-08-31**, not recalled.

The two models
--------------

**Bi-encoder: ``Qwen/Qwen3-Embedding-0.6B``**, revision
``97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3``, apache-2.0, ``lastModified``
2026-04-20, 6,808,765 downloads in the trailing 30 days.  Chosen because it
has a *genuine* asymmetric query mode, verified rather than assumed: its
``config_sentence_transformers.json`` ships

    query    = "Instruct: Given a web search query, retrieve relevant
                passages that answer the query\\nQuery:"
    document = ""

and a live load confirmed ``encode_query(t) != encode_document(t)`` for the
same ``t``, at cosine 0.8399.  Dimension 1024, ``max_position_embeddings``
32768, last-token pooling, and a ``Normalize`` module so embeddings arrive
L2-normalized and cosine equals dot product.

**Cross-encoder: ``Qwen/Qwen3-Reranker-0.6B``**, revision
``e61197ed45024b0ed8a2d74b80b4d909f1255473``, apache-2.0, ``lastModified``
2026-04-16.  It ships ``config_sentence_transformers.json``, ``modules.json``
and a ``1_LogitScore/`` module, so ``CrossEncoder(...)`` loads it natively with
no ``trust_remote_code``.  The widely-cited ``tomaarsen/...-seq-cls``
workaround is obsolete; it still carries 183,058 downloads, so stale tutorials
will keep recommending it.  ``activation_fn`` is ``Identity``, so scores are
raw logits and the half-precision sigmoid collapse that afflicts
sentence-transformers 5.x does not apply here.

Alternatives checked live and rejected
--------------------------------------

- ``jinaai/jina-reranker-v3`` **and** ``jinaai/jina-reranker-v3.5``
  (``lastModified`` 2026-07-30): both **cc-by-nc-4.0**, non-commercial.  The
  v3.5 sibling is not in the project record and is rejected for the same
  reason as v3.
- ``zeroentropy/zerank-2-reranker``: 8.04 GB, larger than the whole card.
- ``Qwen/Qwen3-Reranker-4B``: apache-2.0 and the most-downloaded of the
  family, but ~8 GB of bf16 weights against 8.0 GiB total VRAM, so it cannot
  be co-resident with the embedder.
- ``infgrad/Prism-Qwen3.5-Reranker-0.8B`` (MIT, ``lastModified`` 2026-04-29):
  **a real alternative the project record does not mention.**  It ships the
  same ``1_LogitScore`` layout, so it loads through this module unchanged by
  passing ``model_id``.  Not adopted: single-author repo, no measured
  comparison on scientific corpora, and 0.8B against 0.6B for no demonstrated
  gain.  Recorded so the option is not lost.
- No ``Qwen3.5`` or ``Qwen4`` reranker exists under the ``Qwen`` org.

What is *not* established
-------------------------

There is still no measured head-to-head of these rerankers on scientific
corpora from a primary source.  The cross-encoder choice rests on license,
footprint, and native-API fit -- not on measured quality.  And the
bi-encoder's aggregate lead on MTEB scientific retrieval is driven largely by
TRECCOVID; it is the weakest model in its comparison set on SciFact.  Both
model identities are therefore parameters, not constants.

The assertion this module exists for
------------------------------------

Read from the installed ``sentence_transformers`` 6.0.1 source:

.. code-block:: python

    if prompt_name is None and prompt is None and "query" in self.prompts:
        prompt_name = "query"

**If the model ships no ``query`` prompt, ``encode_query()`` applies no prefix
and falls through to a plain symmetric encode.  No error, no warning.**  So
calling ``encode_query`` is necessary and not sufficient: it is a no-op on
exactly the models that need it most, and the resulting badly-ordered list is
indistinguishable from a well-ordered one.  :func:`load_bi_encoder` therefore
asserts the prompt exists and raises
:exc:`~lit_search.exceptions.ModelContractError` when it does not.

**That assertion is necessary and still not sufficient**, which a type check
surfaced rather than a reading: ``sentence_transformers`` types prompt values
as ``str | None``, so a model can carry a ``query`` key whose value is null or
empty.  ``"query" in prompts`` passes, no prefix is applied, and the silent
no-op arrives through the one door the key check cannot close.  So the loader
also **measures**: it encodes one probe text both ways and refuses the model
when the two vectors are the same.  Measurement subsumes inspection here -- it
also admits a model that is asymmetric through a task router with no prefix at
all, which the key check alone would have wrongly accepted as fine and the
prefix check alone would have wrongly rejected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from lit_search.exceptions import ModelContractError

logger = logging.getLogger(__name__)

#: Default bi-encoder, with the revision it was verified at.
DEFAULT_BI_ENCODER = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_BI_ENCODER_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"

#: Default cross-encoder, with the revision it was verified at.
DEFAULT_CROSS_ENCODER = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_CROSS_ENCODER_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"

#: Name of the prompt entry a bi-encoder must ship to have a query mode.
QUERY_PROMPT_NAME = "query"
#: Name of the prompt entry used for documents.  Absence is tolerated: an
#: absent document prompt means "no document prefix", which is a coherent
#: convention.  An absent *query* prompt is not.
DOCUMENT_PROMPT_NAME = "document"

#: Tolerance for calling an embedding L2-normalized.  Generous because the
#: probe is encoded in bfloat16, where a unit vector lands within about 1e-3.
_NORM_TOLERANCE = 1e-2

#: Text used for the load-time probe.  Content is irrelevant; it only has to
#: be non-empty and stable.
_PROBE_TEXT = "specification inference from call sites"

#: Above this cosine, the query and document encodings of one text are the same
#: vector and the model's query mode is nominal.  The chosen bi-encoder
#: measures 0.8399 here (verified 2026-08-31), so the threshold has a very wide
#: margin and only fires on a model that is genuinely symmetric.
_SYMMETRY_THRESHOLD = 0.9999


class SentenceEncoder(Protocol):
    """The subset of ``SentenceTransformer`` this package depends on.

    Declared as a protocol so the pipeline can be tested against a stand-in
    without a 1.2 GB download and without a GPU, and so that the exact surface
    being relied on is written down.
    """

    # Values are optional because sentence-transformers types them that way: a
    # model may ship a prompt key whose value is null. That matters here --
    # ``"query" in prompts`` would pass while no prefix is applied -- and it is
    # why the query mode is verified by measurement as well as by inspection.
    prompts: dict[str, str | None]

    def encode_query(self, inputs: Any, **kwargs: Any) -> Any:
        """Encode text as a query."""
        ...

    def encode_document(self, inputs: Any, **kwargs: Any) -> Any:
        """Encode text as a document."""
        ...


class PairScorer(Protocol):
    """The subset of ``CrossEncoder`` this package depends on."""

    def predict(self, inputs: Any, **kwargs: Any) -> Any:
        """Score query/document pairs jointly."""
        ...


@dataclass(frozen=True)
class BiEncoder:
    """A loaded bi-encoder together with the facts a ranking must record.

    Attributes
    ----------
    model : SentenceEncoder
        The loaded model.
    model_id : str
        Repository identifier, for example ``"Qwen/Qwen3-Embedding-0.6B"``.
    revision : str | None
        Pinned revision, or ``None`` when the caller declined to pin one --
        which is itself worth recording, because two runs against an unpinned
        model can silently use different weights.
    dimension : int
        Embedding width, **read from the loaded model and confirmed against a
        probe vector**, never assumed.
    normalization : str
        ``"l2"`` or ``"none"``, determined by measuring a probe vector rather
        than by inspecting the module list.  Whether cosine and dot product
        agree depends on this, and a ``truncate_dim`` or a swapped module can
        change it without changing the module names.
    query_prompt : str
        The literal query prefix the model will apply.
    document_prompt : str
        The literal document prefix, empty string when the model applies none.
    max_seq_length : int
        Token budget; text beyond it is truncated, and truncated text is
        partly unscored.
    """

    model: SentenceEncoder
    model_id: str
    revision: str | None
    dimension: int
    normalization: str
    query_prompt: str
    document_prompt: str
    max_seq_length: int

    def prompt_convention(self, *, symmetric: bool) -> str:
        """Describe the prompting actually in force, literally.

        The literal prefixes are recorded rather than a label such as
        ``"qwen3"``, because a label cannot be checked afterwards and the
        prefixes cannot be recovered from a stored handle.  A wrong prefix is
        then visible in the provenance instead of being invisible in the
        scores.

        Parameters
        ----------
        symmetric : bool
            ``True`` when both sides were encoded as documents, which is what
            seed-mode does.

        Returns
        -------
        str
            Human-readable, machine-diffable description.
        """
        document = _escape(self.document_prompt)
        if symmetric:
            return f"{self.model_id}: both sides encode_document(prompt={document})"
        return (
            f"{self.model_id}: encode_query(prompt={_escape(self.query_prompt)}) / "
            f"encode_document(prompt={document})"
        )


@dataclass(frozen=True)
class CrossEncoderModel:
    """A loaded cross-encoder together with the facts a ranking must record.

    Attributes
    ----------
    model : PairScorer
        The loaded model.
    model_id : str
        Repository identifier.
    revision : str | None
        Pinned revision, or ``None``.
    default_prompt : str
        The instruction the model applies when the caller supplies none, read
        off the model.  Recorded because passing ``prompt=""`` does **not**
        suppress it -- observed on 2026-08-31, an empty prompt produced scores
        identical to the default.
    activation : str
        Name of the output activation.  ``"Identity"`` means raw logits, which
        is what makes bfloat16 safe here; a sigmoid in bfloat16 saturates and
        collapses top candidates onto tied values, randomizing their order.
    max_seq_length : int
        Token budget for the concatenated pair.
    """

    model: PairScorer
    model_id: str
    revision: str | None
    default_prompt: str
    activation: str
    max_seq_length: int

    def prompt_convention(self, instruction: str | None) -> str:
        """Describe the instruction that will actually be applied.

        Parameters
        ----------
        instruction : str | None
            Caller-supplied instruction, or ``None`` to use the model default.

        Returns
        -------
        str
            Description naming the effective instruction verbatim.
        """
        effective = self.default_prompt if instruction is None else instruction
        origin = "model default" if instruction is None else "caller override"
        return f"{self.model_id}: instruction={_escape(effective)} ({origin})"


def _escape(value: str) -> str:
    """Render a prompt as a single-line quoted literal.

    Parameters
    ----------
    value : str
        Prompt text, possibly containing newlines.

    Returns
    -------
    str
        Quoted, newline-escaped text.
    """
    return "'" + value.replace("\\", "\\\\").replace("\n", "\\n") + "'"


def _read_dimension(model: Any) -> int:
    """Read the embedding width from the model, tolerating the rename.

    ``get_sentence_embedding_dimension`` was renamed to
    ``get_embedding_dimension`` in sentence-transformers 6.0.1 and the old name
    now emits a ``FutureWarning``.  Both are tried so that neither a pinned
    older release nor a future removal turns into a crash here.

    Parameters
    ----------
    model : Any
        Loaded sentence encoder.

    Returns
    -------
    int
        Declared embedding width.

    Raises
    ------
    ModelContractError
        If the model declares no dimension at all.
    """
    for attribute in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        reader = getattr(model, attribute, None)
        if reader is None:
            continue
        value = reader()
        if value:
            return int(value)
    raise ModelContractError(
        "The bi-encoder does not report an embedding dimension. The dimension must be "
        "read from the model and stored with the index; hardcoding it is how an index "
        "and a model silently stop matching."
    )


def _cosine_between(left: Any, right: Any) -> float:
    """Cosine similarity of two vectors, without requiring numpy.

    Parameters
    ----------
    left : Any
        First vector.
    right : Any
        Second vector.

    Returns
    -------
    float
        Cosine similarity, or ``0.0`` when either vector is degenerate.
    """
    dot = sum(float(a) * float(b) for a, b in zip(left, right, strict=True))
    left_norm = sum(float(a) ** 2 for a in left) ** 0.5
    right_norm = sum(float(b) ** 2 for b in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return float(dot) / float(left_norm * right_norm)


def _probe(model: Any) -> tuple[int, str, float]:
    """Encode one text both ways, to measure what the model actually does.

    Reading the module list and the prompt dictionary reports what the model
    *says*.  This reports what it did, which is the claim being recorded.
    Three facts come out of one pair of forward passes:

    - the real embedding width, to be checked against the declared one;
    - whether vectors arrive L2-normalized, which decides whether cosine and
      dot product agree;
    - whether ``encode_query`` and ``encode_document`` actually diverge.

    The third is the important one.  A model can satisfy every declarative
    check and still be symmetric -- a ``query`` prompt whose value is null or
    empty passes ``"query" in prompts`` and applies no prefix -- and a model
    can be genuinely asymmetric through a task router with no prefix at all.
    Measuring settles both cases; inspecting settles neither.

    Parameters
    ----------
    model : Any
        Loaded sentence encoder.

    Returns
    -------
    tuple[int, str, float]
        Observed width, ``"l2"`` or ``"none"``, and the cosine between the
        same text encoded as a query and as a document.
    """
    as_document = model.encode_document([_PROBE_TEXT])[0]
    as_query = model.encode_query([_PROBE_TEXT])[0]
    magnitude = float(sum(float(component) ** 2 for component in as_document) ** 0.5)
    normalization = "l2" if abs(magnitude - 1.0) <= _NORM_TOLERANCE else "none"
    return len(as_document), normalization, _cosine_between(as_query, as_document)


def load_bi_encoder(
    model_id: str = DEFAULT_BI_ENCODER,
    *,
    revision: str | None = DEFAULT_BI_ENCODER_REVISION,
    device: str | None = None,
    dtype: str | None = None,
) -> BiEncoder:
    """Load a bi-encoder and refuse it if it has no real query mode.

    Parameters
    ----------
    model_id : str
        Repository identifier.
    revision : str | None
        Revision to pin.  Pass ``None`` to accept whatever the hub currently
        serves; that is recorded rather than hidden.
    device : str | None
        Torch device string.  ``None`` lets sentence-transformers choose.
    dtype : str | None
        Torch dtype name, for example ``"bfloat16"``.  ``None`` uses the
        model's own default.

    Returns
    -------
    BiEncoder
        Loaded model with its recorded identity, dimension, normalization, and
        prompt literals.

    Raises
    ------
    ModelContractError
        If the model ships no ``query`` prompt, if it reports no dimension, or
        if the dimension it reports disagrees with the vector it produced.
    """
    from sentence_transformers import SentenceTransformer

    kwargs: dict[str, Any] = {}
    if dtype is not None:
        kwargs["model_kwargs"] = {"dtype": dtype}
    model = SentenceTransformer(model_id, revision=revision, device=device, **kwargs)

    prompts: dict[str, str | None] = dict(getattr(model, "prompts", {}) or {})
    if QUERY_PROMPT_NAME not in prompts:
        raise ModelContractError(
            f"Bi-encoder {model_id!r} ships no {QUERY_PROMPT_NAME!r} prompt, so it has no "
            f"asymmetric query mode. sentence-transformers' encode_query() applies a prefix "
            f"only when that prompt exists and otherwise performs a plain symmetric encode "
            f"with no error, which would produce a confident and badly ordered ranking. "
            f"Available prompts: {sorted(prompts)}. Use a model with a genuine query mode "
            f"(the default is {DEFAULT_BI_ENCODER}), or SPECTER2 with its 'adhoc_query' "
            f"adapter -- never its 'proximity' adapter, which is document-to-document."
        )

    declared = _read_dimension(model)
    observed, normalization, asymmetry = _probe(model)
    if declared != observed:
        raise ModelContractError(
            f"Bi-encoder {model_id!r} declares dimension {declared} but produced a vector of "
            f"width {observed}. Storing the declared value would leave the index and the "
            f"model disagreeing with nothing to show it."
        )
    if asymmetry > _SYMMETRY_THRESHOLD:
        raise ModelContractError(
            f"Bi-encoder {model_id!r} declares a {QUERY_PROMPT_NAME!r} prompt but encodes the "
            f"same text almost identically as a query and as a document (cosine "
            f"{asymmetry:.6f}). Its query mode is nominal, not real -- a null or empty prompt "
            f"value satisfies the declaration and applies no prefix. Using it would compare "
            f"document-space vectors and call the result query-to-document retrieval."
        )

    encoder = BiEncoder(
        model=model,
        model_id=model_id,
        revision=revision,
        dimension=observed,
        normalization=normalization,
        query_prompt=prompts[QUERY_PROMPT_NAME] or "",
        document_prompt=prompts.get(DOCUMENT_PROMPT_NAME) or "",
        max_seq_length=int(getattr(model, "max_seq_length", 0) or 0),
    )
    logger.info(
        "[rerank] bi-encoder %s revision=%s dim=%d normalization=%s max_seq_length=%d",
        encoder.model_id,
        encoder.revision or "(unpinned)",
        encoder.dimension,
        encoder.normalization,
        encoder.max_seq_length,
    )
    if revision is None:
        logger.warning(
            "[rerank] bi-encoder %s was loaded without a pinned revision. Embeddings from "
            "two revisions must never be compared, and nothing downstream can tell them "
            "apart.",
            model_id,
        )
    return encoder


def load_cross_encoder(
    model_id: str = DEFAULT_CROSS_ENCODER,
    *,
    revision: str | None = DEFAULT_CROSS_ENCODER_REVISION,
    device: str | None = None,
    dtype: str | None = None,
) -> CrossEncoderModel:
    """Load a cross-encoder for stage two.

    Parameters
    ----------
    model_id : str
        Repository identifier.
    revision : str | None
        Revision to pin, or ``None``.
    device : str | None
        Torch device string.
    dtype : str | None
        Torch dtype name.

    Returns
    -------
    CrossEncoderModel
        Loaded model with its recorded identity, default instruction, and
        output activation.
    """
    from sentence_transformers import CrossEncoder

    kwargs: dict[str, Any] = {}
    if dtype is not None:
        kwargs["model_kwargs"] = {"dtype": dtype}
    model = CrossEncoder(model_id, revision=revision, device=device, **kwargs)

    prompts = dict(getattr(model, "prompts", {}) or {})
    default_name = getattr(model, "default_prompt_name", None)
    default_prompt = prompts.get(default_name, "") if default_name else ""
    activation = type(getattr(model, "activation_fn", None)).__name__

    scorer = CrossEncoderModel(
        model=model,
        model_id=model_id,
        revision=revision,
        default_prompt=default_prompt,
        activation=activation,
        max_seq_length=int(getattr(model, "max_seq_length", 0) or 0),
    )
    logger.info(
        "[rerank] cross-encoder %s revision=%s activation=%s default_instruction=%s",
        scorer.model_id,
        scorer.revision or "(unpinned)",
        scorer.activation,
        _escape(scorer.default_prompt),
    )
    if activation.lower().startswith("sigmoid"):
        logger.warning(
            "[rerank] cross-encoder %s uses a sigmoid activation. In half precision a "
            "sigmoid saturates and collapses top candidates onto tied values, which "
            "randomizes their order with no symptom. Use float32, or an Identity "
            "activation.",
            model_id,
        )
    return scorer
