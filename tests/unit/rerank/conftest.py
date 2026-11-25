"""Stand-in models for the re-rank tests.

The real models are 1.2 GB each and need a GPU to be quick, and unit tests run
under ``pytest-socket`` with the network blocked.  These fakes implement
exactly the protocol surface declared in :mod:`lit_search.rerank.models`, and
they are built so that a *prefix* mistake changes the numbers -- which is what
lets the prompt tests fail when the prefix is wrong rather than merely pass
when it is right.
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest

from lit_search.core.paper import Paper
from lit_search.rerank.models import BiEncoder, CrossEncoderModel

QWEN_QUERY_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)


def make_paper(
    title: str,
    abstract: str = "",
    doi: str | None = None,
    keywords: set[str] | None = None,
) -> Paper:
    """Build a minimal paper for ranking tests.

    Parameters
    ----------
    title : str
        Paper title.
    abstract : str
        Paper abstract.
    doi : str | None
        DOI, used as the identity when present.
    keywords : set[str] | None
        Author keywords.

    Returns
    -------
    Paper
        The record.
    """
    return Paper(
        title=title,
        abstract=abstract,
        authors=[],
        source=None,
        publication_date=datetime.date(2026, 4, 1),
        doi=doi,
        keywords=keywords,
    )


class FakeSentenceEncoder:
    """A deterministic encoder whose output depends on the applied prefix.

    Embeddings are a hashed bag-of-characters projection, L2-normalized.  The
    only property the tests rely on is that the vector is a function of the
    *full string including any prefix*, so an omitted or wrong prefix produces
    a different vector -- exactly as it does with a real model, and exactly
    what makes the prefix mistake detectable.
    """

    def __init__(
        self,
        prompts: dict[str, str | None] | None = None,
        dimension: int = 16,
        normalize: bool = True,
    ) -> None:
        """Configure the fake.

        Parameters
        ----------
        prompts : dict[str, str] | None
            Prompt dictionary.  Omitting ``"query"`` reproduces the model class
            that makes ``encode_query`` a silent no-op.
        dimension : int
            Embedding width to report and produce.
        normalize : bool
            Whether to L2-normalize, so the normalization probe can be tested
            in both directions.
        """
        self.prompts: dict[str, str | None] = (
            {"query": QWEN_QUERY_PROMPT, "document": ""} if prompts is None else prompts
        )
        self._dimension = dimension
        self._normalize = normalize
        self.max_seq_length = 512
        #: Every string actually handed to the model, prefix included.  The
        #: prompt tests assert against this rather than against a mock call
        #: record, because what matters is the text the model saw.
        self.seen: list[str] = []

    def get_embedding_dimension(self) -> int:
        """Report the declared embedding width.

        Returns
        -------
        int
            The width.
        """
        return self._dimension

    def _embed(self, text: str) -> np.ndarray:
        """Project one string to a deterministic vector.

        Parameters
        ----------
        text : str
            Text to embed, prefix included.

        Returns
        -------
        np.ndarray
            The vector.
        """
        vector = np.zeros(self._dimension, dtype=np.float64)
        for position, character in enumerate(text):
            vector[(ord(character) + position) % self._dimension] += 1.0
        magnitude = np.linalg.norm(vector)
        if self._normalize and magnitude:
            vector = vector / magnitude
        elif not self._normalize:
            vector = vector * 3.0
        return vector

    def _encode(self, inputs, prefix: str):
        """Embed a batch of strings under a prefix.

        Parameters
        ----------
        inputs : Sequence[str]
            Texts.
        prefix : str
            Prefix applied to every text.

        Returns
        -------
        np.ndarray
            Matrix of vectors.
        """
        texts = [prefix + text for text in inputs]
        self.seen.extend(texts)
        return np.stack([self._embed(text) for text in texts])

    def encode_query(self, inputs, **kwargs):
        """Encode as a query, mirroring sentence-transformers' own logic.

        The prefix is applied only when a ``"query"`` prompt exists.  That
        conditional is copied deliberately: it is the behaviour that makes the
        call a silent no-op on a model without one.

        Parameters
        ----------
        inputs : Sequence[str]
            Texts.
        **kwargs : Any
            Ignored.

        Returns
        -------
        np.ndarray
            Matrix of vectors.
        """
        # ``or ""`` mirrors sentence-transformers: a prompt key whose value is
        # None resolves to no prefix at all, which is the exact hole the
        # measured-asymmetry check exists to close.
        return self._encode(inputs, self.prompts.get("query") or "")

    def encode_document(self, inputs, **kwargs):
        """Encode as a document.

        Parameters
        ----------
        inputs : Sequence[str]
            Texts.
        **kwargs : Any
            Ignored.

        Returns
        -------
        np.ndarray
            Matrix of vectors.
        """
        return self._encode(inputs, self.prompts.get("document") or "")


class FakePairScorer:
    """A cross-encoder stand-in that scores by shared vocabulary.

    Records the instruction it was given, because whether the seed-mode
    instruction actually replaces the model's query-retrieval default is a
    correctness question, not a cosmetic one.
    """

    def __init__(self, default_prompt: str = "Given a web search query") -> None:
        """Configure the fake.

        Parameters
        ----------
        default_prompt : str
            Instruction applied when the caller supplies none.
        """
        self.prompts = {"query": default_prompt}
        self.default_prompt_name = "query"
        self.activation_fn: object = Identity()
        self.max_seq_length = 1024
        #: Instructions seen, one entry per ``predict`` call.
        self.instructions: list[str | None] = []
        #: Pairs seen, flattened across calls.
        self.pairs: list[tuple[str, str]] = []

    def predict(self, inputs, **kwargs):
        """Score pairs by token overlap.

        Parameters
        ----------
        inputs : Sequence[tuple[str, str]]
            Pairs of texts.
        **kwargs : Any
            ``prompt`` is recorded; the rest is ignored.

        Returns
        -------
        list[float]
            One score per pair.
        """
        self.instructions.append(kwargs.get("prompt"))
        self.pairs.extend(inputs)
        scores = []
        for left, right in inputs:
            left_tokens = set(left.lower().split())
            right_tokens = set(right.lower().split())
            scores.append(float(len(left_tokens & right_tokens)))
        return scores


class Identity:
    """Stands in for ``torch.nn.Identity`` so the activation name reads right."""


class Sigmoid:
    """Stands in for ``torch.nn.Sigmoid``, whose name triggers the bf16 warning."""


def blank_title(paper: Paper) -> Paper:
    """Return *paper* with its title emptied.

    ``Paper.__init__`` refuses an empty title, so a title-less record cannot be
    constructed -- but ``title`` is a public mutable attribute that
    ``Paper.merge`` and the enrichment paths write to, so an emptied title is
    still a reachable state.  Emptying it here tests the path that can actually
    occur rather than one the constructor forbids.

    Parameters
    ----------
    paper : Paper
        Record to modify in place.

    Returns
    -------
    Paper
        The same record, title emptied.
    """
    paper.title = ""
    return paper


@pytest.fixture
def bi_encoder() -> BiEncoder:
    """Return a loaded-looking bi-encoder over the fake encoder."""
    model = FakeSentenceEncoder()
    return BiEncoder(
        model=model,
        model_id="fake/bi-encoder",
        revision="feedface",
        dimension=16,
        normalization="l2",
        query_prompt=model.prompts["query"] or "",
        document_prompt=model.prompts["document"] or "",
        max_seq_length=512,
    )


@pytest.fixture
def cross_encoder() -> CrossEncoderModel:
    """Return a loaded-looking cross-encoder over the fake pair scorer."""
    model = FakePairScorer()
    return CrossEncoderModel(
        model=model,
        model_id="fake/cross-encoder",
        revision="c0ffee",
        default_prompt=model.prompts["query"],
        activation="Identity",
        max_seq_length=1024,
    )


@pytest.fixture
def candidates() -> list[Paper]:
    """Return a small candidate set with one deliberately degraded record."""
    return [
        make_paper(
            "Scaling formal methods via LLM-based Hoare-style reasoning",
            "We derive a function specification from its call sites so the specification "
            "reflects developer intent even when the implementation is buggy.",
            doi="10.1000/fm-agent",
            keywords={"model checking", "specification inference"},
        ),
        make_paper(
            "Butterfly wing pigmentation across altitudes",
            "A field survey of pigmentation gradients in montane lepidoptera populations "
            "sampled over three seasons.",
            doi="10.1000/butterfly",
        ),
        make_paper(
            "Agentic model checking with dual-source generation",
            "Caller-intent and implementation-behaviour prompts generate candidate "
            "specifications, and disagreements between them are flagged for review.",
            doi="10.1000/bmc-agent",
        ),
        # No abstract at all: the degraded case, present on purpose.
        make_paper("Specula: heterogeneous repository evidence", "", doi="10.1000/specula"),
    ]
