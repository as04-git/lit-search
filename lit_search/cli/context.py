"""Resolving the handle store, the index root and the embedding model.

One place, because all three are choices a user can get wrong silently.  The
embedding model in particular: a run that quietly fell back to a stand-in model
would return a plausible ranking with nothing behind it, so the fallback is
never quiet and never automatic.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from lit_search.index.embedding import (
    DeterministicEmbedder,
    EmbeddingModel,
    SentenceTransformerEmbedder,
)
from lit_search.index.paper_index import PaperIndex
from lit_search.store.store import HandleStore, default_store_root

#: Environment variable naming the default embedding model.
MODEL_ENV_VAR = "LIT_SEARCH_EMBEDDING_MODEL"

#: Model used when neither the flag nor the environment names one.
#:
#: ``O7``'s answer as of 2026-08-30, pending owner ratification: best of 14
#: candidates at this size on MTEB scientific retrieval, and it ships real
#: ``query`` / ``document`` prompts, which
#: :class:`~lit_search.index.embedding.SentenceTransformerEmbedder` requires.
#: It is a default, not a decision: the flag and the environment variable both
#: override it, and whatever runs is recorded in the index manifest and in
#: every ranking provenance.
DEFAULT_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

#: Subdirectory of the store root holding the index.
INDEX_SUBDIRECTORY = "index"

#: Dimension of the deterministic stand-in.  Small on purpose: it exists to
#: exercise plumbing, and a wide vector would only make the tests slower.
DETERMINISTIC_DIMENSION = 64


def resolve_store(args: argparse.Namespace) -> HandleStore:
    """Return the handle store for this invocation.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments; ``--store-root`` overrides the default location.

    Returns
    -------
    HandleStore
        Store rooted at the resolved path.
    """
    root = getattr(args, "store_root", None)
    return HandleStore(root) if root else HandleStore()


def resolve_index_root(args: argparse.Namespace) -> Path:
    """Return the directory the paper index lives in.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments; ``--index-root`` overrides the default location.

    Returns
    -------
    Path
        Index root.
    """
    override = getattr(args, "index_root", None)
    if override:
        return Path(override).expanduser()
    store_root = getattr(args, "store_root", None)
    base = Path(store_root).expanduser() if store_root else default_store_root()
    return base / INDEX_SUBDIRECTORY


def resolve_embedder(args: argparse.Namespace) -> EmbeddingModel:
    """Return the embedding model for this invocation.

    The deterministic stand-in is reachable only through an explicit flag.
    There is no automatic fallback to it when the real model cannot be loaded:
    a search that silently ranked by a non-semantic hash would look exactly
    like one that ranked by meaning.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    EmbeddingModel
        Loaded model.

    Raises
    ------
    EmbeddingError
        If the real model cannot be loaded, including when
        ``sentence-transformers`` is not installed.
    """
    if getattr(args, "deterministic_embedder", False):
        return DeterministicEmbedder(dimension=DETERMINISTIC_DIMENSION)
    model_id = getattr(args, "model", None) or os.environ.get(MODEL_ENV_VAR) or DEFAULT_MODEL_ID
    return SentenceTransformerEmbedder(
        model_id,
        revision=getattr(args, "model_revision", None),
        require_query_prompt=not getattr(args, "allow_symmetric_encoding", False),
    )


def open_index(args: argparse.Namespace, embedder: EmbeddingModel) -> PaperIndex:
    """Open the paper index for this invocation.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    embedder : EmbeddingModel
        Model whose identity the index is pinned to.

    Returns
    -------
    PaperIndex
        Opened index.

    Raises
    ------
    PaperIndexError
        If the index was built with a different model.
    """
    return PaperIndex(resolve_index_root(args), embedder)


__all__ = [
    "DEFAULT_MODEL_ID",
    "DETERMINISTIC_DIMENSION",
    "INDEX_SUBDIRECTORY",
    "MODEL_ENV_VAR",
    "open_index",
    "resolve_embedder",
    "resolve_index_root",
    "resolve_store",
]
