"""The Arrow schema of the paper index, built from the model's real dimension.

The schema is a function of the loaded embedding model, not a constant.  That
is the whole point of this module.  The design reference for this index,
``TKasperczyk/claude-memory``, defines ``EMBEDDING_DIM = 4096`` in a constants
module, builds its ``FixedSizeList`` field from it, and then only *warns* when
the configured model reports a different width -- and names the hardcoding as a
known limitation in its own notes.  Here the dimension comes from
:class:`~lit_search.index.embedding.EmbeddingIdentity`, which read it from the
model.

Every row carries the **full model identity**, not just the dimension.  Two
different models can both produce 1024-dimension unit vectors, so a dimension
check passes while the vectors are mutually meaningless.  Storing
``model_id``, ``model_revision`` and ``prompt_convention`` per row means a
mixed-model table is detectable after the fact -- which matters because the
mixing is what a partly-completed re-index produces, and the symptom of
comparing across two vector spaces is a plausible ranking rather than an error.

Row metadata beyond the vector, and one deliberate omission
-----------------------------------------------------------

Kept, because each answers a question a report has to answer:

* provenance -- ``found_in``, ``source_handle``, ``indexed_at``
* measurement quality -- ``text_policy``, ``missing_fields`` (``R16``)
* the version relation -- ``preprint_doi`` beside ``doi`` (``D24``)

**Not** kept: the reference's per-row usage counters (``retrieval_count``,
``success_count``).  Nothing in this tool ranks by them, and maintaining them
means a write on every read, which turns a read-mostly index into a contended
one for no gain.  If a future feature ranks by usage, it should add them then.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from typing import TYPE_CHECKING, Any

from lit_search.core.identity import record_identity_keys
from lit_search.index.embedding import EmbeddingIdentity
from lit_search.index.text import PaperText, TextPolicy
from lit_search.utils.normalization import normalize_doi

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pyarrow as pa  # type: ignore[import-untyped]

    from lit_search.core.paper import Paper

#: Column holding the vector.  Named ``vector`` because that is LanceDB's
#: default and diverging from it means passing ``vector_column_name`` on every
#: single query, which is one forgettable argument too many.
VECTOR_COLUMN = "vector"

#: Column holding the normalised text: embedded, and full-text indexed.
SEARCH_TEXT_COLUMN = "search_text"

#: Column holding raw text for literal ``LIKE`` matching.
EXACT_TEXT_COLUMN = "exact_text"

#: Column holding the stable per-record key.
KEY_COLUMN = "paper_key"

#: Every column the index writes, in schema order.
COLUMNS: tuple[str, ...] = (
    KEY_COLUMN,
    "title",
    "doi",
    "preprint_doi",
    "year",
    "venue",
    "found_in",
    SEARCH_TEXT_COLUMN,
    EXACT_TEXT_COLUMN,
    "text_policy",
    "missing_fields",
    "model_id",
    "model_revision",
    "prompt_convention",
    "embedding_dim",
    "source_handle",
    "indexed_at",
    VECTOR_COLUMN,
)

#: Columns returned by a query.  The vector itself is excluded by default: it
#: is the largest column by far and no caller reads it except the diversifier,
#: which asks for it explicitly.
PROJECTION: tuple[str, ...] = tuple(column for column in COLUMNS if column != VECTOR_COLUMN)


def paper_key(paper: Paper) -> str:
    """Return the stable index key for *paper*.

    Explicit DOI, arXiv, Semantic Scholar or OpenAlex identity is preferred.
    DOI keys retain their historical bare-DOI spelling for index compatibility.
    An identifier-free record receives a digest of its complete serialized
    snapshot, never a title fallback.  That ``snapshot:`` key is local index
    bookkeeping, not a claim that matching content establishes scholarly-work
    identity; a changed snapshot intentionally receives a different row key.

    Parameters
    ----------
    paper : Paper
        Paper to key.

    Returns
    -------
    str
        Explicit record key, or a ``snapshot:`` content digest when no explicit
        identity is available.
    """
    keys = record_identity_keys(paper)
    if keys:
        if paper.doi:
            return (normalize_doi(paper.doi) or paper.doi.strip()).lower()
        return sorted(keys)[0]

    persisted = type(paper).from_dict(paper.to_dict()).to_dict()
    payload = json.dumps(
        persisted,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()
    return f"snapshot:{digest}"


def build_schema(identity: EmbeddingIdentity) -> pa.Schema:
    """Build the Arrow schema for an index over *identity*'s vector space.

    Parameters
    ----------
    identity : EmbeddingIdentity
        Identity whose dimension the vector column is sized from.

    Returns
    -------
    pa.Schema
        Schema with a ``FixedSizeList`` vector column of the model's width.
    """
    import pyarrow as pa

    return pa.schema(
        [
            pa.field(KEY_COLUMN, pa.string()),
            pa.field("title", pa.string()),
            pa.field("doi", pa.string()),
            pa.field("preprint_doi", pa.string()),
            pa.field("year", pa.int32()),
            pa.field("venue", pa.string()),
            pa.field("found_in", pa.string()),
            pa.field(SEARCH_TEXT_COLUMN, pa.string()),
            pa.field(EXACT_TEXT_COLUMN, pa.string()),
            pa.field("text_policy", pa.string()),
            pa.field("missing_fields", pa.string()),
            pa.field("model_id", pa.string()),
            pa.field("model_revision", pa.string()),
            pa.field("prompt_convention", pa.string()),
            pa.field("embedding_dim", pa.int32()),
            pa.field("source_handle", pa.string()),
            pa.field("indexed_at", pa.string()),
            pa.field(VECTOR_COLUMN, pa.list_(pa.float32(), identity.dimension)),
        ]
    )


def build_row(
    paper: Paper,
    text: PaperText,
    vector: list[float],
    *,
    identity: EmbeddingIdentity,
    policy: TextPolicy,
    source_handle: str | None,
    indexed_at: datetime.datetime | None = None,
) -> dict[str, Any]:
    """Build one index row.

    Parameters
    ----------
    paper : Paper
        Paper being indexed.
    text : PaperText
        Its built strings and missing-field record.
    vector : list[float]
        Its embedding.
    identity : EmbeddingIdentity
        Identity the vector was produced under, written into the row.
    policy : TextPolicy
        Text policy the strings were built under.
    source_handle : str | None
        Handle this record came from, when it came from one.
    indexed_at : datetime.datetime | None
        Timestamp.  Defaults to now, in UTC.

    Returns
    -------
    dict[str, Any]
        Row matching :func:`build_schema`.
    """
    stamp = indexed_at or datetime.datetime.now(datetime.UTC)
    year = paper.publication_date.year if paper.publication_date is not None else 0
    source = getattr(paper, "source", None)
    venue = getattr(source, "title", None) if source is not None else None
    return {
        KEY_COLUMN: paper_key(paper),
        "title": paper.title,
        "doi": paper.doi or "",
        "preprint_doi": paper.preprint_doi or "",
        "year": int(year),
        "venue": str(venue) if venue else "",
        "found_in": ",".join(sorted(paper.found_in)),
        SEARCH_TEXT_COLUMN: text.search_text,
        EXACT_TEXT_COLUMN: text.exact_text,
        "text_policy": policy.describe(),
        "missing_fields": text.degradation_label,
        "model_id": identity.model_id,
        "model_revision": identity.model_revision or "",
        "prompt_convention": identity.prompt_convention,
        "embedding_dim": int(identity.dimension),
        "source_handle": source_handle or "",
        "indexed_at": stamp.isoformat(),
        VECTOR_COLUMN: [float(value) for value in vector],
    }


__all__ = [
    "COLUMNS",
    "EXACT_TEXT_COLUMN",
    "KEY_COLUMN",
    "PROJECTION",
    "SEARCH_TEXT_COLUMN",
    "VECTOR_COLUMN",
    "build_row",
    "build_schema",
    "paper_key",
]
