"""The embedded LanceDB index: creation, ingest, index maintenance, health.

Verified live on 2026-08-31 against the installed ``lancedb`` **0.37.1**.  Four
findings, each of which changes what this module does, and two of which
correct the written record:

**The FTS index must exist before keyword search works, and the call to make
it is not the one the record names.**  ``table.create_fts_index("col")`` is
**deprecated as of lancedb 0.25.0** and emits a ``DeprecatedWarning`` on the
*sync* table -- "Use create_index() with config=FTS() instead".  The current
call is ``table.create_index(column, config=FTS(...))``, where the first
positional argument is the column name whenever ``config`` is supplied (the
parameter is named ``metric`` in the merged signature purely for backwards
compatibility with the legacy vector-index API).  Observed to create an index
named ``<column>_idx`` of type ``FTS``.

**Hybrid search cannot be given a query tuple.**  With no embedding function
registered on the table -- which is our case, because embedding is ours to
control -- ``table.search((vector, "text"), query_type="hybrid")`` raises
``ValueError: Vector query must be either a string or a vector``, and passing
a bare string raises ``ValueError: No embedding function for vector``.  The
working form is ``table.search(query_type="hybrid").vector(v).text(s)``.

**Rows added after the index are genuinely not in it, and the flat-scan
fallback is what keeps results complete.**  Measured: after one ``add`` past a
built index, ``index_stats("search_text_idx").num_unindexed_rows == 1``, and a
normal full-text query still returned the new row.  The same query with
``.fast_search()`` returned only the indexed row and **silently dropped the
new one**.  That is why ``.fast_search()`` is forbidden here and why the ban is
covered by a test rather than a comment.

**0.38.0 exists and is not installable on this machine.**  It was uploaded to
PyPI at 2026-08-31T08:46Z, hours before this module was written, and publishes
only ``macosx_11_0_arm64`` and ``manylinux_2_28_aarch64`` wheels -- no
``x86_64`` Linux wheel, where 0.37.1 has one.  ``uv pip install
lancedb==0.38.0`` fails here with "no wheels with a matching platform tag".
So the build stays on 0.37.1; re-check when the x86_64 wheel appears.

Maintenance policy, and why it is stricter than the documented guidance
----------------------------------------------------------------------

LanceDB's guidance is to optimize after roughly 100,000 row changes or 20
data-modification operations.  Those numbers are for tables far larger than a
literature corpus: at a few thousand papers, 100,000 row changes means
"never", and the unindexed tail grows for the whole life of the index while
every query pays a flat scan over it.  This module instead keeps
``num_unindexed_rows`` at zero by optimizing at the end of any ingest that
leaves rows unindexed, because at this scale the compaction is milliseconds and
the degradation it prevents is invisible.  The 100,000-row and
20-operation thresholds are retained as *upper* bounds for the case where a
caller turns automatic maintenance off.
"""

from __future__ import annotations

import datetime
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lit_search.exceptions import LitSearchError
from lit_search.index.embedding import EmbeddingIdentity, EmbeddingModel
from lit_search.index.schema import (
    EXACT_TEXT_COLUMN,
    KEY_COLUMN,
    PROJECTION,
    SEARCH_TEXT_COLUMN,
    VECTOR_COLUMN,
    build_row,
    build_schema,
)
from lit_search.index.text import DEFAULT_TEXT_POLICY, TextPolicy, build_paper_text
from lit_search.utils.persistence import atomic_write_text

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lit_search.core.paper import Paper

logger = logging.getLogger(__name__)

#: Marker identifying an index manifest written by this module.
MANIFEST_SCHEMA = "lit-search/index"

#: Manifest version.  Bumped for any change to the manifest keys.
MANIFEST_VERSION = 1

#: Manifest filename inside the index root.
MANIFEST_NAME = "index.json"

#: Table name inside the LanceDB database.
TABLE_NAME = "papers"

#: Upper bound on unindexed rows before maintenance is mandatory, from
#: LanceDB's own guidance.  Reached only when automatic maintenance is off.
MAX_UNINDEXED_ROWS = 100_000

#: Upper bound on table versions since the last optimize, from the same
#: guidance ("20 data-modification operations").  ``table.version`` is
#: LanceDB's own transactional counter, so this needs no bookkeeping of ours
#: and cannot drift between processes.
MAX_VERSIONS_SINCE_OPTIMIZE = 20


class PaperIndexError(LitSearchError):
    """Raised when the index cannot be used as declared.

    Covers a missing ``lancedb`` install, an index built under a different
    embedding model than the one now loaded, and a table whose vector width
    does not match its manifest.  Every case is a refusal rather than a
    warning: vectors from two models are still the same shape and still
    cosine-comparable, so continuing produces a confidently ordered list with
    no meaning behind it.
    """


@dataclass(frozen=True, slots=True)
class IndexHealth:
    """What state the index and its full-text index are in.

    Parameters
    ----------
    rows : int
        Rows in the table.
    indexed_rows : int
        Rows covered by the full-text index.
    unindexed_rows : int
        Rows the full-text index does not cover.  Queries still find them --
        LanceDB falls back to a flat scan -- but the scan cost grows with this
        number.
    has_text_index : bool
        Whether the full-text index exists at all.  Without it, keyword
        retrieval is unavailable, which is why it is reported rather than
        assumed.
    versions_since_optimize : int
        Table versions since the last recorded optimize.
    """

    rows: int
    indexed_rows: int
    unindexed_rows: int
    has_text_index: bool
    versions_since_optimize: int

    @property
    def needs_optimize(self) -> bool:
        """Whether maintenance is due under the documented thresholds.

        Returns
        -------
        bool
            ``True`` when the unindexed tail or the version count has crossed
            a bound.
        """
        return self.unindexed_rows >= MAX_UNINDEXED_ROWS or (
            self.versions_since_optimize >= MAX_VERSIONS_SINCE_OPTIMIZE and self.unindexed_rows > 0
        )

    def describe(self) -> str:
        """Return a one-line human-readable summary.

        Returns
        -------
        str
            Summary naming the unindexed tail explicitly, because a growing
            tail is the only way this index degrades and it degrades silently.
        """
        if not self.has_text_index:
            return (
                f"{self.rows} row(s); NO full-text index, so keyword retrieval is "
                "unavailable and hybrid search would run on the vector lane alone"
            )
        state = "maintenance due" if self.needs_optimize else "healthy"
        return (
            f"{self.rows} row(s); full-text index covers {self.indexed_rows}, "
            f"{self.unindexed_rows} served by flat scan; "
            f"{self.versions_since_optimize} version(s) since optimize; {state}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, Any]
            Serialized health record.
        """
        return {
            "rows": self.rows,
            "indexed_rows": self.indexed_rows,
            "unindexed_rows": self.unindexed_rows,
            "has_text_index": self.has_text_index,
            "versions_since_optimize": self.versions_since_optimize,
            "needs_optimize": self.needs_optimize,
        }


@dataclass(frozen=True, slots=True)
class IngestReport:
    """What one ingest actually wrote.

    Parameters
    ----------
    submitted : int
        Papers handed to :meth:`PaperIndex.add`.
    written : int
        Rows written.
    skipped_existing : int
        Papers already present under the same key and left alone.
    degraded_text : tuple[str, ...]
        Keys of records built from less text than the policy asks for, with
        the missing fields.  Surfaced because a silently title-only embedding
        is the ``R16`` defect, one level below the model choice.
    health : IndexHealth
        State of the index after the ingest.
    """

    submitted: int
    written: int
    skipped_existing: int
    degraded_text: tuple[str, ...]
    health: IndexHealth

    def describe(self) -> str:
        """Return a one-line human-readable summary.

        Returns
        -------
        str
            Summary including the degraded-text count.
        """
        degraded = len(self.degraded_text)
        parts = [
            f"{self.written} row(s) written from {self.submitted} paper(s)",
            f"{self.skipped_existing} already present",
        ]
        if degraded:
            parts.append(f"{degraded} embedded on incomplete text")
        return "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, Any]
            Serialized report.
        """
        return {
            "submitted": self.submitted,
            "written": self.written,
            "skipped_existing": self.skipped_existing,
            "degraded_text": list(self.degraded_text),
            "health": self.health.to_dict(),
        }


class PaperIndex:
    """An embedded LanceDB table of papers, with its embedding identity pinned.

    Opening an index whose manifest names a different model than the loaded
    embedder is refused.  That refusal is the point of the class: it is the
    positive form of the design reference's negative lesson, which hardcodes
    one dimension and warns when the model disagrees.  A dimension check alone
    would not be enough anyway -- two 1024-dimension models pass it and mean
    entirely different things.
    """

    def __init__(
        self,
        root: Path | str,
        embedder: EmbeddingModel,
        *,
        policy: TextPolicy = DEFAULT_TEXT_POLICY,
        auto_optimize: bool = True,
    ) -> None:
        """Open or create an index rooted at *root*.

        Parameters
        ----------
        root : Path | str
            Directory holding the LanceDB database and the manifest.
        embedder : EmbeddingModel
            Model used for ingest and for query embedding.  Its identity is
            written into the manifest on creation and checked on every open.
        policy : TextPolicy
            Which paper fields are indexed.
        auto_optimize : bool
            Whether an ingest that leaves unindexed rows compacts them
            immediately.  Leaving this on keeps the flat-scan tail at zero.

        Raises
        ------
        PaperIndexError
            If the manifest names a different embedding identity, or if the
            manifest cannot be read.
        """
        self._root = Path(root).expanduser()
        self._embedder = embedder
        self._policy = policy
        self._auto_optimize = auto_optimize
        self._manifest = self._load_or_create_manifest()

    # ------------------------------------------------------------------
    # Identity and manifest
    # ------------------------------------------------------------------

    @property
    def root(self) -> Path:
        """Return the index root directory.

        Returns
        -------
        Path
            Directory holding the database and the manifest.
        """
        return self._root

    @property
    def identity(self) -> EmbeddingIdentity:
        """Return the embedding identity this index is pinned to.

        Returns
        -------
        EmbeddingIdentity
            Identity from the manifest.
        """
        return EmbeddingIdentity.from_dict(self._manifest["identity"])

    @property
    def text_policy(self) -> TextPolicy:
        """Return the text policy in use.

        Returns
        -------
        TextPolicy
            Policy supplied at construction.
        """
        return self._policy

    @property
    def _manifest_path(self) -> Path:
        """Return the manifest file path.

        Returns
        -------
        Path
            Path to the manifest JSON.
        """
        return self._root / MANIFEST_NAME

    def _load_or_create_manifest(self) -> dict[str, Any]:
        """Read the manifest, or write a new one, refusing a mismatch.

        Returns
        -------
        dict[str, Any]
            Manifest contents.

        Raises
        ------
        PaperIndexError
            If the manifest is unreadable, of an unknown version, or names a
            different embedding identity than the loaded model.
        """
        path = self._manifest_path
        if not path.is_file():
            manifest = {
                "schema": MANIFEST_SCHEMA,
                "schema_version": MANIFEST_VERSION,
                "identity": self._embedder.identity.to_dict(),
                "text_policy": self._policy.describe(),
                "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
                "optimized_at_version": 0,
            }
            self._write_manifest(manifest)
            if not self._embedder.identity.is_semantic:
                logger.warning(
                    "[index] creating %s with a NON-SEMANTIC embedder (%s); its similarity "
                    "scores are reproducible but meaningless, and every ranking built from "
                    "it will say so",
                    self._root,
                    self._embedder.identity.describe(),
                )
            return manifest

        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PaperIndexError(f"Index manifest at {path} could not be read: {error}") from error
        self._check_manifest(loaded, path)
        return dict(loaded)

    def _check_manifest(self, loaded: Any, path: Path) -> None:
        """Refuse a manifest this build cannot honour.

        Parameters
        ----------
        loaded : Any
            Parsed manifest.
        path : Path
            Where it came from, for the error message.

        Returns
        -------
        None

        Raises
        ------
        PaperIndexError
            If the schema, version, or embedding identity does not match.
        """
        if not isinstance(loaded, dict) or loaded.get("schema") != MANIFEST_SCHEMA:
            raise PaperIndexError(
                f"{path} is not a {MANIFEST_SCHEMA} manifest; refusing to treat an "
                "unknown file as an index"
            )
        if loaded.get("schema_version") != MANIFEST_VERSION:
            raise PaperIndexError(
                f"{path} declares manifest version {loaded.get('schema_version')!r} and "
                f"this build writes {MANIFEST_VERSION}; no migration is registered"
            )
        stored = EmbeddingIdentity.from_dict(loaded.get("identity"))
        active = self._embedder.identity
        if stored != active:
            raise PaperIndexError(
                f"Index at {self._root} was built with {stored.describe()} and the loaded "
                f"model is {active.describe()}. Vectors from two models are the same shape "
                "and are not comparable, so a mixed index returns a confidently ordered "
                "list with nothing behind it. Rebuild the index, or load the model it was "
                "built with."
            )

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        """Write the manifest atomically.

        Parameters
        ----------
        manifest : dict[str, Any]
            Manifest to persist.

        Returns
        -------
        None
        """
        self._root.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            self._manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        )

    # ------------------------------------------------------------------
    # Table access
    # ------------------------------------------------------------------

    def _database(self) -> Any:
        """Connect to the LanceDB database under the index root.

        Returns
        -------
        Any
            LanceDB connection.

        Raises
        ------
        PaperIndexError
            If ``lancedb`` is not installed.
        """
        try:
            import lancedb
        except ImportError as error:  # pragma: no cover - lancedb is a hard dependency
            raise PaperIndexError(
                "lancedb is not installed, so the paper index is unavailable. Install it "
                "with 'pip install \"lancedb>=0.37.1,<0.38\"'. Refusing to fall back to an "
                "in-memory scan, which would silently change what a search means."
            ) from error
        self._root.mkdir(parents=True, exist_ok=True)
        return lancedb.connect(self._root / "lance")

    def table(self) -> Any:
        """Return the papers table, creating it empty when it does not exist.

        Returns
        -------
        Any
            LanceDB table.
        """
        database = self._database()
        if TABLE_NAME in _existing_tables(database):
            return database.open_table(TABLE_NAME)
        return database.create_table(TABLE_NAME, schema=build_schema(self._embedder.identity))

    def count(self) -> int:
        """Return how many rows the index holds.

        Returns
        -------
        int
            Row count.
        """
        return int(self.table().count_rows())

    def keys(self) -> set[str]:
        """Return every record key in the index.

        Returns
        -------
        set[str]
            Keys, as produced by :func:`lit_search.index.schema.paper_key`.
        """
        table = self.table()
        if table.count_rows() == 0:
            return set()
        return {
            str(row["paper_key"])
            for row in table.search().select(["paper_key"]).limit(table.count_rows()).to_list()
        }

    # ------------------------------------------------------------------
    # Ingest and maintenance
    # ------------------------------------------------------------------

    def add(self, papers: Sequence[Paper], *, source_handle: str | None = None) -> IngestReport:
        """Embed and write papers not already present.

        Parameters
        ----------
        papers : Sequence[Paper]
            Papers to index.
        source_handle : str | None
            Handle they came from, recorded per row as provenance.

        Returns
        -------
        IngestReport
            What was written, what was skipped, and which records were
            embedded on incomplete text.
        """
        from lit_search.index.schema import paper_key

        existing = self.keys()
        fresh: list[Paper] = []
        seen: set[str] = set()
        for paper in papers:
            key = paper_key(paper)
            if key in existing or key in seen:
                continue
            seen.add(key)
            fresh.append(paper)

        texts = [build_paper_text(paper, self._policy) for paper in fresh]
        degraded = tuple(
            f"{paper_key(paper)} (missing {text.degradation_label})"
            for paper, text in zip(fresh, texts, strict=True)
            if text.is_degraded
        )
        if fresh:
            vectors = self._embedder.encode_documents([text.search_text for text in texts])
            rows = [
                build_row(
                    paper,
                    text,
                    vector,
                    identity=self._embedder.identity,
                    policy=self._policy,
                    source_handle=source_handle,
                )
                for paper, text, vector in zip(fresh, texts, vectors, strict=True)
            ]
            self.table().add(rows)
            self.ensure_text_index()
            if self._auto_optimize:
                self.optimize()
        if degraded:
            logger.warning(
                "[index] %d of %d record(s) were embedded on incomplete text; a "
                "title-only embedding is a weaker measurement than a title-plus-abstract "
                "one and the ranking will not look any different",
                len(degraded),
                len(fresh),
            )
        return IngestReport(
            submitted=len(papers),
            written=len(fresh),
            skipped_existing=len(papers) - len(fresh),
            degraded_text=degraded,
            health=self.health(),
        )

    def ensure_text_index(self) -> bool:
        """Create the full-text index when it is absent.

        Uses ``create_index(column, config=FTS(...))``.  The older
        ``create_fts_index`` is deprecated as of lancedb 0.25.0 and warns on
        the sync table.  ``with_position=True`` is set so that phrase queries
        work: a phrase query is exactly the kind of exact-terminology lookup
        this index exists to support.

        Returns
        -------
        bool
            ``True`` when an index was created by this call, ``False`` when one
            already existed or the table is empty.
        """
        from lancedb.index import FTS

        table = self.table()
        if table.count_rows() == 0:
            return False
        if self._text_index_name(table) is not None:
            return False
        table.create_index(SEARCH_TEXT_COLUMN, config=FTS(with_position=True, language="English"))
        logger.info("[index] created full-text index on %s", SEARCH_TEXT_COLUMN)
        return True

    @staticmethod
    def _text_index_name(table: Any) -> str | None:
        """Return the name of the full-text index, or ``None``.

        Parameters
        ----------
        table : Any
            LanceDB table.

        Returns
        -------
        str | None
            Index name.
        """
        for index in table.list_indices():
            if SEARCH_TEXT_COLUMN in list(index.columns):
                return str(index.name)
        return None

    def optimize(self) -> IndexHealth:
        """Compact the table so the full-text index covers every row.

        Returns
        -------
        IndexHealth
            State after the compaction.
        """
        table = self.table()
        table.optimize()
        self._manifest["optimized_at_version"] = int(table.version)
        self._write_manifest(self._manifest)
        return self.health()

    def health(self) -> IndexHealth:
        """Report the index's state, including the unindexed tail.

        Returns
        -------
        IndexHealth
            Row counts, index presence, and versions since the last optimize.
        """
        table = self.table()
        rows = int(table.count_rows())
        name = self._text_index_name(table)
        indexed, unindexed = 0, rows
        if name is not None:
            stats = table.index_stats(name)
            if stats is not None:
                indexed = int(stats.num_indexed_rows or 0)
                unindexed = int(stats.num_unindexed_rows or 0)
        return IndexHealth(
            rows=rows,
            indexed_rows=indexed,
            unindexed_rows=unindexed,
            has_text_index=name is not None,
            versions_since_optimize=max(
                0, int(table.version) - int(self._manifest.get("optimized_at_version", 0))
            ),
        )

    # ------------------------------------------------------------------
    # Retrieval lanes
    # ------------------------------------------------------------------

    def vector_lane(
        self,
        query_vector: Sequence[float],
        limit: int,
        *,
        keys: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the nearest rows by cosine distance.

        Parameters
        ----------
        query_vector : Sequence[float]
            Query embedding.
        limit : int
            How many rows to return.  Always set explicitly; LanceDB's own
            default is 10, which silently truncates a re-rank pool.
        keys : Sequence[str] | None
            Restrict the lane to these record keys.  Used when re-ranking one
            stored set rather than searching the whole index.

        Returns
        -------
        list[dict[str, Any]]
            Rows with a ``_distance`` column.
        """
        query = (
            self.table()
            .search(list(query_vector), vector_column_name=VECTOR_COLUMN)
            .distance_type("cosine")
        )
        predicate = key_predicate(keys)
        if predicate is not None:
            query = query.where(predicate)
        return list(query.select([*PROJECTION, "_distance"]).limit(limit).to_list())

    def text_lane(
        self, query_text: str, limit: int, *, keys: Sequence[str] | None = None
    ) -> list[dict[str, Any]]:
        """Return rows matching *query_text* by BM25 over the full-text index.

        Note the absence of ``.fast_search()``.  It skips the flat scan over
        rows added since the last compaction, which makes it an API that
        silently omits records -- in a tool whose entire purpose is that a
        result never silently omits anything.

        Parameters
        ----------
        query_text : str
            Keyword query.
        limit : int
            How many rows to return.

        Returns
        -------
        list[dict[str, Any]]
            Rows with a ``_score`` column.
        """
        table = self.table()
        if self._text_index_name(table) is None:
            return []
        query = table.search(query_text, query_type="fts", fts_columns=SEARCH_TEXT_COLUMN)
        predicate = key_predicate(keys)
        if predicate is not None:
            query = query.where(predicate)
        return list(query.select([*PROJECTION, "_score"]).limit(limit).to_list())

    def exact_lane(
        self, needles: Sequence[str], limit: int, *, keys: Sequence[str] | None = None
    ) -> list[dict[str, Any]]:
        """Return rows whose raw text literally contains any of *needles*.

        This is the lane that answers "does the string ``FM-Agent`` occur
        here".  The BM25 lane cannot: it stems, folds case, and drops stop
        words, all of which destroy an exact phrase.  The predicate is a
        ``LIKE`` pushed down to the query engine over a dedicated raw-text
        column -- the shape taken from ``TKasperczyk/claude-memory``, which
        uses ``lower(exact_text) LIKE lower('%needle%') ESCAPE '\\'`` for the
        same reason.

        Parameters
        ----------
        needles : Sequence[str]
            Literal substrings to look for.  Blank entries are dropped.
        limit : int
            How many rows to return.

        Returns
        -------
        list[dict[str, Any]]
            Matching rows, no score column.
        """
        clauses = [_like_clause(needle) for needle in needles if needle.strip()]
        if not clauses:
            return []
        predicate = clauses[0] if len(clauses) == 1 else "(" + " OR ".join(clauses) + ")"
        restriction = key_predicate(keys)
        if restriction is not None:
            predicate = f"{predicate} AND {restriction}"
        return list(
            self.table().search().where(predicate).select(list(PROJECTION)).limit(limit).to_list()
        )

    def vectors_for(self, keys: Sequence[str]) -> dict[str, list[float]]:
        """Return the stored vectors for *keys*.

        Needed by the diversifier, which measures candidate-to-candidate
        similarity and therefore needs the vectors themselves rather than
        distances to the query.

        Parameters
        ----------
        keys : Sequence[str]
            Record keys.

        Returns
        -------
        dict[str, list[float]]
            Key to vector, omitting keys not in the index.
        """
        wanted = [key for key in keys if key.strip()]
        if not wanted:
            return {}
        clauses = " OR ".join(f"paper_key = {_quote(key)}" for key in wanted)
        rows = (
            self.table()
            .search()
            .where(clauses)
            .select(["paper_key", VECTOR_COLUMN])
            .limit(len(wanted))
            .to_list()
        )
        return {
            str(row["paper_key"]): [float(value) for value in row[VECTOR_COLUMN]] for row in rows
        }


def _existing_tables(database: Any) -> set[str]:
    """List the table names in a LanceDB database, following pagination.

    ``list_tables`` returns a ``ListTablesResponse`` carrying a ``tables`` list
    and an optional ``page_token``, not a plain list.  The paging matters even
    for a one-table database: reading only the first page is the kind of
    almost-correct that works until it does not.  The older ``table_names()``
    does return a plain list and is deprecated as of lancedb 0.37.1.

    Parameters
    ----------
    database : Any
        LanceDB connection.

    Returns
    -------
    set[str]
        Every table name in the database.
    """
    names: set[str] = set()
    token: str | None = None
    while True:
        response = database.list_tables(page_token=token) if token else database.list_tables()
        names.update(str(name) for name in getattr(response, "tables", response))
        token = getattr(response, "page_token", None)
        if not token:
            return names


def key_predicate(keys: Sequence[str] | None) -> str | None:
    """Build a SQL predicate restricting a lane to a set of record keys.

    Parameters
    ----------
    keys : Sequence[str] | None
        Record keys, or ``None`` for no restriction.

    Returns
    -------
    str | None
        Predicate, or ``None`` when there is nothing to restrict to.  An empty
        sequence yields a predicate matching nothing, which is deliberately
        different from ``None``: "restrict to no keys" and "do not restrict"
        are opposite instructions and must not collapse into each other.
    """
    if keys is None:
        return None
    wanted = sorted({key for key in keys if key.strip()})
    if not wanted:
        return "1 = 0"
    joined = ", ".join(_quote(key) for key in wanted)
    return f"{KEY_COLUMN} IN ({joined})"


def _quote(value: str) -> str:
    """Return *value* as a single-quoted SQL literal.

    Parameters
    ----------
    value : str
        Raw value.

    Returns
    -------
    str
        Quoted literal with embedded quotes doubled.
    """
    return "'" + value.replace("'", "''") + "'"


def _like_clause(needle: str) -> str:
    """Build a case-insensitive ``LIKE`` predicate for one literal substring.

    Both ``LIKE`` wildcards and the escape character itself are escaped, so a
    query containing ``%`` or ``_`` matches those characters rather than acting
    as a wildcard.

    Parameters
    ----------
    needle : str
        Literal substring.

    Returns
    -------
    str
        SQL predicate over the exact-text column.
    """
    escaped = needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    pattern = _quote(f"%{escaped.lower()}%")
    return f"lower({EXACT_TEXT_COLUMN}) LIKE {pattern} ESCAPE '\\'"


__all__ = [
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA",
    "MANIFEST_VERSION",
    "MAX_UNINDEXED_ROWS",
    "MAX_VERSIONS_SINCE_OPTIMIZE",
    "TABLE_NAME",
    "IndexHealth",
    "IngestReport",
    "PaperIndex",
    "PaperIndexError",
    "key_predicate",
]
