"""Immutable local storage for query-family revisions.

Query families are research artifacts rather than disposable run state.  Each
revision therefore occupies one file named by its semantic content address,
under the same root as the handle store.  Parent revisions must already be
present, making every accepted write a closed lineage DAG at that instant.

The store deliberately has no mutable names or "latest" pointer.  A caller may
resolve an unambiguous digest prefix for convenience, but the result is always
the immutable revision identifier itself.
"""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lit_search.core.query_family import QueryFamilyRevision, QueryFamilySchemaError
from lit_search.store.store import HandleStore, default_store_root
from lit_search.utils.persistence import atomic_write_text

QUERY_FAMILIES_SUBDIRECTORY = "query-families"
QUERY_FAMILY_ID_PREFIX = "qf_"
MINIMUM_QUERY_FAMILY_PREFIX_LENGTH = 4

_QUERY_FAMILY_ID_PATTERN = re.compile(r"qf_[0-9a-f]{32}\Z")


class QueryFamilyStoreError(RuntimeError):
    """Base error for immutable query-family persistence."""


class QueryFamilyRevisionNotFoundError(QueryFamilyStoreError, LookupError):
    """A requested revision or revision prefix is absent."""


class AmbiguousQueryFamilyRevisionError(QueryFamilyStoreError, LookupError):
    """A revision prefix identifies more than one stored revision."""


class QueryFamilyRevisionIntegrityError(QueryFamilyStoreError):
    """Stored bytes disagree with their claimed immutable identity or schema."""


class QueryFamilyLineageError(QueryFamilyStoreError):
    """A query-family ancestry is missing a node or contains a cycle."""


@dataclass(frozen=True, slots=True)
class QueryFamilyLineage:
    """A deterministic, fully verified query-family ancestry DAG.

    ``nodes`` is breadth-first from ``root``.  ``edges`` contains
    ``(child_revision_id, parent_revision_id)`` pairs in the same deterministic
    traversal order.
    """

    root: str
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]


class QueryFamilyStore:
    """Read and write immutable query-family revisions beside result handles."""

    def __init__(self, store_root: Path | str | HandleStore | None = None) -> None:
        """Open the revision store without creating it.

        ``store_root`` may be an existing :class:`HandleStore`, its root path,
        or ``None`` for the normal application cache root.
        """

        if isinstance(store_root, HandleStore):
            self._root = store_root.root
        elif store_root is None:
            self._root = default_store_root()
        else:
            self._root = Path(store_root).expanduser()

    @property
    def root(self) -> Path:
        """Return the shared application-store root."""

        return self._root

    @property
    def revisions_directory(self) -> Path:
        """Return the directory containing immutable revision files."""

        return self._root / QUERY_FAMILIES_SUBDIRECTORY

    def path_for(self, revision_id: str) -> Path:
        """Return the path for one full, validated revision identifier."""

        _validate_revision_id(revision_id)
        return self.revisions_directory / f"{revision_id}.json"

    def exists(self, revision_id: str) -> bool:
        """Report whether a full, well-formed revision identifier is present."""

        try:
            return self.path_for(revision_id).is_file()
        except QueryFamilyRevisionIntegrityError:
            return False

    def put(self, revision: QueryFamilyRevision) -> str:
        """Persist a revision after verifying its identity and closed lineage.

        Repeating an identical write is a no-op.  An existing path containing
        anything else is never replaced.
        """

        if not isinstance(revision, QueryFamilyRevision):
            raise TypeError(f"revision must be QueryFamilyRevision, got {type(revision).__name__}")

        payload = revision.to_dict()
        try:
            verified = QueryFamilyRevision.from_dict(payload)
        except (QueryFamilySchemaError, TypeError, ValueError) as error:
            raise QueryFamilyRevisionIntegrityError(
                f"Revision cannot survive strict serialization: {error}"
            ) from error
        revision_id = verified.revision_id

        for parent_id in sorted(set(verified.parent_revision_ids)):
            if parent_id == revision_id:
                raise QueryFamilyLineageError(
                    f"Revision {revision_id} cannot name itself as a parent"
                )
            if not self.exists(parent_id):
                raise QueryFamilyLineageError(
                    f"Revision {revision_id} refers to missing parent {parent_id}"
                )
            # Refuse to extend a graph whose existing ancestry has been
            # corrupted or made incomplete outside this API.
            self.lineage(parent_id)

        target = self.path_for(revision_id)
        if target.is_file():
            existing = self.get(revision_id)
            if existing.to_dict() == verified.to_dict():
                return revision_id
            raise QueryFamilyRevisionIntegrityError(
                f"Revision {revision_id} already exists with different content at {target}; "
                "refusing to overwrite an immutable revision"
            )

        text = json.dumps(
            verified.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        atomic_write_text(target, text)
        return revision_id

    def get(self, revision_id: str) -> QueryFamilyRevision:
        """Load and strictly verify one full revision identifier."""

        path = self.path_for(revision_id)
        if not path.is_file():
            raise QueryFamilyRevisionNotFoundError(
                f"No query-family revision {revision_id} in {self.revisions_directory}"
            )
        try:
            payload: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise QueryFamilyRevisionIntegrityError(
                f"Query-family revision {revision_id} is not readable valid JSON: {error}"
            ) from error
        try:
            revision = QueryFamilyRevision.from_dict(payload)
        except (QueryFamilySchemaError, TypeError, ValueError) as error:
            raise QueryFamilyRevisionIntegrityError(
                f"Query-family revision {revision_id} fails strict verification: {error}"
            ) from error
        if revision.revision_id != revision_id:
            raise QueryFamilyRevisionIntegrityError(
                f"Query-family file {revision_id} contains revision {revision.revision_id}"
            )
        return revision

    def list_ids(self) -> list[str]:
        """List all well-formed revision identifiers in lexical order."""

        directory = self.revisions_directory
        if not directory.is_dir():
            return []
        return sorted(
            path.stem
            for path in directory.glob(f"{QUERY_FAMILY_ID_PREFIX}*.json")
            if path.is_file() and _is_revision_id(path.stem)
        )

    def resolve(self, prefix: str) -> str:
        """Resolve one full identifier or unambiguous digest prefix."""

        candidate = prefix.strip().lower()
        if not candidate.startswith(QUERY_FAMILY_ID_PREFIX):
            candidate = f"{QUERY_FAMILY_ID_PREFIX}{candidate}"
        if self.exists(candidate):
            return candidate

        digits = candidate[len(QUERY_FAMILY_ID_PREFIX) :]
        if len(digits) < MINIMUM_QUERY_FAMILY_PREFIX_LENGTH:
            raise QueryFamilyRevisionNotFoundError(
                f"Query-family prefix {prefix!r} is shorter than "
                f"{MINIMUM_QUERY_FAMILY_PREFIX_LENGTH} digest characters"
            )
        if not re.fullmatch(r"[0-9a-f]+", digits):
            raise QueryFamilyRevisionNotFoundError(
                f"Query-family prefix {prefix!r} is not hexadecimal"
            )

        matches = [known for known in self.list_ids() if known.startswith(candidate)]
        if not matches:
            raise QueryFamilyRevisionNotFoundError(
                f"No query-family revision matching {prefix!r} in {self.revisions_directory}"
            )
        if len(matches) > 1:
            raise AmbiguousQueryFamilyRevisionError(
                f"Query-family prefix {prefix!r} matches {len(matches)} revisions: "
                f"{', '.join(matches)}"
            )
        return matches[0]

    def lineage(self, revision_id: str) -> QueryFamilyLineage:
        """Return the complete deterministic ancestor DAG or refuse it.

        Unlike a diagnostic best-effort graph, this method never launders a
        missing parent or cycle into a partial lineage result.
        """

        root = self.resolve(revision_id)
        loaded: dict[str, QueryFamilyRevision] = {}
        active: list[str] = []
        complete: set[str] = set()

        def verify(current_id: str) -> None:
            """Depth-first verify and cache the ancestry rooted at ``current_id``."""

            if current_id in active:
                start = active.index(current_id)
                cycle = (*active[start:], current_id)
                raise QueryFamilyLineageError(
                    "Query-family lineage contains a cycle: " + " -> ".join(cycle)
                )
            if current_id in complete:
                return
            if not self.exists(current_id):
                child = active[-1] if active else root
                raise QueryFamilyLineageError(
                    f"Revision {child} refers to missing parent {current_id}"
                )
            current = self.get(current_id)
            loaded[current_id] = current
            active.append(current_id)
            for parent_id in sorted(set(current.parent_revision_ids)):
                verify(parent_id)
            active.pop()
            complete.add(current_id)

        verify(root)

        nodes: list[str] = []
        edges: list[tuple[str, str]] = []
        seen: set[str] = set()
        queue: deque[str] = deque([root])
        while queue:
            current_id = queue.popleft()
            if current_id in seen:
                continue
            seen.add(current_id)
            nodes.append(current_id)
            parents = sorted(set(loaded[current_id].parent_revision_ids))
            edges.extend((current_id, parent_id) for parent_id in parents)
            queue.extend(parent_id for parent_id in parents if parent_id not in seen)

        return QueryFamilyLineage(root=root, nodes=tuple(nodes), edges=tuple(edges))

    def ancestors(self, revision_id: str) -> list[str]:
        """Return verified ancestors breadth-first, nearest first."""

        return list(self.lineage(revision_id).nodes[1:])


def _is_revision_id(value: str) -> bool:
    """Report whether ``value`` is a full lower-case revision identifier."""

    return _QUERY_FAMILY_ID_PATTERN.fullmatch(value) is not None


def _validate_revision_id(value: str) -> None:
    """Refuse an identifier that is unsafe or cannot name a revision file."""

    if not isinstance(value, str) or not _is_revision_id(value):
        raise QueryFamilyRevisionIntegrityError(
            f"{value!r} is not a query-family revision identifier"
        )


__all__ = [
    "MINIMUM_QUERY_FAMILY_PREFIX_LENGTH",
    "QUERY_FAMILIES_SUBDIRECTORY",
    "AmbiguousQueryFamilyRevisionError",
    "QueryFamilyLineage",
    "QueryFamilyLineageError",
    "QueryFamilyRevisionIntegrityError",
    "QueryFamilyRevisionNotFoundError",
    "QueryFamilyStore",
    "QueryFamilyStoreError",
]
