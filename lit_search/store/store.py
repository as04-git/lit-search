"""The on-disk handle store.

Handles are shared between a command line and an MCP server, which are
different processes that may be running at the same time.  That rules out any
in-memory or per-process registry: a handle returned by one must be usable by
the other with no coordination beyond the filesystem.  The store is therefore a
known cache directory holding one JSON file per handle.

Immutability does the concurrency work.  Every operation writes a *new* handle
rather than editing an existing one, so an unattended overnight run and a
person at a terminal never contend for the same file and no locking is needed.
The only write that can target an existing path is a re-write of byte-identical
content, which is treated as the no-op it is.

Every write goes through an atomic temp-file-and-rename, so a crash leaves
either the previous file or the new one, never a half-written JSON document
that the next read cannot parse.
"""

from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from typing import Any

from lit_search.exceptions import (
    AmbiguousHandleError,
    HandleNotFoundError,
    HandleSchemaError,
)
from lit_search.store.lineage import (
    ExternalReference,
    LineageEdge,
    LineageGraph,
    OperandKind,
)
from lit_search.store.record import (
    HANDLE_ID_PREFIX,
    HandleRecord,
    canonical_json,
    verify_payload_id,
)
from lit_search.utils.persistence import atomic_write_text

#: Directory name used under the cache root.
#:
#: PROVISIONAL.  The tool's final name is an open decision, and this constant
#: is the single place that has to change when it is settled.  It is defined
#: here rather than inlined so that a rename cannot leave two processes reading
#: different directories.
APP_DIRECTORY_NAME = "lit-search"

#: Environment variable that overrides the cache root entirely.
STORE_ROOT_ENV_VAR = "LIT_SEARCH_HOME"

#: Subdirectory holding one JSON file per handle.
HANDLES_SUBDIRECTORY = "handles"

#: Minimum length of an abbreviated identifier accepted by :meth:`HandleStore.resolve`.
MINIMUM_PREFIX_LENGTH = 4


def default_store_root() -> Path:
    """Resolve the cache root every process must agree on.

    Resolution order, first match wins:

    1. ``$LIT_SEARCH_HOME``
    2. ``$XDG_CACHE_HOME/<app>``
    3. ``~/.cache/<app>``

    Parameters
    ----------
    None

    Returns
    -------
    Path
        Cache root directory.  Not created by this call.
    """
    override = os.environ.get(STORE_ROOT_ENV_VAR)
    if override:
        return Path(override).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return base / APP_DIRECTORY_NAME


class HandleStore:
    """Reads and writes immutable handles in a shared cache directory."""

    def __init__(self, root: Path | str | None = None) -> None:
        """Open a store rooted at *root*.

        Parameters
        ----------
        root : Path | str | None
            Cache root.  Defaults to :func:`default_store_root`.  The
            directory is created lazily on first write.
        """
        self._root = Path(root).expanduser() if root is not None else default_store_root()

    @property
    def root(self) -> Path:
        """Return the cache root this store reads and writes.

        Returns
        -------
        Path
            Cache root directory.
        """
        return self._root

    @property
    def handles_directory(self) -> Path:
        """Return the directory holding handle files.

        Returns
        -------
        Path
            Directory containing one JSON file per handle.
        """
        return self._root / HANDLES_SUBDIRECTORY

    def path_for(self, handle_id: str) -> Path:
        """Return the file path a handle occupies.

        Parameters
        ----------
        handle_id : str
            Full handle identifier.

        Returns
        -------
        Path
            Path to the handle's JSON file.

        Raises
        ------
        HandleSchemaError
            If *handle_id* is not a well-formed identifier.  Validated because
            an identifier is used as a filename, and an unvalidated one could
            escape the store directory.
        """
        _validate_handle_id(handle_id)
        return self.handles_directory / f"{handle_id}.json"

    def exists(self, handle_id: str) -> bool:
        """Report whether a handle is present in the store.

        Parameters
        ----------
        handle_id : str
            Full handle identifier.

        Returns
        -------
        bool
            ``True`` when the handle's file exists.
        """
        try:
            return self.path_for(handle_id).is_file()
        except HandleSchemaError:
            return False

    def put(self, record: HandleRecord) -> str:
        """Write a handle and return its identifier.

        Three guarantees, in the order they are checked:

        1. **No dangling lineage.**  Every handle operand must already exist in
           this store.  A lineage edge that points at nothing is caught when it
           is created rather than during an audit months later.
        2. **Idempotent retry.**  Re-writing byte-identical content is a no-op,
           so a crashed and restarted operation does not duplicate anything.
        3. **No silent overwrite.**  Since the path is a content address, an
           existing file with *different* content means either corruption or a
           digest collision, and both are reported rather than overwritten.

        Parameters
        ----------
        record : HandleRecord
            Handle to write.

        Returns
        -------
        str
            The handle's identifier.

        Raises
        ------
        HandleNotFoundError
            If an operand handle is not in this store.
        HandleIntegrityError
            If a different handle already occupies the content address.
        """
        from lit_search.exceptions import HandleIntegrityError

        for operand in record.operation.operands:
            if (
                operand.kind is OperandKind.HANDLE
                and operand.handle_id is not None
                and not self.exists(operand.handle_id)
            ):
                raise HandleNotFoundError(
                    f"Operation {record.operation.name.value!r} refers to operand "
                    f"{operand.handle_id} (role {operand.role!r}), which is not in "
                    f"the store at {self.handles_directory}"
                )

        payload = record.to_dict()
        handle_id = str(payload["handle_id"])
        target = self.path_for(handle_id)
        text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)

        if target.is_file():
            existing = target.read_text(encoding="utf-8")
            if existing == text:
                return handle_id
            raise HandleIntegrityError(
                f"Handle {handle_id} already exists with different content at {target}; "
                "refusing to overwrite an immutable handle"
            )

        atomic_write_text(target, text)
        return handle_id

    def read_payload(self, handle_id: str) -> dict[str, Any]:
        """Read a handle's raw payload and verify its content address.

        Verification is done against the bytes on disk rather than against
        re-serialized objects, so it does not depend on any round-trip being
        byte-exact.

        Parameters
        ----------
        handle_id : str
            Full handle identifier.

        Returns
        -------
        dict[str, Any]
            Parsed envelope.

        Raises
        ------
        HandleNotFoundError
            If the handle is not in the store.
        HandleSchemaError
            If the file is not parsable JSON or not an object.
        HandleIntegrityError
            If the content does not hash to the identifier it claims.
        """
        path = self.path_for(handle_id)
        if not path.is_file():
            raise HandleNotFoundError(f"No handle {handle_id} in {self.handles_directory}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HandleSchemaError(f"Handle {handle_id} is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise HandleSchemaError(
                f"Handle {handle_id} is a {type(payload).__name__}, expected an object"
            )
        verify_payload_id(payload)
        return payload

    def get(self, handle_id: str) -> HandleRecord:
        """Load a handle.

        Parameters
        ----------
        handle_id : str
            Full handle identifier.  Use :meth:`resolve` first to expand an
            abbreviation.

        Returns
        -------
        HandleRecord
            The stored handle.

        Raises
        ------
        HandleNotFoundError
            If the handle is not in the store.
        HandleSchemaError
            If the payload is not exactly what this schema version defines.
        HandleIntegrityError
            If the content does not hash to its identifier.
        """
        return HandleRecord.from_dict(self.read_payload(handle_id))

    def list_ids(self) -> list[str]:
        """List every handle identifier in the store, sorted.

        Returns
        -------
        list[str]
            Identifiers, or an empty list when the store has never been
            written to.
        """
        directory = self.handles_directory
        if not directory.is_dir():
            return []
        return sorted(
            path.stem
            for path in directory.glob(f"{HANDLE_ID_PREFIX}*.json")
            if path.is_file() and _is_handle_id(path.stem)
        )

    def resolve(self, prefix: str) -> str:
        """Expand an abbreviated identifier to a full one.

        Content addresses are long, so a person at a terminal will type a
        prefix.  An ambiguous prefix is an error rather than a first match,
        because silently picking one of two candidate sets is precisely the
        kind of plausible wrong answer this store exists to prevent.

        Parameters
        ----------
        prefix : str
            Full identifier, or a prefix of at least
            :data:`MINIMUM_PREFIX_LENGTH` characters after the ``h_`` marker.

        Returns
        -------
        str
            The full identifier.

        Raises
        ------
        HandleNotFoundError
            If nothing matches, or the prefix is too short to be safe.
        AmbiguousHandleError
            If more than one handle matches.
        """
        candidate = prefix.strip()
        if not candidate.startswith(HANDLE_ID_PREFIX):
            candidate = f"{HANDLE_ID_PREFIX}{candidate}"
        if self.exists(candidate):
            return candidate

        digits = candidate[len(HANDLE_ID_PREFIX) :]
        if len(digits) < MINIMUM_PREFIX_LENGTH:
            raise HandleNotFoundError(
                f"Handle prefix {prefix!r} is shorter than {MINIMUM_PREFIX_LENGTH} "
                "characters, which is too short to identify a handle safely"
            )

        matches = [known for known in self.list_ids() if known.startswith(candidate)]
        if not matches:
            raise HandleNotFoundError(f"No handle matching {prefix!r} in {self.handles_directory}")
        if len(matches) > 1:
            raise AmbiguousHandleError(
                f"Handle prefix {prefix!r} matches {len(matches)} handles: "
                f"{', '.join(sorted(matches))}"
            )
        return matches[0]

    def lineage(self, handle_id: str) -> LineageGraph:
        """Walk a handle's full ancestry as a directed acyclic graph.

        Traversal is breadth-first from *handle_id* and is guarded against
        revisiting a node.  A referenced handle that is no longer present is
        reported in :attr:`~lit_search.store.lineage.LineageGraph.missing`
        rather than skipped, because an ancestry with a hole in it is a
        different fact from a complete one.

        Parameters
        ----------
        handle_id : str
            Full identifier of the handle to explain.

        Returns
        -------
        LineageGraph
            Nodes, edges, external snapshots, and any missing ancestors.

        Raises
        ------
        HandleNotFoundError
            If *handle_id* itself is not in the store.
        """
        root = handle_id
        record = self.get(root)

        nodes: list[str] = [root]
        seen: set[str] = {root}
        edges: list[LineageEdge] = []
        external: list[tuple[str, ExternalReference]] = []
        missing: list[str] = []
        queue: deque[tuple[str, HandleRecord]] = deque([(root, record)])

        while queue:
            current_id, current = queue.popleft()
            for operand in current.operation.operands:
                if operand.kind is OperandKind.EXTERNAL and operand.external is not None:
                    external.append((current_id, operand.external))
                    continue
                parent_id = operand.handle_id
                if parent_id is None:
                    continue
                edges.append(
                    LineageEdge(
                        child=current_id,
                        parent=parent_id,
                        role=operand.role,
                        operation=current.operation.name,
                    )
                )
                if parent_id in seen:
                    continue
                seen.add(parent_id)
                if not self.exists(parent_id):
                    missing.append(parent_id)
                    continue
                nodes.append(parent_id)
                queue.append((parent_id, self.get(parent_id)))

        return LineageGraph(
            root=root,
            nodes=tuple(nodes),
            edges=tuple(edges),
            external=tuple(external),
            missing=tuple(missing),
        )

    def ancestors(self, handle_id: str) -> list[str]:
        """List every handle a handle was derived from.

        Parameters
        ----------
        handle_id : str
            Full identifier.

        Returns
        -------
        list[str]
            Ancestor identifiers in breadth-first order, excluding
            *handle_id* itself and excluding ancestors missing from the store.
        """
        return list(self.lineage(handle_id).nodes[1:])

    def canonical_text(self, handle_id: str) -> str:
        """Return a handle's canonical JSON, for diffing two handles by eye.

        Parameters
        ----------
        handle_id : str
            Full identifier.

        Returns
        -------
        str
            Canonical JSON text of the stored payload.
        """
        return canonical_json(self.read_payload(handle_id))


def _is_handle_id(value: str) -> bool:
    """Report whether a string is a well-formed handle identifier.

    Parameters
    ----------
    value : str
        Candidate identifier.

    Returns
    -------
    bool
        ``True`` when the value is ``h_`` followed by lowercase hex digits of
        the expected length.
    """
    if not value.startswith(HANDLE_ID_PREFIX):
        return False
    digits = value[len(HANDLE_ID_PREFIX) :]
    return len(digits) == 32 and all(character in "0123456789abcdef" for character in digits)


def _validate_handle_id(value: str) -> None:
    """Refuse a malformed identifier before it is used as a filename.

    Parameters
    ----------
    value : str
        Candidate identifier.

    Returns
    -------
    None

    Raises
    ------
    HandleSchemaError
        If the value is not a well-formed handle identifier.
    """
    if not _is_handle_id(value):
        raise HandleSchemaError(
            f"{value!r} is not a handle identifier; expected {HANDLE_ID_PREFIX!r} "
            "followed by 32 lowercase hex characters"
        )
