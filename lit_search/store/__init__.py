"""Persistent result handles and the disposable operational stores beside them.

Sets are the primary abstraction: a search produces one, a re-rank orders one,
a snowball expands one, a difference subtracts one, an export emits one.  A
handle names a set.

Four properties hold everything together:

**On disk, shared between processes.**  A handle produced at a terminal must be
usable from an MCP call and the reverse, so the store is a known cache
directory and never a per-process registry.

**Immutable.**  Every operation writes a new handle.  An unattended run and a
person at a terminal never contend for the same file, which removes write
conflicts without locking.

**Lineage is a directed acyclic graph.**  Union and intersection take two or
more operands, difference takes a base and one or more subtrahends whose
positions matter, and a mutable external source such as a Zotero library is
pinned into its own handle so that the dependency is an operand rather than an
assumption.

**Nothing loads that cannot be fully understood.**  Payloads carry a schema
version, and loading refuses on an unknown version, a missing key, a mistyped
value, or an unrecognised key.  Identifiers are content addresses, so a
truncated or hand-edited file is detected rather than trusted.

The shared :class:`SnapshotCache` and per-run :class:`RunStateStore` live under
the same root but are not handles.  They are mutable operational truth used to
avoid repeated provider work and resume interrupted traversal; only a finalized
handle is an immutable result artifact.
"""

from lit_search.store.lineage import (
    ExternalReference,
    LineageEdge,
    LineageGraph,
    Operand,
    OperandKind,
    Operation,
    OperationName,
)
from lit_search.store.operations import (
    deduplicated,
    difference,
    filtered,
    from_search_result,
    from_snowball_result,
    imported,
    intersection,
    reranked,
    searched,
    union,
)
from lit_search.store.order import (
    OrderRecord,
    OrderState,
    RankingProvenance,
    RankMode,
)
from lit_search.store.propagation import (
    CoverageOperand,
    CoveragePropagator,
    LedgerCoveragePropagator,
    RefusingCoveragePropagator,
)
from lit_search.store.query_family_store import QueryFamilyStore
from lit_search.store.record import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    HandleRecord,
    compute_handle_id,
)
from lit_search.store.run_state import (
    CredentialIdentity,
    InvalidRunTransitionError,
    RunEvent,
    RunLeaseError,
    RunLifecycle,
    RunPhase,
    RunSchemaError,
    RunSnapshot,
    RunStateError,
    RunStateStore,
    SemanticSpecMismatchError,
    TargetState,
    WorkUnit,
    WorkUnitState,
)
from lit_search.store.snapshot_cache import (
    DEFAULT_BUDGET_BYTES,
    MAX_BUDGET_BYTES,
    AliasResolution,
    CacheUsage,
    CacheWriteResult,
    NegativeRead,
    PhysicalUsage,
    PruneResult,
    SnapshotCache,
    SnapshotCacheError,
    SnapshotCacheSchemaError,
    SnapshotCacheValueError,
    SnapshotQuality,
    SnapshotRead,
)
from lit_search.store.store import (
    APP_DIRECTORY_NAME,
    STORE_ROOT_ENV_VAR,
    HandleStore,
    default_store_root,
)

__all__ = [
    "APP_DIRECTORY_NAME",
    "DEFAULT_BUDGET_BYTES",
    "MAX_BUDGET_BYTES",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "STORE_ROOT_ENV_VAR",
    "AliasResolution",
    "CacheUsage",
    "CacheWriteResult",
    "CoverageOperand",
    "CoveragePropagator",
    "CredentialIdentity",
    "ExternalReference",
    "HandleRecord",
    "HandleStore",
    "InvalidRunTransitionError",
    "LedgerCoveragePropagator",
    "LineageEdge",
    "LineageGraph",
    "NegativeRead",
    "Operand",
    "OperandKind",
    "Operation",
    "OperationName",
    "OrderRecord",
    "OrderState",
    "PhysicalUsage",
    "PruneResult",
    "QueryFamilyStore",
    "RankMode",
    "RankingProvenance",
    "RefusingCoveragePropagator",
    "RunEvent",
    "RunLeaseError",
    "RunLifecycle",
    "RunPhase",
    "RunSchemaError",
    "RunSnapshot",
    "RunStateError",
    "RunStateStore",
    "SemanticSpecMismatchError",
    "SnapshotCache",
    "SnapshotCacheError",
    "SnapshotCacheSchemaError",
    "SnapshotCacheValueError",
    "SnapshotQuality",
    "SnapshotRead",
    "TargetState",
    "WorkUnit",
    "WorkUnitState",
    "compute_handle_id",
    "deduplicated",
    "default_store_root",
    "difference",
    "filtered",
    "from_search_result",
    "from_snowball_result",
    "imported",
    "intersection",
    "reranked",
    "searched",
    "union",
]
