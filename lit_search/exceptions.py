"""Exception hierarchy for the lit-search library.

All library-specific exceptions inherit from :class:`LitSearchError`,
allowing callers to catch ``LitSearchError`` as a single entry point for
any error raised by this package.  Each leaf exception also inherits from
the corresponding built-in type (``ValueError``, ``TypeError``) so that
existing ``except ValueError`` handlers continue to work.
"""


class LitSearchError(Exception):
    """Base exception for all lit-search-specific errors."""


class UnsupportedQueryError(LitSearchError, ValueError):
    """Raised when a query uses features not supported by a specific database."""


class ConnectorError(LitSearchError):
    """Raised when an external API connector encounters an unrecoverable error."""


class PersistenceError(LitSearchError):
    """Raised when persistence encounters an unsupported data type or format."""


class QueryValidationError(LitSearchError, ValueError):
    """Raised when a query string is invalid."""


class ModelValidationError(LitSearchError, ValueError):
    """Raised when a model object (Paper, Author, Source) has invalid data."""


class IdentityConflictError(LitSearchError, ValueError):
    """Raised when an operation would combine conflicting explicit identities."""


class InvalidParameterError(LitSearchError, ValueError):
    """Raised when a function or runner receives an invalid argument."""


class MissingApiKeyError(InvalidParameterError):
    """Raised when a connector that requires an API key is created without one."""


class CoverageContractError(LitSearchError, TypeError):
    """Raised when a search connector violates the coverage contract.

    Inherits from ``TypeError`` because every violation is a type-level
    mistake: a connector that returns a bare sequence instead of a
    :class:`~lit_search.core.coverage.CoveredPapers`, that annotates
    ``_fetch_papers`` with the wrong return type, or that overrides the final
    :meth:`~lit_search.connectors.search_base.SearchConnectorBase.search`
    entry point.  It is raised at class-definition time where possible, so
    the failure lands on the author of the connector rather than on a user
    reading an incomplete result weeks later.
    """


class RateLimitError(LitSearchError):
    """Base class for failures of the cross-process rate limiter.

    Deliberately **not** a :class:`ConnectorError`.  Connector code treats
    ``ConnectorError`` as a recoverable per-database failure and has
    historically converted it into an empty result list, which is
    indistinguishable from a database that genuinely found nothing.  A run whose
    requests cannot be paced must fail visibly instead: an unpaced run earns
    HTTP 429s, and 429s become apparent zero recall.
    """


class RateLimitLockTimeoutError(RateLimitError):
    """Raised when the shared rate-limit lock could not be acquired in time.

    The limiter's critical section is sub-millisecond, so exhausting the
    acquisition budget indicates a wedged holder or a broken filesystem rather
    than ordinary contention.  A holder that *dies* cannot cause this: the
    kernel releases its ``flock`` when its file descriptor closes.
    """


class RateLimitStoreUnavailableError(RateLimitError):
    """Raised when the shared rate-limit state cannot be created or opened.

    Raised rather than falling back to unsynchronised per-process limiting,
    because an un-shared clock is precisely the defect the limiter exists to
    remove and its symptom is silent.
    """


class RateLimitBusyError(RateLimitError):
    """Raised when the next free slot lies past the caller's deadline.

    Nothing is reserved, so an abandoned call never pushes the shared queue
    back for other processes.  The caller reports the source as busy, with
    :attr:`retry_after`, instead of waiting out a deadline it cannot meet.
    """

    def __init__(self, scope_key: str, retry_after: float, deadline_in: float) -> None:
        self.scope_key = scope_key
        self.retry_after = retry_after
        self.deadline_in = deadline_in
        super().__init__(
            f"rate-limited: the shared '{scope_key.split(':')[0]}' quota is busy; next free slot "
            f"in {retry_after:.0f}s, past this call's deadline ({max(deadline_in, 0):.0f}s left); "
            "nothing was sent"
        )


class HandleStoreError(LitSearchError):
    """Base exception for every failure raised by the handle store."""


class HandleNotFoundError(HandleStoreError):
    """Raised when a handle identifier does not exist in the store."""


class AmbiguousHandleError(HandleStoreError, ValueError):
    """Raised when an abbreviated handle prefix matches more than one handle."""


class HandleSchemaError(HandleStoreError, ValueError):
    """Raised when a serialized handle cannot be trusted to mean what it says.

    Covers an unknown or unsupported ``schema_version``, a missing required
    key, a key of the wrong type, and unrecognised extra keys.  The store
    **refuses** in all of those cases rather than defaulting the field, because
    a silently defaulted field is indistinguishable from a correctly recorded
    one.
    """


class HandleIntegrityError(HandleStoreError):
    """Raised when a stored handle's content does not match its identifier.

    Handle identifiers are content addresses.  A mismatch means the file was
    truncated, hand-edited, or corrupted after it was written.
    """


class OrderSemanticsError(HandleStoreError, ValueError):
    """Raised when an ordering claim is internally inconsistent.

    For example, a handle declaring itself ranked without ranking provenance,
    or a score list whose length does not match its member list.
    """


class RerankError(LitSearchError):
    """Base class for failures of the two-stage re-ranking pipeline."""


class ModelContractError(RerankError):
    """Raised when a re-ranking model does not honour the contract it needs to.

    The archetype is a bi-encoder with no ``query`` entry in its ``prompts``
    dictionary.  ``SentenceTransformer.encode_query`` applies the query prefix
    only when that entry exists, and otherwise falls through to a plain
    symmetric encode with no error and no warning -- so the call that is
    supposed to guarantee an asymmetric query mode is a silent no-op on
    exactly the models that most need it.  A model that cannot support
    query-to-document retrieval must fail at load, not produce a plausible
    and badly ordered list.
    """


class RerankInputError(RerankError, ValueError):
    """Raised when the input to a re-rank cannot support a ranking claim.

    For example a blank question: ranking a set against nothing produces an
    order that looks like a ranking and means nothing.
    """


class CoveragePropagationError(HandleStoreError):
    """Raised when a set operation cannot propagate a coverage ledger.

    The handle store deliberately does not know how to combine coverage
    ledgers.  When an operand carries one and no propagator was supplied, the
    operation fails loudly rather than dropping the ledger, because a derived
    set that silently loses its operands' coverage is exactly the confident
    incomplete answer this project exists to prevent.
    """
