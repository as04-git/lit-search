"""Coverage contract: per-source retrieval status that cannot be left unsaid.

This module exists to make one specific failure inexpressible.  A connector
that can return a bare ``list[Paper]`` can always return ``[]`` when its
upstream API rejects the credential, times out, or changes its response
shape.  Downstream, that empty list is indistinguishable from "this database
genuinely holds no matching paper", and the search reports a confident,
well-formed, incomplete answer.

The types here replace the bare list with a pair: the papers **and** an
explicit :class:`SourceCoverage` saying how complete that list is and why.

Three levels are defined:

:class:`CoverageStatus`
    The closed enum of retrieval outcomes.
:class:`SourceCoverage`
    One source's outcome, carrying the detail needed to act on it (what the
    cap was, which credential was missing, which error was raised).
:class:`CoverageLedger`
    A whole result set's coverage: one entry per source consulted, plus
    :class:`DerivationNote` records for shortfalls inherited from set
    operations rather than from a source.

:class:`CoveredPapers` is what a connector returns.  It deliberately does
**not** implement the sequence protocol: code that wants the papers must say
``result.papers``, which is the friction that stops coverage being dropped by
accident.

See ``R7`` in the project specification.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lit_search.core.paper import Paper


class CoverageStatus(StrEnum):
    """How completely a source was retrieved.

    The enum is closed.  A connector may only produce the first six members;
    :data:`UNKNOWN` is reserved for ledger-level aggregation and cannot be
    attached to a :class:`SourceCoverage` (see :meth:`SourceCoverage.__post_init__`).

    Attributes
    ----------
    COMPLETE
        Everything the source could match for this query was retrieved.  This
        is a positive claim, and the type refuses to record it unqualified:
        :class:`SourceCoverage` requires a ``reported_total`` from the source
        and requires ``retrieved`` to have reached it.  The claim is checked
        again, against evidence the connector did not author, by
        :func:`~lit_search.core.retrieval_audit.verify_coverage_claim`.
    CAPPED
        Retrieval stopped at an explicit, known limit -- typically the
        caller's ``max_papers``.  Incomplete, but deliberately and by a known
        amount.
    PARTIAL
        Retrieval was incomplete for a reason that was not a deliberate cap:
        a pagination shortfall, a page that failed mid-run, a source that
        cannot express the full query, or a connector that has not been
        instrumented well enough to justify :data:`COMPLETE`.
    QUERY_UNSUPPORTED
        The source could not express the query at all, so nothing was
        retrieved.  Determinate and reproducible: re-running changes nothing.
    SKIPPED_NO_CREDENTIAL
        The source was never contacted because a required credential was
        absent.  Determinate, and fixable by configuration.
    SKIPPED_BY_POLICY
        The source was deliberately excluded by the selected search profile.
        Determinate and reproducible; an explicit all-source selection changes
        the plan.  This is distinct from a missing credential.
    FAILED
        The source was contacted and errored, contributing nothing.  Unlike
        the two states above, the cause may be transient, so a re-run may
        give a different answer.
    UNKNOWN
        Provenance was never recorded.  The aggregate status of a ledger with
        no entries and no derivation notes.  Ranked worst deliberately: a set
        whose coverage nobody wrote down must never read as complete.
    """

    COMPLETE = "complete"
    CAPPED = "capped"
    PARTIAL = "partial"
    QUERY_UNSUPPORTED = "query_unsupported"
    SKIPPED_NO_CREDENTIAL = "skipped_no_credential"
    SKIPPED_BY_POLICY = "skipped_by_policy"
    FAILED = "failed"
    UNKNOWN = "unknown"


class ExecutionStatus(StrEnum):
    """Whether the declared operation ran without an unplanned shortfall.

    This is deliberately separate from :class:`CoverageStatus`.  A retrieval
    can finish exactly as declared while remaining epistemically bounded (for
    example, because it reached a caller-supplied cap), and a source excluded
    by the declared profile was never an execution obligation at all.
    """

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


def _default_execution_status(status: CoverageStatus) -> ExecutionStatus:
    """Infer execution conservatively for legacy coverage records."""
    if status in {CoverageStatus.COMPLETE, CoverageStatus.CAPPED}:
        return ExecutionStatus.COMPLETE
    if status is CoverageStatus.SKIPPED_BY_POLICY:
        return ExecutionStatus.NOT_APPLICABLE
    if status is CoverageStatus.UNKNOWN:
        return ExecutionStatus.UNKNOWN
    return ExecutionStatus.INCOMPLETE


# Severity ordering used by :func:`worst`.  Higher is worse.
#
# The ordering tracks how much is unknown about what is missing:
#
#   COMPLETE               nothing is missing.
#   CAPPED                 a known amount is missing, on purpose.
#   PARTIAL                an unquantified amount is missing, but the source
#                          did respond and contributed records.
#   QUERY_UNSUPPORTED      the source contributed nothing; determinate.
#   SKIPPED_NO_CREDENTIAL  the source contributed nothing; determinate and
#                          fixable, but it also means the source was never
#                          even asked, so nothing at all is known about it.
#   FAILED                 the source contributed nothing and the reason may
#                          be transient, so not even the zero is reproducible.
#   UNKNOWN                coverage was never recorded.
_SEVERITY: dict[CoverageStatus, int] = {
    CoverageStatus.COMPLETE: 0,
    CoverageStatus.CAPPED: 1,
    CoverageStatus.PARTIAL: 2,
    CoverageStatus.QUERY_UNSUPPORTED: 3,
    CoverageStatus.SKIPPED_BY_POLICY: 4,
    CoverageStatus.SKIPPED_NO_CREDENTIAL: 5,
    CoverageStatus.FAILED: 6,
    CoverageStatus.UNKNOWN: 7,
}

# Statuses under which a source contributed no records at all.
ZERO_COVERAGE_STATUSES: frozenset[CoverageStatus] = frozenset(
    {
        CoverageStatus.QUERY_UNSUPPORTED,
        CoverageStatus.SKIPPED_BY_POLICY,
        CoverageStatus.SKIPPED_NO_CREDENTIAL,
        CoverageStatus.FAILED,
        CoverageStatus.UNKNOWN,
    }
)


def severity(status: CoverageStatus) -> int:
    """Return the severity rank of *status*.

    Parameters
    ----------
    status : CoverageStatus
        Status to rank.

    Returns
    -------
    int
        Rank, where a larger number is a worse coverage outcome.
    """
    return _SEVERITY[status]


def worst(*statuses: CoverageStatus) -> CoverageStatus:
    """Return the worst of *statuses* under the documented severity order.

    This is the conservative propagation primitive: a derived set is never
    better covered than the least-covered thing it was derived from.

    Parameters
    ----------
    *statuses : CoverageStatus
        Statuses to compare.  Calling with no arguments yields
        :data:`CoverageStatus.UNKNOWN`, because nothing is known.

    Returns
    -------
    CoverageStatus
        The worst status supplied.
    """
    if not statuses:
        return CoverageStatus.UNKNOWN
    return max(statuses, key=severity)


@dataclass(frozen=True, slots=True)
class SourceCoverage:
    """One source's retrieval outcome for one operation.

    Instances are immutable.  Prefer the classmethod constructors
    (:meth:`complete`, :meth:`capped`, :meth:`partial`,
    :meth:`query_unsupported`, :meth:`skipped_no_credential`, :meth:`failed`,
    :meth:`from_pagination`) over the raw constructor: they enforce that each
    status carries the detail needed to act on it.

    Parameters
    ----------
    source : str
        Source identifier (for example ``"arxiv"``).
    status : CoverageStatus
        Retrieval outcome.  :data:`CoverageStatus.UNKNOWN` is rejected -- it
        is a ledger-level aggregate only, never something a connector may
        claim.
    retrieved : int
        Number of records this source contributed.  Must be ``0`` for every
        status in :data:`ZERO_COVERAGE_STATUSES`.
    detail : str | None
        Human-readable reason.  Required for
        :data:`~CoverageStatus.PARTIAL` and
        :data:`~CoverageStatus.QUERY_UNSUPPORTED`.
    cap : int | None
        The limit retrieval stopped at.  Required for
        :data:`~CoverageStatus.CAPPED`.
    reported_total : int | None
        Total number of matching records the source itself reported, when it
        reports one.  ``None`` means the source published no total, which is
        itself a reason completeness cannot be verified -- and is therefore
        **rejected** for :data:`~CoverageStatus.COMPLETE`.
    missing_credential : str | None
        Name of the absent credential.  Required for
        :data:`~CoverageStatus.SKIPPED_NO_CREDENTIAL`.
    error : str | None
        Error message.  Required for :data:`~CoverageStatus.FAILED`.
    error_type : str | None
        Exception class name, when the failure came from an exception.

    Raises
    ------
    ValueError
        If the status/detail combination is not actionable, or if a
        zero-coverage status is paired with a non-zero *retrieved*.
    """

    source: str
    status: CoverageStatus
    retrieved: int = 0
    detail: str | None = None
    cap: int | None = None
    reported_total: int | None = None
    missing_credential: str | None = None
    error: str | None = None
    error_type: str | None = None
    execution_status: ExecutionStatus | None = None

    def __post_init__(self) -> None:
        """Validate that the status carries enough detail to be acted on.

        Raises
        ------
        ValueError
            If a required field for the given status is absent, or if the
            record is internally inconsistent.
        """
        if not self.source:
            raise ValueError("SourceCoverage.source must be a non-empty identifier")
        if self.execution_status is None:
            object.__setattr__(self, "execution_status", _default_execution_status(self.status))
        if (
            self.status is CoverageStatus.SKIPPED_BY_POLICY
            and self.execution_status is not ExecutionStatus.NOT_APPLICABLE
        ):
            raise ValueError("SKIPPED_BY_POLICY execution must be NOT_APPLICABLE")
        if (
            self.status is not CoverageStatus.SKIPPED_BY_POLICY
            and self.execution_status is ExecutionStatus.NOT_APPLICABLE
        ):
            raise ValueError("NOT_APPLICABLE execution is reserved for SKIPPED_BY_POLICY")
        if self.status is CoverageStatus.UNKNOWN:
            raise ValueError(
                "CoverageStatus.UNKNOWN is a ledger-level aggregate and cannot be "
                "attached to a source. A source that produced nothing must say why: "
                "use failed(), skipped_no_credential(), or query_unsupported()."
            )
        if self.retrieved < 0:
            raise ValueError("SourceCoverage.retrieved must not be negative")
        if self.status in ZERO_COVERAGE_STATUSES and self.retrieved != 0:
            raise ValueError(
                f"status '{self.status}' means the source contributed nothing, "
                f"but retrieved={self.retrieved}. A source that returned records "
                f"and then failed is PARTIAL, not {self.status}."
            )
        if self.status is CoverageStatus.COMPLETE:
            self._check_complete_is_justifiable()
        if self.status is CoverageStatus.CAPPED and self.cap is None:
            raise ValueError("CAPPED requires 'cap': the limit retrieval stopped at")
        if self.status is CoverageStatus.PARTIAL and not self.detail:
            raise ValueError("PARTIAL requires 'detail': why the retrieval fell short")
        if self.status is CoverageStatus.QUERY_UNSUPPORTED and not self.detail:
            raise ValueError("QUERY_UNSUPPORTED requires 'detail': what could not be expressed")
        if self.status is CoverageStatus.SKIPPED_BY_POLICY and not self.detail:
            raise ValueError("SKIPPED_BY_POLICY requires 'detail': which policy excluded it")
        if self.status is CoverageStatus.SKIPPED_NO_CREDENTIAL and not self.missing_credential:
            raise ValueError(
                "SKIPPED_NO_CREDENTIAL requires 'missing_credential': which credential was absent"
            )
        if self.status is CoverageStatus.FAILED and not self.error:
            raise ValueError("FAILED requires 'error': what went wrong")

    def _check_complete_is_justifiable(self) -> None:
        """Refuse a ``COMPLETE`` that is not even internally justifiable.

        ``COMPLETE`` is a claim about the source, not about the connector's
        satisfaction with its own run, so the record has to carry the source's
        own count and has to have reached it.  Without this, ``COMPLETE`` is a
        bare assertion: a connector that never asked how many records existed
        could claim to have retrieved all of them.

        This is only the *arithmetic* half.  ``reported_total`` is still a
        number the connector supplied, and it is checked against independently
        recorded evidence by
        :func:`~lit_search.core.retrieval_audit.verify_coverage_claim`.

        Raises
        ------
        ValueError
            If no ``reported_total`` is present, or fewer records were
            retrieved than the source said existed.
        """
        if self.reported_total is None:
            raise ValueError(
                f"COMPLETE requires 'reported_total' for source '{self.source}': the "
                "number of matching records the source itself reported. A connector "
                "that cannot say how many records existed cannot claim to have "
                "retrieved all of them -- use partial() and say what is unverified."
            )
        if self.retrieved < self.reported_total:
            raise ValueError(
                f"COMPLETE claims everything was retrieved from '{self.source}', but "
                f"retrieved={self.retrieved} is fewer than the {self.reported_total} "
                f"records the source reported. That is PARTIAL, not COMPLETE."
            )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def complete(
        cls,
        source: str,
        retrieved: int,
        *,
        reported_total: int,
    ) -> SourceCoverage:
        """Build a :data:`~CoverageStatus.COMPLETE` record.

        Only use this when the connector can justify the claim.  If in doubt,
        use :meth:`partial` and say what is unverified.

        *reported_total* is **required**.  Completeness is a claim about the
        source, so it needs the source's own count; a connector with no total
        to compare against has nothing to justify the claim with.

        Parameters
        ----------
        source : str
            Source identifier.
        retrieved : int
            Number of records retrieved.
        reported_total : int
            Total the source itself reported.  Must not exceed *retrieved*.

        Returns
        -------
        SourceCoverage
            Coverage record.

        Raises
        ------
        ValueError
            If fewer records were retrieved than the source reported.
        """
        return cls(
            source=source,
            status=CoverageStatus.COMPLETE,
            retrieved=retrieved,
            reported_total=reported_total,
            execution_status=ExecutionStatus.COMPLETE,
        )

    @classmethod
    def capped(
        cls,
        source: str,
        retrieved: int,
        *,
        cap: int,
        reported_total: int | None = None,
        detail: str | None = None,
        execution_status: ExecutionStatus = ExecutionStatus.COMPLETE,
    ) -> SourceCoverage:
        """Build a :data:`~CoverageStatus.CAPPED` record.

        Parameters
        ----------
        source : str
            Source identifier.
        retrieved : int
            Number of records retrieved.
        cap : int
            The limit retrieval stopped at.
        reported_total : int | None
            Total the source reported, when it reports one.
        detail : str | None
            Optional extra context.

        Returns
        -------
        SourceCoverage
            Coverage record.
        """
        return cls(
            source=source,
            status=CoverageStatus.CAPPED,
            retrieved=retrieved,
            cap=cap,
            reported_total=reported_total,
            detail=detail,
            execution_status=execution_status,
        )

    @classmethod
    def partial(
        cls,
        source: str,
        retrieved: int,
        *,
        detail: str,
        reported_total: int | None = None,
        error: str | None = None,
        error_type: str | None = None,
        execution_status: ExecutionStatus = ExecutionStatus.INCOMPLETE,
    ) -> SourceCoverage:
        """Build a :data:`~CoverageStatus.PARTIAL` record.

        Parameters
        ----------
        source : str
            Source identifier.
        retrieved : int
            Number of records retrieved before the shortfall.
        detail : str
            Why the retrieval fell short.  Required.
        reported_total : int | None
            Total the source reported, when it reports one.
        error : str | None
            Error message, when the shortfall was caused by an error partway
            through a run that had already produced records.
        error_type : str | None
            Exception class name for that error.

        Returns
        -------
        SourceCoverage
            Coverage record.
        """
        return cls(
            source=source,
            status=CoverageStatus.PARTIAL,
            retrieved=retrieved,
            detail=detail,
            reported_total=reported_total,
            error=error,
            error_type=error_type,
            execution_status=execution_status,
        )

    @classmethod
    def query_unsupported(cls, source: str, *, detail: str) -> SourceCoverage:
        """Build a :data:`~CoverageStatus.QUERY_UNSUPPORTED` record.

        Parameters
        ----------
        source : str
            Source identifier.
        detail : str
            What the source could not express.

        Returns
        -------
        SourceCoverage
            Coverage record.
        """
        return cls(
            source=source,
            status=CoverageStatus.QUERY_UNSUPPORTED,
            detail=detail,
            execution_status=ExecutionStatus.INCOMPLETE,
        )

    @classmethod
    def skipped_no_credential(cls, source: str, *, credential: str) -> SourceCoverage:
        """Build a :data:`~CoverageStatus.SKIPPED_NO_CREDENTIAL` record.

        Parameters
        ----------
        source : str
            Source identifier.
        credential : str
            Name of the absent credential, phrased so a user can supply it
            (for example ``"ieee_api_key"``).

        Returns
        -------
        SourceCoverage
            Coverage record.
        """
        return cls(
            source=source,
            status=CoverageStatus.SKIPPED_NO_CREDENTIAL,
            missing_credential=credential,
            detail=f"required credential '{credential}' was not provided",
            execution_status=ExecutionStatus.INCOMPLETE,
        )

    @classmethod
    def skipped_by_policy(cls, source: str, *, detail: str) -> SourceCoverage:
        """Build a deliberate search-profile exclusion record."""
        return cls(
            source=source,
            status=CoverageStatus.SKIPPED_BY_POLICY,
            detail=detail,
            execution_status=ExecutionStatus.NOT_APPLICABLE,
        )

    @classmethod
    def failed(
        cls,
        source: str,
        *,
        error: str,
        error_type: str | None = None,
    ) -> SourceCoverage:
        """Build a :data:`~CoverageStatus.FAILED` record.

        Parameters
        ----------
        source : str
            Source identifier.
        error : str
            What went wrong.
        error_type : str | None
            Exception class name.

        Returns
        -------
        SourceCoverage
            Coverage record.
        """
        return cls(
            source=source,
            status=CoverageStatus.FAILED,
            error=error,
            error_type=error_type,
            detail=error,
            execution_status=ExecutionStatus.INCOMPLETE,
        )

    @classmethod
    def from_pagination(
        cls,
        source: str,
        *,
        retrieved: int,
        reported_total: int | None,
        cap: int | None = None,
        shortfall_detail: str | None = None,
    ) -> SourceCoverage:
        """Derive coverage from a paginated retrieval's own counters.

        This encodes the decision procedure once so that every connector
        makes the same call:

        1. If *cap* is set and *retrieved* reached it, the run was
           :data:`~CoverageStatus.CAPPED`.
        2. Else if the source reported no total, completeness cannot be
           verified: :data:`~CoverageStatus.PARTIAL`.
        3. Else if the number retrieved does not equal the total the source
           reported -- in *either* direction -- the run is
           :data:`~CoverageStatus.PARTIAL`.  Fewer is an obvious shortfall.
           **More is not a bonus**: a cursor walk that returns more distinct
           records than the source says exist has re-served part of the set,
           and a walk that goes backwards has also skipped what lay ahead of
           the point it restarted from.  Both CrossRef and HAL were observed
           doing exactly this on 2026-08-31 -- CrossRef answers a cursor it
           cannot parse with HTTP 200, page one again, and a **fresh**
           next-cursor token, so a seen-cursor guard does not see it -- and
           before this branch existed the result came back ``COMPLETE``.
        4. Else if *shortfall_detail* was supplied, something was lost that
           the counters alone do not show -- records the source returned and
           the connector could not read, most often -- so the run is
           :data:`~CoverageStatus.PARTIAL` and carries the detail.
        5. Otherwise :data:`~CoverageStatus.COMPLETE`.

        Step 2 is deliberately pessimistic.  A source that publishes no
        result count gives a connector no way to prove it saw everything, and
        assuming it did is exactly the failure this module exists to prevent.

        Step 4 exists because the earlier version of this helper threaded
        *shortfall_detail* into every branch **except** the one claiming
        nothing was missing, so a connector that dutifully reported "2 of 10
        records could not be parsed" got back ``COMPLETE`` with the note
        silently dropped.  A record the source returned and the connector
        could not read is a coverage loss like any other.

        Parameters
        ----------
        source : str
            Source identifier.
        retrieved : int
            Number of records retrieved.
        reported_total : int | None
            Total the source reported.  Pass ``None`` only when the source
            genuinely publishes no count.
        cap : int | None
            Caller-imposed maximum, when one was applied.
        shortfall_detail : str | None
            Extra context appended to the PARTIAL detail (for example a page
            that failed).

        Returns
        -------
        SourceCoverage
            Coverage record.
        """
        if cap is not None and retrieved >= cap:
            return cls.capped(
                source,
                retrieved,
                cap=cap,
                reported_total=reported_total,
                detail=shortfall_detail,
                execution_status=(
                    ExecutionStatus.INCOMPLETE if shortfall_detail else ExecutionStatus.COMPLETE
                ),
            )
        if reported_total is None:
            detail = "source reported no result total, so completeness cannot be verified" + (
                f"; {shortfall_detail}" if shortfall_detail else ""
            )
            return cls.partial(
                source,
                retrieved,
                detail=detail,
                execution_status=(
                    ExecutionStatus.INCOMPLETE if shortfall_detail else ExecutionStatus.COMPLETE
                ),
            )
        if retrieved < reported_total:
            detail = f"retrieved {retrieved} of {reported_total} records the source reported" + (
                f"; {shortfall_detail}" if shortfall_detail else ""
            )
            return cls.partial(source, retrieved, detail=detail, reported_total=reported_total)
        if retrieved > reported_total:
            detail = (
                f"retrieved {retrieved} records while the source reported only "
                f"{reported_total}; a paginated walk cannot honestly return more distinct "
                "records than the source says exist, so either some records were served "
                "twice -- in which case the walk also skipped others -- or the total is "
                "not a count of this result set. Completeness is not established either way"
            ) + (f"; {shortfall_detail}" if shortfall_detail else "")
            return cls.partial(source, retrieved, detail=detail, reported_total=reported_total)
        if shortfall_detail:
            return cls.partial(
                source,
                retrieved,
                detail=(
                    f"retrieved {retrieved} of {reported_total} records the source "
                    f"reported, but something was lost on the way: {shortfall_detail}"
                ),
                reported_total=reported_total,
            )
        return cls.complete(source, retrieved, reported_total=reported_total)

    # ------------------------------------------------------------------
    # Rendering and serialization
    # ------------------------------------------------------------------

    @property
    def is_complete(self) -> bool:
        """Whether this source was retrieved completely.

        Returns
        -------
        bool
            ``True`` only for :data:`~CoverageStatus.COMPLETE`.
        """
        return self.status is CoverageStatus.COMPLETE

    def describe(self) -> str:
        """Return a one-line human-readable summary.

        Returns
        -------
        str
            Summary line, for example
            ``"arxiv: partial (retrieved 40 of 512 ...)"``.
        """
        head = f"{self.source}: {self.status.value} ({self.retrieved} papers)"
        if self.status is CoverageStatus.CAPPED:
            head += f" -- stopped at cap {self.cap}"
            if self.reported_total is not None:
                head += f" of {self.reported_total} reported"
        if self.detail:
            head += f" -- {self.detail}"
        if self.error_type and self.error_type not in (self.detail or ""):
            head += f" [{self.error_type}]"
        return head

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary with every non-``None`` field.
        """
        data = dataclasses.asdict(self)
        data["status"] = self.status.value
        execution_status = self.execution_status
        assert execution_status is not None
        data["execution_status"] = execution_status.value
        return {k: v for k, v in data.items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceCoverage:
        """Reconstruct from the format produced by :meth:`to_dict`.

        Parameters
        ----------
        data : dict[str, Any]
            Serialized record.

        Returns
        -------
        SourceCoverage
            Reconstructed record.
        """
        return cls(
            source=data["source"],
            status=CoverageStatus(data["status"]),
            retrieved=int(data.get("retrieved", 0)),
            detail=data.get("detail"),
            cap=data.get("cap"),
            reported_total=data.get("reported_total"),
            missing_credential=data.get("missing_credential"),
            error=data.get("error"),
            error_type=data.get("error_type"),
            execution_status=(
                ExecutionStatus(data["execution_status"])
                if data.get("execution_status") is not None
                else _default_execution_status(CoverageStatus(data["status"]))
            ),
        )


@dataclass(frozen=True, slots=True)
class DerivationNote:
    """A coverage shortfall inherited from a set operation, not from a source.

    A difference taken against an incomplete set is itself incomplete even
    when every source of the left-hand set was complete: records that should
    have been subtracted were never seen.  That shortfall belongs to the
    *operation*, not to any source the result was searched from, so recording
    it as a source entry would claim something false ("we searched Zotero and
    it was partial" when in fact Zotero was subtracted).

    Parameters
    ----------
    operation : str
        Operation that introduced the shortfall (``"difference"``,
        ``"union"``, ``"intersection"``).
    role : str
        Which operand fell short (for example ``"subtrahend"``).
    status : CoverageStatus
        Floor the note imposes on the derived set's aggregate status.
    detail : str
        Human-readable explanation of the consequence.
    """

    operation: str
    role: str
    status: CoverageStatus
    detail: str
    execution_status: ExecutionStatus | None = None

    def __post_init__(self) -> None:
        """Infer execution for legacy notes while preserving explicit evidence."""
        if self.execution_status is None:
            object.__setattr__(self, "execution_status", _default_execution_status(self.status))

    def describe(self) -> str:
        """Return a one-line human-readable summary.

        Returns
        -------
        str
            Summary line.
        """
        return f"{self.operation} ({self.role}): {self.status.value} -- {self.detail}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, Any]
            Serialized note.
        """
        execution_status = self.execution_status
        assert execution_status is not None
        return {
            "operation": self.operation,
            "role": self.role,
            "status": self.status.value,
            "detail": self.detail,
            "execution_status": execution_status.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DerivationNote:
        """Reconstruct from the format produced by :meth:`to_dict`.

        Parameters
        ----------
        data : dict[str, Any]
            Serialized note.

        Returns
        -------
        DerivationNote
            Reconstructed note.
        """
        return cls(
            operation=data["operation"],
            role=data["role"],
            status=CoverageStatus(data["status"]),
            detail=data["detail"],
            execution_status=(
                ExecutionStatus(data["execution_status"])
                if data.get("execution_status") is not None
                else _default_execution_status(CoverageStatus(data["status"]))
            ),
        )


@dataclass(frozen=True, slots=True)
class CoverageScope:
    """The declared source boundary, separate from execution outcomes."""

    profile: str
    selected_sources: tuple[str, ...] = ()
    registered_sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Freeze stable, duplicate-free source lists and validate the boundary."""
        if not self.profile:
            raise ValueError("CoverageScope.profile must be non-empty")
        selected = tuple(dict.fromkeys(self.selected_sources))
        registered = tuple(dict.fromkeys(self.registered_sources))
        if any(source not in registered for source in selected):
            raise ValueError("CoverageScope.selected_sources must be in registered_sources")
        object.__setattr__(self, "selected_sources", selected)
        object.__setattr__(self, "registered_sources", registered)

    @property
    def excluded_sources(self) -> tuple[str, ...]:
        """Registered sources outside the declared selection."""
        selected = set(self.selected_sources)
        return tuple(source for source in self.registered_sources if source not in selected)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the immutable declaration."""
        return {
            "profile": self.profile,
            "selected_sources": list(self.selected_sources),
            "registered_sources": list(self.registered_sources),
            "excluded_sources": list(self.excluded_sources),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CoverageScope:
        """Read a declaration, recomputing exclusions rather than trusting them."""
        return cls(
            profile=str(data.get("profile") or "unknown"),
            selected_sources=tuple(str(item) for item in data.get("selected_sources", ())),
            registered_sources=tuple(str(item) for item in data.get("registered_sources", ())),
        )


@dataclass(frozen=True, slots=True)
class CoverageLedger:
    """A whole result set's coverage: one entry per source, plus derivations.

    Duplicate entries for the same source are collapsed on construction,
    worst status winning, so that merging ledgers never needs the caller to
    deduplicate by hand.

    Parameters
    ----------
    entries : tuple[SourceCoverage, ...]
        Per-source outcomes for the sources that were actually consulted for
        content.
    derivations : tuple[DerivationNote, ...]
        Shortfalls inherited from set operations.
    """

    entries: tuple[SourceCoverage, ...] = ()
    derivations: tuple[DerivationNote, ...] = field(default=())
    scope: CoverageScope | None = None

    def __post_init__(self) -> None:
        """Normalize inputs to tuples and collapse duplicate sources."""
        object.__setattr__(self, "entries", _collapse(self.entries))
        object.__setattr__(self, "derivations", tuple(self.derivations))

    @classmethod
    def from_sources(cls, entries: Iterable[SourceCoverage]) -> CoverageLedger:
        """Build a ledger from an iterable of source records.

        Parameters
        ----------
        entries : Iterable[SourceCoverage]
            Per-source outcomes.

        Returns
        -------
        CoverageLedger
            Ledger with duplicates collapsed worst-wins.
        """
        return cls(entries=tuple(entries))

    # ------------------------------------------------------------------
    # Aggregate view
    # ------------------------------------------------------------------

    @property
    def status(self) -> CoverageStatus:
        """Worst status across every entry and derivation note.

        Returns
        -------
        CoverageStatus
            The aggregate status.  :data:`CoverageStatus.UNKNOWN` when the
            ledger records nothing at all, because a set whose provenance was
            never written down must not read as complete.
        """
        return worst(
            *(entry.status for entry in self.entries),
            *(note.status for note in self.derivations),
        )

    @property
    def is_complete(self) -> bool:
        """Whether every source was complete and no derivation degraded it.

        Returns
        -------
        bool
            ``True`` only when the aggregate status is
            :data:`~CoverageStatus.COMPLETE`.
        """
        return self.status is CoverageStatus.COMPLETE

    @property
    def execution_status(self) -> ExecutionStatus:
        """Aggregate declared-plan execution, ignoring policy exclusions."""
        statuses = [
            status
            for status in (
                *(entry.execution_status for entry in self.entries),
                *(note.execution_status for note in self.derivations),
            )
            if status is not None and status is not ExecutionStatus.NOT_APPLICABLE
        ]
        if any(status is ExecutionStatus.INCOMPLETE for status in statuses):
            return ExecutionStatus.INCOMPLETE
        if not statuses or any(status is ExecutionStatus.UNKNOWN for status in statuses):
            return ExecutionStatus.UNKNOWN
        return ExecutionStatus.COMPLETE

    @property
    def execution_is_complete(self) -> bool:
        """Whether every applicable declared unit completed."""
        return self.execution_status is ExecutionStatus.COMPLETE

    @property
    def execution_shortfalls(self) -> tuple[str, ...]:
        """Labels for definite or unknown execution shortfalls."""
        source_labels = tuple(
            f"{entry.source} ({entry.status.value})"
            for entry in self.entries
            if entry.execution_status in {ExecutionStatus.INCOMPLETE, ExecutionStatus.UNKNOWN}
        )
        derivation_labels = tuple(
            f"{note.operation}/{note.role} ({note.status.value})"
            for note in self.derivations
            if note.execution_status in {ExecutionStatus.INCOMPLETE, ExecutionStatus.UNKNOWN}
        )
        return source_labels + derivation_labels

    @property
    def sources(self) -> tuple[str, ...]:
        """Identifiers of every source consulted, in ledger order.

        Returns
        -------
        tuple[str, ...]
            Source identifiers.
        """
        return tuple(entry.source for entry in self.entries)

    @property
    def complete_sources(self) -> tuple[str, ...]:
        """Identifiers of the sources that were retrieved completely.

        Returns
        -------
        tuple[str, ...]
            Source identifiers.
        """
        return tuple(e.source for e in self.entries if e.is_complete)

    @property
    def incomplete_entries(self) -> tuple[SourceCoverage, ...]:
        """Entries whose status is anything other than complete.

        Returns
        -------
        tuple[SourceCoverage, ...]
            Entries needing the reader's attention.
        """
        return tuple(e for e in self.entries if not e.is_complete)

    def get(self, source: str) -> SourceCoverage | None:
        """Return the entry for *source*, if present.

        Parameters
        ----------
        source : str
            Source identifier.

        Returns
        -------
        SourceCoverage | None
            The entry, or ``None`` when the source was not consulted.
        """
        for entry in self.entries:
            if entry.source == source:
                return entry
        return None

    def __iter__(self) -> Iterator[SourceCoverage]:
        """Iterate over per-source entries.

        Returns
        -------
        Iterator[SourceCoverage]
            Iterator over entries.
        """
        return iter(self.entries)

    def __len__(self) -> int:
        """Return the number of sources consulted.

        Returns
        -------
        int
            Entry count.
        """
        return len(self.entries)

    # ------------------------------------------------------------------
    # Rendering and serialization
    # ------------------------------------------------------------------

    def render(self) -> str:
        """Render the ledger as honest multi-line text.

        The rendering always states the aggregate first, then every source
        that was consulted with its outcome, then any derivation shortfall.
        It never omits a non-complete source.

        Returns
        -------
        str
            Multi-line report.
        """
        lines = [
            f"Execution: {self.execution_status.value}",
            f"Coverage: {self.status.value}",
        ]
        if not self.entries and not self.derivations:
            lines.append("  (no coverage was recorded for this set)")
            return "\n".join(lines)

        if self.entries:
            selected_entries = tuple(
                entry
                for entry in self.entries
                if entry.status is not CoverageStatus.SKIPPED_BY_POLICY
            )
            complete = tuple(entry for entry in selected_entries if entry.is_complete)
            lines.append(
                f"  sources in declared plan: {len(selected_entries)} "
                f"({len(complete)} semantically complete, "
                f"{len(selected_entries) - len(complete)} bounded or incomplete)"
            )
            for entry in self.entries:
                if entry.execution_status in {
                    ExecutionStatus.INCOMPLETE,
                    ExecutionStatus.UNKNOWN,
                }:
                    marker = "  !! "
                elif entry.is_complete:
                    marker = "  ok "
                else:
                    marker = "  -- "
                lines.append(f"{marker}{entry.describe()}")
        for note in self.derivations:
            marker = (
                "  !! "
                if note.execution_status in {ExecutionStatus.INCOMPLETE, ExecutionStatus.UNKNOWN}
                else "  -- "
            )
            lines.append(f"{marker}{note.describe()}")
        if self.scope is not None:
            lines.append(
                f"  scope {self.scope.profile}: {len(self.scope.selected_sources)} of "
                f"{len(self.scope.registered_sources)} registered source(s) selected"
            )
            if self.scope.excluded_sources:
                lines.append(f"  -- excluded: {', '.join(self.scope.excluded_sources)}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        The aggregate ``status`` is written too.  It is derived, not stored,
        but writing it means a reader of the raw JSON cannot miss it.

        Returns
        -------
        dict[str, Any]
            Serialized ledger.
        """
        return {
            "status": self.status.value,
            "sources": [entry.to_dict() for entry in self.entries],
            "derivations": [note.to_dict() for note in self.derivations],
            "execution": {
                "status": self.execution_status.value,
                "complete": self.execution_is_complete,
                "shortfalls": list(self.execution_shortfalls),
            },
            "scope": self.scope.to_dict() if self.scope is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CoverageLedger:
        """Reconstruct from the format produced by :meth:`to_dict`.

        The stored aggregate ``status`` is ignored and recomputed, so a
        hand-edited or stale file cannot claim a completeness its entries do
        not support.

        Parameters
        ----------
        data : dict[str, Any]
            Serialized ledger.

        Returns
        -------
        CoverageLedger
            Reconstructed ledger.
        """
        entries = tuple(SourceCoverage.from_dict(e) for e in data.get("sources", ()))
        scope_raw = data.get("scope")
        if isinstance(scope_raw, dict):
            scope = CoverageScope.from_dict(scope_raw)
        elif "scope" not in data and entries:
            selected = tuple(
                entry.source
                for entry in entries
                if entry.status is not CoverageStatus.SKIPPED_BY_POLICY
            )
            scope = CoverageScope(
                profile="unknown",
                selected_sources=selected,
                registered_sources=tuple(entry.source for entry in entries),
            )
        else:
            scope = None
        return cls(
            entries=entries,
            derivations=tuple(DerivationNote.from_dict(d) for d in data.get("derivations", ())),
            scope=scope,
        )


def _collapse(entries: Iterable[SourceCoverage]) -> tuple[SourceCoverage, ...]:
    """Collapse duplicate source entries, keeping the worst per source.

    First-seen order is preserved so that renderings stay stable.

    Parameters
    ----------
    entries : Iterable[SourceCoverage]
        Entries, possibly with repeated sources.

    Returns
    -------
    tuple[SourceCoverage, ...]
        One entry per source.
    """
    merged: dict[str, SourceCoverage] = {}
    for entry in entries:
        existing = merged.get(entry.source)
        if existing is None or (
            severity(entry.status),
            _execution_severity(entry.execution_status),
        ) > (severity(existing.status), _execution_severity(existing.execution_status)):
            merged[entry.source] = entry
    return tuple(merged.values())


def _execution_severity(status: ExecutionStatus | None) -> int:
    """Rank execution evidence when semantic statuses tie during collapse."""

    return {
        None: 3,
        ExecutionStatus.COMPLETE: 0,
        ExecutionStatus.NOT_APPLICABLE: 0,
        ExecutionStatus.INCOMPLETE: 2,
        ExecutionStatus.UNKNOWN: 3,
    }[status]


# ----------------------------------------------------------------------
# Set-operation propagation
# ----------------------------------------------------------------------


def union_coverage(*ledgers: CoverageLedger) -> CoverageLedger:
    """Propagate coverage through a union.

    A union is missing anything that either operand was missing, so every
    source appears in the result and each keeps its worst observed status.

    Parameters
    ----------
    *ledgers : CoverageLedger
        Operand ledgers.

    Returns
    -------
    CoverageLedger
        Ledger for the union.
    """
    return _merge_operands("union", ledgers)


def intersection_coverage(*ledgers: CoverageLedger) -> CoverageLedger:
    """Propagate coverage through an intersection.

    An intersection inherits the same worst-per-source rule as a union.  It
    is not safer for being smaller: a record that truly belongs in both sets
    is absent from the intersection whenever *either* operand failed to
    retrieve it, so an incomplete operand makes the intersection under-count.

    Parameters
    ----------
    *ledgers : CoverageLedger
        Operand ledgers.

    Returns
    -------
    CoverageLedger
        Ledger for the intersection.
    """
    return _merge_operands("intersection", ledgers)


def _merge_operands(operation: str, ledgers: tuple[CoverageLedger, ...]) -> CoverageLedger:
    """Merge operand ledgers for a content-combining operation.

    Parameters
    ----------
    operation : str
        Operation name, used only when no operand was supplied.
    ledgers : tuple[CoverageLedger, ...]
        Operand ledgers.

    Returns
    -------
    CoverageLedger
        Merged ledger.
    """
    if not ledgers:
        return CoverageLedger(
            derivations=(
                DerivationNote(
                    operation=operation,
                    role="operands",
                    status=CoverageStatus.UNKNOWN,
                    detail="no operand carried a coverage ledger",
                    execution_status=ExecutionStatus.UNKNOWN,
                ),
            )
        )
    entries: list[SourceCoverage] = []
    derivations: list[DerivationNote] = []
    for index, ledger in enumerate(ledgers):
        entries.extend(ledger.entries)
        derivations.extend(ledger.derivations)
        # An operand that recorded nothing at all contributes no entry, so
        # concatenation alone would let its UNKNOWN vanish and the merged set
        # would inherit only the well-covered operand's status. That is the
        # motivating failure with extra steps, so it is recorded explicitly.
        if not ledger.entries and not ledger.derivations:
            derivations.append(
                DerivationNote(
                    operation=operation,
                    role=f"operand {index}",
                    status=CoverageStatus.UNKNOWN,
                    detail=(
                        "an operand carried no coverage ledger, so nothing is known about "
                        "what it may have failed to retrieve"
                    ),
                    execution_status=ExecutionStatus.UNKNOWN,
                )
            )
    merged = CoverageLedger(
        entries=tuple(entries),
        derivations=tuple(derivations),
        scope=_merge_scope(ledgers),
    )

    # Conservative-propagation invariant: the merged set must never report
    # better coverage than the worst operand did.
    floor = worst(*(ledger.status for ledger in ledgers))
    if severity(merged.status) < severity(floor):
        merged = CoverageLedger(
            entries=merged.entries,
            derivations=(
                *merged.derivations,
                DerivationNote(
                    operation=operation,
                    role="operands",
                    status=floor,
                    detail=(
                        f"an operand was '{floor.value}'; the derived set cannot be better "
                        "covered than what it was derived from"
                    ),
                    execution_status=_aggregate_execution(ledgers),
                ),
            ),
            scope=merged.scope,
        )
    operand_execution = _aggregate_execution(ledgers)
    if merged.execution_status is not operand_execution:
        merged = CoverageLedger(
            entries=merged.entries,
            derivations=(
                *merged.derivations,
                DerivationNote(
                    operation=operation,
                    role="operand execution",
                    status=CoverageStatus.COMPLETE,
                    detail=(
                        f"operand execution aggregated to '{operand_execution.value}'; "
                        "duplicate-source coverage collapse must not erase that verdict"
                    ),
                    execution_status=operand_execution,
                ),
            ),
            scope=merged.scope,
        )
    return merged


def _aggregate_execution(ledgers: Iterable[CoverageLedger]) -> ExecutionStatus:
    """Aggregate operand execution without confusing scope with failure."""
    statuses: list[ExecutionStatus] = []
    saw_policy_only = False
    for ledger in ledgers:
        raw = (
            *(entry.execution_status for entry in ledger.entries),
            *(note.execution_status for note in ledger.derivations),
        )
        if any(
            status is not None and status is not ExecutionStatus.NOT_APPLICABLE for status in raw
        ):
            statuses.append(ledger.execution_status)
        elif not ledger.entries and not ledger.derivations:
            statuses.append(ExecutionStatus.UNKNOWN)
        else:
            saw_policy_only = True
    if any(status is ExecutionStatus.INCOMPLETE for status in statuses):
        return ExecutionStatus.INCOMPLETE
    if not statuses:
        return ExecutionStatus.COMPLETE if saw_policy_only else ExecutionStatus.UNKNOWN
    if any(status is ExecutionStatus.UNKNOWN for status in statuses):
        return ExecutionStatus.UNKNOWN
    return ExecutionStatus.COMPLETE


def _merge_scope(ledgers: Iterable[CoverageLedger]) -> CoverageScope | None:
    """Preserve the union of declared boundaries for a derived set."""
    scopes = [ledger.scope for ledger in ledgers if ledger.scope is not None]
    if not scopes:
        return None
    if len(scopes) == 1:
        return scopes[0]
    selected = tuple(dict.fromkeys(source for scope in scopes for source in scope.selected_sources))
    registered = tuple(
        dict.fromkeys(source for scope in scopes for source in scope.registered_sources)
    )
    return CoverageScope(
        profile="derived",
        selected_sources=selected,
        registered_sources=registered,
    )


def difference_coverage(
    minuend: CoverageLedger,
    subtrahend: CoverageLedger,
    *,
    subtrahend_label: str = "subtrahend set",
) -> CoverageLedger:
    """Propagate coverage through a difference (``minuend`` minus ``subtrahend``).

    This is the asymmetric case, and the one that motivated the rule.  The
    subtrahend's sources are **not** sources of the result: nothing was
    searched from them, they were subtracted.  So they do not become entries.
    But if the subtrahend was not complete, then records that should have been
    removed were never seen, and the result confidently contains rows it
    should not -- the "already in the Zotero library but reported as new"
    failure.  That shortfall is recorded as a :class:`DerivationNote`, which
    floors the aggregate status without making a false claim about which
    sources were searched.

    A complete subtrahend adds nothing: the result is exactly as covered as
    the minuend.

    Parameters
    ----------
    minuend : CoverageLedger
        Coverage of the set being subtracted from.
    subtrahend : CoverageLedger
        Coverage of the set being subtracted.
    subtrahend_label : str
        Human-readable name for the subtrahend, used in the note.

    Returns
    -------
    CoverageLedger
        Ledger for the difference.
    """
    subtrahend_status = subtrahend.status
    if subtrahend_status is CoverageStatus.COMPLETE:
        return CoverageLedger(
            entries=minuend.entries,
            derivations=minuend.derivations,
            scope=minuend.scope,
        )

    if subtrahend.entries:
        offenders = ", ".join(f"{e.source}={e.status.value}" for e in subtrahend.incomplete_entries)
        cause = offenders or "no source was complete"
    else:
        cause = "no coverage was recorded for it"

    note = DerivationNote(
        operation="difference",
        role="subtrahend",
        status=subtrahend_status,
        detail=(
            f"{subtrahend_label} was '{subtrahend_status.value}' ({cause}), so records it "
            "failed to retrieve were not subtracted and may be present here in error"
        ),
        execution_status=_aggregate_execution((subtrahend,)),
    )
    return CoverageLedger(
        entries=minuend.entries,
        derivations=(*minuend.derivations, note),
        scope=minuend.scope,
    )


# ----------------------------------------------------------------------
# Connector return type
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CoveredPapers:
    """Papers returned by one source, inseparable from that source's coverage.

    This is the only thing a search connector may return.  It deliberately
    does **not** implement ``__iter__``, ``__len__`` or ``__getitem__``:
    making it usable as a drop-in list would let every downstream caller keep
    treating it as one and drop :attr:`coverage` by accident, which is the
    exact failure the type exists to prevent.  Callers must say
    ``result.papers``, and in doing so are one attribute away from
    ``result.coverage``.

    Parameters
    ----------
    papers : tuple[Paper, ...]
        Records retrieved.  Any iterable of papers is accepted and frozen
        into a tuple.
    coverage : SourceCoverage
        How complete :attr:`papers` is, and why.

    Raises
    ------
    ValueError
        If ``coverage.retrieved`` disagrees with ``len(papers)``.  A mismatch
        means the connector's own accounting is wrong, and a wrong count is
        how an unnoticed shortfall gets reported as a complete result.
    """

    papers: tuple[Paper, ...]
    coverage: SourceCoverage

    def __post_init__(self) -> None:
        """Freeze *papers* into a tuple and check it against the count.

        Raises
        ------
        ValueError
            If the coverage record's ``retrieved`` count does not match the
            number of papers supplied.
        """
        object.__setattr__(self, "papers", tuple(self.papers))
        if self.coverage.retrieved != len(self.papers):
            raise ValueError(
                f"coverage.retrieved ({self.coverage.retrieved}) does not match the number "
                f"of papers returned ({len(self.papers)}) for source "
                f"'{self.coverage.source}'. Records that were seen but could not be parsed "
                f"are a coverage loss: report them with SourceCoverage.partial()."
            )

    @property
    def status(self) -> CoverageStatus:
        """Shortcut to ``self.coverage.status``.

        Returns
        -------
        CoverageStatus
            This source's status.
        """
        return self.coverage.status

    @classmethod
    def none(cls, coverage: SourceCoverage) -> CoveredPapers:
        """Build an empty result carrying a zero-coverage explanation.

        Parameters
        ----------
        coverage : SourceCoverage
            A record whose status is in :data:`ZERO_COVERAGE_STATUSES`, or
            any record with ``retrieved == 0``.

        Returns
        -------
        CoveredPapers
            Empty result.
        """
        return cls(papers=(), coverage=coverage)

    def with_source(self, source: str) -> CoveredPapers:
        """Return a copy whose coverage is labelled with *source*.

        Used by the base class so that the source identifier on the ledger is
        always the connector's registered name, not whatever string the
        connector happened to pass.

        Parameters
        ----------
        source : str
            Source identifier to stamp.

        Returns
        -------
        CoveredPapers
            This instance when the label already matches, otherwise a copy.
        """
        if self.coverage.source == source:
            return self
        return CoveredPapers(
            papers=self.papers,
            coverage=dataclasses.replace(self.coverage, source=source),
        )
