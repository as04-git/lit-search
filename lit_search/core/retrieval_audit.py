"""Evidence about a retrieval that the connector did not author, and what it proves.

The coverage contract in :mod:`lit_search.core.coverage` makes silence
inexpressible: a connector cannot return papers without also returning a
status.  It does **not** make that status *true*.  Four connectors were built
against the contract that satisfy every enforcement layer and lie:

=========================================================  ==================
Attack                                                     Claimed
=========================================================  ==================
swallow its own exception mid-pagination                   ``COMPLETE``
feed its own paper count back as ``reported_total``        ``COMPLETE``
drop 97 of 100 records it received                         ``COMPLETE``
stop silently at the caller's ``max_papers``               ``COMPLETE``
=========================================================  ==================

Every one of those is a statement about the *world* -- how many records the
source held, how many arrived, whether a request failed -- made by the only
party with an interest in the answer being "all of them".

This module supplies the second opinion.  :class:`RetrievalAudit` is written
**by base-class code only**, at points a connector must pass through and
cannot route around:

``ConnectorBase._request_with_retry``
    every HTTP attempt, and every failure that escaped it -- recorded *before*
    the connector's own ``except`` clause can swallow it.
``SearchConnectorBase.search``
    the progress callback handed to ``_fetch_papers`` is always the base
    class's own recorder, so the two numbers a connector reports for the
    progress bar -- how many records it has seen, and what total the source
    published -- are also observations the base class holds.
the caller
    ``max_papers`` is an input, not a connector output.

:func:`verify_coverage_claim` then compares a ``COMPLETE`` claim against that
evidence and downgrades it when the evidence does not support it.  It never
upgrades: evidence can contradict a completeness claim, never establish one.

What this cannot do, stated plainly
-----------------------------------

A connector that lies **consistently across every channel** -- reports a
fabricated total through the progress callback *and* repeats it as
``reported_total`` -- is not caught, and no check short of the base class
parsing the response body itself would catch it.  What changes is that the
honest path and the lying path are no longer the same keystrokes.  A
mechanical port that simply forgets to instrument now fails closed, at
``PARTIAL``, instead of reporting a confident ``COMPLETE``.

The request evidence covers only what goes through
``ConnectorBase._request_with_retry``, which is every call made via ``_get``
and ``_post``.  **A connector that reaches for ``requests.get`` directly
escapes it** -- so a port must use the base class's helpers, which it has to
anyway to be rate limited at all.  ``WebScrapingConnector`` uses
``curl_cffi`` directly and is deliberately outside this: it is an enrichment
path, not a search connector, and it makes no coverage claim.

See ``R7`` and ``R10`` in the project specification.
"""

from __future__ import annotations

import contextvars
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from lit_search.core.coverage import CoverageStatus, CoveredPapers, SourceCoverage, severity

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RequestFailure:
    """One HTTP request that failed after the base class exhausted its retries.

    Recorded at the point the exception leaves
    :meth:`~lit_search.connectors.connector_base.ConnectorBase._request_with_retry`,
    which is *upstream* of any ``except`` clause a connector writes.  A
    connector that catches this and continues paginating has still lost
    records, and the audit remembers that even when the connector does not
    mention it.

    Parameters
    ----------
    error_type : str
        Exception class name.
    message : str
        Exception message.
    """

    error_type: str
    message: str
    status_code: int | None = None

    def describe(self) -> str:
        """Return a one-line human-readable summary.

        Returns
        -------
        str
            Summary line, for example ``"HTTPError: 503 Server Error"``.
        """
        return f"{self.error_type}: {self.message}" if self.message else self.error_type


@dataclass
class RetrievalAudit:
    """Base-class observations about one connector's retrieval.

    Mutable by design: it accumulates during a single ``search`` call and is
    read once at the end.  It is scoped to the calling thread by
    :func:`auditing`, so concurrent connector runs in the search runner's
    thread pool never see each other's evidence.

    Attributes
    ----------
    source : str
        Connector name this audit belongs to.
    requests_attempted : int
        Number of HTTP requests the base class issued on this connector's
        behalf during the retrieval.
    requests_failed : list[RequestFailure]
        Requests that raised out of the base class's retry loop.
    observed_items : int
        Largest ``items_processed`` value the connector reported through the
        progress channel.  This is the connector's own count of records it
        *saw*, which is a different number from how many papers it returned.
    observed_totals : list[int]
        Distinct non-``None`` totals the connector reported through the
        progress channel -- what the source itself said existed.
    """

    source: str
    requests_attempted: int = 0
    requests_failed: list[RequestFailure] = field(default_factory=list)
    observed_items: int = 0
    observed_totals: list[int] = field(default_factory=list)

    def note_request_attempt(self) -> None:
        """Record that an HTTP request is about to be issued."""
        self.requests_attempted += 1

    def note_request_failure(self, exc: BaseException) -> None:
        """Record a request that failed after retries were exhausted.

        Parameters
        ----------
        exc : BaseException
            Exception leaving the base class's retry loop.
        """
        response = getattr(exc, "response", None)
        raw_status = getattr(response, "status_code", None)
        status_code = raw_status if isinstance(raw_status, int) else None
        self.requests_failed.append(
            RequestFailure(
                error_type=type(exc).__name__,
                message=str(exc),
                status_code=status_code,
            )
        )

    def note_progress(self, items_processed: int, total: int | None) -> None:
        """Record one progress observation made by the connector.

        Parameters
        ----------
        items_processed : int
            Candidate records the connector has seen so far, parsed or not.
        total : int | None
            Total the source reported, when it reports one.
        """
        if isinstance(items_processed, int) and items_processed > self.observed_items:
            self.observed_items = items_processed
        if (
            isinstance(total, int)
            and not isinstance(total, bool)
            and total not in self.observed_totals
        ):
            self.observed_totals.append(total)

    def describe(self) -> str:
        """Return a one-line human-readable summary of the evidence.

        Returns
        -------
        str
            Summary line.
        """
        return (
            f"{self.source}: {self.requests_attempted} request(s), "
            f"{len(self.requests_failed)} failed, {self.observed_items} item(s) seen, "
            f"totals observed {self.observed_totals or 'none'}"
        )


#: Audit for the retrieval running in this thread, if any.  A context variable
#: rather than an argument because the evidence is collected at
#: ``ConnectorBase._request_with_retry``, which is several frames below
#: ``search`` and shared with connectors that never search at all.
_ACTIVE_AUDIT: contextvars.ContextVar[RetrievalAudit | None] = contextvars.ContextVar(
    "lit_search_retrieval_audit", default=None
)


def active_audit() -> RetrievalAudit | None:
    """Return the audit for the retrieval running in this thread.

    Returns
    -------
    RetrievalAudit | None
        The active audit, or ``None`` when no retrieval is in progress -- for
        example a DOI enrichment lookup, which is not a search and has no
        coverage claim to check.
    """
    return _ACTIVE_AUDIT.get()


@contextmanager
def auditing(source: str) -> Iterator[RetrievalAudit]:
    """Open an audit scope for one retrieval.

    Parameters
    ----------
    source : str
        Connector name.

    Yields
    ------
    RetrievalAudit
        The audit that base-class code will write into for the duration.
    """
    audit = RetrievalAudit(source=source)
    token = _ACTIVE_AUDIT.set(audit)
    try:
        yield audit
    finally:
        _ACTIVE_AUDIT.reset(token)


def record_request_attempt() -> None:
    """Note an HTTP attempt on the active audit, if there is one."""
    audit = _ACTIVE_AUDIT.get()
    if audit is not None:
        audit.note_request_attempt()


def record_request_failure(exc: BaseException) -> None:
    """Note a failed HTTP request on the active audit, if there is one.

    Parameters
    ----------
    exc : BaseException
        Exception leaving the base class's retry loop.
    """
    audit = _ACTIVE_AUDIT.get()
    if audit is not None:
        audit.note_request_failure(exc)


# ----------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Contradiction:
    """One reason the evidence does not support a ``COMPLETE`` claim."""

    status: CoverageStatus
    detail: str


def _cap_contradiction(
    *, retrieved: int, reported_total: int, max_papers: int | None
) -> _Contradiction | None:
    """Detect truncation at the caller's cap without trusting the connector.

    ``max_papers`` is an input the caller chose, and ``retrieved`` is counted
    from the returned tuple, so neither is a connector self-report.  When a
    retrieval stops at or above the cap, the cap is a *candidate* reason it
    stopped, and ``COMPLETE`` survives only when the source itself said fewer
    records existed than the cap allowed -- in which case the cap cannot have
    been what ended the run.

    ``reported_total == max_papers == retrieved`` is deliberately treated as a
    contradiction.  "Exactly exhausted the source" and "stopped dead on the
    cap" are indistinguishable from here, and ``CAPPED`` loses nothing: it
    still records the count and the limit.

    Parameters
    ----------
    retrieved : int
        Papers actually returned.
    reported_total : int
        Total the connector says the source reported.
    max_papers : int | None
        Cap the caller imposed.

    Returns
    -------
    _Contradiction | None
        A ``CAPPED`` contradiction, or ``None``.
    """
    if max_papers is None or retrieved < max_papers or reported_total < max_papers:
        return None
    return _Contradiction(
        status=CoverageStatus.CAPPED,
        detail=(
            f"retrieval returned {retrieved} record(s) at the caller's cap of {max_papers} "
            f"while the source reported {reported_total}, so the cap is indistinguishable "
            "from the stopping reason and completeness cannot be claimed"
        ),
    )


def _audit_contradictions(
    *, retrieved: int, reported_total: int, audit: RetrievalAudit
) -> list[_Contradiction]:
    """Collect every way the recorded evidence contradicts ``COMPLETE``.

    Parameters
    ----------
    retrieved : int
        Papers actually returned.
    reported_total : int
        Total the connector says the source reported.
    audit : RetrievalAudit
        Evidence gathered by base-class code during the retrieval.

    Returns
    -------
    list[_Contradiction]
        Contradictions found, possibly empty.
    """
    found: list[_Contradiction] = []

    if audit.requests_failed:
        detail = "; ".join(failure.describe() for failure in audit.requests_failed)
        found.append(
            _Contradiction(
                status=CoverageStatus.PARTIAL,
                detail=(
                    f"{len(audit.requests_failed)} request(s) failed during this retrieval "
                    f"and the connector did not report it ({detail})"
                ),
            )
        )

    if audit.observed_items > retrieved:
        found.append(
            _Contradiction(
                status=CoverageStatus.PARTIAL,
                detail=(
                    f"the connector reported seeing {audit.observed_items} record(s) but "
                    f"returned {retrieved}; records seen and not returned are a coverage "
                    "loss, not a complete result"
                ),
            )
        )

    if reported_total not in audit.observed_totals:
        observed = (
            ", ".join(str(value) for value in audit.observed_totals)
            if audit.observed_totals
            else "none"
        )
        found.append(
            _Contradiction(
                status=CoverageStatus.PARTIAL,
                detail=(
                    f"COMPLETE was claimed against a source total of {reported_total}, but "
                    f"no such total was ever observed coming back from the source "
                    f"(totals observed: {observed}). A total the connector did not read "
                    "from a response cannot corroborate completeness"
                ),
            )
        )

    return found


def verify_coverage_claim(
    result: CoveredPapers,
    *,
    max_papers: int | None,
    audit: RetrievalAudit | None,
) -> CoveredPapers:
    """Downgrade a ``COMPLETE`` claim the evidence does not support.

    Only ``COMPLETE`` is checked, and the result is only ever made *worse*.
    Evidence can contradict a completeness claim; it can never establish one,
    so an already-degraded status is returned untouched.

    Parameters
    ----------
    result : CoveredPapers
        What ``_fetch_papers`` returned.
    max_papers : int | None
        Cap the caller imposed, or ``None``.
    audit : RetrievalAudit | None
        Evidence gathered during the retrieval.  ``None`` means the retrieval
        ran outside an audit scope, in which case only the cap check -- which
        needs no audit -- applies.

    Returns
    -------
    CoveredPapers
        *result* unchanged, or a copy carrying the downgraded status.
    """
    coverage = result.coverage
    if coverage.status is not CoverageStatus.COMPLETE:
        return result

    retrieved = len(result.papers)
    reported_total = coverage.reported_total
    if reported_total is None:  # pragma: no cover - forbidden by SourceCoverage
        reported_total = retrieved

    contradictions: list[_Contradiction] = []
    cap_note = _cap_contradiction(
        retrieved=retrieved, reported_total=reported_total, max_papers=max_papers
    )
    if cap_note is not None:
        contradictions.append(cap_note)
    if audit is not None:
        contradictions.extend(
            _audit_contradictions(retrieved=retrieved, reported_total=reported_total, audit=audit)
        )

    if not contradictions:
        return result

    worst_status = max((note.status for note in contradictions), key=severity)
    detail = "; ".join(note.detail for note in contradictions)
    logger.warning("[%s] refusing the connector's COMPLETE claim: %s", coverage.source, detail)

    if worst_status is CoverageStatus.CAPPED:
        downgraded = SourceCoverage.capped(
            coverage.source,
            retrieved,
            cap=max_papers if max_papers is not None else retrieved,
            reported_total=reported_total,
            detail=detail,
        )
    else:
        downgraded = SourceCoverage.partial(
            coverage.source,
            retrieved,
            detail=detail,
            reported_total=reported_total,
        )
    return CoveredPapers(papers=result.papers, coverage=downgraded)
