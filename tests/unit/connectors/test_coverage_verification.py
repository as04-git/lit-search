"""Tests that a connector cannot claim COMPLETE while lying.

``test_coverage_contract.py`` proves a connector cannot stay *silent*.  This
file proves it cannot *speak falsely*, which is the failure that actually
matters: a bare ``[]`` at least looks empty, while a false ``COMPLETE`` looks
like an answer.

Every test in :class:`TestTheFourAttacks` reproduces, connector for connector,
one of the four builds that an adversarial review got past every layer of the
contract on 2026-08-30:

===================================================  ==========  ============
Attack                                               Then        Now
===================================================  ==========  ============
swallow its own exception mid-pagination             complete    partial
feed its own paper count back as reported_total      complete    partial
drop 97 of 100 records it received                   complete    partial
stop silently at the caller's max_papers             complete    capped
===================================================  ==========  ============

Each attack is written to satisfy the enforcement layers deliberately: correct
return annotation, no override of ``search``, a real
:class:`~lit_search.core.coverage.CoveredPapers`, and a ``retrieved`` count
that agrees with the papers returned.  What defeats them is evidence collected
by base-class code that the connector did not author.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import requests

from lit_search.connectors.search_base import SearchConnectorBase
from lit_search.core.coverage import CoverageStatus, CoveredPapers, SourceCoverage
from lit_search.core.paper import Database, Paper
from lit_search.core.query import Query
from lit_search.core.retrieval_audit import (
    RetrievalAudit,
    active_audit,
    auditing,
    verify_coverage_claim,
)
from lit_search.query.builder import QueryBuilder, QueryValidationResult
from lit_search.runners.search_runner import SearchRunner

_SEARCH_URL = "https://example.invalid/search"


def _paper(index: int) -> Paper:
    """Build a distinct paper.

    Parameters
    ----------
    index : int
        Distinguishes the record.

    Returns
    -------
    Paper
        A minimal paper.
    """
    return Paper(
        title=f"Paper {index}",
        abstract="An abstract.",
        authors=[],
        source=None,
        publication_date=None,
        doi=f"10.1000/{index}",
    )


class _AttackBase(SearchConnectorBase):
    """Shared scaffolding for a connector built to claim a false COMPLETE.

    Everything here is contract-conforming on purpose.  The lie lives in the
    subclass's ``_fetch_papers`` body.
    """

    def __init__(self) -> None:
        """Create the connector with an always-valid query builder."""
        super().__init__()
        self._builder = MagicMock()
        self._builder.validate_query.return_value = QueryValidationResult(
            is_valid=True, error_message=None
        )

    @property
    def name(self) -> str:
        """Connector name.

        Returns
        -------
        str
            ``"liar"``.
        """
        return "liar"

    @property
    def query_builder(self) -> QueryBuilder:
        """Query builder.

        Returns
        -------
        QueryBuilder
            The permissive stub.
        """
        return self._builder

    @property
    def min_request_interval(self) -> float:
        """Rate limit.

        Returns
        -------
        float
            ``0.0``.
        """
        return 0.0


class _SwallowsItsOwnException(_AttackBase):
    """Attack 1: catch a mid-pagination request failure, then claim COMPLETE.

    Page one succeeds and yields two records; page two raises.  The connector
    catches the exception, keeps what it has, and reports the run as complete
    against the total it read from page one.
    """

    def _fetch_papers(
        self,
        query: Query,
        max_papers: int | None,
        progress_callback: Callable[[int, int | None], None] | None,
        since: datetime.date | None = None,
        until: datetime.date | None = None,
    ) -> CoveredPapers:
        """Paginate, swallow the second page's failure, claim COMPLETE.

        Parameters
        ----------
        query : Query
            Ignored.
        max_papers : int | None
            Ignored.
        progress_callback : Callable[[int, int | None], None] | None
            Fed the honest counts from page one.
        since : datetime.date | None
            Ignored.
        until : datetime.date | None
            Ignored.

        Returns
        -------
        CoveredPapers
            Two papers, claimed COMPLETE.
        """
        papers: list[Paper] = []
        for page in range(2):
            try:
                self._get(_SEARCH_URL, {"page": page})
            except requests.RequestException:
                # The donor-tree idiom this project exists to remove.
                break
            papers.extend([_paper(page * 2), _paper(page * 2 + 1)])
            if progress_callback is not None:
                progress_callback(len(papers), 2)
        return CoveredPapers(
            papers=tuple(papers),
            coverage=SourceCoverage.complete(self.name, len(papers), reported_total=len(papers)),
        )


class _InventsItsOwnTotal(_AttackBase):
    """Attack 2: pass its own paper count to ``from_pagination`` as the total.

    ``from_pagination`` compares ``retrieved`` against ``reported_total`` and,
    finding them equal, returns COMPLETE.  It has no way to know the total was
    never read from a response.
    """

    def _fetch_papers(
        self,
        query: Query,
        max_papers: int | None,
        progress_callback: Callable[[int, int | None], None] | None,
        since: datetime.date | None = None,
        until: datetime.date | None = None,
    ) -> CoveredPapers:
        """Return one paper and claim the source said there was exactly one.

        Parameters
        ----------
        query : Query
            Ignored.
        max_papers : int | None
            Ignored.
        progress_callback : Callable[[int, int | None], None] | None
            Never called: the connector has no observation to report.
        since : datetime.date | None
            Ignored.
        until : datetime.date | None
            Ignored.

        Returns
        -------
        CoveredPapers
            One paper, claimed COMPLETE.
        """
        papers = (_paper(0),)
        return CoveredPapers(
            papers=papers,
            coverage=SourceCoverage.from_pagination(
                self.name,
                retrieved=len(papers),
                reported_total=len(papers),
            ),
        )


class _DropsMostRecords(_AttackBase):
    """Attack 3: receive 100 records, return 3, and call the run complete.

    ``CoveredPapers.__post_init__`` only checks ``retrieved`` against the
    papers it was handed, and a connector that drops records *before* counting
    satisfies that trivially.
    """

    def _fetch_papers(
        self,
        query: Query,
        max_papers: int | None,
        progress_callback: Callable[[int, int | None], None] | None,
        since: datetime.date | None = None,
        until: datetime.date | None = None,
    ) -> CoveredPapers:
        """See 100 records, keep 3, claim COMPLETE.

        Parameters
        ----------
        query : Query
            Ignored.
        max_papers : int | None
            Ignored.
        progress_callback : Callable[[int, int | None], None] | None
            Told the truth about the 100 records seen.
        since : datetime.date | None
            Ignored.
        until : datetime.date | None
            Ignored.

        Returns
        -------
        CoveredPapers
            Three papers, claimed COMPLETE.
        """
        if progress_callback is not None:
            progress_callback(100, 100)
        kept = tuple(_paper(index) for index in range(3))
        return CoveredPapers(
            papers=kept,
            coverage=SourceCoverage.complete(self.name, len(kept), reported_total=len(kept)),
        )


class _TruncatesAtTheCap(_AttackBase):
    """Attack 4: stop at the caller's ``max_papers`` and claim COMPLETE.

    The connector never mentions the cap; it simply reports the count it
    stopped at as the whole result set.
    """

    def _fetch_papers(
        self,
        query: Query,
        max_papers: int | None,
        progress_callback: Callable[[int, int | None], None] | None,
        since: datetime.date | None = None,
        until: datetime.date | None = None,
    ) -> CoveredPapers:
        """Fill exactly to the cap and claim COMPLETE.

        Parameters
        ----------
        query : Query
            Ignored.
        max_papers : int | None
            The cap the connector silently stops at.
        progress_callback : Callable[[int, int | None], None] | None
            Fed counts consistent with the lie.
        since : datetime.date | None
            Ignored.
        until : datetime.date | None
            Ignored.

        Returns
        -------
        CoveredPapers
            ``max_papers`` papers, claimed COMPLETE.
        """
        limit = max_papers if max_papers is not None else 5
        kept = tuple(_paper(index) for index in range(limit))
        if progress_callback is not None:
            progress_callback(len(kept), len(kept))
        return CoveredPapers(
            papers=kept,
            coverage=SourceCoverage.complete(self.name, len(kept), reported_total=len(kept)),
        )


class TestTheFourAttacks:
    """Each build that got past the contract on 2026-08-30 now fails."""

    def test_swallowed_exception_cannot_claim_complete(self) -> None:
        """The base class saw the failed request before the connector hid it."""
        responses = [MagicMock(), requests.ConnectionError("page 2 refused")]

        def _fake_request(*_args, **_kwargs):
            outcome = responses.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        connector = _SwallowsItsOwnException()
        with patch.object(connector, "_request_with_retry_uncounted", side_effect=_fake_request):
            result = connector.search(MagicMock())

        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 2
        assert "request(s) failed" in (result.coverage.detail or "")
        assert "ConnectionError" in (result.coverage.detail or "")

    def test_self_reported_total_cannot_claim_complete(self) -> None:
        """A total the connector never read from a response corroborates nothing."""
        result = _InventsItsOwnTotal().search(MagicMock())

        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 1
        assert "no such total was ever observed" in (result.coverage.detail or "")

    def test_dropped_records_cannot_claim_complete(self) -> None:
        """Records seen and not returned are a coverage loss, not a complete result."""
        result = _DropsMostRecords().search(MagicMock())

        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 3
        assert "seeing 100 record(s) but returned 3" in (result.coverage.detail or "")

    def test_truncation_at_the_cap_cannot_claim_complete(self) -> None:
        """The cap is the caller's input, so the connector cannot talk it away."""
        result = _TruncatesAtTheCap().search(MagicMock(), max_papers=5)

        assert result.status is CoverageStatus.CAPPED
        assert result.coverage.cap == 5
        assert len(result.papers) == 5
        assert "cap of 5" in (result.coverage.detail or "")


class TestCompleteIsNotConstructibleUnqualified:
    """``COMPLETE`` needs the source's own total, and needs to have reached it."""

    def test_complete_requires_a_reported_total(self) -> None:
        """A connector that never asked how many records existed cannot claim all."""
        with pytest.raises(ValueError, match="COMPLETE requires 'reported_total'"):
            SourceCoverage(source="x", status=CoverageStatus.COMPLETE, retrieved=3)

    def test_complete_requires_reaching_the_reported_total(self) -> None:
        """Retrieving fewer records than the source reported is PARTIAL."""
        with pytest.raises(ValueError, match="fewer than the 10 records"):
            SourceCoverage(
                source="x",
                status=CoverageStatus.COMPLETE,
                retrieved=3,
                reported_total=10,
            )

    def test_constructor_no_longer_defaults_the_total(self) -> None:
        """The looser signature is gone, so eleven ports cannot be written to it."""
        with pytest.raises(TypeError, match="reported_total"):
            SourceCoverage.complete("x", 3)  # type: ignore[call-arg]


class TestFromPaginationKeepsItsShortfall:
    """A shortfall note must not vanish in the branch claiming nothing is missing."""

    def test_unparsed_records_force_partial(self) -> None:
        """The detail used to be threaded into every branch except COMPLETE."""
        coverage = SourceCoverage.from_pagination(
            "epmc",
            retrieved=8,
            reported_total=8,
            shortfall_detail="2 of 10 records could not be parsed into papers",
        )
        assert coverage.status is CoverageStatus.PARTIAL
        assert "could not be parsed" in (coverage.detail or "")

    def test_a_clean_run_still_reaches_complete(self) -> None:
        """The fix must not make COMPLETE unreachable."""
        coverage = SourceCoverage.from_pagination("epmc", retrieved=8, reported_total=8)
        assert coverage.status is CoverageStatus.COMPLETE


class TestVerificationOnlyEverDowngrades:
    """Evidence can contradict completeness; it can never establish it."""

    def test_a_partial_claim_is_left_alone(self) -> None:
        """Verification never upgrades a status the connector chose."""
        claim = CoveredPapers(
            papers=(_paper(0),),
            coverage=SourceCoverage.partial("s", 1, detail="stated shortfall"),
        )
        assert verify_coverage_claim(claim, max_papers=None, audit=None) is claim

    def test_a_corroborated_complete_survives(self) -> None:
        """An honest connector is not punished for the liars."""
        audit = RetrievalAudit(source="s")
        audit.note_progress(4, 4)
        claim = CoveredPapers(
            papers=tuple(_paper(index) for index in range(4)),
            coverage=SourceCoverage.complete("s", 4, reported_total=4),
        )
        assert verify_coverage_claim(claim, max_papers=100, audit=audit) is claim

    def test_no_audit_still_catches_the_cap(self) -> None:
        """The cap check needs no audit: max_papers is the caller's own input."""
        claim = CoveredPapers(
            papers=tuple(_paper(index) for index in range(4)),
            coverage=SourceCoverage.complete("s", 4, reported_total=4),
        )
        verified = verify_coverage_claim(claim, max_papers=4, audit=None)
        assert verified.status is CoverageStatus.CAPPED


class TestTheRunnerRechecksTheCap:
    """A connector object replaced wholesale never runs the final ``search``."""

    def test_a_monkeypatched_connector_cannot_claim_complete_at_the_cap(self) -> None:
        """The runner re-checks the cap for the same reason it re-checks the type."""
        runner = SearchRunner(
            query="[ml]",
            databases=["arxiv"],
            max_papers_per_database=3,
            enrichment_databases=[],
        )
        searcher = MagicMock()
        searcher.name = Database.ARXIV
        searcher.search.return_value = CoveredPapers(
            papers=tuple(_paper(index) for index in range(3)),
            coverage=SourceCoverage.complete("arxiv", 3, reported_total=3),
        )
        runner._searchers = [searcher]

        result = runner.run(show_progress=False)

        entry = result.coverage.get("arxiv")
        assert entry is not None
        assert entry.status is CoverageStatus.CAPPED
        assert entry.cap == 3

    def test_a_result_below_the_cap_is_left_complete(self) -> None:
        """The backstop must not degrade an honest under-cap result."""
        runner = SearchRunner(
            query="[ml]",
            databases=["arxiv"],
            max_papers_per_database=10,
            enrichment_databases=[],
        )
        searcher = MagicMock()
        searcher.name = Database.ARXIV
        searcher.search.return_value = CoveredPapers(
            papers=tuple(_paper(index) for index in range(3)),
            coverage=SourceCoverage.complete("arxiv", 3, reported_total=3),
        )
        runner._searchers = [searcher]

        result = runner.run(show_progress=False)

        entry = result.coverage.get("arxiv")
        assert entry is not None
        assert entry.status is CoverageStatus.COMPLETE


class TestTheAuditIsScopedAndBaseClassOwned:
    """The evidence channel belongs to the base class, not to the connector."""

    def test_no_audit_outside_a_search(self) -> None:
        """A DOI enrichment lookup is not a search and has no claim to check."""
        assert active_audit() is None

    def test_the_scope_is_restored_after_the_search(self) -> None:
        """Nested scopes must not leak into the caller."""
        with auditing("outer") as outer:
            assert active_audit() is outer
            with auditing("inner") as inner:
                assert active_audit() is inner
            assert active_audit() is outer
        assert active_audit() is None

    def test_the_connector_cannot_skip_the_progress_channel(self) -> None:
        """The callback handed to ``_fetch_papers`` is never the caller's None."""
        seen: list[object] = []

        class _Recorder(_AttackBase):
            def _fetch_papers(
                self,
                query: Query,
                max_papers: int | None,
                progress_callback: Callable[[int, int | None], None] | None,
                since: datetime.date | None = None,
                until: datetime.date | None = None,
            ) -> CoveredPapers:
                """Record what the base class handed over.

                Parameters
                ----------
                query : Query
                    Ignored.
                max_papers : int | None
                    Ignored.
                progress_callback : Callable[[int, int | None], None] | None
                    The value under test.
                since : datetime.date | None
                    Ignored.
                until : datetime.date | None
                    Ignored.

                Returns
                -------
                CoveredPapers
                    An empty, honest result.
                """
                seen.append(progress_callback)
                return CoveredPapers.none(
                    SourceCoverage.partial(self.name, 0, detail="not the point of this test")
                )

        _Recorder().search(MagicMock(), progress_callback=None)
        assert len(seen) == 1
        assert callable(seen[0])

    def test_the_callers_callback_still_receives_progress(self) -> None:
        """Recording the observations must not steal them from the caller."""
        received: list[tuple[int, int | None]] = []
        _DropsMostRecords().search(
            MagicMock(), progress_callback=lambda current, total: received.append((current, total))
        )
        assert received == [(100, 100)]
