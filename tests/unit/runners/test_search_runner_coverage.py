"""The search runner's ledger assembly, one failure shape at a time.

``tests/unit/connectors/test_coverage_contract.py`` checks that the ledger
exists and that a skipped database reaches it.  This module checks the paths
that produce a ledger entry when something goes wrong *inside* the runner --
the ones most likely to be quietly dropped, because each of them is an
exception handler and an exception handler that forgets is invisible.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from lit_search.core.coverage import CoverageStatus, CoveredPapers, SourceCoverage
from lit_search.core.paper import Database
from lit_search.exceptions import MissingApiKeyError, UnsupportedQueryError
from lit_search.runners.search_runner import DEFAULT_SEARCH_DATABASES, SearchRunner


def _searcher(name: Database, **behaviour: object) -> MagicMock:
    """Build a stub connector with the given ``search`` behaviour.

    Parameters
    ----------
    name : Database
        Name the runner will key the ledger on.
    **behaviour : object
        Passed straight to ``search``: ``return_value`` or ``side_effect``.

    Returns
    -------
    MagicMock
        Stub connector.
    """
    stub = MagicMock()
    stub.name = name
    stub.search.configure_mock(**behaviour)
    return stub


def _complete(source: str, count: int = 0) -> CoveredPapers:
    """Build an empty, honestly-complete result for *source*.

    Parameters
    ----------
    source : str
        Source identifier.
    count : int
        Records retrieved; only ``0`` is used here.

    Returns
    -------
    CoveredPapers
        A COMPLETE zero-record result.
    """
    return CoveredPapers.none(SourceCoverage.complete(source, count, reported_total=count))


class TestEveryFailureShapeReachesTheLedger:
    """No searcher leaves the runner without a coverage entry."""

    def test_a_raised_missing_credential_names_the_credential(self) -> None:
        """A key checked at search time, not construction time, still names it.

        ``_build_searchers`` catches the credential error for connectors that
        validate in ``__init__``.  One that validates on the first request
        raises from ``search`` instead, and must not land in the generic
        FAILED bucket: FAILED reads as an upstream outage, and the fix here is
        the user supplying a key.
        """
        runner = SearchRunner(query="[ml]", databases=["arxiv", "scopus"], enrichment_databases=[])
        runner._searchers = [
            _searcher(Database.SCOPUS, side_effect=MissingApiKeyError("scopus_api_key"))
        ]

        entry = runner.run(show_progress=False).coverage.get("scopus")

        assert entry is not None
        assert entry.status is CoverageStatus.SKIPPED_NO_CREDENTIAL
        assert entry.missing_credential == "scopus_api_key"

    def test_an_unsupported_query_is_not_a_failure_and_not_a_zero(self) -> None:
        """QUERY_UNSUPPORTED is its own status and stays out of failed_databases."""
        runner = SearchRunner(query="[ml]", databases=["arxiv"], enrichment_databases=[])
        runner._searchers = [
            _searcher(Database.ARXIV, side_effect=UnsupportedQueryError("no phrase support"))
        ]

        result = runner.run(show_progress=False)

        entry = result.coverage.get("arxiv")
        assert entry is not None
        assert entry.status is CoverageStatus.QUERY_UNSUPPORTED
        assert "no phrase support" in (entry.detail or "")
        assert result.failed_databases == []

    def test_a_thrown_error_carries_its_type_into_the_ledger(self) -> None:
        """The exception class survives, so a reader can tell what to retry."""
        runner = SearchRunner(query="[ml]", databases=["arxiv"], enrichment_databases=[])
        runner._searchers = [_searcher(Database.ARXIV, side_effect=RuntimeError("socket closed"))]

        entry = runner.run(show_progress=False).coverage.get("arxiv")

        assert entry is not None
        assert entry.status is CoverageStatus.FAILED
        assert entry.error_type == "RuntimeError"
        assert "socket closed" in (entry.error or "")

    def test_a_searcher_with_no_entry_at_all_is_failed_not_absent(self) -> None:
        """A runner bug is recorded as a failure, never as a silent omission.

        ``_build_ledger`` is asked for a database it collected nothing for.
        Leaving the entry out would shrink the ledger to the databases that
        worked, which is the exact shape this project exists to refuse.
        """
        runner = SearchRunner(query="[ml]", databases=["arxiv"], enrichment_databases=[])
        runner._searchers = [_searcher(Database.ARXIV, return_value=_complete("arxiv"))]

        ledger = runner._build_ledger({})

        entry = ledger.get("arxiv")
        assert entry is not None
        assert entry.status is CoverageStatus.FAILED
        assert entry.error_type == "MissingCoverage"

    def test_a_result_that_is_neither_papers_nor_error_is_failed(self) -> None:
        """``(None, None)`` from the executor is unexplained, so it is FAILED."""
        entry = SearchRunner._coverage_for_error("arxiv", None)

        assert entry.status is CoverageStatus.FAILED
        assert entry.error_type == "NoResult"


class TestTheLedgerReadsTheSameEveryRun:
    """Completion order must not change what the ledger looks like."""

    def test_entries_follow_configuration_order_then_skipped(self) -> None:
        """Parallel completion order does not reorder the report."""
        runner = SearchRunner(
            query="[ml]",
            databases=["arxiv", "pubmed", "ieee"],
            num_workers=2,
            enrichment_databases=[],
        )
        runner._searchers = [
            _searcher(Database.ARXIV, return_value=_complete("arxiv")),
            _searcher(Database.PUBMED, return_value=_complete("pubmed")),
        ]

        result = runner.run(show_progress=False)

        assert result.coverage.sources == ("arxiv", "pubmed", "ieee")

    def test_the_headline_names_databases_that_were_never_reached(self) -> None:
        """A skipped database is in the headline, not only in the detail.

        ``databases`` records the connectors that were built, so a headline
        driven by it would read "across arxiv" for a three-database request
        and quietly narrow the question the reader thinks was answered.
        """
        runner = SearchRunner(query="[ml]", databases=["arxiv", "ieee"], enrichment_databases=[])
        runner._searchers = [_searcher(Database.ARXIV, return_value=_complete("arxiv"))]

        headline = runner.run(show_progress=False).headline()

        assert "arxiv" in headline
        assert "ieee" in headline


class TestTheVerdictIsAnnouncedOnTheLogger:
    """A caller who never renders the object still gets told."""

    def test_an_incomplete_set_logs_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """WARNING, because the set cannot be read as a finished answer."""
        runner = SearchRunner(query="[ml]", databases=["arxiv", "ieee"], enrichment_databases=[])
        runner._searchers = [_searcher(Database.ARXIV, return_value=_complete("arxiv"))]

        with caplog.at_level(logging.WARNING, logger="lit_search.runners.search_runner"):
            runner.run(show_progress=False)

        assert any("INCOMPLETE" in record.message for record in caplog.records)

    def test_a_complete_set_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        """A ledger that always warns is a ledger nobody reads."""
        runner = SearchRunner(query="[ml]", databases=["arxiv"], enrichment_databases=[])
        runner._searchers = [_searcher(Database.ARXIV, return_value=_complete("arxiv"))]

        with caplog.at_level(logging.INFO, logger="lit_search.runners.search_runner"):
            result = runner.run(show_progress=False)

        assert result.is_complete
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_default_policy_exclusions_do_not_fail_a_fulfilled_plan() -> None:
    """All selected sources can finish while registry-wide coverage stays bounded."""
    runner = SearchRunner(query="[ml]", enrichment_databases=[])
    runner._searchers = [
        _searcher(Database(name), return_value=_complete(name)) for name in DEFAULT_SEARCH_DATABASES
    ]

    result = runner.run(show_progress=False)

    assert result.execution_is_complete
    assert not result.is_complete
    assert result.coverage.scope is not None
    assert result.coverage.scope.profile == "default"
    assert result.databases == list(DEFAULT_SEARCH_DATABASES)
    assert any(
        entry.status is CoverageStatus.SKIPPED_BY_POLICY for entry in result.coverage.entries
    )
