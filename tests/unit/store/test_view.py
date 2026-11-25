"""Tests for rendering a stored handle with its coverage.

A handle's ledger is a plain dictionary on disk, and the core renderer refuses
anything that is not a real ledger -- deliberately, so a bare string cannot
pass as a claim.  This view is the one honest bridge between the two, and the
tests are about what it refuses as much as what it converts.
"""

from __future__ import annotations

from lit_search.core.coverage import CoverageLedger, CoverageStatus, SourceCoverage
from lit_search.core.rendering import coverage_view, render_result
from lit_search.store.lineage import Operation, OperationName
from lit_search.store.record import HandleRecord
from lit_search.store.view import handle_view


def _record(coverage: object) -> HandleRecord:
    """Build a handle carrying an arbitrary coverage payload.

    Parameters
    ----------
    coverage : object
        Payload to attach.

    Returns
    -------
    HandleRecord
        Handle.
    """
    return HandleRecord(
        operation=Operation(name=OperationName.SEARCH, parameters={"query": "[q]"}),
        coverage=coverage,
    )


class TestRehydration:
    """A serialized ledger becomes a real one, so the renderer can read it."""

    def test_a_ledger_dictionary_is_read_back(self) -> None:
        """Otherwise every stored handle renders as "coverage is a dict"."""
        ledger = CoverageLedger.from_sources(
            [SourceCoverage.complete("arxiv", 2, reported_total=2)]
        )
        view = handle_view("h_" + "0" * 32, _record(ledger.to_dict()))
        assert view.coverage is not None
        assert view.coverage.status is CoverageStatus.COMPLETE
        assert coverage_view(view).is_complete is True

    def test_an_incomplete_ledger_renders_its_shortfalls(self) -> None:
        """D18 asks for the failed list in every rendering, handles included."""
        ledger = CoverageLedger.from_sources(
            [
                SourceCoverage.complete("arxiv", 1, reported_total=1),
                SourceCoverage.failed("wos", error="502 from upstream"),
            ]
        )
        view = handle_view("h_" + "1" * 32, _record(ledger.to_dict()))
        rendered = render_result(view, headline=view.headline(), limit=5)
        assert "INCOMPLETE" in rendered
        assert "wos" in rendered


class TestRefusals:
    """An unreadable payload is stated, never believed."""

    def test_a_bare_string_is_not_a_claim(self) -> None:
        """``'complete'`` round-trips through the strict reader; it must not pass."""
        view = handle_view("h_" + "2" * 32, _record("complete"))
        assert view.coverage is None
        assert "not a serialized ledger" in (view.coverage_unavailable_reason or "")
        assert coverage_view(view).is_complete is False

    def test_an_unreadable_mapping_is_reported(self) -> None:
        """A mapping that is not a ledger is a different fact from no ledger."""
        view = handle_view("h_" + "3" * 32, _record({"sources": [{"nonsense": True}]}))
        assert view.coverage is None
        assert "could not be read as a ledger" in (view.coverage_unavailable_reason or "")

    def test_no_ledger_is_explained_rather_than_skipped(self) -> None:
        """ "Nothing recorded" must not read as "nothing missed"."""
        view = handle_view("h_" + "4" * 32, _record(None))
        assert view.coverage is None
        assert "not the same as nothing having been missed" in (
            view.coverage_unavailable_reason or ""
        )
        assert coverage_view(view).status is CoverageStatus.UNKNOWN


class TestHeadline:
    """The one line a listing shows."""

    def test_it_names_the_operation_and_the_order_state(self) -> None:
        """Order state is what tells a reader whether a sequence means anything."""
        view = handle_view("h_" + "5" * 32, _record(None))
        headline = view.headline()
        assert "search" in headline
        assert "order unordered" in headline
