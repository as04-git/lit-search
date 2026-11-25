"""Tests for the coverage-first rendering layer.

The question every test here asks is the same one: **can a reader mistake an
incomplete set for a complete one?**  So the assertions are mostly about the
difference between the two renderings, not about their wording.
"""

from __future__ import annotations

import datetime
import logging
import subprocess
import sys
from typing import ClassVar

import pytest

from lit_search.core.coverage import (
    CoverageLedger,
    CoverageScope,
    CoverageStatus,
    ExecutionStatus,
    SourceCoverage,
)
from lit_search.core.paper import Paper
from lit_search.core.rendering import (
    WARN,
    CoverageView,
    coverage_banner,
    coverage_footer,
    coverage_view,
    log_coverage,
    render_coverage_report,
    render_result,
    result_repr,
    result_summary,
)
from lit_search.core.search_result import SearchResult
from lit_search.core.similar_result import SimilarResult
from lit_search.core.snowball_result import SnowballResult
from lit_search.core.source import Source, SourceType


def _paper(title: str, doi: str | None = None) -> Paper:
    """Build a minimal paper for rendering tests.

    Parameters
    ----------
    title : str
        Paper title.
    doi : str | None
        Optional DOI.

    Returns
    -------
    Paper
        Minimal paper.
    """
    return Paper(
        title=title,
        abstract="",
        authors=[],
        source=Source(title="arXiv", source_type=SourceType.REPOSITORY),
        doi=doi,
        publication_date=datetime.date(2026, 4, 13),
        found_in={"arxiv"},
    )


def _complete_ledger() -> CoverageLedger:
    """Build a ledger whose every source is complete.

    Returns
    -------
    CoverageLedger
        Complete ledger.
    """
    return CoverageLedger.from_sources(
        [
            SourceCoverage.complete("arxiv", 2, reported_total=2),
            SourceCoverage.complete("europe_pmc", 0, reported_total=0),
        ]
    )


def _incomplete_ledger() -> CoverageLedger:
    """Build a ledger with one complete and two failed-or-skipped sources.

    Returns
    -------
    CoverageLedger
        Incomplete ledger.
    """
    return CoverageLedger.from_sources(
        [
            SourceCoverage.complete("arxiv", 2, reported_total=2),
            SourceCoverage.skipped_no_credential("ieee", credential="ieee_api_key"),
            SourceCoverage.failed("wos", error="502 from upstream", error_type="HTTPError"),
        ]
    )


def _bounded_ledger() -> CoverageLedger:
    """Build a fulfilled default plan with visible policy exclusions."""
    return CoverageLedger(
        entries=(
            SourceCoverage.complete("arxiv", 0, reported_total=0),
            SourceCoverage.skipped_by_policy("ieee", detail="default profile"),
        ),
        scope=CoverageScope(
            profile="default",
            selected_sources=("arxiv",),
            registered_sources=("arxiv", "ieee"),
        ),
    )


class TestCoverageView:
    """A view normalises the three states a result object can be in."""

    def test_ledger_backed_view_reports_the_ledger_status(self) -> None:
        """A real ledger is used as-is."""
        view = CoverageView(ledger=_complete_ledger())

        assert view.available is True
        assert view.is_complete is True
        assert view.status is CoverageStatus.COMPLETE
        assert view.execution_status is ExecutionStatus.COMPLETE

    def test_absent_ledger_without_a_reason_is_refused(self) -> None:
        """An unexplained absence of coverage cannot be constructed."""
        with pytest.raises(ValueError, match="unavailable_reason"):
            CoverageView()

    def test_absent_ledger_is_unknown_and_never_complete(self) -> None:
        """'We cannot say' must not read as 'everything is fine'."""
        view = CoverageView(unavailable_reason="the runner records none")

        assert view.status is CoverageStatus.UNKNOWN
        assert view.is_complete is False
        assert "records none" in view.headline()

    def test_object_without_coverage_attribute_says_so(self) -> None:
        """A container that records no coverage is rendered, not skipped."""

        class Bare:
            papers: ClassVar[list[Paper]] = []

        view = coverage_view(Bare())

        assert view.available is False
        assert "Bare" in str(view.unavailable_reason)

    def test_coverage_that_is_not_a_ledger_is_refused_not_believed(self) -> None:
        """A bare string 'complete' must not pass as a coverage claim.

        ``HandleRecord.coverage`` is typed ``Any``, so a hand-edited payload
        can put a string there.  Treating it as a ledger would let the word
        'complete' stand in for the record.
        """

        class Liar:
            coverage = "complete"
            papers: ClassVar[list[Paper]] = []

        view = coverage_view(Liar())

        assert view.is_complete is False
        assert view.status is CoverageStatus.UNKNOWN
        assert "not a CoverageLedger" in str(view.unavailable_reason)

    def test_declared_reason_wins_over_the_generic_message(self) -> None:
        """An object may explain its own gap precisely."""

        class Explained:
            coverage_unavailable_reason = "the snowball runner records no ledger"
            papers: ClassVar[list[Paper]] = []

        assert coverage_view(Explained()).unavailable_reason == (
            "the snowball runner records no ledger"
        )

    def test_to_dict_always_carries_every_key(self) -> None:
        """An omitted key reads as 'fine', so no key is ever omitted."""
        payload = CoverageView(ledger=_incomplete_ledger()).to_dict()

        assert set(payload) == {
            "available",
            "status",
            "complete",
            "execution",
            "scope",
            "unavailable_reason",
            "shortfalls",
            "ledger",
        }
        assert payload["complete"] is False
        assert payload["ledger"] is not None

    def test_to_dict_of_an_unavailable_view_still_carries_every_key(self) -> None:
        """Nothing to report is reported, rather than left out."""
        payload = CoverageView(unavailable_reason="no ledger").to_dict()

        assert payload["available"] is False
        assert payload["ledger"] is None
        assert payload["status"] == "unknown"


class TestBanners:
    """The two renderings differ in structure, not only in adjective."""

    def test_complete_banner_carries_no_warning_marker(self) -> None:
        """A clean run must not cry wolf."""
        text = coverage_banner(CoverageView(ledger=_complete_ledger()))

        assert WARN not in text
        assert "complete" in text

    def test_incomplete_banner_marks_every_line_and_names_each_shortfall(self) -> None:
        """Every source that fell short is named in the banner itself."""
        text = coverage_banner(CoverageView(ledger=_incomplete_ledger()))

        assert "INCOMPLETE RESULT" in text
        assert "ieee (skipped_no_credential)" in text
        assert "wos (failed)" in text
        assert "arxiv" not in text

    def test_policy_exclusions_are_neutral_but_never_hidden(self) -> None:
        """A bounded scope is structurally distinct from an execution failure."""
        view = CoverageView(ledger=_bounded_ledger())

        banner = coverage_banner(view)

        assert "BOUNDED RESULT" in banner
        assert WARN not in banner
        assert "scope default: 1 of 2" in banner
        assert "ieee" in banner
        assert coverage_footer(view, paper_count=0) == ""

    def test_complete_footer_is_empty(self) -> None:
        """There is nothing to repeat under a complete set."""
        assert coverage_footer(CoverageView(ledger=_complete_ledger())) == ""

    def test_incomplete_footer_repeats_the_verdict_after_the_content(self) -> None:
        """A reader who scrolls past the banner still ends on the warning."""
        text = coverage_footer(CoverageView(ledger=_incomplete_ledger()), paper_count=5)

        assert "Do not read this set as a complete answer" in text
        assert "ieee (skipped_no_credential)" in text

    def test_empty_set_with_incomplete_coverage_is_called_out_explicitly(self) -> None:
        """Zero papers plus a shortfall is the shape most easily misread."""
        text = coverage_footer(CoverageView(ledger=_incomplete_ledger()), paper_count=0)

        assert "not evidence that no matching paper exists" in text

    def test_empty_set_with_complete_coverage_makes_no_such_claim(self) -> None:
        """A genuinely exhaustive zero is allowed to be a zero."""
        text = coverage_footer(CoverageView(ledger=_complete_ledger()), paper_count=0)

        assert text == ""

    def test_report_of_an_unavailable_view_states_the_reason(self) -> None:
        """A missing ledger is rendered rather than rendered as nothing."""
        text = render_coverage_report(CoverageView(unavailable_reason="runner records none"))

        assert "no coverage ledger" in text
        assert "runner records none" in text


class TestRenderResult:
    """Papers are never displayed without the coverage of the set they came from."""

    def test_papers_are_sandwiched_between_two_coverage_statements(self) -> None:
        """The warning appears both above and below the paper list."""
        result = SearchResult(
            query="[fm]",
            databases=["arxiv", "ieee", "wos"],
            papers=[_paper("FM-Agent", "10.0/fm"), _paper("BMC-Agent")],
            coverage=_incomplete_ledger(),
        )

        text = render_result(result, headline=result.headline(), limit=None)
        lines = text.splitlines()
        first_paper = next(i for i, line in enumerate(lines) if "FM-Agent" in line)
        warnings = [i for i, line in enumerate(lines) if line.startswith(WARN)]

        assert any(i < first_paper for i in warnings)
        assert any(i > first_paper for i in warnings)

    def test_every_paper_carries_its_originating_databases(self) -> None:
        """Per-paper provenance already exists and is surfaced, not rebuilt."""
        result = SearchResult(
            query="[fm]",
            papers=[_paper("FM-Agent", "10.0/fm")],
            coverage=_complete_ledger(),
        )

        text = render_result(result, headline="x", limit=None)

        assert "found_in: arxiv" in text
        assert "doi:10.0/fm" in text

    def test_display_truncation_is_stated_and_is_not_a_coverage_claim(self) -> None:
        """A shortened display must not read as a shortened set."""
        result = SearchResult(
            query="[fm]",
            papers=[_paper(f"paper {n}") for n in range(5)],
            coverage=_complete_ledger(),
        )

        text = render_result(result, headline="x", limit=2)

        assert "showing 2 of 5" in text
        assert WARN not in text

    def test_complete_result_rendering_contains_no_warning_marker(self) -> None:
        """The two renderings are distinguishable by a single grep."""
        result = SearchResult(
            query="[fm]",
            papers=[_paper("FM-Agent")],
            coverage=_complete_ledger(),
        )

        assert WARN not in render_result(result, headline="x", limit=None)


class TestReprAndStr:
    """The default Python renderings cannot hide an incomplete set."""

    def test_repr_of_a_complete_set_is_quiet(self) -> None:
        """A complete set reprs without alarm and without a shortfalls field."""
        result = SearchResult(query="[fm]", coverage=_complete_ledger())

        text = repr(result)

        assert "coverage='complete'" in text
        assert "shortfalls" not in text

    def test_repr_of_an_incomplete_set_differs_in_shape(self) -> None:
        """An extra field appears, so the difference is structural."""
        result = SearchResult(query="[fm]", coverage=_incomplete_ledger())

        text = repr(result)

        assert "INCOMPLETE:failed" in text
        assert "shortfalls=" in text
        assert "ieee (skipped_no_credential)" in text

    def test_repr_of_a_set_with_no_recorded_coverage_is_not_quiet(self) -> None:
        """A hand-built result must not repr like a verified one."""
        assert "INCOMPLETE:unknown" in repr(SearchResult(query="[fm]"))

    def test_str_shows_coverage_and_a_count_but_never_the_papers(self) -> None:
        """The short form has no papers-without-coverage variant to reach for."""
        result = SearchResult(
            query="[fm]",
            papers=[_paper("FM-Agent")],
            coverage=_incomplete_ledger(),
        )

        text = str(result)

        assert "Papers: 1" in text
        assert "FM-Agent" not in text
        assert "INCOMPLETE RESULT" in text

    def test_result_repr_helper_falls_back_to_the_reason_when_there_is_no_ledger(self) -> None:
        """With no shortfall list, the repr carries the explanation instead."""

        class Bare:
            coverage_unavailable_reason = "nothing recorded here"

        assert "nothing recorded here" in result_repr(Bare())

    def test_result_summary_helper_renders_the_headline(self) -> None:
        """The summary states the operation as well as the coverage."""

        class Bare:
            coverage_unavailable_reason = "nothing recorded here"
            papers: ClassVar[list[Paper]] = []

        assert "my headline" in result_summary(Bare(), headline="my headline")


class TestRunnerContainersWithoutLedgers:
    """Snowball and similar record no coverage, and say which gap that is."""

    def test_snowball_result_names_the_edge_fetch_gap(self) -> None:
        """A dry frontier and a failed lookup are not yet distinguishable."""
        result = SnowballResult(
            seed_paper=_paper("seed", "10.0/seed"),
            max_depth=1,
            direction="both",
            papers=[_paper("neighbour")],
        )

        text = result.render_coverage()

        assert "coverage UNKNOWN" in text
        assert "legacy DOI-only lookup path" in text
        assert "INCOMPLETE:unknown" in repr(result)

    def test_similar_result_reports_its_legacy_lists_beside_the_gap(self) -> None:
        """failed_databases is shown, and named as the strict subset it is."""
        result = SimilarResult(
            seed_paper=_paper("seed", "10.0/seed"),
            databases=["semantic_scholar"],
            failed_databases=["pubmed"],
        )

        text = result.render_coverage()

        assert "coverage UNKNOWN" in text
        assert "failed sources : pubmed" in text
        assert "skipped sources: none recorded" in text

    def test_snowball_render_lists_papers_under_the_warning(self) -> None:
        """The full rendering still refuses to show papers alone."""
        result = SnowballResult(
            seed_paper=_paper("seed", "10.0/seed"),
            max_depth=2,
            direction="backward",
            papers=[_paper("neighbour")],
        )

        text = result.render(limit=None)

        assert "neighbour" in text
        assert text.count("coverage UNKNOWN") >= 2


class TestLogCoverage:
    """The log channel announces the verdict, and is honest about its reach."""

    def test_incomplete_set_logs_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """WARNING is the level a configured application will surface."""
        result = SearchResult(query="[fm]", coverage=_incomplete_ledger())
        with caplog.at_level(logging.INFO, logger="probe.coverage"):
            log_coverage(result, logging.getLogger("probe.coverage"))

        records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(records) == 1
        assert "INCOMPLETE" in records[0].getMessage()

    def test_complete_set_logs_at_info_and_does_not_warn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A clean run must not cry wolf in the log either."""
        result = SearchResult(query="[fm]", coverage=_complete_ledger())
        with caplog.at_level(logging.INFO, logger="probe.coverage"):
            log_coverage(result, logging.getLogger("probe.coverage"))

        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []
        assert [r for r in caplog.records if r.levelno == logging.INFO] != []

    def test_lastresort_is_defeated_for_this_package(self) -> None:
        """Pin the measured fact, so the false claim cannot be re-asserted.

        ``logging.lastResort`` prints a WARNING to stderr when a record
        reaches no handler, which would make a warning audible with no
        logging setup at all.  It does not happen here:
        ``lit_search.utils.logging_config`` installs a ``_SanitizingHandler``
        on the ``lit-search`` logger at import, and ``Logger.callHandlers``
        counts it, so no record under ``lit-search`` ever reaches
        ``lastResort``.

        Measured in a subprocess because pytest attaches its own handlers to
        the root logger, which would mask the effect.  If this test starts
        failing, the package logger changed and :func:`log_coverage`'s
        docstring must be revisited -- it currently tells the reader that the
        log channel is a convenience and not the guarantee.
        """
        script = (
            "import logging, lit_search.utils.logging_config as c;"
            "lg = logging.getLogger('lit_search');"
            "logging.getLogger('lit_search.probe').warning('SILENCED');"
            "[lg.removeHandler(h) for h in list(lg.handlers)];"
            "logging.getLogger('lit_search.probe').warning('AUDIBLE')"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
        )

        assert "SILENCED" not in completed.stderr
        assert "AUDIBLE" in completed.stderr
