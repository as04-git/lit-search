"""Tests for the ``lit-search`` command-line surface.

Three invariants are under test, and they are the reason the CLI exists in
this stage at all:

1. No command prints papers without printing coverage.
2. The exit code reports coverage, so a script cannot treat a partial answer
   as a finished one by default.
3. Nothing that cannot be done is faked.  ``rerank`` errors rather than
   returning the input order, and a failed lookup errors rather than
   printing an empty set.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import ClassVar

import pytest

from lit_search.cli import commands as cli_commands
from lit_search.cli import parser as cli_parser
from lit_search.cli.main import main
from lit_search.cli.output import (
    EXIT_ERROR,
    EXIT_INCOMPLETE,
    EXIT_OK,
    EXIT_USAGE,
)
from lit_search.core.coverage import CoverageLedger, CoverageScope, SourceCoverage
from lit_search.core.lookup_result import LookupResult
from lit_search.core.paper import Paper
from lit_search.core.rendering import WARN
from lit_search.core.search_result import SearchResult
from lit_search.core.snowball_result import SnowballResult
from lit_search.core.source import Source, SourceType
from lit_search.store import operations as store_operations
from lit_search.store.store import HandleStore
from lit_search.utils.persistence import save_to_json


def _paper(title: str = "FM-Agent", doi: str = "10.0/fm") -> Paper:
    """Build a minimal paper.

    Parameters
    ----------
    title : str
        Paper title.
    doi : str
        Paper DOI.

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


def _complete_result() -> SearchResult:
    """Build a search result whose coverage is complete.

    Returns
    -------
    SearchResult
        Complete result.
    """
    return SearchResult(
        query="[fm]",
        databases=["arxiv"],
        papers=[_paper()],
        coverage=CoverageLedger.from_sources(
            [SourceCoverage.complete("arxiv", 1, reported_total=1)]
        ),
    )


def _incomplete_result() -> SearchResult:
    """Build a search result with one skipped and one failed source.

    Returns
    -------
    SearchResult
        Incomplete result.
    """
    return SearchResult(
        query="[fm]",
        databases=["arxiv", "ieee", "wos"],
        papers=[_paper()],
        coverage=CoverageLedger.from_sources(
            [
                SourceCoverage.complete("arxiv", 1, reported_total=1),
                SourceCoverage.skipped_no_credential("ieee", credential="ieee_api_key"),
                SourceCoverage.failed("wos", error="502 from upstream"),
            ]
        ),
    )


def _bounded_result() -> SearchResult:
    """Build a fulfilled default plan with an intentionally narrower scope."""
    return SearchResult(
        query="[fm]",
        databases=["arxiv"],
        papers=[_paper()],
        coverage=CoverageLedger(
            entries=(
                SourceCoverage.complete("arxiv", 1, reported_total=1),
                SourceCoverage.skipped_by_policy("ieee", detail="default profile"),
            ),
            scope=CoverageScope(
                profile="default",
                selected_sources=("arxiv",),
                registered_sources=("arxiv", "ieee"),
            ),
        ),
    )


class FakeEngine:
    """Stand-in for :class:`lit_search.engine.Engine` in CLI tests."""

    search_result: ClassVar[SearchResult | None] = None
    get_result: ClassVar[LookupResult | Paper | None] = None
    snowball_result: ClassVar[SnowballResult | None] = None
    calls: ClassVar[list[tuple[str, dict]]] = []

    def __init__(self, **_kwargs: object) -> None:
        """Accept and ignore engine configuration."""

    def search(self, query: str, **kwargs: object) -> SearchResult:
        """Return the configured search result.

        Parameters
        ----------
        query : str
            Query string.
        **kwargs : object
            Ignored search options, recorded for assertions.

        Returns
        -------
        SearchResult
            The configured result.
        """
        FakeEngine.calls.append(("search", {"query": query, **kwargs}))
        if FakeEngine.search_result is None:
            raise AssertionError("FakeEngine.search_result was not configured")
        return FakeEngine.search_result

    def get(self, identifier: str, **kwargs: object) -> LookupResult | Paper | None:
        """Return the configured lookup result.

        Parameters
        ----------
        identifier : str
            DOI or URL.
        **kwargs : object
            Ignored lookup options, recorded for assertions.

        Returns
        -------
        Paper | None
            The configured paper, or ``None``.
        """
        FakeEngine.calls.append(("get", {"identifier": identifier, **kwargs}))
        return FakeEngine.get_result

    def snowball(self, paper: Paper | list[Paper], **kwargs: object) -> SnowballResult:
        """Return the configured traversal result.

        Parameters
        ----------
        paper : Paper
            Seed paper.
        **kwargs : object
            Ignored traversal options, recorded for assertions.

        Returns
        -------
        SnowballResult
            The configured result.
        """
        seeds = [paper.doi] if isinstance(paper, Paper) else [item.doi for item in paper]
        FakeEngine.calls.append(("snowball", {"seed": seeds, **kwargs}))
        if FakeEngine.snowball_result is None:
            raise AssertionError("FakeEngine.snowball_result was not configured")
        return FakeEngine.snowball_result


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> type[FakeEngine]:
    """Install :class:`FakeEngine` in place of the real engine.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Patching fixture.

    Returns
    -------
    type[FakeEngine]
        The fake, with its class attributes reset.
    """
    FakeEngine.search_result = None
    FakeEngine.get_result = None
    FakeEngine.snowball_result = None
    FakeEngine.calls = []
    monkeypatch.setattr(cli_commands, "Engine", FakeEngine)
    return FakeEngine


class TestSurface:
    """The five subcommands exist and the argument conventions hold."""

    def test_the_registered_subcommands_are_exactly_the_documented_surface(self) -> None:
        """The three set-producing commands, the three that manage sets, and export.

        Pinned as an exact set: a subcommand that exists and is not documented
        is a surface nobody reviewed, and one that is documented and missing is
        a promise the tool does not keep.
        """
        parser = cli_parser.build_parser()
        action = next(
            a for a in parser._actions if isinstance(a, cli_parser.argparse._SubParsersAction)
        )

        assert set(action.choices) == {
            "find",
            "hunt",
            "related",
            "snippets",
            "search",
            "get",
            "snowball",
            "handle",
            "set",
            "index",
            "rerank",
            "export",
            "config",
            "doctor",
            "eval",
            "fetch",
        }

    def test_version_prints_and_exits_zero(self, capsys: pytest.CaptureFixture) -> None:
        """--version is a query, not a run."""
        code = main(["--version"])

        assert code == EXIT_OK
        assert "lit-search" in capsys.readouterr().out

    def test_no_command_prints_help_and_exits_usage(self, capsys: pytest.CaptureFixture) -> None:
        """A bare invocation is a usage error, not a silent success."""
        code = main([])

        assert code == EXIT_USAGE
        assert "COMMAND" in capsys.readouterr().out

    def test_unknown_database_is_rejected_by_the_parser(self) -> None:
        """A typo in -d must not silently narrow the search."""
        with pytest.raises(SystemExit) as excinfo:
            main(["search", "[fm]", "-d", "not_a_database"])

        assert excinfo.value.code == EXIT_USAGE

    def test_iso_date_rejects_a_non_date(self) -> None:
        """--since takes YYYY-MM-DD and says so."""
        with pytest.raises(cli_parser.argparse.ArgumentTypeError, match="YYYY-MM-DD"):
            cli_parser.iso_date("last tuesday")

    def test_positive_int_rejects_zero(self) -> None:
        """A cap of zero is a mistake, not a request for nothing."""
        with pytest.raises(cli_parser.argparse.ArgumentTypeError, match="1 or more"):
            cli_parser.positive_int("0")

    def test_display_limit_treats_zero_as_no_limit(self) -> None:
        """--limit 0 shows everything."""
        assert cli_parser.display_limit("0") is None
        assert cli_parser.display_limit("7") == 7

    def test_display_limit_rejects_a_negative(self) -> None:
        """A negative display limit is a usage error."""
        with pytest.raises(cli_parser.argparse.ArgumentTypeError, match="0 or more"):
            cli_parser.display_limit("-1")


class TestSearchRendersCoverage:
    """Search shows what it consulted, and its exit code says whether that was all."""

    def test_complete_search_exits_zero_and_shows_no_warning(
        self, engine: type[FakeEngine], capsys: pytest.CaptureFixture
    ) -> None:
        """A clean run is quiet and green."""
        engine.search_result = _complete_result()

        code = main(["search", "[fm]"])
        out = capsys.readouterr().out

        assert code == EXIT_OK
        assert WARN not in out
        assert "FM-Agent" in out

    def test_incomplete_search_exits_three_and_names_every_shortfall(
        self, engine: type[FakeEngine], capsys: pytest.CaptureFixture
    ) -> None:
        """The two sources that did not answer are named in the output."""
        engine.search_result = _incomplete_result()

        code = main(["search", "[fm]"])
        out = capsys.readouterr().out

        assert code == EXIT_INCOMPLETE
        assert "INCOMPLETE RESULT" in out
        assert "ieee (skipped_no_credential)" in out
        assert "wos (failed)" in out
        assert "ieee_api_key" in out

    def test_bounded_default_scope_exits_zero_without_hiding_exclusions(
        self, engine: type[FakeEngine], capsys: pytest.CaptureFixture
    ) -> None:
        """Policy breadth is machine-visible but does not impersonate an outage."""
        engine.search_result = _bounded_result()

        code = main(["search", "[fm]", "--json"])
        payload = json.loads(capsys.readouterr().out)

        assert code == EXIT_OK
        assert payload["complete"] is False
        assert payload["execution_complete"] is True
        assert payload["exit_basis"] == "execution_complete"
        assert payload["coverage"]["scope"]["excluded_sources"] == ["ieee"]

    def test_incomplete_search_warns_above_and_below_the_papers(
        self, engine: type[FakeEngine], capsys: pytest.CaptureFixture
    ) -> None:
        """Scrolling past the banner still ends on a warning."""
        engine.search_result = _incomplete_result()

        main(["search", "[fm]"])
        lines = capsys.readouterr().out.splitlines()
        paper_line = next(i for i, line in enumerate(lines) if "FM-Agent" in line)
        marks = [i for i, line in enumerate(lines) if line.startswith(WARN)]

        assert any(i < paper_line for i in marks)
        assert any(i > paper_line for i in marks)

    def test_json_output_always_carries_a_coverage_key(
        self, engine: type[FakeEngine], capsys: pytest.CaptureFixture
    ) -> None:
        """A machine consumer cannot read the papers without the verdict."""
        engine.search_result = _incomplete_result()

        code = main(["search", "[fm]", "--json"])
        payload = json.loads(capsys.readouterr().out)

        assert code == EXIT_INCOMPLETE
        assert payload["complete"] is False
        assert payload["coverage"]["status"] == "failed"
        assert "ieee (skipped_no_credential)" in payload["coverage"]["shortfalls"]
        assert payload["paper_count"] == 1

    def test_quiet_suppresses_the_report_but_not_the_exit_code(
        self, engine: type[FakeEngine], capsys: pytest.CaptureFixture
    ) -> None:
        """--quiet is for scripts; the verdict survives it."""
        engine.search_result = _incomplete_result()

        code = main(["search", "[fm]", "--quiet"])

        assert code == EXIT_INCOMPLETE
        assert capsys.readouterr().out == ""

    def test_output_writes_a_file_and_prints_its_path_not_its_content(
        self, engine: type[FakeEngine], capsys: pytest.CaptureFixture, tmp_path: Path
    ) -> None:
        """D9: a path, never inline content."""
        engine.search_result = _complete_result()
        destination = tmp_path / "run.json"

        main(["search", "[fm]", "-o", str(destination)])
        out = capsys.readouterr().out

        assert destination.exists()
        assert f"Written: {destination}" in out
        assert '"metadata"' not in out

    def test_written_file_carries_the_coverage_ledger(
        self, engine: type[FakeEngine], tmp_path: Path
    ) -> None:
        """A saved set does not lose the record of what it is missing."""
        engine.search_result = _incomplete_result()
        destination = tmp_path / "run.json"

        main(["search", "[fm]", "-o", str(destination), "--quiet"])
        payload = json.loads(destination.read_text(encoding="utf-8"))

        assert payload["metadata"]["coverage"]["status"] == "failed"

    def test_database_restriction_is_passed_through(self, engine: type[FakeEngine]) -> None:
        """-d is repeatable and reaches the engine."""
        engine.search_result = _complete_result()

        main(["search", "[fm]", "-d", "arxiv", "-d", "pubmed", "--quiet"])

        assert engine.calls[0][1]["databases"] == ["arxiv", "pubmed"]


class TestGet:
    """A lookup that finds nothing is an error, not an empty set."""

    def test_missing_record_exits_error_and_prints_no_result(
        self, engine: type[FakeEngine], capsys: pytest.CaptureFixture
    ) -> None:
        """The runner swallows per-source errors, so absence proves nothing."""
        engine.get_result = None

        code = main(["get", "10.0/nope"])
        captured = capsys.readouterr()

        assert code == EXIT_ERROR
        assert captured.out == ""
        assert "not evidence the paper does not exist" in captured.err

    def test_found_record_still_reports_unknown_coverage(
        self, engine: type[FakeEngine], capsys: pytest.CaptureFixture
    ) -> None:
        """get records no ledger, and says which gap that is on every run."""
        engine.get_result = _paper()

        code = main(["get", "10.0/fm"])
        out = capsys.readouterr().out

        assert code == EXIT_INCOMPLETE
        assert "coverage unknown" in out
        assert "no coverage was recorded" in out
        assert "FM-Agent" in out

    def test_failed_lookup_renders_ledger_and_exits_incomplete(
        self, engine: type[FakeEngine], capsys: pytest.CaptureFixture
    ) -> None:
        """A failed provider is visible in both output shape and exit code."""
        engine.get_result = LookupResult(
            identifier="10.0/nope",
            coverage=CoverageLedger.from_sources(
                [SourceCoverage.failed("crossref", error="HTTP 429", error_type="HTTPError")]
            ),
        )

        code = main(["get", "10.0/nope"])
        out = capsys.readouterr().out

        assert code == EXIT_INCOMPLETE
        assert "crossref: failed" in out
        assert "not evidence that no matching paper exists" in out

    def test_explicit_not_found_renders_complete_empty_result(
        self, engine: type[FakeEngine], capsys: pytest.CaptureFixture
    ) -> None:
        """An explicit exact zero is a complete, empty lookup rather than an error."""
        engine.get_result = LookupResult(
            identifier="10.0/nope",
            coverage=CoverageLedger.from_sources(
                [SourceCoverage.complete("crossref", 0, reported_total=0)]
            ),
        )

        code = main(["get", "10.0/nope"])
        out = capsys.readouterr().out

        assert code == EXIT_OK
        assert "coverage complete" in out
        assert "no record" in out


class TestSnowball:
    """Traversal reports that it cannot yet tell closure from a failed fetch."""

    def test_unresolvable_seed_is_an_error_not_an_empty_neighbourhood(
        self, engine: type[FakeEngine], capsys: pytest.CaptureFixture
    ) -> None:
        """No traversal was attempted, and the output says so."""
        engine.get_result = None

        code = main(["snowball", "10.0/nope"])
        captured = capsys.readouterr()

        assert code == EXIT_ERROR
        assert captured.out == ""
        assert "not an empty neighbourhood" in captured.err

    def test_traversal_reports_the_missing_ledger(
        self, engine: type[FakeEngine], capsys: pytest.CaptureFixture
    ) -> None:
        """R14 is stated in the output rather than left to the reader."""
        engine.get_result = _paper()
        engine.snowball_result = SnowballResult(
            seed_paper=_paper(),
            max_depth=1,
            direction="both",
            papers=[_paper("BMC-Agent", "10.0/bmc")],
        )

        code = main(["snowball", "10.0/fm"])
        out = capsys.readouterr().out

        assert code == EXIT_INCOMPLETE
        assert "legacy DOI-only lookup path" in out
        assert "BMC-Agent" in out

    def test_managed_run_is_announced_before_and_forwarded(
        self,
        engine: type[FakeEngine],
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The recoverable run ID is visible even if traversal is later interrupted."""
        run_id = "r_0123456789abcdef0123456789abcdef"
        engine.get_result = _paper()
        engine.snowball_result = SnowballResult(
            seed_paper=_paper(),
            max_depth=1,
            direction="both",
            papers=[],
            coverage=CoverageLedger.from_sources(
                [SourceCoverage.complete("semantic_scholar", 0, reported_total=0)]
            ),
            run_id=run_id,
        )
        monkeypatch.setattr(
            cli_commands.uuid, "uuid4", lambda: type("U", (), {"hex": run_id[2:]})()
        )

        code = main(
            [
                "snowball",
                "10.0/fm",
                "--managed-run",
                "--store-root",
                str(tmp_path),
            ]
        )
        captured = capsys.readouterr()

        assert code == EXIT_OK
        assert f"started managed run {run_id}" in captured.err
        call = next(payload for name, payload in engine.calls if name == "snowball")
        assert call["managed_store_root"] == tmp_path
        assert call["managed_run_id"] == run_id
        assert call["resume_managed_run"] is False

    def test_resume_run_forwards_exact_id_and_semantics(
        self,
        engine: type[FakeEngine],
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Resume is explicit and reuses the user-visible run identifier."""
        run_id = "r_fedcba9876543210fedcba9876543210"
        engine.get_result = _paper()
        engine.snowball_result = SnowballResult(
            seed_paper=_paper(),
            max_depth=1,
            direction="both",
            papers=[],
            coverage=CoverageLedger.from_sources(
                [SourceCoverage.complete("semantic_scholar", 0, reported_total=0)]
            ),
            run_id=run_id,
        )

        code = main(
            [
                "snowball",
                "10.0/fm",
                "--resume-run",
                run_id,
                "--store-root",
                str(tmp_path),
            ]
        )
        captured = capsys.readouterr()

        assert code == EXIT_OK
        assert f"resuming managed run {run_id}" in captured.err
        call = next(payload for name, payload in engine.calls if name == "snowball")
        assert call["managed_run_id"] == run_id
        assert call["resume_managed_run"] is True

    def test_handle_seeds_every_member_and_records_lineage(
        self,
        engine: type[FakeEngine],
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """The preferred protocol expands the whole handle, not a hand-picked DOI."""
        seeds = [_paper("Code-Augur", "10.0/code-augur"), _paper("Other", "10.0/other")]
        ledger = CoverageLedger.from_sources(
            [SourceCoverage.complete("semantic_scholar", 2, reported_total=2)]
        )
        store = HandleStore(tmp_path)
        seed_handle = store.put(
            store_operations.searched(
                seeds,
                query="[specification inference]",
                databases=["semantic_scholar"],
                coverage=ledger.to_dict(),
            )
        )
        engine.snowball_result = SnowballResult(
            seed_paper=seeds[0],
            seed_papers=seeds,
            max_depth=2,
            direction="both",
            papers=[_paper("FM-Agent", "10.0/fm")],
            coverage=ledger,
        )

        code = main(
            [
                "snowball",
                seed_handle,
                "--handle",
                "--no-enrich",
                "--store-root",
                str(tmp_path),
            ]
        )
        out = capsys.readouterr().out

        assert code == EXIT_OK
        call = next(payload for name, payload in engine.calls if name == "snowball")
        assert call["seed"] == ["10.0/code-augur", "10.0/other"]
        assert call["max_depth"] == 2
        assert call["enrichment_databases"] == []
        result_handle = next(
            line.split(": ", 1)[1] for line in out.splitlines() if line.startswith("Handle: ")
        )
        stored = store.get(result_handle)
        assert {paper.title for paper in stored.members} == {"Code-Augur", "Other", "FM-Agent"}
        assert stored.operation.handle_operand_ids() == [seed_handle]


class TestRerankRefuses:
    """When a ranking cannot be produced, the command errors rather than pass through.

    ``rerank`` is implemented now (see ``tests/unit/cli/test_cli_sets.py`` for
    the working path).  What must not change is the refusal: an input that
    cannot be ranked has to fail, because returning the input order would be
    indistinguishable from a ranked list.
    """

    def test_an_unknown_handle_is_an_error_and_prints_nothing_to_stdout(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """A missing input must not yield an ordering of nothing."""
        code = main(
            ["rerank", "h_deadbeef", "--query", "spec inference", "--store-root", str(tmp_path)]
        )
        captured = capsys.readouterr()

        assert code == EXIT_ERROR
        assert captured.out == ""
        assert "No handle matching" in captured.err

    def test_rerank_requires_a_mode(self) -> None:
        """Query mode and seed mode are different operations (D14)."""
        with pytest.raises(SystemExit) as excinfo:
            main(["rerank", "h_deadbeef"])

        assert excinfo.value.code == EXIT_USAGE


class TestExport:
    """An export writes files, prints paths, and never travels without coverage."""

    @pytest.fixture
    def saved(self, tmp_path: Path) -> Path:
        """Write an incomplete search result to disk.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory.

        Returns
        -------
        Path
            Path of the saved JSON.
        """
        path = tmp_path / "run.json"
        save_to_json(_incomplete_result(), str(path))
        return path

    def test_json_export_writes_the_file_and_a_coverage_sidecar(
        self, saved: Path, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """The sidecar is written in every format and whatever the verdict."""
        destination = tmp_path / "out.json"

        code = main(["export", str(saved), "-f", "json", "-o", str(destination)])
        out = capsys.readouterr().out

        assert code == EXIT_INCOMPLETE
        assert destination.exists()
        sidecar = Path(f"{destination}.coverage.txt")
        assert sidecar.exists()
        assert "ieee (skipped_no_credential)" in sidecar.read_text(encoding="utf-8")
        assert f"Written: {destination}" in out
        assert f"Written: {sidecar}" in out

    def test_export_never_prints_the_file_content(
        self, saved: Path, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """D9: paths out, content on disk."""
        destination = tmp_path / "out.csv"

        main(["export", str(saved), "-f", "csv", "-o", str(destination)])
        out = capsys.readouterr().out

        assert "FM-Agent" not in out
        assert destination.read_text(encoding="utf-8").count("FM-Agent") == 1

    def test_bibtex_export_is_refused_rather_than_warned_about(
        self, saved: Path, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """D7 makes Better BibTeX the single authority for citation keys.

        This test previously asserted that a ``.bib`` was written with a
        warning on stderr.  That was the weaker behaviour and it was wrong: the
        file outlives the warning, and whoever merges it discovers the key
        disagreement instead of the person who was warned.  A ``.bib`` is now
        written only from keys Better BibTeX supplied.
        """
        destination = tmp_path / "out.bib"

        code = main(["export", str(saved), "-f", "bibtex", "-o", str(destination)])
        err = capsys.readouterr().err

        assert code == EXIT_ERROR
        assert not destination.exists()
        assert "Better BibTeX" in err
        assert "--citation-keys" in err

    def test_complete_set_exports_with_a_clean_sidecar_and_exits_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """A complete set is allowed to look complete."""
        source = tmp_path / "clean.json"
        save_to_json(_complete_result(), str(source))
        destination = tmp_path / "out.json"

        code = main(["export", str(source), "-f", "json", "-o", str(destination)])
        sidecar = Path(f"{destination}.coverage.txt").read_text(encoding="utf-8")

        assert code == EXIT_OK
        assert WARN not in sidecar
        assert "complete" in capsys.readouterr().out

    def test_json_envelope_names_the_sidecar(
        self, saved: Path, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """A machine consumer is told where the coverage record went."""
        destination = tmp_path / "out.json"

        main(["export", str(saved), "-f", "json", "-o", str(destination), "--json"])
        payload = json.loads(capsys.readouterr().out)

        assert payload["coverage_sidecar"] == f"{destination}.coverage.txt"
        assert payload["complete"] is False

    def test_missing_input_is_an_error_not_an_empty_export(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """A path that does not exist must not produce an empty file."""
        destination = tmp_path / "out.json"

        code = main(["export", str(tmp_path / "absent.json"), "-o", str(destination)])

        assert code == EXIT_ERROR
        assert not destination.exists()
        assert "error" in capsys.readouterr().err


class TestPackageSurface:
    """The new abstractions are reachable from the package root."""

    def test_coverage_and_handle_abstractions_are_exported(self) -> None:
        """They were invisible from ``lit_search.__all__`` before this stage."""
        import lit_search

        for name in (
            "CoverageLedger",
            "CoverageStatus",
            "CoverageView",
            "CoveredPapers",
            "DerivationNote",
            "HandleRecord",
            "HandleStore",
            "SourceCoverage",
            "coverage_view",
            "render_coverage_report",
            "render_result",
        ):
            assert name in lit_search.__all__
            assert hasattr(lit_search, name)

    def test_every_exported_name_resolves(self) -> None:
        """``__all__`` and the module namespace agree."""
        import lit_search

        assert [n for n in lit_search.__all__ if not hasattr(lit_search, n)] == []
