"""Tests for the handle, set, index and rerank command surface.

These are the commands ``O4`` was open about, plus the two that make them
usable.  What is under test is not that they run, but that the properties the
design rests on survive the trip through the command line:

* a difference against an incomplete subtrahend exits non-zero and says why;
* deduplication reports a residue it did not examine;
* a re-rank stores the relevance order and shows the diversified one;
* export refuses to mint citation keys.
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path

import pytest

from lit_search.cli import parser as cli_parser
from lit_search.cli.indexing import _order_members, _seed_paper
from lit_search.cli.main import main
from lit_search.cli.output import EXIT_ERROR, EXIT_INCOMPLETE, EXIT_OK
from lit_search.core.author import Author
from lit_search.core.citation import IdentifierKind, PaperIdentifier
from lit_search.core.coverage import CoverageLedger, SourceCoverage
from lit_search.core.paper import Paper
from lit_search.core.search_result import SearchResult
from lit_search.core.source import Source, SourceType
from lit_search.index.hybrid import Candidate
from lit_search.index.schema import paper_key
from lit_search.utils.persistence import save_to_json

ARXIV_DOI = "10.48550/arXiv.2604.11556"
PUBLISHED_DOI = "10.1145/3712345.3712999"
FM_TITLE = "FM-Agent: Scaling Formal Methods via LLM-Based Hoare-Style Reasoning"
HELD_DOI = "10.48550/arXiv.2604.08792"


def _paper(
    title: str, doi: str | None, abstract: str = "", authors: tuple[str, ...] = (), **extra: object
) -> Paper:
    """Build a paper for the fixtures.

    Parameters
    ----------
    title : str
        Title.
    doi : str | None
        DOI.
    abstract : str
        Abstract.
    authors : tuple[str, ...]
        Author names.
    **extra : object
        Passed through to :class:`Paper`.

    Returns
    -------
    Paper
        Paper.
    """
    return Paper(
        title=title,
        abstract=abstract,
        authors=[Author(name=name) for name in authors],
        source=Source(title="arXiv", source_type=SourceType.REPOSITORY),
        publication_date=datetime.date(2026, 4, 13),
        doi=doi,
        found_in={"arxiv"},
        **extra,  # type: ignore[arg-type]
    )


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, str]:
    """Write two result files and return the paths plus the shared CLI flags."""
    sweep = SearchResult(
        query="[spec synthesis]",
        databases=["arxiv"],
        papers=[
            _paper(
                FM_TITLE,
                ARXIV_DOI,
                "Derive a specification from a function's callers.",
                ("Wei Li",),
            ),
            _paper(
                "BMC-Agent: Agentic Model Checking",
                "10.48550/arXiv.2605.21434",
                "Dual-source generation contrasts caller intent with behaviour.",
            ),
            _paper(
                "Socrates: Active Learning for Program Semantics",
                HELD_DOI,
                "Version spaces and distinguishing inputs.",
            ),
            _paper(
                FM_TITLE,
                PUBLISHED_DOI,
                "Short camera-ready abstract.",
                ("Wei Li",),
                preprint_doi=ARXIV_DOI,
            ),
        ],
        coverage=CoverageLedger.from_sources(
            [SourceCoverage.complete("arxiv", 4, reported_total=4)]
        ),
    )
    library = SearchResult(
        query="[library]",
        databases=["zotero"],
        papers=[_paper("Socrates: Active Learning for Program Semantics", HELD_DOI)],
        coverage=CoverageLedger.from_sources(
            [SourceCoverage.partial("zotero", 1, detail="the local library was only partly synced")]
        ),
    )
    sweep_path = tmp_path / "sweep.json"
    library_path = tmp_path / "library.json"
    save_to_json(sweep, str(sweep_path))
    save_to_json(library, str(library_path))
    return {
        "root": str(tmp_path / "store"),
        "sweep": str(sweep_path),
        "library": str(library_path),
        "tmp": str(tmp_path),
    }


def _handle_id(capsys: pytest.CaptureFixture, *args: str) -> str:
    """Run a command and return the handle identifier it printed.

    Parameters
    ----------
    capsys : pytest.CaptureFixture
        Capture fixture.
    *args : str
        Command-line arguments.

    Returns
    -------
    str
        Handle identifier.
    """
    import re

    main(list(args))
    match = re.search(r"h_[0-9a-f]{32}", capsys.readouterr().out)
    assert match is not None
    return match.group(0)


@pytest.fixture
def imported(workspace: dict[str, str], capsys: pytest.CaptureFixture) -> dict[str, str]:
    """Import both result files and return their handle identifiers."""
    root = ["--store-root", workspace["root"]]
    sweep = _handle_id(capsys, "handle", "import", workspace["sweep"], "--label", "sweep", *root)
    library = _handle_id(
        capsys, "handle", "import", workspace["library"], "--label", "library", *root
    )
    return {**workspace, "sweep_file": workspace["sweep"], "sweep": sweep, "library": library}


class TestSetDifference:
    """Subtracting an incomplete set is the workflow the ledger was built for."""

    def test_it_exits_incomplete_and_names_the_subtrahend(
        self, imported: dict[str, str], capsys: pytest.CaptureFixture
    ) -> None:
        """Already-held papers reported as new is the failure being prevented."""
        code = main(
            [
                "set",
                "difference",
                imported["sweep"],
                imported["library"],
                "--store-root",
                imported["root"],
            ]
        )
        out = capsys.readouterr().out
        assert code == EXIT_INCOMPLETE
        assert "difference (subtrahend)" in out
        assert "may be present here in error" in out

    def test_the_held_paper_is_actually_removed(
        self, imported: dict[str, str], capsys: pytest.CaptureFixture
    ) -> None:
        """The warning is about what could not be removed, not about nothing working."""
        main(
            [
                "set",
                "difference",
                imported["sweep"],
                imported["library"],
                "--store-root",
                imported["root"],
                "--json",
            ]
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["paper_count"] == 3
        assert HELD_DOI not in {paper["doi"] for paper in payload["papers"]}

    def test_an_abbreviated_prefix_resolves(
        self, imported: dict[str, str], capsys: pytest.CaptureFixture
    ) -> None:
        """Content addresses are long; a person types a prefix."""
        code = main(
            [
                "set",
                "difference",
                imported["sweep"][:10],
                imported["library"][:10],
                "--store-root",
                imported["root"],
                "--quiet",
            ]
        )
        assert code == EXIT_INCOMPLETE


class TestSetUnionAndIntersect:
    """Membership operations, with coverage carried through."""

    def test_union_carries_the_worst_operand_status(
        self, imported: dict[str, str], capsys: pytest.CaptureFixture
    ) -> None:
        """A union is missing whatever either operand was missing."""
        code = main(
            [
                "set",
                "union",
                imported["sweep"],
                imported["library"],
                "--store-root",
                imported["root"],
                "--json",
            ]
        )
        payload = json.loads(capsys.readouterr().out)
        assert code == EXIT_INCOMPLETE
        assert payload["coverage"]["status"] == "partial"

    def test_intersection_keeps_only_shared_members(
        self, imported: dict[str, str], capsys: pytest.CaptureFixture
    ) -> None:
        """And is not safer for being smaller."""
        main(
            [
                "set",
                "intersect",
                imported["sweep"],
                imported["library"],
                "--store-root",
                imported["root"],
                "--json",
            ]
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["paper_count"] == 1
        assert payload["papers"][0]["doi"] == HELD_DOI


class TestSetDedup:
    """Versions are linked, and the unexamined residue is reported."""

    def test_it_links_the_preprint_and_the_published_record(
        self, imported: dict[str, str], capsys: pytest.CaptureFixture
    ) -> None:
        """Both survive; a merge would drop one of the two DOIs."""
        main(["set", "dedup", imported["sweep"], "--store-root", imported["root"], "--limit", "0"])
        out = capsys.readouterr().out
        assert "READ" in out
        assert "CITE" in out
        assert ARXIV_DOI.lower() in out.lower()
        assert PUBLISHED_DOI.lower() in out.lower()

    def test_it_says_the_residue_was_not_examined(
        self, imported: dict[str, str], capsys: pytest.CaptureFixture
    ) -> None:
        """Without --semantic, "no further duplicates" was never established."""
        main(["set", "dedup", imported["sweep"], "--store-root", imported["root"]])
        assert "DID NOT RUN" in capsys.readouterr().out

    def test_the_semantic_flag_with_the_stand_in_is_refused_not_faked(
        self, imported: dict[str, str], capsys: pytest.CaptureFixture
    ) -> None:
        """A non-semantic model must not produce scores that look like measurements."""
        main(
            [
                "set",
                "dedup",
                imported["sweep"],
                "--semantic",
                "--deterministic-embedder",
                "--store-root",
                imported["root"],
            ]
        )
        assert "is_semantic=False" in capsys.readouterr().out


class TestHandleDiscovery:
    """Set algebra is unusable without it."""

    def test_listing_shows_operation_order_and_label(
        self, imported: dict[str, str], capsys: pytest.CaptureFixture
    ) -> None:
        """A listing of bare hashes would not be discovery."""
        code = main(["handle", "list", "--store-root", imported["root"]])
        out = capsys.readouterr().out
        assert code == EXIT_OK
        assert "import" in out
        assert "'sweep'" in out
        assert "order=unordered" in out

    def test_lineage_names_the_external_source(
        self, imported: dict[str, str], capsys: pytest.CaptureFixture
    ) -> None:
        """An imported set depends on a file at a moment in time."""
        code = main(["handle", "lineage", imported["sweep"], "--store-root", imported["root"]])
        out = capsys.readouterr().out
        assert code == EXIT_OK
        assert "external result_file" in out

    def test_show_renders_coverage(
        self, imported: dict[str, str], capsys: pytest.CaptureFixture
    ) -> None:
        """A handle looked at a day later must still state what it is missing."""
        code = main(
            [
                "handle",
                "show",
                imported["library"],
                "--store-root",
                imported["root"],
                "--limit",
                "5",
            ]
        )
        out = capsys.readouterr().out
        assert code == EXIT_INCOMPLETE
        assert "partly synced" in out


class TestIndexAndRerank:
    """The index, and the ordering built on it."""

    def test_index_status_fails_before_anything_is_indexed(
        self, imported: dict[str, str], capsys: pytest.CaptureFixture
    ) -> None:
        """No full-text index means keyword retrieval is unavailable, not merely absent."""
        code = main(
            ["index", "status", "--store-root", imported["root"], "--deterministic-embedder"]
        )
        assert code == EXIT_ERROR
        assert "NO full-text index" in capsys.readouterr().out

    def test_adding_a_handle_reports_degraded_text(
        self, imported: dict[str, str], capsys: pytest.CaptureFixture
    ) -> None:
        """A record embedded without its keywords is a weaker measurement."""
        code = main(
            [
                "index",
                "add",
                imported["sweep"],
                "--store-root",
                imported["root"],
                "--deterministic-embedder",
            ]
        )
        out = capsys.readouterr().out
        assert code == EXIT_OK
        assert "embedded on incomplete text" in out

    def test_index_search_reports_every_lane(
        self, imported: dict[str, str], capsys: pytest.CaptureFixture
    ) -> None:
        """A lane that did not run must be named in the output."""
        main(
            [
                "index",
                "add",
                imported["sweep"],
                "--store-root",
                imported["root"],
                "--deterministic-embedder",
            ]
        )
        capsys.readouterr()
        main(
            [
                "index",
                "search",
                '"FM-Agent"',
                "--store-root",
                imported["root"],
                "--deterministic-embedder",
            ]
        )
        out = capsys.readouterr().out
        assert "lane vector:" in out
        assert "lane bm25:" in out
        assert "lane exact:" in out

    def test_rerank_stores_a_permutation_in_relevance_order(
        self, imported: dict[str, str], capsys: pytest.CaptureFixture
    ) -> None:
        """A re-rank orders a set; it does not select from one."""
        code = main(
            [
                "rerank",
                imported["sweep"],
                "--query",
                "deriving specifications from call sites",
                "--scoring",
                "hybrid",
                "--store-root",
                imported["root"],
                "--deterministic-embedder",
                "--json",
            ]
        )
        payload = json.loads(capsys.readouterr().out)
        assert code == EXIT_OK
        assert payload["paper_count"] == 4
        assert payload["handle"].startswith("h_")

    def test_rerank_shows_a_diversified_order_and_says_it_is_not_seeds(
        self, imported: dict[str, str], capsys: pytest.CaptureFixture
    ) -> None:
        """The stored order and the shown order are different things."""
        main(
            [
                "rerank",
                imported["sweep"],
                "--query",
                "specification inference",
                "--scoring",
                "hybrid",
                "--store-root",
                imported["root"],
                "--deterministic-embedder",
                "--limit",
                "2",
            ]
        )
        out = capsys.readouterr().out
        assert "diversified display order" in out
        assert "must not be used as a source of snowball seeds" in out

    def test_rerank_on_an_empty_handle_is_an_error_not_an_empty_ranking(
        self, imported: dict[str, str], capsys: pytest.CaptureFixture
    ) -> None:
        """An empty input and an empty answer are different facts."""
        empty = str(Path(imported["tmp"]) / "empty.json")
        save_to_json(SearchResult(query="[none]", databases=["arxiv"], papers=[]), empty)
        handle = _handle_id(capsys, "handle", "import", empty, "--store-root", imported["root"])
        code = main(
            [
                "rerank",
                handle,
                "--query",
                "anything",
                "--scoring",
                "hybrid",
                "--store-root",
                imported["root"],
                "--deterministic-embedder",
            ]
        )
        assert code == EXIT_ERROR
        assert "nothing to order" in capsys.readouterr().err

    def test_seed_mode_is_accepted_and_recorded(
        self, imported: dict[str, str], capsys: pytest.CaptureFixture
    ) -> None:
        """Ranking against a paper is document-to-document, and must say so."""
        code = main(
            [
                "rerank",
                imported["sweep"],
                "--seed",
                ARXIV_DOI,
                "--scoring",
                "hybrid",
                "--store-root",
                imported["root"],
                "--deterministic-embedder",
                "--quiet",
            ]
        )
        assert code == EXIT_OK

    def test_two_stage_refuses_the_non_semantic_stand_in(
        self, imported: dict[str, str], capsys: pytest.CaptureFixture
    ) -> None:
        """The default scoring mode needs real models and will not substitute.

        This is the refusal that keeps the stand-in out of a real ranking.  A
        fallback here would produce a confidently ordered list with nothing
        behind it, which is indistinguishable from a good one.
        """
        code = main(
            [
                "rerank",
                imported["sweep"],
                "--query",
                "anything",
                "--store-root",
                imported["root"],
                "--deterministic-embedder",
            ]
        )
        assert code == EXIT_ERROR
        err = capsys.readouterr().err
        assert "needs a real bi-encoder" in err
        assert "--scoring hybrid" in err

    def test_hybrid_scoring_is_not_advertised_as_a_semantic_rerank(self) -> None:
        """The two modes make different claims, and the help says which is which."""
        parser = cli_parser.build_parser()
        action = next(
            a for a in parser._actions if isinstance(a, cli_parser.argparse._SubParsersAction)
        )
        help_text = " ".join(action.choices["rerank"].format_help().split())
        assert "NOT a semantic re-rank" in help_text
        assert "two-stage is the semantic re-rank" in help_text

    def test_hybrid_order_does_not_match_identifier_free_members_by_title(self) -> None:
        """An ambiguous legacy title row cannot score or remove either record."""
        first = _paper("Shared title", None, "First distinct abstract.")
        second = _paper("Shared title", None, "Second distinct abstract.")
        identified = _paper("Identified", "10.1000/identified", "Third abstract.")
        candidates = [
            Candidate(
                key="shared title",
                row={"title": "Shared title"},
                lane_ranks={"vector": 1},
                fused_score=0.9,
            ),
            Candidate(
                key="10.1000/identified",
                row={"title": "Identified"},
                lane_ranks={"vector": 2},
                fused_score=0.5,
            ),
        ]

        ordered, scores = _order_members([first, second, identified], candidates)

        assert ordered == [identified, first, second]
        assert scores == [0.5, 0.0, 0.0]
        assert {id(paper) for paper in ordered} == {id(first), id(second), id(identified)}

    def test_hybrid_order_scores_identifier_free_members_by_snapshot_row(self) -> None:
        """The index's snapshot keys retrieve sparse records without title identity."""
        first = _paper("Shared title", None, "First distinct abstract.")
        second = _paper("Shared title", None, "Second distinct abstract.")
        candidates = [
            Candidate(
                key=paper_key(second),
                row={"title": second.title},
                lane_ranks={"vector": 1},
                fused_score=0.9,
            ),
            Candidate(
                key=paper_key(first),
                row={"title": first.title},
                lane_ranks={"vector": 2},
                fused_score=0.7,
            ),
        ]

        ordered, scores = _order_members([first, second], candidates)

        assert ordered == [second, first]
        assert scores == [0.9, 0.7]

    def test_same_title_seed_is_ambiguous_without_an_identifier(self) -> None:
        """Seed selection cannot silently choose the first same-title record."""
        first = _paper("Shared title", None, "First distinct abstract.")
        second = _paper("Shared title", None, "Second distinct abstract.")
        assert _seed_paper([first, second], "Shared title") is None


class TestExportCitationKeys:
    """Better BibTeX is the single authority, enforced rather than requested."""

    def test_bibtex_without_a_key_map_is_refused(
        self, workspace: dict[str, str], capsys: pytest.CaptureFixture
    ) -> None:
        """A warning does not help: the file outlives it."""
        out_path = str(Path(workspace["tmp"]) / "out.bib")
        code = main(["export", workspace["sweep"], "-f", "bibtex", "-o", out_path])
        assert code == EXIT_ERROR
        assert "refused" in capsys.readouterr().err
        assert not Path(out_path).exists()

    def test_a_record_missing_from_the_map_is_an_error(
        self, workspace: dict[str, str], capsys: pytest.CaptureFixture
    ) -> None:
        """Falling back to a minted key for one entry defeats the whole rule."""
        keys = Path(workspace["tmp"]) / "keys.json"
        keys.write_text(json.dumps({ARXIV_DOI.lower(): "li2026fmagent"}), encoding="utf-8")
        out_path = str(Path(workspace["tmp"]) / "out.bib")
        code = main(
            [
                "export",
                workspace["sweep"],
                "-f",
                "bibtex",
                "-o",
                out_path,
                "--citation-keys",
                str(keys),
            ]
        )
        assert code == EXIT_ERROR
        assert "no Better BibTeX key" in capsys.readouterr().err

    def test_a_complete_map_writes_those_exact_keys(
        self, workspace: dict[str, str], capsys: pytest.CaptureFixture
    ) -> None:
        """The keys in the file are Zotero's, not ours."""
        mapping = {
            ARXIV_DOI.lower(): "li2026fmagent",
            "10.48550/arxiv.2605.21434": "anon2026bmc",
            HELD_DOI.lower(): "anon2026socrates",
            PUBLISHED_DOI.lower(): "li2026fmagentPublished",
        }
        keys = Path(workspace["tmp"]) / "keys.json"
        keys.write_text(json.dumps(mapping), encoding="utf-8")
        out_path = Path(workspace["tmp"]) / "out.bib"
        code = main(
            [
                "export",
                workspace["sweep"],
                "-f",
                "bibtex",
                "-o",
                str(out_path),
                "--citation-keys",
                str(keys),
            ]
        )
        capsys.readouterr()
        assert code == EXIT_OK
        written = out_path.read_text(encoding="utf-8")
        for key in mapping.values():
            assert f"@misc{{{key}," in written
        assert Path(f"{out_path}.coverage.txt").exists()

    def test_ambiguous_identifier_free_title_requires_distinct_record_selectors(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """One title alias must not silently satisfy two preserved records."""
        source = tmp_path / "same-title.json"
        papers = [
            _paper("Shared title", None, abstract="first source record"),
            _paper("Shared title", None, abstract="second source record"),
        ]
        save_to_json(
            SearchResult(
                query="[shared title]",
                databases=["manual"],
                papers=papers,
                coverage=CoverageLedger.from_sources(
                    [SourceCoverage.complete("manual", 2, reported_total=2)]
                ),
            ),
            str(source),
        )
        title_map = tmp_path / "title-keys.json"
        title_map.write_text(json.dumps({"shared title": "shared2026"}), encoding="utf-8")
        destination = tmp_path / "ambiguous.bib"

        code = main(
            [
                "export",
                str(source),
                "-f",
                "bibtex",
                "-o",
                str(destination),
                "--citation-keys",
                str(title_map),
            ]
        )

        error = capsys.readouterr().err
        selectors = re.findall(r"record:[0-9a-f]{16}(?:#[0-9]+)?", error)
        assert code == EXIT_ERROR
        assert "ambiguous title fallback" in error
        assert len(selectors) == 2
        assert len(set(selectors)) == 2
        assert not destination.exists()

        explicit_map = tmp_path / "explicit-keys.json"
        explicit_map.write_text(
            json.dumps({selectors[0]: "first2026", selectors[1]: "second2026"}),
            encoding="utf-8",
        )
        code = main(
            [
                "export",
                str(source),
                "-f",
                "bibtex",
                "-o",
                str(destination),
                "--citation-keys",
                str(explicit_map),
            ]
        )
        capsys.readouterr()
        assert code == EXIT_OK
        written = destination.read_text(encoding="utf-8")
        assert "@misc{first2026," in written
        assert "@misc{second2026," in written

    def test_provider_record_keys_map_doi_less_same_title_records(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """Provider identities take precedence over unsafe title fallback."""
        source = tmp_path / "provider-records.json"
        papers = [
            _paper(
                "Shared title",
                None,
                identifiers=[PaperIdentifier(IdentifierKind.SEMANTIC_SCHOLAR, "S2-A")],
            ),
            _paper(
                "Shared title",
                None,
                identifiers=[PaperIdentifier(IdentifierKind.OPENALEX, "W123")],
            ),
        ]
        save_to_json(
            SearchResult(
                query="[shared title]",
                databases=["provider records"],
                papers=papers,
                coverage=CoverageLedger.from_sources(
                    [SourceCoverage.complete("provider records", 2, reported_total=2)]
                ),
            ),
            str(source),
        )
        key_map = tmp_path / "provider-keys.json"
        key_map.write_text(
            json.dumps(
                {
                    "semantic_scholar:s2-a": "semantic2026",
                    "openalex:w123": "openalex2026",
                }
            ),
            encoding="utf-8",
        )
        destination = tmp_path / "providers.bib"

        code = main(
            [
                "export",
                str(source),
                "-f",
                "bibtex",
                "-o",
                str(destination),
                "--citation-keys",
                str(key_map),
            ]
        )
        capsys.readouterr()

        assert code == EXIT_OK
        written = destination.read_text(encoding="utf-8")
        assert "@misc{semantic2026," in written
        assert "@misc{openalex2026," in written

    def test_duplicate_better_bibtex_key_is_refused(
        self, workspace: dict[str, str], capsys: pytest.CaptureFixture
    ) -> None:
        """A fully covered map may still be invalid when its values collide."""
        mapping = {
            ARXIV_DOI.lower(): "duplicate2026",
            "10.48550/arxiv.2605.21434": "duplicate2026",
            HELD_DOI.lower(): "socrates2026",
            PUBLISHED_DOI.lower(): "published2026",
        }
        keys = Path(workspace["tmp"]) / "duplicate-keys.json"
        keys.write_text(json.dumps(mapping), encoding="utf-8")
        destination = Path(workspace["tmp"]) / "duplicates.bib"

        code = main(
            [
                "export",
                workspace["sweep"],
                "-f",
                "bibtex",
                "-o",
                str(destination),
                "--citation-keys",
                str(keys),
            ]
        )

        error = capsys.readouterr().err
        assert code == EXIT_ERROR
        assert "duplicate BibTeX keys" in error
        assert not destination.exists()
