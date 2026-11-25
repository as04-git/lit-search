"""Tests for the text-field policy and its degradation record.

The failure being guarded against: a record with no abstract is scored on a
bare title, sinks in the ranking for a reason unrelated to its relevance, and
nothing in the output says so.
"""

from __future__ import annotations

from lit_search.rerank import (
    MIN_USABLE_ABSTRACT_CHARS,
    TEXT_POLICY_ID,
    TextQuality,
    build_document_text,
    build_documents,
    degraded_identities,
    quality_counts,
)
from tests.unit.rerank.conftest import blank_title, make_paper


class TestBuiltText:
    """What actually gets embedded."""

    def test_full_record_carries_all_three_labelled_fields(self) -> None:
        """Title, abstract, and keywords, each labelled."""
        paper = make_paper(
            "Agentic model checking",
            "A" * 80,
            doi="10.1/x",
            keywords={"verification", "agents"},
        )
        built = build_document_text(paper)
        assert built.text.startswith("Title: Agentic model checking")
        assert "\nAbstract: " in built.text
        assert "\nKeywords: agents; verification" in built.text
        assert built.quality is TextQuality.FULL

    def test_title_comes_first_so_truncation_takes_the_abstract_tail(self) -> None:
        """Truncation removes from the end, so the title must lead."""
        built = build_document_text(make_paper("A title", "B" * 200))
        assert built.text.index("Title:") < built.text.index("Abstract:")

    def test_keywords_are_sorted_for_reproducibility(self) -> None:
        """Set iteration order must not leak into the embedded string."""
        first = build_document_text(make_paper("T", "A" * 80, keywords={"z", "a", "m"}))
        second = build_document_text(make_paper("T", "A" * 80, keywords={"m", "z", "a"}))
        assert first.text == second.text
        assert "Keywords: a; m; z" in first.text

    def test_empty_fields_are_omitted_not_left_dangling(self) -> None:
        """No 'Abstract:' label with nothing after it."""
        built = build_document_text(make_paper("Only a title", ""))
        assert "Abstract:" not in built.text
        assert "Keywords:" not in built.text


class TestDegradationIsRecorded:
    """Incompleteness is a recorded fact, not a silent fallback."""

    def test_missing_abstract_is_labelled(self) -> None:
        """The common case: Crossref and WoS records without abstracts."""
        built = build_document_text(make_paper("Specula", "", doi="10.1/specula"))
        assert built.quality is TextQuality.NO_ABSTRACT
        assert built.is_degraded

    def test_stub_abstract_counts_as_missing(self) -> None:
        """A stub is not an abstract, and calling it one hides degradation."""
        stub = "Abstract not available."
        assert len(stub) < MIN_USABLE_ABSTRACT_CHARS
        built = build_document_text(make_paper("T", stub))
        assert built.quality is TextQuality.NO_ABSTRACT
        assert stub not in built.text

    def test_abstract_at_the_threshold_is_kept(self) -> None:
        """The boundary is inclusive, so the constant means what it says."""
        built = build_document_text(make_paper("T", "x" * MIN_USABLE_ABSTRACT_CHARS))
        assert built.quality is TextQuality.FULL

    def test_record_with_no_text_is_still_scored_and_named(self) -> None:
        """Dropping it would shrink the set with nothing to show it."""
        built = build_document_text(blank_title(make_paper("t", "", doi="10.1/blank")))
        assert built.quality is TextQuality.EMPTY
        assert built.text, "an empty string would be ranked by a model artifact"
        assert built.identity == "doi:10.1/blank"

    def test_title_only_absent_is_distinguished_from_both_absent(self) -> None:
        """Four outcomes, four labels; collapsing them loses information."""
        no_title = blank_title(make_paper("t", "A" * 80))
        assert build_document_text(no_title).quality is TextQuality.NO_TITLE
        neither = blank_title(make_paper("t", ""))
        assert build_document_text(neither).quality is TextQuality.EMPTY

    def test_degraded_identities_lists_only_degraded_records(self, candidates) -> None:
        """The fixture set has exactly one abstract-less record."""
        documents = build_documents(candidates)
        assert degraded_identities(documents) == ("doi:10.1000/specula",)

    def test_quality_counts_reports_every_label(self, candidates) -> None:
        """A caller rendering the report should not need the label set."""
        counts = quality_counts(build_documents(candidates))
        assert set(counts) == {quality.value for quality in TextQuality}
        assert counts["full"] == 3
        assert counts["no_abstract"] == 1


class TestIdentity:
    """Identity is both the attribution key and the tie-break key."""

    def test_doi_wins_and_is_lowercased(self) -> None:
        """DOIs are case-insensitive; two cases must not be two papers."""
        built = build_document_text(make_paper("T", "A" * 80, doi="10.1000/ABC"))
        assert built.identity == "doi:10.1000/abc"

    def test_identifier_free_record_uses_snapshot_and_position_not_title(self) -> None:
        """A title is evidence to inspect, not source-record identity."""
        built = build_document_text(make_paper("Some Title", "A" * 80), position=3)
        assert built.identity.startswith("snapshot:")
        assert built.identity.endswith("@position:3")
        assert "some title" not in built.identity

    def test_textless_record_still_has_snapshot_local_identity(self) -> None:
        """A record with no scoring text still cannot vanish from reporting."""
        anonymous = blank_title(make_paper("t", ""))
        identity = build_document_text(anonymous, position=7).identity
        assert identity.startswith("snapshot:")
        assert identity.endswith("@position:7")

    def test_same_title_distinct_snapshots_do_not_share_identity(self) -> None:
        """Different identifier-free records must not overwrite one another."""
        documents = build_documents(
            [
                make_paper("Shared title", "A" * 80),
                make_paper("Shared title", "B" * 80),
            ]
        )
        assert documents[0].identity != documents[1].identity

    def test_repeated_explicit_identifier_keeps_both_occurrences(self) -> None:
        """Malformed duplicate input remains a two-member permutation."""
        documents = build_documents(
            [
                make_paper("First snapshot", "A" * 80, doi="10.1000/same"),
                make_paper("Second snapshot", "B" * 80, doi="10.1000/same"),
            ]
        )
        assert documents[0].identity == "doi:10.1000/same"
        assert documents[1].identity == "doi:10.1000/same@occurrence:1"


def test_policy_id_is_versioned() -> None:
    """Rankings built under two policies are not comparable."""
    assert TEXT_POLICY_ID.endswith("/v1")
    assert "title" in TEXT_POLICY_ID
    assert "abstract" in TEXT_POLICY_ID
