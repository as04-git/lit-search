"""Grading must not award a target to a paper that only mentions it."""

from __future__ import annotations

from lit_search.evaluation.grading import Item, matches


def _item(title: str, abstract: str = "", arxiv: set[str] | None = None) -> Item:
    return Item(
        key="c1",
        title=title,
        arxiv=arxiv or set(),
        doi="",
        text=f"{title} {abstract}",
        position=None,
        note="",
    )


def test_identified_short_name_is_not_matched_by_a_mention() -> None:
    entry = {"title": "Specula", "arxiv": "2509.00001"}
    citing = _item("Follow-up work", abstract="We compare against Specula and others.")
    assert not matches(entry, citing)


def test_identified_short_name_matches_by_id_or_title_start() -> None:
    entry = {"title": "Specula", "arxiv": "2509.00001"}
    assert matches(entry, _item("Unrelated title", arxiv={"2509.00001"}))
    assert matches(entry, _item("Specula: Scaling formal specifications for systems code"))


def test_unidentified_short_name_still_matches_text() -> None:
    entry = {"title": "Socrates"}
    assert matches(entry, _item("A tool", abstract="We present Socrates, which"))


def test_speculative_does_not_match_specula() -> None:
    entry = {"title": "Specula"}
    assert not matches(entry, _item("Speculative decoding for LLMs"))
