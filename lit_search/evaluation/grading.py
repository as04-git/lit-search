"""Grade a hunt's candidates against a confidence-set case.

A case (``docs/confidence/cases.json``) lists ``targets`` (tier 1) and
optionally ``tier2`` and ``controls``.  For each listed paper the grade says
whether it was retrieved, its shortlist position if the report ran, and its
position in the tool's own score order (what an agent sees first).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lit_search.core.citation import IdentifierKind
from lit_search.hunt.session import HuntSession

TIERS = (("targets", "tier 1"), ("tier2", "tier 2"), ("controls", "controls"))


def _norm(text: str) -> str:
    return re.sub(r"\W+", "", text.casefold())


@dataclass
class Item:
    """One graded result: a hunt candidate or an external tool's hit."""

    key: str
    title: str
    arxiv: set[str]
    doi: str
    text: str
    position: int | None
    note: str
    score_rank: int | None = None


@dataclass
class EntryGrade:
    """How one case paper fared."""

    name: str
    retrieved: bool
    position: int | None = None
    score_rank: int | None = None
    key: str | None = None
    note: str = ""


@dataclass
class TierGrade:
    """Grades for one tier of a case."""

    label: str
    entries: list[EntryGrade] = field(default_factory=list)

    @property
    def retrieved(self) -> int:
        """Number of entries retrieved."""
        return sum(e.retrieved for e in self.entries)

    @property
    def shortlisted(self) -> int:
        """Number of entries in the shortlist."""
        return sum(e.position is not None for e in self.entries)


def load_case(cases_path: Path, case_id: str) -> dict[str, Any]:
    """Return one case from a cases file."""
    cases: dict[str, dict[str, Any]] = {
        c["id"]: c for c in json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    }
    return cases[case_id]


def matches(entry: dict[str, Any], item: Item) -> bool:
    """Whether *item* is the case paper *entry* (arXiv id, DOI or exact title)."""
    if entry.get("arxiv") and entry["arxiv"] in item.arxiv:
        return True
    if entry.get("doi") and item.doi == entry["doi"].lower():
        return True
    full = _norm(entry["title"])
    if len(full) > 20 and full == _norm(item.title):
        return True
    if entry.get("arxiv") or entry.get("doi"):
        # Identified entries match by identifier or exact title only: a paper that merely
        # mentions a short target name ("Specula") in its abstract is not the target.
        return len(full) <= 20 and _norm(item.title).startswith(full)
    return len(full) <= 20 and re.search(rf"\b{re.escape(entry['title'])}\b", item.text) is not None


def session_items(session: HuntSession) -> list[Item]:
    """Turn a hunt's candidates into gradable items."""
    order = (session.data.get("report") or {}).get("shortlist") or []
    score_order = {c.cid: n for n, c in enumerate(session.ranked(include_dropped=True), start=1)}
    items = []
    for c in session.candidates.values():
        p = c.paper
        first = c.routes[0] if c.routes else {}
        items.append(
            Item(
                key=c.cid,
                title=p.title or "",
                arxiv={i.value for i in p.identifiers if i.kind is IdentifierKind.ARXIV},
                doi=(p.doi or "").lower(),
                text=f"{p.title} {p.abstract or ''}",
                position=order.index(c.cid) + 1 if c.cid in order else None,
                note=f"{c.mark or 'unjudged'}, first via move {first.get('move')} "
                f"{first.get('label')}",
                score_rank=score_order.get(c.cid),
            )
        )
    return items


def paper_finder_items(path: Path, shortlist: int = 10) -> list[Item]:
    """Turn an Ai2 Paper Finder result file into gradable items."""
    results = json.loads(path.read_text(encoding="utf-8"))["results"]
    return [
        Item(
            key=f"#{n}",
            title=r.get("title") or "",
            arxiv=set(),
            doi="",
            text=f"{r.get('title')} {r.get('abstract') or ''}",
            position=n if n <= shortlist else None,
            note=f"relevance {r.get('relevanceScore')}",
            score_rank=n,
        )
        for n, r in enumerate(results, start=1)
    ]


def grade(case: dict[str, Any], items: list[Item]) -> tuple[list[TierGrade], list[Item]]:
    """Grade *items* against every tier of *case*.

    Returns
    -------
    tuple[list[TierGrade], list[Item]]
        Per-tier grades, and shortlist items that match no case paper.
    """
    in_key: set[str] = set()
    tiers = []
    for field_name, label in TIERS:
        entries = case.get(field_name) or []
        if not entries:
            continue
        tier = TierGrade(label)
        for entry in entries:
            hits = [i for i in items if matches(entry, i)]
            in_key.update(i.key for i in hits)
            short = len(entry["title"]) <= 20
            name = entry["title"] if short else (entry.get("note") or entry["title"])
            if not hits:
                tier.entries.append(EntryGrade(name, retrieved=False))
                continue
            best = min(hits, key=lambda i: (i.position or 10**6, i.score_rank or 10**6))
            ranks = [i.score_rank for i in hits if i.score_rank is not None]
            tier.entries.append(
                EntryGrade(
                    name,
                    retrieved=True,
                    position=best.position,
                    score_rank=min(ranks) if ranks else None,
                    key=best.key,
                    note=best.note,
                )
            )
        tiers.append(tier)
    outside = sorted(
        (i for i in items if i.position and i.key not in in_key), key=lambda i: i.position or 0
    )
    return tiers, outside


def render_grade(header: str, tiers: list[TierGrade], outside: list[Item]) -> str:
    """Render a grade as the text ``tools/grade_hunt.py`` has always printed."""
    lines = [header]
    for tier in tiers:
        lines.append(f"{tier.label}:")
        for e in tier.entries:
            if not e.retrieved:
                lines.append(f"  MISSED      {e.name[:70]}")
                continue
            where = f"#{e.position}" if e.position else "retrieved"
            rank = f", score rank {e.score_rank}" if e.score_rank else ""
            lines.append(f"  {where:<11} {e.name[:70]} as {e.key} ({e.note}{rank})")
    if outside:
        lines.append(
            "shortlist entries outside the key (noise, or relevant papers the key has not judged):"
        )
        lines += [f"  #{i.position:<3} {i.title[:90]}" for i in outside]
    lines.append(
        " · ".join(
            f"{t.label} {t.retrieved}/{len(t.entries)} retrieved, {t.shortlisted} shortlisted"
            for t in tiers
        )
    )
    return "\n".join(lines)
