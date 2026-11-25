"""What a hunt has covered and which vocabulary it has used.

Two views over a session's move log, shown by ``hunt status`` and written into
the report:

* **coverage**: per provider, how many steps ran, answered, failed or could not
  run the query, and how many papers came back; plus what the hunt never did;
* **vocabulary**: every phrasing searched, with what it reached, and terms that
  recur in the kept papers but appear in no query, so a hunt can leave the
  vocabulary it started in.
"""

from __future__ import annotations

import itertools
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lit_search.hunt.session import HuntSession

_ANSWERED = ("complete", "capped", "partial")
_NOT_RUN = ("query_unsupported", "skipped_no_credential", "skipped_by_policy")
_TEXT_MOVES = ("find", "search", "snippets")
_WORD_RE = re.compile(r"[a-z][a-z0-9\-]+")
_STOPWORDS = frozenset(
    [
        "a",
        "about",
        "above",
        "across",
        "after",
        "again",
        "against",
        "all",
        "also",
        "among",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "here",
        "how",
        "however",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "more",
        "most",
        "much",
        "must",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "out",
        "over",
        "own",
        "same",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "upon",
        "very",
        "via",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "within",
        "without",
        "would",
        "you",
        "your",
        "yet",
        "new",
        "novel",
        "approach",
        "approaches",
        "method",
        "methods",
        "model",
        "models",
        "paper",
        "propose",
        "proposed",
        "results",
        "show",
        "shows",
        "using",
        "use",
        "used",
        "based",
        "study",
        "task",
        "tasks",
        "via",
        "toward",
        "towards",
        "work",
        "works",
        "first",
        "one",
        "two",
        "three",
        "different",
        "large",
        "small",
        "existing",
        "recent",
        "significantly",
        "effective",
        "efficient",
        "improve",
        "improves",
        "improved",
        "performance",
        "framework",
        "problem",
        "problems",
        "data",
    ]
)


@dataclass
class Phrasing:
    """One searched phrasing and what it reached."""

    move: int
    kind: str
    text: str
    new: int
    kept_first: int
    kept_reached: int


def provider_of(step_source: str, move_kind: str) -> str:
    """Name the service behind one step of a move."""
    source = step_source.lower()
    if "openalex" in source:
        return "openalex"
    if "arxiv-source" in source:
        return "arxiv (LaTeX source)"
    if source.startswith("add:arxiv"):
        return "arxiv"
    if source.startswith("add:h_"):
        return "handles"
    if source.startswith(("snowball:", "related:", "snippets:", "add:")):
        return "semantic_scholar"
    return source.split(":")[0]


def coverage_rows(session: HuntSession) -> list[dict[str, Any]]:
    """Per provider: steps, answered, failed, not run, papers returned, and the move kinds."""
    rows: dict[str, dict[str, Any]] = {}
    for move in session.data["moves"]:
        for step in move["steps"]:
            name = provider_of(step["source"], move["kind"])
            row = rows.setdefault(
                name,
                {
                    "provider": name,
                    "steps": 0,
                    "answered": 0,
                    "failed": 0,
                    "not_run": 0,
                    "papers": 0,
                    "moves": Counter(),
                },
            )
            row["steps"] += 1
            row["moves"][move["kind"]] += 1
            status = step.get("status")
            if status in _ANSWERED:
                row["answered"] += 1
                row["papers"] += int(step.get("retrieved") or 0)
            elif status == "failed":
                row["failed"] += 1
            elif status in _NOT_RUN:
                row["not_run"] += 1
    return sorted(rows.values(), key=lambda r: (-r["steps"], r["provider"]))


def coverage_lines(session: HuntSession) -> list[str]:
    """Markdown lines describing the hunt's coverage."""
    rows = coverage_rows(session)
    if not rows:
        return ["- no moves ran"]
    lines = [
        "| provider | steps | answered | failed | could not run | papers returned | moves |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        kinds = ", ".join(f"{k} {n}" for k, n in row["moves"].most_common())
        lines.append(
            f"| {row['provider']} | {row['steps']} | {row['answered']} | {row['failed']} | "
            f"{row['not_run']} | {row['papers']} | {kinds} |"
        )
    by_kind = Counter(m["kind"] for m in session.data["moves"])
    lines.append("")
    lines.append(
        "moves: "
        + ", ".join(f"{k} {n}" for k, n in sorted(by_kind.items()))
        + f" ({sum(by_kind.values())})"
    )
    crawls = session.data.get("crawls") or []
    for crawl in crawls:
        lines.append(
            f"crawl from {' '.join(crawl['seeds'])}: depth {crawl['depth_reached']} of "
            f"{crawl['depth']}, {sum(level['new'] for level in crawl['levels'])} new candidates, "
            f"stopped: {crawl.get('stopped') or 'still running'}"
        )
    return lines


def _first_move(candidate: Any) -> int | None:
    moves = [r["move"] for r in candidate.routes]
    return min(moves) if moves else None


def phrasings(session: HuntSession) -> list[Phrasing]:
    """Every searched phrasing, oldest first, with the kept papers it reached."""
    kept = [c for c in session.candidates.values() if c.mark in ("core", "keep")]
    rows = []
    for move in session.data["moves"]:
        if move["kind"] not in _TEXT_MOVES:
            continue
        text = move["args"].get("text") or move["args"].get("query") or ""
        number = move["number"]
        rows.append(
            Phrasing(
                move=number,
                kind=move["kind"],
                text=str(text),
                new=len(move["new"]),
                kept_first=sum(1 for c in kept if _first_move(c) == number),
                kept_reached=sum(1 for c in kept if any(r["move"] == number for r in c.routes)),
            )
        )
    return rows


def _terms(text: str) -> set[str]:
    words = [w for w in _WORD_RE.findall(text.casefold()) if w not in _STOPWORDS and len(w) > 2]
    bigrams = {f"{a} {b}" for a, b in itertools.pairwise(words)}
    return set(words) | bigrams


def unexplored_terms(session: HuntSession, limit: int = 10) -> list[str]:
    """Terms common in kept papers, rarer elsewhere, and absent from every query.

    Scored by how many kept papers use the term times how distinctive it is
    among all candidates (an idf); words of the question and of any searched
    phrasing are excluded, and a bigram is preferred over its own words.
    """
    kept = [c for c in session.candidates.values() if c.mark in ("core", "keep")]
    if len(kept) < 2:
        return []
    used = _terms(session.data.get("question") or "")
    for row in phrasings(session):
        used |= _terms(row.text)
    used_words = {t for t in used if " " not in t}
    documents = {
        c.cid: _terms(f"{c.paper.title or ''}. {c.paper.abstract or ''}")
        for c in session.candidates.values()
    }
    frequency = Counter(t for terms in documents.values() for t in terms)
    in_kept = Counter(t for c in kept for t in documents[c.cid])
    total = len(documents)
    scored = []
    for term, count in in_kept.items():
        if count < 2 or term in used or any(w in used_words for w in term.split()):
            continue
        idf = math.log((1 + total) / (1 + frequency[term]))
        scored.append((count * idf * (1.3 if " " in term else 1.0), term))
    chosen: list[str] = []
    for _, term in sorted(scored, reverse=True):
        if any(term in other or other in term for other in chosen):
            continue
        chosen.append(term)
        if len(chosen) >= limit:
            break
    return chosen


def vocabulary_lines(session: HuntSession) -> list[str]:
    """Markdown lines: the phrasings searched and the unexplored terms."""
    rows = phrasings(session)
    lines = [
        f'- move {r.move} {r.kind} "{r.text}": +{r.new} new, kept papers first reached '
        f"{r.kept_first}, reached {r.kept_reached}"
        for r in rows
    ] or ["- no keyword, Boolean or snippet search ran"]
    terms = unexplored_terms(session)
    if terms:
        lines.append("- in kept papers but never searched: " + ", ".join(terms))
    return lines


__all__ = [
    "Phrasing",
    "coverage_lines",
    "coverage_rows",
    "phrasings",
    "provider_of",
    "unexplored_terms",
    "vocabulary_lines",
]
