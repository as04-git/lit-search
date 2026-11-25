"""What earlier hunts decided about a paper.

Every saved hunt holds judgments (core, keep, drop) with reasons.  This module
indexes them across hunts so that a paper reappearing in a new hunt shows what
was concluded about it before, and in answer to which question.  The index is
a JSON file beside the hunts, refreshed from any session whose file changed.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lit_search.utils.latex_bib import arxiv_id_in, normalise

_INDEX = "judgments.json"
_VERSION = 1


@dataclass(frozen=True)
class Judgment:
    """One earlier hunt's mark on a paper."""

    hunt: str
    question: str
    mark: str
    note: str


def paper_keys(paper: dict[str, Any] | Any) -> list[str]:
    """Identity keys of a paper (DOI, arXiv id, normalised title), most specific first."""
    if not isinstance(paper, dict):
        paper = paper.to_dict()
    keys = []
    doi = (paper.get("doi") or "").lower()
    if doi:
        keys.append(f"doi:{doi}")
        if doi.startswith("10.48550/arxiv."):
            keys.append(f"arxiv:{doi.split('arxiv.')[-1]}")
    for identifier in paper.get("identifiers") or []:
        if identifier.get("kind") == "arxiv" and (found := arxiv_id_in(identifier.get("value"))):
            keys.append(f"arxiv:{found}")
    title = normalise(paper.get("title"))
    if len(title) > 20:
        keys.append(f"title:{title}")
    return list(dict.fromkeys(keys))


class JudgmentMemory:
    """Judgments of every saved hunt, by paper key."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / _INDEX
        self._data: dict[str, Any] = {"version": _VERSION, "hunts": {}}
        self._load()
        self._refresh()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if data.get("version") == _VERSION:
            self._data = data

    def _refresh(self) -> None:
        hunts: dict[str, Any] = self._data["hunts"]
        changed = False
        present = set()
        for session_file in self.root.glob("hunt-*/session.json"):
            hunt = session_file.parent.name
            present.add(hunt)
            try:
                mtime = session_file.stat().st_mtime
            except OSError:
                continue
            if hunts.get(hunt, {}).get("mtime") == mtime:
                continue
            try:
                session = json.loads(session_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            judged = []
            for candidate in (session.get("candidates") or {}).values():
                if not candidate.get("mark"):
                    continue
                judged.append(
                    {
                        "keys": paper_keys(candidate.get("paper") or {}),
                        "mark": candidate["mark"],
                        "note": candidate.get("note") or "",
                    }
                )
            hunts[hunt] = {
                "mtime": mtime,
                "question": session.get("question") or "",
                "judged": judged,
            }
            changed = True
        for gone in set(hunts) - present:
            del hunts[gone]
            changed = True
        self._by_key: dict[str, list[Judgment]] = {}
        for hunt, record in hunts.items():
            for item in record["judged"]:
                judgment = Judgment(hunt, record["question"], item["mark"], item["note"])
                for key in item["keys"]:
                    self._by_key.setdefault(key, []).append(judgment)
        if changed:
            self._save()

    def _save(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(dir=self.root, prefix=".judgments-")
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(self._data, stream)
            os.replace(temporary, self.path)
        except OSError:
            pass

    def lookup(self, paper: Any, *, exclude_hunt: str | None = None) -> list[Judgment]:
        """Earlier judgments of *paper*, newest hunt first, one per hunt."""
        found: dict[str, Judgment] = {}
        for key in paper_keys(paper):
            for judgment in self._by_key.get(key, []):
                if judgment.hunt != exclude_hunt:
                    found.setdefault(judgment.hunt, judgment)
        return sorted(found.values(), key=lambda j: j.hunt, reverse=True)


def describe(judgments: list[Judgment], limit: int = 2) -> str:
    """One line: ``kept in hunt-… ("why") · dropped in hunt-…``."""
    parts = []
    for judgment in judgments[:limit]:
        verb = {"core": "core", "keep": "kept", "drop": "dropped"}.get(judgment.mark, judgment.mark)
        note = f' ("{judgment.note[:80]}")' if judgment.note else ""
        parts.append(f"{verb} in {judgment.hunt}{note}")
    more = f" · +{len(judgments) - limit} more" if len(judgments) > limit else ""
    return " · ".join(parts) + more


__all__ = ["Judgment", "JudgmentMemory", "describe", "paper_keys"]
