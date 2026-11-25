"""A hunt: a time-budgeted, durable working set that an agent drives move by move.

The session does the bookkeeping an agent should not have to: it gives every
paper a short stable id (``c1``, ``c2`` ...), merges a paper seen again by a
later move into the same candidate, keeps every route that reached it and every
evidence sentence, refuses to repeat an identical move, enforces the wall-clock
budget, and scores candidates for the purpose the hunt was started with.  The
agent spends judgment on what to search next, which candidates are on target,
and when to stop.

State is one JSON file per hunt, rewritten atomically after every move, so a
hunt survives an interrupted agent and can be resumed with its id.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import json
import math
import os
import re
import secrets
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from lit_search.core.coverage import CoverageStatus, ExecutionStatus, SourceCoverage
from lit_search.core.identity import (
    has_version_identity_conflict,
    same_record_identity,
)
from lit_search.core.paper import Paper
from lit_search.exceptions import InvalidParameterError
from lit_search.store.store import default_store_root

_T = TypeVar("_T")

MIN_BUDGET_SHARE = 0.8
PROFILES: tuple[str, ...] = ("prior-art", "orient", "frontier", "known-item")
DEFAULT_MINUTES = 10.0
DEFAULT_SHORTLIST = 10
_ID_RE = re.compile(r"^c\d+$")

PROFILE_GUIDE: dict[str, str] = {
    "prior-art": (
        "recall first: a near-duplicate in other words is the expensive miss. Berrypick in "
        "several communities' vocabularies, snowball both ways from every core paper."
    ),
    "orient": "surveys, canonical work and their reference lists; citations count for more.",
    "frontier": (
        "what is new: snowball forward from the area's established papers with --since, plus a "
        "fresh keyword sweep (citation links lag); recency counts for more."
    ),
    "known-item": "precision: distinctive phrases, names, venues; stop at a confident hit.",
}


def hunts_root() -> Path:
    """Directory holding hunt sessions.

    Returns
    -------
    Path
        ``<store root>/hunts``.
    """
    return default_store_root() / "hunts"


def _title_key(paper: Paper) -> tuple[str, int] | None:
    title = re.sub(r"\W+", "", (paper.title or "").casefold())
    if len(title) < 12 or paper.publication_date is None:
        return None
    return title, paper.publication_date.year


def _find_same(paper: Paper, pool: list[tuple[Paper, _T]]) -> _T | None:
    for other, value in pool:
        if same_record_identity(other, paper):
            return value
    key = _title_key(paper)
    if key is None:
        return None
    same_title = [(other, value) for other, value in pool if _title_key(other) == key]
    if len(same_title) == 1 and not has_version_identity_conflict(same_title[0][0], paper):
        return same_title[0][1]
    return None


def format_seconds(seconds: float) -> str:
    """Render a duration as ``6m12s`` (negative as ``-1m03s``).

    Parameters
    ----------
    seconds : float
        Duration.

    Returns
    -------
    str
        Compact duration.
    """
    sign = "-" if seconds < 0 else ""
    whole = int(abs(seconds))
    return f"{sign}{whole // 60}m{whole % 60:02d}s"


@dataclass
class Candidate:
    """One paper in the working set.

    Parameters
    ----------
    cid : str
        Short id, ``c<n>``.
    paper : Paper
        Merged record.
    routes : list[dict[str, Any]]
        Every route that reached it: ``{"move", "label", "weight"}``.
    evidence : list[str]
        Evidence sentences (snippet matches).
    mark : str | None
        ``"core"`` (does the central move: must read), ``"keep"`` (relevant),
        ``"drop"`` or ``None`` (not judged).
    note : str
        The agent's reason for its mark.
    """

    cid: str
    paper: Paper
    routes: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    mark: str | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping.

        Returns
        -------
        dict[str, Any]
            Serialised candidate.
        """
        return {
            "cid": self.cid,
            "paper": self.paper.to_dict(),
            "routes": self.routes,
            "evidence": self.evidence,
            "mark": self.mark,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Candidate:
        """Rebuild a candidate.

        Parameters
        ----------
        data : dict[str, Any]
            Output of :meth:`to_dict`.

        Returns
        -------
        Candidate
            The candidate.
        """
        return cls(
            cid=data["cid"],
            paper=Paper.from_dict(data["paper"]),
            routes=list(data.get("routes") or []),
            evidence=list(data.get("evidence") or []),
            mark=data.get("mark"),
            note=data.get("note") or "",
        )


@dataclass
class MoveRecord:
    """One completed move, as logged in the session."""

    number: int
    kind: str
    key: str
    args: dict[str, Any]
    started: float
    seconds: float
    steps: list[dict[str, Any]]
    new: list[str]
    again: list[str]
    replay: dict[str, Any] = field(default_factory=dict)
    excluded: int = 0


class HuntSession:
    """A hunt's state and the operations on it.

    Use :meth:`start` or :meth:`load`; every mutating call is followed by
    :meth:`save`.
    """

    def __init__(self, path: Path, data: dict[str, Any]) -> None:
        self.path = path
        self._lock_depth = 0
        self._adopt(data)

    def _adopt(self, data: dict[str, Any]) -> None:
        self.data = data
        self.candidates: dict[str, Candidate] = {
            cid: Candidate.from_dict(raw) for cid, raw in (data.get("candidates") or {}).items()
        }
        self.excluded: list[Paper] = [
            Paper.from_dict(raw) for raw in (data.get("excluded") or {}).get("papers", [])
        ]

    @contextlib.contextmanager
    def locked(self) -> Iterator[HuntSession]:
        """Hold the hunt's lock, reload its state, and save on exit.

        Several processes may drive one hunt (an agent's moves and a background
        ``hunt retry``); every change happens on freshly read state under this
        lock, so none overwrites another's.

        Yields
        ------
        HuntSession
            This session, holding the current on-disk state.
        """
        if self._lock_depth:
            self._lock_depth += 1
            try:
                yield self
            finally:
                self._lock_depth -= 1
            return
        fd = os.open(self.path.parent / "session.lock", os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            if self.path.exists():
                self._adopt(json.loads(self.path.read_text(encoding="utf-8")))
            self._lock_depth = 1
            try:
                yield self
                self._write()
            finally:
                self._lock_depth = 0
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def start(
        cls,
        question: str,
        *,
        profile: str,
        minutes: float = DEFAULT_MINUTES,
        shortlist: int = DEFAULT_SHORTLIST,
        root: Path | None = None,
        strict: bool = False,
        wide: bool = False,
        min_share: float = MIN_BUDGET_SHARE,
    ) -> HuntSession:
        """Create a new hunt.

        Parameters
        ----------
        question : str
            What the hunt is for, in the owner's words.
        profile : str
            One of :data:`PROFILES`.
        minutes : float
            Wall-clock budget, counted from now, including the agent's own time.
        shortlist : int
            Default shortlist size for the report.
        root : Path | None
            Hunts directory; defaults to :func:`hunts_root`.
        wide : bool
            Moves default to several times more candidates.
        strict : bool
            Enforce the written-reasoning and inspection rules (see
            :meth:`move_blockers` and :meth:`report_blockers`).
        min_share : float
            Share of the budget that must be used before the report opens (0 to 1).

        Returns
        -------
        HuntSession
            The saved, empty session.
        """
        if profile not in PROFILES:
            raise InvalidParameterError(f"profile must be one of {', '.join(PROFILES)}")
        if not question.strip():
            raise InvalidParameterError("a hunt needs a question")
        if not 0 <= min_share <= 1:
            raise InvalidParameterError("min_share must be between 0 and 1")
        if minutes <= 0 or shortlist < 1:
            raise InvalidParameterError("minutes must be positive and shortlist at least 1")
        now = time.time()
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
        hunt_id = f"hunt-{stamp}-{secrets.token_hex(2)}"
        directory = (root or hunts_root()) / hunt_id
        directory.mkdir(parents=True, exist_ok=False)
        session = cls(
            directory / "session.json",
            {
                "id": hunt_id,
                "question": question.strip(),
                "profile": profile,
                "minutes": minutes,
                "shortlist": shortlist,
                "started": now,
                "deadline": now + minutes * 60,
                "next_cid": 1,
                "strict": strict,
                "wide": wide,
                "min_share": min_share,
                "notes": [],
                "moves": [],
                "candidates": {},
            },
        )
        session.save()
        return session

    @classmethod
    def load(cls, hunt_id: str, root: Path | None = None) -> HuntSession:
        """Open an existing hunt by id or unique id prefix.

        Parameters
        ----------
        hunt_id : str
            Full id or a unique prefix (``hunt-20261010`` is enough when unique).
        root : Path | None
            Hunts directory.

        Returns
        -------
        HuntSession
            The session.
        """
        base = root or hunts_root()
        matches = sorted(p for p in base.glob(f"{hunt_id}*") if (p / "session.json").exists())
        if not matches:
            raise InvalidParameterError(f"no hunt matches {hunt_id!r} under {base}")
        if len(matches) > 1:
            names = ", ".join(p.name for p in matches)
            raise InvalidParameterError(f"{hunt_id!r} is ambiguous: {names}")
        path = matches[0] / "session.json"
        return cls(path, json.loads(path.read_text(encoding="utf-8")))

    def save(self) -> None:
        """Write the session atomically, under the hunt's lock unless already held."""
        if self._lock_depth:
            return
        self._write()

    def _write(self) -> None:
        self.data["candidates"] = {cid: c.to_dict() for cid, c in self.candidates.items()}
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".session-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(self.data, handle, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------

    @property
    def remaining(self) -> float:
        """Seconds left in the budget (negative when over)."""
        return float(self.data["deadline"]) - time.time()

    @property
    def budget_used(self) -> float:
        """Fraction of the budget used so far."""
        return self.elapsed / (float(self.data["minutes"]) * 60)

    @property
    def elapsed(self) -> float:
        """Seconds since the hunt started."""
        return time.time() - float(self.data["started"])

    def extend(self, minutes: float) -> None:
        """Add time to the budget.

        Parameters
        ----------
        minutes : float
            Minutes to add.
        """
        with self.locked():
            self.data["deadline"] = float(self.data["deadline"]) + minutes * 60
            self.data["minutes"] = float(self.data["minutes"]) + minutes

    # ------------------------------------------------------------------
    # Moves
    # ------------------------------------------------------------------

    def previous_move(self, key: str) -> dict[str, Any] | None:
        """Return the earlier move with this key, if one ran.

        Parameters
        ----------
        key : str
            Normalised move key.

        Returns
        -------
        dict[str, Any] | None
            The logged move.
        """
        return next((m for m in self.data["moves"] if m["key"] == key), None)

    def _match(self, paper: Paper) -> Candidate | None:
        return _find_same(paper, [(c.paper, c) for c in self.candidates.values()])

    def is_excluded(self, paper: Paper) -> bool:
        """Return whether *paper* is in a set the hunt was told to leave out.

        Parameters
        ----------
        paper : Paper
            A paper a move returned.

        Returns
        -------
        bool
            True when it matches an excluded record by identity or unique title.
        """
        return _find_same(paper, [(p, p) for p in self.excluded]) is not None

    def exclude(self, papers: list[Paper], label: str) -> int:
        """Leave *papers* out of every later move (papers already known elsewhere).

        Candidates already in the working set stay; only later moves skip them.

        Parameters
        ----------
        papers : list[Paper]
            Papers to leave out, e.g. the members of a handle.
        label : str
            Where they came from (a handle id), recorded in the session.

        Returns
        -------
        int
            How many papers were newly excluded.
        """
        with self.locked():
            fresh = [p for p in papers if not self.is_excluded(p)]
            block = self.data.setdefault("excluded", {"sources": [], "papers": []})
            if label not in block["sources"]:
                block["sources"].append(label)
            block["papers"].extend(p.to_dict() for p in fresh)
            self.excluded.extend(fresh)
            return len(fresh)

    def absorb(
        self,
        *,
        kind: str,
        key: str,
        args: dict[str, Any],
        started: float,
        steps: list[SourceCoverage],
        papers: list[tuple[Paper, str, float, list[str]]],
        replay: dict[str, Any] | None = None,
    ) -> MoveRecord:
        """Fold one move's results into the working set and log the move.

        Parameters
        ----------
        kind : str
            Move kind (``find``, ``related``, ``snippets``).
        key : str
            Normalised key used to refuse repeats.
        args : dict[str, Any]
            Move arguments, for the log.
        started : float
            ``time.time()`` when the move began.
        steps : list[SourceCoverage]
            Per-source / per-step coverage of the move.
        papers : list[tuple[Paper, str, float, list[str]]]
            ``(paper, route label, route weight, evidence sentences)`` in rank order.
        replay : dict[str, Any] | None
            The move's full parameters, so ``hunt retry`` can run it again.

        Returns
        -------
        MoveRecord
            What the move added.
        """
        with self.locked():
            return self._absorb(kind, key, args, started, steps, papers, replay or {})

    def _absorb(
        self,
        kind: str,
        key: str,
        args: dict[str, Any],
        started: float,
        steps: list[SourceCoverage],
        papers: list[tuple[Paper, str, float, list[str]]],
        replay: dict[str, Any],
    ) -> MoveRecord:
        number = len(self.data["moves"]) + 1
        new: list[str] = []
        again: list[str] = []
        excluded = 0
        for paper, label, weight, evidence in papers:
            candidate = self._match(paper)
            if candidate is None and self.excluded and self.is_excluded(paper):
                excluded += 1
                continue
            if candidate is None:
                cid = f"c{self.data['next_cid']}"
                self.data["next_cid"] += 1
                candidate = Candidate(cid=cid, paper=paper)
                self.candidates[cid] = candidate
                new.append(cid)
            else:
                if not has_version_identity_conflict(candidate.paper, paper):
                    candidate.paper.merge(paper)
                if candidate.cid not in new and candidate.cid not in again:
                    again.append(candidate.cid)
            candidate.routes.append({"move": number, "label": label, "weight": round(weight, 4)})
            for sentence in evidence:
                if sentence not in candidate.evidence:
                    candidate.evidence.append(sentence)
        record = MoveRecord(
            number=number,
            kind=kind,
            key=key,
            args=args,
            started=started,
            seconds=round(time.time() - started, 2),
            steps=[s.to_dict() for s in steps],
            new=new,
            again=again,
            replay=replay,
            excluded=excluded,
        )
        self.data["moves"].append(record.__dict__)
        self.save()
        return record

    def resolve(self, ref: str) -> Candidate:
        """Return the candidate for ``c<n>``.

        Parameters
        ----------
        ref : str
            Candidate id.

        Returns
        -------
        Candidate
            The candidate.
        """
        if not _ID_RE.match(ref) or ref not in self.candidates:
            raise InvalidParameterError(f"{ref!r} is not a candidate in {self.data['id']}")
        return self.candidates[ref]

    def mark(self, refs: list[str], mark: str | None, note: str = "") -> list[Candidate]:
        """Record the agent's judgment on candidates.

        Parameters
        ----------
        refs : list[str]
            Candidate ids.
        mark : str | None
            ``"core"``, ``"keep"``, ``"drop"`` or ``None`` to clear.
        note : str
            Reason.

        Returns
        -------
        list[Candidate]
            The marked candidates.
        """
        with self.locked():
            chosen = [self.resolve(ref) for ref in refs]
            for candidate in chosen:
                candidate.mark = mark
                if note:
                    candidate.note = note
        return chosen

    # ------------------------------------------------------------------
    # Reasoning notes and strict-mode rules
    # ------------------------------------------------------------------

    @property
    def strict(self) -> bool:
        """Whether the strict reasoning/inspection rules apply."""
        return bool(self.data.get("strict"))

    def add_note(self, text: str, kind: str = "note") -> None:
        """Record the agent's reasoning.

        Parameters
        ----------
        text : str
            The note.
        kind : str
            ``"plan"``, ``"note"`` or ``"final"``.
        """
        if not text.strip():
            raise InvalidParameterError("a note needs text")
        if kind == "plan":
            communities = [ln for ln in text.splitlines() if ln.strip().startswith("- ")]
            if len(communities) < 3:
                raise InvalidParameterError(
                    "a plan lists at least 3 research communities, one per line starting "
                    "with '- ', each with how that community would phrase the idea"
                )
        with self.locked():
            self.data.setdefault("notes", []).append(
                {
                    "kind": kind,
                    "after_move": len(self.data["moves"]),
                    "at": time.time(),
                    "text": text.strip(),
                }
            )

    def _notes(self, kind: str | None = None) -> list[dict[str, Any]]:
        notes = self.data.get("notes") or []
        return [n for n in notes if kind is None or n["kind"] == kind]

    def _unjudged_from_last_move(self, top: int = 5) -> list[str]:
        if not self.data["moves"]:
            return []
        new = [self.candidates[c] for c in self.data["moves"][-1]["new"] if c in self.candidates]
        new.sort(key=lambda c: -self.score(c))
        return [c.cid for c in new[:top] if c.mark is None]

    def snowball_queue(self) -> dict[str, list[str]]:
        """Core and kept candidates not yet snowballed, per direction.

        Returns
        -------
        dict[str, list[str]]
            ``{"forward": [...], "backward": [...]}`` candidate ids.
        """
        crawled = self.data.get("crawled") or {}
        judged = [c.cid for c in self.ranked() if c.mark in ("core", "keep")]
        return {
            direction: [cid for cid in judged if direction not in crawled.get(cid, [])]
            for direction in ("forward", "backward")
        }

    def move_blockers(self) -> list[str]:
        """Strict-mode reasons the next move may not run yet.

        Returns
        -------
        list[str]
            Empty when the move may run (always empty outside strict mode).
        """
        if not self.strict:
            return []
        blockers: list[str] = []
        if not self._notes("plan"):
            blockers.append(
                "write the plan first: `hunt note ID --plan` with at least 3 lines '- <community>: "
                "<how they would title a paper on this>'"
            )
        last = len(self.data["moves"])
        if last and not any(n["after_move"] == last for n in self._notes()):
            blockers.append(
                f"write a note on move {last}: what it showed, what is still missing, and why "
                'your next move follows (`hunt note ID "..."`)'
            )
        unjudged = self._unjudged_from_last_move()
        if unjudged:
            blockers.append(
                f"judge the top new candidates of move {last} first: {' '.join(unjudged)} "
                "(`hunt show`, then `hunt mark --core/--keep/--drop --note`)"
            )
        return blockers

    def report_blockers(self) -> list[str]:
        """Strict-mode reasons the report may not be written yet.

        Returns
        -------
        list[str]
            Empty when the report may be written (always empty outside strict mode).
        """
        if not self.strict:
            return []
        blockers = [b for b in self.move_blockers() if not b.startswith("write the plan")]
        unjudged_top = [c.cid for c in self.ranked()[:20] if c.mark is None]
        if unjudged_top:
            blockers.append(
                f"judge every top-20 candidate: {' '.join(unjudged_top)} "
                "(`hunt status ID --unjudged`)"
            )
        if self.data["profile"] == "prior-art":
            kinds = [m["kind"] for m in self.data["moves"]]
            if kinds.count("find") < 3:
                blockers.append("prior-art needs at least 3 `find` moves (different communities)")
            directions = [
                m["args"].get("direction") for m in self.data["moves"] if m["kind"] == "snowball"
            ]
            if not any(d in ("forward", "both") for d in directions) or not any(
                d in ("backward", "both") for d in directions
            ):
                blockers.append("prior-art needs snowballing in both directions (`hunt snowball`)")
            core_left = [
                cid
                for cids in self.snowball_queue().values()
                for cid in cids
                if self.candidates[cid].mark == "core"
            ]
            if core_left:
                blockers.append(
                    f"snowball every core paper both ways: {' '.join(dict.fromkeys(core_left))}"
                )
            if kinds.count("snippets") < 1:
                blockers.append("prior-art needs at least 1 `snippets` (full-text) move")
        if not self._notes("final"):
            blockers.append(
                "write the final note: `hunt note ID --final` with what was found, the closest "
                "papers and why, and what was not covered"
            )
        return blockers

    def reasoning_log(self) -> list[str]:
        """The notes as report lines.

        Returns
        -------
        list[str]
            ``"after move N (kind): text"`` lines.
        """
        return [
            f"- after move {n['after_move']} ({n['kind']}): {n['text']}"
            for n in self.data.get("notes") or []
        ]

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, candidate: Candidate) -> float:
        """Heuristic priority for the hunt's profile.

        Route weights add up (a paper several moves reached ranks higher); the
        best single route also counts once more, so a strong one-route find is
        not buried by weak agreement.  ``orient`` multiplies by a citation
        factor and ``frontier`` by a recency factor.

        Parameters
        ----------
        candidate : Candidate
            Candidate to score.

        Returns
        -------
        float
            Score; higher is better.
        """
        weights = [float(r["weight"]) for r in candidate.routes] or [0.0]
        base = sum(weights) + max(weights)
        profile = self.data["profile"]
        paper = candidate.paper
        if profile == "orient":
            base *= 1 + math.log10(1 + (paper.citations or 0)) / 3
        elif profile == "frontier" and paper.publication_date is not None:
            age_days = (dt.date.today() - paper.publication_date).days
            base *= 2.0 if age_days <= 365 else 1.0 if age_days <= 3 * 365 else 0.6
        return base

    def set_order(self, refs: list[str]) -> list[Candidate]:
        """Record the agent's own ranking; it leads the shortlist in this order.

        Parameters
        ----------
        refs : list[str]
            Candidate ids, best first. Replaces any earlier order.

        Returns
        -------
        list[Candidate]
            The ranked candidates.
        """
        with self.locked():
            chosen = [self.resolve(ref) for ref in refs]
            self.data["order"] = list(dict.fromkeys(c.cid for c in chosen))
        return chosen

    @property
    def min_share(self) -> float:
        """Share of the budget that must be used before the report opens."""
        return float(self.data.get("min_share", MIN_BUDGET_SHARE))

    def needs_more_time(self) -> bool:
        """Whether the report must wait: under the minimum budget share (known-item exempt).

        Returns
        -------
        bool
            ``True`` while time remains below the minimum share.
        """
        return (
            self.data["profile"] != "known-item"
            and self.remaining > 0
            and self.budget_used < self.min_share
        )

    def ranked(self, *, include_dropped: bool = False) -> list[Candidate]:
        """Candidates in the agent's order (``hunt rank``), then by mark, then score.

        Parameters
        ----------
        include_dropped : bool
            Include candidates marked ``drop``.

        Returns
        -------
        list[Candidate]
            Ordered candidates.
        """
        pool = [c for c in self.candidates.values() if include_dropped or c.mark != "drop"]
        tier = {"core": 0, "keep": 1}
        order = {cid: i for i, cid in enumerate(self.data.get("order") or [])}
        return sorted(
            pool,
            key=lambda c: (order.get(c.cid, len(order)), tier.get(c.mark or "", 2), -self.score(c)),
        )

    # ------------------------------------------------------------------
    # Coverage summary
    # ------------------------------------------------------------------

    @staticmethod
    def _failed_steps(move: dict[str, Any]) -> list[SourceCoverage]:
        steps = [SourceCoverage.from_dict(raw) for raw in move["steps"]]
        return [
            s
            for s in steps
            if s.status is CoverageStatus.FAILED
            or s.error
            or (s.detail or "").startswith("cooling down")
        ]

    def failed_moves(self) -> list[dict[str, Any]]:
        """Moves with a failed or refused step that no later retry has fixed.

        Returns
        -------
        list[dict[str, Any]]
            Logged moves, oldest first; only those that can be replayed.
        """
        return [
            m
            for m in self.data["moves"]
            if m.get("replay")
            and not m.get("resolved_by")
            and not m["args"].get("retry_of")
            and self._failed_steps(m)
        ]

    def shortfalls(self) -> list[str]:
        """One line per step that failed or fell short, across all moves.

        Returns
        -------
        list[str]
            ``"move 3 find: openaire FAILED (HTTPError: 503 ...)"`` style lines.
        """
        lines: list[str] = []
        for move in self.data["moves"]:
            if move.get("resolved_by") or move["args"].get("retry_of"):
                continue
            for raw in move["steps"]:
                step = SourceCoverage.from_dict(raw)
                failed = step.status is CoverageStatus.FAILED
                short = step.execution_status is not ExecutionStatus.COMPLETE
                if failed or short:
                    reason = step.error or step.detail or step.status.value
                    lines.append(
                        f"move {move['number']} {move['kind']}: {step.source} "
                        f"{'FAILED' if failed else 'fell short'} ({reason[:140]})"
                    )
        return lines


__all__ = [
    "DEFAULT_MINUTES",
    "DEFAULT_SHORTLIST",
    "MIN_BUDGET_SHARE",
    "PROFILES",
    "PROFILE_GUIDE",
    "Candidate",
    "HuntSession",
    "MoveRecord",
    "format_seconds",
    "hunts_root",
]
