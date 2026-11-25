"""Presentation layer: a result set is never shown without its coverage.

:mod:`lit_search.core.coverage` records, per source, what was retrieved and
what was not.  Recording it is only half the job.  The failure that motivated
this project was not an *absent* record -- it was a confident, well-organised,
incomplete answer that nobody could tell apart from a complete one.  **A
ledger that exists and is never shown is a ledger that does not exist.**

This module is the single place that knows how a result set is presented, and
every presentation path in the tree routes through it: ``__repr__`` and
``__str__`` on the result containers, the search runner's end-of-run log line,
and the command-line interface.  There is deliberately **no** public function
here that renders papers without also rendering coverage.

Three consequences, each deliberate:

* :class:`CoverageView` normalises the three states a result object can be in
  -- it carries a ledger, it carries something that is *not* a ledger, or its
  type records no coverage at all.  All three are rendered.  None is silently
  treated as "fine": the second and third both read as
  :data:`~lit_search.core.coverage.CoverageStatus.UNKNOWN`, and both say why.
* An incomplete result differs in **structure**, not only in wording: a banner
  above the content, a repeated warning below it, and an extra field in the
  ``repr``.  A reader who scrolls past the banner still ends on the warning.
* Zero papers together with non-complete coverage gets its own sentence,
  because that is the shape most easily misread as "nothing matching exists".

See ``D18``, ``R1`` and ``R7`` in the project specification, and the
"default rendering path for coverage" entry under ``O19``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lit_search.core.coverage import CoverageLedger, CoverageScope, CoverageStatus, ExecutionStatus

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lit_search.core.paper import Paper

#: Width of the horizontal rules used by the banners.
RULE_WIDTH = 78

_HEAVY_RULE = "=" * RULE_WIDTH
_LIGHT_RULE = "-" * RULE_WIDTH

#: Marker prefixing every line of an incomplete-coverage banner.  Grep-able on
#: purpose: an unattended run's log can be scanned for it.
WARN = "!!"

_NO_COVERAGE_ATTRIBUTE = (
    "{type_name} records no coverage at all, so nothing is known about which "
    "sources were reached or what they failed to return"
)
_NOT_A_LEDGER = (
    "{type_name}.coverage is {actual}, not a CoverageLedger, so its claim "
    "cannot be checked and must not be believed"
)
_NO_PAPERS_ATTRIBUTE = "{type_name} exposes no 'papers' attribute"


@dataclass(frozen=True, slots=True)
class CoverageView:
    """A result set's coverage, or an explicit statement that it has none.

    The point of the type is that "no ledger" is a *state to be rendered*
    rather than an absence to be skipped.  Every consumer therefore gets a
    :class:`CoverageView`, never a ``CoverageLedger | None`` it has to
    remember to check.

    Parameters
    ----------
    ledger : CoverageLedger | None
        The set's ledger, or ``None`` when it has none.
    unavailable_reason : str | None
        Why there is no ledger.  Required whenever *ledger* is ``None`` --
        an unexplained absence is exactly what this module exists to prevent.

    Raises
    ------
    ValueError
        If *ledger* is ``None`` and no reason was supplied.
    """

    ledger: CoverageLedger | None = None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        """Reject an unexplained absence of coverage.

        Raises
        ------
        ValueError
            If no ledger and no reason were supplied.
        """
        if self.ledger is None and not self.unavailable_reason:
            raise ValueError(
                "CoverageView with no ledger must carry an unavailable_reason: "
                "an absence nobody explains is the failure this module prevents"
            )

    @property
    def available(self) -> bool:
        """Whether a real ledger backs this view.

        Returns
        -------
        bool
            ``True`` when a :class:`~lit_search.core.coverage.CoverageLedger`
            is present.
        """
        return self.ledger is not None

    @property
    def status(self) -> CoverageStatus:
        """Aggregate coverage status.

        Returns
        -------
        CoverageStatus
            The ledger's aggregate, or
            :data:`~lit_search.core.coverage.CoverageStatus.UNKNOWN` when
            there is no ledger.
        """
        return CoverageStatus.UNKNOWN if self.ledger is None else self.ledger.status

    @property
    def is_complete(self) -> bool:
        """Whether the set can be shown as complete.

        Returns
        -------
        bool
            ``True`` only when a ledger is present and aggregates to
            :data:`~lit_search.core.coverage.CoverageStatus.COMPLETE`.
        """
        return self.ledger is not None and self.ledger.is_complete

    @property
    def execution_status(self) -> ExecutionStatus:
        """Aggregate declared-plan execution status."""
        return ExecutionStatus.UNKNOWN if self.ledger is None else self.ledger.execution_status

    @property
    def execution_is_complete(self) -> bool:
        """Whether the declared operation had no unplanned shortfall."""
        return self.ledger is not None and self.ledger.execution_is_complete

    @property
    def execution_shortfalls(self) -> tuple[str, ...]:
        """Compact labels for execution failures or unknowns."""
        return () if self.ledger is None else self.ledger.execution_shortfalls

    @property
    def scope(self) -> CoverageScope | None:
        """Declared source boundary, when recorded."""
        return None if self.ledger is None else self.ledger.scope

    @property
    def shortfalls(self) -> tuple[str, ...]:
        """Compact labels for everything that fell short.

        Returns
        -------
        tuple[str, ...]
            One ``"source (status)"`` label per incomplete source, followed
            by one per derivation note.  Empty when the view is complete or
            has no ledger.
        """
        if self.ledger is None:
            return ()
        sources = tuple(f"{e.source} ({e.status.value})" for e in self.ledger.incomplete_entries)
        derived = tuple(
            f"{n.operation}/{n.role} ({n.status.value})" for n in self.ledger.derivations
        )
        return sources + derived

    def headline(self) -> str:
        """Return the single sentence that states the coverage verdict.

        Returns
        -------
        str
            One line, safe to use on its own in a log record.
        """
        if self.ledger is None:
            return f"execution UNKNOWN; coverage UNKNOWN -- {self.unavailable_reason}"
        if self.execution_is_complete:
            return (
                f"execution complete; coverage {self.status.value} -- declared operation "
                "finished without an unplanned shortfall"
            )
        short = len(self.execution_shortfalls)
        selected = len(self.scope.selected_sources) if self.scope is not None else len(self.ledger)
        if selected == 0:
            return (
                f"execution {self.execution_status.value}; coverage {self.status.value} -- no "
                "applicable source execution was recorded for this set"
            )
        return (
            f"execution {self.execution_status.value} -- {short} declared unit(s) fell short; "
            f"matching papers may exist and not be listed"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the view for machine-readable output.

        Every key is always present.  An omitted key reads as "fine", which
        is the misreading this whole module exists to prevent.

        Returns
        -------
        dict[str, Any]
            Keys ``available``, ``status``, ``complete``,
            ``unavailable_reason``, ``shortfalls`` and ``ledger``.
        """
        return {
            "available": self.available,
            "status": self.status.value,
            "complete": self.is_complete,
            "execution": {
                "status": self.execution_status.value,
                "complete": self.execution_is_complete,
                "shortfalls": list(self.execution_shortfalls),
            },
            "scope": self.scope.to_dict() if self.scope is not None else None,
            "unavailable_reason": self.unavailable_reason,
            "shortfalls": list(self.shortfalls),
            "ledger": None if self.ledger is None else self.ledger.to_dict(),
        }


def coverage_view(result: object) -> CoverageView:
    """Build a :class:`CoverageView` for any result-like object.

    Three cases, all rendered rather than skipped:

    1. ``result.coverage`` is a
       :class:`~lit_search.core.coverage.CoverageLedger` -- the normal path.
    2. ``result.coverage`` exists but is something else.  ``HandleRecord``
       types its ``coverage`` as ``Any``, so a bare string can reach here;
       treating it as a ledger would let ``'complete'`` pass as a claim.
    3. ``result`` has no ``coverage`` attribute at all.  Today that is every
       snowball, similar and get result (see ``R14``).

    In cases 2 and 3 the object may declare a
    ``coverage_unavailable_reason`` attribute naming the specific gap.  That
    is preferred over the generic message: "the snowball runner records no
    ledger, so a dry frontier and a failed edge fetch look identical" tells
    the reader what to do, and "no coverage recorded" does not.

    Parameters
    ----------
    result : object
        Any result container.

    Returns
    -------
    CoverageView
        A view that is renderable in all three cases.
    """
    type_name = type(result).__name__
    if not hasattr(result, "coverage"):
        return CoverageView(
            unavailable_reason=_declared_reason(result)
            or _NO_COVERAGE_ATTRIBUTE.format(type_name=type_name)
        )
    raw = result.coverage
    if isinstance(raw, CoverageLedger):
        return CoverageView(ledger=raw)
    if raw is None and (declared := _declared_reason(result)) is not None:
        return CoverageView(unavailable_reason=declared)
    return CoverageView(
        unavailable_reason=_NOT_A_LEDGER.format(type_name=type_name, actual=type(raw).__name__)
    )


def _declared_reason(result: object) -> str | None:
    """Return the object's own explanation for having no coverage, if any.

    Parameters
    ----------
    result : object
        Result container.

    Returns
    -------
    str | None
        A non-empty ``coverage_unavailable_reason``, or ``None``.
    """
    declared = getattr(result, "coverage_unavailable_reason", None)
    return declared if isinstance(declared, str) and declared else None


# ----------------------------------------------------------------------
# Banners
# ----------------------------------------------------------------------


def coverage_banner(view: CoverageView) -> str:
    """Render the banner that opens any display of a result set.

    A complete set gets a quiet two-line block.  Anything else gets a
    marked block naming every shortfall, so that the difference is
    structural rather than a change of adjective.

    Parameters
    ----------
    view : CoverageView
        Coverage to announce.

    Returns
    -------
    str
        Multi-line banner, no trailing newline.
    """
    if view.execution_is_complete:
        label = "" if view.is_complete else "BOUNDED RESULT -- "
        lines = [_HEAVY_RULE, f"{label}{view.headline()}"]
        lines.extend(_scope_lines(view))
        lines.append(_HEAVY_RULE)
        return "\n".join(lines)
    lines = [_HEAVY_RULE, f"{WARN} INCOMPLETE RESULT -- {view.headline()}"]
    lines.extend(_shortfall_lines(view))
    lines.append(_HEAVY_RULE)
    return "\n".join(lines)


def coverage_footer(view: CoverageView, *, paper_count: int | None = None) -> str:
    """Render the closing warning that repeats an incomplete verdict.

    Returned empty for a complete set: there is nothing to repeat.  The
    repetition exists because a reader who scrolls past a long paper list
    never sees the opening banner again.

    Parameters
    ----------
    view : CoverageView
        Coverage to restate.
    paper_count : int | None
        Number of papers displayed, when known.  Zero papers under
        non-complete coverage is called out explicitly: it is the shape most
        easily misread as "nothing matching exists".

    Returns
    -------
    str
        Multi-line footer, or ``""`` when coverage is complete.
    """
    if view.execution_is_complete:
        return ""
    lines = [_LIGHT_RULE, f"{WARN} {view.headline()}"]
    lines.extend(_shortfall_lines(view))
    if paper_count == 0:
        lines.append(
            f"{WARN} This set is EMPTY and its coverage is not complete. That is not "
            "evidence that no matching paper exists."
        )
    lines.append(f"{WARN} Do not read this set as a complete answer.")
    lines.append(_HEAVY_RULE)
    return "\n".join(lines)


def _shortfall_lines(view: CoverageView) -> list[str]:
    """Return the marked lines naming what fell short.

    Parameters
    ----------
    view : CoverageView
        Coverage to describe.

    Returns
    -------
    list[str]
        One line per shortfall, or a single line when there is no ledger.
    """
    if view.ledger is None:
        return [f"{WARN} no coverage ledger: {view.unavailable_reason}"]
    return [f"{WARN}   {label}" for label in view.execution_shortfalls]


def _scope_lines(view: CoverageView) -> list[str]:
    """Render a declared scope as a neutral boundary, never a failure."""
    scope = view.scope
    if scope is None:
        return []
    lines = [
        f"scope {scope.profile}: {len(scope.selected_sources)} of "
        f"{len(scope.registered_sources)} registered source(s) selected"
    ]
    if scope.excluded_sources:
        lines.append(f"excluded by declared scope: {', '.join(scope.excluded_sources)}")
    return lines


# ----------------------------------------------------------------------
# Result rendering
# ----------------------------------------------------------------------


def render_coverage_report(view: CoverageView) -> str:
    """Render coverage on its own: banner, per-source detail, closing warning.

    Parameters
    ----------
    view : CoverageView
        Coverage to render.

    Returns
    -------
    str
        Multi-line report.
    """
    parts = [coverage_banner(view)]
    if view.ledger is not None:
        parts.append(view.ledger.render())
    footer = coverage_footer(view)
    if footer:
        parts.append(footer)
    return "\n".join(parts)


def render_result(
    result: object,
    *,
    headline: str,
    limit: int | None = 10,
) -> str:
    """Render a whole result set: coverage, then papers, then coverage again.

    This is the only public function that renders papers, and it cannot be
    called without rendering coverage.  That is the enforcement: there is no
    papers-only rendering path to reach for under time pressure.

    Parameters
    ----------
    result : object
        A result container exposing ``papers``, and optionally ``coverage``.
    headline : str
        One line describing the operation, for example
        ``'search "[machine learning]" across 8 databases'``.
    limit : int | None
        Maximum number of papers to list.  ``None`` lists all of them.  When
        the list is truncated the rendering says so -- a truncated *display*
        is not a coverage shortfall, but it is still something the reader
        must not mistake for the whole set.

    Returns
    -------
    str
        Multi-line report.
    """
    view = coverage_view(result)
    papers = _papers_of(result)
    parts = [coverage_banner(view), headline]
    if view.ledger is not None:
        parts.append(view.ledger.render())
    parts.append(_render_papers(papers, limit=limit))
    footer = coverage_footer(view, paper_count=len(papers))
    if footer:
        parts.append(footer)
    return "\n".join(part for part in parts if part)


def _papers_of(result: object) -> Sequence[Paper]:
    """Return the papers carried by *result*, or an empty sequence.

    Parameters
    ----------
    result : object
        Result container.

    Returns
    -------
    Sequence[Paper]
        The papers, or ``()`` when the object exposes none.
    """
    papers = getattr(result, "papers", None)
    if isinstance(papers, Sequence):
        return papers
    return ()


def _render_papers(papers: Sequence[Paper], *, limit: int | None) -> str:
    """Render the paper list, stating explicitly when it is truncated.

    Parameters
    ----------
    papers : Sequence[Paper]
        Papers to list.
    limit : int | None
        Maximum entries to show, or ``None`` for all.

    Returns
    -------
    str
        Multi-line listing.
    """
    total = len(papers)
    if total == 0:
        return "Papers: none"
    shown = total if limit is None else min(limit, total)
    header = (
        f"Papers ({total}):"
        if shown == total
        else f"Papers (showing {shown} of {total}; raise --limit to see the rest):"
    )
    lines = [header]
    width = len(str(shown))
    for index, paper in enumerate(papers[:shown], start=1):
        lines.append(f"{index:>{width}}. {getattr(paper, 'title', None) or '(untitled)'}")
        lines.append(f"{' ' * (width + 2)}{_paper_detail(paper)}")
    return "\n".join(lines)


def _paper_detail(paper: Paper) -> str:
    """Return the one-line provenance suffix shown under a paper's title.

    Parameters
    ----------
    paper : Paper
        Paper to describe.

    Returns
    -------
    str
        Pipe-separated date, DOI and originating databases.
    """
    date = getattr(paper, "publication_date", None)
    doi = getattr(paper, "doi", None)
    found_in = sorted(getattr(paper, "found_in", None) or ())
    bits = [
        str(date) if date else "no date",
        f"doi:{doi}" if doi else "no doi",
        f"found_in: {', '.join(found_in)}" if found_in else "found_in: unrecorded",
    ]
    if getattr(paper, "is_retracted", False):
        bits.append("RETRACTED")
    return " | ".join(bits)


# ----------------------------------------------------------------------
# repr / str
# ----------------------------------------------------------------------


def result_repr(result: object, **fields: Any) -> str:
    """Build a ``repr`` that cannot hide an incomplete set.

    A complete set renders ``coverage=complete``.  Anything else renders
    ``coverage=INCOMPLETE:<status>`` **and** an extra ``shortfalls=`` field,
    so the two differ in shape and not merely in a word.

    Parameters
    ----------
    result : object
        The object being represented.
    **fields : Any
        Extra ``name=value`` pairs, rendered before the coverage fields.

    Returns
    -------
    str
        Single-line ``repr`` string.
    """
    view = coverage_view(result)
    parts = [f"{name}={value!r}" for name, value in fields.items()]
    if view.execution_is_complete:
        parts.append("coverage='complete'")
        if not view.is_complete:
            parts.append(f"retrieval_coverage='BOUNDED:{view.status.value}'")
    else:
        parts.append(f"coverage='INCOMPLETE:{view.status.value}'")
        parts.append(f"execution={view.execution_status.value!r}")
        parts.append(f"shortfalls={list(view.execution_shortfalls) or [view.unavailable_reason]!r}")
    return f"<{type(result).__name__} {' '.join(parts)}>"


def log_coverage(result: object, logger: logging.Logger) -> None:
    """Announce a finished set's coverage on *logger*, at a matching level.

    A complete set logs at INFO; anything else logs at WARNING.  One line
    either way: the full report belongs to whatever renders the result, and
    logging it as well would print the same block twice under the command
    line, which teaches readers to skip it.

    **This channel is a convenience, not the guarantee.**  ``logging``'s
    ``lastResort`` handler normally prints a WARNING to stderr even when the
    application configures no logging -- but not for this package.
    ``lit_search/utils/logging_config.py`` installs a ``_SanitizingHandler``
    on the ``lit-search`` logger at import time, and ``Logger.callHandlers``
    counts it as a handler, so ``lastResort`` never fires for any record
    emitted anywhere under ``lit-search``.  Measured 2026-08-30: the same
    record is silent with that handler installed and printed with it removed.

    The guarantee that a partial set is noticed therefore rests on the
    *renderings* -- :func:`render_result`, :func:`result_repr` and
    :func:`result_summary`, none of which can omit coverage -- and on the
    command line, which configures logging itself.

    Parameters
    ----------
    result : object
        The finished result whose coverage is being announced.
    logger : logging.Logger
        Logger to announce on, normally the calling runner's.

    Returns
    -------
    None
    """
    view = coverage_view(result)
    if view.execution_is_complete:
        logger.info("%s", view.headline())
        return
    logger.warning(
        "Result set is INCOMPLETE: %s. Call render_coverage() for the per-source detail.",
        view.headline(),
    )


def result_summary(result: object, *, headline: str) -> str:
    """Render the short form used by ``__str__``: coverage, never papers.

    Papers are omitted on purpose.  ``str()`` is used in log records and
    f-strings where a hundred-line dump is unwelcome, and a short form that
    listed papers would be a papers-without-coverage path by the back door.
    So the short form shows coverage and a count, and nothing else.

    Parameters
    ----------
    result : object
        Result container.
    headline : str
        One line describing the operation.

    Returns
    -------
    str
        Multi-line summary.
    """
    view = coverage_view(result)
    parts = [coverage_banner(view), headline, f"Papers: {len(_papers_of(result))}"]
    if view.ledger is not None:
        parts.append(view.ledger.render())
    footer = coverage_footer(view, paper_count=len(_papers_of(result)))
    if footer:
        parts.append(footer)
    return "\n".join(parts)
