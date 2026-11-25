"""Coverage-bearing result of an exact identifier lookup."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lit_search.core.coverage import CoverageLedger, CoverageStatus
from lit_search.core.paper import Paper
from lit_search.core.rendering import coverage_view, render_coverage_report, render_result


@dataclass
class LookupResult:
    """A single-paper lookup whose absence is qualified by source evidence.

    Parameters
    ----------
    identifier : str
        What was looked up.
    papers : list[Paper]
        The single paper found, or an empty list.
    """

    identifier: str
    papers: list[Paper] = field(default_factory=list)
    coverage: CoverageLedger = field(default_factory=CoverageLedger)

    def __post_init__(self) -> None:
        """Reject an impossible multi-paper exact lookup result."""
        if len(self.papers) > 1:
            raise ValueError("LookupResult may contain at most one paper")

    @property
    def paper(self) -> Paper | None:
        """Return the found paper while the ledger remains on this result."""
        return self.papers[0] if self.papers else None

    @property
    def coverage_status(self) -> CoverageStatus:
        """Return the worst per-source lookup status."""
        return self.coverage.status

    @property
    def is_complete(self) -> bool:
        """Return whether every requested source has a complete outcome."""
        return self.coverage.is_complete

    def headline(self) -> str:
        """Return the one-line description of the lookup.

        Returns
        -------
        str
            Identifier and outcome.
        """
        found = "1 record" if self.papers else "no record"
        return f"get {self.identifier} -- {found}"

    def render_coverage(self) -> str:
        """Render per-source lookup coverage."""
        return render_coverage_report(coverage_view(self))

    def render(self, *, limit: int | None = 10) -> str:
        """Render the lookup, paper, and coverage warning when incomplete."""
        return render_result(self, headline=self.headline(), limit=limit)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-ready dictionary.

        Returns
        -------
        dict[str, Any]
            Identifier and papers; the ``"type"`` discriminator is added by
            the persistence layer.
        """
        return {
            "identifier": self.identifier,
            "papers": [paper.to_dict() for paper in self.papers],
            "coverage": self.coverage.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LookupResult:
        """Rebuild a lookup result from its serialized form.

        Parameters
        ----------
        data : dict[str, Any]
            Payload produced by :meth:`to_dict`.

        Returns
        -------
        LookupResult
            The reconstructed lookup.
        """
        return cls(
            identifier=str(data.get("identifier", "")),
            papers=[Paper.from_dict(p) for p in data.get("papers", [])],
            coverage=CoverageLedger.from_dict(data.get("coverage", {})),
        )
