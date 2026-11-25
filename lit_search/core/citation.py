"""Typed citation-graph identities and provenance-bearing edges.

The original snowball path reduced every citation target to a DOI string.  A
DOI is optional in both Semantic Scholar and OpenAlex, so that representation
discarded graph edges before traversal even began.  These types keep the
identifiers a graph source actually supplied and record which source asserted
the edge.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lit_search.core.paper import Paper

from lit_search.utils.normalization import normalize_doi


class IdentifierKind(StrEnum):
    """Identifier namespaces understood by native snowball routing."""

    DOI = "doi"
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    OPENALEX = "openalex"


@dataclass(frozen=True, slots=True)
class PaperIdentifier:
    """One typed identifier for a paper."""

    kind: IdentifierKind
    value: str

    def __post_init__(self) -> None:
        value = self.value.strip()
        if not value:
            raise ValueError("PaperIdentifier.value must not be blank")
        if self.kind is IdentifierKind.DOI:
            value = normalize_doi(value) or value
        elif self.kind is IdentifierKind.ARXIV:
            value = value.removeprefix("ARXIV:").removeprefix("arXiv:")
            if value.lower().startswith("10.48550/arxiv."):
                value = value[len("10.48550/arxiv.") :]
        elif self.kind is IdentifierKind.OPENALEX:
            value = value.rsplit("/", 1)[-1]
        object.__setattr__(self, "value", value)

    @property
    def key(self) -> str:
        """Return a namespace-qualified, case-insensitive identity key."""
        return f"{self.kind.value}:{self.value.lower()}"

    def to_dict(self) -> dict[str, str]:
        """Serialize this identifier."""
        return {"kind": self.kind.value, "value": self.value}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaperIdentifier:
        """Reconstruct an identifier from :meth:`to_dict`."""
        return cls(kind=IdentifierKind(data["kind"]), value=str(data["value"]))


@dataclass(frozen=True, slots=True)
class CitationTarget:
    """A citation target carrying every identifier supplied for it."""

    identifiers: tuple[PaperIdentifier, ...]
    title: str | None = None

    def __post_init__(self) -> None:
        if not self.identifiers:
            raise ValueError("CitationTarget requires at least one typed identifier")
        unique: dict[str, PaperIdentifier] = {}
        for identifier in self.identifiers:
            unique.setdefault(identifier.key, identifier)
        object.__setattr__(self, "identifiers", tuple(unique.values()))

    @classmethod
    def from_values(
        cls,
        *,
        doi: str | None = None,
        arxiv_id: str | None = None,
        semantic_scholar_id: str | None = None,
        openalex_id: str | None = None,
        title: str | None = None,
    ) -> CitationTarget | None:
        """Build a target from the non-empty identifiers supplied."""
        pairs = (
            (IdentifierKind.SEMANTIC_SCHOLAR, semantic_scholar_id),
            (IdentifierKind.OPENALEX, openalex_id),
            (IdentifierKind.ARXIV, arxiv_id),
            (IdentifierKind.DOI, doi),
        )
        identifiers = tuple(PaperIdentifier(kind, value) for kind, value in pairs if value)
        return cls(identifiers=identifiers, title=title) if identifiers else None

    @property
    def keys(self) -> frozenset[str]:
        """Return every namespace-qualified alias for visited-set matching."""
        return frozenset(identifier.key for identifier in self.identifiers)

    def get(self, kind: IdentifierKind) -> str | None:
        """Return the value in *kind*, if present."""
        return next((item.value for item in self.identifiers if item.kind is kind), None)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this target."""
        return {
            "identifiers": [identifier.to_dict() for identifier in self.identifiers],
            "title": self.title,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CitationTarget:
        """Reconstruct a target from :meth:`to_dict`."""
        return cls(
            identifiers=tuple(PaperIdentifier.from_dict(item) for item in data["identifiers"]),
            title=data.get("title"),
        )


@dataclass(frozen=True, slots=True)
class CitationEdge:
    """A source assertion that one paper cites another target."""

    target: CitationTarget
    source: str

    def __post_init__(self) -> None:
        source = self.source.strip().lower()
        if not source:
            raise ValueError("CitationEdge.source must not be blank")
        object.__setattr__(self, "source", source)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this edge."""
        return {"source": self.source, "target": self.target.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CitationEdge:
        """Reconstruct an edge from :meth:`to_dict`."""
        return cls(source=str(data["source"]), target=CitationTarget.from_dict(data["target"]))


@dataclass(frozen=True, slots=True)
class NativeBatchResult:
    """Outcome of one or more source-native exact-identifier batches.

    ``requested`` counts request *positions*, not distinct scholarly works.
    Consequently, when two requested aliases resolve to the same work, a
    connector must either deduplicate the request before issuing it or retain
    two resolved positions in :attr:`papers`.  This positional contract makes
    every requested identifier terminally accountable: it resolved, was not
    found, could not be parsed, or failed at request time.
    """

    source: str
    papers: tuple[Paper, ...]
    requested: int
    not_found: int = 0
    parse_failures: int = 0
    truncated_edge_records: int = 0
    request_failures: int = 0
    edge_parse_failures: int = 0
    edge_request_failures: int = 0
    error_messages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject outcomes that leave requested identifiers unaccounted for."""
        counters = {
            "requested": self.requested,
            "not_found": self.not_found,
            "parse_failures": self.parse_failures,
            "truncated_edge_records": self.truncated_edge_records,
            "request_failures": self.request_failures,
            "edge_parse_failures": self.edge_parse_failures,
            "edge_request_failures": self.edge_request_failures,
        }
        negative = [name for name, value in counters.items() if value < 0]
        if negative:
            raise ValueError(
                "NativeBatchResult counters must be non-negative: " + ", ".join(negative)
            )

        terminal_total = (
            len(self.papers) + self.not_found + self.parse_failures + self.request_failures
        )
        if self.requested != terminal_total:
            raise ValueError(
                "NativeBatchResult must account for every requested position: "
                f"requested={self.requested}, resolved={len(self.papers)}, "
                f"not_found={self.not_found}, parse_failures={self.parse_failures}, "
                f"request_failures={self.request_failures}; terminal total={terminal_total}. "
                "Alias duplicates are positional: deduplicate before requesting or return "
                "one resolved paper position per requested alias."
            )

    @property
    def complete(self) -> bool:
        """Whether every request completed and no returned graph was truncated."""
        return (
            self.parse_failures == 0
            and self.truncated_edge_records == 0
            and self.request_failures == 0
            and self.edge_parse_failures == 0
            and self.edge_request_failures == 0
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize a replayable, coverage-bearing runner operation outcome."""
        return {
            "source": self.source,
            "papers": [paper.to_dict() for paper in self.papers],
            "requested": self.requested,
            "not_found": self.not_found,
            "parse_failures": self.parse_failures,
            "truncated_edge_records": self.truncated_edge_records,
            "request_failures": self.request_failures,
            "edge_parse_failures": self.edge_parse_failures,
            "edge_request_failures": self.edge_request_failures,
            "error_messages": list(self.error_messages),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NativeBatchResult:
        """Restore an outcome persisted by :meth:`to_dict`."""
        from lit_search.core.paper import Paper

        return cls(
            source=str(data["source"]),
            papers=tuple(Paper.from_dict(item) for item in data.get("papers", [])),
            requested=int(data["requested"]),
            not_found=int(data.get("not_found", 0)),
            parse_failures=int(data.get("parse_failures", 0)),
            truncated_edge_records=int(data.get("truncated_edge_records", 0)),
            request_failures=int(data.get("request_failures", 0)),
            edge_parse_failures=int(data.get("edge_parse_failures", 0)),
            edge_request_failures=int(data.get("edge_request_failures", 0)),
            error_messages=tuple(str(item) for item in data.get("error_messages", [])),
        )


def merge_identifiers(
    first: list[PaperIdentifier], second: list[PaperIdentifier]
) -> list[PaperIdentifier]:
    """Union two identifier lists while preserving first-seen order."""
    merged: dict[str, PaperIdentifier] = {identifier.key: identifier for identifier in first}
    for identifier in second:
        merged.setdefault(identifier.key, identifier)
    return list(merged.values())


def merge_edges(first: list[CitationEdge], second: list[CitationEdge]) -> list[CitationEdge]:
    """Union provenance-bearing edges without collapsing assertions by source."""
    merged: dict[tuple[str, tuple[str, ...]], CitationEdge] = {}
    for edge in [*first, *second]:
        key = (edge.source, tuple(sorted(edge.target.keys)))
        merged.setdefault(key, edge)
    return list(merged.values())
