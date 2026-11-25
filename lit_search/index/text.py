"""What text gets embedded, and which records were scored on less of it.

``O7`` settles which model runs.  It says nothing about what string the model
is handed, and that gap is ``R16``: a record with an empty abstract is scored
on its title alone, which is a materially weaker measurement than a record
scored on title plus abstract -- and a ranking that mixes the two reads as one
uniform measurement.  Empty abstracts are not exotic: Crossref and Web of
Science records frequently lack one, and so does any record whose enrichment
pass partly failed.

So two objects, not one string:

:class:`TextPolicy`
    The declared field set, recorded in ranking provenance.  Changing which
    fields are embedded changes what a score means, so the policy is named
    rather than implied by whatever the code happened to concatenate.

:class:`PaperText`
    The built strings **plus** the list of fields that were absent.  That list
    travels into the index row and back out of every retrieval, so a report can
    say "4 of these 20 were scored on title only" instead of presenting twenty
    equally-weighted rows.

Two strings are built, not one, and the distinction matters for ``D21``:

``search_text``
    Normalised, lower-cased, stripped of punctuation runs.  Fed to the
    embedding model and to the BM25 full-text index, both of which stem and
    drop stop words.
``exact_text``
    The raw fields joined with separators, case preserved.  Matched with a
    pushed-down ``LIKE``, which is the only lane that can answer "does the
    string ``FM-Agent`` literally occur here".  Stemming and stop-word removal
    destroy exactly that, which is why an FTS index is not a substitute.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lit_search.core.paper import Paper

#: Cap on the exact-text column.  Taken from the design reference
#: (``TKasperczyk/claude-memory``, which caps its own exact-text column at
#: 4096 characters), and kept because an unbounded raw column makes the
#: ``LIKE`` scan cost grow with the longest abstract in the corpus.
EXACT_TEXT_MAX_LENGTH = 4096

#: Cap on the searchable text column.
SEARCH_TEXT_MAX_LENGTH = 8192

_WHITESPACE = re.compile(r"\s+")
_NON_SEARCHABLE = re.compile(r"[^0-9a-z\-\s]+")


@dataclass(frozen=True, slots=True)
class TextPolicy:
    """Which paper fields are embedded and indexed.

    Parameters
    ----------
    name : str
        Short label recorded in ranking provenance, for example
        ``"title+abstract+keywords+venue"``.  A ranking whose policy is not
        recorded cannot be compared with one built under a different policy.
    use_abstract : bool
        Whether the abstract is included.
    use_keywords : bool
        Whether keywords are included.
    use_venue : bool
        Whether the publication venue is included.
    use_authors : bool
        Whether author names are included.  Off by default for the searchable
        text -- author names inflate topical similarity between unrelated
        papers by the same group -- but always present in the exact text, which
        is the lane author lookup belongs in.
    """

    name: str = "title+abstract+keywords+venue"
    use_abstract: bool = True
    use_keywords: bool = True
    use_venue: bool = True
    use_authors: bool = False

    def describe(self) -> str:
        """Return the policy label used in provenance records.

        Returns
        -------
        str
            The policy name.
        """
        return self.name


#: The policy used when a caller does not choose one.
DEFAULT_TEXT_POLICY = TextPolicy()


@dataclass(frozen=True, slots=True)
class PaperText:
    """The strings indexed for one paper, and what was missing from them.

    Parameters
    ----------
    search_text : str
        Normalised text for the embedding model and the full-text index.
    exact_text : str
        Raw text, case preserved, for literal substring matching.
    missing_fields : tuple[str, ...]
        Names of policy fields that were absent or empty for this record.
        Non-empty means this record was scored on less text than the policy
        asks for.
    """

    search_text: str
    exact_text: str
    missing_fields: tuple[str, ...] = field(default=())

    @property
    def is_degraded(self) -> bool:
        """Whether this record was scored on less text than the policy asks.

        Returns
        -------
        bool
            ``True`` when any policy field was absent.
        """
        return bool(self.missing_fields)

    @property
    def degradation_label(self) -> str:
        """Return a compact label for the missing fields.

        Returns
        -------
        str
            Comma-separated field names, or ``""`` when nothing was missing.
        """
        return ",".join(self.missing_fields)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, Any]
            Serialized text record.
        """
        return {
            "search_text": self.search_text,
            "exact_text": self.exact_text,
            "missing_fields": list(self.missing_fields),
        }


def normalise_for_search(text: str) -> str:
    """Lower-case *text* and collapse everything that is not a word or a dash.

    Parameters
    ----------
    text : str
        Raw text.

    Returns
    -------
    str
        Normalised text, whitespace collapsed.
    """
    lowered = _NON_SEARCHABLE.sub(" ", text.lower())
    return _WHITESPACE.sub(" ", lowered).strip()


def _venue_of(paper: Paper) -> str:
    """Return a paper's venue title, or ``""`` when it has none.

    Parameters
    ----------
    paper : Paper
        Paper to read.

    Returns
    -------
    str
        Venue title.
    """
    source = getattr(paper, "source", None)
    title = getattr(source, "title", None) if source is not None else None
    return str(title).strip() if title else ""


def _collect_parts(paper: Paper, policy: TextPolicy) -> tuple[list[str], list[str]]:
    """Gather the text parts a policy asks for, and note the absent ones.

    Parameters
    ----------
    paper : Paper
        Paper to read.
    policy : TextPolicy
        Fields to include.

    Returns
    -------
    tuple[list[str], list[str]]
        The present parts, and the names of the absent policy fields.
    """
    parts: list[str] = [paper.title.strip()]
    missing: list[str] = []
    if not parts[0]:  # pragma: no cover - Paper refuses a blank title
        missing.append("title")

    candidates: list[tuple[bool, str, str]] = [
        (policy.use_abstract, "abstract", (paper.abstract or "").strip()),
        (
            policy.use_keywords,
            "keywords",
            " ".join(sorted(word.strip() for word in (paper.keywords or set()) if word.strip())),
        ),
        (policy.use_venue, "venue", _venue_of(paper)),
        (
            policy.use_authors,
            "authors",
            " ".join(
                author.name.strip() for author in (paper.authors or []) if author.name.strip()
            ),
        ),
    ]
    for enabled, name, value in candidates:
        if not enabled:
            continue
        if value:
            parts.append(value)
        else:
            missing.append(name)
    return [part for part in parts if part], missing


def build_paper_text(paper: Paper, policy: TextPolicy = DEFAULT_TEXT_POLICY) -> PaperText:
    """Build the indexed strings for one paper under *policy*.

    The exact text always includes the identifiers and author names regardless
    of policy, because the literal lane exists precisely to answer "does this
    exact author name / arXiv identifier / phrase occur", and a policy that
    excluded them from the searchable text should not also make them
    unfindable.

    Parameters
    ----------
    paper : Paper
        Paper to index.
    policy : TextPolicy
        Which fields to include in the searchable text.

    Returns
    -------
    PaperText
        Searchable text, exact text, and the absent policy fields.
    """
    parts, missing = _collect_parts(paper, policy)
    search_text = normalise_for_search(" ".join(parts))[:SEARCH_TEXT_MAX_LENGTH]

    exact_parts: list[str] = [paper.title.strip()]
    if paper.doi:
        exact_parts.append(paper.doi)
    if paper.preprint_doi:
        exact_parts.append(paper.preprint_doi)
    exact_parts.extend(
        author.name.strip() for author in (paper.authors or []) if author.name.strip()
    )
    venue = _venue_of(paper)
    if venue:
        exact_parts.append(venue)
    exact_parts.extend(sorted(word.strip() for word in (paper.keywords or set()) if word.strip()))
    if policy.use_abstract and (paper.abstract or "").strip():
        exact_parts.append(paper.abstract.strip())
    exact_text = _WHITESPACE.sub(" ", " | ".join(exact_parts)).strip()[:EXACT_TEXT_MAX_LENGTH]

    return PaperText(
        search_text=search_text,
        exact_text=exact_text,
        missing_fields=tuple(missing),
    )


__all__ = [
    "DEFAULT_TEXT_POLICY",
    "EXACT_TEXT_MAX_LENGTH",
    "SEARCH_TEXT_MAX_LENGTH",
    "PaperText",
    "TextPolicy",
    "build_paper_text",
    "normalise_for_search",
]
