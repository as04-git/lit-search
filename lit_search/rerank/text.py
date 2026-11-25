"""What string gets scored, and what to record when it is incomplete.

A re-ranker does not score papers.  It scores *text built from* papers, and
the build step is a decision with the same silent-failure shape as the model
choice one level above it: a record scored on a bare title sinks in the
ranking for a reason that has nothing to do with its relevance, and nothing
in the ordering shows it.

Crossref and Web of Science records routinely arrive without abstracts, and
any record arrives without one when enrichment partly failed.  So this is not
an edge case to be handled defensively -- it is the common case, and the
policy has to name it.

The policy
----------

Title, abstract, and keywords, each on its own labelled line, empty fields
omitted:

    Title: Scaling Formal Methods to Large Systems ...
    Abstract: We derive a function's specification from its call sites ...
    Keywords: model checking; specification inference

Three decisions are worth their reasons.

**Keywords are included.**  They cost almost nothing and they carry the one
thing the motivating failure turned on.  That failure was terminological: the
same idea was published in formal-methods vocabulary while the searches were
organised around program-synthesis vocabulary.  Author-supplied keywords are
often exactly where a neighbouring community's terms appear, so they are the
cheapest available bridge across the boundary that caused the miss.

**Fields are labelled rather than concatenated.**  The chosen bi-encoder pools
on the last token, so the text is read as a whole rather than averaged, and an
explicit ``Abstract:`` marker keeps a title from running into an abstract as
one ungrammatical sentence.

**Title comes first.**  Truncation takes from the end, so the field that is
always present and always relevant is the field that survives.

Degradation is a recorded fact, not a fallback
----------------------------------------------

Every built document carries a :class:`TextQuality`.  Anything short of
:attr:`TextQuality.FULL` is reported through to
:attr:`~lit_search.store.order.RankingProvenance.degraded_text`, so a low rank
is explainable after the fact.  A record with no usable text at all is still
scored -- dropping it would silently shrink the set, and the whole point of a
re-rank is that it reorders a set without changing what is in it -- but it is
named.

An abstract shorter than :data:`MIN_USABLE_ABSTRACT_CHARS` is treated as
absent.  Sources emit stubs ("Abstract not available.", "No abstract", a bare
copyright line), and counting a stub as a real abstract is the silent case:
the record then looks fully scored and is not.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from lit_search.core.identity import source_record_keys, version_identity_key
from lit_search.core.paper import Paper

#: Identifier of this policy, recorded in every ranking provenance.  Bump the
#: version suffix when the built string changes, because rankings produced
#: under two different policies are not comparable and nothing else in the
#: record would show it.
TEXT_POLICY_ID = "title+abstract+keywords/labelled-lines/v1"

#: Shortest abstract that is treated as an abstract.  Below this the field is
#: a stub rather than content, and calling it an abstract would mark a
#: degraded record as fully scored.
MIN_USABLE_ABSTRACT_CHARS = 40

#: Keyword count cap.  Long auto-generated keyword lists otherwise crowd out
#: the abstract under truncation, which would trade a real signal for a weak
#: one.
MAX_KEYWORDS = 12


class TextQuality(StrEnum):
    """How complete the text built for a record was.

    The distinction between "this record was fully scored" and "this record
    was scored on a fragment" is the whole reason this enum exists.  Both
    produce a number; only one of them produces a number that means what the
    ranking claims it means.
    """

    #: Title and a usable abstract were both present.
    FULL = "full"
    #: Title present, abstract absent or a stub.  Scored on title and any
    #: keywords.
    NO_ABSTRACT = "no_abstract"
    #: Abstract present, title absent.  Unusual, and worth seeing.
    NO_TITLE = "no_title"
    #: Neither title nor abstract.  Scored, and named, because dropping it
    #: would shrink the set silently.
    EMPTY = "empty"


@dataclass(frozen=True)
class DocumentText:
    """The text built for one record, with the identity to attribute it to.

    Attributes
    ----------
    identity : str
        Operation-local identity of this record occurrence.  Explicit DOI,
        arXiv or provider-native keys are preferred.  Identifier-free records
        use a deterministic snapshot digest plus their input position; a title
        is never treated as record identity.  Used both to attribute
        degradation and as the deterministic tie-break key when two records
        score equally.
    text : str
        The string that will be embedded or paired.
    quality : TextQuality
        How complete the source record was.
    """

    identity: str
    text: str
    quality: TextQuality

    @property
    def is_degraded(self) -> bool:
        """Report whether this text is short of what the policy asks for.

        Returns
        -------
        bool
            ``True`` for anything other than :attr:`TextQuality.FULL`.
        """
        return self.quality is not TextQuality.FULL


def paper_identity(paper: Paper, position: int) -> str:
    """Return an operation-local identity for one record occurrence.

    Prefers an explicit publication-version identity, then a provider-native
    source-record identity.  An identifier-free record is named by a digest of
    its complete serialized snapshot plus its input position.  The position is
    intentional: two identifier-free records are not silently declared equal
    merely because their snapshots -- or, especially, their titles -- happen
    to match.

    Parameters
    ----------
    paper : Paper
        Record to identify.
    position : int
        Index of the record in its set, used only for the placeholder.

    Returns
    -------
    str
        Identity string.
    """
    version_key = version_identity_key(paper)
    if version_key is not None:
        return version_key
    provider_keys = sorted(source_record_keys(paper))
    if provider_keys:
        return provider_keys[0]
    snapshot = json.dumps(
        paper.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(snapshot).hexdigest()[:20]
    return f"snapshot:{digest}@position:{position}"


def _clean(value: str | None) -> str:
    """Collapse a possibly-absent field to a stripped string.

    Parameters
    ----------
    value : str | None
        Raw field value.

    Returns
    -------
    str
        Stripped value, or the empty string when the field is absent or holds
        only whitespace.
    """
    return (value or "").strip()


def _usable_abstract(paper: Paper) -> str:
    """Return the abstract when it is long enough to be one.

    Parameters
    ----------
    paper : Paper
        Record to read.

    Returns
    -------
    str
        The abstract, or the empty string when it is absent or a stub.
    """
    abstract = _clean(paper.abstract)
    if len(abstract) < MIN_USABLE_ABSTRACT_CHARS:
        return ""
    return abstract


def _keyword_line(paper: Paper) -> str:
    """Return the keyword line, or the empty string when there is none.

    Keywords are sorted so that the built text does not depend on set
    iteration order; an unstable input string would make scores irreproducible
    across runs for no reason.

    Parameters
    ----------
    paper : Paper
        Record to read.

    Returns
    -------
    str
        Semicolon-separated keywords, capped at :data:`MAX_KEYWORDS`.
    """
    keywords = sorted({_clean(k) for k in (paper.keywords or set())} - {""})
    if not keywords:
        return ""
    return "; ".join(keywords[:MAX_KEYWORDS])


def _classify(title: str, abstract: str) -> TextQuality:
    """Decide the quality label for a title/abstract pair.

    Parameters
    ----------
    title : str
        Cleaned title.
    abstract : str
        Cleaned, length-checked abstract.

    Returns
    -------
    TextQuality
        The label.
    """
    if title and abstract:
        return TextQuality.FULL
    if title:
        return TextQuality.NO_ABSTRACT
    if abstract:
        return TextQuality.NO_TITLE
    return TextQuality.EMPTY


def build_document_text(paper: Paper, position: int = 0) -> DocumentText:
    """Build the scoring text for one record under the policy above.

    Parameters
    ----------
    paper : Paper
        Record to render.
    position : int
        Index of the record in its set, used only to name a record that has
        neither DOI nor title.

    Returns
    -------
    DocumentText
        The built text, its identity, and its quality label.
    """
    title = _clean(paper.title)
    abstract = _usable_abstract(paper)
    keywords = _keyword_line(paper)

    lines: list[str] = []
    if title:
        lines.append(f"Title: {title}")
    if abstract:
        lines.append(f"Abstract: {abstract}")
    if keywords:
        lines.append(f"Keywords: {keywords}")

    quality = _classify(title, abstract)
    if not lines:
        # A model handed an empty string returns an arbitrary vector, and the
        # record would then be ranked by an artifact.  Say what the record is
        # instead, so the score at least reflects the only thing known about
        # it, and let the quality label carry the warning.
        lines.append("Title: (no title or abstract available for this record)")

    return DocumentText(
        identity=paper_identity(paper, position),
        text="\n".join(lines),
        quality=quality,
    )


def build_documents(papers: Sequence[Paper]) -> tuple[DocumentText, ...]:
    """Build scoring text for a whole set, preserving order.

    Parameters
    ----------
    papers : Sequence[Paper]
        Records to render.

    Returns
    -------
    tuple[DocumentText, ...]
        One entry per input record, in input order.
    """
    documents: list[DocumentText] = []
    occurrences: dict[str, int] = {}
    for position, paper in enumerate(papers):
        document = build_document_text(paper, position)
        occurrence = occurrences.get(document.identity, 0)
        occurrences[document.identity] = occurrence + 1
        if occurrence:
            # A handle should not normally contain two records carrying the
            # same explicit identifier, but re-ranking must remain a
            # permutation even when it does.  Keep both occurrences rather
            # than letting a dictionary silently overwrite the first.
            document = replace(
                document,
                identity=f"{document.identity}@occurrence:{occurrence}",
            )
        documents.append(document)
    return tuple(documents)


def degraded_identities(documents: Iterable[DocumentText]) -> tuple[str, ...]:
    """List the identities of records whose text fell short of the policy.

    Parameters
    ----------
    documents : Iterable[DocumentText]
        Built documents.

    Returns
    -------
    tuple[str, ...]
        Identities of the degraded records, in input order.
    """
    return tuple(document.identity for document in documents if document.is_degraded)


def quality_counts(documents: Iterable[DocumentText]) -> dict[str, int]:
    """Count documents by quality label, for reporting.

    Parameters
    ----------
    documents : Iterable[DocumentText]
        Built documents.

    Returns
    -------
    dict[str, int]
        Label value to count, including zero-valued labels so that a caller
        rendering the report does not have to know the label set.
    """
    counts = {quality.value: 0 for quality in TextQuality}
    for document in documents:
        counts[document.quality.value] += 1
    return counts
