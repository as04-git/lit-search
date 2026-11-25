"""Conservative, provenance-honest record deduplication.

Record identity may collapse two records only when an explicit DOI, arXiv
version, or provider-native record identifier establishes sameness.  Version
identity may link two retained records only when metadata explicitly declares
the relationship through ``preprint_doi``.  Title/year and semantic similarity
are candidate evidence: they never remove a member and never become a
:class:`VersionLink` by themselves.

Similarity is not identity.  The semantic report therefore says how many pairs
crossed the configured threshold while leaving them unresolved and retaining
every record.  When no semantic model is supplied, it says the residue was not
examined rather than reporting it clean.  No weak relation participates in
transitive closure.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from lit_search.core.identity import record_identity_keys, same_record_identity
from lit_search.core.paper import Paper, identity_key
from lit_search.exceptions import InvalidParameterError
from lit_search.store.lineage import Operand, Operation, OperationName
from lit_search.store.propagation import (
    BASE_ROLE,
    CoverageOperand,
    CoveragePropagator,
    LedgerCoveragePropagator,
)
from lit_search.store.record import HandleRecord
from lit_search.store.store import HandleStore
from lit_search.utils.normalization import is_preprint_doi

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lit_search.index.embedding import EmbeddingModel

#: Cosine similarity at or above which a residue pair is reported as a
#: candidate relationship.  Crossing the threshold never changes membership.
DEFAULT_SEMANTIC_THRESHOLD = 0.92

#: Deprecated compatibility argument.  Title/year evidence is never used to
#: collapse or link records; callers may still pass the former option.
DEFAULT_YEAR_SLACK = 1

_PUNCTUATION = re.compile(r"[^0-9a-z]+")


class MatchMethod(StrEnum):
    """How two records were judged to be related.

    Attributes
    ----------
    EXACT_RECORD
        Same explicit publication-version or provider-native record identity.
    PREPRINT_LINK
        One record's ``preprint_doi`` is the other's ``doi``.  An explicit
        version relationship the metadata itself carries.
    EXACT_DOI, EXACT_TITLE, SEMANTIC
        Legacy values retained so old serialized reports remain intelligible.
        New plans do not emit them.
    """

    EXACT_RECORD = "exact:record"
    EXACT_DOI = "exact:doi"
    PREPRINT_LINK = "exact:preprint-link"
    EXACT_TITLE = "exact:title"
    SEMANTIC = "semantic:cosine"


@dataclass(frozen=True, slots=True)
class DuplicateGroup:
    """Records collapsed into one, and the evidence for doing it.

    Parameters
    ----------
    kept_key : str
        Identity of the surviving record.
    collapsed_keys : tuple[str, ...]
        Identities of the records removed.
    method : MatchMethod
        How the match was made.
    evidence : str
        Human-readable justification.
    """

    kept_key: str
    collapsed_keys: tuple[str, ...]
    method: MatchMethod
    evidence: str

    def describe(self) -> str:
        """Return a one-line human-readable summary.

        Returns
        -------
        str
            Summary line.
        """
        return f"kept {self.kept_key}, collapsed {list(self.collapsed_keys)} [{self.evidence}]"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, Any]
            Serialized group.
        """
        return {
            "kept_key": self.kept_key,
            "collapsed_keys": list(self.collapsed_keys),
            "method": self.method.value,
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class VersionPair:
    """One detected version relationship between two records in a set.

    An intermediate produced by each stage and consumed by the link builder, so
    that two stages detecting the same relationship contribute to one link
    instead of producing two overlapping ones.

    Parameters
    ----------
    left : int
        Index of the first record in set order.
    right : int
        Index of the second.
    method : MatchMethod
        How the relationship was detected.
    basis : str
        Human-readable justification.
    """

    left: int
    right: int
    method: MatchMethod
    basis: str


@dataclass(frozen=True, slots=True)
class VersionLink:
    """Two or more records that are versions of one work, all kept.

    Parameters
    ----------
    member_keys : tuple[str, ...]
        Identities of every version, in the order they appeared in the set.
    read_target : str
        Identity of the version a reading list should point at.
    cite_target : str
        Identity of the version an export should cite.
    method : MatchMethod
        How the relationship was established.
    basis : str
        Human-readable justification.
    divergence : str | None
        Present exactly when the read and cite targets differ, and states what
        the difference is.  A divergence is information, not a problem to
        resolve: it means the best version to read is not the one the field
        cites, and a reader needs to know that rather than have it decided for
        them.
    """

    member_keys: tuple[str, ...]
    read_target: str
    cite_target: str
    methods: tuple[MatchMethod, ...]
    basis: str
    divergence: str | None = None

    def describe(self) -> str:
        """Return a one-line human-readable summary.

        Returns
        -------
        str
            Summary line, naming the divergence when there is one.
        """
        found_by = "+".join(method.value for method in self.methods)
        head = f"versions {list(self.member_keys)} [{found_by}: {self.basis}]"
        if self.divergence is None:
            return f"{head}; read and cite both {self.read_target}"
        return f"{head}; READ {self.read_target} but CITE {self.cite_target} -- {self.divergence}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, Any]
            Serialized link.
        """
        return {
            "member_keys": list(self.member_keys),
            "read_target": self.read_target,
            "cite_target": self.cite_target,
            "methods": [method.value for method in self.methods],
            "basis": self.basis,
            "divergence": self.divergence,
        }


@dataclass(frozen=True, slots=True)
class SemanticStageReport:
    """Whether the second stage ran, and over how much.

    Parameters
    ----------
    ran : bool
        Whether the semantic pass executed.
    reason : str
        Why it did not run, when it did not.  Required in that case.
    residue_size : int
        How many records the exact stage left undecided.
    threshold : float
        Similarity at or above which a pair was judged the same work.
    pairs_matched : int
        How many residue pairs crossed the similarity threshold.  These are
        unresolved candidates, not duplicates or proven version links.
    model : str | None
        Identity of the model used, when one was.

    Raises
    ------
    ValueError
        If the stage did not run and no reason was given.
    """

    ran: bool
    residue_size: int
    threshold: float
    pairs_matched: int = 0
    reason: str = ""
    model: str | None = None

    def __post_init__(self) -> None:
        """Reject a skipped stage that does not say why it was skipped.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the stage did not run and carries no reason.
        """
        if not self.ran and not self.reason:
            raise ValueError(
                "A semantic deduplication stage that did not run must say why: "
                "'no further duplicates' and 'nobody looked' are opposite claims"
            )

    def describe(self) -> str:
        """Return a one-line human-readable summary.

        Returns
        -------
        str
            Summary line.
        """
        if not self.ran:
            return (
                f"semantic stage DID NOT RUN over {self.residue_size} unmatched record(s) "
                f"-- {self.reason}; those records are unexamined, not clean"
            )
        return (
            f"semantic stage examined {self.residue_size} record(s) at cosine >= "
            f"{self.threshold} with {self.model}: {self.pairs_matched} unresolved "
            "candidate pair(s); candidates were retained and not called versions"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, Any]
            Serialized report.
        """
        return {
            "ran": self.ran,
            "residue_size": self.residue_size,
            "threshold": self.threshold,
            "pairs_matched": self.pairs_matched,
            "reason": self.reason,
            "model": self.model,
        }


@dataclass(frozen=True, slots=True)
class DeduplicationPlan:
    """The finished deduplication: what survives, what collapsed, what links.

    Parameters
    ----------
    members : tuple[Paper, ...]
        The surviving members, in the base set's order.  Version-linked
        records are **all** present here: linking keeps both.
    submitted : int
        How many records were considered.
    input_digest : str
        SHA-256 commitment to the ordered input snapshots.  This prevents a
        plan from being attached to an unrelated same-sized handle.
    duplicate_groups : tuple[DuplicateGroup, ...]
        Collapses performed.
    version_links : tuple[VersionLink, ...]
        Version relationships recorded, with nothing removed.
    semantic_stage : SemanticStageReport
        Whether the second stage ran.
    """

    members: tuple[Paper, ...]
    submitted: int
    input_digest: str
    duplicate_groups: tuple[DuplicateGroup, ...] = ()
    version_links: tuple[VersionLink, ...] = ()
    semantic_stage: SemanticStageReport = field(
        default_factory=lambda: SemanticStageReport(
            ran=False,
            residue_size=0,
            threshold=DEFAULT_SEMANTIC_THRESHOLD,
            reason="no records reached the residue",
        )
    )

    @property
    def collapsed_count(self) -> int:
        """Return how many records were removed.

        Returns
        -------
        int
            Count of collapsed records.
        """
        return sum(len(group.collapsed_keys) for group in self.duplicate_groups)

    @property
    def divergent_links(self) -> tuple[VersionLink, ...]:
        """Return the version links whose read and cite targets differ.

        Returns
        -------
        tuple[VersionLink, ...]
            Divergent links.
        """
        return tuple(link for link in self.version_links if link.divergence is not None)

    def strategy(self) -> str:
        """Return the strategy string recorded on the handle.

        Self-describing on purpose: a handle whose strategy reads
        ``"deduplicated"`` tells a later reader nothing about whether a
        semantic pass ran.

        Returns
        -------
        str
            Strategy description.
        """
        stage = "semantic:candidates" if self.semantic_stage.ran else "semantic:SKIPPED"
        return (
            "collapse(explicit-record-identity); link(explicit-preprint_doi); "
            f"{stage}(cosine>={self.semantic_stage.threshold}, non-destructive)"
        )

    def describe(self) -> str:
        """Return a multi-line human-readable report.

        Returns
        -------
        str
            Report naming the collapses, the links, and the divergences.
        """
        lines = [
            f"{len(self.members)} of {self.submitted} record(s) survive; "
            f"{self.collapsed_count} collapsed, {len(self.version_links)} version link(s)",
            self.semantic_stage.describe(),
        ]
        lines.extend(f"  collapse: {group.describe()}" for group in self.duplicate_groups)
        lines.extend(f"  link: {link.describe()}" for link in self.version_links)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, Any]
            Serialized plan, without the papers themselves.
        """
        return {
            "submitted": self.submitted,
            "input_digest": self.input_digest,
            "survived": len(self.members),
            "collapsed": self.collapsed_count,
            "duplicate_groups": [group.to_dict() for group in self.duplicate_groups],
            "version_links": [link.to_dict() for link in self.version_links],
            "semantic_stage": self.semantic_stage.to_dict(),
        }


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------


def normalise_title(title: str) -> str:
    """Reduce a title to a comparison key.

    Parameters
    ----------
    title : str
        Raw title.

    Returns
    -------
    str
        Lower-cased title with punctuation collapsed to single spaces.
    """
    return _PUNCTUATION.sub(" ", title.lower()).strip()


def _paper_snapshot_digest(paper: Paper) -> str:
    """Return an exact canonical digest of one paper snapshot."""
    payload = json.dumps(
        paper.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _ordered_input_digest(papers: Sequence[Paper]) -> str:
    """Commit to ordered input membership and every persisted field."""
    digests = [_paper_snapshot_digest(paper) for paper in papers]
    payload = json.dumps(digests, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _diagnostic_record_key(paper: Paper) -> str:
    """Return an explicit identity key or a non-fuzzy snapshot diagnostic."""
    explicit = sorted(record_identity_keys(paper))
    if explicit:
        return "+".join(explicit)
    return f"snapshot:{_paper_snapshot_digest(paper)[:16]}"


def _normalise_doi(doi: str | None) -> str | None:
    """Return a lower-cased bare DOI, or ``None``.

    Parameters
    ----------
    doi : str | None
        Raw DOI.

    Returns
    -------
    str | None
        Normalised DOI.
    """
    if not doi:
        return None
    value = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break
    return value or None


def _completeness(paper: Paper) -> tuple[int, int]:
    """Return a sort key for "most complete version".

    Parameters
    ----------
    paper : Paper
        Record to score.

    Returns
    -------
    tuple[int, int]
        Abstract length and publication ordinal, both larger-is-later.
    """
    ordinal = paper.publication_date.toordinal() if paper.publication_date is not None else 0
    return (len(paper.abstract or ""), ordinal)


def _choose_targets(members: Sequence[Paper]) -> tuple[str, str, str | None]:
    """Pick the read target and the cite target for a version group.

    Read target: the most complete version, by abstract length then date.  Cite
    target: the version of record, meaning the one whose DOI is not a known
    preprint DOI; when every version is a preprint, the preprint is also the
    citation target, because citing nothing is not an option.

    Parameters
    ----------
    members : Sequence[Paper]
        Versions of one work.

    Returns
    -------
    tuple[str, str, str | None]
        Read target key, cite target key, and the divergence description or
        ``None``.
    """
    read = max(members, key=_completeness)
    published = [paper for paper in members if not is_preprint_doi(paper.doi)]
    cite = published[0] if published else members[0]
    read_key, cite_key = identity_key(read), identity_key(cite)
    if read_key == cite_key:
        return read_key, cite_key, None
    divergence = (
        "the most complete version is not the version of record, so a reading list and "
        "a bibliography point at different records"
    )
    return read_key, cite_key, divergence


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def _exact_record_stage(
    papers: Sequence[Paper],
) -> tuple[dict[int, int], list[DuplicateGroup]]:
    """Collapse records sharing an explicit, non-conflicting record identity.

    Parameters
    ----------
    papers : Sequence[Paper]
        Records in set order.

    Returns
    -------
    tuple[dict[int, int], list[DuplicateGroup]]
        Map from collapsed index to surviving index, and the groups formed.
    """
    collapsed: dict[int, int] = {}
    collapsed_by_survivor: dict[int, list[int]] = {}
    survivors: list[int] = []
    for index, paper in enumerate(papers):
        keeper = next(
            (
                candidate
                for candidate in survivors
                if same_record_identity(papers[candidate], paper)
            ),
            None,
        )
        if keeper is None:
            survivors.append(index)
            continue
        collapsed[index] = keeper
        collapsed_by_survivor.setdefault(keeper, []).append(index)

    def evidence(keeper: int, others: Sequence[int]) -> str:
        keeper_keys = record_identity_keys(papers[keeper])
        pairwise: list[tuple[int, list[str]]] = []
        for other in others:
            shared = sorted(keeper_keys.intersection(record_identity_keys(papers[other])))
            if shared:
                pairwise.append((other, shared))
        if pairwise:
            identities = sorted({key for _, keys in pairwise for key in keys})
            return f"pairwise shared explicit record identity {identities!r}"
        return "the same in-memory record object was submitted more than once"

    groups = [
        DuplicateGroup(
            kept_key=_diagnostic_record_key(papers[keeper]),
            collapsed_keys=tuple(_diagnostic_record_key(papers[other]) for other in others),
            method=MatchMethod.EXACT_RECORD,
            evidence=evidence(keeper, others),
        )
        for keeper, others in collapsed_by_survivor.items()
    ]
    return collapsed, groups


def _preprint_link_stage(
    papers: Sequence[Paper], collapsed: Mapping[int, int]
) -> list[VersionPair]:
    """Find version pairs the metadata states outright.

    A record carrying both a ``doi`` and a ``preprint_doi`` names its own other
    identity.  When the other identity is present in the set as its own record,
    that is a version relationship with no inference involved.

    Parameters
    ----------
    papers : Sequence[Paper]
        Records in set order.
    collapsed : Mapping[int, int]
        Indices already collapsed by an earlier stage.

    Returns
    -------
    list[VersionPair]
        Detected pairs.
    """
    by_doi: dict[str, int] = {}
    for index, paper in enumerate(papers):
        if index in collapsed:
            continue
        doi = _normalise_doi(paper.doi)
        if doi is not None:
            by_doi.setdefault(doi, index)

    pairs: list[VersionPair] = []
    for index, paper in enumerate(papers):
        if index in collapsed:
            continue
        preprint = _normalise_doi(paper.preprint_doi)
        if preprint is None:
            continue
        other = by_doi.get(preprint)
        if other is None or other == index:
            continue
        pairs.append(
            VersionPair(
                left=index,
                right=other,
                method=MatchMethod.PREPRINT_LINK,
                basis=f"record declares preprint_doi {preprint}",
            )
        )
    return pairs


def _semantic_stage(
    papers: Sequence[Paper],
    residue: Sequence[int],
    embedder: EmbeddingModel | None,
    threshold: float,
) -> SemanticStageReport:
    """Compare the residue pairwise and report unresolved candidates.

    A high cosine score is not record identity and does not establish that two
    records are versions of one work.  This stage is therefore observational:
    it changes neither membership nor version links.

    Parameters
    ----------
    papers : Sequence[Paper]
        Records in set order.
    residue : Sequence[int]
        Indices the exact stage left undecided.
    embedder : EmbeddingModel | None
        Semantic model.  ``None`` means the stage does not run and says so.
    threshold : float
        Cosine similarity at or above which a pair matches.

    Returns
    -------
    SemanticStageReport
        Whether the stage ran and how many candidate pairs crossed the
        threshold.
    """
    if embedder is None:
        return SemanticStageReport(
            ran=False,
            residue_size=len(residue),
            threshold=threshold,
            reason=(
                "no embedding model was supplied, so the residue was not examined "
                "semantically; possible duplicate or version relationships remain "
                "unresolved"
            ),
        )
    if not embedder.identity.is_semantic:
        return SemanticStageReport(
            ran=False,
            residue_size=len(residue),
            threshold=threshold,
            model=embedder.identity.describe(),
            reason=(
                f"the supplied model {embedder.identity.model_id!r} declares "
                "is_semantic=False, so its similarity scores are reproducible but "
                "meaningless; refusing to emit them as if they were measurements"
            ),
        )
    if len(residue) < 2:
        return SemanticStageReport(
            ran=True,
            residue_size=len(residue),
            threshold=threshold,
            model=embedder.identity.describe(),
        )

    from lit_search.index.mmr import cosine

    texts = [f"{papers[index].title}. {papers[index].abstract or ''}".strip() for index in residue]
    vectors = embedder.encode_documents(texts)
    matched = 0
    for left in range(len(residue)):
        for right in range(left + 1, len(residue)):
            similarity = cosine(vectors[left], vectors[right])
            if similarity < threshold:
                continue
            matched += 1
    return SemanticStageReport(
        ran=True,
        residue_size=len(residue),
        threshold=threshold,
        pairs_matched=matched,
        model=embedder.identity.describe(),
    )


def _components(pairs: Sequence[VersionPair]) -> list[list[int]]:
    """Group version pairs into connected components.

    Only explicit ``preprint_doi`` relationships reach this function.  Weak
    title and semantic evidence is intentionally excluded, so connected
    components cannot turn a chain of guesses into a proven multi-version work.

    Parameters
    ----------
    pairs : Sequence[VersionPair]
        Index pairs.

    Returns
    -------
    list[list[int]]
        Sorted index groups, in order of first appearance.
    """
    parent: dict[int, int] = {}

    def find(node: int) -> int:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    order: list[int] = []
    for pair in pairs:
        for node in (pair.left, pair.right):
            if node not in parent:
                order.append(node)
                parent[node] = node
        parent[find(pair.left)] = find(pair.right)

    grouped: dict[int, list[int]] = {}
    for node in order:
        grouped.setdefault(find(node), []).append(node)
    return [sorted(members) for members in grouped.values()]


def _build_links(papers: Sequence[Paper], pairs: Sequence[VersionPair]) -> list[VersionLink]:
    """Turn every version pair into one link per connected component.

    Parameters
    ----------
    papers : Sequence[Paper]
        Records in set order.
    pairs : Sequence[VersionPair]
        Explicit metadata-backed version pairs.

    Returns
    -------
    list[VersionLink]
        One link per group of versions.
    """
    links: list[VersionLink] = []
    for group in _components(pairs):
        indices = set(group)
        involved = [pair for pair in pairs if pair.left in indices and pair.right in indices]
        members = [papers[index] for index in group]
        read, cite, divergence = _choose_targets(members)
        links.append(
            VersionLink(
                member_keys=tuple(identity_key(paper) for paper in members),
                read_target=read,
                cite_target=cite,
                methods=tuple(dict.fromkeys(pair.method for pair in involved)),
                basis="; ".join(dict.fromkeys(pair.basis for pair in involved)),
                divergence=divergence,
            )
        )
    return links


def deduplicate(
    papers: Sequence[Paper],
    *,
    embedder: EmbeddingModel | None = None,
    semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    year_slack: int = DEFAULT_YEAR_SLACK,
) -> DeduplicationPlan:
    """Compute an exact-identity plan and inspect its unresolved residue.

    Parameters
    ----------
    papers : Sequence[Paper]
        Records to deduplicate, in set order.
    embedder : EmbeddingModel | None
        Semantic model for the second stage.  ``None`` runs the exact stage
        only and records that the residue is unexamined rather than clean.
    semantic_threshold : float
        Cosine similarity at or above which a residue pair is reported as an
        unresolved candidate.  It never changes membership or creates a link.
    year_slack : int
        Deprecated compatibility option.  Title/year evidence is not used to
        collapse or link records.

    Returns
    -------
    DeduplicationPlan
        Surviving members, collapses, version links and the stage report.
    """
    del year_slack
    collapsed, groups = _exact_record_stage(papers)

    version_pairs = _preprint_link_stage(papers, collapsed)

    decided = set(collapsed)
    for pair in version_pairs:
        decided.update({pair.left, pair.right})
    residue = [index for index in range(len(papers)) if index not in decided]

    report = _semantic_stage(papers, residue, embedder, semantic_threshold)
    links = _build_links(papers, version_pairs)
    survivors = tuple(paper for index, paper in enumerate(papers) if index not in collapsed)
    return DeduplicationPlan(
        members=survivors,
        submitted=len(papers),
        input_digest=_ordered_input_digest(papers),
        duplicate_groups=tuple(groups),
        version_links=tuple(links),
        semantic_stage=report,
    )


def _is_one_to_one_subset(members: Sequence[Paper], base: Sequence[Paper]) -> bool:
    """Return whether every survivor maps once to an explicit or exact base record."""
    available = set(range(len(base)))
    for member in members:
        member_digest = _paper_snapshot_digest(member)
        match = next(
            (
                index
                for index in sorted(available)
                if _paper_snapshot_digest(base[index]) == member_digest
                or same_record_identity(member, base[index])
            ),
            None,
        )
        if match is None:
            return False
        available.remove(match)
    return True


def record_deduplication(
    store: HandleStore,
    base_id: str,
    plan: DeduplicationPlan,
    *,
    label: str | None = None,
    propagator: CoveragePropagator | None = None,
) -> HandleRecord:
    """Write the plan as a new handle, carrying the version links in lineage.

    The version links live in the operation's ``parameters``, which is an
    arbitrary JSON mapping the handle schema already allows.  They are
    deliberately **not** a new envelope key: the handle schema refuses any
    version but 1 and refuses unrecognised keys, so adding one would force a
    schema bump that invalidates every handle already written -- and the tool's
    own name is still an open decision that will force a bump of its own.
    Putting the links in the lineage costs nothing and keeps them where a later
    reader looks for how a set came to be.

    The member-count change is recorded in the parameters too, and the
    coverage **status** is left to the propagator.  Whether a
    membership-reducing operation should floor coverage is open item ``O10``
    with two competing answers, and settling it here by default would resolve
    an open question silently.

    Parameters
    ----------
    store : HandleStore
        Store holding the base handle.
    base_id : str
        Full identifier of the set that was deduplicated.
    plan : DeduplicationPlan
        Computed plan.
    label : str | None
        Optional human-readable name.
    propagator : CoveragePropagator | None
        Coverage propagator.

    Returns
    -------
    HandleRecord
        The deduplicated handle, order invalidated when the base was ranked.

    Raises
    ------
    InvalidParameterError
        If the plan does not describe the base handle's members.
    """
    base = store.get(base_id)
    if plan.submitted != len(base.members):
        raise InvalidParameterError(
            f"deduplication plan was computed over {plan.submitted} record(s) but "
            f"{base_id} holds {len(base.members)}; the plan does not describe this set"
        )
    base_digest = _ordered_input_digest(base.members)
    if plan.input_digest != base_digest:
        raise InvalidParameterError(
            "deduplication plan input snapshot does not match the base handle; "
            "equal member counts do not establish lineage"
        )
    if len(plan.members) + plan.collapsed_count != plan.submitted:
        raise InvalidParameterError(
            "deduplication plan survivor and collapse counts do not account for "
            "the submitted records"
        )
    if not _is_one_to_one_subset(plan.members, base.members):
        raise InvalidParameterError(
            "deduplication plan survivors are not a one-to-one explicit-identity "
            "or exact-snapshot subset of the base handle"
        )
    active = propagator if propagator is not None else LedgerCoveragePropagator()
    coverage = active.combine(
        OperationName.DEDUPLICATE,
        [CoverageOperand(role=BASE_ROLE, coverage=base.coverage, member_count=len(base.members))],
    )
    return HandleRecord(
        operation=Operation(
            name=OperationName.DEDUPLICATE,
            operands=(Operand.of_handle(base_id, BASE_ROLE),),
            parameters={
                "strategy": plan.strategy(),
                "submitted": plan.submitted,
                "input_digest": plan.input_digest,
                "survived": len(plan.members),
                "collapsed": plan.collapsed_count,
                "duplicate_groups": [group.to_dict() for group in plan.duplicate_groups],
                "version_links": [link.to_dict() for link in plan.version_links],
                "semantic_stage": plan.semantic_stage.to_dict(),
            },
        ),
        order=base.order.invalidate(OperationName.DEDUPLICATE.value),
        members=plan.members,
        label=label,
        coverage=coverage,
    )


__all__ = [
    "DEFAULT_SEMANTIC_THRESHOLD",
    "DEFAULT_YEAR_SLACK",
    "DeduplicationPlan",
    "DuplicateGroup",
    "MatchMethod",
    "SemanticStageReport",
    "VersionLink",
    "VersionPair",
    "deduplicate",
    "normalise_title",
    "record_deduplication",
]
