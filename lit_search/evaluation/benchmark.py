"""Strict manifests for frozen, adversarial prior-art evaluation.

The case manifest is an answer key.  It is frozen independently of a run and
identified by a SHA-256 digest over its complete semantic content.  A run
records that digest; the grader refuses a changed case definition instead of
quietly evaluating against a moving target.

Acquisition and ranking are deliberately different fields.  No ranking state
can establish that a target entered the candidate union, and ``UNKNOWN`` is a
closed state rather than an alias for either found or missed.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from lit_search.core.coverage import CoverageStatus, ExecutionStatus

BENCHMARK_SUITE_SCHEMA = "lit-search/adversarial-benchmark-suite"
BENCHMARK_RUN_SCHEMA = "lit-search/adversarial-benchmark-run"
BENCHMARK_SCHEMA_VERSION = 1
GRADER_VERSION = "adversarial-prior-art-v1"
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class BenchmarkManifestError(ValueError):
    """A benchmark or run manifest violates the frozen contract."""


class Stratum(StrEnum):
    """Adversarial mechanisms represented by the suite."""

    CITATION_RECOVERABLE_DEVELOPMENT = "citation_recoverable_development"
    CITATION_DISCONNECTED = "citation_disconnected"
    RECENT_UNCITED = "recent_uncited"
    TERMINOLOGY_COMMUNITY_SHIFT = "terminology_community_shift"
    AGGREGATOR_ASYMMETRY = "aggregator_asymmetry"
    ACTOR_VENUE_CLASSIFICATION = "actor_venue_classification"
    IDENTIFIER_VERSION_STRESS = "identifier_version_stress"
    AMBIGUOUS_LANGUAGE = "ambiguous_language"
    QUESTION_MODEL_REVISION = "question_model_revision"


REQUIRED_HELD_OUT_STRATA: frozenset[Stratum] = frozenset(
    {
        Stratum.CITATION_DISCONNECTED,
        Stratum.RECENT_UNCITED,
        Stratum.TERMINOLOGY_COMMUNITY_SHIFT,
        Stratum.AGGREGATOR_ASYMMETRY,
        Stratum.ACTOR_VENUE_CLASSIFICATION,
        Stratum.IDENTIFIER_VERSION_STRESS,
        Stratum.AMBIGUOUS_LANGUAGE,
        Stratum.QUESTION_MODEL_REVISION,
    }
)


class CitationReachability(StrEnum):
    """Scoped citation-path result at the case freeze point."""

    NO_USABLE_PATH = "no_usable_path"
    PATH_EXISTS = "path_exists"
    UNKNOWN = "unknown"


class ProviderPresence(StrEnum):
    """What a frozen provider observation says about a critical record."""

    PRESENT = "present"
    ABSENT = "absent"
    MISREPRESENTED = "misrepresented"
    UNKNOWN = "unknown"


class BenchmarkTrack(StrEnum):
    """Mechanism-isolation track executed by a run."""

    ORIGINAL_QUERY = "original_query"
    SEMANTIC_RERANKING_ONLY = "semantic_ranking_only"
    STRUCTURAL_ONLY = "structural_only"
    QUERY_FAMILY_ONLY = "query_family_only"
    QUERY_FAMILY_AND_STRUCTURAL = "query_family_and_structural"
    INTEGRATED = "integrated"


class AcquisitionStatus(StrEnum):
    """Whether a target entered the candidate union."""

    ACQUIRED = "acquired"
    NOT_ACQUIRED = "not_acquired"
    UNKNOWN = "unknown"


class AcquisitionMechanism(StrEnum):
    """Discovery mechanism that introduced a target before ranking."""

    ORIGINAL_QUERY = "original_query"
    QUERY_FAMILY = "query_family"
    CITATION_TRAVERSAL = "citation_traversal"
    ACTOR_VENUE_CLASSIFICATION = "actor_venue_classification"


class RankingStatus(StrEnum):
    """A target's separate surfacing outcome at the frozen cutoff."""

    RANKED = "ranked"
    NOT_IN_TOP_K = "not_in_top_k"
    NOT_RUN = "not_run"
    UNKNOWN = "unknown"


class ExecutionOutcome(StrEnum):
    """Whether the declared work for a case executed."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    UNKNOWN = "unknown"


class ExecutionIssueKind(StrEnum):
    """Typed location of an execution shortfall or unknown."""

    SOURCE = "source"
    QUERY_CELL = "query_cell"
    GRAPH_WORK_UNIT = "graph_work_unit"
    IDENTITY = "identity"
    RANKING = "ranking"
    OTHER = "other"


class IssueState(StrEnum):
    """Whether an issue is a known shortfall or an unknown outcome."""

    INCOMPLETE = "incomplete"
    UNKNOWN = "unknown"


def canonical_json(value: Any) -> str:
    """Return deterministic JSON for digests and fixture comparison."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_digest(value: Any) -> str:
    """Return a tagged SHA-256 digest of a semantic JSON value."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _mapping(value: Any, context: str) -> dict[str, Any]:
    """Require a string-keyed mapping at *context*."""

    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise BenchmarkManifestError(f"{context}: expected an object with string keys")
    return value


def _keys(data: dict[str, Any], expected: set[str], context: str) -> None:
    """Reject missing or unknown keys in a strict manifest object."""

    missing = sorted(expected - set(data))
    unknown = sorted(set(data) - expected)
    if missing:
        raise BenchmarkManifestError(f"{context}: missing required key(s) {missing}")
    if unknown:
        raise BenchmarkManifestError(f"{context}: unknown key(s) {unknown}")


def _text(data: dict[str, Any], key: str, context: str, *, nonempty: bool = True) -> str:
    """Read a string field, requiring content by default."""

    value = data[key]
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise BenchmarkManifestError(f"{context}.{key}: expected a non-empty string")
    return value


def _optional_text(data: dict[str, Any], key: str, context: str) -> str | None:
    """Read a nullable, non-empty string field."""

    value = data[key]
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise BenchmarkManifestError(f"{context}.{key}: expected null or a non-empty string")
    return value


def _integer(data: dict[str, Any], key: str, context: str, *, minimum: int = 0) -> int:
    """Read a non-boolean integer at or above *minimum*."""

    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BenchmarkManifestError(f"{context}.{key}: expected integer >= {minimum}")
    return int(value)


def _enum(data: dict[str, Any], key: str, enum: type[StrEnum], context: str) -> Any:
    """Read one member of a closed string enum."""

    raw = _text(data, key, context)
    try:
        return enum(raw)
    except ValueError as error:
        raise BenchmarkManifestError(f"{context}.{key}: unsupported value {raw!r}") from error


def _list(data: dict[str, Any], key: str, context: str) -> list[Any]:
    """Read a list field without coercion."""

    value = data[key]
    if not isinstance(value, list):
        raise BenchmarkManifestError(f"{context}.{key}: expected a list")
    return value


def _text_tuple(data: dict[str, Any], key: str, context: str) -> tuple[str, ...]:
    """Read a list of non-empty strings as an immutable tuple."""

    values = _list(data, key, context)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise BenchmarkManifestError(f"{context}.{key}: expected non-empty strings")
    return tuple(values)


def _digest(value: str, context: str) -> str:
    """Validate and return one tagged lowercase SHA-256 digest."""

    if not _DIGEST_RE.fullmatch(value):
        raise BenchmarkManifestError(f"{context}: expected sha256:<64 lowercase hex characters>")
    return value


def _unique(values: tuple[str, ...], context: str) -> None:
    """Refuse duplicate strings in a semantically set-like field."""

    if len(values) != len(set(values)):
        raise BenchmarkManifestError(f"{context}: duplicate values are forbidden")


@dataclass(frozen=True, slots=True)
class TargetIdentity:
    """One acceptable publication-version identity for a critical target."""

    kind: str
    value: str

    _KINDS: ClassVar[frozenset[str]] = frozenset(
        {"doi", "arxiv", "semantic_scholar", "openalex", "pmid", "other"}
    )

    def __post_init__(self) -> None:
        """Reject unsupported identity kinds and empty values."""

        if self.kind not in self._KINDS:
            raise BenchmarkManifestError(f"target identity: unsupported kind {self.kind!r}")
        if not self.value.strip():
            raise BenchmarkManifestError("target identity: value must be non-empty")

    def to_dict(self) -> dict[str, str]:
        """Return the strict JSON representation."""

        return {"kind": self.kind, "value": self.value}

    @classmethod
    def from_dict(cls, value: Any, context: str) -> TargetIdentity:
        """Read one strict identity object."""

        data = _mapping(value, context)
        _keys(data, {"kind", "value"}, context)
        return cls(kind=_text(data, "kind", context), value=_text(data, "value", context))


@dataclass(frozen=True, slots=True)
class CriticalTarget:
    """A pre-judged relevant paper and its acceptable version identities."""

    target_id: str
    acceptable_identities: tuple[TargetIdentity, ...]
    relevance_evidence_digest: str

    def __post_init__(self) -> None:
        """Require a target identifier set and frozen relevance evidence."""

        if not self.target_id.strip():
            raise BenchmarkManifestError("critical target: target_id must be non-empty")
        if not self.acceptable_identities:
            raise BenchmarkManifestError(f"{self.target_id}: at least one identity is required")
        identity_pairs = tuple(
            (item.kind, item.value.casefold()) for item in self.acceptable_identities
        )
        if len(identity_pairs) != len(set(identity_pairs)):
            raise BenchmarkManifestError(f"{self.target_id}: duplicate acceptable identities")
        _digest(self.relevance_evidence_digest, f"{self.target_id}.relevance_evidence_digest")

    def to_dict(self) -> dict[str, Any]:
        """Return the strict JSON representation."""

        return {
            "target_id": self.target_id,
            "acceptable_identities": [item.to_dict() for item in self.acceptable_identities],
            "relevance_evidence_digest": self.relevance_evidence_digest,
        }

    @classmethod
    def from_dict(cls, value: Any, context: str) -> CriticalTarget:
        """Read one strict target object."""

        data = _mapping(value, context)
        _keys(data, {"target_id", "acceptable_identities", "relevance_evidence_digest"}, context)
        identities = tuple(
            TargetIdentity.from_dict(item, f"{context}.acceptable_identities[{index}]")
            for index, item in enumerate(_list(data, "acceptable_identities", context))
        )
        return cls(
            target_id=_text(data, "target_id", context),
            acceptable_identities=identities,
            relevance_evidence_digest=_digest(
                _text(data, "relevance_evidence_digest", context),
                f"{context}.relevance_evidence_digest",
            ),
        )


@dataclass(frozen=True, slots=True)
class ProviderObservation:
    """A provider-presence claim with coverage, execution, and evidence."""

    source: str
    presence: ProviderPresence
    coverage_status: CoverageStatus
    execution_status: ExecutionStatus
    evidence_digest: str

    def __post_init__(self) -> None:
        """Validate the provider claim and its absence preconditions."""

        if not self.source.strip():
            raise BenchmarkManifestError("provider observation source must be non-empty")
        _digest(self.evidence_digest, f"provider observation {self.source}.evidence_digest")
        if self.presence is ProviderPresence.ABSENT and (
            self.coverage_status is not CoverageStatus.COMPLETE
            or self.execution_status is not ExecutionStatus.COMPLETE
        ):
            raise BenchmarkManifestError(
                f"provider observation {self.source}: absence requires complete coverage and execution"
            )

    def to_dict(self) -> dict[str, str]:
        """Return the strict JSON representation."""

        return {
            "source": self.source,
            "presence": self.presence.value,
            "coverage_status": self.coverage_status.value,
            "execution_status": self.execution_status.value,
            "evidence_digest": self.evidence_digest,
        }

    @classmethod
    def from_dict(cls, value: Any, context: str) -> ProviderObservation:
        """Read one strict provider observation."""

        data = _mapping(value, context)
        _keys(
            data,
            {
                "source",
                "presence",
                "coverage_status",
                "execution_status",
                "evidence_digest",
            },
            context,
        )
        return cls(
            source=_text(data, "source", context),
            presence=_enum(data, "presence", ProviderPresence, context),
            coverage_status=_enum(data, "coverage_status", CoverageStatus, context),
            execution_status=_enum(data, "execution_status", ExecutionStatus, context),
            evidence_digest=_digest(
                _text(data, "evidence_digest", context), f"{context}.evidence_digest"
            ),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One frozen research question and its adversarial answer key."""

    case_id: str
    primary_stratum: Stratum
    secondary_strata: tuple[Stratum, ...]
    research_question: str
    initial_formulation: str
    initial_query_family_digest: str
    initial_query_visible_terms: tuple[str, ...]
    prohibited_answer_terms: tuple[str, ...]
    critical_targets: tuple[CriticalTarget, ...]
    freeze_timestamp: str
    source_snapshot_digests: tuple[str, ...]
    citation_reachability: CitationReachability
    citation_evidence_digest: str
    provider_observations: tuple[ProviderObservation, ...]
    relevance_protocol_digest: str
    max_wall_seconds: int
    max_provider_requests: int

    def __post_init__(self) -> None:
        """Validate every case invariant at construction time."""

        self._validate_required_fields()
        self._validate_strata_evidence()
        self._validate_target_uniqueness()
        self._validate_initial_input_leakage()
        self._validate_evidence_digests()

    def _validate_required_fields(self) -> None:
        """Require the fields needed to execute and audit one case."""

        if not self.case_id.strip():
            raise BenchmarkManifestError("case_id must be non-empty")
        if not self.research_question.strip() or not self.initial_formulation.strip():
            raise BenchmarkManifestError(f"{self.case_id}: question and formulation are required")
        if not self.initial_query_visible_terms:
            raise BenchmarkManifestError(
                f"{self.case_id}: flattened run-visible initial query terms are required"
            )
        if not self.freeze_timestamp.strip():
            raise BenchmarkManifestError(f"{self.case_id}: freeze_timestamp is required")
        if not self.critical_targets:
            raise BenchmarkManifestError(
                f"{self.case_id}: at least one critical target is required"
            )
        if not self.source_snapshot_digests:
            raise BenchmarkManifestError(f"{self.case_id}: source snapshots are required")
        if not self.provider_observations:
            raise BenchmarkManifestError(f"{self.case_id}: provider observations are required")
        if self.max_wall_seconds <= 0 or self.max_provider_requests <= 0:
            raise BenchmarkManifestError(f"{self.case_id}: budgets must be positive")

    def _validate_strata_evidence(self) -> None:
        """Ensure adversarial tags are backed by compatible observations."""

        tags = (self.primary_stratum, *self.secondary_strata)
        if len(tags) != len(set(tags)):
            raise BenchmarkManifestError(f"{self.case_id}: duplicate stratum tags")
        if (
            Stratum.CITATION_DISCONNECTED in tags
            and self.citation_reachability is not CitationReachability.NO_USABLE_PATH
        ):
            raise BenchmarkManifestError(
                f"{self.case_id}: citation_disconnected requires scoped no_usable_path evidence"
            )
        if Stratum.AGGREGATOR_ASYMMETRY in tags and not any(
            observation.presence in {ProviderPresence.ABSENT, ProviderPresence.MISREPRESENTED}
            for observation in self.provider_observations
        ):
            raise BenchmarkManifestError(
                f"{self.case_id}: aggregator_asymmetry requires absent or misrepresented provider evidence"
            )

    def _validate_target_uniqueness(self) -> None:
        """Prevent duplicate target IDs or double-counted identities."""

        target_ids = tuple(target.target_id for target in self.critical_targets)
        _unique(target_ids, f"{self.case_id}.critical_targets")
        target_identity_pairs = tuple(
            (identity.kind, identity.value.casefold())
            for target in self.critical_targets
            for identity in target.acceptable_identities
        )
        if len(target_identity_pairs) != len(set(target_identity_pairs)):
            raise BenchmarkManifestError(
                f"{self.case_id}: one acceptable identity cannot count as multiple critical targets"
            )

    def _validate_initial_input_leakage(self) -> None:
        """Reject answer-key terms from all run-visible initial inputs."""

        answer_terms = tuple(term.casefold() for term in self.prohibited_answer_terms)
        _unique(answer_terms, f"{self.case_id}.prohibited_answer_terms")
        _unique(
            tuple(term.casefold() for term in self.initial_query_visible_terms),
            f"{self.case_id}.initial_query_visible_terms",
        )
        run_visible_input = "\n".join(
            (self.research_question, self.initial_formulation, *self.initial_query_visible_terms)
        ).casefold()
        leaked = [term for term in answer_terms if term and term in run_visible_input]
        if leaked:
            raise BenchmarkManifestError(
                f"{self.case_id}: run-visible initial input leaks prohibited answer term(s) {leaked}"
            )

    def _validate_evidence_digests(self) -> None:
        """Validate every case-level evidence and snapshot digest."""

        for name, digest in (
            ("initial_query_family_digest", self.initial_query_family_digest),
            ("citation_evidence_digest", self.citation_evidence_digest),
            ("relevance_protocol_digest", self.relevance_protocol_digest),
        ):
            _digest(digest, f"{self.case_id}.{name}")
        for index, digest in enumerate(self.source_snapshot_digests):
            _digest(digest, f"{self.case_id}.evidence_digest[{index}]")

    @property
    def strata(self) -> frozenset[Stratum]:
        """Return primary and secondary strata without duplication."""

        return frozenset((self.primary_stratum, *self.secondary_strata))

    def to_dict(self) -> dict[str, Any]:
        """Return the strict JSON representation."""

        return {
            "case_id": self.case_id,
            "primary_stratum": self.primary_stratum.value,
            "secondary_strata": [item.value for item in self.secondary_strata],
            "research_question": self.research_question,
            "initial_formulation": self.initial_formulation,
            "initial_query_family_digest": self.initial_query_family_digest,
            "initial_query_visible_terms": list(self.initial_query_visible_terms),
            "prohibited_answer_terms": list(self.prohibited_answer_terms),
            "critical_targets": [target.to_dict() for target in self.critical_targets],
            "freeze_timestamp": self.freeze_timestamp,
            "source_snapshot_digests": list(self.source_snapshot_digests),
            "citation_reachability": self.citation_reachability.value,
            "citation_evidence_digest": self.citation_evidence_digest,
            "provider_observations": [item.to_dict() for item in self.provider_observations],
            "relevance_protocol_digest": self.relevance_protocol_digest,
            "max_wall_seconds": self.max_wall_seconds,
            "max_provider_requests": self.max_provider_requests,
        }

    @classmethod
    def from_dict(cls, value: Any, context: str) -> BenchmarkCase:
        """Read one strict case object."""

        data = _mapping(value, context)
        expected = {
            "case_id",
            "primary_stratum",
            "secondary_strata",
            "research_question",
            "initial_formulation",
            "initial_query_family_digest",
            "initial_query_visible_terms",
            "prohibited_answer_terms",
            "critical_targets",
            "freeze_timestamp",
            "source_snapshot_digests",
            "citation_reachability",
            "citation_evidence_digest",
            "provider_observations",
            "relevance_protocol_digest",
            "max_wall_seconds",
            "max_provider_requests",
        }
        _keys(data, expected, context)
        raw_secondary = _text_tuple(data, "secondary_strata", context)
        try:
            secondary = tuple(Stratum(item) for item in raw_secondary)
        except ValueError as error:
            raise BenchmarkManifestError(
                f"{context}.secondary_strata: unsupported stratum"
            ) from error
        targets = tuple(
            CriticalTarget.from_dict(item, f"{context}.critical_targets[{index}]")
            for index, item in enumerate(_list(data, "critical_targets", context))
        )
        return cls(
            case_id=_text(data, "case_id", context),
            primary_stratum=_enum(data, "primary_stratum", Stratum, context),
            secondary_strata=secondary,
            research_question=_text(data, "research_question", context),
            initial_formulation=_text(data, "initial_formulation", context),
            initial_query_family_digest=_digest(
                _text(data, "initial_query_family_digest", context),
                f"{context}.initial_query_family_digest",
            ),
            initial_query_visible_terms=_text_tuple(data, "initial_query_visible_terms", context),
            prohibited_answer_terms=_text_tuple(data, "prohibited_answer_terms", context),
            critical_targets=targets,
            freeze_timestamp=_text(data, "freeze_timestamp", context),
            source_snapshot_digests=_text_tuple(data, "source_snapshot_digests", context),
            citation_reachability=_enum(
                data, "citation_reachability", CitationReachability, context
            ),
            citation_evidence_digest=_digest(
                _text(data, "citation_evidence_digest", context),
                f"{context}.citation_evidence_digest",
            ),
            provider_observations=tuple(
                ProviderObservation.from_dict(item, f"{context}.provider_observations[{index}]")
                for index, item in enumerate(_list(data, "provider_observations", context))
            ),
            relevance_protocol_digest=_digest(
                _text(data, "relevance_protocol_digest", context),
                f"{context}.relevance_protocol_digest",
            ),
            max_wall_seconds=_integer(data, "max_wall_seconds", context, minimum=1),
            max_provider_requests=_integer(data, "max_provider_requests", context, minimum=1),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    """A frozen collection of 8--12 adversarial cases."""

    suite_id: str
    cases: tuple[BenchmarkCase, ...]
    freeze_digest: str
    schema: str = BENCHMARK_SUITE_SCHEMA
    schema_version: int = BENCHMARK_SCHEMA_VERSION
    grader_version: str = GRADER_VERSION

    def semantic_dict(self) -> dict[str, Any]:
        """Return all content covered by the freeze digest."""

        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "grader_version": self.grader_version,
            "suite_id": self.suite_id,
            "cases": [case.to_dict() for case in self.cases],
        }

    def expected_freeze_digest(self) -> str:
        """Compute the digest required by the current case definitions."""

        return sha256_digest(self.semantic_dict())

    def with_computed_freeze_digest(self) -> BenchmarkSuite:
        """Return a copy frozen over its current semantic content."""

        return replace(self, freeze_digest=self.expected_freeze_digest())

    def validate(self) -> None:
        """Validate suite composition and the recorded freeze digest."""

        if self.schema != BENCHMARK_SUITE_SCHEMA or self.schema_version != BENCHMARK_SCHEMA_VERSION:
            raise BenchmarkManifestError("suite schema or schema_version is unsupported")
        if self.grader_version != GRADER_VERSION:
            raise BenchmarkManifestError("suite grader_version is unsupported")
        if not self.suite_id.strip():
            raise BenchmarkManifestError("suite_id must be non-empty")
        if not 8 <= len(self.cases) <= 12:
            raise BenchmarkManifestError("suite must contain between 8 and 12 cases")
        case_ids = tuple(case.case_id for case in self.cases)
        _unique(case_ids, "suite.cases")
        represented = frozenset().union(*(case.strata for case in self.cases))
        missing = sorted(item.value for item in REQUIRED_HELD_OUT_STRATA - represented)
        if missing:
            raise BenchmarkManifestError(f"suite is missing required stratum/strata {missing}")
        disconnected = sum(Stratum.CITATION_DISCONNECTED in case.strata for case in self.cases)
        if disconnected < 2:
            raise BenchmarkManifestError("suite requires at least two citation-disconnected cases")
        _digest(self.freeze_digest, "suite.freeze_digest")
        expected = self.expected_freeze_digest()
        if self.freeze_digest != expected:
            raise BenchmarkManifestError(
                "suite definitions changed after freeze: recorded freeze_digest does not match"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete strict JSON representation."""

        return {**self.semantic_dict(), "freeze_digest": self.freeze_digest}

    @classmethod
    def from_dict(cls, value: Any, *, verify_freeze: bool = True) -> BenchmarkSuite:
        """Read a strict suite, optionally allowing a draft before freezing."""

        data = _mapping(value, "suite")
        _keys(
            data,
            {"schema", "schema_version", "grader_version", "suite_id", "cases", "freeze_digest"},
            "suite",
        )
        schema_version = _integer(data, "schema_version", "suite", minimum=1)
        cases = tuple(
            BenchmarkCase.from_dict(item, f"suite.cases[{index}]")
            for index, item in enumerate(_list(data, "cases", "suite"))
        )
        suite = cls(
            schema=_text(data, "schema", "suite"),
            schema_version=schema_version,
            grader_version=_text(data, "grader_version", "suite"),
            suite_id=_text(data, "suite_id", "suite"),
            cases=cases,
            freeze_digest=_text(data, "freeze_digest", "suite"),
        )
        if verify_freeze:
            suite.validate()
        return suite

    @classmethod
    def read_json(cls, path: str | Path, *, verify_freeze: bool = True) -> BenchmarkSuite:
        """Load a suite from UTF-8 JSON."""

        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BenchmarkManifestError(f"cannot read benchmark suite {path}: {error}") from error
        return cls.from_dict(value, verify_freeze=verify_freeze)


@dataclass(frozen=True, slots=True)
class DiscoveryEvidence:
    """Provenance for the acquisition operation that introduced a target."""

    mechanism: AcquisitionMechanism
    operation_id: str
    evidence_digest: str

    def __post_init__(self) -> None:
        """Require an auditable operation ID and evidence digest."""

        if not self.operation_id.strip():
            raise BenchmarkManifestError("discovery evidence operation_id must be non-empty")
        _digest(self.evidence_digest, "discovery evidence digest")

    def to_dict(self) -> dict[str, str]:
        """Return the strict JSON representation."""

        return {
            "mechanism": self.mechanism.value,
            "operation_id": self.operation_id,
            "evidence_digest": self.evidence_digest,
        }

    @classmethod
    def from_dict(cls, value: Any, context: str) -> DiscoveryEvidence:
        """Read one strict acquisition evidence object."""

        data = _mapping(value, context)
        _keys(data, {"mechanism", "operation_id", "evidence_digest"}, context)
        return cls(
            mechanism=_enum(data, "mechanism", AcquisitionMechanism, context),
            operation_id=_text(data, "operation_id", context),
            evidence_digest=_digest(
                _text(data, "evidence_digest", context), f"{context}.evidence_digest"
            ),
        )


@dataclass(frozen=True, slots=True)
class RankingOutcome:
    """A target's surfacing state, independent of acquisition."""

    status: RankingStatus
    best_rank: int | None

    def __post_init__(self) -> None:
        """Keep rank values consistent with the closed ranking state."""

        if self.status is RankingStatus.RANKED:
            if self.best_rank is None or self.best_rank <= 0:
                raise BenchmarkManifestError("ranked target requires positive best_rank")
        elif self.best_rank is not None:
            raise BenchmarkManifestError("only ranked targets may carry best_rank")

    def to_dict(self) -> dict[str, Any]:
        """Return the strict JSON representation."""

        return {"status": self.status.value, "best_rank": self.best_rank}

    @classmethod
    def from_dict(cls, value: Any, context: str) -> RankingOutcome:
        """Read one strict ranking result."""

        data = _mapping(value, context)
        _keys(data, {"status", "best_rank"}, context)
        raw_rank = data["best_rank"]
        if raw_rank is not None and (
            isinstance(raw_rank, bool) or not isinstance(raw_rank, int) or raw_rank <= 0
        ):
            raise BenchmarkManifestError(f"{context}.best_rank: expected null or positive integer")
        return cls(status=_enum(data, "status", RankingStatus, context), best_rank=raw_rank)


@dataclass(frozen=True, slots=True)
class TargetOutcome:
    """Separate acquisition and ranking observations for one target."""

    target_id: str
    acquisition_status: AcquisitionStatus
    observed_identity: TargetIdentity | None
    discovery_evidence: tuple[DiscoveryEvidence, ...]
    ranking: RankingOutcome

    def __post_init__(self) -> None:
        """Keep acquisition evidence, observed identity, and rank coherent."""

        if not self.target_id.strip():
            raise BenchmarkManifestError("target outcome target_id must be non-empty")
        if self.acquisition_status is AcquisitionStatus.ACQUIRED and not self.discovery_evidence:
            raise BenchmarkManifestError(
                f"{self.target_id}: acquired target requires discovery evidence"
            )
        if self.acquisition_status is AcquisitionStatus.ACQUIRED and self.observed_identity is None:
            raise BenchmarkManifestError(
                f"{self.target_id}: acquired target requires observed identity"
            )
        if self.acquisition_status is not AcquisitionStatus.ACQUIRED and self.discovery_evidence:
            raise BenchmarkManifestError(
                f"{self.target_id}: non-acquired target cannot claim discovery evidence"
            )
        if (
            self.acquisition_status is not AcquisitionStatus.ACQUIRED
            and self.observed_identity is not None
        ):
            raise BenchmarkManifestError(
                f"{self.target_id}: non-acquired target cannot claim an observed identity"
            )
        if (
            self.acquisition_status is AcquisitionStatus.NOT_ACQUIRED
            and self.ranking.status is not RankingStatus.NOT_RUN
        ):
            raise BenchmarkManifestError(
                f"{self.target_id}: a known absent target must have ranking status not_run"
            )
        if self.acquisition_status is AcquisitionStatus.UNKNOWN and self.ranking.status not in {
            RankingStatus.NOT_RUN,
            RankingStatus.UNKNOWN,
        }:
            raise BenchmarkManifestError(
                f"{self.target_id}: unknown acquisition cannot have a determinate rank"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the strict JSON representation."""

        return {
            "target_id": self.target_id,
            "acquisition_status": self.acquisition_status.value,
            "observed_identity": (
                self.observed_identity.to_dict() if self.observed_identity is not None else None
            ),
            "discovery_evidence": [item.to_dict() for item in self.discovery_evidence],
            "ranking": self.ranking.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any, context: str) -> TargetOutcome:
        """Read one strict target outcome."""

        data = _mapping(value, context)
        _keys(
            data,
            {
                "target_id",
                "acquisition_status",
                "observed_identity",
                "discovery_evidence",
                "ranking",
            },
            context,
        )
        evidence = tuple(
            DiscoveryEvidence.from_dict(item, f"{context}.discovery_evidence[{index}]")
            for index, item in enumerate(_list(data, "discovery_evidence", context))
        )
        return cls(
            target_id=_text(data, "target_id", context),
            acquisition_status=_enum(data, "acquisition_status", AcquisitionStatus, context),
            observed_identity=(
                TargetIdentity.from_dict(data["observed_identity"], f"{context}.observed_identity")
                if data["observed_identity"] is not None
                else None
            ),
            discovery_evidence=evidence,
            ranking=RankingOutcome.from_dict(data["ranking"], f"{context}.ranking"),
        )


@dataclass(frozen=True, slots=True)
class ExecutionIssue:
    """A typed known shortfall or unresolved execution outcome."""

    kind: ExecutionIssueKind
    state: IssueState
    unit_id: str
    evidence_digest: str

    def __post_init__(self) -> None:
        """Require an auditable work-unit ID and issue evidence."""

        if not self.unit_id.strip():
            raise BenchmarkManifestError("execution issue unit_id must be non-empty")
        _digest(self.evidence_digest, "execution issue evidence_digest")

    def to_dict(self) -> dict[str, str]:
        """Return the strict JSON representation."""

        return {
            "kind": self.kind.value,
            "state": self.state.value,
            "unit_id": self.unit_id,
            "evidence_digest": self.evidence_digest,
        }

    @classmethod
    def from_dict(cls, value: Any, context: str) -> ExecutionIssue:
        """Read one strict execution issue."""

        data = _mapping(value, context)
        _keys(data, {"kind", "state", "unit_id", "evidence_digest"}, context)
        return cls(
            kind=_enum(data, "kind", ExecutionIssueKind, context),
            state=_enum(data, "state", IssueState, context),
            unit_id=_text(data, "unit_id", context),
            evidence_digest=_digest(
                _text(data, "evidence_digest", context), f"{context}.evidence_digest"
            ),
        )


@dataclass(frozen=True, slots=True)
class CaseRun:
    """One run's outcomes for one frozen case."""

    case_id: str
    execution: ExecutionOutcome
    execution_issues: tuple[ExecutionIssue, ...]
    targets: tuple[TargetOutcome, ...]

    def __post_init__(self) -> None:
        """Derive the only valid aggregate execution outcome from issues."""

        target_ids = tuple(target.target_id for target in self.targets)
        _unique(target_ids, f"case run {self.case_id}.targets")
        states = {issue.state for issue in self.execution_issues}
        expected = (
            ExecutionOutcome.UNKNOWN
            if IssueState.UNKNOWN in states
            else ExecutionOutcome.INCOMPLETE
            if IssueState.INCOMPLETE in states
            else ExecutionOutcome.COMPLETE
        )
        if self.execution is not expected:
            raise BenchmarkManifestError(
                f"{self.case_id}: execution={self.execution.value} contradicts typed issues; "
                f"expected {expected.value}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the strict JSON representation."""

        return {
            "case_id": self.case_id,
            "execution": self.execution.value,
            "execution_issues": [issue.to_dict() for issue in self.execution_issues],
            "targets": [target.to_dict() for target in self.targets],
        }

    @classmethod
    def from_dict(cls, value: Any, context: str) -> CaseRun:
        """Read one strict case-run object."""

        data = _mapping(value, context)
        _keys(data, {"case_id", "execution", "execution_issues", "targets"}, context)
        return cls(
            case_id=_text(data, "case_id", context),
            execution=_enum(data, "execution", ExecutionOutcome, context),
            execution_issues=tuple(
                ExecutionIssue.from_dict(item, f"{context}.execution_issues[{index}]")
                for index, item in enumerate(_list(data, "execution_issues", context))
            ),
            targets=tuple(
                TargetOutcome.from_dict(item, f"{context}.targets[{index}]")
                for index, item in enumerate(_list(data, "targets", context))
            ),
        )


_TRACK_MECHANISMS: dict[BenchmarkTrack, frozenset[AcquisitionMechanism]] = {
    BenchmarkTrack.ORIGINAL_QUERY: frozenset({AcquisitionMechanism.ORIGINAL_QUERY}),
    BenchmarkTrack.SEMANTIC_RERANKING_ONLY: frozenset({AcquisitionMechanism.ORIGINAL_QUERY}),
    BenchmarkTrack.STRUCTURAL_ONLY: frozenset(
        {AcquisitionMechanism.ORIGINAL_QUERY, AcquisitionMechanism.CITATION_TRAVERSAL}
    ),
    BenchmarkTrack.QUERY_FAMILY_ONLY: frozenset(
        {
            AcquisitionMechanism.ORIGINAL_QUERY,
            AcquisitionMechanism.QUERY_FAMILY,
            AcquisitionMechanism.ACTOR_VENUE_CLASSIFICATION,
        }
    ),
    BenchmarkTrack.QUERY_FAMILY_AND_STRUCTURAL: frozenset(
        {
            AcquisitionMechanism.ORIGINAL_QUERY,
            AcquisitionMechanism.QUERY_FAMILY,
            AcquisitionMechanism.CITATION_TRAVERSAL,
            AcquisitionMechanism.ACTOR_VENUE_CLASSIFICATION,
        }
    ),
    BenchmarkTrack.INTEGRATED: frozenset(AcquisitionMechanism),
}


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Outcomes for one mechanism-isolation run against one frozen suite."""

    run_id: str
    suite_id: str
    suite_freeze_digest: str
    track: BenchmarkTrack
    ranking_cutoff: int
    candidate_union_digest: str
    baseline_candidate_union_digest: str | None
    cases: tuple[CaseRun, ...]
    schema: str = BENCHMARK_RUN_SCHEMA
    schema_version: int = BENCHMARK_SCHEMA_VERSION
    grader_version: str = GRADER_VERSION

    def validate_against(self, suite: BenchmarkSuite) -> None:
        """Validate this run against the exact frozen suite."""

        suite.validate()
        self._validate_header(suite)
        self._validate_candidate_union()
        self._validate_case_outcomes(suite)

    def _validate_header(self, suite: BenchmarkSuite) -> None:
        """Validate schema, suite binding, freeze binding, and cutoff."""

        if self.schema != BENCHMARK_RUN_SCHEMA or self.schema_version != BENCHMARK_SCHEMA_VERSION:
            raise BenchmarkManifestError("run schema or schema_version is unsupported")
        if self.grader_version != GRADER_VERSION:
            raise BenchmarkManifestError("run grader_version is unsupported")
        if self.suite_id != suite.suite_id:
            raise BenchmarkManifestError("run suite_id does not match benchmark suite")
        if self.suite_freeze_digest != suite.freeze_digest:
            raise BenchmarkManifestError(
                "run was recorded against a different suite freeze digest; scoring refused"
            )
        if self.ranking_cutoff <= 0:
            raise BenchmarkManifestError("ranking_cutoff must be positive")

    def _validate_candidate_union(self) -> None:
        """Validate membership digests and reranking-only immutability."""

        _digest(self.candidate_union_digest, "run.candidate_union_digest")
        if self.baseline_candidate_union_digest is not None:
            _digest(self.baseline_candidate_union_digest, "run.baseline_candidate_union_digest")
        if self.track is BenchmarkTrack.SEMANTIC_RERANKING_ONLY:
            if self.baseline_candidate_union_digest is None:
                raise BenchmarkManifestError(
                    "semantic-ranking-only run requires a baseline candidate-union digest"
                )
            if self.baseline_candidate_union_digest != self.candidate_union_digest:
                raise BenchmarkManifestError(
                    "semantic ranking changed candidate membership; it cannot claim recall"
                )

    def _validate_case_outcomes(self, suite: BenchmarkSuite) -> None:
        """Require exactly one outcome for every frozen case."""

        expected_cases = {case.case_id: case for case in suite.cases}
        actual_ids = tuple(case.case_id for case in self.cases)
        _unique(actual_ids, "run.cases")
        if set(actual_ids) != set(expected_cases):
            missing = sorted(set(expected_cases) - set(actual_ids))
            unknown = sorted(set(actual_ids) - set(expected_cases))
            raise BenchmarkManifestError(
                f"run cases do not match suite; missing={missing}, unknown={unknown}"
            )
        allowed_mechanisms = _TRACK_MECHANISMS[self.track]
        for case_run in self.cases:
            self._validate_case_targets(
                case_run, expected_cases[case_run.case_id], allowed_mechanisms
            )

    def _validate_case_targets(
        self,
        case_run: CaseRun,
        case_definition: BenchmarkCase,
        allowed_mechanisms: frozenset[AcquisitionMechanism],
    ) -> None:
        """Require exactly one outcome for every target in one case."""

        target_definitions = {item.target_id: item for item in case_definition.critical_targets}
        actual_targets = {item.target_id for item in case_run.targets}
        if actual_targets != set(target_definitions):
            raise BenchmarkManifestError(
                f"{case_run.case_id}: target outcomes do not match frozen critical targets"
            )
        for target in case_run.targets:
            self._validate_target(
                case_run.case_id,
                target,
                target_definitions[target.target_id],
                allowed_mechanisms,
            )

    def _validate_target(
        self,
        case_id: str,
        target: TargetOutcome,
        target_definition: CriticalTarget,
        allowed_mechanisms: frozenset[AcquisitionMechanism],
    ) -> None:
        """Validate identity, mechanism, and rank for one target outcome."""

        if target.observed_identity is not None:
            acceptable = {
                (identity.kind, identity.value.casefold())
                for identity in target_definition.acceptable_identities
            }
            observed = (target.observed_identity.kind, target.observed_identity.value.casefold())
            if observed not in acceptable:
                raise BenchmarkManifestError(
                    f"{case_id}/{target.target_id}: observed identity is not an "
                    "acceptable frozen version identity"
                )
        mechanisms = {item.mechanism for item in target.discovery_evidence}
        forbidden = mechanisms - allowed_mechanisms
        if forbidden:
            raise BenchmarkManifestError(
                f"{case_id}/{target.target_id}: mechanism(s) "
                f"{sorted(item.value for item in forbidden)} forbidden in {self.track.value}"
            )
        if (
            target.ranking.status is RankingStatus.RANKED
            and target.ranking.best_rank is not None
            and target.ranking.best_rank > self.ranking_cutoff
        ):
            raise BenchmarkManifestError(
                f"{case_id}/{target.target_id}: best_rank exceeds frozen cutoff"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete strict JSON representation."""

        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "grader_version": self.grader_version,
            "run_id": self.run_id,
            "suite_id": self.suite_id,
            "suite_freeze_digest": self.suite_freeze_digest,
            "track": self.track.value,
            "ranking_cutoff": self.ranking_cutoff,
            "candidate_union_digest": self.candidate_union_digest,
            "baseline_candidate_union_digest": self.baseline_candidate_union_digest,
            "cases": [case.to_dict() for case in self.cases],
        }

    @classmethod
    def from_dict(cls, value: Any) -> RunManifest:
        """Read a strict run manifest."""

        data = _mapping(value, "run")
        _keys(
            data,
            {
                "schema",
                "schema_version",
                "grader_version",
                "run_id",
                "suite_id",
                "suite_freeze_digest",
                "track",
                "ranking_cutoff",
                "candidate_union_digest",
                "baseline_candidate_union_digest",
                "cases",
            },
            "run",
        )
        return cls(
            schema=_text(data, "schema", "run"),
            schema_version=_integer(data, "schema_version", "run", minimum=1),
            grader_version=_text(data, "grader_version", "run"),
            run_id=_text(data, "run_id", "run"),
            suite_id=_text(data, "suite_id", "run"),
            suite_freeze_digest=_digest(
                _text(data, "suite_freeze_digest", "run"), "run.suite_freeze_digest"
            ),
            track=_enum(data, "track", BenchmarkTrack, "run"),
            ranking_cutoff=_integer(data, "ranking_cutoff", "run", minimum=1),
            candidate_union_digest=_digest(
                _text(data, "candidate_union_digest", "run"), "run.candidate_union_digest"
            ),
            baseline_candidate_union_digest=_optional_text(
                data, "baseline_candidate_union_digest", "run"
            ),
            cases=tuple(
                CaseRun.from_dict(item, f"run.cases[{index}]")
                for index, item in enumerate(_list(data, "cases", "run"))
            ),
        )

    @classmethod
    def read_json(cls, path: str | Path) -> RunManifest:
        """Load a run manifest from UTF-8 JSON."""

        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BenchmarkManifestError(f"cannot read benchmark run {path}: {error}") from error
        return cls.from_dict(value)


__all__ = [
    "BENCHMARK_RUN_SCHEMA",
    "BENCHMARK_SCHEMA_VERSION",
    "BENCHMARK_SUITE_SCHEMA",
    "GRADER_VERSION",
    "REQUIRED_HELD_OUT_STRATA",
    "AcquisitionMechanism",
    "AcquisitionStatus",
    "BenchmarkCase",
    "BenchmarkManifestError",
    "BenchmarkSuite",
    "BenchmarkTrack",
    "CaseRun",
    "CitationReachability",
    "CriticalTarget",
    "DiscoveryEvidence",
    "ExecutionIssue",
    "ExecutionIssueKind",
    "ExecutionOutcome",
    "IssueState",
    "ProviderObservation",
    "ProviderPresence",
    "RankingOutcome",
    "RankingStatus",
    "RunManifest",
    "Stratum",
    "TargetIdentity",
    "TargetOutcome",
    "canonical_json",
    "sha256_digest",
]
