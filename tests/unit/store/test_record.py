"""Tests for the handle envelope: content addressing and schema refusal."""

from __future__ import annotations

import copy
import datetime

import pytest

from lit_search.core.paper import Paper
from lit_search.exceptions import HandleIntegrityError, HandleSchemaError
from lit_search.store.lineage import Operation, OperationName
from lit_search.store.order import OrderRecord
from lit_search.store.record import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    HandleRecord,
    canonical_json,
    compute_handle_id,
    verify_payload_id,
)

from .conftest import FIXED_TIME


class TestContentAddressing:
    """A handle's identifier is a hash of everything it asserts."""

    def test_identifier_is_prefixed_hex(self, make_record, papers) -> None:
        """The identifier is the h_ marker plus 32 lowercase hex characters."""
        handle_id = make_record(papers).handle_id
        assert handle_id.startswith("h_")
        assert len(handle_id) == 34
        assert all(character in "0123456789abcdef" for character in handle_id[2:])

    def test_identical_content_yields_identical_identifier(self, make_record, papers) -> None:
        """Two records asserting the same thing address the same file."""
        assert make_record(papers).handle_id == make_record(papers).handle_id

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(lambda kwargs: kwargs.update(label="renamed"), id="label"),
            pytest.param(lambda kwargs: kwargs.update(coverage={"arxiv": "capped"}), id="coverage"),
            pytest.param(lambda kwargs: kwargs.update(query="[c]"), id="parameters"),
        ],
    )
    def test_any_change_changes_the_identifier(self, make_record, papers, mutate) -> None:
        """Changing anything the handle asserts changes its address."""
        kwargs: dict = {}
        mutate(kwargs)
        assert make_record(papers).handle_id != make_record(papers, **kwargs).handle_id

    def test_changing_a_member_changes_the_identifier(self, make_record, papers) -> None:
        """Membership is part of what a handle asserts."""
        assert make_record(papers).handle_id != make_record(papers[:2]).handle_id

    def test_member_order_changes_the_identifier(self, make_record, papers) -> None:
        """Sequence is recorded, so reordering produces a different handle."""
        assert make_record(papers).handle_id != make_record(list(reversed(papers))).handle_id

    def test_with_coverage_produces_a_new_identifier(self, make_record, papers) -> None:
        """Attaching a ledger changes the assertion, so it cannot be in place."""
        original = make_record(papers)
        attached = original.with_coverage({"arxiv": "complete"})
        assert attached.handle_id != original.handle_id
        assert original.coverage is None

    def test_canonical_json_sorts_keys(self) -> None:
        """Canonical form is independent of insertion order."""
        assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})

    def test_canonical_json_refuses_non_finite_numbers(self) -> None:
        """NaN is not JSON and must never reach a stored payload."""
        with pytest.raises(ValueError, match="Out of range"):
            canonical_json({"score": float("nan")})


class TestRoundTrip:
    """Serializing and reloading preserves everything the handle asserts."""

    def test_round_trip_preserves_fields(self, make_record, papers) -> None:
        """Every envelope field survives a to_dict/from_dict cycle."""
        original = make_record(papers, label="round one", coverage={"arxiv": "complete"})
        restored = HandleRecord.from_dict(original.to_dict())
        assert restored.handle_id == original.handle_id
        assert restored.label == "round one"
        assert restored.coverage == {"arxiv": "complete"}
        assert restored.created_by == "test"
        assert restored.created_at == FIXED_TIME
        assert [p.title for p in restored.members] == [p.title for p in papers]
        assert restored.operation.name is OperationName.SEARCH
        assert restored.operation.parameters["databases"] == ["arxiv"]

    def test_naive_timestamps_are_normalized_to_utc(self, papers) -> None:
        """A naive creation time is stored as UTC rather than kept ambiguous."""
        record = HandleRecord(
            operation=Operation(name=OperationName.SEARCH),
            members=tuple(papers),
            created_at=datetime.datetime(2026, 8, 30, 12, 0, 0),
        )
        assert record.created_at.tzinfo is datetime.UTC


class TestSchemaRefusal:
    """Loading refuses anything it cannot fully understand."""

    @pytest.fixture
    def payload(self, make_record, papers) -> dict:
        """Return a valid on-disk payload to corrupt in each test."""
        return make_record(papers).to_dict()

    def test_newer_schema_version_is_refused(self, payload) -> None:
        """A future version is refused rather than read with v1 assumptions."""
        payload["schema_version"] = SCHEMA_VERSION + 1
        with pytest.raises(HandleSchemaError, match="schema version 2"):
            HandleRecord.from_dict(payload)

    def test_older_schema_version_is_refused(self, payload) -> None:
        """An older version is refused because no migration is registered."""
        payload["schema_version"] = 0
        with pytest.raises(HandleSchemaError, match="older schema"):
            HandleRecord.from_dict(payload)

    def test_missing_schema_version_is_refused(self, payload) -> None:
        """An unversioned payload cannot be interpreted safely."""
        del payload["schema_version"]
        with pytest.raises(HandleSchemaError, match="schema_version: required key is missing"):
            HandleRecord.from_dict(payload)

    def test_wrong_schema_marker_is_refused(self, payload) -> None:
        """A payload from another producer is refused."""
        payload["schema"] = "some-other-tool/handle"
        with pytest.raises(HandleSchemaError, match="not written by the handle store"):
            HandleRecord.from_dict(payload)

    def test_string_schema_version_is_not_coerced(self, payload) -> None:
        """A version of the wrong type is refused, never coerced."""
        payload["schema_version"] = "1"
        with pytest.raises(HandleSchemaError, match="expected int, got str"):
            HandleRecord.from_dict(payload)

    @pytest.mark.parametrize(
        "key", ["created_at", "created_by", "label", "operation", "order", "coverage", "members"]
    )
    def test_missing_required_key_is_refused_not_defaulted(self, payload, key) -> None:
        """No required key is ever supplied from a default on load."""
        del payload[key]
        with pytest.raises(HandleSchemaError, match=f"{key}: required key is missing"):
            HandleRecord.from_dict(payload)

    def test_unknown_envelope_key_is_refused(self, payload) -> None:
        """An unrecognised key means the writer knew something this reader does not."""
        payload["provenance_v2"] = {"depth": 3}
        with pytest.raises(HandleSchemaError, match="unrecognised key"):
            HandleRecord.from_dict(payload)

    def test_unknown_nested_key_is_refused(self, payload) -> None:
        """Refusal reaches into nested payloads, not just the envelope."""
        payload["operation"]["cost"] = 1
        with pytest.raises(HandleSchemaError, match=r"handle\.operation: unrecognised key"):
            HandleRecord.from_dict(payload)

    def test_null_coverage_is_accepted_and_distinguishable(self, payload) -> None:
        """Present-and-null records "no ledger"; absent is refused."""
        assert payload["coverage"] is None
        assert HandleRecord.from_dict(payload).coverage is None

    def test_non_mapping_payload_is_refused(self) -> None:
        """A list is not a handle."""
        with pytest.raises(HandleSchemaError, match="expected an object"):
            HandleRecord.from_dict([1, 2, 3])

    def test_member_without_title_is_refused(self, payload) -> None:
        """A member that cannot be reconstructed stops the load."""
        payload["members"][0]["title"] = ""
        with pytest.raises(Exception, match="title cannot be null"):
            HandleRecord.from_dict(payload)


class TestIntegrityVerification:
    """A handle that no longer hashes to its own name is not trusted."""

    def test_untouched_payload_verifies(self, make_record, papers) -> None:
        """A freshly serialized payload verifies against its identifier."""
        payload = make_record(papers).to_dict()
        assert verify_payload_id(payload) == payload["handle_id"]

    def test_tampered_payload_is_detected(self, make_record, papers) -> None:
        """Editing a field without recomputing the address is caught."""
        payload = make_record(papers).to_dict()
        payload["label"] = "quietly renamed"
        with pytest.raises(HandleIntegrityError, match="does not match its own content"):
            verify_payload_id(payload)

    def test_dropped_member_is_detected(self, make_record, papers) -> None:
        """Removing a paper by hand is caught, which is the point."""
        payload = make_record(papers).to_dict()
        payload["members"].pop()
        with pytest.raises(HandleIntegrityError):
            verify_payload_id(payload)

    def test_verification_uses_raw_payload_not_reserialized_objects(
        self, make_record, papers
    ) -> None:
        """Verification does not depend on a byte-exact object round trip."""
        payload = make_record(papers).to_dict()
        reparsed = copy.deepcopy(payload)
        assert verify_payload_id(reparsed) == payload["handle_id"]


class TestIdentifierComputation:
    """The identifier is computed over the payload minus the identifier."""

    def test_compute_matches_to_dict(self, make_record, papers) -> None:
        """to_dict stamps the same address compute_handle_id derives."""
        record = make_record(papers)
        payload = record.to_dict()
        without_id = {k: v for k, v in payload.items() if k != "handle_id"}
        assert compute_handle_id(without_id) == payload["handle_id"]

    def test_schema_marker_participates_in_the_address(self, papers: list[Paper]) -> None:
        """Two payloads differing only by schema marker address differently."""
        base = {
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "created_at": FIXED_TIME.isoformat(),
            "created_by": "test",
            "label": None,
            "operation": Operation(name=OperationName.SEARCH).to_dict(),
            "order": OrderRecord.unordered().to_dict(),
            "coverage": None,
            "members": [],
        }
        other = dict(base, schema="other/handle")
        assert compute_handle_id(base) != compute_handle_id(other)


def test_ranked_record_rejects_mismatched_score_count(papers: list[Paper], query_ranking) -> None:
    """A ranked handle whose scores do not line up with members is refused."""
    from lit_search.exceptions import OrderSemanticsError

    with pytest.raises(OrderSemanticsError, match="score"):
        HandleRecord(
            operation=Operation(name=OperationName.SEARCH),
            order=OrderRecord.ranked(query_ranking, [1.0, 0.5]),
            members=tuple(papers),
        )
