"""Unit tests for the coverage contract's value types and propagation rules.

The tests are organised around the two claims the contract has to keep:

1. **A status must be actionable.**  Every status carries the specific detail
   a user would need to act on it, and constructing one without that detail
   fails.
2. **Propagation is conservative.**  A derived set is never reported as
   better covered than the worst thing it came from, and a difference taken
   against a non-complete set is itself non-complete.
"""

from __future__ import annotations

import pytest

from lit_search.core.coverage import (
    ZERO_COVERAGE_STATUSES,
    CoverageLedger,
    CoverageScope,
    CoverageStatus,
    CoveredPapers,
    DerivationNote,
    ExecutionStatus,
    SourceCoverage,
    difference_coverage,
    intersection_coverage,
    severity,
    union_coverage,
)

ALL_STATUSES = tuple(CoverageStatus)
CONNECTOR_STATUSES = tuple(s for s in CoverageStatus if s is not CoverageStatus.UNKNOWN)


# ---------------------------------------------------------------------------
# The enum itself
# ---------------------------------------------------------------------------


class TestCoverageStatusEnum:
    """Tests for the closed status enum and its severity ordering."""

    def test_covers_every_status_the_spec_requires(self) -> None:
        """R1's six required states all exist."""
        required = {
            "complete",
            "capped",
            "partial",
            "skipped_no_credential",
            "query_unsupported",
            "failed",
        }
        assert required <= {s.value for s in CoverageStatus}

    def test_severity_is_a_strict_total_order(self) -> None:
        """No two statuses share a severity, so 'worst' is never ambiguous."""
        ranks = [severity(s) for s in ALL_STATUSES]
        assert len(set(ranks)) == len(ranks)

    def test_complete_is_the_only_best_status(self) -> None:
        """COMPLETE ranks strictly below everything else."""
        assert all(
            severity(CoverageStatus.COMPLETE) < severity(s)
            for s in ALL_STATUSES
            if s is not CoverageStatus.COMPLETE
        )

    def test_unknown_is_the_worst_status(self) -> None:
        """Unrecorded provenance outranks every recorded shortfall.

        A set whose coverage nobody wrote down must never sort as better than
        one that honestly reported a failure.
        """
        assert all(
            severity(CoverageStatus.UNKNOWN) > severity(s)
            for s in ALL_STATUSES
            if s is not CoverageStatus.UNKNOWN
        )

    def test_zero_coverage_statuses_rank_above_partial(self) -> None:
        """A source that contributed nothing is worse than one that fell short."""
        assert all(severity(s) > severity(CoverageStatus.PARTIAL) for s in ZERO_COVERAGE_STATUSES)


# ---------------------------------------------------------------------------
# SourceCoverage: a status must be actionable
# ---------------------------------------------------------------------------


class TestSourceCoverageIsActionable:
    """Each status must carry the detail needed to act on it."""

    def test_capped_requires_the_cap(self) -> None:
        """CAPPED without the cap value is rejected."""
        with pytest.raises(ValueError, match="CAPPED requires 'cap'"):
            SourceCoverage(source="arxiv", status=CoverageStatus.CAPPED, retrieved=50)

    def test_partial_requires_a_reason(self) -> None:
        """PARTIAL without a reason is rejected."""
        with pytest.raises(ValueError, match="PARTIAL requires 'detail'"):
            SourceCoverage(source="arxiv", status=CoverageStatus.PARTIAL, retrieved=3)

    def test_query_unsupported_requires_a_reason(self) -> None:
        """QUERY_UNSUPPORTED without a reason is rejected."""
        with pytest.raises(ValueError, match="QUERY_UNSUPPORTED requires 'detail'"):
            SourceCoverage(source="wos", status=CoverageStatus.QUERY_UNSUPPORTED)

    def test_skipped_requires_the_credential_name(self) -> None:
        """SKIPPED_NO_CREDENTIAL without the credential name is rejected."""
        with pytest.raises(ValueError, match="requires 'missing_credential'"):
            SourceCoverage(source="ieee", status=CoverageStatus.SKIPPED_NO_CREDENTIAL)

    def test_failed_requires_the_error(self) -> None:
        """FAILED without an error message is rejected."""
        with pytest.raises(ValueError, match="FAILED requires 'error'"):
            SourceCoverage(source="scopus", status=CoverageStatus.FAILED)

    def test_unknown_cannot_be_claimed_by_a_source(self) -> None:
        """UNKNOWN is a ledger aggregate, never an escape hatch for a connector.

        Without this a porting agent could satisfy the contract by labelling
        everything UNKNOWN, which reintroduces exactly the silence the type
        exists to forbid.
        """
        with pytest.raises(ValueError, match="ledger-level aggregate"):
            SourceCoverage(source="arxiv", status=CoverageStatus.UNKNOWN)

    @pytest.mark.parametrize("status", sorted(ZERO_COVERAGE_STATUSES - {CoverageStatus.UNKNOWN}))
    def test_zero_coverage_status_cannot_carry_records(self, status: CoverageStatus) -> None:
        """A status meaning 'contributed nothing' cannot report records."""
        with pytest.raises(ValueError, match="contributed nothing"):
            SourceCoverage(
                source="arxiv",
                status=status,
                retrieved=4,
                detail="d",
                missing_credential="c",
                error="e",
            )

    def test_source_must_be_named(self) -> None:
        """An unnamed source is rejected: a ledger row with no key is useless."""
        with pytest.raises(ValueError, match="non-empty identifier"):
            SourceCoverage.complete("", 0, reported_total=0)

    def test_constructors_produce_valid_records(self) -> None:
        """Every classmethod constructor satisfies its own validation."""
        records = [
            SourceCoverage.complete("a", 3, reported_total=3),
            SourceCoverage.capped("b", 50, cap=50),
            SourceCoverage.partial("c", 2, detail="page 3 failed"),
            SourceCoverage.query_unsupported("d", detail="no wildcard support"),
            SourceCoverage.skipped_no_credential("e", credential="ieee_api_key"),
            SourceCoverage.skipped_by_policy("f", detail="excluded by the default profile"),
            SourceCoverage.failed("g", error="401 Unauthorized", error_type="HTTPError"),
        ]
        assert [r.status for r in records] == list(CONNECTOR_STATUSES)

    def test_skipped_names_the_credential_in_its_detail(self) -> None:
        """The rendered detail tells the user what to supply."""
        record = SourceCoverage.skipped_no_credential("ieee", credential="ieee_api_key")
        assert "ieee_api_key" in record.describe()


class TestFromPagination:
    """Tests for the shared cap / shortfall / unverifiable decision procedure."""

    def test_reaching_the_cap_is_capped(self) -> None:
        """Stopping at the caller's limit is CAPPED, not COMPLETE."""
        record = SourceCoverage.from_pagination("s", retrieved=50, reported_total=900, cap=50)
        assert record.status is CoverageStatus.CAPPED
        assert record.cap == 50
        assert record.execution_status is ExecutionStatus.COMPLETE

    def test_no_reported_total_is_partial_not_complete(self) -> None:
        """A source that publishes no count cannot be certified complete.

        This is the pessimistic branch: assuming completeness because nothing
        contradicted it is the failure mode the whole design targets.
        """
        record = SourceCoverage.from_pagination("s", retrieved=12, reported_total=None)
        assert record.status is CoverageStatus.PARTIAL
        assert "no result total" in (record.detail or "")
        assert record.execution_status is ExecutionStatus.COMPLETE

    def test_shortfall_against_reported_total_is_partial(self) -> None:
        """Fewer records than the source said existed is PARTIAL, with the numbers."""
        record = SourceCoverage.from_pagination("s", retrieved=40, reported_total=512)
        assert record.status is CoverageStatus.PARTIAL
        assert "40" in (record.detail or "")
        assert "512" in (record.detail or "")
        assert record.execution_status is ExecutionStatus.INCOMPLETE

    def test_parse_loss_makes_a_reached_cap_execution_incomplete(self) -> None:
        """A declared cap does not erase an unplanned loss on the way to it."""
        record = SourceCoverage.from_pagination(
            "s",
            retrieved=50,
            reported_total=900,
            cap=50,
            shortfall_detail="2 returned records could not be parsed",
        )

        assert record.status is CoverageStatus.CAPPED
        assert record.execution_status is ExecutionStatus.INCOMPLETE

    def test_full_retrieval_is_complete(self) -> None:
        """Matching the reported total is the only path to COMPLETE."""
        record = SourceCoverage.from_pagination("s", retrieved=7, reported_total=7, cap=None)
        assert record.status is CoverageStatus.COMPLETE

    def test_cap_takes_precedence_over_completeness(self) -> None:
        """A cap equal to the total still reports CAPPED, not COMPLETE."""
        record = SourceCoverage.from_pagination("s", retrieved=7, reported_total=7, cap=7)
        assert record.status is CoverageStatus.CAPPED

    def test_over_retrieval_is_partial_not_complete(self) -> None:
        """More records than the source reported is a shortfall, not a surplus.

        ``_check_complete_is_justifiable`` rejects ``retrieved < reported_total``
        and says nothing about the other direction, so before this branch
        existed a walk that returned 20 records against a reported 17 came back
        ``COMPLETE``.  It is reachable: CrossRef answers an unparseable cursor
        with page one and a **fresh** token, and Solr's ``cursorMark`` re-serves
        records when a document is inserted before the current sort position.
        Records served twice mean records never served at all.
        """
        record = SourceCoverage.from_pagination("s", retrieved=20, reported_total=17)
        assert record.status is CoverageStatus.PARTIAL
        assert "20" in (record.detail or "")
        assert "17" in (record.detail or "")

    def test_over_retrieval_keeps_the_shortfall_note(self) -> None:
        """The unparsed-record note must not be dropped by the new branch."""
        record = SourceCoverage.from_pagination(
            "s", retrieved=20, reported_total=17, shortfall_detail="2 of 22 records unparsed"
        )
        assert record.status is CoverageStatus.PARTIAL
        assert "2 of 22 records unparsed" in (record.detail or "")


# ---------------------------------------------------------------------------
# CoveredPapers
# ---------------------------------------------------------------------------


class TestCoveredPapers:
    """Tests for the connector return type."""

    def test_is_not_a_sequence(self) -> None:
        """CoveredPapers deliberately refuses to impersonate a list.

        If it were iterable and sized, every existing call site would keep
        working and would silently drop the coverage, which is the whole
        failure this type exists to make impossible.
        """
        result = CoveredPapers.none(SourceCoverage.complete("s", 0, reported_total=0))
        with pytest.raises(TypeError):
            len(result)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            list(result)  # type: ignore[call-overload]

    def test_count_mismatch_is_rejected(self, make_paper) -> None:
        """A coverage record whose count disagrees with the papers is a bug."""
        with pytest.raises(ValueError, match="does not match the number"):
            CoveredPapers(
                papers=(make_paper(),),
                coverage=SourceCoverage.complete("s", 5, reported_total=5),
            )

    def test_papers_are_frozen_into_a_tuple(self, make_paper) -> None:
        """A mutable list handed in cannot be mutated afterwards."""
        supplied = [make_paper()]
        result = CoveredPapers(
            papers=supplied, coverage=SourceCoverage.complete("s", 1, reported_total=1)
        )
        supplied.append(make_paper(title="sneaky"))
        assert len(result.papers) == 1

    def test_with_source_relabels_coverage(self) -> None:
        """The base class can stamp its own name onto a connector's record."""
        result = CoveredPapers.none(
            SourceCoverage.complete("typo", 0, reported_total=0)
        ).with_source("arxiv")
        assert result.coverage.source == "arxiv"

    def test_with_source_is_identity_when_already_correct(self) -> None:
        """Relabelling to the same name does not copy."""
        result = CoveredPapers.none(SourceCoverage.complete("arxiv", 0, reported_total=0))
        assert result.with_source("arxiv") is result


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class TestCoverageLedger:
    """Tests for the set-level ledger."""

    def test_empty_ledger_is_unknown_not_complete(self) -> None:
        """A set with no recorded coverage reports UNKNOWN.

        Reporting COMPLETE here would make an unrecorded set look like a
        verified one, which is the motivating failure in miniature.
        """
        assert CoverageLedger().status is CoverageStatus.UNKNOWN
        assert not CoverageLedger().is_complete

    def test_aggregate_is_the_worst_entry(self) -> None:
        """One failed source degrades the whole set."""
        ledger = CoverageLedger.from_sources(
            [
                SourceCoverage.complete("arxiv", 10, reported_total=10),
                SourceCoverage.complete("openalex", 20, reported_total=20),
                SourceCoverage.failed("ieee", error="401"),
            ]
        )
        assert ledger.status is CoverageStatus.FAILED
        assert ledger.complete_sources == ("arxiv", "openalex")
        assert [e.source for e in ledger.incomplete_entries] == ["ieee"]

    def test_all_complete_aggregates_to_complete(self) -> None:
        """Only an all-complete ledger reports COMPLETE."""
        ledger = CoverageLedger.from_sources(
            [
                SourceCoverage.complete("arxiv", 1, reported_total=1),
                SourceCoverage.complete("pubmed", 2, reported_total=2),
            ]
        )
        assert ledger.is_complete

    def test_duplicate_sources_collapse_worst_wins(self) -> None:
        """Two records for one source collapse to the worse of the two."""
        ledger = CoverageLedger.from_sources(
            [
                SourceCoverage.complete("arxiv", 10, reported_total=10),
                SourceCoverage.partial("arxiv", 3, detail="page 2 failed"),
            ]
        )
        assert len(ledger) == 1
        assert ledger.status is CoverageStatus.PARTIAL

    def test_get_returns_none_for_unconsulted_source(self) -> None:
        """A source that was never consulted has no entry."""
        ledger = CoverageLedger.from_sources(
            [SourceCoverage.complete("arxiv", 1, reported_total=1)]
        )
        assert ledger.get("arxiv") is not None
        assert ledger.get("scopus") is None

    def test_render_names_every_incomplete_source_and_why(self) -> None:
        """The rendering is honest: it states the aggregate and each shortfall."""
        ledger = CoverageLedger.from_sources(
            [
                SourceCoverage.complete("arxiv", 10, reported_total=10),
                SourceCoverage.skipped_no_credential("ieee", credential="ieee_api_key"),
                SourceCoverage.capped("openalex", 50, cap=50, reported_total=4000),
            ]
        )
        text = ledger.render()
        assert "Coverage: skipped_no_credential" in text
        assert "arxiv" in text
        assert "ieee" in text
        assert "openalex" in text
        assert "ieee_api_key" in text
        assert "cap 50" in text

    def test_render_says_so_when_nothing_was_recorded(self) -> None:
        """An empty ledger renders as unknown rather than as a clean bill of health."""
        text = CoverageLedger().render()
        assert "unknown" in text
        assert "no coverage was recorded" in text

    def test_round_trip_through_dict(self) -> None:
        """Serialization preserves every entry and derivation."""
        ledger = CoverageLedger(
            entries=(
                SourceCoverage.capped("arxiv", 50, cap=50, reported_total=900),
                SourceCoverage.failed("ieee", error="boom", error_type="HTTPError"),
            ),
            derivations=(
                DerivationNote(
                    "difference", "subtrahend", CoverageStatus.PARTIAL, "zotero partial"
                ),
            ),
        )
        restored = CoverageLedger.from_dict(ledger.to_dict())
        assert restored == ledger

    def test_stored_aggregate_status_cannot_lie(self) -> None:
        """A hand-edited status in the JSON is ignored and recomputed."""
        data = CoverageLedger.from_sources([SourceCoverage.failed("ieee", error="boom")]).to_dict()
        data["status"] = "complete"
        assert CoverageLedger.from_dict(data).status is CoverageStatus.FAILED

    def test_policy_scope_does_not_impersonate_execution_failure(self) -> None:
        """A fulfilled default plan is green while its narrower scope stays visible."""
        ledger = CoverageLedger(
            entries=(
                SourceCoverage.complete("arxiv", 0, reported_total=0),
                SourceCoverage.complete("openalex", 0, reported_total=0),
                SourceCoverage.skipped_by_policy("ieee", detail="default profile"),
            ),
            scope=CoverageScope(
                profile="default",
                selected_sources=("arxiv", "openalex"),
                registered_sources=("arxiv", "openalex", "ieee"),
            ),
        )

        assert ledger.status is CoverageStatus.SKIPPED_BY_POLICY
        assert not ledger.is_complete
        assert ledger.execution_status is ExecutionStatus.COMPLETE
        assert ledger.execution_is_complete
        assert ledger.scope is not None
        assert ledger.scope.excluded_sources == ("ieee",)

    def test_caps_complete_execution_but_not_semantic_coverage(self) -> None:
        """Reaching a declared bound is not an unplanned shortfall."""
        ledger = CoverageLedger.from_sources(
            [SourceCoverage.capped("openalex", 10, cap=10, reported_total=100)]
        )

        assert ledger.status is CoverageStatus.CAPPED
        assert ledger.execution_status is ExecutionStatus.COMPLETE

    def test_partial_is_incomplete_unless_evidence_explicitly_overrides_it(self) -> None:
        """Legacy partial records remain conservative; planned boundaries are explicit."""
        legacy = SourceCoverage.from_dict(
            {"source": "archive", "status": "partial", "retrieved": 0, "detail": "no total"}
        )
        planned = SourceCoverage.partial(
            "archive",
            0,
            detail="source publishes no total",
            execution_status=ExecutionStatus.COMPLETE,
        )

        assert legacy.execution_status is ExecutionStatus.INCOMPLETE
        assert planned.execution_status is ExecutionStatus.COMPLETE

    def test_duplicate_source_collapse_cannot_erase_execution_shortfall(self) -> None:
        """Equal semantic statuses use execution severity as a conservative tie-break."""
        planned = SourceCoverage.partial(
            "archive",
            1,
            detail="declared source window",
            execution_status=ExecutionStatus.COMPLETE,
        )
        failed_page = SourceCoverage.partial(
            "archive",
            1,
            detail="second page failed",
            execution_status=ExecutionStatus.INCOMPLETE,
        )

        ledger = CoverageLedger.from_sources((planned, failed_page))

        assert ledger.get("archive") == failed_page
        assert ledger.execution_status is ExecutionStatus.INCOMPLETE

    def test_execution_and_scope_round_trip_and_stored_execution_cannot_lie(self) -> None:
        """Execution is recomputed from evidence and scope survives persistence."""
        ledger = CoverageLedger(
            entries=(SourceCoverage.complete("arxiv", 0, reported_total=0),),
            scope=CoverageScope(
                profile="explicit",
                selected_sources=("arxiv",),
                registered_sources=("arxiv", "ieee"),
            ),
        )
        payload = ledger.to_dict()
        payload["execution"]["status"] = "incomplete"

        restored = CoverageLedger.from_dict(payload)

        assert restored.execution_status is ExecutionStatus.COMPLETE
        assert restored.scope == ledger.scope


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------

_COMPLETE = CoverageLedger.from_sources([SourceCoverage.complete("arxiv", 5, reported_total=5)])
_PARTIAL = CoverageLedger.from_sources(
    [SourceCoverage.partial("zotero", 3, detail="local API returned one page only")]
)
_FAILED = CoverageLedger.from_sources([SourceCoverage.failed("ieee", error="401")])
_UNRECORDED = CoverageLedger()


class TestUnionAndIntersectionPropagation:
    """A derived set inherits the worst status of its operands."""

    @pytest.mark.parametrize("operation", [union_coverage, intersection_coverage])
    def test_worst_operand_wins(self, operation) -> None:
        """Combining a complete set with a failed one is not complete."""
        assert operation(_COMPLETE, _FAILED).status is CoverageStatus.FAILED

    @pytest.mark.parametrize("operation", [union_coverage, intersection_coverage])
    def test_all_complete_stays_complete(self, operation) -> None:
        """Combining complete sets stays complete."""
        other = CoverageLedger.from_sources(
            [SourceCoverage.complete("pubmed", 2, reported_total=2)]
        )
        assert operation(_COMPLETE, other).status is CoverageStatus.COMPLETE

    @pytest.mark.parametrize("operation", [union_coverage, intersection_coverage])
    def test_every_operand_source_survives(self, operation) -> None:
        """Both operands' sources appear, so the render can name them all."""
        assert set(operation(_COMPLETE, _FAILED).sources) == {"arxiv", "ieee"}

    @pytest.mark.parametrize("operation", [union_coverage, intersection_coverage])
    def test_unrecorded_operand_poisons_the_result(self, operation) -> None:
        """A set with no ledger drags the result to UNKNOWN rather than vanishing."""
        assert operation(_COMPLETE, _UNRECORDED).status is CoverageStatus.UNKNOWN

    @pytest.mark.parametrize("operation", [union_coverage, intersection_coverage])
    def test_no_operands_is_unknown(self, operation) -> None:
        """Combining nothing yields UNKNOWN with a note saying why."""
        result = operation()
        assert result.status is CoverageStatus.UNKNOWN
        assert result.derivations

    @pytest.mark.parametrize("operation", [union_coverage, intersection_coverage])
    def test_derivation_notes_carry_through(self, operation) -> None:
        """A shortfall an operand inherited is not laundered away by combining."""
        tainted = difference_coverage(_COMPLETE, _PARTIAL)
        assert operation(tainted, _COMPLETE).status is CoverageStatus.PARTIAL

    @pytest.mark.parametrize("operation", [union_coverage, intersection_coverage])
    def test_policy_shadow_cannot_erase_an_operands_execution(self, operation) -> None:
        """Worst semantic collapse and execution aggregation remain independent."""
        excluded = CoverageLedger.from_sources(
            [SourceCoverage.skipped_by_policy("ieee", detail="default profile")]
        )
        selected = CoverageLedger.from_sources(
            [SourceCoverage.complete("ieee", 0, reported_total=0)]
        )

        result = operation(excluded, selected)

        assert result.status is CoverageStatus.SKIPPED_BY_POLICY
        assert result.execution_status is ExecutionStatus.COMPLETE


class TestDifferencePropagation:
    """A difference against a non-complete operand is itself non-complete."""

    def test_complete_minus_complete_stays_complete(self) -> None:
        """Subtracting a fully-known set does not degrade coverage."""
        other = CoverageLedger.from_sources(
            [SourceCoverage.complete("zotero", 40, reported_total=40)]
        )
        assert difference_coverage(_COMPLETE, other).status is CoverageStatus.COMPLETE

    def test_complete_minus_partial_is_not_complete(self) -> None:
        """The Zotero case: an incomplete subtrahend makes the result incomplete.

        Every source of the left-hand set was complete, yet the difference is
        not, because records that should have been removed were never seen.
        Without this rule the result confidently labels already-held papers
        as new.
        """
        result = difference_coverage(_COMPLETE, _PARTIAL, subtrahend_label="zotero library")
        assert result.status is CoverageStatus.PARTIAL
        assert not result.is_complete
        assert result.execution_status is ExecutionStatus.INCOMPLETE

    def test_difference_keeps_a_planned_subtrahend_boundary_out_of_execution_failure(
        self,
    ) -> None:
        """Semantic subtraction risk remains visible without inventing an outage."""
        bounded = CoverageLedger.from_sources(
            [
                SourceCoverage.partial(
                    "archive",
                    1,
                    detail="declared window",
                    execution_status=ExecutionStatus.COMPLETE,
                )
            ]
        )

        result = difference_coverage(_COMPLETE, bounded)

        assert result.status is CoverageStatus.PARTIAL
        assert result.execution_status is ExecutionStatus.COMPLETE
        assert result.derivations[-1].execution_status is ExecutionStatus.COMPLETE

    def test_complete_minus_unrecorded_is_unknown(self) -> None:
        """Subtracting a set with no ledger at all is the worst case."""
        result = difference_coverage(_COMPLETE, _UNRECORDED)
        assert result.status is CoverageStatus.UNKNOWN

    def test_subtrahend_sources_do_not_become_result_sources(self) -> None:
        """A subtracted source was not searched, so it is not listed as one.

        Recording it as an entry would claim we queried it for content and it
        came back partial, which is a different and false statement.
        """
        result = difference_coverage(_COMPLETE, _PARTIAL)
        assert result.sources == ("arxiv",)
        assert "zotero" not in result.sources

    def test_shortfall_is_recorded_as_a_derivation_note(self) -> None:
        """The reason is preserved, and it names the offending source."""
        result = difference_coverage(_COMPLETE, _PARTIAL, subtrahend_label="zotero library")
        assert len(result.derivations) == 1
        note = result.derivations[0]
        assert note.operation == "difference"
        assert note.role == "subtrahend"
        assert "zotero library" in note.detail
        assert "zotero=partial" in note.detail

    def test_note_explains_the_consequence_not_just_the_cause(self) -> None:
        """The rendering says what may now be wrong with the result set."""
        text = difference_coverage(_COMPLETE, _PARTIAL).render()
        assert "not subtracted" in text
        assert "may be present here in error" in text

    def test_minuend_shortfalls_are_preserved(self) -> None:
        """A difference does not repair the left-hand set's own gaps."""
        result = difference_coverage(_FAILED, _PARTIAL)
        assert result.status is CoverageStatus.FAILED
        assert result.sources == ("ieee",)
        assert result.derivations

    def test_difference_is_not_symmetric(self) -> None:
        """Swapping the operands changes which shortfall is recorded where."""
        forward = difference_coverage(_COMPLETE, _PARTIAL)
        backward = difference_coverage(_PARTIAL, _COMPLETE)
        assert forward.sources == ("arxiv",)
        assert backward.sources == ("zotero",)
        assert forward.derivations
        assert not backward.derivations
