"""Verification tests for immutable query-family handle evidence."""

from __future__ import annotations

import dataclasses

import pytest

from lit_search.core.paper import Paper
from lit_search.core.query_family import Evidence, EvidenceKind, QueryFamilyRevision
from lit_search.query.family_evidence import (
    HandleEvidenceVerifier,
    QueryFamilyEvidenceError,
    field_evidence_digest,
    member_evidence_digest,
)
from lit_search.store.operations import searched
from lit_search.store.store import HandleStore


def _paper(title: str, doi: str | None = None) -> Paper:
    return Paper(
        title=title,
        abstract=None,
        authors=[],
        source=None,
        publication_date=None,
        doi=doi,
    )


def _revision(handle_id: str, paper: Paper, *, field: str = "title") -> QueryFamilyRevision:
    payload = paper.to_dict()
    return QueryFamilyRevision(
        research_question="Which vocabulary did this paper use?",
        evidence=(
            Evidence(
                evidence_id="ev-paper-title",
                kind=EvidenceKind.HANDLE_MEMBER,
                statement="The retrieved paper uses this term in its title.",
                handle_id=handle_id,
                member_digest=member_evidence_digest(paper),
                field=field,
                field_value_digest=field_evidence_digest(payload[field]),
            ),
        ),
    )


def test_handle_evidence_reproduces_exact_member_and_field(tmp_path) -> None:
    store = HandleStore(tmp_path)
    paper = _paper("Bounded model checking", "10.1000/bmc")
    handle_id = store.put(searched([paper], query="[model checking]", databases=["arxiv"]))

    HandleEvidenceVerifier(store).verify(_revision(handle_id, paper))


def test_handle_evidence_refuses_member_or_field_drift(tmp_path) -> None:
    store = HandleStore(tmp_path)
    paper = _paper("Bounded model checking", "10.1000/bmc")
    handle_id = store.put(searched([paper], query="[model checking]", databases=["arxiv"]))
    revision = _revision(handle_id, paper)
    evidence = revision.evidence[0]

    with pytest.raises(QueryFamilyEvidenceError, match=r"member digest.*absent"):
        HandleEvidenceVerifier(store).verify(
            dataclasses.replace(
                revision,
                evidence=(dataclasses.replace(evidence, member_digest="qfm_deadbeef"),),
            )
        )
    with pytest.raises(QueryFamilyEvidenceError, match="field digest changed"):
        HandleEvidenceVerifier(store).verify(
            dataclasses.replace(
                revision,
                evidence=(dataclasses.replace(evidence, field_value_digest="qfv_deadbeef"),),
            )
        )


def test_handle_evidence_refuses_unknown_field_and_ambiguous_member(tmp_path) -> None:
    store = HandleStore(tmp_path)
    paper = _paper("A sparse record")
    handle_id = store.put(
        searched([paper, Paper.from_dict(paper.to_dict())], query="[sparse]", databases=["arxiv"])
    )

    with pytest.raises(QueryFamilyEvidenceError, match="byte-equivalent members"):
        HandleEvidenceVerifier(store).verify(_revision(handle_id, paper))

    unique_handle = store.put(searched([paper], query="[sparse]", databases=["semantic_scholar"]))
    revision = _revision(unique_handle, paper)
    malformed = dataclasses.replace(
        revision,
        evidence=(
            dataclasses.replace(
                revision.evidence[0],
                field="invented",
                field_value_digest=field_evidence_digest("invented"),
            ),
        ),
    )
    with pytest.raises(QueryFamilyEvidenceError, match="unknown Paper field"):
        HandleEvidenceVerifier(store).verify(malformed)
