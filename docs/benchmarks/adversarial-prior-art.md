# Adversarial prior-art benchmark contract

Status: **selection protocol frozen; cases chosen and frozen by each user**

Updated: 2026-08-31

This benchmark asks whether the search process can acquire critical prior art under distinct
failure mechanisms. It is not a leaderboard for one query, one citation graph, or one reranker.
The benchmark is defined before policy tuning so that one development case cannot keep sculpting the architecture.

## What is graded

Grade these dimensions separately:

- **acquisition:** whether each critical paper entered the candidate union at all;
- **surfacing:** its best rank at fixed cutoffs after an identical reranking budget;
- **execution:** whether every declared query cell or graph work unit completed as planned;
- **scope:** declared sources, query families, graph directions/depth, freshness, and caps;
- **cost:** cold and self-warm wall time, provider requests/pages, protected quota, and cache use;
- **explanation:** the exact query revision or traversal evidence that introduced the paper;
- **identity safety:** whether versions and same-title records remained correctly distinct.

Execution completeness never substitutes for acquisition. A completely executed search that
finds zero critical papers is a valid, failed-quality observation.

## Frozen strata

Freeze approximately 8–12 cases. Every case must name one primary adversarial stratum and may
carry secondary tags. At least two cases must satisfy the citation-disconnected stratum.

1. **Citation-recoverable vocabulary shift (development only).** A retrieved paper
   exposes a citation path to the critical one even though the vocabularies differ. Keep such a
   case as a regression, never as the only quality gate.
2. **No usable citation path (mandatory held-out).** No initially retrieved relevant member has a
   usable path to the critical paper within the frozen sources, directions, depth, freshness, and
   request budget. State this scoped fact; do not claim global graph disconnection.
3. **Recent and effectively uncited.** The critical paper is too new for citation traversal to be
   a credible primary discovery mechanism at the freeze date.
4. **Terminology/community shift.** The critical paper uses a foreign disciplinary vocabulary;
   the initial user formulation must not already contain its answer-key terms.
5. **Aggregator absence or asymmetry.** At least one major provider omits or materially
   misrepresents the critical record, so source complementarity is tested.
6. **Actor/venue/classification discovery.** Text similarity alone is weak; author, venue,
   controlled vocabulary, subject, or topic classification supplies the useful lane.
7. **Identifier/version stress.** DOI-less records, preprint/publisher versions, corrected or
   retracted records, and conflicting aggregator aliases test non-destructive identity.
8. **Ambiguous language.** A high-volume irrelevant sense competes with the intended sense. The
   “contract” versus blockchain literature is the development seed; include a relevant smart-
   contract/formal-verification paper so a naïve global negative also fails.
9. **Question-model revision.** Relevance becomes apparent only after retrieved evidence changes
   the searcher's understanding of the research question; the next query family must cite that
   evidence explicitly.

## Case-freezing protocol

For each case, version:

- the research question and the user's initial formulation;
- the initial accepted query-family revision, with answer-key terminology prohibited unless the
  user actually supplied it;
- critical papers and acceptable version identities;
- the freeze timestamp and source snapshots or replay fixtures;
- scoped citation reachability evidence from the initial relevant set;
- provider presence/absence observations and their coverage status;
- relevance rationale written before the system run;
- allowed budgets and the grader version.

Keep case selection and critical-paper grading hidden from policy tuning where practical. A case
is not held out merely because its paper title is absent from a prompt; answer-derived vocabulary,
citations, or source choices also leak the answer.

## Mechanism-isolation tracks

Run every frozen case under equal acquisition and reranking budgets:

1. original single query;
2. semantic reranking only (membership must remain identical to track 1);
3. structural traversal only from the original candidate set;
4. typed query-family revisions only;
5. query-family plus structural traversal;
6. the final integrated policy.

Use the same identity reducer and reranker across tracks. A selector must not inspect an
unexpanded candidate's hidden outgoing graph. Frozen replay decides policy comparisons; live
cold/self-warm runs validate latency and provider behavior for finalists.

## Required falsifiers

The suite must fail an implementation that:

- inserts critical-paper terminology into the initial revision;
- calls reranking a recall improvement while membership is unchanged;
- treats an empty or failed graph response as frontier exhaustion;
- reports closure with failed, stale, truncated, deferred, unroutable, or pending required work;
- lets a planned source exclusion trigger the same exit signal as a provider failure;
- merges different DOI versions or unrelated DOI-less same-title records;
- silently falls back from an invalid source-specific query translation;
- executes an unreviewed model-generated term proposal;
- collapses repeated variants against one source into one source-level coverage record;
- tunes on its development case and passes only citation-recoverable cases.

## Provisional acceptance reporting

Report the Pareto frontier rather than silently weakening the target. The normal interactive goal
remains cold-process p90 at or below 600 seconds from an existing handle or accepted query-family
revision to a persisted, reranked, coverage-bearing result. If no policy recovers every critical
paper inside that budget, report acquisition/surfacing by stratum, residual work, and the next
most valuable round; do not relabel a bounded run exhaustive.

The final live acceptance run is user-executed. Before handoff, offline replay, deterministic
serialization, interruption equivalence, coverage/scope rendering, and identity falsifiers must
already pass.

## Executable manifest contract

`lit_search.evaluation` implements this protocol as strict, versioned JSON objects. A frozen suite
contains 8--12 cases, covers every held-out stratum above, and includes at least two cases tagged
`citation_disconnected`. A disconnected tag is valid only with scoped `no_usable_path` evidence.
Critical targets require acceptable publication-version identifiers and digests for the relevance
evidence written before the run.

The suite freeze digest covers every case-definition field. The scorer first recomputes it, then
requires the run to name that exact digest. Editing a question, target, budget, snapshot, stratum,
or evidence reference after freezing therefore makes both validation and scoring fail closed.
The manifest also freezes the exhaustive flattened terms visible in the initial query-family
revision (including source translations); prohibited answer-key terms are rejected across those
terms, the research question, and the user's initial formulation.

Acquisition and ranking are separate objects in a run manifest. An acquired target must name the
observed identifier that matches its frozen acceptable-version set, and discovery evidence names
the pre-ranking operation and mechanism that introduced it. Ranking has only rank/surfacing state
and cannot add candidate membership. In the semantic-ranking-only track, the baseline and output
candidate-union digests must match exactly.

Scores retain three acquisition states: `acquired`, `not_acquired`, and `unknown`. With any unknown
target, the report exposes lower and upper recall bounds and withholds point recall. Execution uses
typed issues for source, query-cell, graph-unit, identity, and ranking failures or unknowns; these
are reported separately from quality.

[`synthetic-case.template.json`](synthetic-case.template.json) is deliberately fictional. It shows
the strict case shape but is not a held-out case, evidence bundle, or complete 8--12-case suite.
Replace every synthetic value, assemble user-selected cases, instantiate `BenchmarkSuite`, call
`with_computed_freeze_digest()` once at the freeze boundary, and persist `suite.to_dict()`. Loading
with `BenchmarkSuite.read_json()` verifies the freeze; `score_run(suite, run)` refuses any drift.
