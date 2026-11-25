# Snowball traversal

`Engine.snowball()` expands one paper or a whole retrieved set through source-native citation
edges. The current default follows both directions to depth two. It is an exhaustive-style BFS,
not the planned targeted interactive mode, and a large seed handle can take far longer than ten
minutes.

```python
import lit_search

engine = findpapers.Engine()
searched = engine.search("[specification inference]", databases=["semantic_scholar"])
result = engine.snowball(searched.papers, enrichment_databases=[])

print(len(result.papers))      # discoveries only
print(len(result.all_papers))  # seeds plus discoveries
print(result.render_coverage())
```

The CLI prefers a handle because it preserves the complete seed set and lineage:

```bash
lit-search snowball h_1234abcd --handle --no-enrich
```

A DOI or landing-page URL remains valid for a focused traversal.

## Managed request/page resume

For expensive work, start a managed run:

```bash
lit-search snowball h_1234abcd --managed-run --no-enrich \
  -d semantic_scholar --store-root .lit-search
```

The command prints an `r_<digest>` ID to stderr before provider work. If interrupted, repeat the
same semantic invocation with that ID:

```bash
lit-search snowball h_1234abcd \
  --resume-run r_0123456789abcdef0123456789abcdef \
  --no-enrich -d semantic_scholar --store-root .lit-search
```

Resume validates the complete semantic spec and non-secret credential fingerprints. Changing the
seed payload, sources, direction, depth, caps, date filters, enrichment, worker count, SSL policy,
or credential identity refuses resume; create a new run instead.

Native graph work checkpoints actual provider requests/pages before I/O and commits their typed
outcomes before local reduction. Completed units replay without another request; an interrupted
in-flight unit is retried. The source-snapshot cache accepts only fresh, complete, directional
evidence for the exhaustive path. Partial, truncated, or stale observations remain visible but do
not satisfy the request.

Legacy composite `GetRunner` and web-enrichment work is still checkpointed at the coarser runner-
operation boundary and may repeat connector-internal work after interruption. That limitation is
part of the result contract, not hidden by the native request/page guarantee.

Operational state lives under:

```text
<store-root>/runs/<run-id>/run.sqlite
<store-root>/graph-cache/cache.sqlite
```

The shared cache defaults to 1 GiB and refuses configuration above 5 GiB. It is disposable
acceleration, separate from immutable result handles.

## Parameters

```python
result = engine.snowball(
    paper_or_papers,
    max_depth=2,
    direction="both",
    max_papers_per_level=None,
    max_expansion_per_level=None,
    max_cited_by=100,
    databases=["semantic_scholar"],
    enrichment_databases=None,
    since=None,
    until=None,
    num_workers=1,
)
```

| Parameter | Meaning |
|---|---|
| `paper_or_papers` | One `Paper` or non-empty sequence with at least one typed identifier. |
| `max_depth` | BFS depth; default 2. |
| `direction` | `backward`, `forward`, or `both`. |
| `max_papers_per_level` | Result-membership cap. A paper outside it may still serve as a bridge. |
| `max_expansion_per_level` | Frontier-expansion cap. Suppressed work remains a policy residual. |
| `max_cited_by` | Forward-edge cap where a source paginates citers. Binding caps remain visible. |
| `databases` | Graph providers: Crossref, OpenAlex, and Semantic Scholar. |
| `enrichment_databases` | Metadata-only enrichment after graph traversal; `[]` disables it. |

`Engine.snowball()` additionally accepts `managed_store_root`, `managed_run_id`, and
`resume_managed_run` for programmatic lifecycle control. The CLI generates the run ID before work
so it can be recovered after interruption.

## Typed edges and identity

Native traversal uses:

- `PaperIdentifier` for DOI, arXiv, Semantic Scholar, and OpenAlex identities;
- `CitationTarget` for one target with multiple asserted aliases;
- `CitationEdge` for provenance and direction;
- source-native batches for acquisition.

Routing preserves DOI-less native edges. arXiv IDs and `10.48550/arXiv.*` DOIs are sent to arXiv,
not Crossref. Target coalescing follows strict primary-version identity: a non-synthetic DOI is
preferred, otherwise a canonical arXiv ID; provider aliases only identify two unversioned records.
A publisher DOI record that also carries its arXiv alias coalesces with the DOI version, not with
the arXiv-only version.

## Deterministic traversal flow

`lit_search.snowball.flow` now separates pure planning, routing, acquisition intent, current cap
policy, and level reduction from the impure runner adapter. This permits frozen replay and policy
falsification without network work. The current policy preserves the existing BFS behavior.

Targeted lane selection, persistent policy decisions/residuals, quota brokerage, and moving the
remaining native execution adapter out of `SnowballRunner` are deferred until the frozen
adversarial benchmark exists.

## Coverage and termination

`SnowballResult.coverage` distinguishes:

- provider request or parse failure;
- explicit not-found;
- missing or unroutable identifiers;
- edge-count truncation;
- planned result/frontier caps;
- source acquisition coverage versus policy-suppressed residuals.

An empty frontier is closure evidence only when all required work was acquired completely and no
failed, stale, truncated, deferred, unroutable, pending, or policy-suppressed target remains under
the declared exhaustive contract. A bounded run may complete successfully without claiming
closure.

## Result shape

| Attribute | Meaning |
|---|---|
| `seed_papers` | All enriched seeds. |
| `seed_paper` | Compatibility alias for the first seed. |
| `papers` | Newly discovered records only. |
| `all_papers` | Seeds plus discoveries under strict record identity. |
| `coverage` | Acquisition and policy coverage ledger. |
| `run_id` | Managed operational run ID, or `None` for an unmanaged traversal. |

Serialization retains typed edges, seed records, coverage, and the managed run reference.
Immutable snowball handles store `all_papers` and name the seed handle as lineage.
