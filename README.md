# lit-search

Literature hunts an agent can drive and a person can audit.

You give an agent (or yourself) a research question and a time budget. The agent runs a **hunt**:
it searches, follows citations, reads, and judges, while `lit-search` keeps the books. Every
candidate paper has a short id, every route that reached it, the citation sentences that mention
it, and the agent's judgment with a reason. The report is a shortlist you can check, plus an
honest account of what was searched and what was not.

That last part is the point. Literature tools fail silently: a source that was rate-limited,
capped, or could not parse the query returns an empty list that looks exactly like "nothing
exists". In lit-search every step carries typed coverage (`complete`, `capped`, `partial`,
`failed`, `skipped_no_credential`, ...), so a failure is a visible fact, never a missing result.

```text
$ lit-search hunt start "has anyone made linear RNNs track state by changing the recurrent transition?" --profile prior-art
hunt hunt-20261125-0108-31ce · profile prior-art · budget 10 min · shortlist 10 · report opens at 80%

$ lit-search hunt find hunt-20261125-0108-31ce linear RNN state tracking parity --brief
move 1 find text=linear RNN state tracking parity · 3/5 sources answered · 10.3s · +27 new, 0 again
  FAIL semantic_scholar: HTTPError: 429 Client Error:  for url: https://api.semanticscholar.org/graph/v1/paper/search?...
  FAIL openaire: OpenAIREAuthenticationError: OpenAIRE access-token refresh failed with HTTP 400 (HTTPError).
  c1   Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues
       2025-05 · arXiv:2411.12537 · cited 60
  ...

$ lit-search hunt search hunt-20261125-0108-31ce 'abs[state tracking] AND abs[linear RNN]' --brief
move 2 search query=abs[state tracking] AND abs[linear RNN] · 2/2 sources answered · 0.4s · +6 new, 9 again (c27 c16 c2 ...)
```

## How it fits together

```
           agent / you                                    programs
               │                                              │
     ┌─────────▼──────────────────────────────┐               │
     │ hunt: the workspace                    │               │
     │ question · budget · candidates c1..cN  │   handles     │
     │ routes · evidence · marks · notes      │◄─────────────►│
     │ moves: find related snippets snowball  │ (stored sets  │
     │        add search read                 │  of papers)   │
     └─────────┬──────────────────────────────┘               │
               │                                   building blocks: search, snowball,
     sources: Semantic Scholar, OpenAlex, arXiv,   set algebra, rerank, index, export
     OpenAIRE, Hugging Face, DBLP, PubMed, ... 25
```

- **The hunt is the workspace.** It holds the question, the budget, and every candidate. Moves
  add candidates; `mark`, `rank` and `note` record judgment; `report` writes the shortlist.
  The CLI enforces a few rules that measurably changed agent behaviour (the report opens only
  after 80% of the budget is used; a repeated move is refused) and leaves reasoning to the model.
- **Moves are the expert search methods.** `find` (relevance search across sources), `related`
  (recommendations plus one citation hop), `snippets` (full-text sentence search), `snowball`
  (every citation edge of a seed, forward or backward, with the citing sentences), `add` (papers
  you already know), `search` (an exact-term Boolean sweep), and `read` (a candidate's intro or
  related-work section, from its PDF).
- **Handles connect hunts to everything else.** A handle is an immutable, content-addressed set
  of papers with its coverage and lineage. Each report writes two (shortlist and all
  candidates). `hunt add --from HANDLE` starts from a set; `hunt start --exclude HANDLE` leaves
  out papers you already know, so a follow-up hunt shows only what is new. The building-block
  commands read and write handles too: an exhaustive resumable snowball, set algebra,
  embedding re-ranking, BibTeX export.
- **`find`, `related` and `snippets` also run on their own**, as quick lookups without a
  session.

## Install

Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/as04-git/lit-search.git && cd lit-search
make setup                     # creates .venv; EXTRAS=rerank adds the local re-ranker (torch)
make install-cli               # links ~/.local/bin/lit-search
lit-search doctor              # checks each source with one small request; prints fixes
```

No key is required to start (arXiv and Hugging Face need none). A free
[Semantic Scholar key](https://www.semanticscholar.org/product/api) makes hunts far more
reliable. Add keys with `lit-search config set SEMANTIC_SCHOLAR_API_KEY --stdin` or
`lit-search config import keys.env`; `lit-search config list` shows every setting.
A [Voyage AI](https://dash.voyageai.com) key (`VOYAGE_API_KEY`; the first 200M rerank tokens
are free) makes every hunt move list its new candidates by relevance to the question; without
it, `tools/rerank_server.py` serves a local cross-encoder, or moves keep the hunt's own order.
Set `LIT_SEARCH_EMAIL` so the User-Agent carries your contact address.

## Use

Quick lookups:

```bash
lit-search find state tracking in linear attention          # ~3-4 s, relevance-fused
lit-search related 2404.08819 --not 2306.09340               # recommendations, steer away
lit-search snippets diagonal recurrences fail on parity     # papers whose text says this
lit-search fetch 2404.08819 10.1038/nature14539              # PDFs; every attempt recorded
```

A hunt:

```bash
lit-search hunt start "has X been done?" --profile prior-art   # prior-art | orient | frontier | known-item
lit-search hunt add HUNT 2404.08819 "Exact title of a paper you know"
lit-search hunt snowball HUNT c1 --forward --since 2025-01-01
lit-search hunt find HUNT a mechanism in the field's own words
lit-search hunt mark HUNT c4 c9 --keep --note "same mechanism, different name"
lit-search hunt status HUNT --unjudged
lit-search hunt report HUNT                                     # writes report.md and two handles
```

Hunts are designed for an agent to drive. [`docs/hunt-playbook.md`](docs/hunt-playbook.md) is the
short prompt we give hunting agents; any agent that can run shell commands can use it. Every
command takes `--json`.

Exit codes: 0 when the operation ran (bounded scope stays visible in the output), 1 when it
could not run, 2 for usage errors, 3 when part of it fell short (a source failed, a budget was
spent).

## Why it is built this way

It started from a literature review that missed three closely related papers despite several
search passes. The searches used one community's vocabulary while the papers described
themselves in another field's terms. Re-ranking could not recover papers that never entered the
candidate set, and a citation path that would have reached them was lost because the tool
dropped records without a DOI. Hence:

1. **Several routes, not one.** Keyword search, citation traversal, recommendations and full-text
   snippets reach different papers; none substitutes for the others. Starting from papers the
   agent already knows and snowballing from them is often what finds the target.
2. **Coverage is data.** A result set says which sources answered, which failed and why, and
   what caps bounded it.
3. **Identity is narrow.** Records merge only on exact, version-aware identifiers; title matches
   and embeddings are evidence, never a destructive merge. Preprint and published versions stay
   linked rather than collapsed.
4. **Measure, then change.** `lit-search eval usage` summarises saved hunts (failures per
   source, which moves first reached kept papers); `eval replay` and `eval bench` re-run recorded
   moves against graded cases in [`docs/confidence/cases.json`](docs/confidence/cases.json).
   One finding that shaped the tool: ordering each move's new candidates by a cross-encoder put
   target papers in an agent's first ten 81% of the time, against 43% for the default order
   (Voyage's hosted reranker by default, or `tools/rerank_server.py` locally).

## More

- [`docs/hunt-playbook.md`](docs/hunt-playbook.md): the agent prompt and every hunt command;
  [`docs/hunt-playbook-strict.md`](docs/hunt-playbook-strict.md) for a stricter variant.
- [`docs/search.md`](docs/search.md), [`docs/query-syntax.md`](docs/query-syntax.md): Boolean
  search across 25 sources. [`docs/databases.md`](docs/databases.md) lists them.
- [`docs/snowball.md`](docs/snowball.md): exhaustive, resumable citation traversal.
- [`docs/configuration.md`](docs/configuration.md): credentials and settings.
- [`docs/api-reference.md`](docs/api-reference.md): the Python API.
- [`docs/benchmarks/adversarial-prior-art.md`](docs/benchmarks/adversarial-prior-art.md): a
  protocol for benchmarks that reward finding papers, not ranking them.

Status: research software, used daily by its author. Interfaces may still change.

## Development

```bash
make lint && make types
uv run pytest -q tests/unit --ignore=tests/unit/index
```

## License and origin

MIT. lit-search began as a fork of
[Findpapers](https://github.com/jonatasgrosman/findpapers) by Jonatas Grosman and includes code
adapted from [paper-search-mcp](https://github.com/openags/paper-search-mcp); see
[`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
