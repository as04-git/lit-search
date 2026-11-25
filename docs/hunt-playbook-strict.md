# Strict hunt playbook

The strict variant of `docs/hunt-playbook.md`. Start hunts with `--strict`; the CLI then refuses
moves and the report until the reasoning and inspection steps below are on record. Follow this
file alone; where it differs from the standard playbook, this file wins.

The failure this exists to prevent: a search that looks thorough but never left the question's own
vocabulary, kept everything it saw, and judged papers by their titles. The paper that kills a
novelty claim usually calls the idea something else.

## 0. Think before searching (about 1 minute, no tool calls)

Write down, for yourself:

1. **The central move** of the question in one sentence: what would a paper have to do to make
   the user's idea not new?
2. **2–4 sub-claims** (A, B, C, ...) that together make up the idea, each phrased as a concrete mechanism (operation + artifact), see "Mechanism queries" below.
3. **At least 5 research communities** (prior-art) or 3 (other profiles) that could have done
   the central move under their own names: the obvious one, more theoretical ones, more applied
   ones, and fields that solve the same problem for a different reason. For each: how *they*
   would title a paper on it.

Then start and record the plan:

```bash
lit-search hunt start "<question>" --profile prior-art --minutes 10 --strict
lit-search hunt note ID --plan "Central move: ...
Sub-claims: A ... ; B ... ; C ...
- <community 1>: <their title for it>
- <community 2>: <their title for it>
- <community 3>: ...
- <community 4>: ...
- <community 5>: ..."
```

The plan must contain at least three lines that start with `- `.

### Mechanism queries (required for prior-art)

Search engines reward one specific mechanism, not a composite idea. A query about the whole idea
("multi-source evidence specification inference with uncertainty") lands in a crowded middle of
loosely related papers; the paper that kills a novelty claim usually does **one** of its
mechanisms and names that mechanism concretely.

1. Break the idea into its concrete mechanisms: each is one operation on one artifact (for
   example "derive specifications from a function's callers", "check generated specifications by
   model checking", "resolve conflicts between documentation and tests").
2. For each mechanism, write the query as *operation + artifact + method*, 3–7 words, and also
   the version the most technical neighbouring community would write (the method by its proper
   name: e.g. a specific logic, checker, solver, or statistical technique).
3. Run one `find` per mechanism (both versions if time allows) before any `related` hop.
4. A paper that does one mechanism directly and well is more important than a paper that gestures
   at all of them. Before marking anything core, check whether your kept papers include one
   that does a single mechanism *exactly*; read its abstract with `hunt show`.

## 1. The move loop

Every move follows the same three steps. The CLI blocks the next move until steps 2 and 3 are done.

1. **Run one move** (`find`, `related` or `snippets`).
2. **Inspect the top new candidates.** The CLI names the ones you must judge (the top 5 new by
   score). For each: `hunt show ID cN`, read the abstract, then
   `hunt mark ID cN --core|--keep|--drop --note "<sub-claim(s) it covers>: <a short phrase quoted from the abstract>"`.
   - `--core`: it does the central move (or most of it), whatever it calls it.
   - `--keep`: it covers at least one sub-claim.
   - `--drop`: neither.
   Judge from the abstract, never the title.
3. **Write the move note:** `hunt note ID "Learned: ... | Sub-claims: A covered by c3; B none yet; C partly c7 | Next: <move> because <reason>"`.

## 2. Required move mix (prior-art)

Plan for all of these within the budget; the report is refused without the first three.

- `find` in **at least 3 different communities** from your plan, in their own words (3–7
  content words each). Aim for all 5.
- **Snowballing in both directions** (`hunt snowball`): backward from your most recent
  core/kept paper, forward (with `--since`) from an older one; then snowball every core paper
  both ways (the report is refused until you do).
- **1 `snippets`** with a sentence a matching paper would literally contain ("We propose ...").
- **1 semantic paraphrase:** `hunt find ID "<plain-language description of the central move, no jargon>" -d huggingface -k 40`.
- **1 recent sweep:** `hunt find ID "<best community phrasing>" --since <6 months ago>`.

Orient, frontier and known-item: follow the profile guide printed by `hunt start`, with the same
inspect-and-note loop.

## Proven practices (from how expert searchers and systematic reviews work)

- **Write the killer abstract first.** Draft, in 3–4 sentences, the abstract of the paper that
  would make the user's idea not new. Use its sentences for `snippets` and for the semantic
  source (`hunt find ID "<one sentence>" -d huggingface -k 40`). Papers are found by how *they*
  describe themselves; this is the closest you can get before you have seen them.
- **One good paper beats twenty queries.** As soon as one paper is clearly on target, mine it:
  `related` from it, and read the **related-work section of its PDF**; authors there list exactly
  the competing work, often in vocabulary you have not tried:
  `lit-search fetch <arXiv id or DOI> -o /tmp/rw && pdftotext /tmp/rw/<file>.pdf - | sed -n '/Related Work/,/^[0-9]* *Conclusion/p' | head -150`
  (do this for at most 2 papers; it costs about a minute each).
- **Surveys are vocabulary maps.** In an unfamiliar community, first `find "<topic> survey"`; a
  recent survey's section headings tell you what that community calls things.
- **Follow people.** If two kept papers share an author or lab, `find` with that author's name
  plus the topic; groups publish follow-ups under new names.
- **Recency has its own sweep.** Indexes lag by weeks to months. Run one `find --since <3 months
  ago>` per main mechanism; arXiv and Hugging Face are the freshest sources.
- **Saturation is your stop signal.** The move line prints `+N new, M again`. When moves in
  *different* vocabularies keep returning mostly `again` candidates and no new keeps, coverage is
  likely good. When a new vocabulary returns mostly new candidates, that community was unexplored:
  keep going there.
- **Verify before you declare.** Before calling a paper core, read its abstract with `hunt show`;
  if it is the deciding paper, skim its introduction's contribution list in the PDF.
- **Name variants and acronyms.** Search the obvious system/benchmark names the idea would get,
  and acronyms ("BMC", "SMT", "RAG", ...) next to the mechanism.

## 3. Adversarial check (before the report)

Write a note answering: "If this idea has already been done, what are three likely titles of
that paper?" Search the two that use vocabulary you have not tried yet. Inspect and mark as usual.

## 4. Close

1. `hunt status ID --unjudged --top 20`, and judge everything it lists.
2. Final note, one line per sub-claim:
   `hunt note ID --final "A: covered by c3 (how), c9; B: closest c7, which only ...; C: not found in these moves. Closest overall: c3. Not covered: <communities/vocabularies not reached>."`
3. `hunt report ID --shortlist N` with N set by what you found (core papers first). The report
   opens at 80% of the budget (known-item excepted); spend spare time reading your best candidates
   more deeply (`hunt read`) and set the final order with `hunt rank`.

If the CLI refuses a move or the report, do what the refusal names. Do not work around it.

## What to return

- The report path and the shortlist exactly as printed.
- Your per-sub-claim verdict (the final note).
- The communities you searched and the ones you did not reach.

Never say an idea does not exist in the literature. Say what was searched and what was not.

## Wide hunts (`hunt start ... --wide`)

Moves return 100–250 candidates instead of ~30 (`find` 50 per source, `related` 50
recommendations + 120 neighbours, `snippets` 60), each as a title line plus a one-line abstract
excerpt, ordered by score. **Read every line.** That is the point of wide mode: the paper that
matters is often far down a list or phrased in another community's words. Mark what matters
(`--core`/`--keep`); you do not need to drop the rest. Run `hunt show` on anything whose excerpt
might do the central move before deciding.
