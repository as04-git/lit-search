# Porting a connector into lit-search

Read this before porting a connector from
[`openags/paper-search-mcp`](https://github.com/openags/paper-search-mcp) (pinned at
`234678ab231074a7977320978ee0496dcdaddd1f`) or writing a new one.

The worked reference is **Europe PMC**:

| Piece | Path |
|---|---|
| Connector | `lit_search/connectors/europe_pmc.py` |
| Query builder | `lit_search/query/builders/europe_pmc.py` |
| Tests | `tests/unit/connectors/test_europe_pmc.py` |
| Registration | `lit_search/connectors/__init__.py`, `lit_search/core/paper.py`, `lit_search/query/builders/__init__.py` |

Read those four files alongside this one. This document explains *why*; they show *how*.

---

## 1. The central task is rewriting the error contract

**This is not a translation job.** If you port the donor's control flow and only change the
types, you will produce a connector that reports every upstream outage as a database that
holds no matching paper. That is the exact failure this fork exists to prevent: a literature
review missed papers, and nothing in the output said anything was missing.

Budget your time accordingly. Field mapping is the easy half and it is not where the risk is.
Expect to spend most of the port on: *how does this API fail, and how many of those failures
currently look like success?*

### The pattern you are removing

Donor, `paper_search_mcp/academic_platforms/europepmc.py`, lines 43–93 — abridged, with the
load-bearing lines intact:

```python
def search(self, query: str, max_results: int = 10, **kwargs) -> List[Paper]:
    papers = []
    try:
        params = {'query': query, 'pageSize': min(max_results, 100), ...}
        response = self.session.get(f"{self.BASE_URL}/search", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        result_list = data.get('resultList', {}).get('result', [])
        for item in result_list:
            try:
                paper = self._parse_item(item)
                if paper:
                    papers.append(paper)
                    ...
            except Exception as e:
                logger.warning(f"Error parsing Europe PMC item: {e}")
                continue                      # <-- record lost, count not kept

    except requests.RequestException as e:
        logger.error(f"Europe PMC search request error: {e}")   # <-- falls through
    except Exception as e:
        logger.error(f"Unexpected error in Europe PMC search: {e}")   # <-- falls through

    return papers                             # <-- [] on every path above
```

Four separate outcomes reach the caller as the same value:

1. the database genuinely matched nothing,
2. the endpoint was unreachable,
3. the response shape changed and parsing failed,
4. `max_results` truncated the result at 100 with no pagination.

`return papers` is `[]` for the first three and a silently short list for the fourth. There is
no return value the caller can inspect to tell them apart, and the only trace is a log line.

`CORESearcher.search` has the same shape plus a worse one — `if response is None: return
papers` after three failed retries (`core.py:128-129`). Most donor connectors do this. Assume
yours does until you have read it.

### The pattern you are writing

`lit_search/connectors/europe_pmc.py`, the same decision points:

```python
try:
    payload = self._read_body(self._get(_SEARCH_URL, params))
except (requests.RequestException, ConnectorError) as exc:
    self._record_page_failure(run, exc, cursor)  # re-raises when nothing was retrieved
    break
```

and, inside `_record_page_failure`:

```python
if not run.papers:
    if isinstance(exc, EuropePMCBodyError) and exc.is_query_error:
        raise UnsupportedQueryError(str(exc)) from exc
    raise exc  # base class turns this into FAILED
...  # otherwise: keep the records, record the shortfall
run.truncate(f"page request at cursorMark={cursor} failed: {exc}", type(exc).__name__)
```

The asymmetry is deliberate and it is the whole design:

- **Nothing retrieved** → re-raise. There is no partial result to report, so the failure is
  the entire story and must not be softened into a status the connector invented.
  `SearchConnectorBase.search` catches `requests.RequestException` / `ConnectorError` and
  produces `FAILED`, or `QUERY_UNSUPPORTED` for an `UnsupportedQueryError`.
- **Something retrieved** → keep it, mark `PARTIAL`, carry the error message and the exception
  type. Those records are real; discarding them would be its own kind of dishonesty.

And the parse failures the donor drops on the floor:

```python
run.processed += len(items)  # every record seen
run.parsed += 1  # only the ones that became a Paper
...
unparsed = processed - parsed
if unparsed > 0:
    notes.append(f"{unparsed} of {processed} records could not be parsed into papers")
```

A record the source returned and the connector could not read is a coverage loss, not an
absent paper. If you skip this, a schema change degrades recall invisibly — the connector
keeps returning papers, just fewer of them.

---

## 2. Errors do not always arrive as errors

Before writing a line, probe the live API for the failure modes that come back HTTP 200. This
is where the mechanical port is most dangerous, because `raise_for_status()` is silent and
`data.get("resultList", {}).get("result", [])` returns `[]`.

Three found on Europe PMC on 2026-08-30, all HTTP 200, all indistinguishable from an empty
result under the donor contract:

| Trigger | Body | What a naive port does |
|---|---|---|
| `pageSize=1001` | `{"errCode":404,"errMsg":"Invalid page size provided..."}` | reports 0 papers |
| `cursorMark=GARBAGE!!` | `{"version":"6.9"}` — no `hitCount`, no `resultList` | reports 0 papers |
| empty query | `{"errCode":404,"errMsg":"No search criteria provided..."}` | reports 0 papers |

The handling lives in one place, `_read_body`, which raises rather than returning a default:

```python
if "errCode" in payload or "errMsg" in payload:
    ...raise EuropePMCBodyError(..., is_query_error=<errMsg blames the criteria>)
if "hitCount" not in payload:
    raise EuropePMCBodyError(
        "Europe PMC response carried no 'hitCount', so it is not a search result. "
        "Treating it as an empty result set would report a failure as zero papers."
    )
```

Note the distinction the exception carries. "Your query is malformed" and "the service broke"
lead to different next actions: `FAILED` invites a retry, and retrying a query the source
cannot express only spends rate-limit budget. Only messages that blame the search criteria
become `QUERY_UNSUPPORTED`; anything else — including a bug in your own request shaping — is
`FAILED`, loudly.

**A wrong field name is the same class of trap.** `NOTAFIELD:foo` returns `hitCount: 0` with
HTTP 200. That is why the query builder emits only names checked against the live `/fields`
endpoint, and never passes user text through as a field.

---

## 3. Know how pagination really ends

The donor connectors mostly do not paginate at all; they send one request and cap at 100. If
you keep that, mark the result `CAPPED` or `PARTIAL` — never `COMPLETE`.

If you do paginate, find the *actual* termination signal rather than assuming. Europe PMC has
two, and neither is "a short page":

- a result set that fits one page **omits** `nextCursorMark`;
- a multi-page set ends with an **empty page whose `nextCursorMark` repeats the cursor sent**.

An observed 9-hit query paged `2, 2, 2, 2, 1, 0`. A `len(page) < page_size: break` would have
worked there by luck and would truncate elsewhere.

`_advance` returns the next cursor or `None`, and records a truncation for every stop that is
not the cap or a natural exhaustion — including the pathological "fresh cursor, empty page",
which would otherwise spin forever or stop silently.

**Watch the reported total across pages, not just on the first one.** Europe PMC's `hitCount`
can move mid-run because the index is live. Retrieving 3 of a first-page `hitCount` of 3 would
read as `COMPLETE` even though the set shifted underneath, so drift forces `PARTIAL`.

---

## 4. Choosing the status

`SourceCoverage.from_pagination(...)` already encodes the decision procedure — use it rather
than reimplementing the branches. Reach past it only for the cases it cannot see:

| Situation | Status | Note |
|---|---|---|
| retrieved == reported total | `COMPLETE` | the *only* justified completeness claim |
| stopped at `max_papers` | `CAPPED` | must carry `cap` |
| source publishes no total | `PARTIAL` | `from_pagination` does this for you, deliberately |
| retrieved < reported total | `PARTIAL` | |
| failed after some records | `PARTIAL` | carry `error` and `error_type` |
| records seen but unparsed | `PARTIAL` | with the count in `detail` |
| failed with no records | *raise* | base class → `FAILED` |
| source rejected the query | *raise* `UnsupportedQueryError` | → `QUERY_UNSUPPORTED` |
| credential absent | *raise* `MissingApiKeyError` | → `SKIPPED_NO_CREDENTIAL` |

Two rules that are not negotiable:

- **`COMPLETE` is a positive claim.** Only make it when the source told you how many records
  matched and you retrieved that many. A source that publishes no count can never justify it.
- **If you cannot justify a status, do not invent one.** Subclass
  `UninstrumentedSearchConnectorBase`, which reports everything `PARTIAL`. That is an honest
  placeholder; a guessed `COMPLETE` is the failure this whole design exists to prevent.

### 4b. `COMPLETE` is checked, not accepted

The base class does not take your word for it. `SearchConnectorBase.search` opens a
`RetrievalAudit` around your `_fetch_papers`, records evidence you did not author, and
**downgrades a `COMPLETE` the evidence contradicts** (it never upgrades). Four things follow,
and all four are things a port gets wrong by omission rather than by intent:

- **`SourceCoverage.complete()` requires `reported_total`.** It is keyword-required, and
  `retrieved` must have reached it. `SourceCoverage.complete("x", 3)` is now a `TypeError`, and
  `COMPLETE` with `retrieved=3, reported_total=10` is a `ValueError`. This is a signature
  change: a connector written against the older, looser form will not import.

- **Feed the progress callback with the source's own total.** `progress_callback(items_seen,
  total)` is not only a progress bar — it is the channel through which the base class learns
  what the source published. A `COMPLETE` whose `reported_total` was never seen coming back
  through that channel is downgraded to `PARTIAL`, because a total you computed from your own
  paper count corroborates nothing. Call it at least once, and call it again unconditionally
  after the loop, as `arxiv` and `europe_pmc` both do.

- **`items_seen` must count records *seen*, not papers *kept*.** If you report seeing 100 and
  return 3, the base class reads that as a coverage loss and downgrades. That is correct: it
  is exactly the "dropped 97 records" failure. Report the unparsed count in `shortfall_detail`
  and let `from_pagination` mark it `PARTIAL` — which it now does, instead of silently
  discarding the note in the `COMPLETE` branch as it used to.

- **Do not catch a page failure and claim completeness.** `ConnectorBase._request_with_retry`
  records every request that raises out of it, *before* your `except` clause runs. Catching a
  mid-pagination failure and reporting `COMPLETE` is downgraded to `PARTIAL` with the error
  named. Catching it and honestly reporting `PARTIAL` yourself is fine, and is what Europe PMC
  does. The corollary is that all HTTP must go through `_get` / `_post`; a bare `requests.get`
  escapes the evidence channel as well as the rate limiter.

- **Truncation at `max_papers` is detected structurally.** The cap is the caller's input, so
  the base class checks `len(papers) >= max_papers` itself. At or above the cap, `COMPLETE`
  survives only when the source reported strictly fewer records than the cap allowed;
  otherwise it becomes `CAPPED`. `from_pagination(cap=max_papers)` already gets this right —
  pass the cap.

---

## 5. Rate limit

Take `min_request_interval` from the verified rate-limit table. Do not guess, and do not copy
a number from a neighbouring connector.

If the value is **inferred** rather than documented, say so in the docstring, at the value,
with the reasoning and the date. Europe PMC's:

```python
_MIN_REQUEST_INTERVAL = 1.0
"""...**This value is inferred, not documented.** Europe PMC publishes no rate limit:
neither europepmc.org/developers nor europepmc.org/RestfulWebService states a request
frequency, and a live response carries no X-RateLimit-* or Retry-After headers (checked
2026-08-30)..."""
```

Three more things the base class needs from you:

- **`_credential_attrs`.** The default is `("_api_key",)`. If your connector stores its key
  elsewhere, declare it, or the limiter buckets you as anonymous. If there is no credential,
  declare `()` explicitly — that states you checked rather than overlooked.
- **`rate_limit_service`** when several connectors share one quota. It defaults to `name`,
  which is wrong whenever a service hosts more than one endpoint you might port. Europe PMC
  returns the literal `"europepmc"` so a future Europe PMC annotations or full-text connector
  shares the clock instead of doubling the rate. PubMed returns `"ncbi"` for the same reason.
- **`request_quota`** for budget-shaped services — OpenAIRE (60/hour), CORE (tokens/day),
  Zenodo (30/min on search), OpenAlex (USD/day). Interval pacing alone burns an hourly budget
  in seconds and then returns errors for the rest of the hour, which reads as zero recall.

Never add a `time.sleep` retry loop inside the connector, as `CORESearcher.search` does. Pacing
belongs to the shared cross-process limiter; a private sleep is invisible to it.

---

## 6. Checklist for each port

Nothing here is optional. Items marked **[R2]** are the ones that fail silently if skipped.

**Structure**
- [ ] Subclass `SearchConnectorBase`. Implement `name`, `query_builder`,
      `min_request_interval`, `_fetch_papers`.
- [ ] `_fetch_papers` is annotated `-> CoveredPapers`. Anything else is rejected at import.
- [ ] Do not override `search` — it is `@final` and it is where coverage is stamped.
- [ ] Add a `Database` member, register in `SEARCH_REGISTRY`, export the query builder.
- [ ] Register in `DOI_LOOKUP_REGISTRY` / `URL_LOOKUP_REGISTRY` **only** if you implemented and
      tested those paths. A connector registered for a role it has not implemented is how a
      lookup starts returning `None` for reasons nobody recorded.

**Error contract [R2]**
- [ ] No `except ...: return []`, and no `return papers` reachable from an exception handler.
- [ ] Failure with nothing retrieved re-raises; failure after records reports `PARTIAL`.
- [ ] Every HTTP-200-with-an-error-body shape is detected and raised, not defaulted away.
- [ ] Unparsed records are counted and reported, not skipped.
- [ ] `COMPLETE` is returned only against a source-reported total (now enforced: the
      `reported_total` argument is required and must have been reached).
- [ ] `progress_callback(items_seen, total)` is called with the source's own total, at least
      once inside the loop and unconditionally after it. Without it a `COMPLETE` is downgraded
      to `PARTIAL`, because nothing corroborates the total. See §4b.
- [ ] `items_seen` counts records *seen*, not papers *kept*.
- [ ] All HTTP goes through `_get` / `_post`, never a bare `requests.get`.
- [ ] `max_papers` is passed to `from_pagination` as `cap`.

**Query builder [R2]**
- [ ] Every emitted field name verified against the source's own field list. A wrong name is
      usually a silent zero, not an error.
- [ ] A query the source would run but not scope correctly is rejected in `validate_query`,
      not sent. (Europe PMC: `TITLE_ABS:machine learn*` returned 249,661 hits against 378,639
      for `TITLE_ABS:machine` — the second token escaped the field, and the query looked fine.)

**Rate limit**
- [ ] `min_request_interval` from the verified table, marked inferred where it is inferred.
- [ ] `_credential_attrs` declared. `rate_limit_service` set if the quota is shared.
- [ ] `request_quota` set for budget-shaped services. No `time.sleep` retry loop.

**Tests** (`pytest-socket` blocks the network; mock every request)
- [ ] a normal result → `COMPLETE`
- [ ] an upstream failure → `FAILED`
- [ ] an empty-but-successful search → `COMPLETE`, zero papers
- [ ] a capped result → `CAPPED`, with `cap` set
- [ ] **the failure and the empty case assert different statuses in one test.** This is the
      regression test for the port. Everything else can pass while the connector still reports
      an outage as zero; this cannot.
- [ ] each HTTP-200 error body → `FAILED` (or `QUERY_UNSUPPORTED`), never empty
- [ ] a mid-pagination failure → `PARTIAL` with the records kept
- [ ] a rejected query reaches the ledger as `QUERY_UNSUPPORTED` and issues no request
- [ ] a swallowed mid-pagination failure that *claims* `COMPLETE` is downgraded to `PARTIAL`
      by the base class. Write this one even though the base class does the work: it is the
      test that proves your connector's HTTP goes through `_get`.
- [ ] **mutation-check your own tests.** Reintroduce the donor's swallow, run the suite, and
      confirm it goes red. If it stays green the tests are decorative. For the Europe PMC port
      this turned 7 tests red, including the one named above.

**Gates** — run all of them, they are Makefile targets and CI steps:
```
.venv/bin/ruff check findpapers tests
.venv/bin/ruff format --check findpapers tests
MYPYPATH=typings .venv/bin/mypy findpapers tests/unit
.venv/bin/xenon --max-absolute C --max-average B findpapers
.venv/bin/interrogate findpapers/
.venv/bin/python -m pytest
```
`xenon` is the one people miss. A `_fetch_papers` that tracks every stop reason inline will
rank D and fail the gate; split the pagination bookkeeping into a small state object and a
cursor-advance helper, as `_Retrieval` / `_advance` do.

**Do not do this**
- [ ] Do not bump a hard-coded count to make a test pass (`assert len(Database) == 8`). If a
      test breaks only because a database now exists, the test was asserting the wrong thing —
      rewrite it against `SEARCH_REGISTRY`. Bumping the number teaches the next porter that
      this file is bookkeeping rather than a decision.

---

## 7. Read the source list before you start

Not every donor module is a search connector, and the classification is in
the project's design notes. Briefly:

- **Unpaywall** and **Sci-Hub** are not search sources. Registering them as queryable
  databases makes an ordinary Boolean query return zero, silently.
- **bioRxiv / medRxiv** turn the query into a category string and only consult the last 30
  days. Either drive them properly or mark every result `PARTIAL` — never `COMPLETE`.
- **BASE** filters locally over OAI-PMH prefixes and can return an order-biased sample. It
  also requires IP registration; until that exists its status is `SKIPPED_NO_CREDENTIAL`.
- **Google Scholar** returns whatever it accumulated when it hits a CAPTCHA. That is directly
  incompatible with the contract and needs a policy read besides.
- **CiteSeerX is dead.** Every path 301s to a Wayback snapshot; a port would parse Wayback's
  error pages and report zero. Do not port it.
- **ACM** and **IEEE** in the donor tree are stubs (`# TODO: implement real ACM DL API call
  here`), and findpapers already has a real IEEE connector.
