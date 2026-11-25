# Configuration

Findpapers is configured through the `Engine` constructor. Credentials and source-specific
settings can be passed explicitly or supplied through environment variables. Explicit
constructor arguments take precedence.

## Setup

lit-search reads each credential from the environment, or else from its credentials file
(`~/.config/lit-search/credentials.env`, plain `NAME=value` lines, mode 600; `lit-search config
path` prints it). On a new machine, one of:

```bash
lit-search config import ~/.config/api-keys.env   # copy every recognised key from a file
lit-search config set SEMANTIC_SCHOLAR_API_KEY --stdin < key.txt   # or one at a time
lit-search doctor                                  # what is set, and does each key work
```

`config import` reads `NAME=value` files without executing them and also converts the old
`FINDPAPERS_*` names. `config list` shows every setting, where it comes from and what it is for
(values redacted). `doctor` makes one small authenticated request per configured source and
prints the fix for anything missing, rejected or expired (OpenAIRE refresh tokens last a month).

## Common credentials

Most open sources work anonymously, but credentials can raise their quota or unlock an
otherwise unavailable source. Pass them once when constructing the engine:

```python
import lit_search

engine = findpapers.Engine(
    semantic_scholar_api_key="your-s2-key",
    openalex_api_key="your-openalex-key",
    pubmed_api_key="your-ncbi-key",
    ncbi_tool="my-literature-review",
    ncbi_email="researcher@university.edu",
    email="researcher@university.edu",
)
```

Semantic Scholar and OpenAlex keys are optional but useful. The NCBI key, tool name, and
contact email are shared by PubMed and PubMed Central. `email` is the general contact identity
sent to Crossref through its official `mailto` request parameter; `ncbi_email` and
`chemrxiv_email` override it for those
providers.

> **Tip:** Semantic Scholar, OpenAlex, and NCBI keys are useful free credentials. IEEE,
> Scopus, and Web of Science access depends on the user's entitlement and should be configured
> only when that source is intentionally used. Missing credentials remain visible in coverage.

## Additional source configuration

The less commonly needed source options remain available through the same facade:

```python
engine = findpapers.Engine(
    core_api_key="your-core-key",  # optional; CORE also supports anonymous use
    zenodo_access_token="your-zenodo-token",  # optional; anonymous search remains usable
    chemrxiv_email="chem-contact@example.edu",
    ssrn_api_key="your-ssrn-key",  # required for SSRN free-text search
    openreview_username="user@example.edu",
    openreview_password="...",
    openreview_venue_ids=["ICLR.cc/2026/Conference"],
)
```

OpenReview's venue list and cache controls are operation boundaries, not merely performance knobs:
`openreview_venue_ids` selects the proceedings searched, while `openreview_cache_dir` and
`openreview_cache_ttl_seconds` control reuse of harvested venue metadata.

### OpenAIRE renewable authentication

OpenAIRE can run anonymously, but its anonymous request allowance is intentionally paced very
conservatively. For routine use, either supply a current one-hour access token or configure the
refresh flow:

```python
engine = findpapers.Engine(
    openaire_access_token="current-access-token",  # optional when refresh data is supplied
    openaire_refresh_token="one-month-refresh-token",
    openaire_access_token_expires_in=3200,  # remaining lifetime, if both tokens are supplied
)
```

The refresh token is used to obtain a replacement access token when necessary. Findpapers does
not write refreshed secrets back to your environment or configuration file. A token or refresh
failure is reported as a source failure; it is never converted into an empty OpenAIRE result.
When both tokens are supplied without `openaire_access_token_expires_in`, Findpapers refreshes on
first use rather than guessing the independently acquired access token's remaining lifetime.

## Proxy configuration

To route all HTTP requests through a proxy:

```python
engine = findpapers.Engine(proxy="http://proxy.example.com:8080")
```

The same proxy URL is used for both HTTP and HTTPS destinations across engine-managed remote
search, lookup, enrichment, and download transports. Local arXiv snapshots and Zotero's local
API do not need an external proxy. The setting may also be supplied with `LIT_SEARCH_PROXY`.

## SSL verification

SSL certificate verification is enabled by default. To disable it (e.g., behind a corporate proxy with a custom CA):

```python
engine = findpapers.Engine(ssl_verify=False)
```

> **Warning:** Disabling SSL verification makes requests vulnerable to man-in-the-middle attacks. Only use this in trusted network environments.

Set `LIT_SEARCH_SSL_VERIFY=0`, `false`, or `no` to disable verification through the
environment. Prefer installing the institutional CA certificate instead. Credentials and
refresh tokens can traverse the configured proxy, so use only a proxy you trust.

## Full Constructor Reference

```python
engine = findpapers.Engine(
    ieee_api_key=None,  # str | None - IEEE Xplore API key
    scopus_api_key=None,  # str | None - Elsevier Scopus API key
    pubmed_api_key=None,  # str | None - NCBI PubMed API key
    ncbi_tool=None,  # str | None - NCBI E-utilities tool identity
    ncbi_email=None,  # str | None - NCBI contact email; falls back to email
    openalex_api_key=None,  # str | None - OpenAlex API key
    email=None,  # str | None - shared contact email
    semantic_scholar_api_key=None,  # str | None - Semantic Scholar API key
    wos_api_key=None,  # str | None - Clarivate Web of Science API key
    core_api_key=None,  # str | None - optional CORE API key
    openaire_access_token=None,  # str | None - current OpenAIRE bearer token
    openaire_refresh_token=None,  # str | None - OpenAIRE refresh token
    openaire_access_token_expires_in=None,  # float | None - remaining token lifetime
    zenodo_access_token=None,  # str | None - optional Zenodo bearer token
    chemrxiv_email=None,  # str | None - ChemRxiv/Crossref contact email
    ssrn_api_key=None,  # str | None - SSRN search credential
    openreview_username=None,  # str | None - OpenReview account username
    openreview_password=None,  # str | None - OpenReview account password
    openreview_venue_ids=None,  # list[str] | None - proceedings to search
    openreview_cache_dir=None,  # str | None - venue cache location
    openreview_cache_ttl_seconds=None,  # float | None - venue cache lifetime
    arxiv_snapshot_path=None,  # str | None - local arXiv metadata snapshot
    arxiv_categories=None,  # Collection[str] | None - snapshot category boundary
    zotero_base_url=None,  # str | None - Zotero local API base URL
    zotero_library=None,  # str | None - Zotero library identifier
    proxy=None,  # str | None - HTTP/HTTPS proxy URL
    ssl_verify=True,  # bool - verify SSL certificates
)
```

### Environment variables as fallback

Credentials and selected path settings fall back to environment variables when not provided
explicitly. This is useful for keeping credentials out of source code:

```bash
export IEEE_API_KEY="your-ieee-key"
export SCOPUS_API_KEY="your-scopus-key"
export NCBI_API_KEY="your-pubmed-key"
export LIT_SEARCH_NCBI_TOOL="my-literature-review"
export LIT_SEARCH_NCBI_EMAIL="researcher@university.edu"
export OPENALEX_API_KEY="your-openalex-key"
export SEMANTIC_SCHOLAR_API_KEY="your-s2-key"
export WOS_API_KEY="your-wos-key"
export CORE_API_KEY="your-core-key"
export OPENAIRE_ACCESS_TOKEN="current-access-token"
export OPENAIRE_REFRESH_TOKEN="one-month-refresh-token"
export ZENODO_ACCESS_TOKEN="your-zenodo-token"
export LIT_SEARCH_CHEMRXIV_EMAIL="chem-contact@example.edu"
export SSRN_API_KEY="your-ssrn-key"
export OPENREVIEW_USERNAME="user@example.edu"
export OPENREVIEW_PASSWORD="..."
export LIT_SEARCH_OPENREVIEW_CACHE_DIR="/path/to/openreview-cache"
export LIT_SEARCH_ARXIV_SNAPSHOT="/path/to/arxiv-metadata-oai-snapshot.json"
export LIT_SEARCH_ZOTERO_BASE_URL="http://127.0.0.1:23119/api"
export LIT_SEARCH_ZOTERO_LIBRARY="users/0"
export LIT_SEARCH_EMAIL="researcher@university.edu"
export LIT_SEARCH_PROXY="http://proxy.example.com:8080"
export LIT_SEARCH_SSL_VERIFY="true"
```

Values set in the environment win over the credentials file (`lit-search config path`). The old
`FINDPAPERS_*` names are no longer read; `lit-search config import` converts a file that uses
them, and `lit-search doctor` warns when they are still set. Keep secrets out of git.

## Parallel Execution

Several methods support parallel execution via the `num_workers` parameter:

```python
# Search databases in parallel
result = engine.search("[machine learning]", num_workers=4)

# Download PDFs in parallel
engine.download(result.papers, "./pdfs", num_workers=8)

# Snowball in parallel
graph = engine.snowball(result.papers[0], num_workers=4)
```

When `num_workers=1` (default), operations run sequentially. When greater than 1, a thread pool is used. The optimal number depends on your network and the API rate limits.

## Verbose Logging

Enable detailed logging for debugging:

```python
result = engine.search("[machine learning]", verbose=True)
```

When `verbose=True`, Findpapers logs HTTP requests, responses, rate limiting events, and processing details to the console.

## Progress Bars

Progress bars are shown by default using [tqdm](https://tqdm.github.io/). Disable them with:

```python
result = engine.search("[machine learning]", show_progress=False)
```

## Timeouts

Download, enrichment, and paper-lookup operations accept a `timeout` parameter (in seconds) for individual HTTP requests:

```python
engine.download(result.papers, "./pdfs", timeout=30.0)
paper = engine.get("10.1038/nature12373", timeout=15.0)
paper = engine.get("https://arxiv.org/abs/1706.03762", timeout=15.0)
```

Set to `None` to disable timeouts (not recommended).

## Bounded cloud runs

The checked-in [`.env.example`](../.env.example) is a secret-free template listing the
names. Besides the environment, the CLI reads the credentials file described in
[Setup](#setup); it reads no other dotenv or JSON file.

For a programmatic runner, inject credentials into the process environment through
its approved secret store, under these names:

| Secret-store name | Process environment variable | HTTPS credential destination |
|---|---|---|
| `SEMANTIC_SCHOLAR_API_KEY` | `SEMANTIC_SCHOLAR_API_KEY` | `api.semanticscholar.org` (`x-api-key` header) |
| `OPENALEX_API_KEY` | `OPENALEX_API_KEY` | `api.openalex.org` (`api_key` query parameter) |
| `OPENAIRE_ACCESS_TOKEN` | `OPENAIRE_ACCESS_TOKEN` | `api.openaire.eu` (Bearer header) |
| `OPENAIRE_REFRESH_TOKEN` | `OPENAIRE_REFRESH_TOKEN` | `services.openaire.eu` (`refreshToken` query parameter) |

The aliases in the first column are **not** read by lit-search itself. For a
shell-based runner that already receives those aliases, this mapping preserves
any explicitly supplied canonical variables:

```bash
# Keep shell tracing disabled wherever secrets are available.
set +x
export SEMANTIC_SCHOLAR_API_KEY="${SEMANTIC_SCHOLAR_API_KEY:-${SEMANTIC_SCHOLAR_API_KEY:-}}"
export OPENALEX_API_KEY="${OPENALEX_API_KEY:-${OPENALEX_API_KEY:-}}"
export OPENAIRE_ACCESS_TOKEN="${OPENAIRE_ACCESS_TOKEN:-${OPENAIRE_ACCESS_TOKEN:-}}"
export OPENAIRE_REFRESH_TOKEN="${OPENAIRE_REFRESH_TOKEN:-${OPENAIRE_REFRESH_TOKEN:-}}"
```

With the project installed, a deliberately small starting profile is two sources,
at most 25 papers per source, and no post-search enrichment:

```bash
python -m lit_search.cli search '[specification inference]' \
  -d openalex -d semantic_scholar \
  --max-papers 25 --no-enrich --handle --json \
  --store-root /tmp/lit-search-radar \
  -o /tmp/lit-search-radar-results.json
```

For a periodic radar, provide an explicit publication window with
`--since YYYY-MM-DD --until YYYY-MM-DD`; choose the dates for that run. Persist
the result JSON and handle store through the runner's approved artifact storage
if they must survive an ephemeral job. Inspect the coverage and scope fields,
including caps: exit 0 does not claim exhaustive literature coverage, while exit
3 reports an unplanned or unknown shortfall.

OpenAIRE is optional in this bounded profile. Add `-d openaire` only when wanted;
a refresh-token setup needs both OpenAIRE destinations reachable. The manual
[Bounded Literature Radar workflow](../.github/workflows/literature_radar.yml)
uses only the OpenAlex and Semantic Scholar secrets for its first-stage profile.
The OpenAIRE secrets can remain in the repository secret store without being
injected into that job.

### Run the manual GitHub Actions job

1. In repository Settings, add Actions secrets named `OPENALEX_API_KEY` and
   `SEMANTIC_SCHOLAR_API_KEY`. Do not put their values in workflow YAML or files.
2. In Actions, select **Bounded Literature Radar**, then **Run workflow** on the
   default branch. Enter a Boolean query such as `[specification inference]`.
3. Optionally enter `since` and `until` as `YYYY-MM-DD`. A blank `until` means
   today in UTC; a blank `since` starts a 14-day inclusive window. The job rejects
   windows longer than 31 days, reversed dates and future dates.
4. Check the acknowledgement for live API quota and run the job.
5. Download the `literature-radar-<run-id>-<attempt>` artifact from that run.
   Its `result.json` contains paper metadata and typed per-source coverage.

The workflow is manual only and serializes overlapping runs. It allows five
minutes for provider work and 15 minutes for the whole job, installs this checkout
without optional reranking models, and retains the artifact for seven days.
Credentials are injected only into the search step; checkout does not persist its
GitHub credential, and the job has read-only repository permissions.

The cloud export deliberately omits raw provider errors, request metadata and
handles. It redacts the active keys from allowed text fields and removes URL
query strings, fragments and user information. Raw stdout/stderr are discarded
because upstream HTTP exceptions can include authenticated URLs. This is a review
artifact rather than a lossless `SearchResult` file, so do not import it as a handle.
A failed/partial operation still uploads its safe diagnostic artifact when one
was written; a timeout may leave no artifact. Exit 2 means invalid inputs, exit 1
means missing credentials or a run failure, and exit 3 preserves incomplete or
unknown execution. Review the first live run before adding any recurring schedule.

For a personally managed local run, `.env` and `.env.*` are already ignored by
Git; `.env.example` is intentionally tracked. If you choose to use a populated
local `.env`, load it explicitly in your shell before invoking the CLI. Do not
commit it, print the environment, enable shell tracing, or include credentials in
commands, logs, result files, or workflow source.
