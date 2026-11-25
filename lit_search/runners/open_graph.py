"""Citation edges without Semantic Scholar.

* **Backward** (what a seed cites): the seed's arXiv e-print carries the authors'
  own bibliography; entries are resolved to OpenAlex works by arXiv id or DOI in
  batches, then by title.  One arXiv request per seed.  The ``.tex`` also gives
  the sentence each reference is cited in.
* **Forward** (who cites a seed): OpenAlex has indexed the full text of most
  recent papers even where it has not linked their references, so a phrase
  search for the seed's title (and title variants, and its arXiv id) in full
  text, from the seed's date on, finds its citers.  These are unverified: a
  paper that names the title without citing it would match.

Measured 2026-11-25 on 16 recent arXiv ML seeds against Semantic Scholar's own
lists: backward recall 95% of linked references; forward 77% of citers plus
about 25% more that Semantic Scholar lacks (55 of 56 checked cite the seed).
"""

from __future__ import annotations

import datetime
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import requests

from lit_search.connectors.arxiv import ArxivConnector
from lit_search.connectors.openalex import OpenAlexConnector
from lit_search.core.citation import IdentifierKind
from lit_search.core.coverage import SourceCoverage
from lit_search.core.paper import Paper
from lit_search.exceptions import ConnectorError, RateLimitBusyError
from lit_search.runners.explore_runner import ExploreOutcome
from lit_search.utils import latex_bib

_WORKS_URL = "https://api.openalex.org/works"
_EPRINT_URL = "https://export.arxiv.org/e-print/{arxiv_id}"
_SELECT = (
    "id,doi,title,display_name,publication_date,authorships,abstract_inverted_index,"
    "cited_by_count,open_access,locations,primary_location,keywords,type,biblio,"
    "primary_topic,language,is_retracted"
)
_DOI_BATCH = 50
_SAME_TITLE = 0.9
#: Title lookups per seed (one OpenAlex request each); entries with ids resolve in batches.
TITLE_LOOKUPS = 80
_ERRORS = (requests.RequestException, ConnectorError, RateLimitBusyError, ValueError)


@dataclass
class OpenSeed:
    """What the open routes need to know about a seed."""

    label: str
    title: str
    arxiv: str | None
    doi: str | None
    published: datetime.date | None


def seed_from_paper(label: str, paper: Paper) -> OpenSeed:
    """Describe a stored paper as a seed."""
    by_kind = {identifier.kind: identifier.value for identifier in paper.identifiers}
    arxiv = by_kind.get(IdentifierKind.ARXIV)
    doi = paper.doi
    if not arxiv and doi and doi.lower().startswith("10.48550/arxiv."):
        arxiv = doi.split(".", 2)[2]
    return OpenSeed(label, paper.title or label, arxiv, doi, paper.publication_date)


def _work_arxiv(work: dict[str, Any]) -> str | None:
    for location in work.get("locations") or []:
        for key in ("landing_page_url", "pdf_url"):
            url = location.get(key) or ""
            if "arxiv" in url and (found := latex_bib.arxiv_id_in(url)):
                return found
    doi = (work.get("doi") or "").lower()
    return doi.split("arxiv.")[-1] if "10.48550/arxiv." in doi else None


def _bare_doi(work: dict[str, Any]) -> str:
    return (work.get("doi") or "").lower().replace("https://doi.org/", "")


def _works(openalex: OpenAlexConnector, params: dict[str, Any]) -> list[dict[str, Any]]:
    data = openalex._get(_WORKS_URL, {"select": _SELECT, **params}).json()
    return list(data.get("results") or [])


def _date_ok(
    work: dict[str, Any], since: datetime.date | None, until: datetime.date | None
) -> bool:
    raw = work.get("publication_date")
    if not raw or not (since or until):
        return True
    try:
        date = datetime.date.fromisoformat(raw)
    except ValueError:
        return True
    return not (since and date < since) and not (until and date > until)


def resolve_seed(openalex: OpenAlexConnector, ref: str) -> OpenSeed | None:
    """Look up a raw arXiv id or DOI in OpenAlex for its title and date."""
    arxiv = latex_bib.arxiv_id_in(ref)
    doi = f"10.48550/arxiv.{arxiv}" if arxiv else ref.lower().removeprefix("doi:")
    results = _works(openalex, {"filter": f"doi:{doi}", "per-page": 1})
    if not results:
        return None
    work = results[0]
    date = work.get("publication_date")
    return OpenSeed(
        ref,
        work.get("title") or ref,
        arxiv or _work_arxiv(work),
        _bare_doi(work) or None,
        datetime.date.fromisoformat(date) if date else None,
    )


def lookup_papers(
    openalex: OpenAlexConnector,
    refs: Sequence[str],
    arxiv_connector: ArxivConnector | None = None,
) -> tuple[dict[str, Paper], list[SourceCoverage]]:
    """Find papers by arXiv id, DOI or exact title in OpenAlex, then arXiv.

    Ids go in one batched request; each title is one title search, accepted only
    for a near-identical title.  arXiv ids OpenAlex does not know as DOIs (it
    folds a preprint into its published version) are read from the arXiv API in
    one request.

    Returns
    -------
    tuple[dict[str, Paper], list[SourceCoverage]]
        Papers keyed by the ref that found them, and one step per request.
    """
    found: dict[str, Paper] = {}
    steps: list[SourceCoverage] = []
    by_doi: dict[str, str] = {}
    titles: list[str] = []
    for ref in refs:
        value = ref.strip()
        lowered = value.lower().removeprefix("doi:").removeprefix("arxiv:")
        arxiv_id = latex_bib.arxiv_id_in(lowered) if re.fullmatch(r"[\d.v]+", lowered) else None
        if arxiv_id:
            by_doi[f"10.48550/arxiv.{arxiv_id}"] = ref
        elif lowered.startswith("10."):
            by_doi[lowered] = ref
        else:
            titles.append(ref)
    if by_doi:
        try:
            works = _works(openalex, {"filter": "doi:" + "|".join(by_doi), "per-page": 100})
            for work in works:
                owner = by_doi.get(_bare_doi(work))
                paper = openalex._parse_paper(work) if owner else None
                if owner and paper is not None:
                    found[owner] = paper
            steps.append(
                SourceCoverage.complete("add:openalex:ids", len(found), reported_total=len(found))
                if len(found) == len(by_doi)
                else SourceCoverage.partial(
                    "add:openalex:ids",
                    len(found),
                    detail=f"not in OpenAlex: {', '.join(r for r in by_doi.values() if r not in found)}",
                )
            )
        except _ERRORS as exc:
            steps.append(
                SourceCoverage.failed(
                    "add:openalex:ids", error=str(exc), error_type=type(exc).__name__
                )
            )
    unresolved_arxiv = {
        doi.split("arxiv.")[-1]: ref
        for doi, ref in by_doi.items()
        if doi.startswith("10.48550/arxiv.") and ref not in found
    }
    if unresolved_arxiv:
        arxiv_connector = arxiv_connector or ArxivConnector()
        try:
            batch = arxiv_connector.fetch_papers_batch(list(unresolved_arxiv))
            for paper in batch.papers:
                ids = {i.value for i in paper.identifiers if i.kind is IdentifierKind.ARXIV}
                for arxiv_id, ref in unresolved_arxiv.items():
                    if arxiv_id in ids or any(i.startswith(arxiv_id) for i in ids):
                        found[ref] = paper
            got = sum(1 for r in unresolved_arxiv.values() if r in found)
            steps.append(
                SourceCoverage.complete("add:arxiv:ids", got, reported_total=got)
                if got == len(unresolved_arxiv)
                else SourceCoverage.partial("add:arxiv:ids", got, detail="some ids not found")
            )
        except _ERRORS as exc:
            steps.append(
                SourceCoverage.failed(
                    "add:arxiv:ids", error=str(exc), error_type=type(exc).__name__
                )
            )
    for title in titles:
        step = f"add:openalex:title:{title[:40]}"
        try:
            works = _works(
                openalex,
                {"filter": f"title.search:{latex_bib.normalise(title)[:200]}", "per-page": 5},
            )
        except _ERRORS as exc:
            steps.append(SourceCoverage.failed(step, error=str(exc), error_type=type(exc).__name__))
            continue
        best = max(
            works, key=lambda w: latex_bib.title_similarity(w.get("title"), title), default=None
        )
        paper = (
            openalex._parse_paper(best)
            if best and latex_bib.title_similarity(best.get("title"), title) >= _SAME_TITLE
            else None
        )
        if paper is None:
            steps.append(SourceCoverage.partial(step, 0, detail="no title match"))
        else:
            found[title] = paper
            steps.append(SourceCoverage.complete(step, 1, reported_total=1))
    return found, steps


def complete_seed(openalex: OpenAlexConnector, seed: OpenSeed) -> OpenSeed:
    """Fill in a seed's arXiv id and earliest date from OpenAlex records of the same title.

    A published paper's record often carries only the venue DOI, while its
    preprint record (a separate OpenAlex work) carries the arXiv id the open
    routes need.
    """
    if seed.arxiv or len(latex_bib.normalise(seed.title)) <= 8:
        return seed
    title = latex_bib.normalise(seed.title)
    works = _works(openalex, {"filter": f"title.search:{title[:200]}", "per-page": 10})
    same = [w for w in works if latex_bib.title_similarity(w.get("title"), seed.title) >= 0.95]
    arxiv = next((a for a in (_work_arxiv(w) for w in same) if a), None)
    dates = [
        datetime.date.fromisoformat(w["publication_date"])
        for w in same
        if w.get("publication_date")
    ]
    earliest = min([*dates, *([seed.published] if seed.published else [])], default=None)
    return OpenSeed(seed.label, seed.title, arxiv, seed.doi, earliest)


def fetch_eprint(arxiv: ArxivConnector, arxiv_id: str) -> bytes:
    """Download an e-print through the arXiv connector's shared rate limit."""
    return arxiv._get(_EPRINT_URL.format(arxiv_id=arxiv_id)).content


def _resolve_entries(
    openalex: OpenAlexConnector, entries: list[latex_bib.BibEntry], deadline: float
) -> tuple[dict[int, dict[str, Any]], int]:
    """Map entry index -> OpenAlex work; also return how many requests failed."""
    resolved: dict[int, dict[str, Any]] = {}
    failures = 0
    wanted: dict[str, list[int]] = {}
    for index, entry in enumerate(entries):
        doi = f"10.48550/arxiv.{entry.arxiv}" if entry.arxiv else entry.doi
        if doi:
            wanted.setdefault(doi.lower(), []).append(index)
    dois = list(wanted)
    for start in range(0, len(dois), _DOI_BATCH):
        if time.time() > deadline:
            break
        chunk = dois[start : start + _DOI_BATCH]
        try:
            works = _works(openalex, {"filter": "doi:" + "|".join(chunk), "per-page": 200})
        except _ERRORS:
            failures += 1
            continue
        for work in works:
            for index in wanted.get(_bare_doi(work), []):
                resolved[index] = work
    lookups = 0
    for index, entry in enumerate(entries):
        title = latex_bib.normalise(entry.title)
        if index in resolved or len(title) <= 8 or lookups >= TITLE_LOOKUPS:
            continue
        if time.time() > deadline:
            break
        lookups += 1
        try:
            works = _works(openalex, {"filter": f"title.search:{title[:200]}", "per-page": 5})
        except _ERRORS:
            failures += 1
            continue
        best = max(
            works, key=lambda w: latex_bib.title_similarity(w.get("title"), title), default=None
        )
        if best and latex_bib.title_similarity(best.get("title"), title) >= _SAME_TITLE:
            resolved[index] = best
    return resolved, failures


def backward_from_source(
    arxiv: ArxivConnector,
    openalex: OpenAlexConnector,
    seed: OpenSeed,
    *,
    since: datetime.date | None = None,
    until: datetime.date | None = None,
    deadline: float,
) -> tuple[list[tuple[Paper, list[str]]], SourceCoverage]:
    """References of *seed* read from its arXiv source, with the sentences citing them."""
    step = f"snowball:backward:arxiv-source:{seed.title[:40]}"
    if not seed.arxiv:
        return [], SourceCoverage.skipped_by_policy(step, detail="not an arXiv paper")
    try:
        kind, files = latex_bib.unpack(fetch_eprint(arxiv, seed.arxiv))
    except _ERRORS as exc:
        return [], SourceCoverage.failed(step, error=str(exc), error_type=type(exc).__name__)
    origin, entries = latex_bib.bibliography(files)
    if not entries:
        detail = "the e-print is a PDF only" if kind == "pdf" else f"no bibliography ({kind})"
        return [], SourceCoverage.partial(step, 0, detail=detail)
    resolved, failures = _resolve_entries(openalex, entries, deadline)
    found: list[tuple[Paper, list[str]]] = []
    seen: set[str] = set()
    for index, work in resolved.items():
        if work.get("id") in seen or not _date_ok(work, since, until):
            continue
        seen.add(work.get("id") or "")
        paper = openalex._parse_paper(work)
        if paper is None:
            continue
        sentences = latex_bib.citation_sentences(files, {entries[index].key}, limit=2)
        found.append((paper, sentences))
    detail = (
        f"{len(resolved)} of {len(entries)} {origin} entries resolved in OpenAlex"
        + (f"; {failures} lookups failed" if failures else "")
        + (", stopped at the deadline" if time.time() > deadline else "")
    )
    unresolved = len(entries) - len(resolved)
    if failures or unresolved:
        return found, SourceCoverage.partial(step, len(found), detail=detail)
    return found, SourceCoverage.complete(step, len(found), reported_total=len(found))


def _title_phrases(title: str) -> list[str]:
    phrases = [title]
    if ":" in title:
        head, tail = title.split(":", 1)
        if len(head.split()) >= 3:
            phrases.append(head)
        if len(tail.split()) >= 4:
            phrases.append(tail)
    stripped = re.sub("[\u2019']", "", title)
    if stripped != title:
        phrases.append(stripped)
    cleaned = []
    for phrase in phrases:
        text = re.sub(r"[^a-z0-9 ]+", " ", phrase.casefold().replace("\u2019", "").replace("'", ""))
        text = re.sub(r"\s+", " ", text).strip()
        if len(text.split()) >= 3 and text not in cleaned:
            cleaned.append(text)
    return cleaned


def forward_from_fulltext(
    openalex: OpenAlexConnector,
    seed: OpenSeed,
    *,
    since: datetime.date | None = None,
    until: datetime.date | None = None,
    deadline: float,
) -> tuple[list[Paper], SourceCoverage]:
    """Papers whose full text names *seed*'s title or arXiv id (likely citers)."""
    step = f"snowball:forward:openalex-fulltext:{seed.title[:40]}"
    phrases = _title_phrases(seed.title)
    if seed.arxiv:
        phrases.append(seed.arxiv)
    earliest = max(d for d in (seed.published, since) if d) if (seed.published or since) else None
    filters = [f"from_publication_date:{earliest.isoformat()}"] if earliest else []
    if until:
        filters.append(f"to_publication_date:{until.isoformat()}")
    works: dict[str, dict[str, Any]] = {}
    failed = 0
    for phrase in phrases:
        if time.time() > deadline:
            break
        try:
            results = _works(
                openalex,
                {"filter": ",".join([f'fulltext.search:"{phrase}"', *filters]), "per-page": 200},
            )
        except _ERRORS:
            failed += 1
            continue
        for work in results:
            if latex_bib.title_similarity(work.get("title"), seed.title) >= _SAME_TITLE:
                continue
            if seed.arxiv and _work_arxiv(work) == seed.arxiv:
                continue
            works.setdefault(work.get("id") or str(len(works)), work)
    papers = [p for p in (openalex._parse_paper(w) for w in works.values()) if p is not None]
    if failed == len(phrases):
        return [], SourceCoverage.failed(
            step, error="every full-text query failed", error_type="ConnectorError"
        )
    detail = (
        f"{len(papers)} papers name the title or arXiv id in their full text (citers, "
        f"unverified); {len(phrases) - failed} of {len(phrases)} phrasings searched"
    )
    return papers, SourceCoverage.partial(step, len(papers), detail=detail)


def run_open_snowball(
    seeds: Sequence[OpenSeed],
    *,
    forward: bool,
    backward: bool,
    since: datetime.date | None = None,
    until: datetime.date | None = None,
    seconds: float = 120.0,
    openalex: OpenAlexConnector | None = None,
    arxiv: ArxivConnector | None = None,
) -> ExploreOutcome:
    """Both open routes for every seed, as one exploration outcome.

    Routes read ``cites <label> [fulltext]`` and ``cited by <label> [source]``;
    evidence holds the seed's sentences citing a backward neighbour.
    """
    from lit_search import credentials

    started = time.time()
    deadline = started + seconds
    openalex = openalex or OpenAlexConnector(api_key=credentials.get("OPENALEX_API_KEY"))
    arxiv = arxiv or ArxivConnector()
    outcome = ExploreOutcome()
    by_key: dict[str, Paper] = {}

    def add(paper: Paper, route: str, sentences: list[str]) -> None:
        key = paper.doi or paper.title or str(id(paper))
        kept = by_key.setdefault(key.lower(), paper)
        if kept is paper:
            outcome.papers.append(paper)
        outcome.routes.setdefault(id(kept), []).append(route)
        evidence = outcome.evidence.setdefault(id(kept), [])
        evidence.extend(s for s in sentences if s not in evidence)

    for seed in seeds:
        if forward:
            citers, step = forward_from_fulltext(
                openalex, seed, since=since, until=until, deadline=deadline
            )
            outcome.steps.append(step)
            for paper in citers:
                add(paper, f"cites {seed.label} [fulltext]", [])
        if backward:
            refs, step = backward_from_source(
                arxiv, openalex, seed, since=since, until=until, deadline=deadline
            )
            outcome.steps.append(step)
            for paper, sentences in refs:
                add(paper, f"cited by {seed.label} [source]", sentences)
    outcome.seconds = time.time() - started
    return outcome


def _search_titles(
    openalex: OpenAlexConnector, title: str, since: datetime.date, count: int
) -> list[dict[str, Any]]:
    phrase = latex_bib.normalise(title)[:250]
    return _works(
        openalex,
        {
            "search": phrase,
            "filter": f"from_publication_date:{since.isoformat()}",
            "per-page": count,
        },
    )


def run_open_related(
    seeds: Sequence[OpenSeed],
    *,
    negative: Sequence[str] = (),
    count: int = 20,
    recent_days: int = 90,
    years: int = 2,
    seconds: float = 45.0,
    openalex: OpenAlexConnector | None = None,
    today: datetime.date | None = None,
) -> ExploreOutcome:
    """Papers like the seeds, from OpenAlex: its ``related_works`` and its search.

    Two searches per seed with the seed's title (OpenAlex search matches full
    text, so these pull in work that discusses the seed): one over the last
    *years*, one over the last *recent_days*, the slice Semantic Scholar's
    recommendations covered best.  ``related_works`` adds OpenAlex's own
    neighbours.  Papers whose titles match *negative* are left out.
    """
    from lit_search import credentials

    started = time.time()
    deadline = started + seconds
    openalex = openalex or OpenAlexConnector(api_key=credentials.get("OPENALEX_API_KEY"))
    today = today or datetime.date.today()
    outcome = ExploreOutcome()
    seen: dict[str, Paper] = {}
    avoid = [latex_bib.normalise(t) for t in negative if t]
    seed_titles = [s.title for s in seeds]

    def add(work: dict[str, Any], route: str) -> bool:
        title = work.get("title") or ""
        if any(latex_bib.title_similarity(title, t) >= _SAME_TITLE for t in seed_titles):
            return False
        if any(latex_bib.title_similarity(title, t) >= _SAME_TITLE for t in avoid):
            return False
        key = work.get("id") or title
        paper = seen.get(key)
        if paper is None:
            parsed = openalex._parse_paper(work)
            if parsed is None:
                return False
            paper = seen[key] = parsed
            outcome.papers.append(paper)
        outcome.routes.setdefault(id(paper), []).append(route)
        return True

    for seed in seeds:
        for label, since, size in (
            ("similar", today - datetime.timedelta(days=365 * years), count),
            ("recent", today - datetime.timedelta(days=recent_days), max(5, count // 2)),
        ):
            step = f"related:openalex-{label}:{seed.title[:40]}"
            if time.time() > deadline:
                outcome.steps.append(SourceCoverage.partial(step, 0, detail="move deadline spent"))
                continue
            try:
                works = _search_titles(openalex, seed.title, since, size)
            except _ERRORS as exc:
                outcome.steps.append(
                    SourceCoverage.failed(step, error=str(exc), error_type=type(exc).__name__)
                )
                continue
            added = 0
            for rank, work in enumerate(works, start=1):
                added += add(work, f"{label}#{rank} to {seed.label}")
            outcome.steps.append(
                SourceCoverage.partial(
                    step, added, detail=f"top {len(works)} by OpenAlex relevance since {since}"
                )
            )
        step = f"related:openalex-related-works:{seed.title[:40]}"
        try:
            found = _works(
                openalex,
                {"filter": f"title.search:{latex_bib.normalise(seed.title)[:200]}", "per-page": 3},
            )
            record = max(
                found,
                key=lambda w: latex_bib.title_similarity(w.get("title"), seed.title),
                default=None,
            )
            ids = []
            if (
                record
                and latex_bib.title_similarity(record.get("title"), seed.title) >= _SAME_TITLE
            ):
                data = openalex._get(
                    _WORKS_URL + "/" + record["id"].rsplit("/", 1)[-1], {"select": "related_works"}
                ).json()
                ids = [w.rsplit("/", 1)[-1] for w in data.get("related_works") or []][:count]
            works = (
                _works(openalex, {"filter": "openalex:" + "|".join(ids), "per-page": len(ids)})
                if ids
                else []
            )
        except _ERRORS as exc:
            outcome.steps.append(
                SourceCoverage.failed(step, error=str(exc), error_type=type(exc).__name__)
            )
            continue
        added = sum(add(work, f"related to {seed.label} [openalex]") for work in works)
        outcome.steps.append(
            SourceCoverage.complete(step, added, reported_total=added)
            if ids or record
            else SourceCoverage.partial(step, 0, detail="seed not found in OpenAlex")
        )
    outcome.seconds = time.time() - started
    return outcome


__all__ = [
    "OpenSeed",
    "backward_from_source",
    "complete_seed",
    "fetch_eprint",
    "forward_from_fulltext",
    "lookup_papers",
    "resolve_seed",
    "run_open_related",
    "run_open_snowball",
    "seed_from_paper",
]
