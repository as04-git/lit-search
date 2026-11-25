"""Command handlers for ``lit-search``.

Every handler returns an exit code and never raises past
:func:`lit_search.cli.main.main`.  Two rules hold across all of them:

* **A set is emitted only through** :func:`lit_search.core.rendering.render_result`
  **or** :func:`lit_search.cli.output.json_envelope`, both of which carry
  coverage.  There is no path here that prints papers on their own.
* **An error is never turned into an empty result.**  A lookup that returns
  nothing exits :data:`~lit_search.cli.output.EXIT_ERROR` with a message
  saying so, rather than printing a well-formed set of zero papers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from lit_search.cli.context import resolve_store
from lit_search.cli.emit import emit_set
from lit_search.cli.eval import cmd_eval
from lit_search.cli.explore import cmd_related, cmd_snippets
from lit_search.cli.find import cmd_find
from lit_search.cli.handles import cmd_handle, cmd_set
from lit_search.cli.hunt import cmd_hunt
from lit_search.cli.indexing import cmd_index, cmd_rerank
from lit_search.cli.output import (
    EXIT_ERROR,
    emit_error,
    emit_json,
    emit_text,
    exit_code_for,
    json_envelope,
)
from lit_search.core.coverage import CoverageLedger
from lit_search.core.identity import record_identity_keys, version_identity_key
from lit_search.core.lookup_result import LookupResult
from lit_search.core.paper import Paper
from lit_search.core.rendering import (
    CoverageView,
    coverage_view,
    render_coverage_report,
)
from lit_search.engine import Engine
from lit_search.exceptions import PersistenceError
from lit_search.store.operations import from_search_result, from_snowball_result
from lit_search.store.record import HandleRecord
from lit_search.utils.persistence import (
    Persistable,
    atomic_write_text,
    load_from_json,
    save_to_csv,
    save_to_json,
)

_BIBTEX_REFUSED = (
    "bibtex export is refused, and the refusal is the feature (D7).\n"
    "  Better BibTeX, in Zotero, is the single authority for citation keys. A "
    ".bib written here would carry keys this tool minted, which disagree with "
    "every key in the Zotero library the export is meant to feed -- and a "
    ".bib travels, so the disagreement is discovered by whoever merges it.\n"
    "  A warning was tried and is not enough: the file outlives the warning.\n"
    "  Supply --citation-keys PATH, a JSON object mapping an explicit DOI or "
    "provider identity to its Better BibTeX key. A lower-cased title remains "
    "accepted only for one identifier-free record; ambiguous title matches "
    "must use the distinct record selectors printed by the error. Every entry "
    "will use a Better BibTeX key from that map. A missing or duplicate key is "
    "an error, not a locally-minted fallback.\n"
    "  Until that file exists, export json or csv, import into Zotero, and let "
    "Better BibTeX assign."
)


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------


def _emit_with_optional_handle(
    command: str,
    result: Persistable,
    headline: str,
    args: argparse.Namespace,
    record: HandleRecord,
) -> int:
    """Emit a produced set, storing it as a handle when ``--handle`` was given.

    The handle is written **before** the rendering, so a run whose output is
    piped into a full disk still leaves the set recoverable.

    Parameters
    ----------
    command : str
        Subcommand name.
    result : object
        Result container.
    headline : str
        One-line description.
    args : argparse.Namespace
        Parsed arguments.
    record : HandleRecord
        Handle to store, built from *result*.

    Returns
    -------
    int
        Exit code implied by the set's coverage.
    """
    if not getattr(args, "handle", False):
        return emit_set(command, result, headline, args)
    handle_id = resolve_store(args).put(record)
    return emit_set(
        command, result, headline, args, extra_lines=[f"Handle: {handle_id}"], handle=handle_id
    )


def cmd_search(args: argparse.Namespace) -> int:
    """Run a Boolean search and render its coverage.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code.
    """
    engine = Engine()
    result = engine.search(
        args.query,
        databases=args.databases,
        max_papers_per_database=args.max_papers,
        since=args.since,
        until=args.until,
        num_workers=args.workers,
        verbose=args.verbose,
        show_progress=False,
        enrichment_databases=[] if args.no_enrich else None,
    )
    return _emit_with_optional_handle(
        "search", result, result.headline(), args, from_search_result(result, label=args.label)
    )


def cmd_get(args: argparse.Namespace) -> int:
    """Fetch one paper by DOI or URL and render what is known about coverage.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code.  A lookup that finds nothing is an error, not an empty
        set: the runner swallows per-source failures, so absence here is not
        evidence of absence upstream.
    """
    engine = Engine()
    raw_result = engine.get(args.identifier, databases=args.databases, verbose=args.verbose)
    if raw_result is None:  # compatibility for pre-contract Engine adapters
        emit_error(
            f"legacy lookup adapter returned no result for {args.identifier!r} without coverage; "
            "this is not evidence the paper does not exist"
        )
        return EXIT_ERROR
    result = (
        raw_result
        if isinstance(raw_result, LookupResult)
        else LookupResult(identifier=args.identifier, papers=[raw_result])
    )
    return emit_set("get", result, result.headline(), args)


def cmd_snowball(args: argparse.Namespace) -> int:
    """Expand a seed paper's citation neighbourhood.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code.
    """
    engine = Engine()
    seed_handle: str | None = None
    if args.seed.startswith("h_"):
        store = resolve_store(args)
        seed_handle = store.resolve(args.seed)
        seed_record = store.get(seed_handle)
        seeds: Paper | list[Paper] = list(seed_record.members)
        if not seeds:
            emit_error(f"seed handle {seed_handle} is empty; no traversal was attempted")
            return EXIT_ERROR
    else:
        raw_seed_lookup = engine.get(args.seed, verbose=args.verbose)
        if raw_seed_lookup is None:  # compatibility for pre-contract Engine adapters
            emit_error(
                f"legacy lookup adapter returned no seed for {args.seed!r} without coverage; "
                "this is a failed lookup, not an empty neighbourhood"
            )
            return EXIT_ERROR
        seed_lookup = (
            raw_seed_lookup
            if isinstance(raw_seed_lookup, LookupResult)
            else LookupResult(identifier=args.seed, papers=[raw_seed_lookup])
        )
        seed = seed_lookup.paper
        if seed is None:
            code = emit_set("snowball-seed", seed_lookup, seed_lookup.headline(), args)
            return EXIT_ERROR if code == 0 else code
        seeds = seed
    extra: dict[str, Any] = {}
    if args.databases is not None:
        extra["databases"] = args.databases
    if args.no_enrich:
        extra["enrichment_databases"] = []
    managed_run_id: str | None = None
    if args.managed_run:
        managed_run_id = f"r_{uuid.uuid4().hex}"
    elif args.resume_run:
        managed_run_id = args.resume_run
    if managed_run_id is not None:
        store_root = resolve_store(args).root
        extra.update(
            managed_store_root=store_root,
            managed_run_id=managed_run_id,
            resume_managed_run=bool(args.resume_run),
        )
        action = "resuming" if args.resume_run else "started"
        emit_text(
            f"lit-search: {action} managed run {managed_run_id}; "
            f"resume with --resume-run {managed_run_id} and identical options",
            stream=sys.stderr,
        )
    result = engine.snowball(
        seeds,
        max_depth=args.depth,
        direction=args.direction,
        max_papers_per_level=args.max_papers_per_level,
        verbose=args.verbose,
        show_progress=False,
        **extra,
    )
    if seed_handle is None:
        seed_entries = tuple(
            replace(entry, source=f"seed_lookup:{entry.source}")
            for entry in seed_lookup.coverage.entries
        )
        if seed_entries:
            traversal_entries = result.coverage.entries if result.coverage is not None else ()
            result.coverage = CoverageLedger.from_sources((*traversal_entries, *seed_entries))
    record = from_snowball_result(result, label=args.label, seed_handle=seed_handle)
    code = _emit_with_optional_handle(
        "snowball",
        result,
        result.headline(),
        args,
        record,
    )
    if result.run_id is not None and not args.quiet and not args.json:
        emit_text(f"Managed run: {result.run_id}")
    return code


def cmd_export(args: argparse.Namespace) -> int:
    """Write a stored result to a file and print its path, never its content.

    A coverage sidecar is written beside every export, in every format and
    whatever the verdict.  An export travels: a ``.bib`` handed to a
    co-author carries no ledger of its own, and a sidecar that appears only
    for incomplete sets would teach readers that its absence means "fine".

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code from the loaded set's coverage.
    """
    loaded = load_from_json(args.input)
    papers = loaded if isinstance(loaded, list) else list(loaded.papers)
    view = coverage_view(loaded)
    if args.format == "bibtex":
        keys = _load_citation_keys(args)
        if keys is None:
            emit_error(_BIBTEX_REFUSED)
            return EXIT_ERROR
        resolved_keys = _resolve_citation_keys(papers, keys, source=str(args.citation_keys))
        _write_bibtex(papers, resolved_keys, args.output)
    else:
        _write_export(loaded, papers, args)
    sidecar = _write_coverage_sidecar(args.output, view, source=args.input)
    if not args.quiet:
        if args.json:
            emit_json(
                json_envelope(
                    "export",
                    view,
                    paper_count=len(papers),
                    format=args.format,
                    output=str(args.output),
                    coverage_sidecar=sidecar,
                )
            )
        else:
            emit_text(f"Written: {args.output} ({len(papers)} papers, {args.format})")
            emit_text(f"Written: {sidecar}")
            emit_text(view.headline())
    return exit_code_for(view)


def _write_export(loaded: object, papers: list[Paper], args: argparse.Namespace) -> None:
    """Write the export itself in the requested format.

    Parameters
    ----------
    loaded : object
        The loaded result.
    papers : list[Paper]
        Papers extracted from it.
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    None
    """
    if args.format == "json":
        save_to_json(loaded, args.output)  # type: ignore[arg-type]
    else:
        save_to_csv(papers, args.output)


def _citation_key_for(paper: Paper) -> str:
    """Return the preferred explicit lookup key for a paper.

    Parameters
    ----------
    paper : Paper
        Record to look up.

    Returns
    -------
    str
        Bare lower-cased DOI for compatibility, otherwise a provider-qualified
        record identity.  A lower-cased title is returned only when the record
        states no explicit identity; callers must still check whether that
        fallback is ambiguous within the export.
    """
    if paper.doi:
        return paper.doi.strip().lower()
    explicit = _explicit_map_keys(paper)
    if explicit:
        return explicit[0]
    return paper.title.strip().lower()


def _explicit_map_keys(paper: Paper) -> tuple[str, ...]:
    """Return accepted explicit Better BibTeX map aliases in preference order.

    Bare DOIs remain accepted for compatibility with the original map format.
    Namespace-qualified publication-version and provider-record identities are
    the unambiguous form for new maps.  Titles are intentionally absent.
    """
    aliases: list[str] = []
    if paper.doi:
        aliases.append(paper.doi.strip().lower())
    version_key = version_identity_key(paper)
    if version_key is not None:
        aliases.append(version_key)
    aliases.extend(sorted(record_identity_keys(paper) - ({version_key} if version_key else set())))
    return tuple(dict.fromkeys(alias for alias in aliases if alias))


def _record_selectors(papers: list[Paper]) -> list[str]:
    """Build export-local selectors without asserting scholarly identity.

    The digest describes one persisted record snapshot.  An occurrence suffix
    keeps even byte-equivalent duplicate snapshots addressable as distinct
    export entries.  These selectors are only an escape hatch for supplying
    Better BibTeX keys when no external record identity exists.
    """
    bases: list[str] = []
    for paper in papers:
        payload = json.dumps(
            paper.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        bases.append(f"record:{hashlib.sha256(payload).hexdigest()[:16]}")

    totals = {base: bases.count(base) for base in set(bases)}
    seen: dict[str, int] = {}
    selectors: list[str] = []
    for base in bases:
        seen[base] = seen.get(base, 0) + 1
        selectors.append(f"{base}#{seen[base]}" if totals[base] > 1 else base)
    return selectors


def _load_citation_keys(args: argparse.Namespace) -> dict[str, str] | None:
    """Read the Better BibTeX key map, or return ``None`` when none was given.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    dict[str, str] | None
        Lower-cased identity to citation key, or ``None``.

    Raises
    ------
    PersistenceError
        If the file is not a JSON object of strings.
    """
    path = getattr(args, "citation_keys", None)
    if not path:
        return None
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in raw.items()
    ):
        raise PersistenceError(
            f"{path} must be a JSON object mapping identity to Better BibTeX key"
        )
    normalized = {key.strip().lower(): value.strip() for key, value in raw.items()}
    if any(not key or not value for key, value in normalized.items()):
        raise PersistenceError(
            f"{path} must not contain blank identities or blank Better BibTeX keys"
        )
    return normalized


def _resolve_citation_keys(papers: list[Paper], keys: dict[str, str], *, source: str) -> list[str]:
    """Resolve and validate exactly one Better BibTeX key per export record.

    Parameters
    ----------
    papers : list[Paper]
        Records being exported.
    keys : dict[str, str]
        Identity-to-key map supplied by Better BibTeX.
    source : str
        Map path, included in actionable failures.

    Returns
    -------
    list[str]
        Better BibTeX keys in paper order.

    Raises
    ------
    PersistenceError
        If a record is unmapped, an ambiguous title fallback is used, aliases
        disagree, or one citation key would be emitted for multiple records.
    """
    selectors = _record_selectors(papers)
    title_groups: dict[str, list[int]] = {}
    for index, paper in enumerate(papers):
        if not _explicit_map_keys(paper):
            title_groups.setdefault(paper.title.strip().lower(), []).append(index)

    ambiguous_titles = {
        title: indexes for title, indexes in title_groups.items() if len(indexes) > 1
    }
    for title, indexes in ambiguous_titles.items():
        if title in keys:
            required = ", ".join(selectors[index] for index in indexes)
            raise PersistenceError(
                f"ambiguous title fallback {title!r} in {source} matches "
                f"{len(indexes)} identifier-free records. Remove that title entry and provide "
                f"distinct Better BibTeX mappings for: {required}"
            )

    resolved: list[str] = []
    missing: list[str] = []
    for index, paper in enumerate(papers):
        explicit = _explicit_map_keys(paper)
        matched = {keys[alias] for alias in explicit if alias in keys}
        if len(matched) > 1:
            aliases = ", ".join(alias for alias in explicit if alias in keys)
            raise PersistenceError(
                f"conflicting Better BibTeX keys in {source} for aliases of one record: {aliases}"
            )
        if matched:
            resolved.append(next(iter(matched)))
            continue

        selector = selectors[index]
        if selector in keys:
            resolved.append(keys[selector])
            continue

        title = paper.title.strip().lower()
        if not explicit and title not in ambiguous_titles and title in keys:
            resolved.append(keys[title])
            continue

        required = explicit[0] if explicit else selector
        missing.append(required)

    if missing:
        shown = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise PersistenceError(
            f"{len(missing)} record(s) have no Better BibTeX key in {source}: "
            f"{shown}{suffix}. Add those explicit mappings, or export json and import into "
            "Zotero first; locally minted fallbacks are refused."
        )

    owners: dict[str, list[int]] = {}
    for index, key in enumerate(resolved):
        owners.setdefault(key, []).append(index)
    duplicate = next(((key, indexes) for key, indexes in owners.items() if len(indexes) > 1), None)
    if duplicate is not None:
        key, indexes = duplicate
        identities = ", ".join(selectors[index] for index in indexes)
        raise PersistenceError(
            f"Better BibTeX key {key!r} in {source} resolves to {len(indexes)} exported "
            f"records ({identities}); duplicate BibTeX keys are invalid. Deduplicate the "
            "records or provide distinct Better BibTeX keys."
        )
    return resolved


def _write_bibtex(papers: list[Paper], keys: list[str], output: str) -> None:
    """Write a BibTeX file using keys supplied by Better BibTeX.

    Deliberately does not call
    :func:`lit_search.utils.persistence.save_to_bibtex`, which mints its own
    keys.  Two key schemes in one workflow means silently divergent
    bibliographies, and the divergence is found by whoever merges the file.

    Parameters
    ----------
    papers : list[Paper]
        Records to write.
    keys : list[str]
        Validated Better BibTeX keys in paper order.
    output : str
        Destination path.

    Returns
    -------
    None
    """
    entries: list[str] = []
    for paper, citation_key in zip(papers, keys, strict=True):
        fields = [f"  title = {{{paper.title}}}"]
        if paper.authors:
            fields.append("  author = {" + " and ".join(a.name for a in paper.authors) + "}")
        if paper.publication_date is not None:
            fields.append(f"  year = {{{paper.publication_date.year}}}")
        if paper.doi:
            fields.append(f"  doi = {{{paper.doi}}}")
        if paper.preprint_doi:
            fields.append(f"  note = {{preprint: {paper.preprint_doi}}}")
        body = ",\n".join(fields)
        entries.append(f"@misc{{{citation_key},\n{body}\n}}")
    atomic_write_text(output, "\n\n".join(entries) + "\n")


def _write_coverage_sidecar(output: str, view: CoverageView, *, source: str) -> str:
    """Write the coverage report that accompanies an export.

    Parameters
    ----------
    output : str
        Path of the export the sidecar accompanies.
    view : CoverageView
        Coverage of the exported set.
    source : str
        Path the set was loaded from.

    Returns
    -------
    str
        Path of the sidecar written.
    """
    path = f"{output}.coverage.txt"
    body = "\n".join(
        [
            f"Coverage of {Path(output).name}, exported from {source}",
            "",
            render_coverage_report(view),
            "",
        ]
    )
    atomic_write_text(path, body)
    return path


HANDLERS = {
    "eval": cmd_eval,
    "find": cmd_find,
    "hunt": cmd_hunt,
    "related": cmd_related,
    "snippets": cmd_snippets,
    "search": cmd_search,
    "get": cmd_get,
    "snowball": cmd_snowball,
    "handle": cmd_handle,
    "set": cmd_set,
    "index": cmd_index,
    "rerank": cmd_rerank,
    "export": cmd_export,
}

__all__ = ["HANDLERS"]
