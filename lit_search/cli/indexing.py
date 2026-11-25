"""Index maintenance and re-ranking at the command line.

Re-ranking is the operation this project most needs to get right and most
easily gets wrong invisibly, so three things are structural here rather than
advisory.

**Two operations, not one** (``D14``).  ``--query`` ranks against a question
and embeds it with the model's *query* convention.  ``--seed`` ranks against a
paper and embeds that paper with the *document* convention, because ranking
against a seed is document-to-document similarity.  Applying a query prefix to
a document, or none to a query, degrades the ordering with no symptom, so the
mode is recorded in the handle's provenance rather than inferred later.

**No member can be dropped by the retrieval step.**  The lanes are restricted
to the set's own members with a pushed-down key filter, and the vector lane's
limit is the member count, so it returns every member that is in the index
rather than an approximate top-k of them.  Any member the lanes did not score
-- because it is not in the index at all -- is placed last with a score of
zero **and counted in the report**, never silently omitted.  That matters
because :func:`lit_search.store.operations.reranked` refuses a result that is
not a permutation of its base, precisely so that a retrieval step cannot drop
members and call the remainder an ordering.  A zero here means "no lane scored
this", which is an absence of evidence, not a low ranking.

**Diversification never changes the set.**  MMR reorders what is *displayed*.
The handle is written from the relevance order, and the display order is
reported beside it.  ``DiversifiedView.for_seeding`` exists so that anything
downstream asking for papers to expand gets the relevance order.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace

from lit_search.cli.context import open_index, resolve_embedder, resolve_store
from lit_search.cli.emit import emit_set
from lit_search.cli.output import EXIT_ERROR, EXIT_OK, emit_error, emit_json, emit_text
from lit_search.core.identity import record_identity_keys, version_identity_key
from lit_search.core.paper import Paper, identity_key
from lit_search.exceptions import InvalidParameterError
from lit_search.index.embedding import EmbeddingModel
from lit_search.index.hybrid import Candidate, HybridResult, Lane, hybrid_search
from lit_search.index.mmr import DEFAULT_LAMBDA, DiversifiedView, diversify
from lit_search.index.paper_index import PaperIndex
from lit_search.index.schema import paper_key
from lit_search.store import operations
from lit_search.store.order import RankingProvenance, RankMode
from lit_search.store.view import handle_view

#: Scoring modes offered by ``rerank``.
#:
#: ``two-stage`` is the ordering ``R6`` settled on: a bi-encoder recalls and a
#: cross-encoder rescores the head.  ``hybrid`` orders by the index's three
#: retrieval lanes fused, which is a different claim -- useful when a literal
#: or keyword match should influence the order, and **not** a semantic re-rank.
SCORING_MODES = ("two-stage", "hybrid")

#: Bi-encoder used when the caller names none.  Owned by
#: :mod:`lit_search.rerank`, imported lazily so that the CLI's other
#: subcommands do not pay for torch.
DEFAULT_BI_ENCODER = "Qwen/Qwen3-Embedding-0.6B"


def _index_lines(index: PaperIndex) -> list[str]:
    """Return the standard two lines describing an index's state.

    Parameters
    ----------
    index : PaperIndex
        Index to describe.

    Returns
    -------
    list[str]
        Identity line and health line.
    """
    return [f"Index: {index.root} [{index.identity.describe()}]", index.health().describe()]


# ----------------------------------------------------------------------
# index
# ----------------------------------------------------------------------


def cmd_index_add(args: argparse.Namespace) -> int:
    """Embed a handle's members into the index.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        :data:`~lit_search.cli.output.EXIT_OK`.
    """
    store = resolve_store(args)
    handle_id = store.resolve(args.handle)
    record = store.get(handle_id)
    index = open_index(args, resolve_embedder(args))
    report = index.add(list(record.members), source_handle=handle_id)
    if args.json:
        emit_json({"command": "index add", "handle": handle_id, **report.to_dict()})
    elif not args.quiet:
        emit_text(report.describe())
        for line in _index_lines(index):
            emit_text(line)
        for entry in report.degraded_text:
            emit_text(f"  embedded on incomplete text: {entry}")
    return EXIT_OK


def cmd_index_status(args: argparse.Namespace) -> int:
    """Report the index's identity, size and unindexed tail.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        :data:`~lit_search.cli.output.EXIT_OK`, or
        :data:`~lit_search.cli.output.EXIT_ERROR` when maintenance is overdue
        or the full-text index is missing -- both mean retrieval is not what
        the user thinks it is.
    """
    index = open_index(args, resolve_embedder(args))
    health = index.health()
    if args.json:
        emit_json(
            {
                "command": "index status",
                "root": str(index.root),
                "identity": index.identity.to_dict(),
                "text_policy": index.text_policy.describe(),
                **health.to_dict(),
            }
        )
    elif not args.quiet:
        for line in _index_lines(index):
            emit_text(line)
        emit_text(f"Text policy: {index.text_policy.describe()}")
    return EXIT_ERROR if (health.needs_optimize or not health.has_text_index) else EXIT_OK


def cmd_index_optimize(args: argparse.Namespace) -> int:
    """Build the full-text index if absent and compact the unindexed tail.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        :data:`~lit_search.cli.output.EXIT_OK`.
    """
    index = open_index(args, resolve_embedder(args))
    before = index.health()
    created = index.ensure_text_index()
    after = index.optimize()
    if args.json:
        emit_json(
            {
                "command": "index optimize",
                "created_text_index": created,
                "before": before.to_dict(),
                "after": after.to_dict(),
            }
        )
    elif not args.quiet:
        if created:
            emit_text("Created the full-text index; keyword retrieval is now available.")
        emit_text(f"Before: {before.describe()}")
        emit_text(f"After:  {after.describe()}")
    return EXIT_OK


def cmd_index_search(args: argparse.Namespace) -> int:
    """Query the whole index, reporting which lane surfaced each record.

    A diagnostic rather than a set-producing command: it searches the index,
    which is not the same object as a search handle, so it deliberately does
    not write one.  Use ``rerank`` to order a stored set.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        :data:`~lit_search.cli.output.EXIT_OK`, or
        :data:`~lit_search.cli.output.EXIT_ERROR` when a lane could not run:
        a retrieval missing one of its three mechanisms is not the retrieval
        that was asked for.
    """
    embedder = resolve_embedder(args)
    index = open_index(args, embedder)
    result = hybrid_search(
        index, embedder, args.query, limit=args.limit or 20, exact_terms=args.exact
    )
    _emit_hybrid("index search", result, args)
    return EXIT_ERROR if result.unavailable_lanes else EXIT_OK


def _emit_hybrid(command: str, result: HybridResult, args: argparse.Namespace) -> None:
    """Print a hybrid retrieval, lanes first.

    Parameters
    ----------
    command : str
        Subcommand name.
    result : HybridResult
        Retrieval to print.
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    None
    """
    if args.json:
        emit_json({"command": command, **result.to_dict()})
        return
    if args.quiet:
        return
    emit_text(result.headline())
    for outcome in result.lanes:
        emit_text(f"  lane {outcome.describe()}")
    for position, candidate in enumerate(result.candidates, start=1):
        emit_text(f"  {position:>3}. {candidate.describe()}")


# ----------------------------------------------------------------------
# rerank
# ----------------------------------------------------------------------


def _seed_paper(members: Sequence[Paper], seed: str) -> Paper | None:
    """Find the seed paper among the set's members.

    Parameters
    ----------
    members : Sequence[Paper]
        Members of the set being ranked.
    seed : str
        DOI or title of the seed.

    Returns
    -------
    Paper | None
        The matching member, or ``None``.
    """
    wanted = seed.strip().lower()
    explicit_matches = [paper for paper in members if wanted in _index_keys(paper)]
    if explicit_matches:
        return explicit_matches[0]
    title_matches = [paper for paper in members if paper.title.strip().lower() == wanted]
    # A title can select a seed only when it is unambiguous.  Choosing the
    # first of two same-title records would silently make the seed depend on
    # input order.
    return title_matches[0] if len(title_matches) == 1 else None


def _index_keys(paper: Paper) -> frozenset[str]:
    """Return safe index and selector aliases for *paper*.

    The legacy index stores a bare DOI while strict identity qualifies it as
    ``doi:...`` (and canonicalises arXiv aliases), so both forms are retained
    during the migration. Identifier-free records use the same stable
    ``snapshot:`` key written by :class:`PaperIndex`; this is local row
    addressing, not title identity or scholarly equivalence.

    Parameters
    ----------
    paper : Paper
        Member whose safe aliases are needed.

    Returns
    -------
    frozenset[str]
        Index row key plus explicit version/provider selector aliases.
    """
    version_key = version_identity_key(paper)
    if version_key is not None:
        # Once a publication version is explicit, a provider alias is not
        # allowed to bridge to a record carrying a conflicting DOI.
        keys = {version_key, identity_key(paper), paper_key(paper)}
    else:
        keys = {*record_identity_keys(paper), paper_key(paper)}
    return frozenset(key.lower() for key in keys)


def _build_provenance(
    args: argparse.Namespace,
    index: PaperIndex,
    seed_identity: str | None,
    degraded: Sequence[str],
) -> RankingProvenance:
    """Build the provenance recorded on the ranked handle.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    index : PaperIndex
        Index the vectors came from.
    seed_identity : str | None
        Seed identity in seed mode.
    degraded : Sequence[str]
        Keys whose indexed text was incomplete.

    Returns
    -------
    RankingProvenance
        Provenance whose ``pool_independent`` flag is honest: fused
        reciprocal-rank scores depend on which other members were in the pool,
        plain cosine does not.
    """
    identity = index.identity
    hybrid = args.scoring == "hybrid"
    return RankingProvenance(
        mode=RankMode.SEED if seed_identity else RankMode.QUERY,
        query_text=None if seed_identity else args.query,
        seed_identity=seed_identity,
        model_id=identity.model_id,
        model_revision=identity.model_revision,
        prompt_convention=identity.prompt_convention,
        scoring_convention="rrf(vector+bm25+exact)" if hybrid else "cosine",
        normalization=identity.normalization,
        text_policy=index.text_policy.describe(),
        degraded_text=tuple(degraded),
        pool_independent=not hybrid,
    )


def _order_members(
    members: Sequence[Paper], candidates: Sequence[Candidate]
) -> tuple[list[Paper], list[float]]:
    """Put the set's members into the ranked order, keeping every one.

    Parameters
    ----------
    members : Sequence[Paper]
        The base handle's members.
    candidates : Sequence[Candidate]
        Ranked candidates, best first.

    Returns
    -------
    tuple[list[Paper], list[float]]
        Members in ranked order and their scores.  Any member the retrieval
        did not score is appended at the end with a score of zero rather than
        dropped, and the caller reports the count.
    """
    ordered: list[Paper] = []
    scores: list[float] = []
    consumed: set[int] = set()
    for candidate in candidates:
        candidate_key = candidate.key.lower()
        for index, paper in enumerate(members):
            if index in consumed or candidate_key not in _index_keys(paper):
                continue
            consumed.add(index)
            ordered.append(paper)
            scores.append(candidate.fused_score)
    for index, paper in enumerate(members):
        if index not in consumed:
            ordered.append(paper)
            scores.append(0.0)
    return ordered, scores


def _rank(
    index: PaperIndex,
    embedder: EmbeddingModel,
    args: argparse.Namespace,
    members: Sequence[Paper],
) -> tuple[HybridResult, list[Candidate]]:
    """Score every member, hybrid or plain cosine as requested.

    Parameters
    ----------
    index : PaperIndex
        Index to score against.
    embedder : EmbeddingModel
        Model, used for the query or seed embedding.
    args : argparse.Namespace
        Parsed arguments.
    members : Sequence[Paper]
        Members to score.

    Returns
    -------
    tuple[HybridResult, list[Candidate]]
        The retrieval report and the ranked candidates.
    """
    keys = sorted({paper_key(paper) for paper in members})
    if args.seed:
        seed = _seed_paper(members, args.seed)
        text = f"{seed.title}. {seed.abstract or ''}".strip() if seed else args.seed
        query_vector: Sequence[float] | None = embedder.encode_documents([text])[0]
        query_text = seed.title if seed else args.seed
    else:
        query_vector = None
        query_text = args.query

    result = hybrid_search(
        index,
        embedder,
        query_text,
        limit=len(members),
        keys=keys,
        query_vector=query_vector,
        weights=None if args.scoring == "hybrid" else {Lane.VECTOR: 1.0},
    )
    return result, list(result.candidates)


def _two_stage(args: argparse.Namespace, base_id: str, members: Sequence[Paper]) -> tuple:
    """Run the two-stage bi-encoder then cross-encoder ranking.

    Delegates to :mod:`lit_search.rerank`, which owns the models.  The handle
    is written from :meth:`~lit_search.rerank.pipeline.TwoStageRanking.reading_order`,
    which is the full membership in the order a person should read -- the
    rescored head first, then the remainder.

    The score vector that accompanies it deliberately **mixes two scales**: a
    cross-encoder logit for the head and a cosine for the tail.  That is
    recorded in the scoring convention rather than smoothed over, because the
    two halves genuinely were measured differently and presenting one number
    for both would be an ordering claim nothing supports.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    base_id : str
        Handle being ranked, used only in messages.
    members : Sequence[Paper]
        Members to rank.

    Returns
    -------
    tuple
        Ordered papers, their scores, the provenance, and the report lines.
    """
    from lit_search.rerank import Reranker

    reranker = Reranker.load(
        bi_encoder_id=args.model or DEFAULT_BI_ENCODER,
        with_cross_encoder=not args.no_cross_encoder,
    )
    if args.seed:
        seed = _seed_paper(members, args.seed)
        if seed is None:
            raise InvalidParameterError(
                f"seed {args.seed!r} is not a member of {base_id}. Seed mode ranks a set "
                "against one of its own members; ranking against an outside paper would "
                "make the seed's own position meaningless."
            )
        ranking = (
            reranker.rank_against_seed_paper(list(members), seed, top_k=args.top_k)
            if args.top_k
            else reranker.rank_against_seed_paper(list(members), seed)
        )
    else:
        ranking = (
            reranker.rank_against_question(list(members), args.query, top_k=args.top_k)
            if args.top_k
            else reranker.rank_against_question(list(members), args.query)
        )

    ordered = list(ranking.reading_order())
    ordered_identities = ranking.reading_order_identities()
    final = ranking.rescored if ranking.rescored is not None else ranking.recall
    by_identity = dict(zip(final.identities, final.scores, strict=True))
    fallback = dict(zip(ranking.recall.identities, ranking.recall.scores, strict=True))
    scores = [
        by_identity.get(identity, fallback.get(identity, 0.0)) for identity in ordered_identities
    ]
    provenance = replace(
        final.provenance,
        scoring_convention=(
            f"{final.provenance.scoring_convention}; reading order mixes the rescored head "
            "with the recall tail, and the two score scales are not comparable"
        )
        if ranking.rescored is not None
        else final.provenance.scoring_convention,
        pool_independent=False
        if ranking.rescored is not None
        else final.provenance.pool_independent,
    )
    return ordered, scores, provenance, ranking.describe().splitlines()


def cmd_rerank(args: argparse.Namespace) -> int:
    """Order a stored set against a question or a seed paper.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code implied by the ranked set's coverage.
    """
    store = resolve_store(args)
    base_id = store.resolve(args.handle)
    base = store.get(base_id)
    if not base.members:
        emit_error(
            f"{base_id} holds no members, so there is nothing to order. This is an empty "
            "input, not an empty ranking."
        )
        return EXIT_ERROR
    if args.scoring == "two-stage":
        return _rerank_two_stage(args, store, base_id, base)

    embedder = resolve_embedder(args)
    index = open_index(args, embedder)
    ingest = index.add(list(base.members), source_handle=base_id)
    result, candidates = _rank(index, embedder, args, list(base.members))
    ordered, scores = _order_members(list(base.members), candidates)
    unscored = sum(1 for score in scores if score == 0.0)

    seed_identity = args.seed if args.seed else None
    provenance = _build_provenance(
        args,
        index,
        seed_identity,
        [candidate.key for candidate in candidates if candidate.degraded_text],
    )
    record = operations.reranked(
        store, base_id, members=ordered, scores=scores, ranking=provenance, label=args.label
    )
    handle_id = store.put(record)

    view = handle_view(handle_id, record)
    lines = _rerank_lines(args, index, result, candidates, handle_id, ingest.written, unscored)
    return emit_set("rerank", view, view.headline(), args, extra_lines=lines, handle=handle_id)


def _rerank_lines(
    args: argparse.Namespace,
    index: PaperIndex,
    result: HybridResult,
    candidates: Sequence[Candidate],
    handle_id: str,
    newly_indexed: int,
    unscored: int,
) -> list[str]:
    """Build the extra report lines a ranked handle needs beside its members.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    index : PaperIndex
        Index used.
    result : HybridResult
        Retrieval report.
    candidates : Sequence[Candidate]
        Ranked candidates.
    handle_id : str
        Identifier of the new handle.
    newly_indexed : int
        Members embedded during this run.
    unscored : int
        Members no lane scored.

    Returns
    -------
    list[str]
        Lines to print after the rendering.
    """
    lines = [f"Handle: {handle_id}", *_index_lines(index)]
    if newly_indexed:
        lines.append(f"Embedded {newly_indexed} member(s) not previously in the index.")
    lines.extend(f"  lane {outcome.describe()}" for outcome in result.lanes)
    if unscored:
        lines.append(
            f"  {unscored} member(s) were scored by no lane and are placed last with a "
            "score of zero; they were kept, not dropped, and their position is an "
            "absence of evidence rather than a low ranking"
        )
    if args.no_diversify:
        return lines

    view = _diversified(index, candidates, args)
    lines.append(view.headline())
    lines.extend(
        f"  display {position:>3}. {candidate.describe()}"
        for position, candidate in enumerate(view.for_display(), start=1)
    )
    lines.extend(f"  suppressed by diversity: {item.describe()}" for item in view.suppressed)
    lines.append(
        "  the stored handle keeps the RELEVANCE order; the display order above is a "
        "presentation choice and must not be used as a source of snowball seeds"
    )
    return lines


def _diversified(
    index: PaperIndex, candidates: Sequence[Candidate], args: argparse.Namespace
) -> DiversifiedView:
    """Build the diversified display order for the report.

    Parameters
    ----------
    index : PaperIndex
        Index holding the vectors.
    candidates : Sequence[Candidate]
        Ranked candidates.
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    DiversifiedView
        Both orders and the suppressions.
    """
    window = args.limit if args.limit else len(candidates)
    vectors = index.vectors_for([candidate.key for candidate in candidates])
    return diversify(candidates, vectors, limit=window, lambda_value=args.mmr_lambda)


def _rerank_two_stage(args: argparse.Namespace, store: object, base_id: str, base: object) -> int:
    """Commit a two-stage ranking as a handle and print what qualifies it.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    store : object
        Handle store.
    base_id : str
        Handle being ranked.
    base : object
        The loaded handle.

    Returns
    -------
    int
        Exit code implied by the ranked set's coverage.
    """
    if args.deterministic_embedder:
        emit_error(
            "--scoring two-stage needs a real bi-encoder and cross-encoder, and "
            "--deterministic-embedder supplies neither. Refusing to substitute the "
            "non-semantic stand-in, which would produce a confidently ordered list with "
            "nothing behind it. Use --scoring hybrid to exercise the plumbing, or drop "
            "--deterministic-embedder to load the real models."
        )
        return EXIT_ERROR
    members = list(base.members)  # type: ignore[attr-defined]
    ordered, scores, provenance, report = _two_stage(args, base_id, members)
    record = operations.reranked(
        store,  # type: ignore[arg-type]
        base_id,
        members=ordered,
        scores=scores,
        ranking=provenance,
        label=args.label,
    )
    handle_id = store.put(record)  # type: ignore[attr-defined]
    view = handle_view(handle_id, record)
    lines = [f"Handle: {handle_id}", *report]
    return emit_set("rerank", view, view.headline(), args, extra_lines=lines, handle=handle_id)


INDEX_ACTIONS = {
    "add": cmd_index_add,
    "status": cmd_index_status,
    "optimize": cmd_index_optimize,
    "search": cmd_index_search,
}


def cmd_index(args: argparse.Namespace) -> int:
    """Dispatch an ``index`` subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code from the action.
    """
    return INDEX_ACTIONS[args.action](args)


__all__ = ["DEFAULT_LAMBDA", "INDEX_ACTIONS", "SCORING_MODES", "cmd_index", "cmd_rerank"]
