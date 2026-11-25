"""Query builder for the local arXiv metadata snapshot.

The snapshot is a file, so the query is answered here rather than upstream.
:mod:`lit_search.query.local_match` carries the machinery and the coverage
consequence; this module supplies the field map.

Note what the map does **not** contain, and why the omissions are rejections
rather than silent no-ops.  The Kaggle snapshot's fields are ``id``,
``submitter``, ``authors``, ``title``, ``comments``, ``journal-ref``, ``doi``,
``abstract``, ``categories``, ``versions``, ``update_date`` and
``authors_parsed`` (read from the dataset's own description on 2026-08-31).
There is no affiliation anywhere in the dump, so an ``[aff]`` term cannot be
answered and is rejected.  Accepting it and matching it against the abstract
instead would widen the search without saying so, and a widened search that
still finds nothing is indistinguishable from a narrow one.

``[key]`` **is** supported, and it means the record's arXiv categories.  The
dump has no author-supplied keyword list, and arXiv's category labels are the
only controlled vocabulary the corpus carries; mapping ``[key]`` to them is
therefore the honest reading rather than a substitution.  It also matters for
the failure that motivated this project, which was terminological -- a search
thorough inside one community's words and silent about another's.  Writing
``[key]cs.LO`` beside a free-text term crosses that boundary on purpose.

This is a wider surface than the live arXiv API connector offers, which
supports only ``ti``, ``abs``, ``au`` and ``tiabs``.  Matching locally is what
buys the difference: there is no remote grammar to be constrained by.
"""

from __future__ import annotations

from lit_search.core.query import FilterCode
from lit_search.query.local_match import LocalPredicateQueryBuilder

#: Filter code to the fields
#: :meth:`~lit_search.loaders.arxiv_snapshot.ArxivRecord.as_match_record`
#: supplies.
FIELD_MAP: dict[FilterCode, tuple[str, ...]] = {
    FilterCode.TITLE: ("title",),
    FilterCode.ABSTRACT: ("abstract",),
    FilterCode.AUTHOR: ("authors",),
    FilterCode.SOURCE: ("journal",),
    FilterCode.KEYWORDS: ("categories",),
    FilterCode.TITLE_ABSTRACT: ("title", "abstract"),
    FilterCode.TITLE_ABSTRACT_KEYWORDS: ("title", "abstract", "categories"),
}


class ArxivBulkQueryBuilder(LocalPredicateQueryBuilder):
    """Compile queries into a predicate over snapshot records."""

    FIELD_MAP = FIELD_MAP
    SOURCE_LABEL = "the local arXiv snapshot"
    UNSUPPORTED_FIELDS_NOTE = (
        "the Kaggle arXiv metadata dump carries no author keywords and no affiliations, "
        "so nothing in a record could answer it. Matching it against the abstract instead "
        "would widen the search silently."
    )
