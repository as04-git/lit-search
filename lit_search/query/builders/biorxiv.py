"""Query builder for bioRxiv and medRxiv.

**The donor turned the query into a category name.**  Verbatim from
``paper_search_mcp/academic_platforms/biorxiv.py`` at the pinned commit, as
recorded in the specification: ``search(query, max_results, days=30)`` with
``category = query.lower().replace(' ', '_')``.  So ``machine learning AND
robustness`` became the category ``machine_learning_and_robustness``, which
matches nothing, and the result was an empty list -- reported as bioRxiv holding
no matching paper.

This builder does not do that.  bioRxiv's API cannot be sent a text query at
all (see :mod:`lit_search.connectors.biorxiv`), so the query is compiled into a
predicate and evaluated against records harvested by date window.  The full
boolean grammar therefore works exactly, including ``AND NOT``.

``[key]`` maps to the record's ``category`` -- bioRxiv's own subject
classification, and the only controlled vocabulary its records carry.  That
keeps the donor's one true insight (category is searchable) while dropping its
mistake (that the whole query is a category).

``[src]`` maps to the record ``type`` (for example ``new results``,
``confirmatory results``), which is the closest thing to a venue a preprint
server has.  Affiliation is not supported: the details endpoint carries only
``author_corresponding_institution``, which is one author's affiliation rather
than the paper's, and matching an ``[aff]`` term against it would answer a
different question from the one asked.
"""

from __future__ import annotations

from lit_search.core.query import FilterCode
from lit_search.query.local_match import LocalPredicateQueryBuilder

#: Filter code to the fields
#: :func:`~lit_search.connectors.biorxiv.match_record` supplies.  Every name is
#: a key observed on a live ``api.biorxiv.org/details`` response on 2026-08-31.
FIELD_MAP: dict[FilterCode, tuple[str, ...]] = {
    FilterCode.TITLE: ("title",),
    FilterCode.ABSTRACT: ("abstract",),
    FilterCode.AUTHOR: ("authors",),
    FilterCode.KEYWORDS: ("category",),
    FilterCode.SOURCE: ("type",),
    FilterCode.TITLE_ABSTRACT: ("title", "abstract"),
    FilterCode.TITLE_ABSTRACT_KEYWORDS: ("title", "abstract", "category"),
}


class BioRxivQueryBuilder(LocalPredicateQueryBuilder):
    """Compile queries into a predicate over harvested bioRxiv/medRxiv records."""

    FIELD_MAP = FIELD_MAP
    SOURCE_LABEL = "bioRxiv and medRxiv"
    UNSUPPORTED_FIELDS_NOTE = (
        "their details endpoint carries only the corresponding author's institution, "
        "which is one author's affiliation rather than the paper's, so an affiliation "
        "term would answer a different question from the one asked."
    )
