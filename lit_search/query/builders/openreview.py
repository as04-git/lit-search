"""OpenReview query builder.

**This builder does not build a remote query, because OpenReview has no boolean
search over a venue's accepted papers.**  It compiles the query into a
predicate evaluated locally against notes retrieved for a venue, and its job is
to be honest about which filters that predicate can and cannot answer.

Why there is no remote query, verified live 2026-08-31 against
``https://api2.openreview.net`` and against ``openreview-py`` 2.5.1's own
source:

- The notes endpoint filters by *equality* on note fields.
  ``OpenReviewClient.get_notes`` turns ``content={'venueid': ...}`` into the
  query parameter ``content.venueid=...`` (``api/client.py:1785-1787``); the
  remaining parameters are ``id``, ``forum``, ``invitation``, ``signature``,
  ``number``, ``mintcdate`` and pagination.  There is no operator for "title
  contains X and abstract does not contain Y".
- The client also exposes ``search_notes``, which is a relevance-ranked
  full-text search over the whole site rather than a scoped boolean filter, and
  which cannot express ``AND NOT``.  Using it would answer a *different* query
  from the one the user wrote while the ledger reported the one they wrote.

So a venue's accepted papers are enumerated and filtered here.  Every field the
predicate reads comes from an API v2 note's ``content`` object, whose values are
wrapped as ``{"value": ...}``; the connector unwraps them before the predicate
sees them.

The coverage consequence is stated once, in
:mod:`lit_search.query.local_match`, and enforced by the connector: a result
assembled this way is never ``COMPLETE``.
"""

from __future__ import annotations

from lit_search.core.query import FilterCode
from lit_search.query.local_match import LocalPredicateQueryBuilder, Predicate, Record

__all__ = ["FIELD_MAP", "OpenReviewQueryBuilder", "Predicate", "Record"]

# Mapping from lit_search filter codes to the note ``content`` fields the local
# predicate reads for them.  These are the standard OpenReview submission
# fields; the connector also reads ``venue`` for provenance, but it is
# deliberately absent from every filter's field list -- a venue restriction is
# expressed by choosing which venues to enumerate, not by matching text that
# happens to name one, and offering both would give the same query two answers.
FIELD_MAP: dict[FilterCode, tuple[str, ...]] = {
    FilterCode.TITLE: ("title",),
    FilterCode.ABSTRACT: ("abstract",),
    FilterCode.KEYWORDS: ("keywords",),
    FilterCode.AUTHOR: ("authors",),
    FilterCode.TITLE_ABSTRACT: ("title", "abstract"),
    FilterCode.TITLE_ABSTRACT_KEYWORDS: ("title", "abstract", "keywords"),
}


class OpenReviewQueryBuilder(LocalPredicateQueryBuilder):
    """Compile queries into a local predicate over OpenReview notes.

    The matching machinery is shared with the other sources that cannot be
    asked a question -- see :mod:`lit_search.query.local_match`, which also
    states the coverage consequence.  What is specific to OpenReview is
    :data:`FIELD_MAP`: the ``content`` fields a submission note carries.
    """

    FIELD_MAP = FIELD_MAP
    SOURCE_LABEL = "OpenReview"
    UNSUPPORTED_FIELDS_NOTE = (
        "a submission note carries title, abstract, keywords and authors and no "
        "publisher, ISSN or affiliation metadata. Restrict the venue list rather than "
        "filtering on venue text."
    )
