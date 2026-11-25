"""Zotero local-library query builder.

**This builder does not build a remote query, and the reason is the load-bearing
part of this module.**  Zotero's local API does expose a search parameter, and
using it would be the obvious implementation -- but it answers a different
question from the one lit-search's query grammar asks, and the difference is
invisible in the result.

Verified against the Zotero Web API v3 specification, which the local API
implements (``zotero.org/support/dev/web_api/v3/basics`` and
``.../local_api``, read 2026-08-31):

- ``q`` is documented as "Quick search. Searches titles and individual creator
  fields by default", with ``qmode`` switching between ``titleCreatorYear`` and
  ``everything``.  It takes a **single substring**, not an expression.
- There is no boolean operator, no field-scoped term, and above all no
  negation.  ``[transformer] AND NOT [vision]`` has no expression in ``q``.
- Sending only the first term of a conjunction, or joining terms with spaces and
  hoping quicksearch reads that as ``AND``, both return a set that is not the
  query's answer while the ledger reports the query the user wrote.

The library is therefore enumerated and filtered here.  A local library is the
one corpus in this project where that is cheap: a few thousand records on the
same machine over a loopback socket, with no network cost and no remote quota.

**Match semantics remain an open decision** (``O15`` and silent-failure
watchlist item 5 in the project specification: "Zotero connector semantics --
reopened; D19 settled less than it appeared to").  What is implemented here is
the one shape the tree already has a precedent for -- enumerate, then apply
lit-search's own grammar unchanged, exactly as the IACR connector does -- so
that a term means the same thing against a Zotero record as against a Europe
PMC one.  It is deliberately **not** a resolution of ``O15``, whose harder half
is identity and equivalence for set operations (preprint versus published DOI,
DOI present versus absent).  Nothing in this file settles that.
"""

from __future__ import annotations

from lit_search.core.query import FilterCode
from lit_search.query.local_match import LocalPredicateQueryBuilder, Predicate, Record

__all__ = ["FIELD_MAP", "Predicate", "Record", "ZoteroQueryBuilder"]

# Mapping from lit_search filter codes to the normalised record fields the local
# predicate reads.  The names are the connector's normalised ones rather than
# Zotero's raw field names, because Zotero spells the venue differently per item
# type -- ``publicationTitle``, ``proceedingsTitle``, ``bookTitle``,
# ``repository``, ``institution`` -- and a filter that read only one of them
# would silently miss every record of the other types.
FIELD_MAP: dict[FilterCode, tuple[str, ...]] = {
    FilterCode.TITLE: ("title",),
    FilterCode.ABSTRACT: ("abstract",),
    FilterCode.KEYWORDS: ("keywords",),
    FilterCode.AUTHOR: ("authors",),
    FilterCode.SOURCE: ("source",),
    FilterCode.TITLE_ABSTRACT: ("title", "abstract"),
    FilterCode.TITLE_ABSTRACT_KEYWORDS: ("title", "abstract", "keywords"),
}


class ZoteroQueryBuilder(LocalPredicateQueryBuilder):
    """Compile queries into a local predicate over Zotero library items.

    The matching machinery is shared with the other sources that cannot be
    asked a question -- see :mod:`lit_search.query.local_match`.  What is
    specific to Zotero is :data:`FIELD_MAP`, and the fact that ``src`` *is*
    answerable here: Zotero stores the venue, so a source term is a real filter
    rather than one that would silently match nothing.
    """

    FIELD_MAP = FIELD_MAP
    SOURCE_LABEL = "Zotero library"
    UNSUPPORTED_FIELDS_NOTE = (
        "Zotero stores no affiliation field for creators, so an affiliation term would "
        "match nothing and report that as a library holding no such paper."
    )
