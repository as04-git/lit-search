"""IACR Cryptology ePrint Archive query builder.

**This builder does not build a remote query, because the archive has no
searchable endpoint this project may use.**  It compiles the query into a
predicate evaluated locally against harvested OAI-PMH records, and its job is
to be honest about which filters that predicate can and cannot answer.

Why there is no remote query, verified live 2026-08-31:

- ``https://eprint.iacr.org/robots.txt`` carries ``Disallow: /search`` for
  every named agent and ``User-agent: * -> Disallow: /``, and states that
  crawlers must be whitelisted.  The donor connector
  (``paper_search_mcp/academic_platforms/iacr.py:20,170``) scrapes exactly that
  path.  It is not ported.
- The same file names the sanctioned alternative in its own opening comment:
  *"Metadata is available from this site using OAI-PMH, RSS, and ATOM."*
- IACR's OAI-PMH endpoint has **no query facility at all**.  ``ListSets``
  returns ``<error code="noSetHierarchy">We do not support sets yet.</error>``,
  and ``ListMetadataFormats`` offers only ``oai_dc``.  ``ListRecords`` accepts
  ``from``, ``until`` and ``metadataPrefix`` -- a date window and nothing else.

So the archive is harvested by date window and filtered here.  Every field the
predicate can read comes from ``oai_dc``, whose full tag set on this repository
is ``dc:title``, ``dc:creator``, ``dc:subject``, ``dc:description``,
``dc:identifier``, ``dc:date``, ``dc:publisher``, ``dc:rights``, ``dc:type``
and ``dc:language``.

The consequence for coverage is stated once, here, and enforced by the
connector: **a result assembled this way is never ``COMPLETE``.**  What was
consulted is a slice of the archive chosen by date, not the set of records
matching the query, and the archive never reports how many records match a
query because it cannot be asked one.
"""

from __future__ import annotations

from lit_search.core.query import FilterCode
from lit_search.query.local_match import (
    LocalPredicateQueryBuilder,
    Predicate,
    Record,
)

__all__ = ["FIELD_MAP", "IACRQueryBuilder", "Predicate", "Record"]

# Mapping from lit_search filter codes to the ``oai_dc`` elements the local
# predicate reads for them.  Every element name was read from a live
# ``ListRecords`` response on 2026-08-31.
FIELD_MAP: dict[FilterCode, tuple[str, ...]] = {
    FilterCode.TITLE: ("title",),
    FilterCode.ABSTRACT: ("description",),
    FilterCode.KEYWORDS: ("subject",),
    FilterCode.AUTHOR: ("creator",),
    FilterCode.TITLE_ABSTRACT: ("title", "description"),
    FilterCode.TITLE_ABSTRACT_KEYWORDS: ("title", "description", "subject"),
}


class IACRQueryBuilder(LocalPredicateQueryBuilder):
    """Compile queries into a local predicate over harvested IACR records.

    The matching machinery is shared with the other sources that cannot be
    asked a question -- see :mod:`lit_search.query.local_match`, which also
    states the coverage consequence.  What is specific to IACR is
    :data:`FIELD_MAP`: the set of ``oai_dc`` elements this repository actually
    serves.
    """

    FIELD_MAP = FIELD_MAP
    SOURCE_LABEL = "IACR ePrint"
    UNSUPPORTED_FIELDS_NOTE = (
        "their OAI-PMH endpoint serves only 'oai_dc', which carries no venue or "
        "affiliation for this repository."
    )
