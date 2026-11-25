"""Europe PMC query builder.

Field names and syntax behaviour below were verified against the live API on
2026-08-30 (``https://www.ebi.ac.uk/europepmc/webservices/rest``, service
version 6.9).  Two of those observations are load-bearing and are the reason
this builder validates rather than passing text through:

1. **An unrecognised field name is not an error.**  ``NOTAFIELD:foo`` returns
   HTTP 200 with ``hitCount: 0`` and an empty result list.  A builder that
   emitted a field Europe PMC does not index would therefore produce a
   confident empty search rather than a failure, so every field name this
   builder can emit comes from the whitelist below, checked against the live
   ``/fields`` endpoint (143 indexed terms, all eight mappings present).

2. **A wildcard beside a space silently loses its field scope.**
   ``TITLE_ABS:machine learn*`` returned 249,661 hits, against 378,639 for
   ``TITLE_ABS:machine`` and 998,939 for ``TITLE_ABS:learn*`` -- consistent
   with the second token being applied to the default field instead of
   ``TITLE_ABS``.  The query still runs and still returns plausible papers,
   which is exactly the shape of failure this project exists to prevent, so
   such a term is rejected as unsupported instead.
"""

from __future__ import annotations

from lit_search.core.query import ConnectorType, FilterCode, Query, QueryNode
from lit_search.query.builder import QueryBuilder, QueryValidationResult

# Mapping from lit_search filter codes to Europe PMC indexed search fields.
# Every value was confirmed present in the live ``/fields`` response on
# 2026-08-30.  ``TITLE_ABSTRACT_KEYWORDS`` has no single Europe PMC field and
# is expanded to a disjunction in :meth:`EuropePMCQueryBuilder.convert_query`.
FIELD_MAP: dict[FilterCode, str] = {
    FilterCode.TITLE: "TITLE",
    FilterCode.ABSTRACT: "ABSTRACT",
    FilterCode.KEYWORDS: "KW",
    FilterCode.AUTHOR: "AUTH",
    FilterCode.SOURCE: "JOURNAL",
    FilterCode.AFFILIATION: "AFF",
    FilterCode.TITLE_ABSTRACT: "TITLE_ABS",
}

_WILDCARD_CHARS = ("*", "?")


class EuropePMCQueryBuilder(QueryBuilder):
    """Build Europe PMC search expressions.

    Europe PMC indexes an equivalent of every filter code lit-search defines,
    so no filter is rejected on capability grounds.  What *is* rejected is a
    term Europe PMC would run but not scope the way the query says -- see the
    module docstring.
    """

    _SUPPORTED_FILTERS = frozenset(
        {
            FilterCode.TITLE,
            FilterCode.ABSTRACT,
            FilterCode.KEYWORDS,
            FilterCode.AUTHOR,
            FilterCode.SOURCE,
            FilterCode.AFFILIATION,
            FilterCode.TITLE_ABSTRACT,
            FilterCode.TITLE_ABSTRACT_KEYWORDS,
        }
    )

    def validate_query(self, query: Query) -> QueryValidationResult:
        """Validate whether Europe PMC can express this query faithfully.

        Parameters
        ----------
        query : Query
            Query to validate.

        Returns
        -------
        QueryValidationResult
            Invalid when a term is empty, or when a term combines a wildcard
            with a space.  Both would run upstream and return a plausible
            number of papers for a query other than the one asked.
        """
        for term in self.iter_term_nodes(query.root):
            value = (term.value or "").strip()
            if not value:
                return QueryValidationResult(
                    is_valid=False,
                    error_message=(
                        "Europe PMC rejects an empty search criterion "
                        "(HTTP 200 with errCode 404), so an empty term cannot be run."
                    ),
                )
            if any(char in value for char in _WILDCARD_CHARS) and " " in value:
                return QueryValidationResult(
                    is_valid=False,
                    error_message=(
                        f"Term '{value}' combines a wildcard with a space. Europe PMC "
                        "applies the field prefix only to the first token of an unquoted "
                        "term and does not expand wildcards inside a quoted phrase, so "
                        "this term would silently search a different field than requested."
                    ),
                )
            filter_code = self.get_effective_filter(term)
            if not self.supports_filter(filter_code):
                return QueryValidationResult(
                    is_valid=False,
                    error_message=f"Filter '{filter_code}' is not supported by Europe PMC.",
                )
        return QueryValidationResult(is_valid=True)

    def convert_query(self, query: Query) -> str:
        """Convert a query into Europe PMC search syntax.

        Parameters
        ----------
        query : Query
            Query to convert.  Must have passed :meth:`validate_query`.

        Returns
        -------
        str
            Europe PMC query expression.
        """
        connector_map = {
            ConnectorType.AND: "AND",
            ConnectorType.OR: "OR",
            ConnectorType.AND_NOT: "NOT",
        }

        def convert_term(term_node: QueryNode) -> str:
            value = (term_node.value or "").strip()
            rendered = _render_value(value)
            filter_code = self.get_effective_filter(term_node)
            if filter_code is FilterCode.TITLE_ABSTRACT_KEYWORDS:
                # No single Europe PMC field spans title, abstract and
                # keywords, so the disjunction is written out.
                return f"(TITLE_ABS:{rendered} OR KW:{rendered})"
            field = FIELD_MAP.get(filter_code, "TITLE_ABS")
            return f"{field}:{rendered}"

        return self.convert_expression(query.root, convert_term, connector_map)


def _render_value(value: str) -> str:
    """Render a term value for Europe PMC, quoting where that is safe.

    A term is quoted so that multi-word values search as a phrase rather than
    as separate tokens, which is what the bracketed term syntax means.  A term
    carrying a wildcard is left unquoted, because Europe PMC does not expand
    wildcards inside a quoted phrase; :meth:`EuropePMCQueryBuilder.validate_query`
    has already guaranteed such a term has no space, so leaving it unquoted
    cannot leak past the field prefix.

    Parameters
    ----------
    value : str
        Stripped term value.

    Returns
    -------
    str
        Quoted or bare term, safe to concatenate after a field prefix.
    """
    if any(char in value for char in _WILDCARD_CHARS):
        return value
    return '"' + value.replace('"', r"\"") + '"'
