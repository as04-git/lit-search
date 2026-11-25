"""HAL query builder.

HAL's search API is a Solr front end, so unlike CrossRef it has real Boolean
operators.  Field names and syntax behaviour below were verified against the
live API on 2026-08-31 (``https://api.archives-ouvertes.fr/search/``).

Two observations drive the validation this builder does:

1. **An unrecognised field name is an HTTP 200 error body, not an error.**
   ``q=notafield_zz:catalysis`` returns status 200 with
   ``{"error":{"msg":"Error. See help : /docs"}}`` -- no ``response`` object,
   so ``data.get("response", {}).get("docs", [])`` yields ``[]`` and the donor
   connector reports it as "HAL holds no matching paper".  Every field name
   this builder can emit is therefore taken from the whitelist below, each
   checked live for a plausible non-zero hit count.

2. **A wildcard beside a space silently loses its field scope.**
   ``title_t:machine`` returned 14,537 and ``title_t:learn*`` returned 45,173,
   while ``title_t:machine learn*`` returned 8,131 and the correctly grouped
   ``title_t:(machine learn*)`` returned 7,699.  The ungrouped form is neither
   of the single-term counts and is not the grouped conjunction either, which
   is what you get when the second token is applied to Solr's default field
   instead of ``title_t``.  The query runs and returns plausible papers for a
   question nobody asked, so such a term is rejected rather than sent.

A third observation shapes what the *connector* does rather than this builder:
HAL's error body is the same seven-word string whatever went wrong, so nothing
in a failed response distinguishes "your query is malformed" from "the service
broke".  That is why this builder rejects the malformed-query shapes it can
detect locally -- unbalanced quotes and parentheses both produce that body --
instead of leaving them to be misreported downstream.
"""

from __future__ import annotations

from lit_search.core.query import ConnectorType, FilterCode, Query, QueryNode
from lit_search.query.builder import QueryBuilder, QueryValidationResult

#: Mapping from lit_search filter codes to HAL Solr text fields.  Every value
#: was queried live on 2026-08-31 and returned a plausible non-zero count;
#: ``TITLE_ABSTRACT`` and ``TITLE_ABSTRACT_KEYWORDS`` have no single HAL field
#: and are expanded to disjunctions in :meth:`HALQueryBuilder.convert_query`.
FIELD_MAP: dict[FilterCode, str] = {
    FilterCode.TITLE: "title_t",
    FilterCode.ABSTRACT: "abstract_t",
    FilterCode.KEYWORDS: "keyword_t",
    FilterCode.AUTHOR: "authFullName_t",
    FilterCode.SOURCE: "journalTitle_t",
    FilterCode.AFFILIATION: "structName_t",
}

#: Fields the composite scopes expand to.
_TITLE_ABSTRACT_FIELDS = ("title_t", "abstract_t")
_TITLE_ABSTRACT_KEYWORDS_FIELDS = ("title_t", "abstract_t", "keyword_t")

_WILDCARD_CHARS = ("*", "?")

#: Characters that would end the quoted phrase this builder emits, or escape
#: the character after it, and so change the parse of everything downstream.
#: Both produce HAL's opaque HTTP 200 error body when they unbalance the query.
_QUOTE_BREAKING_CHARS = ('"', "\\")


class HALQueryBuilder(QueryBuilder):
    """Build HAL Solr search expressions.

    HAL indexes an equivalent of every filter code lit-search defines, so no
    filter is rejected on capability grounds.  What *is* rejected is a term HAL
    would run but not scope the way the query says -- see the module docstring.
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
        """Validate whether HAL can express this query faithfully.

        Parameters
        ----------
        query : Query
            Query to validate.

        Returns
        -------
        QueryValidationResult
            Invalid when a term is empty, carries a character that would
            unbalance the emitted phrase, or combines a wildcard with a space.
        """
        for term in self.iter_term_nodes(query.root):
            value = (term.value or "").strip()
            if not value:
                return QueryValidationResult(
                    is_valid=False,
                    error_message=(
                        "HAL cannot run an empty search term: an empty Solr clause "
                        "returns HTTP 200 with an opaque error body, which is "
                        "indistinguishable from a search that matched nothing."
                    ),
                )
            if any(char in value for char in _QUOTE_BREAKING_CHARS):
                return QueryValidationResult(
                    is_valid=False,
                    error_message=(
                        f"Term '{value}' contains a quote or backslash. This builder emits "
                        "terms as quoted Solr phrases, and either character would unbalance "
                        "the expression; HAL answers an unbalanced query with HTTP 200 and "
                        "an error body that reads the same as an empty result."
                    ),
                )
            if any(char in value for char in _WILDCARD_CHARS) and " " in value:
                return QueryValidationResult(
                    is_valid=False,
                    error_message=(
                        f"Term '{value}' combines a wildcard with a space. Solr applies the "
                        "field prefix only to the first token of an unquoted term and does "
                        "not expand wildcards inside a quoted phrase, so this term would "
                        "silently search a different field than requested."
                    ),
                )
            filter_code = self.get_effective_filter(term)
            if not self.supports_filter(filter_code):
                return QueryValidationResult(
                    is_valid=False,
                    error_message=f"Filter '{filter_code}' is not supported by HAL.",
                )
        return QueryValidationResult(is_valid=True)

    def convert_query(self, query: Query) -> str:
        """Convert a query into HAL Solr syntax.

        Parameters
        ----------
        query : Query
            Query to convert.  Must have passed :meth:`validate_query`.

        Returns
        -------
        str
            HAL Solr query expression.
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
            fields: tuple[str, ...]
            if filter_code is FilterCode.TITLE_ABSTRACT:
                fields = _TITLE_ABSTRACT_FIELDS
            elif filter_code is FilterCode.TITLE_ABSTRACT_KEYWORDS:
                fields = _TITLE_ABSTRACT_KEYWORDS_FIELDS
            else:
                return f"{FIELD_MAP[filter_code]}:{rendered}"
            inner = " OR ".join(f"{field}:{rendered}" for field in fields)
            return f"({inner})"

        return self.convert_expression(query.root, convert_term, connector_map)


def _render_value(value: str) -> str:
    """Render a term value for HAL, quoting where that is safe.

    A term is quoted so that a multi-word value searches as a phrase rather
    than as separate tokens, which is what the bracketed term syntax means.  A
    term carrying a wildcard is left unquoted, because Solr does not expand
    wildcards inside a quoted phrase; :meth:`HALQueryBuilder.validate_query`
    has already guaranteed such a term has no space, so leaving it unquoted
    cannot leak past the field prefix.

    Parameters
    ----------
    value : str
        Stripped term value.  Guaranteed by ``validate_query`` to contain no
        quote or backslash.

    Returns
    -------
    str
        Quoted or bare term, safe to concatenate after a field prefix.
    """
    if any(char in value for char in _WILDCARD_CHARS):
        return value
    return f'"{value}"'
