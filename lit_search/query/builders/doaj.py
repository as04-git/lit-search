"""DOAJ query builder.

Field names and syntax behaviour below were verified against the live API on
2026-08-31 (``https://doaj.org/api/search/articles``, which serves the v4
schema).  Three of those observations decide the design of this builder, and
each is a way a plausible-looking query returns the wrong answer with no error.

1. **An unrecognised field name is not an error.**  ``notafield:cancer``
   returns HTTP 200 with ``total: 0`` and an empty ``results`` list, exactly as
   a genuinely unmatched query does.  So every field name here comes from the
   whitelist below and every entry in it was checked against the live index.

2. **The short-hand field names are a trap, and the donor connector walks into
   it.**  DOAJ documents short-hand names for *some* fields only, and the
   plausible short form of the others silently matches nothing.  Measured on
   the same query text:

   ===============================  =========  ================================
   Field expression                 ``total``  Verdict
   ===============================  =========  ================================
   ``bibjson.title:cancer``           292,130  works
   ``title:cancer``                   292,130  documented short-hand, works
   ``bibjson.author.name:smith``       50,171  works
   ``author:smith``                         0  **silent zero**
   ``bibjson.keywords:cancer``        221,539  works
   ``keywords:cancer``                      0  **silent zero**
   ``bibjson.year:2020``              883,363  works
   ``year:2020``                            0  **silent zero**
   ``bibjson.journal.title:oncology``  95,519  works
   ``journal.title:oncology``               0  **silent zero**
   ===============================  =========  ================================

   ``paper_search_mcp``'s ``DOAJSearcher._build_lucene_query`` emits
   ``year:``, ``journal.title:``, ``country:``, ``language:`` and ``subject:``.
   Four of those five return a confident zero.  This builder emits the
   ``bibjson.``-prefixed paths only.

3. **Wildcards are refused outright, with HTTP 400.**  ``bibjson.title:cancer*``
   and ``bibjson.title:can?er`` both return
   ``{"status": "bad_request", "error": "Query contains disallowed Lucene
   features"}``.  A wildcard term is therefore rejected in
   :meth:`DOAJQueryBuilder.validate_query` rather than sent and turned into a
   failure that looks like an outage.
"""

from __future__ import annotations

from lit_search.core.query import ConnectorType, FilterCode, Query, QueryNode
from lit_search.query.builder import QueryBuilder, QueryValidationResult

# Mapping from lit_search filter codes to DOAJ article index paths.  Every
# value returned a non-zero count against the live index on 2026-08-31; see the
# module docstring for why the documented short-hand forms are not used.
FIELD_MAP: dict[FilterCode, str] = {
    FilterCode.TITLE: "bibjson.title",
    FilterCode.ABSTRACT: "bibjson.abstract",
    FilterCode.KEYWORDS: "bibjson.keywords",
    FilterCode.AUTHOR: "bibjson.author.name",
    FilterCode.SOURCE: "bibjson.journal.title",
    FilterCode.AFFILIATION: "bibjson.author.affiliation",
}

# Composite scopes DOAJ has no single field for.  Written out as disjunctions
# rather than approximated by one of their members, because dropping a member
# would narrow the search without saying so.
COMPOSITE_FIELDS: dict[FilterCode, tuple[str, ...]] = {
    FilterCode.TITLE_ABSTRACT: ("bibjson.title", "bibjson.abstract"),
    FilterCode.TITLE_ABSTRACT_KEYWORDS: (
        "bibjson.title",
        "bibjson.abstract",
        "bibjson.keywords",
    ),
}

# Field carrying the publication year.  DOAJ indexes no finer publication-date
# granularity than this, which is why :class:`~lit_search.connectors.doaj.DOAJConnector`
# reports a sub-year date bound as a shortfall rather than pretending to honour it.
YEAR_FIELD = "bibjson.year"

_WILDCARD_CHARS = ("*", "?")


class DOAJQueryBuilder(QueryBuilder):
    """Build DOAJ article-search expressions.

    DOAJ indexes an equivalent of every filter code lit-search defines, so no
    filter is rejected on capability grounds.  What *is* rejected is a term
    DOAJ refuses to run at all -- see the module docstring.
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
        """Validate whether DOAJ can express this query faithfully.

        Parameters
        ----------
        query : Query
            Query to validate.

        Returns
        -------
        QueryValidationResult
            Invalid when a term is empty or carries a wildcard.  DOAJ answers
            both with an HTTP error rather than a result, so sending them
            spends rate-limit budget to learn what is already known here.
        """
        for term in self.iter_term_nodes(query.root):
            value = (term.value or "").strip()
            if not value:
                return QueryValidationResult(
                    is_valid=False,
                    error_message=(
                        "DOAJ has no endpoint for an empty search term "
                        "(the request path 404s), so an empty term cannot be run."
                    ),
                )
            if any(char in value for char in _WILDCARD_CHARS):
                return QueryValidationResult(
                    is_valid=False,
                    error_message=(
                        f"Term '{value}' contains a wildcard. DOAJ refuses wildcard queries "
                        "with HTTP 400 'Query contains disallowed Lucene features' (verified "
                        "2026-08-31), so this query cannot be run against DOAJ at all."
                    ),
                )
            filter_code = self.get_effective_filter(term)
            if not self.supports_filter(filter_code):
                return QueryValidationResult(
                    is_valid=False,
                    error_message=f"Filter '{filter_code}' is not supported by DOAJ.",
                )
        return QueryValidationResult(is_valid=True)

    def convert_query(self, query: Query) -> str:
        """Convert a query into a DOAJ Lucene expression.

        Parameters
        ----------
        query : Query
            Query to convert.  Must have passed :meth:`validate_query`.

        Returns
        -------
        str
            DOAJ query expression, not yet URL-encoded.
        """
        connector_map = {
            ConnectorType.AND: "AND",
            ConnectorType.OR: "OR",
            ConnectorType.AND_NOT: "NOT",
        }

        def convert_term(term_node: QueryNode) -> str:
            value = (term_node.value or "").strip()
            rendered = quote_value(value)
            filter_code = self.get_effective_filter(term_node)
            fields = COMPOSITE_FIELDS.get(filter_code)
            if fields is not None:
                inner = " OR ".join(f"{field}:{rendered}" for field in fields)
                return f"({inner})"
            field = FIELD_MAP.get(filter_code, "bibjson.title")
            return f"{field}:{rendered}"

        return self.convert_expression(query.root, convert_term, connector_map)


def quote_value(value: str) -> str:
    """Render a term value as a quoted DOAJ phrase.

    Every term is quoted.  A quoted phrase is what the bracketed term syntax
    means, and quoting additionally neutralises the Lucene operators a raw term
    could carry -- an unbalanced ``(`` returns HTTP 400 from DOAJ, verified
    2026-08-31, which would read as an outage rather than as a bad term.
    Wildcards are not a concern here because
    :meth:`DOAJQueryBuilder.validate_query` has already refused them.

    Parameters
    ----------
    value : str
        Stripped term value.

    Returns
    -------
    str
        Quoted term, safe to concatenate after a field prefix.
    """
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
