"""Query builder for BASE.

BASE's search interface is IP-gated and unreachable from here (see
:mod:`lit_search.connectors.base_search`), so no query is ever sent.  This
builder exists because the coverage contract runs validation *before*
:meth:`~lit_search.connectors.search_base.SearchConnectorBase.search` reaches
the connector body, and a source that is skipped should be reported as skipped
rather than as unable to express the query.

It therefore accepts what BASE's documented ``func=PerformSearch`` grammar
accepts -- title, abstract, author, subject and the combined scopes -- and
converts a query into the ``dcXX:`` field syntax BASE documents.  The
conversion is **unexercised**: no response has been observed, so the rendering
is checked against BASE's published field list and nothing more.  It is here so
that the day an IP registration exists, the work left is a parser rather than a
parser and a grammar.
"""

from __future__ import annotations

from lit_search.core.query import ConnectorType, FilterCode, Query, QueryNode
from lit_search.query.builder import QueryBuilder, QueryValidationResult

#: BASE Dublin Core field names, from the interface's published documentation.
#: **Not verified against a live response**, because every route refuses this
#: host; see the connector module.
_FIELD_NAMES: dict[FilterCode, str] = {
    FilterCode.TITLE: "dctitle",
    FilterCode.ABSTRACT: "dcdescription",
    FilterCode.KEYWORDS: "dcsubject",
    FilterCode.AUTHOR: "dccreator",
    FilterCode.SOURCE: "dcsource",
}


class BaseSearchQueryBuilder(QueryBuilder):
    """Render a query in BASE's documented field syntax."""

    _SUPPORTED_FILTERS = frozenset(
        {
            *_FIELD_NAMES,
            FilterCode.TITLE_ABSTRACT,
            FilterCode.TITLE_ABSTRACT_KEYWORDS,
        }
    )

    def validate_query(self, query: Query) -> QueryValidationResult:
        """Validate whether BASE's grammar could express this query.

        Parameters
        ----------
        query : Query
            Query to validate.

        Returns
        -------
        QueryValidationResult
            Invalid when a term is empty or names a field BASE has no Dublin
            Core element for.
        """
        for term in self.iter_term_nodes(query.root):
            if not (term.value or "").strip():
                return QueryValidationResult(
                    is_valid=False,
                    error_message="An empty term matches every record in the index.",
                )
            filter_code = self.get_effective_filter(term)
            if not self.supports_filter(filter_code):
                return QueryValidationResult(
                    is_valid=False,
                    error_message=(
                        f"Filter '{filter_code}' has no Dublin Core element in BASE's "
                        f"documented field list, so the query cannot be scoped as asked."
                    ),
                )
        return QueryValidationResult(is_valid=True)

    def convert_query(self, query: Query) -> str:
        """Render the query in BASE's ``dcXX:`` syntax.

        Parameters
        ----------
        query : Query
            Query to convert.  Must have passed :meth:`validate_query`.

        Returns
        -------
        str
            Rendered query string.
        """
        connector_map = {
            ConnectorType.AND: "AND",
            ConnectorType.OR: "OR",
            ConnectorType.AND_NOT: "AND NOT",
        }

        def convert_term(term_node: QueryNode) -> str:
            """Render one term with its BASE field prefix.

            Parameters
            ----------
            term_node : QueryNode
                Term node to render.

            Returns
            -------
            str
                Rendered term.
            """
            value = self.quote_term((term_node.value or "").strip())
            filter_code = self.get_effective_filter(term_node)
            fields = _fields_for(filter_code)
            if len(fields) == 1:
                return f"{fields[0]}:{value}"
            return "(" + " OR ".join(f"{name}:{value}" for name in fields) + ")"

        return self.convert_expression(query.root, convert_term, connector_map)


def _fields_for(filter_code: FilterCode) -> tuple[str, ...]:
    """Return the BASE field names one filter code expands to.

    Parameters
    ----------
    filter_code : FilterCode
        Effective filter for a term.

    Returns
    -------
    tuple[str, ...]
        One or more Dublin Core element names.
    """
    if filter_code is FilterCode.TITLE_ABSTRACT:
        return (_FIELD_NAMES[FilterCode.TITLE], _FIELD_NAMES[FilterCode.ABSTRACT])
    if filter_code is FilterCode.TITLE_ABSTRACT_KEYWORDS:
        return (
            _FIELD_NAMES[FilterCode.TITLE],
            _FIELD_NAMES[FilterCode.ABSTRACT],
            _FIELD_NAMES[FilterCode.KEYWORDS],
        )
    return (_FIELD_NAMES[filter_code],)
