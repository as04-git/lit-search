"""Query builder for Google Scholar.

Scholar takes one free-text ``q`` parameter.  It supports ``AND``, ``OR``,
``-`` for exclusion, quoted phrases, ``intitle:`` and ``author:``, and nothing
resembling an abstract or keyword field.

The interesting decisions here are all rejections, and each exists because the
alternative is a silently wrong scope:

- ``[abs]`` is **rejected**.  Scholar has no abstract-only scope.  Sending the
  term unscoped would search the full text instead, silently widening the
  query; a widened query that still finds nothing looks exactly like a narrow
  one that found nothing.
- ``[key]`` and ``[aff]`` are rejected for the same reason: no such scope
  exists.
- ``[tiabs]`` and ``[tiabskey]`` -- the *defaults* -- are rejected too, which
  is the aggressive one.  A bare term in this project's syntax means "title and
  abstract", and Scholar's unscoped search covers the whole document.  Quietly
  accepting the default would make every ordinary query mean something other
  than what it says on every other database in the ledger, and the difference
  would never surface.  A caller who wants Scholar's own semantics writes
  ``[ti]`` for a title search or asks for full text explicitly through
  ``[src]``, which is mapped to the unscoped form and therefore states the
  intent.

The consequence is that Scholar reports ``QUERY_UNSUPPORTED`` for a plain
query rather than answering a different question.  That is a loud, determinate
outcome the ledger records, which is the whole point.
"""

from __future__ import annotations

from lit_search.core.query import ConnectorType, FilterCode, Query, QueryNode
from lit_search.query.builder import QueryBuilder, QueryValidationResult

#: Scholar's own field prefixes.  ``FilterCode.SOURCE`` maps to the empty
#: prefix, meaning Scholar's default full-document search -- a caller reaching
#: for it has said so explicitly.
_PREFIXES: dict[FilterCode, str] = {
    FilterCode.TITLE: "intitle:",
    FilterCode.AUTHOR: "author:",
    FilterCode.SOURCE: "",
}

_UNSCOPED_NOTE = (
    "Google Scholar has no abstract, keyword or affiliation scope: it searches the whole "
    "document. Accepting the term anyway would answer a broader question than the one "
    "asked, on one database out of many, with nothing in the result to say so. Scope the "
    "term with [ti] or [au], or use [src] to ask for Scholar's full-document search "
    "deliberately."
)


class GoogleScholarQueryBuilder(QueryBuilder):
    """Render a query as a Google Scholar ``q`` string."""

    _SUPPORTED_FILTERS = frozenset(_PREFIXES)

    def validate_query(self, query: Query) -> QueryValidationResult:
        """Validate whether Scholar can scope every term as asked.

        Parameters
        ----------
        query : Query
            Query to validate.

        Returns
        -------
        QueryValidationResult
            Invalid when a term is empty or asks for a scope Scholar does not
            have -- including the project-wide default scope.
        """
        for term in self.iter_term_nodes(query.root):
            if not (term.value or "").strip():
                return QueryValidationResult(
                    is_valid=False,
                    error_message="An empty term matches everything Scholar indexes.",
                )
            filter_code = self.get_effective_filter(term)
            if not self.supports_filter(filter_code):
                return QueryValidationResult(
                    is_valid=False,
                    error_message=f"Filter '{filter_code}' cannot be expressed. {_UNSCOPED_NOTE}",
                )
        return QueryValidationResult(is_valid=True)

    def convert_query(self, query: Query) -> str:
        """Render the query as Scholar's ``q`` value.

        Parameters
        ----------
        query : Query
            Query to convert.  Must have passed :meth:`validate_query`.

        Returns
        -------
        str
            Query string.
        """
        connector_map = {
            ConnectorType.AND: "AND",
            ConnectorType.OR: "OR",
            ConnectorType.AND_NOT: "-",
        }

        def convert_term(term_node: QueryNode) -> str:
            """Render one term with its Scholar prefix.

            Parameters
            ----------
            term_node : QueryNode
                Term node to render.

            Returns
            -------
            str
                Rendered term.
            """
            value = (term_node.value or "").strip()
            quoted = f'"{value}"' if " " in value else value
            return f"{_PREFIXES[self.get_effective_filter(term_node)]}{quoted}"

        return self.convert_expression(query.root, convert_term, connector_map)
