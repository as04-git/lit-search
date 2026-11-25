"""OpenAIRE Graph API query builder.

The OpenAIRE Graph API takes a **query parameter per field** rather than one
query string, and combines those parameters with a single global
``logicalOperator``.  That shape decides the design of this builder, so
:meth:`OpenAIREQueryBuilder.convert_query` returns a one-entry ``dict`` naming
the parameter and its expression rather than a string.

Field names and syntax behaviour were verified live on 2026-08-31 against
``https://api.openaire.eu/graph/v3/research-products``.  Four observations are
load-bearing:

1. **Boolean operators work inside a single field parameter, and grouping
   works.**  ``search=catalysis`` returned 278,812; ``search=(catalysis OR
   photolysis)`` returned 334,539; ``search=catalysis AND enzyme`` returned
   30,758; ``search=catalysis NOT enzyme`` returned 248,054 -- which is exactly
   278,812 minus 30,758.  ``mainTitle=("machine learning" OR "deep learning")``
   returned 824,699.  So a whole boolean tree can be rendered into one
   parameter, provided every term targets the same field.

2. **An unquoted multi-word term is AND-ed, not phrase-matched.**
   ``mainTitle=machine learning`` and ``mainTitle=machine AND learning`` both
   returned **528,357**, while ``mainTitle="machine learning"`` returned
   509,041.  lit-search's bracketed term means a phrase, so multi-word terms are
   quoted; without that, every multi-word term silently widens.

3. **Wildcards do not expand, and fail quietly.**  ``mainTitle=catalys*``
   returned **103** records against **86,158** for ``mainTitle=catalysis``.  A
   wildcard therefore does not broaden the search -- it collapses it to
   something small and plausible.  Wildcard terms are refused rather than sent.

4. **An empty query is the whole graph, and a wrong field name is a silent
   zero.**  ``search=`` returned 386,695,835 records; ``search=NOTAFIELD:x``
   returned 0 with HTTP 200.  Both are rejected here.

Unlike Europe PMC, this builder also refuses a query whose terms target
*different* fields.  The API's ``logicalOperator`` applies one operator across
all field parameters, so a tree such as ``ti[a] AND (abs[b] OR abs[c])`` has no
faithful encoding.  Sending an approximation would return a plausible set for a
different question, which is the failure this fork exists to prevent.
"""

from __future__ import annotations

from lit_search.core.query import ConnectorType, FilterCode, Query, QueryNode
from lit_search.query.builder import QueryBuilder, QueryValidationResult

# Mapping from lit_search filter codes to OpenAIRE Graph API query parameters.
# Each was confirmed to return a plausible non-zero count on 2026-08-31:
# mainTitle 86,158; description 894,586; subjects 181,279; authorFullName 944;
# search 278,812.
FIELD_MAP: dict[FilterCode, str] = {
    FilterCode.TITLE: "mainTitle",
    FilterCode.ABSTRACT: "description",
    FilterCode.KEYWORDS: "subjects",
    FilterCode.AUTHOR: "authorFullName",
    FilterCode.TITLE_ABSTRACT: "search",
}

_WILDCARD_CHARS = ("*", "?")


class OpenAIREQueryBuilder(QueryBuilder):
    """Build OpenAIRE Graph API query parameters.

    :data:`~lit_search.core.query.FilterCode.TITLE_ABSTRACT_KEYWORDS` is
    deliberately **not** supported.  OpenAIRE's ``search`` parameter covers
    title and abstract but not subjects, and there is no parameter spanning all
    three.  Because
    :meth:`~lit_search.query.builder.QueryBuilder.get_effective_filter` falls
    back to :data:`~lit_search.core.query.FilterCode.TITLE_ABSTRACT` when the
    builder does not support the keyword-inclusive default, an unfiltered query
    lands on ``search`` -- which is what it says it is -- instead of on a
    parameter that quietly omits one of the three fields it claims to cover.

    :data:`~lit_search.core.query.FilterCode.SOURCE` and
    :data:`~lit_search.core.query.FilterCode.AFFILIATION` are refused for the
    same reason: the Graph API exposes ``publisher``, which is the publishing
    house rather than the venue, and exposes nothing for affiliation.
    """

    _SUPPORTED_FILTERS = frozenset(FIELD_MAP)

    def validate_query(self, query: Query) -> QueryValidationResult:
        """Validate whether OpenAIRE can express this query faithfully.

        Parameters
        ----------
        query : Query
            Query to validate.

        Returns
        -------
        QueryValidationResult
            Invalid when a term is empty, carries a wildcard, uses a filter
            OpenAIRE does not expose, or when the query's terms do not all
            target one field parameter.
        """
        parameters: set[str] = set()
        for term in self.iter_term_nodes(query.root):
            value = (term.value or "").strip()
            failure = self._reject_term(value)
            if failure is not None:
                return failure
            filter_code = self.get_effective_filter(term)
            if not self.supports_filter(filter_code):
                return QueryValidationResult(
                    is_valid=False,
                    error_message=(
                        f"Filter '{filter_code}' has no faithful OpenAIRE Graph API "
                        "parameter. The API exposes mainTitle, description, subjects and "
                        "authorFullName, plus a title-and-abstract 'search'; it exposes "
                        "no venue title and no affiliation."
                    ),
                )
            parameters.add(FIELD_MAP[filter_code])

        if len(parameters) > 1:
            return QueryValidationResult(
                is_valid=False,
                error_message=(
                    "OpenAIRE takes one query parameter per field and joins them with a "
                    f"single global logicalOperator, so a query spanning "
                    f"{sorted(parameters)} cannot keep its own operator structure. "
                    "Split it into one search per field rather than sending an "
                    "approximation that would answer a different question."
                ),
            )
        return QueryValidationResult(is_valid=True)

    @staticmethod
    def _reject_term(value: str) -> QueryValidationResult | None:
        """Return a rejection for *value*, or ``None`` when it is safe to send.

        Parameters
        ----------
        value : str
            Stripped term value.

        Returns
        -------
        QueryValidationResult | None
            An invalid result when the term would be silently reinterpreted.
        """
        if not value:
            return QueryValidationResult(
                is_valid=False,
                error_message=(
                    "OpenAIRE treats an empty query as a match-all: 'search=' returned "
                    "386,695,835 records on 2026-08-31. An empty term cannot be sent, "
                    "because the result would be the whole graph reported as a match."
                ),
            )
        if any(char in value for char in _WILDCARD_CHARS):
            return QueryValidationResult(
                is_valid=False,
                error_message=(
                    f"Term '{value}' contains a wildcard, which OpenAIRE does not expand. "
                    "Verified 2026-08-31: 'mainTitle=catalys*' returned 103 records while "
                    "'mainTitle=catalysis' returned 86,158, so the wildcard narrows the "
                    "search to a small plausible set instead of broadening it."
                ),
            )
        return None

    def convert_query(self, query: Query) -> dict[str, str]:
        """Convert a query into a single OpenAIRE query parameter.

        Parameters
        ----------
        query : Query
            Query to convert.  Must have passed :meth:`validate_query`, which
            guarantees every term maps to the same parameter.

        Returns
        -------
        dict[str, str]
            One entry mapping the OpenAIRE parameter name to its expression.
        """
        connector_map = {
            ConnectorType.AND: "AND",
            ConnectorType.OR: "OR",
            ConnectorType.AND_NOT: "NOT",
        }

        def convert_term(term_node: QueryNode) -> str:
            return _render_value((term_node.value or "").strip())

        expression = self.convert_expression(query.root, convert_term, connector_map)
        return {self.parameter_for(query): expression}

    def parameter_for(self, query: Query) -> str:
        """Return the single OpenAIRE parameter name this query targets.

        Parameters
        ----------
        query : Query
            Query to inspect.  Must have passed :meth:`validate_query`.

        Returns
        -------
        str
            Parameter name, defaulting to ``"search"`` for a query with no
            terms at all.
        """
        for term in self.iter_term_nodes(query.root):
            return FIELD_MAP[self.get_effective_filter(term)]
        return "search"


def _render_value(value: str) -> str:
    """Render a term value for OpenAIRE as a quoted phrase.

    Every term is quoted.  An unquoted multi-word term is AND-ed rather than
    phrase-matched -- ``mainTitle=machine learning`` and
    ``mainTitle=machine AND learning`` returned the identical 528,357 on
    2026-08-31, against 509,041 for the quoted form -- and lit-search's bracketed
    term means a phrase.  Wildcards, the one case Europe PMC leaves unquoted,
    are rejected outright by :meth:`OpenAIREQueryBuilder.validate_query`, so
    there is no term for which quoting is the wrong choice.

    Parameters
    ----------
    value : str
        Stripped term value.

    Returns
    -------
    str
        Quoted term, safe to place inside a boolean expression.
    """
    return '"' + value.replace('"', r"\"") + '"'
