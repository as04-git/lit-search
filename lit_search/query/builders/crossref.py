"""CrossRef field-query builder.

Used by every connector that is a **CrossRef filter profile** rather than an
independent source -- today that is chemRxiv
(:mod:`lit_search.connectors.chemrxiv`).  It is separate from the connector for
that reason: the query semantics belong to ``api.crossref.org``, not to the
preprint server whose DOI prefix the profile filters on.

Everything below was measured against ``api.crossref.org`` on 2026-08-31,
inside the chemRxiv profile
(``filter=type:posted-content,prefix:10.26434``).  Three of the measurements
decide the whole design, and each of them is a query that runs, returns HTTP
200, and answers a *different question* than the one asked -- the exact shape
this fork exists to catch.

1. **CrossRef has no Boolean operators.**  Tokens inside one ``query.*``
   parameter are combined disjunctively, and ``AND`` / ``NOT`` are ordinary
   stopwords:

   ===========================================  =======
   ``query.bibliographic``                      hits
   ===========================================  =======
   ``catalysis``                                    938
   ``photocatalysis``                               148
   ``catalysis AND photocatalysis``                1079
   ``catalysis NOT photocatalysis``                1079
   ===========================================  =======

   1079 is the union, and the ``AND`` and ``NOT`` forms are byte-identical in
   their counts.  A conjunction sent to CrossRef silently becomes a
   disjunction; a negation silently disappears.  Both are therefore rejected
   in :meth:`CrossRefQueryBuilder.validate_query` rather than sent.

2. **Separate ``query.*`` parameters are combined conjunctively.**
   ``query.title=catalysis`` gave 938 and ``query.author=smith`` gave 563,
   while the two together gave 10.  So the only query shape CrossRef can
   express faithfully is *an AND of per-field OR-groups, at most one group per
   field*, and that is exactly what this builder accepts.

3. **CrossRef does not index abstracts.**  A distinctive phrase taken from one
   preprint's abstract ("comparative analysis using the three major categories
   of substrates") did not return that preprint at all, while four words from
   its title returned it at rank 1.  There is likewise no ``query.abstract``
   and no ``query.keyword``: ``query.abstract=...`` is rejected with HTTP 400
   ``field-query-not-available``.  So ``abs[...]`` and ``key[...]`` cannot be
   run at all, and ``tiabs[...]`` / ``tiabskey[...]`` can only be run in a
   narrowed form -- see :meth:`CrossRefQueryBuilder.scope_degradation`, whose
   note the connector turns into ``PARTIAL`` coverage.

A wrong field-query name is *not* a silent zero here: ``query.notafield``
returns HTTP 400 with a ``field-query-not-available`` body naming the valid
set.  The whitelist below is that set, restricted to the names this project
has a filter code for.
"""

from __future__ import annotations

from lit_search.core.query import ConnectorType, FilterCode, NodeType, Query, QueryNode
from lit_search.query.builder import QueryBuilder, QueryValidationResult

#: Mapping from lit_search filter codes to CrossRef field-query names.  Every
#: value appears in the ``field-query-not-available`` error body CrossRef
#: returns for an unknown name, checked live on 2026-08-31.
FIELD_MAP: dict[FilterCode, str] = {
    FilterCode.TITLE: "title",
    FilterCode.AUTHOR: "author",
    FilterCode.SOURCE: "container-title",
    FilterCode.AFFILIATION: "affiliation",
    FilterCode.TITLE_ABSTRACT: "bibliographic",
    FilterCode.TITLE_ABSTRACT_KEYWORDS: "bibliographic",
}

#: Filter codes whose requested scope is wider than the CrossRef field they
#: are mapped to.  Mapped to the note explaining what the source cannot reach.
_DEGRADED_SCOPES: dict[FilterCode, str] = {
    FilterCode.TITLE_ABSTRACT: (
        "'tiabs' was run against CrossRef's 'bibliographic' field query, which "
        "indexes titles, contributors and container titles but not abstracts, so "
        "records matching only in the abstract were never reachable"
    ),
    FilterCode.TITLE_ABSTRACT_KEYWORDS: (
        "'tiabskey' was run against CrossRef's 'bibliographic' field query, which "
        "indexes titles, contributors and container titles but neither abstracts "
        "nor keywords, so records matching only in those were never reachable"
    ),
}

#: Filter codes CrossRef cannot express at all.  Mapped to the reason, which is
#: reported to the user rather than silently widened to a field that would run.
_UNSUPPORTED_SCOPES: dict[FilterCode, str] = {
    FilterCode.ABSTRACT: (
        "CrossRef has no abstract field query -- 'query.abstract' is refused with "
        "HTTP 400 'field-query-not-available' -- and its free-text search does not "
        "index abstracts, so an abstract-scoped term cannot be run"
    ),
    FilterCode.KEYWORDS: (
        "CrossRef has no keyword field query, so a keyword-scoped term cannot be run"
    ),
}

_WILDCARD_CHARS = ("*", "?")


class UnexpressibleQueryError(ValueError):
    """A query shape CrossRef would run but would not mean.

    Raised internally by the clause analysis and converted into a
    :class:`~lit_search.query.builder.QueryValidationResult` by
    :meth:`CrossRefQueryBuilder.validate_query`.  It is deliberately *not* a
    :class:`~lit_search.exceptions.UnsupportedQueryError`: nothing here has
    reached a connector yet.
    """


class CrossRefQueryBuilder(QueryBuilder):
    """Build CrossRef ``query.*`` field-query parameters.

    :meth:`convert_query` returns a parameter mapping such as
    ``{"query.title": "catalysis photocatalysis", "query.author": "smith"}``,
    which CrossRef reads as *(title matches catalysis OR photocatalysis) AND
    (author matches smith)*.

    Only that shape is accepted.  See the module docstring for the
    measurements behind each restriction.
    """

    _SUPPORTED_FILTERS = frozenset(
        {
            FilterCode.TITLE,
            FilterCode.AUTHOR,
            FilterCode.SOURCE,
            FilterCode.AFFILIATION,
            FilterCode.TITLE_ABSTRACT,
            FilterCode.TITLE_ABSTRACT_KEYWORDS,
        }
    )

    def validate_query(self, query: Query) -> QueryValidationResult:
        """Validate whether CrossRef can express this query faithfully.

        Parameters
        ----------
        query : Query
            Query to validate.

        Returns
        -------
        QueryValidationResult
            Invalid when a term is empty or wildcarded, when a filter has no
            CrossRef field query, or when the Boolean shape is one CrossRef
            would run but not mean.
        """
        for term in self.iter_term_nodes(query.root):
            problem = self._term_problem(term)
            if problem is not None:
                return QueryValidationResult(is_valid=False, error_message=problem)
        try:
            self._clauses(query.root)
        except UnexpressibleQueryError as exc:
            return QueryValidationResult(is_valid=False, error_message=str(exc))
        return QueryValidationResult(is_valid=True)

    def _term_problem(self, term: QueryNode) -> str | None:
        """Return why *term* cannot be sent to CrossRef, or ``None``.

        Parameters
        ----------
        term : QueryNode
            TERM node to inspect.

        Returns
        -------
        str | None
            Explanation, or ``None`` when the term is fine.
        """
        value = (term.value or "").strip()
        if not value:
            return "CrossRef cannot run an empty search term."
        if any(char in value for char in _WILDCARD_CHARS):
            return (
                f"Term '{value}' carries a wildcard. CrossRef's field queries are "
                "relevance-ranked free text with no wildcard operator, so the "
                "wildcard character would be searched as an ordinary token and the "
                "query would silently match a different set."
            )
        filter_code = self.get_effective_filter(term)
        unsupported = _UNSUPPORTED_SCOPES.get(filter_code)
        if unsupported is not None:
            return f"Term '{value}' is scoped to '{filter_code}': {unsupported}."
        if not self.supports_filter(filter_code):
            return f"Filter '{filter_code}' is not supported by CrossRef."
        return None

    def scope_degradation(self, query: Query) -> str | None:
        """Return a note naming every scope CrossRef narrowed, or ``None``.

        The connector turns a non-``None`` return into ``PARTIAL`` coverage.
        That is not pedantry: a ``tiabs`` query answered without an abstract
        index has an unquantified shortfall, and CrossRef's own
        ``total-results`` counts only what it *did* search, so the shortfall is
        invisible to the pagination arithmetic.

        Parameters
        ----------
        query : Query
            Query about to be sent.

        Returns
        -------
        str | None
            Semicolon-joined notes, or ``None`` when nothing was narrowed.
        """
        notes: list[str] = []
        for term in self.iter_term_nodes(query.root):
            note = _DEGRADED_SCOPES.get(self.get_effective_filter(term))
            if note is not None and note not in notes:
                notes.append(note)
        return "; ".join(notes) or None

    def convert_query(self, query: Query) -> dict[str, str]:
        """Convert a query into CrossRef ``query.*`` request parameters.

        Parameters
        ----------
        query : Query
            Query to convert.  Must have passed :meth:`validate_query`.

        Returns
        -------
        dict[str, str]
            Request parameters, one per CrossRef field query.

        Raises
        ------
        UnexpressibleQueryError
            If the query shape is one CrossRef cannot mean.  Callers that ran
            :meth:`validate_query` first will not see this.
        """
        clauses = self._clauses(query.root)
        return {f"query.{field}": " ".join(values) for field, values in clauses.items()}

    # ------------------------------------------------------------------
    # Clause analysis
    # ------------------------------------------------------------------

    def _clauses(self, node: QueryNode) -> dict[str, list[str]]:
        """Reduce a query subtree to CrossRef's one expressible shape.

        That shape is a conjunction of per-field disjunctions, so the return
        value is a mapping from CrossRef field-query name to the tokens that
        go in it.  Anything that does not reduce to it raises.

        Parameters
        ----------
        node : QueryNode
            Subtree root.

        Returns
        -------
        dict[str, list[str]]
            Field-query name to its token list, in tree order.

        Raises
        ------
        UnexpressibleQueryError
            When the subtree needs an operator CrossRef does not have.
        """
        if node.node_type is NodeType.TERM:
            return {FIELD_MAP[self.get_effective_filter(node)]: [(node.value or "").strip()]}

        operands = [child for child in node.children if child.node_type is not NodeType.CONNECTOR]
        connectors = {
            ConnectorType(child.value)
            for child in node.children
            if child.node_type is NodeType.CONNECTOR and child.value
        }
        if len(operands) == 1 and not connectors:
            return self._clauses(operands[0])
        if ConnectorType.AND_NOT in connectors:
            raise UnexpressibleQueryError(
                "CrossRef has no negation: 'catalysis NOT photocatalysis' and "
                "'catalysis AND photocatalysis' both returned 1079 hits against 938 "
                "for 'catalysis' alone, so a NOT term would be silently ignored and "
                "the search would return a superset of what was asked."
            )
        if len(connectors) > 1:
            raise UnexpressibleQueryError(
                "CrossRef cannot express a group that mixes AND and OR at one level, "
                "because AND is only available between different field queries and OR "
                "is only available inside one. Rewrite the query so that each "
                "parenthesised group uses a single connector."
            )
        merge = self._merge_or if ConnectorType.OR in connectors else self._merge_and
        return merge([self._clauses(operand) for operand in operands])

    @staticmethod
    def _merge_or(parts: list[dict[str, list[str]]]) -> dict[str, list[str]]:
        """Combine disjunctive operands, which must share one field.

        Parameters
        ----------
        parts : list[dict[str, list[str]]]
            Clause maps of the operands.

        Returns
        -------
        dict[str, list[str]]
            One field mapped to the concatenated tokens.

        Raises
        ------
        UnexpressibleQueryError
            When the operands do not all use the same CrossRef field query.
        """
        fields = {field for part in parts for field in part}
        if len(fields) != 1:
            raise UnexpressibleQueryError(
                "CrossRef can only OR terms that share one field query, and this "
                f"group spans {sorted(fields)}. Separate 'query.*' parameters are "
                "combined conjunctively (query.title=catalysis with "
                "query.author=smith gave 10 hits against 938 and 563 separately), so "
                "an OR across fields would be sent as an AND."
            )
        field = fields.pop()
        return {field: [value for part in parts for value in part[field]]}

    @staticmethod
    def _merge_and(parts: list[dict[str, list[str]]]) -> dict[str, list[str]]:
        """Combine conjunctive operands, which must use distinct fields.

        Parameters
        ----------
        parts : list[dict[str, list[str]]]
            Clause maps of the operands.

        Returns
        -------
        dict[str, list[str]]
            Union of the operand maps.

        Raises
        ------
        UnexpressibleQueryError
            When two operands would need the same field query twice.
        """
        merged: dict[str, list[str]] = {}
        for part in parts:
            for field, values in part.items():
                if field in merged:
                    raise UnexpressibleQueryError(
                        f"CrossRef cannot AND two terms in the same field query "
                        f"('{field}'): tokens inside one 'query.*' parameter are "
                        "combined disjunctively, so the conjunction would be sent as "
                        "a disjunction and would return a superset of what was asked."
                    )
                merged[field] = list(values)
        return merged
