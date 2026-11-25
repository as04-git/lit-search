"""Compiling a lit-search query into a predicate evaluated locally.

Some sources cannot be asked a question.  IACR's OAI-PMH endpoint accepts a
date window and a metadata prefix and nothing else; bioRxiv and medRxiv accept
a date window and a server name; a Kaggle arXiv snapshot is a file on disk.
For all three the query has to be answered here, against records the connector
already holds, rather than upstream.

That has one consequence which this module exists to make impossible to
forget, and it is stated here once so every user of the module inherits it:

    **A result assembled by local matching is never ``COMPLETE``.**  What was
    consulted is a slice of the source chosen by something other than the
    query -- a date window, a category, a snapshot -- and the source never
    reported how many records match the query, because it was never asked one.
    :meth:`~lit_search.core.coverage.SourceCoverage.complete` requires a
    source-reported total for exactly this reason, and a local match has none
    to offer.

The machinery was written for IACR and is shared rather than copied: a regular
expression compiler and a boolean folder duplicated across three connectors
would drift, and a term that matches in one place and not another is the kind
of difference nobody notices until a paper is missing.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import ClassVar

from lit_search.core.query import ConnectorType, FilterCode, NodeType, Query, QueryNode
from lit_search.query.builder import QueryBuilder, QueryValidationResult

#: One record's text, keyed by whatever field name the source uses.  Values are
#: sequences because several formats repeat an element (``dc:subject`` appears
#: once per keyword) and joining them would create phrase matches that span two
#: unrelated values.
Record = Mapping[str, Sequence[str]]

#: A compiled test over one such record.
Predicate = Callable[[Record], bool]


def term_pattern(value: str) -> re.Pattern[str]:
    """Compile a lit-search term into a case-insensitive regular expression.

    ``*`` matches any run of characters and ``?`` matches exactly one, matching
    the wildcard semantics of lit-search's own query syntax.  Everything else is
    escaped, so a term is searched for literally.

    The pattern is anchored at word boundaries where the term's own edges are
    word characters, so ``[key]`` does not match "monkey".  A term that begins
    or ends with a wildcard deliberately loses that boundary on the wildcard
    side, which is what the wildcard is for.

    Parameters
    ----------
    value : str
        Stripped term value.  Must not be empty; an empty term matches every
        record, and callers reject it in ``validate_query`` rather than letting
        it silently return the whole slice as a query result.

    Returns
    -------
    re.Pattern[str]
        Compiled pattern.
    """
    body = "".join(
        ".*" if char == "*" else "." if char == "?" else re.escape(char) for char in value
    )
    prefix = r"\b" if value[:1].isalnum() else ""
    suffix = r"\b" if value[-1:].isalnum() else ""
    return re.compile(f"{prefix}{body}{suffix}", re.IGNORECASE | re.DOTALL)


def compile_sequence(
    operands: Sequence[Predicate],
    connectors: Sequence[ConnectorType],
) -> Predicate:
    """Combine sibling predicates under their connectors.

    lit-search's parser produces a flat alternating sequence of operands and
    connectors within one group, and the grammar gives them equal precedence,
    so they are folded left to right.  Explicit grouping in the query string is
    what changes the association, and it arrives here as nested GROUP nodes.

    Parameters
    ----------
    operands : Sequence[Predicate]
        Compiled operand predicates, in order.
    connectors : Sequence[ConnectorType]
        Connectors between them, in order.

    Returns
    -------
    Predicate
        Combined predicate.  A group with no operands matches nothing, which
        keeps an empty group from silently admitting every record.
    """
    if not operands:
        return lambda record: False
    combined = operands[0]
    for index, right in enumerate(operands[1:]):
        connector = connectors[index] if index < len(connectors) else ConnectorType.AND
        combined = _combine(combined, right, connector)
    return combined


def _combine(left: Predicate, right: Predicate, connector: ConnectorType) -> Predicate:
    """Join two predicates under one boolean connector.

    Parameters
    ----------
    left : Predicate
        Left operand.
    right : Predicate
        Right operand.
    connector : ConnectorType
        Connector joining them.

    Returns
    -------
    Predicate
        Combined predicate.
    """
    if connector is ConnectorType.OR:
        return lambda record: left(record) or right(record)
    if connector is ConnectorType.AND_NOT:
        return lambda record: left(record) and not right(record)
    return lambda record: left(record) and right(record)


class LocalPredicateQueryBuilder(QueryBuilder):
    """Base for builders that match locally instead of querying upstream.

    Subclasses declare :attr:`FIELD_MAP`, mapping each filter code they can
    answer to the record field names that answer it, plus :attr:`SOURCE_LABEL`
    and :attr:`UNSUPPORTED_FIELDS_NOTE` for the error messages.

    Because matching happens here rather than upstream, the full boolean
    grammar is supported exactly -- including ``AND NOT``, which several remote
    APIs in this project cannot express faithfully.  What is *not* supported is
    a filter code the record format carries no field for, and that is rejected
    in :meth:`validate_query` rather than quietly ignored: a filter dropped on
    the floor widens the search silently, and a wider search that still returns
    nothing looks exactly like a narrow one that found nothing.
    """

    #: Filter code to the record field names that answer it.  Override.
    FIELD_MAP: ClassVar[dict[FilterCode, tuple[str, ...]]] = {}

    #: Human-readable source name used in validation messages.  Override.
    SOURCE_LABEL: str = "this source"

    #: Sentence explaining what the record format does not carry, appended to
    #: the message for an unsupported filter.  Override.
    UNSUPPORTED_FIELDS_NOTE: str = "its records carry no field that could answer it."

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Keep ``_SUPPORTED_FILTERS`` in step with ``FIELD_MAP``.

        The base :class:`~lit_search.query.builder.QueryBuilder` answers
        :meth:`~lit_search.query.builder.QueryBuilder.supports_filter` from
        ``_SUPPORTED_FILTERS``, and
        :meth:`~lit_search.query.builder.QueryBuilder.get_effective_filter`
        uses that to pick the default filter for an unqualified term.  A
        subclass that declared ``FIELD_MAP`` and forgot ``_SUPPORTED_FILTERS``
        would therefore default every bare term to ``tiabs`` and reject every
        explicit filter -- so the two are derived from one declaration instead
        of being kept in step by hand.

        Parameters
        ----------
        **kwargs : object
            Forwarded to ``super().__init_subclass__``.
        """
        super().__init_subclass__(**kwargs)
        cls._SUPPORTED_FILTERS = frozenset(cls.FIELD_MAP)

    def validate_query(self, query: Query) -> QueryValidationResult:
        """Validate whether the local predicate can answer this query.

        Parameters
        ----------
        query : Query
            Query to validate.

        Returns
        -------
        QueryValidationResult
            Invalid when a term is empty or names a field the record format
            does not carry.
        """
        for term in self.iter_term_nodes(query.root):
            value = (term.value or "").strip()
            if not value:
                return QueryValidationResult(
                    is_valid=False,
                    error_message=(
                        "An empty term matches every record consulted, which would "
                        "report the whole slice as a query result."
                    ),
                )
            filter_code = self.get_effective_filter(term)
            if not self.supports_filter(filter_code):
                return QueryValidationResult(
                    is_valid=False,
                    error_message=(
                        f"Filter '{filter_code}' cannot be answered from {self.SOURCE_LABEL} "
                        f"records: {self.UNSUPPORTED_FIELDS_NOTE}"
                    ),
                )
        return QueryValidationResult(is_valid=True)

    def convert_query(self, query: Query) -> str:
        """Render the query as the field-scoped expression matched locally.

        The source is never sent this string.  It exists so that provenance,
        logs and the coverage detail record what the predicate actually tested,
        which is otherwise invisible: a local match leaves no request to
        inspect.

        Parameters
        ----------
        query : Query
            Query to render.  Must have passed :meth:`validate_query`.

        Returns
        -------
        str
            Human-readable rendering of the local match expression.
        """
        connector_map = {
            ConnectorType.AND: "AND",
            ConnectorType.OR: "OR",
            ConnectorType.AND_NOT: "AND NOT",
        }

        def convert_term(term_node: QueryNode) -> str:
            """Render one term with the fields it is matched against.

            Parameters
            ----------
            term_node : QueryNode
                Term node to render.

            Returns
            -------
            str
                Rendering of the form ``title+abstract:"value"``.
            """
            value = (term_node.value or "").strip()
            fields = "+".join(self.FIELD_MAP[self.get_effective_filter(term_node)])
            return f'{fields}:"{value}"'

        return self.convert_expression(query.root, convert_term, connector_map)

    def compile_predicate(self, query: Query) -> Predicate:
        """Compile the query into a predicate over one record.

        Parameters
        ----------
        query : Query
            Query to compile.  Must have passed :meth:`validate_query`.

        Returns
        -------
        Predicate
            Callable returning ``True`` when a record matches.
        """
        return self._compile_node(query.root)

    def _compile_node(self, node: QueryNode) -> Predicate:
        """Compile one query node into a predicate.

        Parameters
        ----------
        node : QueryNode
            Node to compile.

        Returns
        -------
        Predicate
            Callable returning ``True`` when a record matches this node.
        """
        if node.node_type == NodeType.TERM:
            return self._compile_term(node)
        return compile_sequence(
            [
                self._compile_node(child)
                for child in node.children
                if child.node_type != NodeType.CONNECTOR
            ],
            [
                ConnectorType(child.value)
                for child in node.children
                if child.node_type == NodeType.CONNECTOR and child.value
            ],
        )

    def _compile_term(self, node: QueryNode) -> Predicate:
        """Compile one term node into a predicate.

        Parameters
        ----------
        node : QueryNode
            Term node.

        Returns
        -------
        Predicate
            Callable returning ``True`` when a record matches this term.
        """
        fields = self.FIELD_MAP[self.get_effective_filter(node)]
        pattern = term_pattern((node.value or "").strip())

        def matches(record: Record) -> bool:
            """Test one record against this term.

            Parameters
            ----------
            record : Record
                Record text keyed by field name.

            Returns
            -------
            bool
                ``True`` when any of this term's fields matches.
            """
            return any(
                pattern.search(text) for name in fields for text in record.get(name, ()) if text
            )

        return matches
