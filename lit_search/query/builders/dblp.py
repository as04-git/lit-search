"""dblp query builder.

Every syntax claim below was verified against the live search API on
2026-08-31 (``https://dblp.org/search/publ/api``) by reading the ``result.query``
field, which echoes the query **as dblp actually parsed it**.  That echo is the
single most useful diagnostic this API offers, and it is what exposed the three
silent rewrites this builder exists to refuse.

**dblp rewrites queries it cannot express, and returns HTTP 200 either way.**

=========================  ==========================================  ==========
Sent                       Echoed back as ``result.query``             ``@total``
===============================  ======================================  ==========
Sent                             Echoed back as ``result.query``         ``@total``
===============================  ======================================  ==========
``blockchain``                   ``blockchain*``                         32,115
``blockchain consensus``         ``blockchain* consensus*``              1,173
``blockchain -consensus``        ``blockchain* consensus*``              **1,173**
``"neural network"``             ``neural* network*``                    201,098
``blockchain|consensus``         ``blockchain*|consensus*``              50,643
``blockchain | consensus``       ``blockchain*|consensus*``              50,643
``notafield:blockchain``         ``notafield* blockchain*``              **0**
``title:blockchain``             ``:filter:title:blockchain*``           31,519
``author:Goldwasser``            ``:filter:author:Goldwasser*``          515
``venue:CRYPTO``                 ``:filter:venue:CRYPTO*``               44,964
``(blockchain hash)|quantum``    ``blockchain* hash*|quantum*``          **660**
``blockchain hash``              ``blockchain* hash*``                   213
``quantum``                      ``quantum*``                            55,489
===============================  ======================================  ==========

Four of those rows are the reason this builder validates instead of passing
text through:

1. **A leading ``-`` is discarded, and ``NOT`` silently becomes ``AND``.**
   ``blockchain -consensus`` and ``blockchain consensus`` produce a byte-identical
   echo and an identical total of 1,173.  A query meaning "blockchain, excluding
   consensus" therefore returns precisely the papers it asked to exclude -- the
   complement of the intent, with no error and a plausible-looking result.
   :class:`DBLPQueryBuilder` refuses ``AND NOT`` rather than emit it.

2. **An unrecognised field prefix is split into a bare token.**
   ``notafield:blockchain`` becomes the two-term conjunction
   ``notafield* blockchain*``, which matches nothing and returns HTTP 200 with
   ``@total: 0``.  A builder that emitted a field dblp does not index would
   report "dblp holds no matching paper" for every query.  Only the four
   prefixes in :data:`FIELD_MAP` -- each confirmed by its ``:filter:`` echo --
   are ever emitted.

3. **Quotes are stripped, so a phrase becomes a conjunction of prefixes.**
   ``"neural network"`` is parsed as ``neural* network*``.  This is a
   *widening*: every paper containing the phrase is still returned, alongside
   papers containing both words apart.  Unlike (1) and (2) it costs precision
   rather than recall, and the extra papers are visible to whoever reads the
   results, so it is accepted and documented rather than refused.

4. **Parentheses are discarded, and ``|`` binds TIGHTER than the implicit
   whitespace ``AND``.**  Verified 2026-08-31:
   ``(blockchain hash)|quantum`` echoes as ``blockchain* hash*|quantum*`` and
   returns **660**, while ``blockchain hash`` alone returns 213 and ``quantum``
   alone returns 55,489.  The intended ``(blockchain AND hash) OR quantum``
   cannot return fewer than 55,489, so dblp answered
   ``blockchain AND (hash OR quantum)`` -- a different question, roughly a
   hundredth the size, at HTTP 200 with no diagnostic.  Worse, a ``)`` that
   follows a field-scoped value is *absorbed into the value*:
   ``(title:blockchain title:hash) | title:quantum`` echoes as
   ``:filter:title:blockchain* :filter:title:hash)*|:filter:title:quantum*``,
   so the second clause searches titles beginning ``hash)``.

   dblp's expressible grammar is therefore exactly **conjunctive normal form**:
   a whitespace-joined conjunction of ``|``-joined disjunctions of terms, with
   no grouping of any kind.  :class:`DBLPQueryBuilder` emits **no parentheses**
   and refuses any query whose shape needs them; see :func:`_clause_level`.

Two further behaviours shape the mapping:

- **Every term is implicitly prefix-expanded.**  ``blockchain`` is parsed as
  ``blockchain*``; a trailing ``$`` suppresses that (``blockchain$`` -> 30,271
  hits against 32,115).  Implicit expansion widens recall, so it is left alone.
- **dblp indexes bibliographic metadata only -- there are no abstracts in the
  corpus at all.**  Filters that name the abstract or keywords cannot be
  answered, and that is a coverage fact rather than a syntax problem; see
  :meth:`DBLPQueryBuilder.degradation_note`.
"""

from __future__ import annotations

from lit_search.core.query import ConnectorType, FilterCode, NodeType, Query, QueryNode
from lit_search.query.builder import QueryBuilder, QueryValidationResult

# Mapping from lit_search filter codes to dblp search-field prefixes.
#
# Every prefix here was confirmed live on 2026-08-31 by the ``:filter:<name>:``
# form in the echoed ``result.query``.  A prefix dblp does not recognise is
# silently demoted to a search token and yields zero hits, so this map must
# never grow by assumption -- check the echo first.
FIELD_MAP: dict[FilterCode, str] = {
    FilterCode.TITLE: "title",
    FilterCode.AUTHOR: "author",
    FilterCode.SOURCE: "venue",
}

# Filter codes dblp can answer, exactly, with a verified field prefix.
_NATIVE_FILTERS = frozenset(FIELD_MAP)

# Filter codes whose scope includes the title but also names fields dblp does
# not hold.  These are answered against dblp's metadata index, which is a
# *subset* of what was asked, so they are accepted and the shortfall is
# reported as coverage rather than hidden.  See ``degradation_note``.
_DEGRADED_FILTERS = frozenset(
    {
        FilterCode.TITLE_ABSTRACT,
        FilterCode.TITLE_ABSTRACT_KEYWORDS,
    }
)

# Characters dblp gives its own meaning inside a term: ``|`` is OR, ``(``/``)``
# group, ``:`` introduces a field prefix, and quotes are stripped.  A term
# carrying one of these is reparsed as structure rather than searched for.
#
# ``?`` is listed for a different reason.  lit-search's query syntax admits it
# as a single-character wildcard, and dblp's handling of it was **not
# verified** on 2026-08-31 -- only ``*`` (implicit prefix expansion) and ``$``
# (suppressing it) were.  An unrecognised operator on this API does not error:
# it becomes an ordinary search token and returns HTTP 200 with zero hits.
# Refusing a construct whose parse is unknown is the honest direction; emitting
# it and hoping is how a source comes to report a silent zero.
#
# ``$`` is listed for the same reason in the opposite direction: dblp's FAQ
# documents a trailing ``$`` as suppressing the implicit prefix expansion, so a
# term carrying one would silently *narrow* recall -- the direction that hides.
# An attempt to confirm the effect live on 2026-08-31 drew an HTTP 503 throttle
# rather than a parse echo, so the behaviour is **unconfirmed here**, which is
# itself the reason to refuse it: exactly the rule already applied to ``?``.
_STRUCTURAL_CHARS = ("|", "(", ")", ":", '"', "?", "$")

# Levels of the only grammar dblp can express, verified 2026-08-31: a
# conjunction of disjunctions of terms, with no grouping.
_LEVEL_TERM = 0
_LEVEL_DISJUNCTION = 1
_LEVEL_CONJUNCTION = 2


def _clause_level(node: QueryNode) -> int | None:
    """Return the CNF level of *node*, or ``None`` when dblp cannot express it.

    dblp's grammar is a whitespace-joined conjunction of ``|``-joined
    disjunctions of terms.  It has no grouping at all: parentheses are
    discarded from the query string, and ``|`` binds tighter than the implicit
    whitespace ``AND``.  So a node is expressible only when it sits at one of
    three levels -- a term, a disjunction of terms, or a conjunction of those
    -- and any shape that would need a parenthesis to survive is not.

    Two shapes are rejected, both because dblp answers them with a *different*
    query at HTTP 200 rather than with an error:

    - an ``AND`` nested inside an ``OR``, which is what the discarded
      parentheses were carrying;
    - ``AND`` and ``OR`` mixed at one level, where lit-search folds left to
      right (``[a] AND [b] OR [c]`` means ``(a AND b) OR c``) while dblp binds
      ``|`` tighter and reads ``a AND (b OR c)``.

    Parameters
    ----------
    node : QueryNode
        Node to classify.

    Returns
    -------
    int | None
        :data:`_LEVEL_TERM`, :data:`_LEVEL_DISJUNCTION` or
        :data:`_LEVEL_CONJUNCTION`, or ``None`` when the node is not
        expressible.
    """
    if node.node_type == NodeType.TERM:
        return _LEVEL_TERM

    levels: list[int] = []
    for child in node.children:
        if child.node_type == NodeType.CONNECTOR:
            continue
        level = _clause_level(child)
        if level is None:
            return None
        levels.append(level)
    if not levels:
        return None
    if len(levels) == 1:
        # A group wrapping a single operand adds no structure, so it neither
        # needs a parenthesis nor changes the level.
        return levels[0]

    connectors = {
        ConnectorType(child.value)
        for child in node.children
        if child.node_type == NodeType.CONNECTOR and child.value
    }
    if len(connectors) != 1:
        return None
    connector = connectors.pop()
    if connector is ConnectorType.OR:
        return _LEVEL_DISJUNCTION if max(levels) <= _LEVEL_DISJUNCTION else None
    if connector is ConnectorType.AND:
        return _LEVEL_CONJUNCTION if max(levels) <= _LEVEL_CONJUNCTION else None
    # AND NOT, refused earlier by validate_query for its own reason.
    return None


class DBLPQueryBuilder(QueryBuilder):
    """Build dblp search expressions.

    The builder emits only the operators and field prefixes whose parse was
    confirmed against the live echo, and rejects the constructs dblp accepts
    and quietly reinterprets.
    """

    _SUPPORTED_FILTERS = frozenset(_NATIVE_FILTERS | _DEGRADED_FILTERS)

    def validate_query(self, query: Query) -> QueryValidationResult:
        """Validate whether dblp can express this query without rewriting it.

        Parameters
        ----------
        query : Query
            Query to validate.

        Returns
        -------
        QueryValidationResult
            Invalid when the query uses ``AND NOT`` (which dblp silently turns
            into ``AND``), names a field dblp does not index, contains an empty
            term, or contains a term carrying dblp's own operator characters.
        """
        if ConnectorType.AND_NOT in self.iter_connectors(query.root):
            return QueryValidationResult(
                is_valid=False,
                error_message=(
                    "dblp cannot express 'AND NOT'. Verified live 2026-08-31: it "
                    "discards a leading '-', so 'blockchain -consensus' is parsed as "
                    "'blockchain* consensus*' and returns the same 1,173 papers as the "
                    "conjunction -- exactly the records the query asked to exclude. "
                    "Sending it would return a confident, well-formed, inverted result."
                ),
            )

        for term in self.iter_term_nodes(query.root):
            failure = self._validate_term(term)
            if failure is not None:
                return failure

        if _clause_level(query.root) is None:
            return QueryValidationResult(
                is_valid=False,
                error_message=(
                    "dblp cannot express this query's boolean structure. Verified live "
                    "2026-08-31: it discards parentheses and binds '|' tighter than the "
                    "implicit whitespace 'AND', so '(blockchain hash)|quantum' is parsed "
                    "as 'blockchain AND (hash OR quantum)' and returns 660 papers where "
                    "the query asked for at least the 55,489 matching 'quantum' alone -- "
                    "a different question, answered at HTTP 200 with no diagnostic. Only "
                    "a conjunction of disjunctions of terms survives: '[a] OR [b]', "
                    "'[a] AND [b]', '([a] OR [b]) AND [c]'. An 'AND' inside an 'OR', or "
                    "'AND' and 'OR' mixed without grouping, must be split into separate "
                    "searches and combined as sets."
                ),
            )
        return QueryValidationResult(is_valid=True)

    def _validate_term(self, term: QueryNode) -> QueryValidationResult | None:
        """Validate one term node.

        Parameters
        ----------
        term : QueryNode
            Term node to check.

        Returns
        -------
        QueryValidationResult | None
            A failing result, or ``None`` when the term is acceptable.
        """
        value = (term.value or "").strip()
        if not value:
            return QueryValidationResult(
                is_valid=False,
                error_message=(
                    "dblp answers an empty query with the whole database "
                    "(HTTP 200, @total 8,733,592, @sent 0), so an empty term "
                    "cannot be run."
                ),
            )
        found = [char for char in _STRUCTURAL_CHARS if char in value]
        if found:
            return QueryValidationResult(
                is_valid=False,
                error_message=(
                    f"Term '{value}' contains {found!r}, which dblp reads as query "
                    "structure rather than as text: ':' introduces a field prefix, "
                    "'|' is OR, '(' groups, and quotes are stripped. The term would "
                    "be reparsed into a different query."
                ),
            )
        filter_code = self.get_effective_filter(term)
        if not self.supports_filter(filter_code):
            return QueryValidationResult(
                is_valid=False,
                error_message=(
                    f"Filter '{filter_code}' is not supported by dblp. dblp indexes "
                    "bibliographic metadata only -- titles, authors, venues and years "
                    "-- and holds no abstracts, keywords or affiliations. Answering "
                    "this filter from the title index would search a different field "
                    "than the query names and return a different set of papers."
                ),
            )
        return None

    def degradation_note(self, query: Query) -> str | None:
        """Describe how far short of the query dblp's index falls.

        dblp holds no abstracts.  A query scoped to title-and-abstract is
        therefore answered against titles alone, which retrieves a *subset* of
        what was asked: a paper whose abstract matches but whose title does not
        is absent, and nothing in the result says so.  That is a coverage loss,
        so the connector turns this note into a ``PARTIAL`` status rather than
        letting the run claim ``COMPLETE``.

        A query scoped only to fields dblp does hold -- title, author, venue --
        has nothing missing and returns ``None`` here.

        Parameters
        ----------
        query : Query
            Validated query.

        Returns
        -------
        str | None
            Explanation of the shortfall, or ``None`` when there is none.
        """
        degraded = sorted(
            {
                str(self.get_effective_filter(term))
                for term in self.iter_term_nodes(query.root)
                if self.get_effective_filter(term) in _DEGRADED_FILTERS
            }
        )
        if not degraded:
            return None
        return (
            f"dblp holds no abstracts, so the filter(s) {', '.join(degraded)} were "
            "answered against its bibliographic index alone; papers matching only in "
            "an abstract are absent from this result and dblp cannot report how many"
        )

    def convert_query(self, query: Query) -> str:
        """Convert a query into dblp search syntax.

        The shared :meth:`~lit_search.query.builder.QueryBuilder.convert_expression`
        helper is deliberately **not** used here: it wraps every GROUP node in
        parentheses, and dblp discards parentheses -- absorbing a stray ``)``
        into the preceding field value when it follows one.  Emitting a
        character the target silently deletes is how the query that gets
        answered stops being the query that was asked.  Since
        :meth:`validate_query` has already guaranteed the tree is a conjunction
        of disjunctions, rendering it without grouping loses nothing.

        Parameters
        ----------
        query : Query
            Query to convert.  Must have passed :meth:`validate_query`.

        Returns
        -------
        str
            dblp query expression, containing no parentheses.
        """
        return " ".join(self._render(query.root).split())

    def _render(self, node: QueryNode) -> str:
        """Render one validated node as flat dblp syntax.

        Parameters
        ----------
        node : QueryNode
            Node to render.  Its connectors are uniform, because
            :func:`_clause_level` rejected anything else.

        Returns
        -------
        str
            Rendered expression with no grouping.
        """
        if node.node_type == NodeType.TERM:
            value = (node.value or "").strip()
            prefix = FIELD_MAP.get(self.get_effective_filter(node))
            return f"{prefix}:{value}" if prefix else value

        operands = [child for child in node.children if child.node_type != NodeType.CONNECTOR]
        is_or = any(
            child.node_type == NodeType.CONNECTOR and ConnectorType(child.value) is ConnectorType.OR
            for child in node.children
            if child.value
        )
        # ``a | b`` and ``a|b`` were both confirmed to parse as a disjunction on
        # 2026-08-31, for bare and for field-scoped terms alike; the spaced form
        # is used because it is the one verified against field prefixes.
        joiner = " | " if is_or else " "
        return joiner.join(self._render(operand) for operand in operands)
