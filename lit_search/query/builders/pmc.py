"""PubMed Central (PMC) query builder.

PMC is reached through the same NCBI E-utilities endpoints as PubMed, which
makes it tempting to reuse :class:`~lit_search.query.builders.pubmed.PubmedQueryBuilder`
outright.  **Do not.**  The two Entrez databases index different fields, and the
tags that differ fail silently.

Every claim below was checked against the live ``einfo`` field list for
``db=pmc`` and against live ``esearch`` counts on 2026-08-31.

**The trap.**  Entrez does not reject an unknown field tag.  It drops the tag,
re-runs the term against all fields, returns HTTP 200 with a plausible result
count, and mentions what it did only inside ``<ErrorList><FieldNotFound>``:

======================  ===========  =========================================
Term                    ``Count``    What happened
======================  ===========  =========================================
``cancer[ti]``            605,637    title search, as asked
``cancer[tiab]``        1,088,018    title/abstract search, as asked
``cancer[ALL]``         4,013,901    every indexed field, including body text
``cancer[ab]``          4,013,901    **tag dropped** -- ``FieldNotFound: ab``
``cancer[ot]``          4,013,901    **tag dropped** -- ``FieldNotFound: ot``
======================  ===========  =========================================

``[ab]`` and ``[ot]`` are the tags PubMed uses for abstract and author
keywords, and both are absent from PMC's field list.  A query scoped to
abstracts therefore comes back as a four-million-record full-text search that
looks exactly like a successful search.  The connector detects
``FieldNotFound`` at runtime as a backstop; this builder's job is never to emit
one.

**What PMC does index**, from ``einfo``: ``TITL`` (``ti``), ``TIAB``
(``tiab``), ``FULL`` (``au``), ``JOUR`` (``ta``), ``AFFL`` (``ad``), ``MESH``,
``ARTI`` (article body).  Every tag emitted here was confirmed to return a
count with no ``ErrorList``.

**What it does not index, and what that costs:**

- **No standalone abstract field.**  ``FilterCode.ABSTRACT`` is rejected rather
  than approximated by ``tiab`` (which would silently widen the scope to
  titles) or by ``ALL`` (which would silently widen it to full text).
- **No author-keyword field.**  PMC has ``MESH``, but MeSH headings are an
  NLM-assigned controlled vocabulary, not the keywords an author chose --
  precisely the vocabulary mismatch this project exists to make visible.  So
  ``FilterCode.KEYWORDS`` and ``FilterCode.TITLE_ABSTRACT_KEYWORDS`` are
  rejected too.

Rejecting ``tiabskey`` also changes the *default* scope: with it unsupported,
:meth:`~lit_search.query.builder.QueryBuilder.get_effective_filter` falls back
to ``tiabs``, so an unfiltered query becomes a ``[tiab]`` search.  That is the
intended outcome, not an accident of the filter list.
"""

from __future__ import annotations

from lit_search.core.query import ConnectorType, FilterCode, Query, QueryNode
from lit_search.query.builder import QueryBuilder, QueryValidationResult

# Mapping from lit_search filter codes to Entrez field tags valid on ``db=pmc``.
# Every tag was confirmed present in the live ``einfo`` field list and confirmed
# to produce a count with no ``<ErrorList>`` on 2026-08-31.
FIELD_MAP: dict[FilterCode, str] = {
    FilterCode.TITLE: "ti",
    FilterCode.AUTHOR: "au",
    FilterCode.SOURCE: "ta",
    FilterCode.AFFILIATION: "ad",
    FilterCode.TITLE_ABSTRACT: "tiab",
}

# Filter codes PMC has no faithful field for, with the reason stated in the
# terms a user can act on.  Approximating any of these would answer a different
# question with no sign that it had.
UNSUPPORTED_FILTERS: dict[FilterCode, str] = {
    FilterCode.ABSTRACT: (
        "PMC indexes no standalone abstract field (its Entrez field list has TIAB but "
        "no AB), and the '[ab]' tag PubMed uses is silently dropped there: 'cancer[ab]' "
        "returns 4,013,901 records, identical to 'cancer[ALL]', with the tag reported "
        "only inside <ErrorList><FieldNotFound>. Use the 'tiabs' filter to search title "
        "and abstract together, or search PubMed or Europe PMC for an abstract-only scope."
    ),
    FilterCode.KEYWORDS: (
        "PMC indexes no author-keyword field ('cancer[ot]' is silently dropped and "
        "widened to every field). Its MESH field holds NLM-assigned Medical Subject "
        "Headings, which are a controlled vocabulary chosen by indexers rather than the "
        "keywords an author supplied, so mapping 'key' to it would answer a different "
        "question. Search PubMed or Europe PMC for author keywords."
    ),
    FilterCode.TITLE_ABSTRACT_KEYWORDS: (
        "PMC indexes no author-keyword field, so the title/abstract/keywords scope "
        "cannot be expressed. Use 'tiabs' for title and abstract."
    ),
}

# Minimum characters before a '*' truncation.  PubMed documents four; measured
# on db=pmc, 'ne*[ti]' returned 579 records against 369,990 for 'neur*[ti]',
# which is not a prefix expansion of 'ne' by any reading -- the short form is
# quietly not doing what it says.
_MIN_WILDCARD_PREFIX = 4


class PMCQueryBuilder(QueryBuilder):
    """Build PMC-compatible Entrez query expressions."""

    _SUPPORTED_FILTERS = frozenset(FIELD_MAP)

    def validate_query(self, query: Query) -> QueryValidationResult:
        """Validate whether PMC can express this query faithfully.

        Parameters
        ----------
        query : Query
            Query to validate.

        Returns
        -------
        QueryValidationResult
            Invalid when a term is empty, uses an unsupported wildcard form, or
            asks for a scope PMC does not index.  Each of those runs upstream
            and returns a plausible count for a different question.
        """
        for term in self.iter_term_nodes(query.root):
            value = (term.value or "").strip()
            if not value:
                return QueryValidationResult(
                    is_valid=False,
                    error_message=(
                        "NCBI E-utilities reject an empty term with "
                        "'<ERROR>Empty term and query_key - nothing todo</ERROR>' at HTTP 200, "
                        "so an empty term cannot be run."
                    ),
                )
            wildcard_error = _wildcard_error(value)
            if wildcard_error is not None:
                return QueryValidationResult(is_valid=False, error_message=wildcard_error)
            filter_code = self.get_effective_filter(term)
            reason = UNSUPPORTED_FILTERS.get(filter_code)
            if reason is not None:
                return QueryValidationResult(
                    is_valid=False,
                    error_message=f"Filter '{filter_code}' is not supported by PMC. {reason}",
                )
            if not self.supports_filter(filter_code):
                return QueryValidationResult(
                    is_valid=False,
                    error_message=f"Filter '{filter_code}' is not supported by PMC.",
                )
        return QueryValidationResult(is_valid=True)

    def convert_query(self, query: Query) -> str:
        """Convert a query into PMC Entrez syntax.

        Parameters
        ----------
        query : Query
            Query to convert.  Must have passed :meth:`validate_query`.

        Returns
        -------
        str
            Entrez term expression for ``db=pmc``.
        """
        connector_map = {
            ConnectorType.AND: "AND",
            ConnectorType.OR: "OR",
            ConnectorType.AND_NOT: "NOT",
        }

        def convert_term(term_node: QueryNode) -> str:
            value = (term_node.value or "").strip()
            filter_code = self.get_effective_filter(term_node)
            tag = FIELD_MAP.get(filter_code, "tiab")
            return f"{_render_value(value)}[{tag}]"

        return self.convert_expression(query.root, convert_term, connector_map)


def _wildcard_error(value: str) -> str | None:
    """Return why *value*'s wildcard form is unsafe on PMC, or ``None``.

    Parameters
    ----------
    value : str
        Stripped term value.

    Returns
    -------
    str | None
        Explanation when the term cannot be sent, otherwise ``None``.
    """
    if "?" in value:
        return (
            f"Term '{value}' uses '?', which Entrez does not treat as a wildcard. "
            "The query runs and returns a count for a different term "
            "('neu?al[ti]' returned 99 records on db=pmc), so it is refused here."
        )
    index = value.find("*")
    if index >= 0 and index < _MIN_WILDCARD_PREFIX:
        return (
            f"Term '{value}' truncates after {index} character(s); Entrez requires at least "
            f"{_MIN_WILDCARD_PREFIX} before '*'. A shorter prefix does not error, it silently "
            "stops expanding: 'ne*[ti]' returned 579 records against 369,990 for 'neur*[ti]'."
        )
    return None


def _render_value(value: str) -> str:
    """Render a term value for Entrez, quoting where that is safe.

    A term is quoted so that a multi-word value searches as a phrase, which is
    what the bracketed term syntax means.  A term carrying a ``*`` is left
    unquoted because Entrez does not expand a truncation inside a quoted phrase.

    Parameters
    ----------
    value : str
        Stripped term value.

    Returns
    -------
    str
        Quoted or bare term, ready for a ``[tag]`` suffix.
    """
    if "*" in value:
        return value
    return '"' + value.replace('"', "") + '"'
