"""CORE query builder.

CORE's ``/v3/search/works`` endpoint takes one ``q`` string supporting
``field:value`` lookups.  Its searchable-field list was read from CORE's own
API documentation at ``https://api.core.ac.uk/docs/v3`` on 2026-08-31, where
each entry states explicitly whether it is *"Searchable using field lookup via
GET"* and carries a worked example URL.  For the ``works`` entity those are:
``abstract``, ``authors``, ``contributors``, ``doi``, ``fullText``, ``id``,
``publisher``, ``pubmedId``, ``title``, ``updatedDate`` and ``yearPublished``.

**RETRACTED: "None of this builder's behaviour could be confirmed against live
responses, because CORE refuses unauthenticated search."**  That said a keyless
request returns HTTP 429 with a zero-byte body.  Re-checked 2026-08-31 during
adversarial review: a keyless ``GET /v3/search/works/?q=...`` returns HTTP 200
with results.  The earlier 429 was an emptied bucket, not a refusal.  (A request
carrying an *invalid* key does return HTTP 401
``{"message": "The API key you provided is not valid."}`` -- that part stands.)

The mapped field names have now been checked against live responses, and the
important result is that **CORE is loud about a field it does not index**:

- ``abstract:catalysis`` -> HTTP 200, 95,796 records.  ``authors:Zhang`` ->
  HTTP 200, 2,186,825.  Both mapped names are real.
- ``NOTAFIELD:catalysis`` -> **HTTP 500**, *"Illegal arguments in query request:
  NOTAFIELD is not a searchable field."*  This is the opposite of the Europe PMC
  and Zenodo trap: a wrong field name here is an error the connector reports as
  ``FAILED``, not a silent zero.
- ``keywords:catalysis`` -> **HTTP 500**, *"keywords is not a searchable
  field"*, which confirms the refusal of
  :data:`~lit_search.core.query.FilterCode.KEYWORDS` rather than mapping it
  onto something adjacent.

Two conservative choices remain, and both are still unverified rather than
verified-safe:

- Wildcards are refused.  Whether CORE expands them was not tested, and the two
  sibling sources in this port disagree with each other -- Zenodo expands
  (``title:catalys*`` 4,638 against ``title:catalysis`` 772) while OpenAIRE
  silently does not (``mainTitle=catalys*`` 103 against 86,158).  A wildcard
  that does not expand returns a small, plausible, wrong answer, so an
  unverified wildcard is not sent.
- An empty term is refused.  Both sibling sources treat an empty query as a
  match-all over the entire corpus, and there is no reason to assume CORE
  differs.
"""

from __future__ import annotations

from lit_search.core.query import ConnectorType, FilterCode, Query, QueryNode
from lit_search.query.builder import QueryBuilder, QueryValidationResult

# Mapping from lit_search filter codes to CORE searchable fields for the
# ``works`` entity.  Each is documented as searchable via GET field lookup.
FIELD_MAP: dict[FilterCode, str] = {
    FilterCode.TITLE: "title",
    FilterCode.ABSTRACT: "abstract",
    FilterCode.AUTHOR: "authors",
}

# CORE has no combined title/abstract field, so the composite filter code is
# written out as an explicit disjunction.
_COMPOSITE_FIELDS: dict[FilterCode, tuple[str, ...]] = {
    FilterCode.TITLE_ABSTRACT: ("title", "abstract"),
}

_WILDCARD_CHARS = ("*", "?")


class CoreQueryBuilder(QueryBuilder):
    """Build CORE ``q`` expressions.

    :data:`~lit_search.core.query.FilterCode.TITLE_ABSTRACT_KEYWORDS` is not
    supported, so an unfiltered query falls back to
    :data:`~lit_search.core.query.FilterCode.TITLE_ABSTRACT` and searches the
    two fields it names.  CORE's ``works`` documentation lists no searchable
    keyword field, and claiming to search keywords while not doing so is the
    silent-shortfall shape this project exists to prevent.
    """

    _SUPPORTED_FILTERS = frozenset(
        {
            FilterCode.TITLE,
            FilterCode.ABSTRACT,
            FilterCode.AUTHOR,
            FilterCode.TITLE_ABSTRACT,
        }
    )

    def validate_query(self, query: Query) -> QueryValidationResult:
        """Validate whether CORE can express this query faithfully.

        Parameters
        ----------
        query : Query
            Query to validate.

        Returns
        -------
        QueryValidationResult
            Invalid when a term is empty, carries a wildcard, or uses a filter
            with no documented searchable CORE field.
        """
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
                        f"Filter '{filter_code}' has no documented searchable CORE field "
                        "for the 'works' entity. CORE documents title, abstract, authors, "
                        "contributors, doi, fullText, id, publisher, pubmedId, updatedDate "
                        "and yearPublished as searchable; it documents no keyword or "
                        "affiliation field."
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
            An invalid result when the term cannot be sent safely.
        """
        if not value:
            return QueryValidationResult(
                is_valid=False,
                error_message=(
                    "An empty term cannot be sent to CORE. Both sibling sources in this "
                    "port treat an empty query as a match-all over their entire corpus "
                    "(Zenodo 7,218,253 records, OpenAIRE 386,695,835, both verified "
                    "2026-08-31), and CORE's behaviour here is untested."
                ),
            )
        if any(char in value for char in _WILDCARD_CHARS):
            return QueryValidationResult(
                is_valid=False,
                error_message=(
                    f"Term '{value}' contains a wildcard. CORE does not document whether "
                    "it expands wildcards and the behaviour has not been tested. A "
                    "wildcard that is not expanded returns a small plausible result set "
                    "rather than an error -- OpenAIRE's 'mainTitle=catalys*' returned 103 "
                    "against 86,158 for the bare term -- so it is not sent."
                ),
            )
        return None

    def convert_query(self, query: Query) -> str:
        """Convert a query into a CORE ``q`` expression.

        Parameters
        ----------
        query : Query
            Query to convert.  Must have passed :meth:`validate_query`.

        Returns
        -------
        str
            CORE query expression.
        """
        connector_map = {
            ConnectorType.AND: "AND",
            ConnectorType.OR: "OR",
            ConnectorType.AND_NOT: "NOT",
        }

        def convert_term(term_node: QueryNode) -> str:
            rendered = _render_value((term_node.value or "").strip())
            filter_code = self.get_effective_filter(term_node)
            composite = _COMPOSITE_FIELDS.get(filter_code)
            if composite is not None:
                joined = " OR ".join(f"{field}:{rendered}" for field in composite)
                return f"({joined})"
            return f"{FIELD_MAP[filter_code]}:{rendered}"

        return self.convert_expression(query.root, convert_term, connector_map)


def _render_value(value: str) -> str:
    """Render a term value for CORE as a quoted phrase.

    Every term is quoted, so that a multi-word value searches as a phrase
    rather than as separate tokens -- which is what lit-search's bracketed term
    syntax means.  Wildcards, the one case that would need a bare term, are
    refused by :meth:`CoreQueryBuilder.validate_query`.

    Parameters
    ----------
    value : str
        Stripped term value.

    Returns
    -------
    str
        Quoted term, safe to concatenate after a field prefix.
    """
    return '"' + value.replace('"', r"\"") + '"'
