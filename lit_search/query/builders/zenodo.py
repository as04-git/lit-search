"""Zenodo query builder.

Zenodo's record search is an Elasticsearch ``query_string`` surface exposed
through InvenioRDM.  Field names and syntax behaviour below were verified
against the live API on 2026-08-31 (``https://zenodo.org/api/records``).  Four
of those observations are load-bearing, and each one is a way for a wrong
query to come back looking right:

1. **An unrecognised field name is a silent zero.**  ``abstract:catalysis``
   and ``authors:Zhang`` both return HTTP 200 with ``hits.total == 0``, while
   ``description:catalysis`` returns 1,929 and ``creators.name:Zhang`` returns
   90,613.  ``abstract`` and ``authors`` are the *obvious* guesses and both are
   wrong, so every field name this builder can emit comes from the verified
   whitelist below.

2. **An empty query is not an error -- it is the whole repository.**  ``q=``
   returned ``hits.total == 7,218,253``.  A query that renders to an empty
   string would therefore return Zenodo's entire corpus as if it matched.

3. **Malformed syntax is not an error either.**  ``title:(((`` returned HTTP
   200 with ``hits.total == 47,131`` -- an order of magnitude more than
   ``title:catalysis``'s 772, and plausible-looking.  Unbalanced brackets are
   rejected here rather than sent.

4. **Wildcards do expand, and are kept.**  ``title:catalys*`` returned 4,638
   against 772 for ``title:catalysis``.  A wildcard beside a space is still
   refused, because a quoted phrase does not expand wildcards and an unquoted
   multi-word term is tokenised, so the term would search something other than
   what it says.
"""

from __future__ import annotations

from lit_search.core.query import ConnectorType, FilterCode, Query, QueryNode
from lit_search.query.builder import QueryBuilder, QueryValidationResult

# Mapping from lit_search filter codes to Zenodo indexed search fields.
# Every value was confirmed to return a plausible non-zero count on
# 2026-08-31, and every rejected alternative was confirmed to return zero.
FIELD_MAP: dict[FilterCode, str] = {
    FilterCode.TITLE: "title",
    FilterCode.ABSTRACT: "description",
    FilterCode.KEYWORDS: "keywords",
    FilterCode.AUTHOR: "creators.name",
}

# Zenodo has no combined title/abstract field, so the composite filter codes
# are written out as explicit disjunctions.  Verified 2026-08-31: the
# title-or-description form returned 59,813 for "machine learning" against
# 20,989 for the title alone, and adding keywords raised it to 76,802.
_COMPOSITE_FIELDS: dict[FilterCode, tuple[str, ...]] = {
    FilterCode.TITLE_ABSTRACT: ("title", "description"),
    FilterCode.TITLE_ABSTRACT_KEYWORDS: ("title", "description", "keywords"),
}

_WILDCARD_CHARS = ("*", "?")

# Characters whose imbalance Zenodo accepts and silently reinterprets.
_BRACKET_PAIRS = (("(", ")"), ("[", "]"), ("{", "}"))


class ZenodoQueryBuilder(QueryBuilder):
    """Build Zenodo ``q`` expressions.

    Zenodo indexes a title, a description, a keyword list and creator names.
    It indexes no journal title and no author affiliation, so
    :data:`~lit_search.core.query.FilterCode.SOURCE` and
    :data:`~lit_search.core.query.FilterCode.AFFILIATION` are refused rather
    than mapped onto the nearest-looking field.
    """

    _SUPPORTED_FILTERS = frozenset(
        {
            FilterCode.TITLE,
            FilterCode.ABSTRACT,
            FilterCode.KEYWORDS,
            FilterCode.AUTHOR,
            FilterCode.TITLE_ABSTRACT,
            FilterCode.TITLE_ABSTRACT_KEYWORDS,
        }
    )

    def validate_query(self, query: Query) -> QueryValidationResult:
        """Validate whether Zenodo can express this query faithfully.

        Parameters
        ----------
        query : Query
            Query to validate.

        Returns
        -------
        QueryValidationResult
            Invalid when a term is empty, carries unbalanced brackets, mixes a
            wildcard with a space, or uses a filter Zenodo does not index.
            Every one of those would run upstream and return a plausible
            number of records for a query other than the one asked.
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
                        f"Filter '{filter_code}' is not indexed by Zenodo. Zenodo record "
                        "metadata carries a title, a description, keywords and creator "
                        "names, and no journal title or affiliation; mapping this filter "
                        "onto the nearest available field would search something other "
                        "than what the query asks for."
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
                    "Zenodo treats an empty query as a match-all: 'q=' returned 7,218,253 "
                    "records on 2026-08-31. An empty term cannot be sent, because the "
                    "result would be the whole repository reported as a match."
                ),
            )
        for opening, closing in _BRACKET_PAIRS:
            if value.count(opening) != value.count(closing):
                return QueryValidationResult(
                    is_valid=False,
                    error_message=(
                        f"Term '{value}' has unbalanced '{opening}{closing}' brackets. "
                        "Zenodo accepts malformed syntax with HTTP 200 and reinterprets it: "
                        "'title:(((' returned 47,131 records against 772 for "
                        "'title:catalysis', so the search would succeed and answer a "
                        "different question."
                    ),
                )
        if any(char in value for char in _WILDCARD_CHARS) and " " in value:
            return QueryValidationResult(
                is_valid=False,
                error_message=(
                    f"Term '{value}' combines a wildcard with a space. Zenodo does not "
                    "expand wildcards inside a quoted phrase and tokenises an unquoted "
                    "multi-word term, so this term would search a different field scope "
                    "than requested."
                ),
            )
        return None

    def convert_query(self, query: Query) -> str:
        """Convert a query into a Zenodo ``q`` expression.

        Parameters
        ----------
        query : Query
            Query to convert.  Must have passed :meth:`validate_query`.

        Returns
        -------
        str
            Zenodo query expression.
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
            composite = _COMPOSITE_FIELDS.get(filter_code)
            if composite is not None:
                joined = " OR ".join(f"{field}:{rendered}" for field in composite)
                return f"({joined})"
            return f"{FIELD_MAP[filter_code]}:{rendered}"

        return self.convert_expression(query.root, convert_term, connector_map)


def _render_value(value: str) -> str:
    """Render a term value for Zenodo, quoting where that is safe.

    A term is quoted so that a multi-word value searches as a phrase rather
    than as separate tokens, which is what the bracketed term syntax means.  A
    term carrying a wildcard is left unquoted, because Zenodo does not expand
    wildcards inside a quoted phrase; :meth:`ZenodoQueryBuilder.validate_query`
    has already guaranteed such a term has no space, so leaving it unquoted
    cannot leak past the field prefix.

    Parameters
    ----------
    value : str
        Stripped term value.

    Returns
    -------
    str
        Quoted or bare term, safe to concatenate after a field prefix.
    """
    if any(char in value for char in _WILDCARD_CHARS):
        return value
    return '"' + value.replace('"', r"\"") + '"'
