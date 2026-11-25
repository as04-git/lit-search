"""SSRN query builder.

This builder rejects almost everything, and the reason is the point.

SSRN's free-text search route (``api.ssrn.com/content/v1/search``) answers
``HTTP 401 {"message":"Unauthorized"}`` to an unauthenticated request, checked
live on 2026-08-31, so **its query semantics could not be observed at all**.
The HTML search pages the donor connector scraped are no longer usable either:
``www.ssrn.com/index.cfm/en/rps-stage1-results/`` returns HTTP 404 ("Page
Cannot be Found") and ``papers.ssrn.com/sol3/results.cfm`` returns HTTP 403
behind a Cloudflare challenge, with an ordinary client and with a
browser-impersonating one alike.

What that leaves is a choice between two ways to be wrong.  A builder could
map ``[a] AND [b]`` onto whatever string SSRN happens to accept and hope the
operator is honoured -- and if it is not, the search returns a plausible,
well-ordered list for a different question, with nothing in the output saying
so.  That is precisely the failure this fork exists to prevent, and the wrong
answer is silent.  Refusing the query instead is loud: it reaches the ledger as
``QUERY_UNSUPPORTED``, naming SSRN and naming the reason, and the user can see
that SSRN was asked nothing.

So this builder accepts a single term and nothing else.  A single term is the
one shape whose meaning does not depend on an operator, so it is the only one
that can be sent without inventing a fact about SSRN.

A neighbouring observation shows the risk is not theoretical.  The sibling
route ``api.ssrn.com/content/v1/bindings/{id}/papers`` is reachable without a
credential, and adding ``&keyword=catalysis`` to it changes nothing at all: it
returned the same ``"total": 59666`` as the unfiltered request.  SSRN's API
accepts a parameter it does not implement and answers with the unfiltered set.

**Revisit this module the moment an SSRN credential exists.**  Everything here
is a consequence of not being able to look, not a judgement about SSRN's
capabilities.
"""

from __future__ import annotations

from lit_search.core.query import FilterCode, NodeType, Query, QueryNode
from lit_search.query.builder import QueryBuilder, QueryValidationResult

#: Filter codes that denote SSRN's own default search scope rather than a
#: field SSRN would have to scope to.  Anything narrower is refused, because
#: no field-scoped syntax could be observed.
_FREE_TEXT_FILTERS = frozenset(
    {
        FilterCode.TITLE_ABSTRACT,
        FilterCode.TITLE_ABSTRACT_KEYWORDS,
    }
)

#: Recorded once and reused so the ledger entry and the test assert the same
#: sentence.
UNVERIFIED_SEMANTICS_NOTE = (
    "SSRN's free-text search route answers HTTP 401 without a credential "
    "(checked 2026-08-31), so its query semantics have never been observed and "
    "no Boolean or field-scoped query can be sent without guessing what SSRN "
    "would do with it"
)


class SSRNQueryBuilder(QueryBuilder):
    """Build SSRN free-text search strings.

    Accepts a single unscoped term.  See the module docstring for why the
    accepted surface is this narrow.
    """

    _SUPPORTED_FILTERS = frozenset(_FREE_TEXT_FILTERS)

    def validate_query(self, query: Query) -> QueryValidationResult:
        """Validate whether SSRN can be asked this query at all.

        Parameters
        ----------
        query : Query
            Query to validate.

        Returns
        -------
        QueryValidationResult
            Valid only for a single term in SSRN's default scope.
        """
        terms = self.iter_term_nodes(query.root)
        if not terms:
            return QueryValidationResult(
                is_valid=False, error_message="SSRN cannot run a query with no search terms."
            )
        if len(terms) > 1 or self.iter_connectors(query.root):
            return QueryValidationResult(
                is_valid=False,
                error_message=(
                    f"SSRN was sent nothing: this query combines terms with a Boolean "
                    f"operator, and {UNVERIFIED_SEMANTICS_NOTE}. Running it as a single "
                    "string would return a plausible list for a different question."
                ),
            )
        term = terms[0]
        value = (term.value or "").strip()
        if not value:
            return QueryValidationResult(
                is_valid=False, error_message="SSRN cannot run an empty search term."
            )
        filter_code = self.get_effective_filter(term)
        if filter_code not in _FREE_TEXT_FILTERS:
            return QueryValidationResult(
                is_valid=False,
                error_message=(
                    f"SSRN was sent nothing: the term is scoped to '{filter_code}', and "
                    f"{UNVERIFIED_SEMANTICS_NOTE}. A field-scoped term would have to be "
                    "sent as unscoped free text, which searches a wider set than asked."
                ),
            )
        return QueryValidationResult(is_valid=True)

    def convert_query(self, query: Query) -> str:
        """Convert a query into an SSRN free-text search string.

        Parameters
        ----------
        query : Query
            Query to convert.  Must have passed :meth:`validate_query`.

        Returns
        -------
        str
            The single term's value.
        """
        terms = self.iter_term_nodes(query.root)
        return (terms[0].value or "").strip() if terms else ""

    @staticmethod
    def _is_single_term(node: QueryNode) -> bool:
        """Whether *node* is a bare TERM node.

        Parameters
        ----------
        node : QueryNode
            Node to inspect.

        Returns
        -------
        bool
            ``True`` for a TERM node.
        """
        return node.node_type is NodeType.TERM
