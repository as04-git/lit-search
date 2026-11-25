"""Unit tests for BaseSearchConnector.

BASE is registered and never consulted, and these tests pin both halves: that
it reports the skip rather than an empty result, and that it issues no request
while doing so.  A connector that quietly started making a request would be
sending traffic to a host that refuses it, and the refusal arrives as HTTP 200
with an XML error body -- which is exactly the shape that reads as zero papers.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from lit_search.connectors.base_search import CREDENTIAL_NAME, BaseSearchConnector
from lit_search.core.coverage import CoverageStatus, CoveredPapers
from lit_search.core.paper import Database
from lit_search.query.builders.base_search import BaseSearchQueryBuilder

from .conftest import _parse


@pytest.fixture
def connector() -> BaseSearchConnector:
    """Return a BASE connector."""
    return BaseSearchConnector()


class TestItIsSkippedNotEmpty:
    """The degraded label, as the ledger reports it."""

    def test_a_search_reports_skipped_no_credential(self, connector: BaseSearchConnector) -> None:
        """BASE was never asked, and the ledger says so rather than "0 papers"."""
        result = connector.search(_parse("[transformer]"))
        assert result.status is CoverageStatus.SKIPPED_NO_CREDENTIAL
        assert result.papers == ()
        assert result.coverage.missing_credential == CREDENTIAL_NAME

    def test_the_skip_is_not_a_failure(self, connector: BaseSearchConnector) -> None:
        """The distinction matters: a failure invites a retry, a skip does not."""
        result = connector.search(_parse("[transformer]"))
        assert result.status is not CoverageStatus.FAILED

    def test_no_request_is_issued(self, connector: BaseSearchConnector) -> None:
        """Nothing reaches a host that refuses this one with HTTP 200."""
        with patch.object(connector, "_get") as mocked:
            connector.search(_parse("[transformer]"))
        mocked.assert_not_called()

    def test_it_is_registered_so_the_absence_is_visible(self) -> None:
        """An unregistered source is invisible in every coverage ledger."""
        from lit_search.connectors import SEARCH_REGISTRY

        assert SEARCH_REGISTRY[Database.BASE_SEARCH] is BaseSearchConnector

    def test_search_returns_the_contract_type(self, connector: BaseSearchConnector) -> None:
        """The connector cannot hand back a bare list."""
        assert isinstance(connector.search(_parse("[transformer]")), CoveredPapers)

    def test_the_rate_is_the_published_crawl_delay(self, connector: BaseSearchConnector) -> None:
        """Declared correctly now, so the first request ever sent is paced."""
        assert connector.min_request_interval == 2.0


class TestQueryBuilder:
    """The grammar, which is written and unexercised."""

    @pytest.fixture
    def builder(self) -> BaseSearchQueryBuilder:
        """Return a BASE query builder."""
        return BaseSearchQueryBuilder()

    def test_a_scoped_term_renders_with_its_dublin_core_field(
        self, builder: BaseSearchQueryBuilder
    ) -> None:
        """A title term becomes ``dctitle:``."""
        assert builder.convert_query(_parse("ti[transformer]")) == "dctitle:transformer"

    def test_the_default_scope_expands_to_title_abstract_and_subject(
        self, builder: BaseSearchQueryBuilder
    ) -> None:
        """An unqualified term searches all three fields, disjunctively.

        The default is ``tiabskey`` rather than ``tiabs`` because BASE has a
        ``dcsubject`` element, and
        :meth:`~lit_search.query.builder.QueryBuilder.get_effective_filter`
        prefers the wider default wherever keywords are supported.
        """
        rendered = builder.convert_query(_parse("[transformer]"))
        assert rendered == (
            "(dctitle:transformer OR dcdescription:transformer OR dcsubject:transformer)"
        )

    def test_an_unsupported_scope_is_rejected(self, builder: BaseSearchQueryBuilder) -> None:
        """BASE has no affiliation element, so ``[aff]`` cannot be expressed."""
        assert not builder.validate_query(_parse("aff[mit]")).is_valid

    def test_an_unsupported_scope_reaches_the_ledger_as_query_unsupported(
        self, connector: BaseSearchConnector
    ) -> None:
        """Validation runs before the skip, so the more specific status wins."""
        result = connector.search(_parse("aff[mit]"))
        assert result.status is CoverageStatus.QUERY_UNSUPPORTED
