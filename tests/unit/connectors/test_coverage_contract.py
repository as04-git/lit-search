"""Tests that a search connector cannot express silence.

Each test here corresponds to one way a hurried port could return ``[]`` on
failure.  All of them must fail loudly, and the earlier the better: at class
definition if possible, at the first call otherwise, and never at the user
weeks later reading a confident, incomplete result.

The three enforcement layers under test:

1. ``__init_subclass__`` -- rejects a wrong or missing return annotation, and
   any attempt to override the final ``search`` entry point.
2. ``SearchConnectorBase.search`` -- runtime type check on what
   ``_fetch_papers`` actually handed back.
3. ``SearchRunner`` -- re-checks at the fan-in, catching a monkey-patched or
   mocked connector that bypassed both.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import requests

from lit_search.connectors.arxiv import ArxivConnector
from lit_search.connectors.search_base import SearchConnectorBase, _CoverageContractMeta
from lit_search.connectors.uninstrumented_base import UninstrumentedSearchConnectorBase
from lit_search.core.coverage import CoverageStatus, CoveredPapers, SourceCoverage
from lit_search.core.paper import Database, Paper
from lit_search.core.query import Query
from lit_search.exceptions import CoverageContractError
from lit_search.query.builder import QueryBuilder, QueryValidationResult
from lit_search.runners.search_runner import SearchRunner

from .conftest import search_papers


def _valid_builder() -> MagicMock:
    """Build a query builder that accepts everything.

    Returns
    -------
    MagicMock
        Stub builder whose ``validate_query`` always passes.
    """
    builder = MagicMock()
    builder.validate_query.return_value = QueryValidationResult(is_valid=True, error_message=None)
    return builder


class _ConformingConnector(SearchConnectorBase):
    """A minimal connector that honours the coverage contract."""

    def __init__(self, result: CoveredPapers | None = None) -> None:
        """Create the stub.

        Parameters
        ----------
        result : CoveredPapers | None
            Value ``_fetch_papers`` should return.
        """
        super().__init__()
        self._builder = _valid_builder()
        self._result = result or CoveredPapers.none(
            SourceCoverage.complete("stub", 0, reported_total=0)
        )

    @property
    def name(self) -> str:
        """Connector name.

        Returns
        -------
        str
            ``"stub"``.
        """
        return "stub"

    @property
    def query_builder(self) -> QueryBuilder:
        """Query builder.

        Returns
        -------
        QueryBuilder
            The stub builder.
        """
        return self._builder

    @property
    def min_request_interval(self) -> float:
        """Rate limit.

        Returns
        -------
        float
            ``0.0``.
        """
        return 0.0

    def _fetch_papers(
        self,
        query: Query,
        max_papers: int | None,
        progress_callback: Callable[[int, int | None], None] | None,
        since: datetime.date | None = None,
        until: datetime.date | None = None,
    ) -> CoveredPapers:
        """Return the preconfigured result, reporting what it saw first.

        The progress call is what makes this connector *conforming* rather
        than merely contract-shaped.  It is the channel through which a
        connector states how many records it saw and what total the source
        published, and the base class needs both to let a ``COMPLETE`` claim
        stand.

        Parameters
        ----------
        query : Query
            Ignored.
        max_papers : int | None
            Ignored.
        progress_callback : Callable[[int, int | None], None] | None
            Called once with the result's own counts.
        since : datetime.date | None
            Ignored.
        until : datetime.date | None
            Ignored.

        Returns
        -------
        CoveredPapers
            The configured result.
        """
        if progress_callback is not None:
            progress_callback(len(self._result.papers), self._result.coverage.reported_total)
        return self._result


class TestClassDefinitionEnforcement:
    """A connector that forgets coverage fails at import, not at the user."""

    def test_returning_a_bare_list_is_rejected_at_class_definition(self) -> None:
        """The mechanically-ported ``-> list[Paper]`` connector cannot be defined.

        This is the exact shape of every paper-search-mcp connector, and the
        reason prose in a porting brief is not enough.
        """
        with pytest.raises(CoverageContractError, match="list\\[Paper\\]"):

            class _BareListConnector(SearchConnectorBase):
                """Port that kept the old signature."""

                def _fetch_papers(
                    self,
                    query: Query,
                    max_papers: int | None,
                    progress_callback: Callable | None,
                    since: datetime.date | None = None,
                    until: datetime.date | None = None,
                ) -> list[Paper]:
                    """Return papers with no coverage.

                    Returns
                    -------
                    list[Paper]
                        Always empty.
                    """
                    return []

    def test_missing_return_annotation_is_rejected(self) -> None:
        """Dropping the annotation is not a way around the contract."""
        with pytest.raises(CoverageContractError, match="no return annotation"):

            class _UnannotatedConnector(SearchConnectorBase):
                """Port that dropped the annotation."""

                def _fetch_papers(  # type: ignore[override]
                    self,
                    query,
                    max_papers,
                    progress_callback,
                    since=None,
                    until=None,
                ):
                    """Return papers with no coverage."""
                    return []

    def test_error_message_tells_the_author_what_to_do(self) -> None:
        """The failure names the required type and the transitional base."""
        with pytest.raises(CoverageContractError) as excinfo:

            class _AnotherBareConnector(SearchConnectorBase):
                """Port that kept the old signature."""

                def _fetch_papers(
                    self,
                    query: Query,
                    max_papers: int | None,
                    progress_callback: Callable | None,
                    since: datetime.date | None = None,
                    until: datetime.date | None = None,
                ) -> list[Paper]:
                    """Return papers with no coverage.

                    Returns
                    -------
                    list[Paper]
                        Always empty.
                    """
                    return []

        message = str(excinfo.value)
        assert "CoveredPapers" in message
        assert "UninstrumentedSearchConnectorBase" in message
        assert "SourceCoverage" in message

    def test_overriding_search_is_rejected(self) -> None:
        """``search`` is the validation point, so it cannot be replaced."""
        with pytest.raises(CoverageContractError, match="final on SearchConnectorBase"):

            class _OverridingConnector(SearchConnectorBase):
                """Port that routed around the entry point."""

                def search(self, *args, **kwargs):  # type: ignore[override]
                    """Return papers directly.

                    Returns
                    -------
                    list
                        Always empty.
                    """
                    return []

                def _fetch_papers(
                    self,
                    query: Query,
                    max_papers: int | None,
                    progress_callback: Callable | None,
                    since: datetime.date | None = None,
                    until: datetime.date | None = None,
                ) -> CoveredPapers:
                    """Return an empty covered result.

                    Returns
                    -------
                    CoveredPapers
                        Empty result.
                    """
                    return CoveredPapers.none(SourceCoverage.complete("x", 0, reported_total=0))

    def test_conforming_connector_is_accepted(self) -> None:
        """The contract does not reject a correct connector."""
        assert _ConformingConnector().search(MagicMock()).status is CoverageStatus.COMPLETE

    def test_every_registered_connector_satisfies_the_contract(self) -> None:
        """Importing the registry is itself the check for the shipped connectors."""
        from lit_search.connectors import SEARCH_REGISTRY

        assert SEARCH_REGISTRY
        for cls in SEARCH_REGISTRY.values():
            assert issubclass(cls, SearchConnectorBase)


class _Decoys:
    """Namespace holding an impostor whose final dotted name is the right one.

    Reachable from module globals, which is where a string annotation is
    evaluated, so ``-> _Decoys.CoveredPapers`` resolves to a real class that is
    not the contract type.
    """

    class CoveredPapers:
        """Impostor sharing the contract type's bare name."""


class TestMroBypass:
    """The contract follows method resolution, not the class body.

    ``__init_subclass__`` used to inspect ``cls.__dict__`` alone, which asks
    "did *this* class body define it".  A mixin listed before the base answers
    that "no" while supplying the implementation that actually runs, so the
    check concluded the method had been inherited from the validated abstract
    base and let the connector through.  Confirmed against the shipped tree on
    2026-08-30: ``type=list len=2 coverage=<none>``.
    """

    def test_a_mixin_supplying_fetch_papers_is_rejected(self) -> None:
        """The bypass: the mixin wins the MRO and the class body looks empty."""

        class _BareListMixin:
            """Donor-shaped implementation with no coverage at all."""

            def _fetch_papers(
                self,
                query: Query,
                max_papers: int | None,
                progress_callback: Callable | None,
                since: datetime.date | None = None,
                until: datetime.date | None = None,
            ) -> list:
                """Return papers with no coverage.

                Returns
                -------
                list
                    Always empty.
                """
                return []

        with pytest.raises(CoverageContractError, match="resolved through the MRO"):

            class _MixedInConnector(_BareListMixin, SearchConnectorBase):  # type: ignore[misc]
                """Connector whose own body defines nothing to inspect."""

    def test_a_mixin_supplying_search_is_rejected(self) -> None:
        """The final entry point cannot be replaced through the MRO either."""

        class _SearchMixin:
            """Mixin that replaces the validation point."""

            def search(self, *args, **kwargs):
                """Return papers directly.

                Returns
                -------
                list
                    Always empty.
                """
                return []

        with pytest.raises(CoverageContractError, match="final on SearchConnectorBase"):

            class _MixedInSearch(_SearchMixin, SearchConnectorBase):  # type: ignore[misc]
                """Connector that routed around the entry point via a mixin."""

    def test_the_error_names_the_class_that_actually_supplies_the_method(self) -> None:
        """A message pointing at the empty class body would not be actionable."""

        class _WrongAnnotationMixin:
            """Mixin annotated with the wrong return type."""

            def _fetch_papers(
                self,
                query: Query,
                max_papers: int | None,
                progress_callback: Callable | None,
                since: datetime.date | None = None,
                until: datetime.date | None = None,
            ) -> list:
                """Return papers with no coverage.

                Returns
                -------
                list
                    Always empty.
                """
                return []

        with pytest.raises(CoverageContractError, match="_WrongAnnotationMixin"):

            class _Connector(_WrongAnnotationMixin, SearchConnectorBase):  # type: ignore[misc]
                """Connector inheriting the offending method."""

    def test_a_decoy_annotation_is_rejected_at_class_definition(self) -> None:
        """A different class of the same name must not pass the name comparison.

        The string fallback exists for annotations that cannot be evaluated,
        typically because the name is imported only under ``TYPE_CHECKING``.
        It used to run even when evaluation *succeeded* and produced the wrong
        class, leaving the mismatch to be caught at runtime instead.
        """

        with pytest.raises(CoverageContractError, match="coverage contract requires"):

            class _DecoyConnector(SearchConnectorBase):
                """Connector annotated with a dotted path to the impostor."""

                def _fetch_papers(  # type: ignore[override]
                    self,
                    query: Query,
                    max_papers: int | None,
                    progress_callback: Callable | None,
                    since: datetime.date | None = None,
                    until: datetime.date | None = None,
                ) -> _Decoys.CoveredPapers:
                    """Return the impostor.

                    Returns
                    -------
                    _Decoys.CoveredPapers
                        Not the contract type, despite the final name matching.
                    """
                    return _Decoys.CoveredPapers()

    def test_an_inherited_valid_implementation_is_still_accepted(self) -> None:
        """The fix must not reject the transitional base's own final method."""

        class _ThroughTheUninstrumentedBase(UninstrumentedSearchConnectorBase):
            """Connector whose ``_fetch_papers`` comes from the MRO, correctly."""

            def _fetch_papers_uninstrumented(
                self,
                query: Query,
                max_papers: int | None,
                progress_callback: Callable | None,
                since: datetime.date | None = None,
                until: datetime.date | None = None,
            ) -> list:
                """Return no papers.

                Returns
                -------
                list
                    Always empty.
                """
                return []

        assert _ThroughTheUninstrumentedBase is not None


class _HijacksTheHook(_ConformingConnector):
    """A fully conforming connector that switches the enforcement hook off.

    Nothing about this class is a violation: its ``_fetch_papers`` is annotated
    and returns the contract type.  Its only sin is declaring
    ``__init_subclass__`` without calling ``super()``, which is an easy thing to
    do for an unrelated reason (registering plugins, say) and which nobody
    reviewing *this* class would flag.

    The damage lands on its children.  Python calls only the nearest
    ``__init_subclass__`` in the MRO, so every class beneath this one was
    unguarded until the check also moved onto the metaclass.  Confirmed against
    the shipped tree on 2026-08-31: a child of a class like this returned
    ``type=list len=1 coverage=<none>`` from the ``@final`` ``search``.
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Swallow the enforcement chain instead of forwarding it.

        Parameters
        ----------
        **kwargs : object
            Deliberately dropped rather than passed to ``super()``.
        """


class TestHijackedInitSubclassCannotDisableTheContract:
    """A subclass must not be able to opt its own children out of the check.

    ``__init_subclass__`` is cooperative, so it is only ever as strong as the
    least careful class in the chain.  The metaclass is not, which is why the
    check runs from both.
    """

    def test_the_hijacking_class_is_itself_legal(self) -> None:
        """The setup is not the violation, which is what makes it dangerous."""
        assert issubclass(_HijacksTheHook, SearchConnectorBase)

    def test_a_child_of_it_still_cannot_return_a_bare_list(self) -> None:
        """The escape route: wrong annotation, under a hook that never fires."""
        with pytest.raises(CoverageContractError, match="coverage contract requires"):

            class _SneakyChild(_HijacksTheHook):
                """Child relying on the hijacked hook to skip validation."""

                def _fetch_papers(  # type: ignore[override]
                    self,
                    query: Query,
                    max_papers: int | None,
                    progress_callback: Callable | None,
                    since: datetime.date | None = None,
                    until: datetime.date | None = None,
                ) -> list:
                    """Return papers with no coverage.

                    Returns
                    -------
                    list
                        Always empty.
                    """
                    return []

    def test_a_child_of_it_still_cannot_replace_the_final_search(self) -> None:
        """The worse half: overriding the point where coverage is stamped."""
        with pytest.raises(CoverageContractError, match="final on SearchConnectorBase"):

            class _SneakySearch(_HijacksTheHook):
                """Child that supplies its own entry point."""

                def search(self, *args: object, **kwargs: object) -> list:
                    """Return papers with no coverage at all.

                    Returns
                    -------
                    list
                        Always empty.
                    """
                    return []

    def test_the_check_survives_an_arbitrarily_long_hijacked_chain(self) -> None:
        """One guarded generation is not enough; every generation is guarded."""

        class _Middle(_HijacksTheHook):
            """A legal intermediate generation."""

        class _Deeper(_Middle):
            """Another one."""

        with pytest.raises(CoverageContractError, match="coverage contract requires"):

            class _Bottom(_Deeper):
                """The violation, four generations from the base."""

                def _fetch_papers(  # type: ignore[override]
                    self,
                    query: Query,
                    max_papers: int | None,
                    progress_callback: Callable | None,
                    since: datetime.date | None = None,
                    until: datetime.date | None = None,
                ) -> list:
                    """Return papers with no coverage.

                    Returns
                    -------
                    list
                        Always empty.
                    """
                    return []

    def test_the_metaclass_is_inherited_by_every_connector(self) -> None:
        """A connector built with a plain metaclass would escape the check."""
        from lit_search.connectors import SEARCH_REGISTRY

        assert SEARCH_REGISTRY
        for cls in SEARCH_REGISTRY.values():
            assert isinstance(cls, _CoverageContractMeta)


class TestRuntimeEnforcement:
    """A body that ignores its own annotation is still caught."""

    def test_returning_a_list_at_runtime_raises(self, make_paper) -> None:
        """Annotation says CoveredPapers, body returns a list: caught by search()."""
        connector = _ConformingConnector()
        with (
            patch.object(connector, "_fetch_papers", return_value=[make_paper()]),
            pytest.raises(CoverageContractError, match="returned list"),
        ):
            connector.search(MagicMock())

    def test_returning_none_at_runtime_raises(self) -> None:
        """Returning None is caught too, rather than becoming an empty result."""
        connector = _ConformingConnector()
        with (
            patch.object(connector, "_fetch_papers", return_value=None),
            pytest.raises(CoverageContractError, match="returned NoneType"),
        ):
            connector.search(MagicMock())

    def test_runner_rejects_a_connector_returning_a_bare_list(self, make_paper) -> None:
        """The runner re-checks, so a mocked connector cannot smuggle a list through.

        The runner records it as a failed database rather than as a search
        that legitimately found one paper.

        The diagnostic is asserted, not just the outcome.  Deleting the
        runner's ``isinstance`` guard leaves this database ``FAILED`` anyway --
        by ``AttributeError`` from the next line that treats the list as a
        result object -- so an outcome-only assertion passes with the guard
        removed and pins nothing.  Requiring the failure to *name* the contract
        is what makes the test fail when the guard goes.
        """
        runner = SearchRunner(query="[ml]", databases=["arxiv"])
        mock_searcher = MagicMock()
        mock_searcher.name = Database.ARXIV
        mock_searcher.search.return_value = [make_paper()]
        runner._searchers = [mock_searcher]

        result = runner.run(show_progress=False)

        assert result.papers == []
        assert Database.ARXIV in result.failed_databases
        assert result.coverage_status is CoverageStatus.FAILED
        entry = result.coverage.get("arxiv")
        assert entry is not None
        assert entry.error_type == CoverageContractError.__name__
        assert "CoveredPapers" in (entry.error or "")


class TestUninstrumentedBase:
    """The transitional base cannot be used to claim completeness."""

    class _Uninstrumented(UninstrumentedSearchConnectorBase):
        """Connector carried across the boundary without instrumentation."""

        def __init__(self, papers: list[Paper] | None = None) -> None:
            """Create the stub.

            Parameters
            ----------
            papers : list[Paper] | None
                Papers to return.
            """
            super().__init__()
            self._builder = _valid_builder()
            self._papers = papers or []

        @property
        def name(self) -> str:
            """Connector name.

            Returns
            -------
            str
                ``"legacy"``.
            """
            return "legacy"

        @property
        def query_builder(self) -> QueryBuilder:
            """Query builder.

            Returns
            -------
            QueryBuilder
                The stub builder.
            """
            return self._builder

        @property
        def min_request_interval(self) -> float:
            """Rate limit.

            Returns
            -------
            float
                ``0.0``.
            """
            return 0.0

        def _fetch_papers_uninstrumented(
            self,
            query: Query,
            max_papers: int | None,
            progress_callback: Callable[[int, int | None], None] | None,
            since: datetime.date | None = None,
            until: datetime.date | None = None,
        ) -> list[Paper]:
            """Return the configured papers.

            Returns
            -------
            list[Paper]
                Configured papers.
            """
            return self._papers

    def test_result_is_never_complete(self, make_paper) -> None:
        """Even a full-looking result is reported PARTIAL.

        Migrating a connector to real instrumentation can then only improve
        its reported coverage, so the transitional state errs safe.
        """
        result = self._Uninstrumented([make_paper()]).search(MagicMock())
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 1

    def test_detail_says_why_it_is_unverified(self, make_paper) -> None:
        """The reason is specific enough to be a work item."""
        result = self._Uninstrumented([make_paper()]).search(MagicMock())
        assert "not been instrumented" in (result.coverage.detail or "")

    def test_empty_result_is_still_partial_not_complete(self) -> None:
        """An uninstrumented connector returning nothing does not certify a zero."""
        result = self._Uninstrumented([]).search(MagicMock())
        assert result.status is CoverageStatus.PARTIAL

    def test_hitting_the_cap_is_reported(self, make_paper) -> None:
        """The generically-knowable cap signal is still surfaced."""
        connector = self._Uninstrumented([make_paper(), make_paper(title="b")])
        result = connector.search(MagicMock(), max_papers=2)
        assert "cap of 2" in (result.coverage.detail or "")

    def test_shipped_uninstrumented_connectors_report_partial(self) -> None:
        """The six not-yet-migrated connectors inherit the pessimistic status."""
        from lit_search.connectors.ieee import IEEEConnector

        assert issubclass(IEEEConnector, UninstrumentedSearchConnectorBase)


def _feed(n: int, total: int) -> str:
    """Build a minimal synthetic arXiv Atom feed.

    Parameters
    ----------
    n : int
        Number of entries in the page.
    total : int
        Value for ``opensearch:totalResults``.

    Returns
    -------
    str
        Atom XML.
    """
    entries = "".join(
        f"""<entry>
            <id>http://arxiv.org/abs/2301.0000{i}v1</id>
            <title>Synthetic Paper {i}</title>
            <published>2023-01-01T00:00:00Z</published>
            <summary>Synthetic abstract {i}.</summary>
            <author><name>Author {i}</name></author>
        </entry>"""
        for i in range(n)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" '
        'xmlns="http://www.w3.org/2005/Atom">'
        f"<opensearch:totalResults>{total}</opensearch:totalResults>{entries}</feed>"
    )


class TestArxivReferenceConnector:
    """The reference implementation reports each outcome distinctly.

    arXiv is the reference because it publishes a result total, which is what
    makes a COMPLETE claim justifiable rather than assumed.
    """

    def test_full_retrieval_is_complete(self, simple_query, mock_response) -> None:
        """Retrieving everything arXiv reported yields COMPLETE."""
        searcher = ArxivConnector()
        with patch.object(searcher, "_get", return_value=mock_response(text=_feed(3, 3))):
            result = searcher.search(simple_query)
        assert result.status is CoverageStatus.COMPLETE
        assert result.coverage.reported_total == 3

    def test_user_cap_is_capped_not_complete(self, simple_query, mock_response) -> None:
        """Stopping at max_papers is reported as CAPPED, with the cap value."""
        searcher = ArxivConnector()
        with patch.object(searcher, "_get", return_value=mock_response(text=_feed(2, 900))):
            result = searcher.search(simple_query, max_papers=2)
        assert result.status is CoverageStatus.CAPPED
        assert result.coverage.cap == 2

    def test_first_request_failure_is_failed_not_empty(self, simple_query) -> None:
        """A dead first request is FAILED, never a confident zero.

        This is the motivating failure at connector scale: the old base class
        turned this exact case into ``[]``.
        """
        searcher = ArxivConnector()
        with patch.object(searcher, "_get", side_effect=requests.ConnectionError("down")):
            result = searcher.search(simple_query)
        assert result.status is CoverageStatus.FAILED
        assert result.papers == ()
        assert "down" in (result.coverage.error or "")

    def test_failure_after_some_pages_is_partial(self, simple_query, mock_response) -> None:
        """Records already retrieved are kept, and the shortfall is reported."""
        searcher = ArxivConnector()
        with patch.object(
            searcher,
            "_get",
            side_effect=[
                mock_response(text=_feed(100, 900)),
                requests.ConnectionError("dropped"),
            ],
        ):
            result = searcher.search(simple_query)
        assert result.status is CoverageStatus.PARTIAL
        assert len(result.papers) == 100
        assert "dropped" in (result.coverage.detail or "")

    def test_empty_page_before_reported_total_is_partial(self, simple_query, mock_response) -> None:
        """arXiv's known short-paging behaviour degrades the status."""
        searcher = ArxivConnector()
        with patch.object(
            searcher,
            "_get",
            side_effect=[mock_response(text=_feed(100, 900)), mock_response(text=_feed(0, 900))],
        ):
            result = searcher.search(simple_query)
        assert result.status is CoverageStatus.PARTIAL
        assert "empty page" in (result.coverage.detail or "")

    def test_unparsable_entries_degrade_the_status(self, simple_query, mock_response) -> None:
        """Entries seen but not parsed are a coverage loss, not a smaller result.

        Without this, a feed whose schema drifts returns fewer papers and
        reports nothing at all about it.
        """
        searcher = ArxivConnector()
        feed = _feed(3, 3).replace("<title>Synthetic Paper 1</title>", "<title></title>")
        with patch.object(searcher, "_get", return_value=mock_response(text=feed)):
            result = searcher.search(simple_query)
        assert result.status is CoverageStatus.PARTIAL
        assert "could not be parsed" in (result.coverage.detail or "")
        assert len(result.papers) == 2

    def test_unsupported_query_is_reported_not_raised(self, key_query) -> None:
        """An inexpressible query is a coverage status, so it reaches the ledger."""
        result = ArxivConnector().search(key_query)
        assert result.status is CoverageStatus.QUERY_UNSUPPORTED

    def test_papers_still_reach_callers(self, simple_query, mock_response) -> None:
        """The contract does not change what a successful search returns."""
        searcher = ArxivConnector()
        with patch.object(searcher, "_get", return_value=mock_response(text=_feed(3, 3))):
            papers = search_papers(searcher, simple_query)
        assert len(papers) == 3


class TestRunnerLedger:
    """The runner assembles per-source coverage into the result set's ledger."""

    def test_skipped_database_is_recorded_not_a_silent_zero(self) -> None:
        """A database dropped for a missing key appears in the ledger.

        Before the contract this produced ``total_papers_from_ieee = 0``,
        indistinguishable from a database that genuinely found nothing.
        """
        runner = SearchRunner(query="[ml]", databases=["arxiv", "ieee"])
        mock_searcher = MagicMock()
        mock_searcher.name = Database.ARXIV
        mock_searcher.search.return_value = CoveredPapers.none(
            SourceCoverage.complete("arxiv", 0, reported_total=0)
        )
        runner._searchers = [mock_searcher]

        result = runner.run(show_progress=False)

        entry = result.coverage.get("ieee")
        assert entry is not None
        assert entry.status is CoverageStatus.SKIPPED_NO_CREDENTIAL
        assert entry.missing_credential == "ieee_api_key"
        assert result.coverage_status is CoverageStatus.SKIPPED_NO_CREDENTIAL

    def test_genuinely_empty_is_distinguishable_from_skipped(self) -> None:
        """A source that ran and found nothing reports COMPLETE with zero records."""
        runner = SearchRunner(query="[ml]", databases=["arxiv"])
        mock_searcher = MagicMock()
        mock_searcher.name = Database.ARXIV
        mock_searcher.search.return_value = CoveredPapers.none(
            SourceCoverage.complete("arxiv", 0, reported_total=0)
        )
        runner._searchers = [mock_searcher]

        result = runner.run(show_progress=False)

        assert result.coverage_status is CoverageStatus.COMPLETE
        assert result.is_complete

    def test_ledger_survives_serialization(self) -> None:
        """A saved result still knows what it could not reach."""
        from lit_search.core.search_result import SearchResult

        runner = SearchRunner(query="[ml]", databases=["arxiv", "scopus"])
        mock_searcher = MagicMock()
        mock_searcher.name = Database.ARXIV
        mock_searcher.search.return_value = CoveredPapers.none(
            SourceCoverage.complete("arxiv", 0, reported_total=0)
        )
        runner._searchers = [mock_searcher]
        result = runner.run(show_progress=False)

        restored = SearchResult.from_dict(result.to_dict())
        assert restored.coverage == result.coverage
        assert restored.coverage_status is CoverageStatus.SKIPPED_NO_CREDENTIAL

    def test_result_without_recorded_coverage_is_unknown(self) -> None:
        """A hand-built or legacy result does not read as complete."""
        from lit_search.core.search_result import SearchResult

        assert SearchResult(query="[ml]").coverage_status is CoverageStatus.UNKNOWN

    def test_render_lists_consulted_and_unreached_sources(self) -> None:
        """A result set can state its own coverage honestly."""
        runner = SearchRunner(query="[ml]", databases=["arxiv", "ieee"])
        mock_searcher = MagicMock()
        mock_searcher.name = Database.ARXIV
        mock_searcher.search.return_value = CoveredPapers.none(
            SourceCoverage.complete("arxiv", 0, reported_total=0)
        )
        runner._searchers = [mock_searcher]

        text = runner.run(show_progress=False).render_coverage()

        assert "arxiv" in text
        assert "ieee" in text
        assert "ieee_api_key" in text
