"""Transitional base for search connectors not yet instrumented for coverage.

The coverage contract in
:mod:`lit_search.connectors.search_base` requires every connector to state how
complete its result is.  Deciding that honestly means reading the connector's
pagination, cap and error handling and knowing what its source reports -- work
that has to happen once per connector.

This class exists so that a connector can be carried across the contract
boundary **without anyone guessing**.  It implements ``_fetch_papers`` by
delegating to an uninstrumented fetch and labelling the result
:data:`~lit_search.core.coverage.CoverageStatus.PARTIAL`.

The important property is asymmetric: a connector on this base can never
report :data:`~lit_search.core.coverage.CoverageStatus.COMPLETE`.  Migrating it
to the real contract can only ever improve the reported coverage, so the
transitional state errs toward saying "we do not know", which is the safe
direction.  Every use is greppable, and the class is expected to be deleted
once no connector inherits from it.
"""

from __future__ import annotations

import datetime
from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, final

from lit_search.connectors.search_base import SearchConnectorBase
from lit_search.core.coverage import CoveredPapers, SourceCoverage

if TYPE_CHECKING:
    from lit_search.core.paper import Paper
    from lit_search.core.query import Query

UNINSTRUMENTED_DETAIL = (
    "connector has not been instrumented for coverage: it reports no result "
    "total, no pagination shortfall and no per-page failures, so completeness "
    "cannot be verified"
)


class UninstrumentedSearchConnectorBase(SearchConnectorBase):
    """Base for connectors whose coverage accounting has not been written yet.

    Subclasses implement :meth:`_fetch_papers_uninstrumented` with the plain
    ``list[Paper]`` body they already had.  Everything such a connector
    returns is reported as
    :data:`~lit_search.core.coverage.CoverageStatus.PARTIAL`.
    """

    @abstractmethod
    def _fetch_papers_uninstrumented(
        self,
        query: Query,
        max_papers: int | None,
        progress_callback: Callable[[int, int | None], None] | None,
        since: datetime.date | None = None,
        until: datetime.date | None = None,
    ) -> list[Paper]:
        """Fetch papers without reporting coverage.

        Parameters
        ----------
        query : Query
            Pre-validated query object.
        max_papers : int | None
            Maximum papers to retrieve.
        progress_callback : Callable[[int, int | None], None] | None
            Progress callback.
        since : datetime.date | None
            Lower-bound publication date.
        until : datetime.date | None
            Upper-bound publication date.

        Returns
        -------
        list[Paper]
            Retrieved papers, with no statement about completeness.
        """

    @final
    def _fetch_papers(
        self,
        query: Query,
        max_papers: int | None,
        progress_callback: Callable[[int, int | None], None] | None,
        since: datetime.date | None = None,
        until: datetime.date | None = None,
    ) -> CoveredPapers:
        """Wrap the uninstrumented fetch in a deliberately pessimistic status.

        Parameters
        ----------
        query : Query
            Pre-validated query object.
        max_papers : int | None
            Maximum papers to retrieve.
        progress_callback : Callable[[int, int | None], None] | None
            Progress callback.
        since : datetime.date | None
            Lower-bound publication date.
        until : datetime.date | None
            Upper-bound publication date.

        Returns
        -------
        CoveredPapers
            The papers, always labelled ``PARTIAL``.
        """
        papers = self._fetch_papers_uninstrumented(
            query, max_papers, progress_callback, since, until
        )
        detail = UNINSTRUMENTED_DETAIL
        if max_papers is not None and len(papers) >= max_papers:
            detail = f"{detail}; also stopped at the caller's cap of {max_papers}"
        return CoveredPapers(
            papers=tuple(papers),
            coverage=SourceCoverage.partial(self.name, len(papers), detail=detail),
        )
