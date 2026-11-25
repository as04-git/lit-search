"""arXiv bulk metadata connector, reading a local Kaggle snapshot.

This is the search face of :mod:`lit_search.loaders.arxiv_snapshot`.  It answers
a query by scanning a snapshot file rather than by calling arXiv, which buys
three things the live ``arxiv`` connector cannot offer: the whole corpus instead
of a paginated slice, an unrestricted local boolean grammar, and no rate limit.

It costs one thing, and the whole design of this module is about not letting
that cost be silent.

**A snapshot is stale by construction, and a stale corpus answers "no" for
recent work.**  The Kaggle dataset is republished weekly (verified: version 301
on 2026-08-29, 300 on 2026-08-22, 299 on 2026-08-15), so even a snapshot
downloaded the moment it appeared is up to a week behind, and one downloaded
three months ago is silent about a quarter of a year of arXiv.  A search that
reported that silence as ``COMPLETE`` would be the failure this fork exists to
prevent, in its purest form: no network to blame, no error to see, just a
confident empty answer about a paper that exists.

**So this connector never reports ``COMPLETE``, and the reason is structural
rather than cautious.**  ``COMPLETE`` is a claim that everything the source
could match was retrieved, and
:meth:`~lit_search.core.coverage.SourceCoverage.complete` requires a
*source-reported* total to back it -- precisely so that a connector cannot
count its own results and call the count a corroboration.  The snapshot reports
no total.  It is a file: it can be counted, but the count is this connector's
own arithmetic and corroborates nothing about arXiv.  Every result is
``PARTIAL``, and the detail names the dataset, the version, the date, what that
date is a date *of*, and how many records were read.

**The snapshot's provenance is carried into the result, not logged and
forgotten.**  See :class:`~lit_search.loaders.arxiv_snapshot.SnapshotProvenance`.
Where no sidecar manifest exists the only date available is the file's
modification time, and the detail says exactly that rather than presenting it as
a snapshot date.

**Missing snapshot is ``SKIPPED_NO_CREDENTIAL``, not zero papers.**  The
snapshot is a precondition supplied by configuration, which is what that status
means -- the same shape as IACR's whitelisting acknowledgement and BASE's IP
registration.  Registering the database and skipping it keeps "we did not
consult the local arXiv corpus" sayable; leaving the connector unregistered
would make that indistinguishable from the corpus not existing.
"""

from __future__ import annotations

import datetime
import logging
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from lit_search.query.builder import QueryBuilder

from lit_search.connectors.search_base import SearchConnectorBase
from lit_search.core.author import Author
from lit_search.core.coverage import CoveredPapers, SourceCoverage
from lit_search.core.paper import Database, Paper, PaperType
from lit_search.core.query import Query
from lit_search.core.source import Source, SourceType
from lit_search.exceptions import ConnectorError, MissingApiKeyError
from lit_search.loaders.arxiv_snapshot import (
    ARXIV_ML_CATEGORIES,
    SNAPSHOT_PATH_ENV_VAR,
    ArxivRecord,
    SnapshotProvenance,
    SnapshotScan,
    SnapshotUnavailableError,
    iter_records,
    resolve_snapshot_path,
)
from lit_search.query.builders.arxiv_bulk import ArxivBulkQueryBuilder

logger = logging.getLogger(__name__)

#: Interval between requests.  There are none: this connector opens a local
#: file and never reaches the network, so there is no remote quota to pace
#: against and no host to be polite to.
#:
#: ``0.0`` is a legal declaration and the shared limiter treats it as an
#: exemption.  It is recorded here rather than left implicit because a reader
#: comparing this connector against the others will otherwise wonder whether
#: the rate was forgotten; it was not, and the reason it is zero is checkable
#: from the code -- no ``_get`` or ``_post`` call exists in this module.
_MIN_REQUEST_INTERVAL = 0.0

#: Safety ceiling on records handed back from one search when the caller sets
#: no ``max_papers``.  A query matching a large slice of 605,681 records would
#: otherwise return a set nothing downstream can re-rank in reasonable time.
#: Hitting it is reported, never silent.
_DEFAULT_MAX_PAPERS = 5_000


@dataclass(slots=True)
class _Scan:
    """Bookkeeping for one pass over the snapshot.

    Attributes
    ----------
    papers : list[Paper]
        Records that matched the query and the date bounds, in file order.
    matched : int
        Records that satisfied the query and the date bounds, including those
        beyond the cap.
    counters : SnapshotScan
        Read counters from the loader: lines, unreadable lines, unparsed
        records, in-scope records.
    stopped_at_cap : int | None
        The cap that ended the scan, when one did.
    """

    papers: list[Paper] = field(default_factory=list)
    matched: int = 0
    counters: SnapshotScan = field(default_factory=SnapshotScan)
    stopped_at_cap: int | None = None


class ArxivBulkConnector(SearchConnectorBase):
    """Search a local arXiv metadata snapshot.

    Parameters
    ----------
    snapshot_path : str | None
        Snapshot file to read.  Falls back to
        :data:`~lit_search.loaders.arxiv_snapshot.SNAPSHOT_PATH_ENV_VAR`.
    categories : frozenset[str] | None
        arXiv categories to consider.  Defaults to
        :data:`~lit_search.loaders.arxiv_snapshot.ARXIV_ML_CATEGORIES`, the six
        the owner asked for.  Pass ``frozenset()`` to search the whole
        snapshot; the restriction is reported in the coverage detail either
        way, because a category filter is a recall boundary and an unstated
        one is the terminological failure again.
    query_builder : ArxivBulkQueryBuilder | None
        Builder used to validate queries and compile the local predicate.
    """

    _credential_attrs: ClassVar[tuple[str, ...]] = ()
    """No credential: the snapshot is a local file and no request is made.
    Declared explicitly, as the porting guide requires, so that the empty tuple
    reads as checked rather than overlooked.
    """

    def __init__(
        self,
        snapshot_path: str | None = None,
        categories: frozenset[str] | None = None,
        query_builder: ArxivBulkQueryBuilder | None = None,
    ) -> None:
        """Create a bulk arXiv searcher.

        The absent snapshot is deliberately **not** raised here.  Raising in
        the constructor would leave the runner with no connector object and
        only a class-level note that something was skipped; raising inside
        :meth:`_fetch_papers` lets the base class turn it into a coverage entry
        that names the source and names what to configure.

        Parameters
        ----------
        snapshot_path : str | None
            Snapshot file, or ``None`` to read the environment.
        categories : frozenset[str] | None
            Categories to restrict the scan to.
        query_builder : ArxivBulkQueryBuilder | None
            Builder used to validate queries and compile the predicate.
        """
        super().__init__()
        self._configured_path = snapshot_path
        self._categories = ARXIV_ML_CATEGORIES if categories is None else categories
        self._query_builder: ArxivBulkQueryBuilder = query_builder or ArxivBulkQueryBuilder()

    @property
    def name(self) -> str:
        """Return the database identifier.

        Returns
        -------
        str
            Database name.
        """
        return Database.ARXIV_BULK.value

    @property
    def query_builder(self) -> QueryBuilder:
        """Return the snapshot query builder.

        Returns
        -------
        QueryBuilder
            The underlying builder instance.
        """
        return self._query_builder

    @property
    def min_request_interval(self) -> float:
        """Return the minimum seconds between HTTP requests.

        Returns
        -------
        float
            Zero.  See :data:`_MIN_REQUEST_INTERVAL` for why that is a
            statement rather than an omission.
        """
        return _MIN_REQUEST_INTERVAL

    @property
    def rate_limit_service(self) -> str:
        """Return the quota authority these requests are paced against.

        Deliberately **not** ``"arxiv"``.  Sharing a scope with the live arXiv
        API connector would put a local file scan into the bucket that paces
        arXiv's three-second interval, so a long snapshot scan would appear in
        the wait statistics as arXiv contention and, worse, a future change
        that gave this connector a non-zero interval would throttle the API
        connector along with it.

        Returns
        -------
        str
            Quota-authority identifier.
        """
        return Database.ARXIV_BULK.value

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _to_paper(self, record: ArxivRecord) -> Paper:
        """Build a :class:`Paper` that keeps both of the record's identities.

        This is the point at which paperscraper's normalisation loses half the
        identity, so it is the point at which this project must not.
        :attr:`~lit_search.core.paper.Paper.doi` takes the *citable* DOI --
        the publisher's when the record names one -- and
        :attr:`~lit_search.core.paper.Paper.preprint_doi` takes the arXiv DOI
        whenever that is a second identity rather than the same one twice.

        Parameters
        ----------
        record : ArxivRecord
            Normalised snapshot record.

        Returns
        -------
        Paper
            Paper carrying both DOIs, the arXiv URLs, and the categories as
            subjects.
        """
        published = record.publisher_doi is not None
        return Paper(
            title=record.title,
            abstract=record.abstract,
            authors=[Author(name=name) for name in record.authors],
            source=Source(
                title=record.journal_ref or "arXiv",
                source_type=(SourceType.JOURNAL if record.journal_ref else SourceType.REPOSITORY),
            ),
            publication_date=record.submitted,
            url=record.abs_url,
            pdf_url=record.pdf_url,
            doi=record.citable_doi,
            preprint_doi=record.preprint_doi,
            keywords=set(record.categories) or None,
            subjects=set(record.categories),
            comments=record.comments,
            found_in={self.name},
            paper_type=(PaperType.ARTICLE if published else PaperType.UNPUBLISHED),
            # Every arXiv record is openly readable.
            is_open_access=True,
            language="en",
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _locate_snapshot(self) -> str:
        """Return the snapshot to read, distinguishing absent from wrong.

        The two cases must not collapse.  Nobody having configured a snapshot
        is a **skip**: an optional corpus was not set up, and the ledger should
        say so and name the variable.  A snapshot having been configured and
        not being there is a **failure**: somebody meant to search it, and
        reporting that as "no snapshot configured" turns a typo in a path into
        a source that was never expected to run -- which reads as normal and
        is not.

        Returns
        -------
        str
            Path of a readable snapshot file.

        Raises
        ------
        MissingApiKeyError
            When nothing is configured.  Becomes ``SKIPPED_NO_CREDENTIAL``.
        ConnectorError
            When something is configured and cannot be resolved to a readable
            file.  Becomes ``FAILED``.
        """
        path = resolve_snapshot_path(self._configured_path)
        if path is not None:
            return path
        configured = self._configured_path or os.environ.get(SNAPSHOT_PATH_ENV_VAR, "").strip()
        if not configured:
            logger.warning(
                "Skipping the local arXiv snapshot: set %s to the Kaggle "
                "'arxiv-metadata-oai-snapshot.json' file to search it.",
                SNAPSHOT_PATH_ENV_VAR,
            )
            raise MissingApiKeyError(SNAPSHOT_PATH_ENV_VAR)
        raise ConnectorError(
            f"The configured arXiv snapshot '{configured}' is not a readable file, and is "
            f"not a directory holding exactly one '*.json' file. Nothing was searched, so "
            f"reporting zero papers would say arXiv holds nothing on the subject."
        )

    def _fetch_papers(
        self,
        query: Query,
        max_papers: int | None,
        progress_callback: Callable[[int, int | None], None] | None,
        since: datetime.date | None = None,
        until: datetime.date | None = None,
    ) -> CoveredPapers:
        """Scan the snapshot and return the records the query matches.

        Parameters
        ----------
        query : Query
            Validated query object.
        max_papers : int | None
            Maximum papers to retrieve.  ``None`` falls back to
            :data:`_DEFAULT_MAX_PAPERS`, and the fallback is reported.
        progress_callback : Callable[[int, int | None], None] | None
            Progress callback called as ``callback(records_seen, None)``.  The
            total is always ``None``: the snapshot publishes no count, and a
            count this connector computed itself would corroborate nothing.
        since : datetime.date | None
            Only return papers first submitted on or after this date.
        until : datetime.date | None
            Only return papers first submitted on or before this date.

        Returns
        -------
        CoveredPapers
            Matching papers and the coverage status describing them.  Never
            ``COMPLETE``; see the module docstring.

        Raises
        ------
        MissingApiKeyError
            When no snapshot is configured at all.
        ConnectorError
            When a snapshot is configured but cannot be resolved, opened or
            read.  See :meth:`_locate_snapshot` for why those are different.
        """
        path = self._locate_snapshot()

        provenance = SnapshotProvenance.for_file(path)
        cap = max_papers if max_papers is not None else _DEFAULT_MAX_PAPERS
        matches = self._query_builder.compile_predicate(query)
        run = _Scan()

        try:
            self._scan(run, path, matches, cap, since, until, progress_callback)
        except SnapshotUnavailableError as exc:
            # Nothing retrieved and nothing retrievable: re-raise so the base
            # class records FAILED rather than letting an unreadable file
            # become an arXiv that holds no matching paper.
            if not run.papers:
                raise ConnectorError(str(exc)) from exc
            raise

        if progress_callback is not None:
            progress_callback(run.counters.lines, None)

        return CoveredPapers(
            papers=tuple(run.papers),
            coverage=self._build_coverage(
                run=run,
                provenance=provenance,
                max_papers=max_papers,
                cap=cap,
                date_bounded=since is not None or until is not None,
            ),
        )

    def _scan(
        self,
        run: _Scan,
        path: str,
        matches: Callable[[dict[str, tuple[str, ...]]], bool],
        cap: int,
        since: datetime.date | None,
        until: datetime.date | None,
        progress_callback: Callable[[int, int | None], None] | None,
    ) -> None:
        """Walk the snapshot, folding matching records into *run*.

        Parameters
        ----------
        run : _Scan
            Bookkeeping, updated in place.
        path : str
            Snapshot file.
        matches : Callable[[dict[str, tuple[str, ...]]], bool]
            Compiled query predicate.
        cap : int
            Maximum papers to keep.
        since : datetime.date | None
            Lower bound on the submission date.
        until : datetime.date | None
            Upper bound on the submission date.
        progress_callback : Callable[[int, int | None], None] | None
            Progress callback.
        """
        stream: Iterable[ArxivRecord] = iter_records(
            path, categories=self._categories or None, scan=run.counters
        )
        for record in stream:
            if not _within(record, since, until) or not matches(record.as_match_record()):
                continue
            run.matched += 1
            if len(run.papers) >= cap:
                run.stopped_at_cap = cap
                break
            run.papers.append(self._to_paper(record))
            if progress_callback is not None and run.counters.lines % 50_000 == 0:
                progress_callback(run.counters.lines, None)

    def _build_coverage(
        self,
        *,
        run: _Scan,
        provenance: SnapshotProvenance,
        max_papers: int | None,
        cap: int,
        date_bounded: bool,
    ) -> SourceCoverage:
        """Describe what the scan achieved, never claiming completeness.

        Not ``CAPPED`` either, even when the cap ended the scan.  ``CAPPED``
        ranks as a *milder* outcome than ``PARTIAL`` because it means a known
        amount is missing on purpose; here an unknown amount is missing
        whatever the cap did, since everything arXiv published after the
        snapshot was never consulted.  Reporting ``CAPPED`` would make a
        truncated scan read as healthier than an untruncated one.  The cap is
        recorded in the detail instead.

        Parameters
        ----------
        run : _Scan
            Bookkeeping for the finished scan.
        provenance : SnapshotProvenance
            Which snapshot was read.
        max_papers : int | None
            The caller's own limit, if any.
        cap : int
            The limit actually applied.
        date_bounded : bool
            Whether the caller asked for a date range.

        Returns
        -------
        SourceCoverage
            A ``PARTIAL`` record whose detail carries the snapshot provenance.
        """
        notes: list[str] = [
            f"answered from a local arXiv metadata snapshot ({provenance.describe()}), so "
            "every paper posted or revised after that date is absent from this result and "
            "its absence is not evidence that arXiv holds no such paper",
            f"read {run.counters.lines} snapshot records, of which {run.counters.in_scope} "
            f"were in scope and {run.matched} matched the query",
        ]
        if self._categories:
            notes.append(
                "the scan was restricted to arXiv categories "
                f"{', '.join(sorted(self._categories))}; a matching paper filed only under "
                "another category was never examined"
            )
        if run.stopped_at_cap is not None:
            limit_source = "the caller's limit" if max_papers is not None else "the default limit"
            notes.append(
                f"the scan stopped after {run.stopped_at_cap} papers at {limit_source}, while "
                f"at least {run.matched} records had already matched"
            )
        if date_bounded:
            notes.append(
                "the requested date range was applied to each record's first-submission "
                "date, which is the only submission date the snapshot carries; a paper "
                "whose later version falls in range is judged by its v1 date"
            )
        if run.counters.lost:
            notes.append(
                f"{run.counters.lost} of {run.counters.lines} snapshot lines could not be "
                f"read as records ({run.counters.unreadable} were not valid JSON objects, "
                f"{run.counters.unparsed} carried no usable identifier or title)"
            )

        return SourceCoverage.partial(self.name, len(run.papers), detail="; ".join(notes))


def _within(
    record: ArxivRecord,
    since: datetime.date | None,
    until: datetime.date | None,
) -> bool:
    """Whether a record's submission date falls inside the requested bounds.

    A record with no usable date is **kept**, not dropped.  Dropping it would
    remove a real paper on the strength of a missing field, and the loss would
    be invisible in a result whose whole point is completeness; keeping it can
    at worst return a paper slightly outside the range, which the reader can
    see.

    Parameters
    ----------
    record : ArxivRecord
        Record to test.
    since : datetime.date | None
        Lower bound, inclusive.
    until : datetime.date | None
        Upper bound, inclusive.

    Returns
    -------
    bool
        ``True`` when the record is in range.
    """
    if record.submitted is None:
        return True
    if since is not None and record.submitted < since:
        return False
    return not (until is not None and record.submitted > until)
