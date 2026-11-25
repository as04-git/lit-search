"""Abstract base class for academic database search connectors.

Extends :class:`~lit_search.connectors.connector_base.ConnectorBase` with the
search-specific contract: query validation, paper fetching, and the public
:meth:`SearchConnectorBase.search` entry point used by the search runner.

**The coverage contract (R7).**  A connector may not return papers alone.
:meth:`SearchConnectorBase._fetch_papers` returns a
:class:`~lit_search.core.coverage.CoveredPapers`, which pairs the records with
an explicit :class:`~lit_search.core.coverage.SourceCoverage` saying how
complete they are and why.  Three independent mechanisms enforce this:

1. **Static.**  The abstract method is annotated ``-> CoveredPapers`` and
   :meth:`SearchConnectorBase.search` is ``@final``, so mypy rejects a
   connector that returns a list or overrides the entry point.
2. **Class-definition time.**  The check resolves ``_fetch_papers`` and
   ``search`` **through the MRO** and raises
   :class:`~lit_search.exceptions.CoverageContractError` if the implementation
   that would actually run has a missing or wrong return annotation, or if
   anything ahead of :class:`SearchConnectorBase` supplies ``search``.  A
   connector that forgets fails on ``import``, before a single request is made.
   It runs from **two** places -- ``__init_subclass__`` and the metaclass
   :class:`_CoverageContractMeta` -- for the reason given below.
3. **Runtime.**  :meth:`SearchConnectorBase.search` type-checks what
   ``_fetch_papers`` handed back, catching a body that ignores its own
   annotation.  The search runner repeats that check on whatever its connector
   objects return, because an object can be replaced without ever running
   ``search``.

Resolving through the MRO rather than through ``cls.__dict__`` is the fix for
a bypass that worked: a mixin listed before the base supplies ``_fetch_papers``
and wins method resolution, while a ``__dict__``-only check looks at the
connector class, finds nothing, and concludes the method was inherited from the
validated abstract base.  The observed result was ``type=list len=2
coverage=<none>``.

**Why the check also lives on the metaclass.**  ``__init_subclass__`` is a
cooperative hook: it runs on the *nearest* definition, and a subclass that
declares its own and declines to call ``super().__init_subclass__()`` silently
switches the enforcement off for everything beneath it.  That is not
hypothetical -- a two-class probe (a conforming connector that hijacks the hook,
then a child of it) produced ``type=list len=1 coverage=<none>`` from a call to
the ``@final`` :meth:`SearchConnectorBase.search`, which is the same escape the
mixin route gave.  A metaclass ``__init__`` has no such opt-out: it runs for
every class in the hierarchy no matter what the class body contains, so the
enforcement is no longer something a subclass can decline.

The remaining way out is to declare a custom metaclass deriving from
:class:`_CoverageContractMeta` and override ``__init__`` without calling
``super()``.  Python offers no defence against that short of refusing foreign
metaclasses outright, and it is a far louder act than forgetting a ``super()``
call in a hook most authors never think about.  It is named here rather than
left to be discovered.

**The contract is now checked, not only declared.**  Making silence
inexpressible does not make speech true: four connectors were built that
satisfy every layer above and report ``COMPLETE`` while lying.
:meth:`SearchConnectorBase.search` therefore collects evidence the connector
did not author -- failed HTTP requests, records it said it saw, totals the
source published -- and hands the claim to
:func:`~lit_search.core.retrieval_audit.verify_coverage_claim`, which
downgrades a ``COMPLETE`` the evidence contradicts.  See
:mod:`lit_search.core.retrieval_audit`.

To register a connector that has not yet been instrumented for coverage, see
:class:`~lit_search.connectors.uninstrumented_base.UninstrumentedSearchConnectorBase`.
"""

from __future__ import annotations

import datetime
import logging
from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, final, get_type_hints

import requests

if TYPE_CHECKING:
    from lit_search.core.query import Query
    from lit_search.query.builder import QueryBuilder, QueryValidationResult

from lit_search.connectors.connector_base import ConnectorBase
from lit_search.core.coverage import CoveredPapers, SourceCoverage
from lit_search.core.retrieval_audit import auditing, verify_coverage_claim
from lit_search.exceptions import (
    ConnectorError,
    CoverageContractError,
    MissingApiKeyError,
    UnsupportedQueryError,
)

logger = logging.getLogger(__name__)

QUERY_COMBINATIONS_WARNING_THRESHOLD = 20

# Name of the method that carries the coverage contract, and the simple name
# its return annotation must resolve to.
_CONTRACT_METHOD = "_fetch_papers"
_CONTRACT_RETURN_NAME = CoveredPapers.__name__

# Methods a subclass must not override: they are the enforcement points.
_FINAL_METHODS = ("search",)

_CONTRACT_HELP = (
    "Search connectors must return papers together with an explicit coverage "
    "status. Annotate and return "
    "'lit_search.core.coverage.CoveredPapers', built with one of the "
    "SourceCoverage constructors (complete/capped/partial/query_unsupported/"
    "skipped_no_credential/failed, or from_pagination). A connector that has "
    "not been instrumented yet should subclass "
    "UninstrumentedSearchConnectorBase instead of claiming a status it cannot "
    "justify."
)


def _unwrap(attr: object) -> Any:
    """Return the plain function behind a possible descriptor.

    Parameters
    ----------
    attr : object
        Raw value taken from a class ``__dict__``.

    Returns
    -------
    Any
        The underlying function, or *attr* unchanged.
    """
    return getattr(attr, "__func__", attr)


def _return_annotation_matches(func: Any) -> bool:
    """Whether *func*'s return annotation is the coverage contract type.

    Handles both eagerly evaluated annotations and the string form produced
    by ``from __future__ import annotations``.  A string is evaluated in the
    defining module's namespace, so import aliases resolve correctly, and the
    resolved object is compared by identity.

    Only when evaluation *fails* -- typically because the name is imported
    solely under ``TYPE_CHECKING`` -- does the check fall back to comparing the
    final dotted component of the annotation string.  That fallback is
    deliberately unreachable when evaluation succeeds: a name that resolves to
    some other class called ``CoveredPapers`` must be rejected here rather than
    left to the runtime ``isinstance`` check downstream.

    Parameters
    ----------
    func : Any
        Function to inspect.

    Returns
    -------
    bool
        ``True`` when the annotation denotes
        :class:`~lit_search.core.coverage.CoveredPapers`.
    """
    annotations = getattr(func, "__annotations__", {})
    if "return" not in annotations:
        return False
    annotation = annotations["return"]
    if annotation is CoveredPapers:
        return True
    if not isinstance(annotation, str):
        return False
    try:
        resolved: Any = get_type_hints(
            func,
            globalns=getattr(func, "__globals__", {}),
            localns=None,
        )["return"]
    except (KeyError, NameError, TypeError):
        # Unresolvable: fall back to the name, which is all there is to go on.
        return annotation.strip().rsplit(".", 1)[-1] == _CONTRACT_RETURN_NAME
    return resolved is CoveredPapers


def _defining_class(cls: type, method_name: str) -> type | None:
    """Return the class in *cls*'s MRO that supplies *method_name*.

    This is the class whose implementation would actually run, which is the
    only one worth validating.  Looking in ``cls.__dict__`` alone answers a
    different and much weaker question -- "did *this* class body define it" --
    and a mixin ahead of the base in the MRO answers it "no" while supplying
    the implementation.

    Parameters
    ----------
    cls : type
        Class being defined.
    method_name : str
        Attribute to resolve.

    Returns
    -------
    type | None
        The defining class, or ``None`` when nothing in the MRO defines it.
    """
    for ancestor in cls.__mro__:
        if method_name in ancestor.__dict__:
            return ancestor
    return None


def _enforce_coverage_contract(cls: type, base: type) -> None:
    """Reject a subclass that can express silence.

    Both checks resolve through the MRO rather than through ``cls.__dict__``,
    because what matters is the implementation that will run, not where it was
    typed.

    Parameters
    ----------
    cls : type
        The subclass being defined.
    base : type
        :class:`SearchConnectorBase` itself, which is the only class permitted
        to supply the final methods.

    Raises
    ------
    CoverageContractError
        If anything ahead of *base* in the MRO supplies a final enforcement
        method, or if the ``_fetch_papers`` implementation that would run
        lacks the contract return annotation.
    """
    for method_name in _FINAL_METHODS:
        owner = _defining_class(cls, method_name)
        if owner is not None and owner is not base:
            raise CoverageContractError(
                f"{cls.__module__}.{cls.__qualname__} resolves '{method_name}' to "
                f"{owner.__module__}.{owner.__qualname__}, but it is final on "
                f"SearchConnectorBase. It is the point at which coverage is validated "
                f"and stamped; supplying it -- in the class body or through a mixin "
                f"earlier in the MRO -- would let a connector return results with no "
                f"coverage. Implement '{_CONTRACT_METHOD}' instead."
            )

    owner = _defining_class(cls, _CONTRACT_METHOD)
    if owner is None or owner is base:
        # Still the abstract declaration on the base: nothing to validate, and
        # the class cannot be instantiated until something supplies it.
        return
    func = _unwrap(owner.__dict__[_CONTRACT_METHOD])
    if not _return_annotation_matches(func):
        found = getattr(func, "__annotations__", {}).get("return", "<no return annotation>")
        via = (
            ""
            if owner is cls
            else f" (resolved through the MRO to {owner.__module__}.{owner.__qualname__})"
        )
        raise CoverageContractError(
            f"{cls.__module__}.{cls.__qualname__}.{_CONTRACT_METHOD}{via} is annotated "
            f"'{found}', but the coverage contract requires "
            f"'{_CONTRACT_RETURN_NAME}'. {_CONTRACT_HELP}"
        )


class _CoverageContractMeta(ABCMeta):
    """Metaclass that enforces the coverage contract on every subclass.

    This exists because ``__init_subclass__`` can be switched off from inside
    the hierarchy it guards.  Python calls only the *nearest*
    ``__init_subclass__`` in the MRO, so an intermediate class that defines one
    and omits ``super().__init_subclass__(**kwargs)`` disables the check for
    every class beneath it -- and that intermediate class is itself perfectly
    conforming, so nothing about it looks wrong.

    A metaclass ``__init__`` runs for every class built with it, whatever the
    class body says, and is inherited automatically.  It is therefore the layer
    a subclass cannot decline.

    Derives from :class:`abc.ABCMeta` because the connector hierarchy is
    abstract; using a plain ``type`` here would be a metaclass conflict.
    """

    def __init__(
        cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs: Any
    ) -> None:
        """Enforce the coverage contract as the class object is finalised.

        Parameters
        ----------
        name : str
            Name of the class being created.
        bases : tuple[type, ...]
            Its bases.
        namespace : dict[str, Any]
            Its class body namespace.
        **kwargs : Any
            Forwarded to ``ABCMeta.__init__``.

        Raises
        ------
        CoverageContractError
            If the class violates the contract.
        """
        super().__init__(name, bases, namespace, **kwargs)
        # SearchConnectorBase itself is still being built the first time this
        # runs, so there is nothing to validate against yet.
        base = globals().get("SearchConnectorBase")
        if base is not None and cls is not base:
            _enforce_coverage_contract(cls, base)


class SearchConnectorBase(ConnectorBase, metaclass=_CoverageContractMeta):
    """Abstract base class for academic database search connectors.

    Subclasses implement the search logic for a specific database and receive
    a database-specific ``QueryBuilder`` via dependency injection.  Each
    connector is responsible for:

    1. Validating the query against the database capabilities.
    2. Converting the query to the database-specific format.
    3. Executing HTTP requests with proper rate limiting (inherited from
       :class:`~lit_search.connectors.connector_base.ConnectorBase`).
    4. Parsing the API responses into :class:`~lit_search.core.paper.Paper`
       objects.
    5. **Reporting how complete the result is**, as a
       :class:`~lit_search.core.coverage.SourceCoverage` returned alongside
       the papers.  There is no way to skip this step: see the module
       docstring.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Enforce the coverage contract when a subclass is defined.

        Parameters
        ----------
        **kwargs : Any
            Forwarded to ``super().__init_subclass__``.

        Raises
        ------
        CoverageContractError
            If the subclass violates the contract.
        """
        super().__init_subclass__(**kwargs)
        _enforce_coverage_contract(cls, SearchConnectorBase)

    @property
    @abstractmethod
    def query_builder(self) -> QueryBuilder:
        """Return the database-specific query builder.

        Returns
        -------
        QueryBuilder
            Builder used to validate and convert queries.
        """

    @abstractmethod
    def _fetch_papers(
        self,
        query: Query,
        max_papers: int | None,
        progress_callback: Callable[[int, int | None], None] | None,
        since: datetime.date | None = None,
        until: datetime.date | None = None,
    ) -> CoveredPapers:
        """Fetch papers from the database, together with their coverage.

        Subclasses implement HTTP requests, rate limiting, pagination and
        response parsing here, and must report what the retrieval actually
        achieved.  Returning an empty result is allowed; returning an empty
        result *without saying why it is empty* is not, because the return
        type has no way to express it.

        Guidance on choosing a status:

        - Use :meth:`~lit_search.core.coverage.SourceCoverage.from_pagination`
          when the source reports a result total.  It applies the cap /
          shortfall / unverifiable decision procedure for you.
        - Use
          :meth:`~lit_search.core.coverage.SourceCoverage.complete` only when
          the connector can justify the claim.
        - A source that cannot express the query (for example a preprint
          server that only filters by category) reports
          :meth:`~lit_search.core.coverage.SourceCoverage.query_unsupported`,
          or :meth:`~lit_search.core.coverage.SourceCoverage.partial` when it
          answers a degraded form of the query.
        - Records that were seen but failed to parse are a coverage loss:
          report :meth:`~lit_search.core.coverage.SourceCoverage.partial`
          rather than silently returning fewer papers.

        Parameters
        ----------
        query : Query
            Pre-validated query object.
        max_papers : int | None
            Maximum papers to retrieve.  ``None`` means unlimited.
        progress_callback : Callable[[int, int | None], None] | None
            Optional callback invoked after each page / item with
            ``(items_processed, total_or_none)``.  ``items_processed`` counts
            every candidate item attempted (regardless of whether it was
            successfully parsed), so the bar always reaches ``total`` even
            when some items fail to parse.
        since : datetime.date | None
            Only return papers published on or after this date.
        until : datetime.date | None
            Only return papers published on or before this date.

        Returns
        -------
        CoveredPapers
            Retrieved papers and the coverage status that describes them.

        Raises
        ------
        Exception
            Implementations may raise on network or parsing failures;
            :meth:`search` converts the known failure classes into an
            explicit coverage status.
        """

    @final
    def search(
        self,
        query: Query,
        max_papers: int | None = None,
        progress_callback: Callable[[int, int | None], None] | None = None,
        since: datetime.date | None = None,
        until: datetime.date | None = None,
    ) -> CoveredPapers:
        """Execute a search and return papers together with their coverage.

        This method never returns a bare sequence and never converts a
        failure into an empty success.  Every outcome it knows how to name is
        returned as an explicit status:

        =============================  ==================================
        Situation                      Status
        =============================  ==================================
        Query rejected by the builder  ``QUERY_UNSUPPORTED``
        Credential missing             ``SKIPPED_NO_CREDENTIAL``
        Network / connector error      ``FAILED``
        Anything else                  whatever ``_fetch_papers`` reported,
                                       **after verification**
        =============================  ==================================

        Exceptions outside those classes are **not** caught.  An unexpected
        error is a bug in the connector, and turning it into a coverage
        status would hide it behind a plausible-looking result.

        A ``COMPLETE`` reported by ``_fetch_papers`` is not taken at face
        value.  For the duration of the call this method opens a
        :class:`~lit_search.core.retrieval_audit.RetrievalAudit` and supplies
        its own progress callback, so that failed requests, the record count
        the connector reports, and the totals the source published are all
        recorded by base-class code.  ``COMPLETE`` is then downgraded when
        that evidence contradicts it -- never upgraded.

        Parameters
        ----------
        query : Query
            Parsed query object.
        max_papers : int | None
            Maximum number of papers to return.
        progress_callback : Callable[[int, int | None], None] | None
            Progress callback called as ``callback(current, total_or_none)``.
        since : datetime.date | None
            Only return papers published on or after this date.
        until : datetime.date | None
            Only return papers published on or before this date.

        Returns
        -------
        CoveredPapers
            Retrieved papers and this source's coverage status.  The
            coverage is always stamped with :attr:`name`, so ledger keys are
            the connector's registered identifier regardless of what the
            subclass passed.

        Raises
        ------
        CoverageContractError
            If ``_fetch_papers`` returned anything other than a
            :class:`~lit_search.core.coverage.CoveredPapers`.
        """
        validation: QueryValidationResult = self.query_builder.validate_query(query)

        if not validation.is_valid:
            message = validation.error_message or "query is incompatible with this database"
            logger.warning("Query unsupported on '%s': %s", self.name, message)
            return CoveredPapers.none(SourceCoverage.query_unsupported(self.name, detail=message))

        # The callback handed to _fetch_papers is always the base class's own,
        # never the caller's and never None.  It is the channel through which
        # a connector states how many records it has seen and what total the
        # source published, and both of those are evidence against a false
        # COMPLETE.  Passing None here -- which is what a caller with no
        # progress bar wants -- would let a connector skip the channel simply
        # by honouring its own 'if progress_callback is not None' guard.
        with auditing(self.name) as audit:

            def _record(items_processed: int, total: int | None) -> None:
                """Record a progress observation, then forward it.

                Parameters
                ----------
                items_processed : int
                    Candidate records seen so far.
                total : int | None
                    Total the source reported, if any.
                """
                audit.note_progress(items_processed, total)
                if progress_callback is not None:
                    progress_callback(items_processed, total)

            try:
                result = self._fetch_papers(query, max_papers, _record, since, until)
            except UnsupportedQueryError as exc:
                logger.warning("Query unsupported on '%s': %s", self.name, exc)
                return CoveredPapers.none(
                    SourceCoverage.query_unsupported(self.name, detail=str(exc))
                )
            except MissingApiKeyError as exc:
                logger.warning("Skipping '%s': %s", self.name, exc)
                return CoveredPapers.none(
                    SourceCoverage.skipped_no_credential(self.name, credential=str(exc))
                )
            except (requests.RequestException, ConnectorError) as exc:
                logger.warning("Error while searching '%s': %s", self.name, exc)
                logger.debug("Search exception details:", exc_info=True)
                return CoveredPapers.none(
                    SourceCoverage.failed(self.name, error=str(exc), error_type=type(exc).__name__)
                )

            if not isinstance(result, CoveredPapers):
                raise CoverageContractError(
                    f"{type(self).__module__}.{type(self).__qualname__}.{_CONTRACT_METHOD} "
                    f"returned {type(result).__name__}, not {_CONTRACT_RETURN_NAME}. "
                    f"{_CONTRACT_HELP}"
                )
            stamped = result.with_source(self.name)
            return verify_coverage_claim(stamped, max_papers=max_papers, audit=audit)
