"""BASE (Bielefeld Academic Search Engine): registered, and never consulted.

BASE is in the specification as a **degraded** source, and this module is the
degraded label made into code.  It is registered as a database and it reports
:data:`~lit_search.core.coverage.CoverageStatus.SKIPPED_NO_CREDENTIAL` on every
search.  It issues no request, because there is no route this project can
currently drive and because writing a parser for a response nobody here has
seen would be a guess dressed as a connector.

**Every route was probed live on 2026-08-31 from this machine.**

=====================================================================  =====================================
Route                                                                   Result
=====================================================================  =====================================
``api.base-search.net/cgi-bin/BaseHttpSearchInterface.fcgi``            **HTTP 200** with
                                                                        ``<error>Access denied for IP
                                                                        address ... and user agent
                                                                        ...</error>``
``oai.base-search.net/cgi-bin/oai.cgi?verb=Identify`` (donor's route)   **HTTP 404**, body ``File not found``
``www.base-search.net/interface/opensearch/search?...``                 HTTP 200, an anti-bot
                                                                        proof-of-work interstitial
                                                                        ("Making sure you're not a bot!")
=====================================================================  =====================================

Three things follow, and each is the reason for a decision here.

**1. The search interface is IP-gated, and the refusal is HTTP 200.**  This is
the ``PORTING-GUIDE`` section 2 trap in its purest form: ``raise_for_status()``
is silent, the body is XML rather than the expected result envelope, and a port
that reached for ``findall(".//doc")`` would get ``[]`` and report BASE as
holding no matching paper.  Access is granted per source IP through BASE's own
registration form; until that exists the source is not merely failing, it is
**never asked**, which is what ``SKIPPED_NO_CREDENTIAL`` means -- determinate,
and fixable by configuration.

**2. The donor's route is gone.**  The donor
(``paper_search_mcp/academic_platforms/base_search.py``, subclassing its
``oaipmh.py``) harvests OAI-PMH and filters locally.  That endpoint now returns
404 to every verb.  A port of it would parse a plain-text ``File not found``
body, find no records, and report zero -- the same failure as CiteSeerX, which
this project already refused to port for exactly that reason.

**3. Even with access, BASE would be degraded rather than genuine.**  The
specification's reading is that it harvests OAI-PMH records and filters
locally over prefixes that can be enormous, so it can time out or return an
order-biased sample.  So its status could never be
:data:`~lit_search.core.coverage.CoverageStatus.COMPLETE` even after
registration.

**Why register a connector that cannot run.**  Because the alternative is
worse.  An unregistered source is absent from every coverage ledger, and "we
never asked BASE" then looks exactly like "BASE does not exist" -- which is the
registry-level version of the silence this fork exists to remove.  With the
connector registered, a search says ``base_search: skipped_no_credential`` and
names what to do about it.  SSRN and IACR are registered on the same argument.

**What is deliberately not here.**  There is no half-written query path behind
an opt-in flag.  The response shape of the authenticated interface has not been
observed, so any parser for it would be invented; a connector that invents a
parser fails by returning nothing when the real shape differs, and returning
nothing is the outcome this project treats as the enemy.  When an IP
registration exists, write the parser against a captured response and delete
:meth:`BaseSearchConnector._refuse` -- not before.
"""

from __future__ import annotations

import datetime
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from lit_search.query.builder import QueryBuilder

from lit_search.connectors.search_base import SearchConnectorBase
from lit_search.core.coverage import CoveredPapers
from lit_search.core.paper import Database
from lit_search.core.query import Query
from lit_search.exceptions import MissingApiKeyError
from lit_search.query.builders.base_search import BaseSearchQueryBuilder

logger = logging.getLogger(__name__)

#: Name of the precondition a caller must satisfy, phrased as the runner
#: reports it.  It is not an API key: BASE grants access per source IP address
#: through a registration form, so the thing to supply is a registration rather
#: than a secret.  ``SKIPPED_NO_CREDENTIAL`` is nonetheless the right status --
#: the source was never contacted, the reason is determinate, and configuration
#: fixes it.
CREDENTIAL_NAME = "base_search_ip_registration"

#: Where the registration is requested.  Named in the log line so a reader is
#: not left to search for it.
REGISTRATION_URL = "https://www.base-search.net/about/en/faq_data.php"

_MIN_REQUEST_INTERVAL = 2.0
"""Seconds between requests.

**Documented by the source, not inferred.**  ``www.base-search.net/robots.txt``
publishes ``Crawl-delay: 2`` for ``User-Agent: *`` (fetched 2026-08-31, file
dated 2025-09-16).  No request is currently made, so the value is unexercised;
it is declared correctly anyway so that the first request ever sent from this
connector is paced by BASE's own number rather than by whatever the author of
the query path happens to pick.
"""


class BaseSearchConnector(SearchConnectorBase):
    """Keep BASE visible in the coverage ledger as a source that was not asked.

    Parameters
    ----------
    query_builder : BaseSearchQueryBuilder | None
        Builder used to validate queries.
    """

    _credential_attrs: ClassVar[tuple[str, ...]] = ()
    """BASE issues no key; access is granted per source IP.  Declared
    explicitly so the empty tuple reads as checked rather than overlooked.
    """

    def __init__(self, query_builder: BaseSearchQueryBuilder | None = None) -> None:
        """Create a BASE searcher.

        Parameters
        ----------
        query_builder : BaseSearchQueryBuilder | None
            Builder used to validate queries.
        """
        super().__init__()
        self._query_builder: BaseSearchQueryBuilder = query_builder or BaseSearchQueryBuilder()

    @property
    def name(self) -> str:
        """Return the database identifier.

        Returns
        -------
        str
            Database name.
        """
        return Database.BASE_SEARCH.value

    @property
    def query_builder(self) -> QueryBuilder:
        """Return the BASE query builder.

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
            Two seconds, taken verbatim from BASE's published ``Crawl-delay``.
            See :data:`_MIN_REQUEST_INTERVAL`.
        """
        return _MIN_REQUEST_INTERVAL

    @property
    def rate_limit_service(self) -> str:
        """Return the quota authority these requests are paced against.

        Stated explicitly so that a future connector against any other
        ``base-search.net`` host shares this clock rather than doubling the
        rate against one operator.

        Returns
        -------
        str
            Quota-authority identifier.
        """
        return "base_search"

    def _refuse(self) -> None:
        """Refuse to contact BASE, naming what would have to change.

        Raises
        ------
        MissingApiKeyError
            Always.  The base class turns it into
            ``SKIPPED_NO_CREDENTIAL``, so BASE stays in the ledger as a source
            that was deliberately not consulted rather than as one that found
            nothing.
        """
        logger.warning(
            "Skipping BASE: its search interface answers HTTP 200 with "
            "'<error>Access denied for IP address ...</error>' until the requesting host is "
            "registered (%s), and the OAI-PMH route the donor connector used now returns "
            "HTTP 404. No request was made, and no papers are being reported as absent.",
            REGISTRATION_URL,
        )
        raise MissingApiKeyError(CREDENTIAL_NAME)

    def _fetch_papers(
        self,
        query: Query,
        max_papers: int | None,
        progress_callback: Callable[[int, int | None], None] | None,
        since: datetime.date | None = None,
        until: datetime.date | None = None,
    ) -> CoveredPapers:
        """Report that BASE was not consulted.

        Parameters
        ----------
        query : Query
            Validated query object.  Unused: no request is made.
        max_papers : int | None
            Unused.
        progress_callback : Callable[[int, int | None], None] | None
            Unused.
        since : datetime.date | None
            Unused.
        until : datetime.date | None
            Unused.

        Returns
        -------
        CoveredPapers
            Never returns; the signature is the coverage contract's.

        Raises
        ------
        MissingApiKeyError
            Always.  See :meth:`_refuse`.
        """
        self._refuse()
        raise AssertionError("unreachable: _refuse always raises")  # pragma: no cover
