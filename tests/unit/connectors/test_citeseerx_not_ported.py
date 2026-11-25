"""CiteSeerX is deliberately not a connector. This records why, as a test.

CiteSeerX was on the original port list and was dropped.  A dropped connector
leaves no trace in the tree, and an absence cannot be distinguished from an
oversight: the donor
(``paper_search_mcp/academic_platforms/citeseerx.py`` at pinned commit
``234678ab231074a7977320978ee0496dcdaddd1f``) is still sitting there looking
portable, and the next person to work through the list has nothing to tell them
it was considered.  So the decision is written down where it runs.

**The service is retired.**  Re-verified live on 2026-08-31, independently of
the project specification's 2026-08-30 check:

===========================================  ======================================
Request                                      Result
===========================================  ======================================
``GET https://citeseerx.ist.psu.edu/``       ``301`` -> ``web.archive.org`` snapshot
``GET .../api/search?q=test``                ``301`` -> ``web.archive.org`` snapshot
``GET .../search?q=test``                    ``301`` -> ``web.archive.org`` snapshot
===========================================  ======================================

Every path redirects to
``https://web.archive.org/web/20251230112235/https://citeseerx.ist.psu.edu/...``,
and following the redirect returns the Internet Archive's Wayback Machine page,
not a search result.

**Why that makes a port actively harmful rather than merely useless.**  The
donor's ``search`` catches its failures and returns the empty list it started
with.  Ported mechanically, every CiteSeerX search would report "this database
holds no matching paper" -- a plausible, well-formed, confident, incomplete
answer, which is the precise failure this fork exists to prevent.  A port
written *correctly* against the coverage contract would be no better in
practice: it would report ``FAILED`` on every query forever, which is a source
that costs a request, a rate-limit slot and a permanently degraded aggregate
status while never returning a paper.

**What would reopen this.**  CiteSeerX serving search results from its own
host again, rather than redirecting to an archived snapshot.  Re-run the three
requests above; if they stop returning ``301`` to ``web.archive.org``, the drop
is worth revisiting.
"""

from __future__ import annotations

from lit_search.connectors import SEARCH_REGISTRY
from lit_search.core.paper import Database


class TestCiteSeerXIsNotRegistered:
    """The drop is a decision, and decisions are asserted rather than assumed."""

    def test_no_database_member_exists_for_citeseerx(self) -> None:
        """A member nobody can construct cannot be requested by name.

        Asserted against the member *names* rather than a count of them, so
        that adding an unrelated database does not break this test and teach
        the next porter that this file is bookkeeping.
        """
        names = {member.name for member in Database}
        values = {member.value for member in Database}
        assert "CITESEERX" not in names
        assert "citeseerx" not in values

    def test_not_in_the_search_registry(self) -> None:
        """Nothing selectable maps to the retired service."""
        registered = {database.value for database in SEARCH_REGISTRY}
        assert "citeseerx" not in registered

    def test_no_connector_module_was_added(self) -> None:
        """The module is absent, not present-and-disabled.

        A disabled connector is worse than none: it reads as a source that
        could be switched on, when the service behind it no longer exists.
        """
        import importlib.util

        assert importlib.util.find_spec("lit_search.connectors.citeseerx") is None
