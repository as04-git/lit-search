"""Registry-wide invariants for the public search configuration facade.

A connector is not actually usable merely because it appears in
``SEARCH_REGISTRY``.  Every constructor option that changes what it can reach
must either be exposed by :class:`SearchRunner` or be deliberately classified
as having no public configuration.  These tests make that classification
exhaustive, then exercise the forwarding path independently of any live API.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lit_search.connectors import SEARCH_REGISTRY
from lit_search.core.coverage import CoverageStatus
from lit_search.core.paper import Database
from lit_search.exceptions import MissingApiKeyError
from lit_search.runners.search_runner import (
    CREDENTIALLED_DEFAULT_SEARCH_DATABASES,
    DEFAULT_SEARCH_DATABASES,
    SEARCH_CONNECTOR_CONFIG_PARAMETERS,
    SearchRunner,
)

_EXPECTED_CONFIG_PARAMETERS: dict[Database, tuple[str, ...]] = {
    Database.ARXIV: (),
    Database.ARXIV_BULK: ("arxiv_snapshot_path", "arxiv_categories"),
    Database.BASE_SEARCH: (),
    Database.BIORXIV: (),
    Database.CHEMRXIV: ("chemrxiv_email",),
    Database.CORE: ("core_api_key",),
    Database.DBLP: (),
    Database.DOAJ: (),
    Database.EUROPE_PMC: (),
    Database.GOOGLE_SCHOLAR: (),
    Database.HAL: (),
    Database.IACR: (),
    Database.IEEE: ("ieee_api_key",),
    Database.MEDRXIV: (),
    Database.OPENAIRE: (
        "openaire_access_token",
        "openaire_refresh_token",
        "openaire_access_token_expires_in",
    ),
    Database.OPENALEX: ("openalex_api_key", "email"),
    Database.OPENREVIEW: (
        "openreview_username",
        "openreview_password",
        "openreview_venue_ids",
        "openreview_cache_dir",
        "openreview_cache_ttl_seconds",
    ),
    Database.PMC: ("pubmed_api_key", "ncbi_tool", "ncbi_email"),
    Database.PUBMED: ("pubmed_api_key", "ncbi_tool", "ncbi_email"),
    Database.SCOPUS: ("scopus_api_key",),
    Database.SEMANTIC_SCHOLAR: ("semantic_scholar_api_key",),
    Database.SSRN: ("ssrn_api_key",),
    Database.WOS: ("wos_api_key",),
    Database.ZENODO: ("zenodo_access_token",),
    Database.ZOTERO: ("zotero_base_url", "zotero_library"),
}


_FORWARDING_CASES: tuple[tuple[Database, dict[str, object], dict[str, object]], ...] = (
    (
        Database.ARXIV_BULK,
        {
            "arxiv_snapshot_path": "/data/arxiv.json",
            "arxiv_categories": frozenset({"cs.AI"}),
        },
        {"snapshot_path": "/data/arxiv.json", "categories": frozenset({"cs.AI"})},
    ),
    (
        Database.CHEMRXIV,
        {"chemrxiv_email": "chem@example.org"},
        {"email": "chem@example.org"},
    ),
    (Database.CORE, {"core_api_key": "core-key"}, {"api_key": "core-key"}),
    (Database.IEEE, {"ieee_api_key": "ieee-key"}, {"api_key": "ieee-key"}),
    (
        Database.OPENAIRE,
        {
            "openaire_access_token": "access-token",
            "openaire_refresh_token": "refresh-token",
            "openaire_access_token_expires_in": 3_000.0,
        },
        {
            "access_token": "access-token",
            "refresh_token": "refresh-token",
            "access_token_expires_in": 3_000.0,
        },
    ),
    (
        Database.OPENALEX,
        {"openalex_api_key": "openalex-key", "email": "owner@example.org"},
        {"api_key": "openalex-key", "email": "owner@example.org"},
    ),
    (
        Database.OPENREVIEW,
        {
            "openreview_username": "owner@example.org",
            "openreview_password": "password",
            "openreview_venue_ids": ("ICLR.cc/2026/Conference",),
            "openreview_cache_dir": Path("/tmp/openreview-cache"),
            "openreview_cache_ttl_seconds": 123.0,
        },
        {
            "username": "owner@example.org",
            "password": "password",
            "venue_ids": ("ICLR.cc/2026/Conference",),
            "cache_dir": Path("/tmp/openreview-cache"),
            "cache_ttl_seconds": 123.0,
        },
    ),
    (
        Database.PMC,
        {
            "pubmed_api_key": "ncbi-key",
            "ncbi_tool": "lit-search-test",
            "ncbi_email": "owner@example.org",
        },
        {
            "api_key": "ncbi-key",
            "tool": "lit-search-test",
            "email": "owner@example.org",
        },
    ),
    (
        Database.PUBMED,
        {
            "pubmed_api_key": "ncbi-key",
            "ncbi_tool": "lit-search-test",
            "ncbi_email": "owner@example.org",
        },
        {
            "api_key": "ncbi-key",
            "tool": "lit-search-test",
            "email": "owner@example.org",
        },
    ),
    (Database.SCOPUS, {"scopus_api_key": "scopus-key"}, {"api_key": "scopus-key"}),
    (
        Database.SEMANTIC_SCHOLAR,
        {"semantic_scholar_api_key": "s2-key"},
        {"api_key": "s2-key"},
    ),
    (Database.SSRN, {"ssrn_api_key": "ssrn-key"}, {"api_key": "ssrn-key"}),
    (Database.WOS, {"wos_api_key": "wos-key"}, {"api_key": "wos-key"}),
    (
        Database.ZENODO,
        {"zenodo_access_token": "zenodo-token"},
        {"access_token": "zenodo-token"},
    ),
    (
        Database.ZOTERO,
        {
            "zotero_base_url": "http://127.0.0.1:23119/api",
            "zotero_library": "groups/42",
        },
        {"base_url": "http://127.0.0.1:23119/api", "library": "groups/42"},
    ),
)


class _SpyConnector:
    """Minimal constructor spy accepted by ``SearchRunner``."""

    def __init__(
        self,
        name: str,
        constructor_kwargs: dict[str, object],
        transport_calls: list[tuple[str | None, bool]],
    ) -> None:
        self.name = name
        self.constructor_kwargs = constructor_kwargs
        self.transport_calls = transport_calls

    def configure_transport(self, *, proxy: str | None, ssl_verify: bool) -> None:
        """Record generic transport configuration."""
        self.transport_calls.append((proxy, ssl_verify))

    def close(self) -> None:
        """Match the connector lifecycle without opening resources."""


def _spy_factory(
    database: Database,
    constructor_calls: list[dict[str, object]],
    transport_calls: list[tuple[str | None, bool]],
) -> Callable[..., _SpyConnector]:
    """Return a connector factory that records constructor and transport args."""

    def factory(**kwargs: object) -> _SpyConnector:
        constructor_calls.append(dict(kwargs))
        return _SpyConnector(database.value, dict(kwargs), transport_calls)

    return factory


def test_configuration_classification_is_exhaustive_and_public() -> None:
    """Every registered connector is explicitly configurable or explicitly bare."""
    assert SEARCH_CONNECTOR_CONFIG_PARAMETERS == _EXPECTED_CONFIG_PARAMETERS

    public_parameters = inspect.signature(SearchRunner).parameters
    for database, names in SEARCH_CONNECTOR_CONFIG_PARAMETERS.items():
        assert database in SEARCH_REGISTRY
        assert set(names) <= set(public_parameters), (
            f"{database.value} declares unreachable SearchRunner parameters: "
            f"{set(names) - set(public_parameters)}"
        )


def test_registry_constructor_options_cannot_outgrow_the_public_manifest() -> None:
    """A new connector option must be routed publicly or classified deliberately."""
    expected_constructor_options = {
        database: set(expected) for database, _public, expected in _FORWARDING_CASES
    }
    for database, connector_type in SEARCH_REGISTRY.items():
        constructor_options = {
            name for name in inspect.signature(connector_type).parameters if name != "query_builder"
        }
        assert constructor_options == expected_constructor_options.get(database, set()), (
            f"{database.value} constructor options changed without updating public routing: "
            f"constructor={sorted(constructor_options)}, "
            f"manifest={sorted(expected_constructor_options.get(database, set()))}"
        )


@pytest.mark.parametrize(
    ("database", "public_kwargs", "expected_constructor_kwargs"),
    _FORWARDING_CASES,
    ids=lambda value: value.value if isinstance(value, Database) else None,
)
def test_connector_specific_configuration_reaches_the_constructor(
    monkeypatch: pytest.MonkeyPatch,
    database: Database,
    public_kwargs: dict[str, object],
    expected_constructor_kwargs: dict[str, object],
) -> None:
    """Public facade values are forwarded under the connector's own names."""
    constructor_calls: list[dict[str, object]] = []
    transport_calls: list[tuple[str | None, bool]] = []
    monkeypatch.setitem(
        SEARCH_REGISTRY,
        database,
        _spy_factory(database, constructor_calls, transport_calls),
    )

    SearchRunner(
        query="[ml]",
        databases=[database.value],
        enrichment_databases=[],
        **public_kwargs,  # type: ignore[arg-type]
    )

    assert constructor_calls == [expected_constructor_kwargs]


@pytest.mark.parametrize("database", tuple(SEARCH_REGISTRY), ids=lambda db: db.value)
def test_generic_transport_configuration_reaches_every_connector(
    monkeypatch: pytest.MonkeyPatch,
    database: Database,
) -> None:
    """Proxy and TLS policy cannot become a connector-specific omission."""
    constructor_calls: list[dict[str, object]] = []
    transport_calls: list[tuple[str | None, bool]] = []
    monkeypatch.setitem(
        SEARCH_REGISTRY,
        database,
        _spy_factory(database, constructor_calls, transport_calls),
    )

    SearchRunner(
        query="[ml]",
        databases=[database.value],
        enrichment_databases=[],
        proxy="http://127.0.0.1:8080",
        ssl_verify=False,
    )

    assert transport_calls == [("http://127.0.0.1:8080", False)]


def test_default_search_is_usable_and_explicit_all_is_exhaustive() -> None:
    """The ordinary default avoids degraded sources; ``all`` remains literal."""
    expected_default = (
        Database.ARXIV,
        Database.BIORXIV,
        Database.CHEMRXIV,
        Database.CORE,
        Database.DBLP,
        Database.DOAJ,
        Database.EUROPE_PMC,
        Database.HAL,
        Database.MEDRXIV,
        Database.OPENALEX,
        Database.PMC,
        Database.PUBMED,
        Database.SEMANTIC_SCHOLAR,
        Database.ZENODO,
    )
    assert expected_default == DEFAULT_SEARCH_DATABASES

    default_runner = SearchRunner(query="[ml]", enrichment_databases=[])
    all_runner = SearchRunner(query="[ml]", databases=["all"], enrichment_databases=[])

    default_names = tuple(db.value for db in expected_default)
    all_names = tuple(db.value for db in SEARCH_REGISTRY)
    policy_excluded = tuple(name for name in all_names if name not in default_names)

    assert tuple(searcher.name for searcher in default_runner._searchers) == default_names
    assert tuple(default_runner._policy_excluded_databases) == policy_excluded
    assert tuple(default_runner._requested_databases) == default_names
    assert tuple(default_runner._registered_databases) == all_names
    assert default_runner._scope_profile == "default"
    assert {
        entry.source: entry.status for entry in default_runner._skipped_coverage()
    } == dict.fromkeys(policy_excluded, CoverageStatus.SKIPPED_BY_POLICY)

    # Explicit ``all`` attempts the entire registry.  Required credentials may
    # keep a connector out of ``_searchers``, but they cannot erase it from the
    # resolved request or the skip provenance.
    attempted_all = tuple(
        [str(searcher.name) for searcher in all_runner._searchers]
        + list(all_runner._skipped_databases)
    )
    assert set(attempted_all) == set(all_names)
    assert all_runner._policy_excluded_databases == []
    assert tuple(all_runner._requested_databases) == tuple(db.value for db in SEARCH_REGISTRY)
    assert all_runner._scope_profile == "all"


def test_credentialed_default_adds_openaire_without_weakening_unconfigured_default() -> None:
    """Renewable OpenAIRE auth expands the hot path only when usable."""
    runner = SearchRunner(
        query="[ml]",
        openaire_refresh_token="month-lived-refresh-token",
        enrichment_databases=[],
    )

    assert CREDENTIALLED_DEFAULT_SEARCH_DATABASES == (Database.OPENAIRE,)
    assert tuple(runner._requested_databases) == (*DEFAULT_SEARCH_DATABASES, "openaire")
    assert runner._scope_profile == "default+credentialed"
    assert "openaire" not in runner._policy_excluded_databases
    assert any(searcher.name == Database.OPENAIRE for searcher in runner._searchers)


def test_requested_but_constructor_skipped_source_survives_in_result_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A constructor-time credential skip is not erased from the saved request."""

    def missing_ssrn(**kwargs: Any) -> _SpyConnector:
        del kwargs
        raise MissingApiKeyError("ssrn_api_key")

    monkeypatch.setitem(SEARCH_REGISTRY, Database.SSRN, missing_ssrn)
    runner = SearchRunner(query="[ml]", databases=["ssrn"], enrichment_databases=[])

    result = runner.run(show_progress=False)

    assert result.databases == ["ssrn"]
    assert runner._requested_databases == ["ssrn"]
    entry = result.coverage.get("ssrn")
    assert entry is not None
    assert entry.status is CoverageStatus.SKIPPED_NO_CREDENTIAL
    assert entry.missing_credential == "ssrn_api_key"
