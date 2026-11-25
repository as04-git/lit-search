"""Unit tests for the Engine facade."""

from __future__ import annotations

import datetime as dt
import os
from unittest.mock import MagicMock, patch

import pytest

import lit_search
from lit_search.core.lookup_result import LookupResult
from lit_search.core.paper import Paper
from lit_search.core.query_family import QueryFamilyRevision
from lit_search.core.query_family_result import QueryFamilyResult
from lit_search.core.search_result import SearchResult
from lit_search.engine import Engine
from lit_search.query.family_compiler import QueryFamilyPlan
from lit_search.runners.snowball_runner import SnowballRunner

_SETTING_ENV_VARS = {
    "IEEE_API_KEY": "",
    "SCOPUS_API_KEY": "",
    "NCBI_API_KEY": "",
    "OPENALEX_API_KEY": "",
    "LIT_SEARCH_EMAIL": "",
    "SEMANTIC_SCHOLAR_API_KEY": "",
    "WOS_API_KEY": "",
    "LIT_SEARCH_PROXY": "",
    "LIT_SEARCH_SSL_VERIFY": "",
}


@pytest.fixture(autouse=True)
def _clear_findpapers_env():
    """Clear lit-search credential variables for every test in this module."""
    with patch.dict(os.environ, _SETTING_ENV_VARS, clear=False):
        yield


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestEngineInit:
    """Tests for Engine initialisation and stored configuration."""

    def test_default_values(self):
        """Engine defaults all API keys to None, ssl_verify to True."""
        # Ensure no credential env vars leak from other tests.
        env_clear = {
            "IEEE_API_KEY": "",
            "SCOPUS_API_KEY": "",
            "NCBI_API_KEY": "",
            "OPENALEX_API_KEY": "",
            "LIT_SEARCH_EMAIL": "",
            "SEMANTIC_SCHOLAR_API_KEY": "",
            "WOS_API_KEY": "",
            "LIT_SEARCH_PROXY": "",
            "LIT_SEARCH_SSL_VERIFY": "",
        }
        with patch.dict(os.environ, env_clear, clear=False):
            engine = Engine()

        assert engine._ieee_api_key is None
        assert engine._scopus_api_key is None
        assert engine._pubmed_api_key is None
        assert engine._openalex_api_key is None
        assert engine._email is None
        assert engine._semantic_scholar_api_key is None
        assert engine._proxy is None
        assert engine._ssl_verify is True

    def test_custom_values(self):
        """Engine stores the supplied configuration values."""
        engine = Engine(
            ieee_api_key="ieee-k",
            scopus_api_key="scopus-k",
            pubmed_api_key="pubmed-k",
            openalex_api_key="oalex-k",
            email="me@example.com",
            semantic_scholar_api_key="s2-k",
            proxy="http://proxy:8080",
            ssl_verify=False,
        )

        assert engine._ieee_api_key == "ieee-k"
        assert engine._scopus_api_key == "scopus-k"
        assert engine._pubmed_api_key == "pubmed-k"
        assert engine._openalex_api_key == "oalex-k"
        assert engine._email == "me@example.com"
        assert engine._semantic_scholar_api_key == "s2-k"
        assert engine._proxy == "http://proxy:8080"
        assert engine._ssl_verify is False

    def test_keyword_only_arguments(self):
        """Engine constructor only accepts keyword arguments."""
        with pytest.raises(TypeError):
            Engine("ieee-key")  # type: ignore[misc]

    def test_env_var_fallbacks(self):
        """Engine reads missing values from credential env vars."""
        env = {
            "IEEE_API_KEY": "env-ieee",
            "SCOPUS_API_KEY": "env-scopus",
            "NCBI_API_KEY": "env-pubmed",
            "OPENALEX_API_KEY": "env-oalex",
            "LIT_SEARCH_EMAIL": "env@example.com",
            "SEMANTIC_SCHOLAR_API_KEY": "env-s2",
            "WOS_API_KEY": "env-wos",
            "LIT_SEARCH_PROXY": "http://env-proxy:8080",
        }
        with patch.dict(os.environ, env, clear=False):
            engine = Engine()

        assert engine._ieee_api_key == "env-ieee"
        assert engine._scopus_api_key == "env-scopus"
        assert engine._pubmed_api_key == "env-pubmed"
        assert engine._openalex_api_key == "env-oalex"
        assert engine._email == "env@example.com"
        assert engine._semantic_scholar_api_key == "env-s2"
        assert engine._wos_api_key == "env-wos"
        assert engine._proxy == "http://env-proxy:8080"

    def test_explicit_values_override_env(self):
        """Explicit arguments take precedence over environment variables."""
        env = {
            "IEEE_API_KEY": "env-ieee",
            "LIT_SEARCH_EMAIL": "env@example.com",
        }
        with patch.dict(os.environ, env, clear=False):
            engine = Engine(ieee_api_key="explicit-ieee", email="explicit@example.com")

        assert engine._ieee_api_key == "explicit-ieee"
        assert engine._email == "explicit@example.com"

    def test_ssl_verify_env_false(self):
        """LIT_SEARCH_SSL_VERIFY=false disables SSL verification."""
        for val in ("false", "0", "no", "False", "NO"):
            with patch.dict(os.environ, {"LIT_SEARCH_SSL_VERIFY": val}, clear=False):
                engine = Engine()
            assert engine._ssl_verify is False, f"Expected False for SSL_VERIFY={val!r}"

    def test_ssl_verify_env_true(self):
        """LIT_SEARCH_SSL_VERIFY=true keeps SSL verification enabled."""
        with patch.dict(os.environ, {"LIT_SEARCH_SSL_VERIFY": "true"}, clear=False):
            engine = Engine()
        assert engine._ssl_verify is True

    def test_ssl_verify_explicit_overrides_env(self):
        """Explicit ssl_verify=False overrides the environment variable."""
        with patch.dict(os.environ, {"LIT_SEARCH_SSL_VERIFY": "true"}, clear=False):
            engine = Engine(ssl_verify=False)
        assert engine._ssl_verify is False

    def test_ssl_verify_false_logs_warning(self, caplog):
        """Engine logs a security warning when SSL verification is disabled."""
        with caplog.at_level("WARNING", logger="lit_search.engine"):
            Engine(ssl_verify=False)
        assert any("SSL certificate verification is disabled" in msg for msg in caplog.messages)

    def test_ssl_verify_true_no_warning(self, caplog):
        """Engine does not log a warning when SSL verification is enabled."""
        env_clear = {"LIT_SEARCH_SSL_VERIFY": ""}
        with (
            patch.dict(os.environ, env_clear, clear=False),
            caplog.at_level("WARNING", logger="lit_search.engine"),
        ):
            Engine(ssl_verify=True)
        assert not any("SSL certificate verification" in msg for msg in caplog.messages)

    def test_ssl_verify_env_false_logs_warning(self, caplog):
        """Engine logs a warning when SSL is disabled via environment variable."""
        with (
            patch.dict(os.environ, {"LIT_SEARCH_SSL_VERIFY": "false"}, clear=False),
            caplog.at_level("WARNING", logger="lit_search.engine"),
        ):
            Engine()
        assert any("SSL certificate verification is disabled" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------


class TestEngineSearch:
    """Tests for Engine.search()."""

    def test_search_delegates_to_runner_with_engine_keys(self):
        """search() passes engine API keys and per-call params to SearchRunner."""
        engine = Engine(
            ieee_api_key="ieee-k",
            scopus_api_key="scopus-k",
            pubmed_api_key="pubmed-k",
            openalex_api_key="oalex-k",
            email="me@example.com",
            semantic_scholar_api_key="s2-k",
        )
        fake_search = MagicMock(spec=SearchResult)
        with patch("lit_search.engine.SearchRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_runner.run.return_value = fake_search
            mock_cls.return_value = mock_runner

            result = engine.search(
                "ti[ml]",
                databases=["arxiv"],
                max_papers_per_database=50,
                num_workers=3,
                verbose=True,
            )

        mock_cls.assert_called_once_with(
            query="ti[ml]",
            databases=["arxiv"],
            max_papers_per_database=50,
            ieee_api_key="ieee-k",
            scopus_api_key="scopus-k",
            pubmed_api_key="pubmed-k",
            openalex_api_key="oalex-k",
            email="me@example.com",
            semantic_scholar_api_key="s2-k",
            wos_api_key=None,
            core_api_key=None,
            openaire_access_token=None,
            openaire_refresh_token=None,
            openaire_access_token_expires_in=None,
            zenodo_access_token=None,
            chemrxiv_email="me@example.com",
            ssrn_api_key=None,
            ncbi_tool=None,
            ncbi_email="me@example.com",
            openreview_username=None,
            openreview_password=None,
            openreview_venue_ids=None,
            openreview_cache_dir=None,
            openreview_cache_ttl_seconds=None,
            arxiv_snapshot_path=None,
            arxiv_categories=None,
            zotero_base_url=None,
            zotero_library=None,
            num_workers=3,
            since=None,
            until=None,
            enrichment_databases=["crossref", "web_scraping"],
            max_cited_by=100,
            proxy=None,
            ssl_verify=True,
        )
        mock_runner.run.assert_called_once_with(verbose=True, show_progress=True)
        assert result is fake_search

    def test_search_default_per_call_params(self):
        """num_workers defaults to 1, verbose to False."""
        engine = Engine()
        with patch("lit_search.engine.SearchRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_runner.run.return_value = MagicMock(spec=SearchResult)
            mock_cls.return_value = mock_runner

            engine.search("[ml]")

        _, kwargs = mock_cls.call_args
        assert kwargs["num_workers"] == 1
        mock_runner.run.assert_called_once_with(verbose=False, show_progress=True)

    def test_search_show_progress_false(self):
        """show_progress=False is forwarded to SearchRunner.run()."""
        engine = Engine()
        with patch("lit_search.engine.SearchRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_runner.run.return_value = MagicMock(spec=SearchResult)
            mock_cls.return_value = mock_runner

            engine.search("[ml]", show_progress=False)

        mock_runner.run.assert_called_once_with(verbose=False, show_progress=False)


# ---------------------------------------------------------------------------
# search_family()
# ---------------------------------------------------------------------------


class TestEngineSearchFamily:
    """Tests for the explicit query-family Engine entry point."""

    def test_routes_exact_revision_scope_plan_verifier_and_engine_configuration(self):
        """The facade adds no query terms and forwards every source credential."""
        venue_ids = ("ICLR.cc/2026/Conference",)
        categories = frozenset({"cs.AI", "cs.SE"})
        engine = Engine(
            ieee_api_key="ieee-k",
            scopus_api_key="scopus-k",
            pubmed_api_key="pubmed-k",
            openalex_api_key="openalex-k",
            email="contact@example.test",
            semantic_scholar_api_key="s2-k",
            wos_api_key="wos-k",
            core_api_key="core-k",
            openaire_access_token="openaire-access",
            openaire_refresh_token="openaire-refresh",
            openaire_access_token_expires_in=1234.5,
            zenodo_access_token="zenodo-k",
            chemrxiv_email="chemrxiv@example.test",
            ssrn_api_key="ssrn-k",
            ncbi_tool="lit-search-test",
            ncbi_email="ncbi@example.test",
            openreview_username="reviewer",
            openreview_password="secret",
            openreview_venue_ids=venue_ids,
            openreview_cache_dir="/tmp/openreview-cache",
            openreview_cache_ttl_seconds=600.0,
            arxiv_snapshot_path="/tmp/arxiv-snapshot.jsonl",
            arxiv_categories=categories,
            zotero_base_url="https://zotero.example.test",
            zotero_library="library-1",
            proxy="http://proxy.example.test:8080",
            ssl_verify=False,
        )
        revision = MagicMock(spec=QueryFamilyRevision)
        plan = MagicMock(spec=QueryFamilyPlan)
        verifier = MagicMock(return_value=True)
        sources = ["openaire", "openalex", "arxiv"]
        fake_result = MagicMock(spec=QueryFamilyResult)
        since = dt.date(2020, 1, 2)
        until = dt.date(2026, 8, 31)

        with patch("lit_search.engine.QueryFamilyRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_runner.run.return_value = fake_result
            mock_cls.return_value = mock_runner

            result = engine.search_family(
                revision,
                sources=sources,
                plan=plan,
                evidence_verifier=verifier,
                max_papers_per_database=75,
                since=since,
                until=until,
                num_workers=4,
                verbose=True,
                show_progress=False,
            )

        mock_cls.assert_called_once_with(
            revision,
            sources,
            plan=plan,
            search_runner_kwargs={
                "max_papers_per_database": 75,
                "ieee_api_key": "ieee-k",
                "scopus_api_key": "scopus-k",
                "pubmed_api_key": "pubmed-k",
                "openalex_api_key": "openalex-k",
                "email": "contact@example.test",
                "semantic_scholar_api_key": "s2-k",
                "wos_api_key": "wos-k",
                "core_api_key": "core-k",
                "openaire_access_token": "openaire-access",
                "openaire_refresh_token": "openaire-refresh",
                "openaire_access_token_expires_in": 1234.5,
                "zenodo_access_token": "zenodo-k",
                "chemrxiv_email": "chemrxiv@example.test",
                "ssrn_api_key": "ssrn-k",
                "ncbi_tool": "lit-search-test",
                "ncbi_email": "ncbi@example.test",
                "openreview_username": "reviewer",
                "openreview_password": "secret",
                "openreview_venue_ids": venue_ids,
                "openreview_cache_dir": "/tmp/openreview-cache",
                "openreview_cache_ttl_seconds": 600.0,
                "arxiv_snapshot_path": "/tmp/arxiv-snapshot.jsonl",
                "arxiv_categories": categories,
                "zotero_base_url": "https://zotero.example.test",
                "zotero_library": "library-1",
                "num_workers": 4,
                "since": since,
                "until": until,
                "proxy": "http://proxy.example.test:8080",
                "ssl_verify": False,
            },
            evidence_verifier=verifier,
        )
        # QueryFamilyRunner, rather than the Engine, owns exact cell planning
        # and its mandatory per-cell enrichment suppression.
        assert "enrichment_databases" not in mock_cls.call_args.kwargs["search_runner_kwargs"]
        mock_runner.run.assert_called_once_with(verbose=True, show_progress=False)
        assert result is fake_result

    def test_requires_an_explicit_source_scope(self):
        """There is no implicit mutable/default source policy for families."""
        engine = Engine()
        revision = MagicMock(spec=QueryFamilyRevision)

        with pytest.raises(TypeError, match="sources"):
            engine.search_family(revision)  # type: ignore[call-arg]

    def test_default_execution_controls_are_forwarded(self):
        """Per-cell limits and execution controls retain documented defaults."""
        engine = Engine()
        revision = MagicMock(spec=QueryFamilyRevision)

        with patch("lit_search.engine.QueryFamilyRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_runner.run.return_value = MagicMock(spec=QueryFamilyResult)
            mock_cls.return_value = mock_runner

            engine.search_family(revision, sources=("arxiv",))

        kwargs = mock_cls.call_args.kwargs["search_runner_kwargs"]
        assert kwargs["max_papers_per_database"] is None
        assert kwargs["since"] is None
        assert kwargs["until"] is None
        assert kwargs["num_workers"] == 1
        mock_runner.run.assert_called_once_with(verbose=False, show_progress=True)


# ---------------------------------------------------------------------------
# download()
# ---------------------------------------------------------------------------


class TestEngineDownload:
    """Tests for Engine.download()."""

    def test_download_uses_engine_proxy_and_ssl(self, make_paper):
        """download() forwards engine proxy and ssl_verify to DownloadRunner."""
        engine = Engine(
            proxy="http://proxy:8080",
            ssl_verify=False,
        )
        fake_metrics = {"total_papers": 1, "downloaded_papers": 1}
        with patch("lit_search.engine.DownloadRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_runner.run.return_value = fake_metrics
            mock_cls.return_value = mock_runner

            papers = [make_paper()]
            result = engine.download(
                papers,
                "/tmp/pdfs",
                num_workers=2,
                timeout=20.0,
                verbose=True,
            )

        mock_cls.assert_called_once_with(
            papers=papers,
            output_directory="/tmp/pdfs",
            num_workers=2,
            timeout=20.0,
            proxy="http://proxy:8080",
            ssl_verify=False,
        )
        mock_runner.run.assert_called_once_with(verbose=True, show_progress=True)
        assert result == fake_metrics

    def test_download_default_per_call_params(self):
        """num_workers defaults to 1, verbose to False, timeout to 30.0."""
        engine = Engine()
        with patch("lit_search.engine.DownloadRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_runner.run.return_value = {}
            mock_cls.return_value = mock_runner

            engine.download([], "/tmp/out")

        _, kwargs = mock_cls.call_args
        assert kwargs["num_workers"] == 1
        assert kwargs["timeout"] == 30.0
        mock_runner.run.assert_called_once_with(verbose=False, show_progress=True)

    def test_download_show_progress_false(self):
        """show_progress=False is forwarded to DownloadRunner.run()."""
        engine = Engine()
        with patch("lit_search.engine.DownloadRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_runner.run.return_value = {}
            mock_cls.return_value = mock_runner

            engine.download([], "/tmp/out", show_progress=False)

        mock_runner.run.assert_called_once_with(verbose=False, show_progress=False)


# ---------------------------------------------------------------------------
# search() enrichment_databases forwarding
# ---------------------------------------------------------------------------


class TestEngineSearchEnrichmentDatabases:
    """Tests that Engine.search() forwards enrichment_databases to SearchRunner."""

    def test_enrichment_databases_forwarded(self):
        """enrichment_databases is passed through to SearchRunner."""
        engine = Engine()
        with patch("lit_search.engine.SearchRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_runner.run.return_value = MagicMock(spec=SearchResult)
            mock_cls.return_value = mock_runner

            engine.search("[ml]", enrichment_databases=["arxiv", "pubmed"])

        _, kwargs = mock_cls.call_args
        assert kwargs["enrichment_databases"] == ["arxiv", "pubmed"]

    def test_enrichment_databases_default_is_list(self):
        """enrichment_databases defaults to DEFAULT_ENRICHMENT_DATABASES (not None)."""
        engine = Engine()
        with patch("lit_search.engine.SearchRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_runner.run.return_value = MagicMock(spec=SearchResult)
            mock_cls.return_value = mock_runner

            engine.search("[ml]")

        _, kwargs = mock_cls.call_args
        assert kwargs["enrichment_databases"] == ["crossref", "web_scraping"]

    def test_enrichment_databases_none_uses_default(self):
        """enrichment_databases=None is forwarded to SearchRunner (uses default databases)."""
        engine = Engine()
        with patch("lit_search.engine.SearchRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_runner.run.return_value = MagicMock(spec=SearchResult)
            mock_cls.return_value = mock_runner

            engine.search("[ml]", enrichment_databases=None)

        _, kwargs = mock_cls.call_args
        assert kwargs["enrichment_databases"] is None

    def test_enrichment_databases_empty_list_disables_enrichment(self):
        """enrichment_databases=[] is forwarded to SearchRunner to disable enrichment."""
        engine = Engine()
        with patch("lit_search.engine.SearchRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_runner.run.return_value = MagicMock(spec=SearchResult)
            mock_cls.return_value = mock_runner

            engine.search("[ml]", enrichment_databases=[])

        _, kwargs = mock_cls.call_args
        assert kwargs["enrichment_databases"] == []


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------


class TestEngineGet:
    """Tests for Engine.get()."""

    def test_fetch_forwards_timeout(self):
        """get() forwards per-call timeout to GetRunner."""
        engine = Engine()
        fake_paper = MagicMock(spec=Paper)
        with patch("lit_search.engine.GetRunner") as mock_cls:
            mock_runner = MagicMock()
            fake_result = LookupResult(identifier="10.1234/test", papers=[fake_paper])
            mock_runner.run.return_value = fake_result
            mock_cls.return_value = mock_runner

            result = engine.get("10.1234/test", timeout=30.0, verbose=True)

        _, call_kwargs = mock_cls.call_args
        assert call_kwargs["identifier"] == "10.1234/test"
        assert call_kwargs["timeout"] == 30.0
        mock_runner.run.assert_called_once_with(verbose=True, max_cited_by=100)
        assert result is fake_result

    def test_fetch_returns_lookup_result_when_not_found(self):
        """get() preserves the runner's empty coverage-bearing result."""
        engine = Engine()
        empty = LookupResult(identifier="10.9999/nonexistent")
        with patch("lit_search.engine.GetRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_runner.run.return_value = empty
            mock_cls.return_value = mock_runner

            result = engine.get("10.9999/nonexistent")

        assert result is empty

    def test_fetch_default_per_call_params(self):
        """get() verbose defaults to False, timeout to 10.0."""
        engine = Engine()
        with patch("lit_search.engine.GetRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_runner.run.return_value = None
            mock_cls.return_value = mock_runner

            engine.get("10.1234/test")

        _, kwargs = mock_cls.call_args
        assert kwargs["timeout"] == 10.0
        mock_runner.run.assert_called_once_with(verbose=False, max_cited_by=100)

    def test_get_forwards_doi_org_url(self):
        """get() passes doi.org URLs to GetRunner as the identifier."""
        engine = Engine()
        fake_paper = MagicMock(spec=Paper)
        with patch("lit_search.engine.GetRunner") as mock_cls:
            mock_runner = MagicMock()
            fake_result = LookupResult(
                identifier="https://doi.org/10.1038/nature12373", papers=[fake_paper]
            )
            mock_runner.run.return_value = fake_result
            mock_cls.return_value = mock_runner

            result = engine.get("https://doi.org/10.1038/nature12373")

        _, kwargs = mock_cls.call_args
        assert kwargs["identifier"] == "https://doi.org/10.1038/nature12373"
        assert result is fake_result

    def test_get_forwards_landing_page_url(self):
        """get() passes landing-page URLs to GetRunner as the identifier."""
        engine = Engine()
        fake_paper = MagicMock(spec=Paper)
        with patch("lit_search.engine.GetRunner") as mock_cls:
            mock_runner = MagicMock()
            fake_result = LookupResult(
                identifier="https://arxiv.org/abs/1706.03762", papers=[fake_paper]
            )
            mock_runner.run.return_value = fake_result
            mock_cls.return_value = mock_runner

            result = engine.get("https://arxiv.org/abs/1706.03762")

        _, kwargs = mock_cls.call_args
        assert kwargs["identifier"] == "https://arxiv.org/abs/1706.03762"
        assert result is fake_result

    def test_get_forwards_proxy_and_ssl(self):
        """get() passes proxy and ssl_verify from the Engine to GetRunner."""
        engine = Engine(proxy="http://proxy:8080", ssl_verify=False)
        with patch("lit_search.engine.GetRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_runner.run.return_value = None
            mock_cls.return_value = mock_runner

            engine.get("https://arxiv.org/abs/1706.03762")

        _, kwargs = mock_cls.call_args
        assert kwargs["proxy"] == "http://proxy:8080"
        assert kwargs["ssl_verify"] is False

    def test_get_forwards_databases_param(self):
        """get() passes the databases list to GetRunner."""
        engine = Engine()
        with patch("lit_search.engine.GetRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_runner.run.return_value = None
            mock_cls.return_value = mock_runner

            engine.get("10.1234/test", databases=["arxiv", "pubmed"])

        _, kwargs = mock_cls.call_args
        assert kwargs["databases"] == ["arxiv", "pubmed"]

    def test_get_databases_defaults_to_none(self):
        """get() passes databases=None to GetRunner when not specified."""
        engine = Engine()
        with patch("lit_search.engine.GetRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_runner.run.return_value = None
            mock_cls.return_value = mock_runner

            engine.get("10.1234/test")

        _, kwargs = mock_cls.call_args
        assert kwargs["databases"] is None

    def test_get_forwards_crossref_in_databases(self):
        """get() forwards databases list including 'crossref' to GetRunner."""
        engine = Engine()
        with patch("lit_search.engine.GetRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_runner.run.return_value = None
            mock_cls.return_value = mock_runner

            engine.get("10.1234/test", databases=["crossref"])

        _, kwargs = mock_cls.call_args
        assert kwargs["databases"] == ["crossref"]

    def test_get_forwards_web_scraping_in_databases(self):
        """get() forwards databases list including 'web_scraping' to GetRunner."""
        engine = Engine()
        with patch("lit_search.engine.GetRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_runner.run.return_value = None
            mock_cls.return_value = mock_runner

            engine.get("https://arxiv.org/abs/1706.03762", databases=["web_scraping"])

        _, kwargs = mock_cls.call_args
        assert kwargs["databases"] == ["web_scraping"]


# ---------------------------------------------------------------------------
# Top-level import
# ---------------------------------------------------------------------------


class TestEngineImport:
    """Verify that Engine is accessible from the lit-search namespace."""

    def test_engine_importable(self):
        """lit_search.Engine is the Engine class."""

        assert lit_search.Engine is Engine


# ---------------------------------------------------------------------------
# Snowball
# ---------------------------------------------------------------------------


class TestEngineSnowball:
    """Tests for Engine.snowball()."""

    def test_snowball_delegates_to_runner(self, make_paper):
        """snowball() creates a SnowballRunner and calls run()."""
        engine = Engine()
        seed = make_paper("Seed", doi="10.1000/seed")

        with patch("lit_search.engine.SnowballRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_graph = MagicMock()
            mock_runner.run.return_value = mock_graph
            mock_cls.return_value = mock_runner

            result = engine.snowball(seed)

        mock_cls.assert_called_once()
        mock_runner.run.assert_called_once_with(verbose=False, show_progress=True)
        assert result is mock_graph

    def test_snowball_passes_parameters(self, make_paper):
        """snowball() passes configuration to SnowballRunner."""
        engine = Engine(openalex_api_key="oakey", email="me@test.com")
        seed = make_paper("Seed", doi="10.1000/seed")

        with patch("lit_search.engine.SnowballRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_runner.run.return_value = MagicMock()
            mock_cls.return_value = mock_runner

            engine.snowball(
                seed,
                max_depth=2,
                direction="backward",
                num_workers=3,
                verbose=True,
            )

        _, kwargs = mock_cls.call_args
        assert kwargs["max_depth"] == 2
        assert kwargs["direction"] == "backward"
        assert kwargs["num_workers"] == 3
        assert kwargs["openalex_api_key"] == "oakey"
        assert kwargs["email"] == "me@test.com"
        mock_runner.run.assert_called_once_with(verbose=True, show_progress=True)

    def test_snowball_since_until_forwarded(self, make_paper):
        """snowball() forwards since and until to SnowballRunner."""
        import datetime as _dt

        engine = Engine()
        seed = make_paper("Seed", doi="10.1000/seed")
        since = _dt.date(2020, 1, 1)
        until = _dt.date(2023, 12, 31)

        with patch("lit_search.engine.SnowballRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_runner.run.return_value = MagicMock()
            mock_cls.return_value = mock_runner

            engine.snowball(seed, since=since, until=until)

        _, kwargs = mock_cls.call_args
        assert kwargs["since"] is since
        assert kwargs["until"] is until

    def test_snowball_since_until_none_by_default(self, make_paper):
        """snowball() passes None for since and until when not provided."""
        engine = Engine()
        seed = make_paper("Seed", doi="10.1000/seed")

        with patch("lit_search.engine.SnowballRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_runner.run.return_value = MagicMock()
            mock_cls.return_value = mock_runner

            engine.snowball(seed)

        _, kwargs = mock_cls.call_args
        assert kwargs["since"] is None
        assert kwargs["until"] is None

    def test_snowball_passes_single_paper_as_seed(self, make_paper):
        """snowball() forwards the given paper as seed_paper to SnowballRunner."""
        engine = Engine()
        seed = make_paper("Seed", doi="10.1000/seed")

        with patch("lit_search.engine.SnowballRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_runner.run.return_value = MagicMock()
            mock_cls.return_value = mock_runner

            engine.snowball(seed)

        call_args = mock_cls.call_args
        assert call_args[1]["seed_paper"] is seed

    def test_snowball_show_progress_false(self, make_paper):
        """show_progress=False is forwarded to SnowballRunner.run()."""
        engine = Engine()
        seed = make_paper("Seed", doi="10.1000/seed")

        with patch("lit_search.engine.SnowballRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_runner.run.return_value = MagicMock()
            mock_cls.return_value = mock_runner

            engine.snowball(seed, show_progress=False)

        mock_runner.run.assert_called_once_with(verbose=False, show_progress=False)

    def test_managed_snowball_creates_state_and_shared_cache(self, make_paper, tmp_path):
        """Managed execution binds exact runner semantics before doing provider work."""
        engine = Engine(openalex_api_key="secret")
        seed = make_paper("Seed", doi="10.1000/seed")
        run_id = "r_0123456789abcdef0123456789abcdef"
        probe = MagicMock()
        probe.semantic_spec = {"depth": 2}
        probe.credential_identities = ("credential-fingerprint",)
        managed = MagicMock()
        expected = MagicMock()
        managed.run.return_value = expected
        state = MagicMock()

        with (
            patch("lit_search.engine.SnowballRunner", side_effect=[probe, managed]) as runner_cls,
            patch("lit_search.engine.RunStateStore.create", return_value=state) as create,
            patch("lit_search.engine.SnapshotCache") as cache_cls,
        ):
            result = engine.snowball(
                seed,
                managed_store_root=tmp_path,
                managed_run_id=run_id,
            )

        create.assert_called_once_with(
            tmp_path,
            probe.semantic_spec,
            credentials=probe.credential_identities,
            run_id=run_id,
        )
        cache_cls.assert_called_once_with(tmp_path)
        assert runner_cls.call_args_list[1].kwargs["run_state"] is state
        assert runner_cls.call_args_list[1].kwargs["snapshot_cache"] is cache_cls.return_value
        assert result is expected

    def test_managed_resume_refuses_to_recreate_run(self, make_paper, tmp_path):
        """Resume opens an existing exact semantic digest and never creates a replacement."""
        engine = Engine()
        seed = make_paper("Seed", doi="10.1000/seed")
        run_id = "r_fedcba9876543210fedcba9876543210"
        probe = MagicMock()
        probe.semantic_spec = {"depth": 2}
        probe.credential_identities = ()
        managed = MagicMock()
        state = MagicMock()

        with (
            patch("lit_search.engine.SnowballRunner", side_effect=[probe, managed]),
            patch("lit_search.engine.RunStateStore.resume", return_value=state) as resume,
            patch("lit_search.engine.RunStateStore.create") as create,
            patch("lit_search.engine.SnapshotCache"),
        ):
            engine.snowball(
                seed,
                managed_store_root=tmp_path,
                managed_run_id=run_id,
                resume_managed_run=True,
            )

        resume.assert_called_once_with(
            tmp_path,
            run_id,
            probe.semantic_spec,
            credentials=probe.credential_identities,
        )
        create.assert_not_called()


class TestSnowballImport:
    """Verify snowball-related classes are accessible from lit_search namespace."""

    def test_snowball_runner_importable(self):
        """lit_search.SnowballRunner is accessible."""

        assert lit_search.SnowballRunner is SnowballRunner


class TestEngineSimilar:
    """Tests for Engine.similar()."""

    def test_similar_delegates_to_runner(self, make_paper):
        """similar() creates a SimilarRunner and calls run()."""
        engine = Engine()
        seed = make_paper("Seed", doi="10.1000/seed")

        with patch("lit_search.engine.SimilarRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_result = MagicMock()
            mock_runner.run.return_value = mock_result
            mock_cls.return_value = mock_runner

            result = engine.similar(seed)

        mock_cls.assert_called_once()
        mock_runner.run.assert_called_once_with(verbose=False, show_progress=True)
        assert result is mock_result

    def test_similar_passes_parameters(self, make_paper):
        """similar() forwards configuration to SimilarRunner."""
        engine = Engine(
            semantic_scholar_api_key="sskey",
            pubmed_api_key="pmkey",
            openalex_api_key="oakey",
            ieee_api_key="ieeekey",
            scopus_api_key="scopuskey",
            wos_api_key="woskey",
        )
        seed = make_paper("Seed", doi="10.1000/seed")

        with patch("lit_search.engine.SimilarRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_runner.run.return_value = MagicMock()
            mock_cls.return_value = mock_runner

            engine.similar(
                seed,
                databases=["semantic_scholar"],
                max_papers_per_database=15,
                timeout=5.0,
                verbose=True,
            )

        _, kwargs = mock_cls.call_args
        assert kwargs["paper"] is seed
        assert kwargs["databases"] == ["semantic_scholar"]
        assert kwargs["max_papers_per_database"] == 15
        assert kwargs["timeout"] == 5.0
        assert kwargs["semantic_scholar_api_key"] == "sskey"
        assert kwargs["pubmed_api_key"] == "pmkey"
        assert kwargs["openalex_api_key"] == "oakey"
        assert kwargs["ieee_api_key"] == "ieeekey"
        assert kwargs["scopus_api_key"] == "scopuskey"
        assert kwargs["wos_api_key"] == "woskey"
        mock_runner.run.assert_called_once_with(verbose=True, show_progress=True)

    def test_similar_since_until_forwarded(self, make_paper):
        """similar() forwards since and until to SimilarRunner."""
        import datetime as _dt

        engine = Engine()
        seed = make_paper("Seed", doi="10.1000/seed")
        since = _dt.date(2020, 1, 1)
        until = _dt.date(2023, 12, 31)

        with patch("lit_search.engine.SimilarRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_runner.run.return_value = MagicMock()
            mock_cls.return_value = mock_runner

            engine.similar(seed, since=since, until=until)

        _, kwargs = mock_cls.call_args
        assert kwargs["since"] is since
        assert kwargs["until"] is until

    def test_similar_enrichment_databases_and_max_cited_by_forwarded(self, make_paper):
        """similar() forwards enrichment_databases, max_cited_by, and num_workers."""
        engine = Engine()
        seed = make_paper("Seed", doi="10.1000/seed")

        with patch("lit_search.engine.SimilarRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_runner.run.return_value = MagicMock()
            mock_cls.return_value = mock_runner

            engine.similar(
                seed,
                enrichment_databases=["crossref"],
                max_cited_by=50,
                num_workers=4,
            )

        _, kwargs = mock_cls.call_args
        assert kwargs["enrichment_databases"] == ["crossref"]
        assert kwargs["max_cited_by"] == 50
        assert kwargs["num_workers"] == 4

    def test_similar_default_enrichment_databases(self, make_paper):
        """When omitted, enrichment_databases defaults to crossref + web_scraping."""
        engine = Engine()
        seed = make_paper("Seed", doi="10.1000/seed")

        with patch("lit_search.engine.SimilarRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_runner.run.return_value = MagicMock()
            mock_cls.return_value = mock_runner

            engine.similar(seed)

        _, kwargs = mock_cls.call_args
        assert kwargs["enrichment_databases"] == ["crossref", "web_scraping"]

    def test_similar_show_progress_false(self, make_paper):
        """show_progress=False is forwarded to SimilarRunner.run()."""
        engine = Engine()
        seed = make_paper("Seed", doi="10.1000/seed")

        with patch("lit_search.engine.SimilarRunner") as mock_cls:
            mock_runner = MagicMock()
            mock_runner.run.return_value = MagicMock()
            mock_cls.return_value = mock_runner

            engine.similar(seed, show_progress=False)

        mock_runner.run.assert_called_once_with(verbose=False, show_progress=False)


class TestSimilarImport:
    """Verify similar-related classes are accessible from lit_search namespace."""

    def test_similar_runner_importable(self):
        """lit_search.SimilarRunner is accessible."""
        from lit_search.runners.similar_runner import SimilarRunner

        assert lit_search.SimilarRunner is SimilarRunner

    def test_similar_result_importable(self):
        """lit_search.SimilarResult is accessible."""
        from lit_search.core.similar_result import SimilarResult

        assert lit_search.SimilarResult is SimilarResult


class TestSaveFunctionsImport:
    """Verify persistence functions are accessible from lit_search namespace."""

    def test_save_to_json_importable(self):
        """lit_search.save_to_json is accessible."""

        assert callable(lit_search.save_to_json)

    def test_load_from_json_importable(self):
        """lit_search.load_from_json is accessible."""

        assert callable(lit_search.load_from_json)

    def test_save_to_bibtex_importable(self):
        """lit_search.save_to_bibtex is accessible."""

        assert callable(lit_search.save_to_bibtex)

    def test_load_from_bibtex_importable(self):
        """lit_search.load_from_bibtex is accessible."""

        assert callable(lit_search.load_from_bibtex)

    def test_save_to_csv_importable(self):
        """lit_search.save_to_csv is accessible."""

        assert callable(lit_search.save_to_csv)

    def test_load_from_csv_importable(self):
        """lit_search.load_from_csv is accessible."""

        assert callable(lit_search.load_from_csv)


# ---------------------------------------------------------------------------
# Save / Load (top-level functions)
# ---------------------------------------------------------------------------


class TestSaveToJson:
    """Tests for lit_search.save_to_json top-level function."""

    def test_save_search_result(self, make_paper, tmp_path):
        """SearchResult is saved to a JSON file."""

        search = SearchResult(query="[test]", databases=["arxiv"])
        search.add_paper(make_paper(doi="10.1/a"))
        path = str(tmp_path / "search.json")
        lit_search.save_to_json(search, path)
        assert os.path.exists(path)

    def test_save_paper_list(self, make_paper, tmp_path):
        """A plain list of papers is saved to a JSON file."""

        papers = [make_paper(doi="10.1/a"), make_paper(title="Paper B", doi="10.1/b")]
        path = str(tmp_path / "papers.json")
        lit_search.save_to_json(papers, path)
        assert os.path.exists(path)


class TestSaveToBibtex:
    """Tests for lit_search.save_to_bibtex top-level function."""

    def test_save_paper_list(self, make_paper, tmp_path):
        """A plain list of papers is saved to a BibTeX file."""

        papers = [make_paper(doi="10.1/a")]
        path = str(tmp_path / "refs.bib")
        lit_search.save_to_bibtex(papers, path)
        with open(path, encoding="utf-8") as file_handle:
            content = file_handle.read()
        assert "@" in content

    def test_empty_list_produces_empty_file(self, tmp_path):
        """An empty paper list creates an empty file."""

        path = str(tmp_path / "empty.bib")
        lit_search.save_to_bibtex([], path)
        with open(path, encoding="utf-8") as file_handle:
            content = file_handle.read()
        assert content == ""


class TestLoadFromBibtex:
    """Tests for lit_search.load_from_bibtex top-level function."""

    def test_round_trip(self, make_paper, tmp_path):
        """Papers survive save -> load round-trip."""

        papers = [make_paper(doi="10.1/a")]
        path = str(tmp_path / "refs.bib")
        lit_search.save_to_bibtex(papers, path)
        loaded = lit_search.load_from_bibtex(path)
        assert len(loaded) == 1
        assert loaded[0].title == papers[0].title


class TestLoadFromJson:
    """Tests for lit_search.load_from_json top-level function."""

    def test_round_trip_search_result(self, make_paper, tmp_path):
        """SearchResult survives save -> load round-trip."""

        search = SearchResult(query="[test]", databases=["arxiv"])
        search.add_paper(make_paper(doi="10.1/a"))
        path = str(tmp_path / "search.json")
        lit_search.save_to_json(search, path)

        loaded = lit_search.load_from_json(path)
        assert isinstance(loaded, SearchResult)
        assert len(loaded.papers) == 1

    def test_round_trip_paper_list(self, make_paper, tmp_path):
        """Paper list survives save -> load round-trip."""

        papers = [make_paper(doi="10.1/a")]
        path = str(tmp_path / "papers.json")
        lit_search.save_to_json(papers, path)

        loaded = lit_search.load_from_json(path)
        assert isinstance(loaded, list)
        assert len(loaded) == 1
        assert loaded[0].title == "Test Paper"


class TestSaveToCsv:
    """Tests for lit_search.save_to_csv top-level function."""

    def test_save_paper_list(self, make_paper, tmp_path):
        """A plain list of papers is saved to a CSV file."""

        papers = [make_paper(doi="10.1/a")]
        path = str(tmp_path / "papers.csv")
        lit_search.save_to_csv(papers, path)
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "title" in content
        assert "Test Paper" in content

    def test_empty_list_produces_header_only(self, tmp_path):
        """An empty paper list creates a CSV with only the header."""

        path = str(tmp_path / "empty.csv")
        lit_search.save_to_csv([], path)
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().strip().splitlines()
        assert len(lines) == 1


class TestLoadFromCsv:
    """Tests for lit_search.load_from_csv top-level function."""

    def test_round_trip(self, make_paper, tmp_path):
        """Papers survive save -> load round-trip."""

        papers = [make_paper(doi="10.1/a")]
        path = str(tmp_path / "papers.csv")
        lit_search.save_to_csv(papers, path)
        loaded = lit_search.load_from_csv(path)
        assert len(loaded) == 1
        assert loaded[0].title == papers[0].title
