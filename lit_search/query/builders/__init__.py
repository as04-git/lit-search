"""Database-specific query builders."""

from lit_search.query.builders.arxiv import ArxivQueryBuilder
from lit_search.query.builders.arxiv_bulk import ArxivBulkQueryBuilder
from lit_search.query.builders.base_search import BaseSearchQueryBuilder
from lit_search.query.builders.biorxiv import BioRxivQueryBuilder
from lit_search.query.builders.core_ac_uk import CoreQueryBuilder
from lit_search.query.builders.crossref import CrossRefQueryBuilder
from lit_search.query.builders.dblp import DBLPQueryBuilder
from lit_search.query.builders.doaj import DOAJQueryBuilder
from lit_search.query.builders.europe_pmc import EuropePMCQueryBuilder
from lit_search.query.builders.google_scholar import GoogleScholarQueryBuilder
from lit_search.query.builders.hal import HALQueryBuilder
from lit_search.query.builders.iacr import IACRQueryBuilder
from lit_search.query.builders.ieee import IEEEQueryBuilder
from lit_search.query.builders.openaire import OpenAIREQueryBuilder
from lit_search.query.builders.openalex import OpenAlexQueryBuilder
from lit_search.query.builders.openreview import OpenReviewQueryBuilder
from lit_search.query.builders.pmc import PMCQueryBuilder
from lit_search.query.builders.pubmed import PubmedQueryBuilder
from lit_search.query.builders.scopus import ScopusQueryBuilder
from lit_search.query.builders.semantic_scholar import SemanticScholarQueryBuilder
from lit_search.query.builders.ssrn import SSRNQueryBuilder
from lit_search.query.builders.zenodo import ZenodoQueryBuilder
from lit_search.query.builders.zotero import ZoteroQueryBuilder

__all__ = [
    "ArxivBulkQueryBuilder",
    "ArxivQueryBuilder",
    "BaseSearchQueryBuilder",
    "BioRxivQueryBuilder",
    "CoreQueryBuilder",
    "CrossRefQueryBuilder",
    "DBLPQueryBuilder",
    "DOAJQueryBuilder",
    "EuropePMCQueryBuilder",
    "GoogleScholarQueryBuilder",
    "HALQueryBuilder",
    "IACRQueryBuilder",
    "IEEEQueryBuilder",
    "OpenAIREQueryBuilder",
    "OpenAlexQueryBuilder",
    "OpenReviewQueryBuilder",
    "PMCQueryBuilder",
    "PubmedQueryBuilder",
    "SSRNQueryBuilder",
    "ScopusQueryBuilder",
    "SemanticScholarQueryBuilder",
    "ZenodoQueryBuilder",
    "ZoteroQueryBuilder",
]
