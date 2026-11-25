"""Turn command-line identifiers and stored papers into fetch targets."""

from __future__ import annotations

import re

from lit_search.core.citation import IdentifierKind
from lit_search.core.paper import Paper
from lit_search.fetch.model import FetchTarget
from lit_search.utils.normalization import normalize_doi

#: Prefix of DataCite DOIs that arXiv mints for every submission.
ARXIV_DOI_PREFIX = "10.48550/arxiv."

_NEW_ARXIV = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_OLD_ARXIV = re.compile(r"^[a-z][a-z.-]*(\.[A-Z]{2})?/\d{7}(v\d+)?$")
_ARXIV_URL = re.compile(
    r"^https?://(?:www\.|export\.)?arxiv\.org/(?:abs|pdf)/(?P<id>.+?)(?:\.pdf)?/?$",
    re.IGNORECASE,
)
_VERSION_SUFFIX = re.compile(r"v\d+$")


def _arxiv_id(raw: str) -> str | None:
    """Return a bare arXiv id if *raw* is one in any accepted form."""
    value = raw.strip()
    match = _ARXIV_URL.match(value)
    if match:
        value = match.group("id")
    if value.lower().startswith("arxiv:"):
        value = value[len("arxiv:") :]
    if _NEW_ARXIV.match(value) or _OLD_ARXIV.match(value):
        return value
    return None


def arxiv_doi(arxiv_id: str) -> str:
    """Return the DataCite DOI arXiv mints for *arxiv_id* (version dropped).

    Parameters
    ----------
    arxiv_id : str
        arXiv identifier, with or without a version suffix.

    Returns
    -------
    str
        E.g. ``10.48550/arXiv.2310.06825``.
    """
    return f"10.48550/arXiv.{_VERSION_SUFFIX.sub('', arxiv_id)}"


def parse_identifier(raw: str) -> FetchTarget:
    """Parse one command-line identifier.

    Accepted: a DOI (bare, ``doi:``-prefixed or a doi.org URL), an arXiv DOI
    (``10.48550/arXiv.<id>``), an arXiv id (``2310.06825``, ``arXiv:...``,
    ``hep-th/9901001``) or an arxiv.org abs/pdf URL.

    Parameters
    ----------
    raw : str
        Text supplied by the user.

    Returns
    -------
    FetchTarget
        Target labelled with the normalised identifier.

    Raises
    ------
    ValueError
        If *raw* is none of the accepted forms.
    """
    arxiv_id = _arxiv_id(raw)
    if arxiv_id is not None:
        return FetchTarget(label=arxiv_id, arxiv_id=arxiv_id, doi=arxiv_doi(arxiv_id))
    doi = normalize_doi(raw)
    if doi is None:
        raise ValueError(f"not a DOI or arXiv id: {raw!r}")
    if doi.lower().startswith(ARXIV_DOI_PREFIX):
        return FetchTarget(label=doi, doi=doi, arxiv_id=doi[len(ARXIV_DOI_PREFIX) :])
    return FetchTarget(label=doi, doi=doi)


def target_from_paper(paper: Paper) -> FetchTarget:
    """Build a target from a stored paper, keeping every identifier it has.

    Parameters
    ----------
    paper : Paper
        Paper from a handle.

    Returns
    -------
    FetchTarget
        Target labelled by DOI, else arXiv id, else title.
    """
    arxiv_id = next(
        (ident.value for ident in paper.identifiers if ident.kind is IdentifierKind.ARXIV),
        None,
    )
    if arxiv_id is None and paper.url:
        arxiv_id = _arxiv_id(paper.url)
    doi = paper.doi
    if doi is None and arxiv_id is not None:
        doi = arxiv_doi(arxiv_id)
    year = getattr(paper.publication_date, "year", None) if paper.publication_date else None
    label = paper.doi or arxiv_id or paper.title
    return FetchTarget(
        label=label,
        doi=doi,
        arxiv_id=arxiv_id,
        pdf_url=paper.pdf_url,
        landing_url=paper.url,
        title=paper.title,
        year=year,
    )


__all__ = ["ARXIV_DOI_PREFIX", "arxiv_doi", "parse_identifier", "target_from_paper"]
