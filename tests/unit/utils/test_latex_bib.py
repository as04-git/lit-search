"""Bibliographies and citation sentences read from LaTeX source."""

from __future__ import annotations

import gzip
import io
import tarfile

from lit_search.utils import latex_bib

NATBIB = r"""
\begin{thebibliography}{2}
\bibitem[Grazzi et~al.(2024)]{grazzi2024}
Riccardo Grazzi, Julien Siems, and Massimiliano Pontil.
\newblock Unlocking state-tracking in linear {RNN}s through negative eigenvalues.
\newblock \emph{arXiv preprint arXiv:2411.12537}, 2024.

\bibitem[Merrill et~al.(2024)]{merrill2024}
William Merrill, Jackson Petty, and Ashish Sabharwal.
\newblock The illusion of state in state-space models.
\newblock In \emph{ICML}, 2024. doi: 10.5555/3692070.3693520.
\end{thebibliography}
"""

BIBLATEX = r"""
\entry{siems2025}{article}{}
  \field{title}{{DeltaProduct}: Improving State-Tracking in Linear {RNNs} via Householder Products}
  \field{eprint}{2502.10297}
\endentry
"""

BIB = r"""
@article{used, title={A Paper That Is Cited}, journal={arXiv preprint arXiv:2301.00001}, year={2023}}
@article{unused, title={A Paper That Is Not Cited}, year={2020}}
"""

TEX = r"""
\section{Related work}
Linear RNNs with negative eigenvalues can solve parity \citep{grazzi2024}. Other work differs.
We also build on \cite{used}.
"""


def test_natbib_entries_carry_titles_and_ids() -> None:
    entries = latex_bib.parse_bbl(NATBIB)
    assert [e.key for e in entries] == ["grazzi2024", "merrill2024"]
    assert entries[0].title.startswith("Unlocking state-tracking in linear RNNs")
    assert entries[0].arxiv == "2411.12537"
    assert entries[1].title == "The illusion of state in state-space models"
    assert entries[1].doi == "10.5555/3692070.3693520"


def test_biblatex_entry() -> None:
    (entry,) = latex_bib.parse_bbl(BIBLATEX)
    assert entry.arxiv == "2502.10297"
    assert entry.title.startswith("DeltaProduct: Improving State-Tracking")


def test_bib_database_keeps_only_cited_entries() -> None:
    origin, entries = latex_bib.bibliography({"refs.bib": BIB, "main.tex": TEX})
    assert origin == "bib"
    assert [e.key for e in entries] == ["used"]
    assert entries[0].arxiv == "2301.00001"


def test_compiled_bibliography_is_preferred() -> None:
    origin, entries = latex_bib.bibliography({"main.bbl": NATBIB, "refs.bib": BIB, "main.tex": TEX})
    assert origin == "bbl"
    assert len(entries) == 2


def test_citation_sentence_is_the_citing_sentence() -> None:
    (sentence,) = latex_bib.citation_sentences({"main.tex": TEX}, {"grazzi2024"})
    assert sentence.startswith("Linear RNNs with negative eigenvalues can solve parity")
    assert "Other work" not in sentence


def test_unpack_reads_a_gzipped_tar_and_recognises_a_pdf() -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, text in (("main.tex", TEX), ("main.bbl", NATBIB), ("fig.png", "x")):
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    kind, files = latex_bib.unpack(gzip.compress(buffer.getvalue()))
    assert kind == "tar"
    assert sorted(files) == ["main.bbl", "main.tex"]
    assert latex_bib.unpack(b"%PDF-1.5 ...") == ("pdf", {})
    kind, files = latex_bib.unpack(gzip.compress(TEX.encode()))
    assert kind == "single"


def test_title_similarity() -> None:
    assert latex_bib.title_similarity("The Illusion of State", "the illusion of state") == 1.0
    assert latex_bib.title_similarity("Something else entirely", "The illusion of state") < 0.6
