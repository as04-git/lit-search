"""Read a paper's bibliography and citation sentences from its LaTeX source.

An arXiv e-print is the authors' own source: its ``.bbl`` (or ``.bib`` plus the
``\\cite`` keys the text uses) lists every reference, and the ``.tex`` holds the
sentence each one is cited in.  Measured 2026-11-25 on 16 recent arXiv ML papers:
15 had usable source, and the parsed lists recovered 95% of Semantic Scholar's
linked references (docs/experiments/2026-11-25-s2-free in the development repo).

Everything here is pure and in memory: archive members are read, never written.
"""

from __future__ import annotations

import difflib
import gzip
import io
import re
import tarfile
from dataclasses import dataclass

_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s}{,\"\\]+", re.I)
_ARXIV_RE = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)")
_CITE_RE = re.compile(
    r"\\(?:[a-zA-Z]*cite[a-zA-Z]*|autocite|parencite|textcite)\*?\s*"
    r"(?:\[[^\]]*\]\s*){0,2}\{([^}]*)\}"
)
_MEMBER_LIMIT = 20_000_000
_STRUCTURE = re.compile(r"\\(?:(?:sub)*section|paragraph|caption)\*?\{[^}]*\}|\\item\b")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z\\])")


@dataclass
class BibEntry:
    """One reference of a paper, as its authors wrote it."""

    key: str
    title: str
    arxiv: str | None = None
    doi: str | None = None
    raw: str = ""


def normalise(text: str | None) -> str:
    """Lowercase, drop LaTeX commands and punctuation, collapse spaces."""
    lowered = (text or "").casefold()
    lowered = re.sub(r"\\[a-z]+\s*", " ", lowered)
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def arxiv_id_in(text: str | None) -> str | None:
    """Return the first new-style arXiv id in *text*, without version."""
    match = _ARXIV_RE.search(text or "")
    return match.group(1) if match else None


def title_similarity(a: str | None, b: str | None) -> float:
    """Similarity of two titles in [0, 1]; containment of a long title counts as 0.95."""
    left, right = normalise(a), normalise(b)
    if not left or not right:
        return 0.0
    if len(left) > 25 and len(right) > 25 and (left in right or right in left):
        return 0.95
    return difflib.SequenceMatcher(None, left, right).ratio()


def unpack(data: bytes) -> tuple[str, dict[str, str]]:
    """Return the kind of an e-print and its ``.tex``/``.bbl``/``.bib`` members.

    Parameters
    ----------
    data : bytes
        The e-print as served: a gzipped tar, a gzipped single file, or a PDF.

    Returns
    -------
    tuple[str, dict[str, str]]
        ``("tar" | "single" | "pdf" | "unreadable", {member name: text})``.
    """
    if data[:4] == b"%PDF":
        return "pdf", {}
    if data[:2] == b"\x1f\x8b":
        try:
            data = gzip.decompress(data)
        except (OSError, EOFError):
            return "unreadable", {}
    if data[:4] == b"%PDF":
        return "pdf", {}
    try:
        files: dict[str, str] = {}
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            for member in archive.getmembers():
                if not member.isfile() or member.size > _MEMBER_LIMIT:
                    continue
                if member.name.lower().endswith((".tex", ".bbl", ".bib", ".ltx")):
                    handle = archive.extractfile(member)
                    if handle is not None:
                        files[member.name] = handle.read().decode("utf-8", "replace")
        return "tar", files
    except tarfile.TarError:
        return "single", {"main.tex": data.decode("utf-8", "replace")}


def _balanced(text: str, start: int, opening: str = "{", closing: str = "}") -> int:
    depth = 0
    index = start
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return len(text)


def delatex(text: str) -> str:
    """Strip LaTeX markup well enough to read and match a title or sentence."""
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(r"\\(?:url|path|href)\s*\{[^}]*\}", " ", text)
    text = re.sub(
        r"\\[a-zA-Z]*cite[a-zA-Z]*\*?\s*(?:\[[^\]]*\]\s*){0,2}\{[^}]*\}", " [cite] ", text
    )
    text = re.sub(
        r"\\(?:emph|textit|textbf|textsc|textrm|texttt|mbox|showarticletitle|showDOI)\s*", "", text
    )
    text = re.sub(r"\\(?:bibinfo|bibfield|bibitemdeclare)\{[^}]*\}", "", text)
    text = re.sub(r"\\['\"`^~=.uvHcdbt]\{?([a-zA-Z])\}?", r"\1", text)
    text = text.replace("~", " ").replace("\\&", "&")
    text = re.sub(
        r"\\(?:newblock|penalty\d*|natexlab\{[^}]*\}|BIBentry\w*|ignorespaces)", " ", text
    )
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    text = text.replace("{", "").replace("}", "").replace("$", "")
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([.,;:])", r"\1", text)


def _ids_in(text: str) -> tuple[str | None, str | None]:
    lowered = text.lower()
    arxiv = None
    if "arxiv" in lowered or "eprint" in lowered or "corr" in lowered:
        match = re.search(
            r"(?:arxiv[:\s./a-z]*|abs/|pdf/|eprint\}?\{?)\s*(\d{4}\.\d{4,5})", lowered
        )
        arxiv = match.group(1) if match else arxiv_id_in(lowered)
    found = _DOI_RE.search(text)
    doi = found.group(0).rstrip(".").lower() if found else None
    if doi and doi.startswith("10.48550/arxiv."):
        arxiv = arxiv or doi.split("arxiv.")[-1]
    return arxiv, doi


def _title_from_bibitem(body: str) -> str:
    blocks = re.split(r"\\newblock", body)
    if len(blocks) >= 2:
        title = re.sub(r"\.\s*$", "", delatex(blocks[1]))
        return re.split(r"\.\s+(?:In|arXiv|CoRR|Proceedings|Advances)\b", title)[0]
    quoted = re.search(r"``(.+?)(?:,?'')", body, re.S)
    if quoted:
        return delatex(quoted.group(1)).rstrip(",.")
    flat = delatex(body)
    parts = re.split(r"(?<=[a-z0-9\)])\.\s+", flat)
    return parts[1] if len(parts) > 1 else flat


def parse_bbl(text: str) -> list[BibEntry]:
    """Parse a compiled bibliography (natbib/plain ``\\bibitem`` or biblatex ``\\entry``)."""
    entries: list[BibEntry] = []
    if "\\entry{" in text and "\\field{" in text:
        for match in re.finditer(r"\\entry\{([^}]*)\}\{([^}]*)\}", text):
            end = text.find("\\endentry", match.end())
            body = text[match.end() : end if end > 0 else len(text)]
            title = ""
            if found := re.search(r"\\field\{title\}\{", body):
                start = found.end() - 1
                title = delatex(body[start : _balanced(body, start)])
            arxiv, doi = _ids_in(body)
            if (eprint := re.search(r"\\field\{eprint\}\{([^}]*)\}", body)) and arxiv_id_in(
                eprint.group(1)
            ):
                arxiv = arxiv_id_in(eprint.group(1))
            if verb := re.search(r"\\verb\{doi\}\s*\\verb\s*([^\n]*)", body):
                doi = verb.group(1).strip().lower()
            entries.append(BibEntry(match.group(1), title, arxiv, doi, delatex(body)[:300]))
        return entries
    starts = [m.start() for m in re.finditer(r"\\bibitem\b", text)]
    for begin, end in zip(starts, [*starts[1:], len(text)], strict=True):
        segment = text[begin + len("\\bibitem") : end].split("\\end{thebibliography}")[0]
        index = len(segment) - len(segment.lstrip())
        if index < len(segment) and segment[index] == "[":
            index = _balanced(segment, index, "[", "]")
        while index < len(segment) and segment[index].isspace():
            index += 1
        key = ""
        if index < len(segment) and segment[index] == "{":
            close = _balanced(segment, index)
            key = segment[index + 1 : close - 1]
            index = close
        body = segment[index:]
        arxiv, doi = _ids_in(body)
        entries.append(
            BibEntry(key.strip(), _title_from_bibitem(body), arxiv, doi, delatex(body)[:300])
        )
    return entries


def parse_bib(text: str) -> dict[str, BibEntry]:
    """Parse a ``.bib`` database into entries keyed by citation key."""
    entries: dict[str, BibEntry] = {}
    for match in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        if match.group(1).lower() in ("string", "comment", "preamble"):
            continue
        end = _balanced(text, text.find("{", match.start()))
        body = text[match.end() : end - 1]
        fields: dict[str, str] = {}
        for field_match in re.finditer(r"(\w+)\s*=\s*", body):
            name = field_match.group(1).lower()
            at = field_match.end()
            if at < len(body) and body[at] == "{":
                value = body[at + 1 : _balanced(body, at) - 1]
            elif at < len(body) and body[at] == '"':
                value = body[at + 1 : body.find('"', at + 1)]
            else:
                value = re.match(r"[^,\n]*", body[at:]).group(0)  # type: ignore[union-attr]
            fields.setdefault(name, value)
        arxiv, doi = _ids_in(
            " ".join(
                fields.get(n, "")
                for n in ("eprint", "journal", "url", "doi", "archiveprefix", "note", "booktitle")
            )
        )
        if fields.get("eprint") and arxiv_id_in(fields["eprint"]):
            arxiv = arxiv_id_in(fields["eprint"])
        if fields.get("doi"):
            doi = fields["doi"].strip().lower()
        entries[match.group(2)] = BibEntry(
            match.group(2), delatex(fields.get("title", "")), arxiv, doi, delatex(body)[:300]
        )
    return entries


def cited_keys(texts: list[str]) -> set[str]:
    """Every citation key the ``.tex`` files cite."""
    keys: set[str] = set()
    for text in texts:
        for match in _CITE_RE.finditer(re.sub(r"(?<!\\)%.*", "", text)):
            keys |= {k.strip() for k in match.group(1).split(",") if k.strip()}
    return keys


def bibliography(files: dict[str, str]) -> tuple[str, list[BibEntry]]:
    """Return the bibliography of an unpacked e-print and where it came from.

    A compiled ``.bbl`` (or a ``thebibliography`` block in the ``.tex``) is
    preferred; otherwise the ``.bib`` entries the text actually cites.

    Returns
    -------
    tuple[str, list[BibEntry]]
        ``("bbl" | "bib" | "none", entries)``.
    """
    texts = [v for k, v in files.items() if k.lower().endswith((".tex", ".ltx"))]
    compiled = [
        v
        for k, v in files.items()
        if k.lower().endswith(".bbl") and ("\\bibitem" in v or "\\entry{" in v)
    ]
    if not compiled:
        compiled = [
            t[t.find("\\begin{thebibliography}") :]
            for t in texts
            if "\\begin{thebibliography}" in t
        ]
    if compiled:
        return "bbl", [entry for block in compiled for entry in parse_bbl(block)]
    databases = [v for k, v in files.items() if k.lower().endswith(".bib")]
    if databases:
        merged: dict[str, BibEntry] = {}
        for database in databases:
            merged.update(parse_bib(database))
        keys = set(merged) if any("\\nocite{*}" in t for t in texts) else cited_keys(texts)
        return "bib", [merged[k] for k in sorted(keys) if k in merged]
    return "none", []


def citation_sentences(files: dict[str, str], keys: set[str], *, limit: int = 3) -> list[str]:
    """Sentences of the ``.tex`` that cite any of *keys* (at most *limit*)."""
    found: list[str] = []
    for name, text in files.items():
        if not name.lower().endswith((".tex", ".ltx")):
            continue
        text = re.sub(r"(?<!\\)%.*", "", text)
        for match in _CITE_RE.finditer(text):
            if not {k.strip() for k in match.group(1).split(",")} & keys:
                continue
            low = max(text.rfind("\n\n", 0, match.start()), match.start() - 400, 0)
            high = text.find("\n\n", match.end())
            high = min(high if high >= 0 else len(text), match.end() + 200)
            head = _SENTENCE_BREAK.split(text[low : match.start()])[-1]
            head = _STRUCTURE.split(head)[-1]
            tail = _SENTENCE_BREAK.split(text[match.start() : high])[0]
            sentence = delatex(head + tail)
            if sentence and sentence not in found:
                found.append(sentence[:400])
            if len(found) >= limit:
                return found
    return found


__all__ = [
    "BibEntry",
    "arxiv_id_in",
    "bibliography",
    "citation_sentences",
    "cited_keys",
    "delatex",
    "normalise",
    "parse_bbl",
    "parse_bib",
    "title_similarity",
    "unpack",
]
