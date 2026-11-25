"""Reading the Kaggle arXiv metadata snapshot, with its provenance attached.

The owner wants complete arXiv machine-learning metadata and abstracts held
locally: roughly 605,681 records across ``cs.LG``, ``cs.AI``, ``cs.CL``,
``cs.CV``, ``cs.NE`` and ``stat.ML`` once the overlap between those six
categories is removed.  Metadata only -- full text is out of scope.

The reference is paperscraper's Kaggle backend
(``paperscraper/arxiv/kaggle.py``, version 1.0.0 read on 2026-08-31).  Its
shape is right and two things in it are not, and both are the kind of loss this
project exists to stop.

**1. No provenance survives the conversion.**  ``arxiv_kaggle`` writes JSONL
with ``title``, ``authors``, ``date``, ``abstract``, ``journal`` and ``doi`` and
records nothing about which snapshot produced them.  A snapshot is a photograph
of arXiv taken on one day; the dataset is republished weekly (verified: version
301 created 2026-08-29, 300 on 2026-08-22, 299 on 2026-08-15).  Without the
date, a paper posted after the snapshot is absent from every answer and its
absence is indistinguishable from "arXiv holds no such paper" -- which is the
motivating failure with the network removed.  :class:`SnapshotProvenance`
carries the dataset reference, the version number and the version's creation
date, and :class:`~lit_search.connectors.arxiv_bulk.ArxivBulkConnector` puts
them in the coverage detail of every result.

**2. The DOI normalisation keeps one identity and destroys the other.**
Verbatim from ``normalize_kaggle_record``::

    "doi": record.get("doi") or f"10.48550/arXiv.{arxiv_id}",

A record that names a journal DOI loses its arXiv identity; a record that does
not gets a synthetic arXiv DOI and never had a journal one.  Either way exactly
one identifier survives, and ``D24``'s version linking needs both -- the arXiv
preprint and its version of record are the same work, indexed under different
DOIs by different databases.  :func:`normalize_record` keeps both:
:attr:`ArxivRecord.arxiv_doi` is always populated and
:attr:`ArxivRecord.publisher_doi` is populated when the record has one.

**A third defect was found while reading that line and is worth naming**,
because it silently mints DOIs that do not exist.  The same function computes
``arxiv_id = str(record.get("id", "")).split("v")[0]``.  arXiv's old-style
identifiers carry the archive name, and one archive's name contains the letter
``v``: ``solv-int``.  So ``solv-int/9506005`` becomes ``sol``, and the record is
given the DOI ``10.48550/arXiv.sol``.  Checked live on 2026-08-31:
``https://doi.org/10.48550/arXiv.solv-int/9506005`` resolves (HTTP 200, to
``arxiv.org/abs/solv-int/9506005``) and ``https://doi.org/10.48550/arXiv.sol``
does not (HTTP 404).  :func:`strip_version_suffix` removes a trailing ``v<n>``
and nothing else.

**What this module deliberately does not do.**  It does not download from
Kaggle.  Downloading needs credentials, writes several gigabytes, and cannot be
exercised under ``pytest-socket``; more to the point, the download is where the
provenance is available, and a loader that fetched silently would have every
opportunity to lose it again.  :func:`write_manifest` records the provenance
beside the snapshot at download time, and everything here reads that manifest.
:func:`fetch_kaggle_provenance` builds one from Kaggle's public dataset-view
API, which needs no credential.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Kaggle dataset the snapshot comes from.
KAGGLE_DATASET_REF = "Cornell-University/arxiv"

#: Kaggle's public dataset-view endpoint.  Readable without a credential
#: (verified 2026-08-31); returns ``currentVersionNumber``, ``lastUpdated`` and
#: a ``versions`` list with a ``creationDate`` per version.
KAGGLE_VIEW_URL = f"https://www.kaggle.com/api/v1/datasets/view/{KAGGLE_DATASET_REF}"

#: Environment variable naming the snapshot file, or the directory holding it.
SNAPSHOT_PATH_ENV_VAR = "LIT_SEARCH_ARXIV_SNAPSHOT"

#: Suffix of the sidecar manifest written beside a snapshot.
MANIFEST_SUFFIX = ".manifest.json"

#: arXiv categories the owner asked for.  The count in the module docstring is
#: the union after deduplication, not the sum: a paper cross-listed to ``cs.LG``
#: and ``stat.ML`` is one record with both categories, and adding the six
#: category totals would roughly double-count it.
ARXIV_ML_CATEGORIES: frozenset[str] = frozenset(
    {"cs.LG", "cs.AI", "cs.CL", "cs.CV", "cs.NE", "stat.ML"}
)

#: Trailing ``v<n>`` on an arXiv identifier.  Anchored, and the digits are
#: required, so an archive name that merely contains ``v`` is untouched.
_VERSION_SUFFIX = re.compile(r"v\d+$")


class SnapshotUnavailableError(Exception):
    """The configured snapshot is missing, unreadable, or not a snapshot.

    Raised rather than returning an empty iterator: a snapshot that is not
    there holds no papers in exactly the way a database that is down holds no
    papers, and the caller has to be able to tell that apart from a query that
    matched nothing.
    """


# ----------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SnapshotProvenance:
    """Which snapshot a result came from, and how well that is known.

    Parameters
    ----------
    path : str
        Absolute path of the snapshot file that was read.
    size_bytes : int
        Its size on disk.
    dataset_ref : str | None
        Kaggle dataset reference, when a manifest recorded one.
    version_number : int | None
        Kaggle dataset version, when a manifest recorded one.  ``None`` means
        the version is unknown, which is a materially weaker claim than any
        particular number and must not be filled in with a guess.
    version_created_at : str | None
        ISO-8601 timestamp at which Kaggle published that version.
    downloaded_at : str | None
        ISO-8601 timestamp at which the operator downloaded it.
    file_mtime : str | None
        ISO-8601 modification time of the snapshot file itself.
    date_basis : str
        Which of the timestamps above :attr:`as_of` reports, spelled out so a
        reader is never left guessing whether a date came from the publisher or
        from a filesystem.
    """

    path: str
    size_bytes: int
    dataset_ref: str | None = None
    version_number: int | None = None
    version_created_at: str | None = None
    downloaded_at: str | None = None
    file_mtime: str | None = None
    date_basis: str = "unknown"

    @property
    def as_of(self) -> str | None:
        """Return the timestamp this snapshot is understood to represent.

        Returns
        -------
        str | None
            The dataset version's creation time when known, else the download
            time, else the file's modification time, else ``None``.  Read
            :attr:`date_basis` alongside it: the three are not equally good,
            and only the first is a statement by the publisher.
        """
        return self.version_created_at or self.downloaded_at or self.file_mtime

    def describe(self) -> str:
        """Return a one-line description for a coverage detail.

        Never returns a bare date.  Every rendering says what the date is a
        date *of*, because "snapshot of 2026-08-29" invites the reader to
        assume arXiv was complete on that day, and a file modification time
        supports no such reading.

        Returns
        -------
        str
            Description naming the dataset, the version and the date basis.
        """
        dataset = self.dataset_ref or "an unidentified arXiv metadata snapshot"
        version = (
            f"version {self.version_number}"
            if self.version_number is not None
            else "an unrecorded version"
        )
        stamp = self.as_of or "an unknown date"
        return f"{dataset}, {version}, {self.date_basis} {stamp}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, Any]
            Every field, including the ``None`` ones: an absent version has to
            stay visible as absent rather than vanish from the record.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, path: str, size_bytes: int) -> SnapshotProvenance:
        """Rebuild provenance from a manifest, keeping the observed file facts.

        *path* and *size_bytes* come from the file actually opened, not from
        the manifest, so a manifest copied next to a different file cannot
        misdescribe what was read.

        Parameters
        ----------
        data : dict[str, Any]
            Manifest contents.
        path : str
            Absolute path of the snapshot that was opened.
        size_bytes : int
            Its observed size.

        Returns
        -------
        SnapshotProvenance
            Provenance record.
        """
        version = data.get("version_number")
        return cls(
            path=path,
            size_bytes=size_bytes,
            dataset_ref=_optional_str(data.get("dataset_ref")),
            version_number=int(version) if isinstance(version, int) else None,
            version_created_at=_optional_str(data.get("version_created_at")),
            downloaded_at=_optional_str(data.get("downloaded_at")),
            file_mtime=_file_mtime(path),
            date_basis=(
                "published by Kaggle at"
                if _optional_str(data.get("version_created_at"))
                else "downloaded at"
                if _optional_str(data.get("downloaded_at"))
                else "last modified on disk at"
            ),
        )

    @classmethod
    def for_file(cls, path: str) -> SnapshotProvenance:
        """Read provenance for the snapshot at *path*.

        Uses the sidecar manifest when one exists.  When it does not, the only
        thing available is the file's modification time, and the result says
        so: a file that was copied, restored from a backup or touched carries a
        modification time that has nothing to do with arXiv, so it is reported
        as what it is rather than dressed up as a snapshot date.

        Parameters
        ----------
        path : str
            Absolute path of the snapshot file.

        Returns
        -------
        SnapshotProvenance
            Provenance record, with :attr:`date_basis` naming its own weakness
            when no manifest was found.
        """
        size = os.path.getsize(path)
        manifest_path = Path(f"{path}{MANIFEST_SUFFIX}")
        if manifest_path.is_file():
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.warning(
                    "arXiv snapshot manifest %s could not be read (%s); falling back to the "
                    "file's modification time, which is a much weaker provenance claim.",
                    manifest_path,
                    exc,
                )
            else:
                if isinstance(data, dict):
                    return cls.from_dict(data, path=path, size_bytes=size)
                logger.warning(
                    "arXiv snapshot manifest %s is not a JSON object; ignoring it.",
                    manifest_path,
                )
        return cls(
            path=path,
            size_bytes=size,
            file_mtime=_file_mtime(path),
            date_basis=(
                "no manifest was found beside the snapshot, so the only date available is "
                "the file's own modification time,"
            ),
        )


def write_manifest(
    snapshot_path: str,
    *,
    dataset_ref: str = KAGGLE_DATASET_REF,
    version_number: int | None = None,
    version_created_at: str | None = None,
    downloaded_at: str | None = None,
) -> str:
    """Write the sidecar manifest that gives a snapshot its provenance.

    Call this at download time, when the version is still known.  Recovering it
    afterwards is guesswork: the file itself carries no version marker.

    Parameters
    ----------
    snapshot_path : str
        Path of the snapshot the manifest describes.
    dataset_ref : str
        Kaggle dataset reference.
    version_number : int | None
        Kaggle version number.
    version_created_at : str | None
        ISO-8601 publication time of that version.
    downloaded_at : str | None
        ISO-8601 download time.  Defaults to now, in UTC.

    Returns
    -------
    str
        Path of the manifest written.
    """
    payload = {
        "dataset_ref": dataset_ref,
        "version_number": version_number,
        "version_created_at": version_created_at,
        "downloaded_at": downloaded_at
        or datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }
    manifest_path = f"{snapshot_path}{MANIFEST_SUFFIX}"
    Path(manifest_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def parse_kaggle_view(payload: dict[str, Any]) -> dict[str, Any]:
    """Turn Kaggle's dataset-view response into manifest fields.

    Kept separate from any HTTP so that it can be tested under
    ``pytest-socket`` against a recorded body.

    Parameters
    ----------
    payload : dict[str, Any]
        Decoded body of :data:`KAGGLE_VIEW_URL`.

    Returns
    -------
    dict[str, Any]
        ``dataset_ref``, ``version_number`` and ``version_created_at``.  A
        field Kaggle did not supply comes back ``None`` rather than being
        inferred from another one.
    """
    version = payload.get("currentVersionNumber")
    version_number = int(version) if isinstance(version, int) else None
    created: str | None = None
    versions = payload.get("versions")
    if isinstance(versions, list):
        for entry in versions:
            if isinstance(entry, dict) and entry.get("versionNumber") == version_number:
                created = _optional_str(entry.get("creationDate"))
                break
    return {
        "dataset_ref": _optional_str(payload.get("ref")) or KAGGLE_DATASET_REF,
        "version_number": version_number,
        "version_created_at": created or _optional_str(payload.get("lastUpdated")),
    }


# ----------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArxivRecord:
    """One normalised snapshot record, with both of its DOI identities.

    Parameters
    ----------
    arxiv_id : str
        Versionless arXiv identifier, new-style (``2106.09685``) or old-style
        (``solv-int/9506005``).
    arxiv_doi : str
        ``10.48550/arXiv.<id>``.  **Always populated.**  This is the preprint
        identity, and it is what OpenAlex, Semantic Scholar and Crossref index
        the preprint under, so dropping it is what makes a preprint and its
        published version look like two unrelated papers.
    publisher_doi : str | None
        The DOI the record itself names, when it names one.  This is the
        version of record: ``D24``'s "cite this".
    title : str
        Title, whitespace-normalised.
    abstract : str
        Abstract, whitespace-normalised.
    authors : tuple[str, ...]
        Author names, one per author, reconstructed from ``authors_parsed``
        when it is present and split from the ``authors`` string otherwise.
    categories : tuple[str, ...]
        arXiv categories, primary first.
    submitted : datetime.date | None
        Date of version 1.
    updated : datetime.date | None
        Date of the latest version.
    latest_version : int | None
        Number of the latest version present in the snapshot.
    version_count : int
        How many versions the snapshot records.
    journal_ref : str | None
        ``journal-ref``, when present.
    comments : str | None
        ``comments``, when present.
    license_url : str | None
        ``license``, when present.
    """

    arxiv_id: str
    arxiv_doi: str
    publisher_doi: str | None
    title: str
    abstract: str
    authors: tuple[str, ...]
    categories: tuple[str, ...]
    submitted: datetime.date | None
    updated: datetime.date | None
    latest_version: int | None
    version_count: int
    journal_ref: str | None
    comments: str | None
    license_url: str | None

    @property
    def citable_doi(self) -> str:
        """Return the DOI a bibliography should use.

        Returns
        -------
        str
            The publisher DOI when the record names one, otherwise the arXiv
            DOI.  Per ``D24`` the published record is the citable one; citing
            the preprint of a published paper silently disagrees with every
            other bibliography in the field.
        """
        return self.publisher_doi or self.arxiv_doi

    @property
    def preprint_doi(self) -> str | None:
        """Return the preprint DOI when it is a *second* identity.

        Returns
        -------
        str | None
            The arXiv DOI when the record also has a publisher DOI, otherwise
            ``None`` -- because when there is no publisher DOI the arXiv DOI is
            already the record's primary one and repeating it would claim two
            identities where there is one.
        """
        return self.arxiv_doi if self.publisher_doi else None

    @property
    def abs_url(self) -> str:
        """Return the arXiv abstract page URL.

        Returns
        -------
        str
            ``https://arxiv.org/abs/<id>``.
        """
        return f"https://arxiv.org/abs/{self.arxiv_id}"

    @property
    def pdf_url(self) -> str:
        """Return the arXiv PDF URL.

        Returns
        -------
        str
            ``https://arxiv.org/pdf/<id>``.
        """
        return f"https://arxiv.org/pdf/{self.arxiv_id}"

    def as_match_record(self) -> dict[str, tuple[str, ...]]:
        """Return this record as fields a local predicate can match against.

        Returns
        -------
        dict[str, tuple[str, ...]]
            Field name to values, matching
            :data:`~lit_search.query.builders.arxiv_bulk.FIELD_MAP`.
        """
        return {
            "title": (self.title,),
            "abstract": (self.abstract,),
            "authors": self.authors,
            "categories": self.categories,
            "journal": (self.journal_ref,) if self.journal_ref else (),
        }


def strip_version_suffix(arxiv_id: str) -> str:
    """Remove a trailing ``v<n>`` from an arXiv identifier and nothing else.

    This exists because the obvious implementation is wrong on real data.
    ``id.split("v")[0]`` -- paperscraper's -- truncates every old-style
    identifier whose archive name contains a ``v``, and one does:
    ``solv-int/9506005`` becomes ``sol``.  See the module docstring for the
    live check.

    Parameters
    ----------
    arxiv_id : str
        Identifier as it appears in the snapshot.

    Returns
    -------
    str
        Identifier without its version suffix.
    """
    return _VERSION_SUFFIX.sub("", arxiv_id.strip())


def arxiv_doi_for(arxiv_id: str) -> str:
    """Return the DataCite DOI arXiv assigns to *arxiv_id*.

    The form is ``10.48550/arXiv.<id>`` for both identifier styles.  Verified
    live on 2026-08-31: ``10.48550/arXiv.2106.09685`` and
    ``10.48550/arXiv.hep-th/9901001`` both resolve through ``doi.org`` to the
    matching ``arxiv.org/abs`` page.

    Parameters
    ----------
    arxiv_id : str
        Versionless identifier.

    Returns
    -------
    str
        The DOI.
    """
    return f"10.48550/arXiv.{arxiv_id}"


def normalize_record(raw: dict[str, Any]) -> ArxivRecord | None:
    """Normalise one raw snapshot line into an :class:`ArxivRecord`.

    Parameters
    ----------
    raw : dict[str, Any]
        One decoded JSON line from the snapshot.

    Returns
    -------
    ArxivRecord | None
        The record, or ``None`` when it carries no identifier or no title.
        Callers **must** count the ``None`` returns: a record the snapshot
        contained and this function could not read is a coverage loss, not an
        absent paper.
    """
    arxiv_id = strip_version_suffix(str(raw.get("id") or ""))
    title = _clean(raw.get("title"))
    if not arxiv_id or not title:
        return None

    raw_versions = raw.get("versions")
    versions: list[Any] = raw_versions if isinstance(raw_versions, list) else []
    submitted, updated, latest = _version_facts(versions)
    if updated is None:
        updated = _iso_date(raw.get("update_date"))

    return ArxivRecord(
        arxiv_id=arxiv_id,
        arxiv_doi=arxiv_doi_for(arxiv_id),
        publisher_doi=_clean(raw.get("doi")) or None,
        title=title,
        abstract=_clean(raw.get("abstract")),
        authors=_authors(raw),
        categories=tuple(str(raw.get("categories") or "").split()),
        submitted=submitted,
        updated=updated,
        latest_version=latest,
        version_count=len(versions),
        journal_ref=_clean(raw.get("journal-ref")) or None,
        comments=_clean(raw.get("comments")) or None,
        license_url=_clean(raw.get("license")) or None,
    )


@dataclass(slots=True)
class SnapshotScan:
    """Counters for one pass over a snapshot.

    Attributes
    ----------
    lines : int
        Non-blank lines read.
    unreadable : int
        Lines that were not valid JSON objects.
    unparsed : int
        Decoded records that :func:`normalize_record` could not turn into an
        :class:`ArxivRecord`.
    in_scope : int
        Records inside the requested categories.
    """

    lines: int = 0
    unreadable: int = 0
    unparsed: int = 0
    in_scope: int = 0

    @property
    def lost(self) -> int:
        """Return how many lines the scan could not turn into a record.

        Returns
        -------
        int
            Unreadable plus unparsed.
        """
        return self.unreadable + self.unparsed


def iter_records(
    path: str,
    *,
    categories: frozenset[str] | None = None,
    scan: SnapshotScan | None = None,
) -> Iterator[ArxivRecord]:
    """Stream the snapshot, yielding records inside *categories*.

    The file is several gigabytes, so it is read line by line and never held in
    memory.

    Parameters
    ----------
    path : str
        Snapshot file.
    categories : frozenset[str] | None
        Categories to keep.  ``None`` keeps every record.  A record is kept
        when *any* of its categories is in the set, which is what makes the
        six-category union the owner asked for one pass rather than six.
    scan : SnapshotScan | None
        Counters to update.  Supply one: without it, records the scan could not
        read are invisible, and invisible losses are the thing this project is
        built against.

    Yields
    ------
    ArxivRecord
        Each in-scope record, in file order.

    Raises
    ------
    SnapshotUnavailableError
        When the file cannot be opened.
    """
    counters = scan if scan is not None else SnapshotScan()
    try:
        handle = open(path, encoding="utf-8", errors="replace")  # noqa: SIM115
    except OSError as exc:
        raise SnapshotUnavailableError(
            f"The arXiv snapshot at {path} could not be opened: {exc}. Reporting this as "
            f"zero papers would say arXiv holds nothing on the subject."
        ) from exc

    with handle:
        for line in handle:
            if not line.strip():
                continue
            counters.lines += 1
            record = _decode(line, counters)
            if record is None:
                continue
            if categories is not None and not categories.intersection(record.categories):
                continue
            counters.in_scope += 1
            yield record


def _decode(line: str, counters: SnapshotScan) -> ArxivRecord | None:
    """Decode and normalise one line, counting both failure modes separately.

    Parameters
    ----------
    line : str
        Raw line from the snapshot.
    counters : SnapshotScan
        Counters to update in place.

    Returns
    -------
    ArxivRecord | None
        The record, or ``None`` when the line was unreadable or unparsable.
    """
    try:
        raw = json.loads(line)
    except ValueError:
        counters.unreadable += 1
        return None
    if not isinstance(raw, dict):
        counters.unreadable += 1
        return None
    record = normalize_record(raw)
    if record is None:
        counters.unparsed += 1
    return record


def resolve_snapshot_path(configured: str | None = None) -> str | None:
    """Return the snapshot file to read, or ``None`` when none is configured.

    Parameters
    ----------
    configured : str | None
        Explicit path.  Falls back to :data:`SNAPSHOT_PATH_ENV_VAR`.

    Returns
    -------
    str | None
        Absolute path of a readable file, or ``None``.  A configured path that
        names a directory is searched for a single ``*.json`` file; a directory
        holding several is **not** resolved by picking the largest, which is
        what paperscraper does and what silently reads last month's snapshot
        when this month's download was interrupted.
    """
    raw = configured or os.environ.get(SNAPSHOT_PATH_ENV_VAR, "").strip()
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    if candidate.is_dir():
        matches = sorted(p for p in candidate.glob("*.json") if p.is_file())
        if len(matches) == 1:
            return str(matches[0].resolve())
        logger.warning(
            "arXiv snapshot directory %s holds %d '*.json' files; refusing to guess which "
            "is the snapshot. Point %s at the file itself.",
            candidate,
            len(matches),
            SNAPSHOT_PATH_ENV_VAR,
        )
        return None
    return None


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------


def _clean(value: Any) -> str:
    """Collapse whitespace in a possibly absent string field.

    Parameters
    ----------
    value : Any
        Raw field value.

    Returns
    -------
    str
        Whitespace-normalised text, empty when the field was absent.
    """
    return " ".join(str(value or "").split())


def _optional_str(value: Any) -> str | None:
    """Return a non-blank string, or ``None``.

    Parameters
    ----------
    value : Any
        Raw value.

    Returns
    -------
    str | None
        Stripped string, or ``None`` when absent or blank.
    """
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _file_mtime(path: str) -> str | None:
    """Return a file's modification time as an ISO-8601 UTC string.

    Parameters
    ----------
    path : str
        File path.

    Returns
    -------
    str | None
        Timestamp, or ``None`` when the file cannot be stat-ed.
    """
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        return None
    return (
        datetime.datetime.fromtimestamp(stamp, tz=datetime.UTC).replace(microsecond=0).isoformat()
    )


def _authors(raw: dict[str, Any]) -> tuple[str, ...]:
    """Extract author names, preferring the structured field.

    ``authors_parsed`` is a list of ``[last, first, suffix]`` triples and is
    unambiguous.  The ``authors`` string is free text with "and", commas and
    LaTeX in it, so it is only used when the structured field is absent.

    Parameters
    ----------
    raw : dict[str, Any]
        Raw snapshot record.

    Returns
    -------
    tuple[str, ...]
        Author names.
    """
    parsed = raw.get("authors_parsed")
    if isinstance(parsed, list) and parsed:
        names = []
        for entry in parsed:
            if not isinstance(entry, list):
                continue
            parts = [str(part).strip() for part in entry if str(part).strip()]
            if not parts:
                continue
            last = parts[0]
            rest = " ".join(parts[1:])
            names.append(f"{rest} {last}".strip())
        if names:
            return tuple(names)
    flat = _clean(raw.get("authors"))
    if not flat:
        return ()
    return tuple(part.strip() for part in re.split(r",| and ", flat) if part.strip())


def _version_facts(
    versions: list[Any],
) -> tuple[datetime.date | None, datetime.date | None, int | None]:
    """Extract submission date, latest date and latest version number.

    Parameters
    ----------
    versions : list[Any]
        The record's ``versions`` list.

    Returns
    -------
    tuple[datetime.date | None, datetime.date | None, int | None]
        Date of v1, date of the latest version, and that version's number.
    """
    dates: list[tuple[int, datetime.date]] = []
    for entry in versions:
        if not isinstance(entry, dict):
            continue
        number = _version_number(entry.get("version"))
        created = _rfc2822_date(entry.get("created"))
        if number is not None and created is not None:
            dates.append((number, created))
    if not dates:
        return None, None, None
    dates.sort()
    return dates[0][1], dates[-1][1], dates[-1][0]


def _version_number(value: Any) -> int | None:
    """Parse ``"v3"`` into ``3``.

    Parameters
    ----------
    value : Any
        Raw version label.

    Returns
    -------
    int | None
        Version number, or ``None`` when unparsable.
    """
    text = str(value or "").strip().lstrip("vV")
    return int(text) if text.isdigit() else None


def _rfc2822_date(value: Any) -> datetime.date | None:
    """Parse an RFC 2822 timestamp into a date.

    Parameters
    ----------
    value : Any
        Raw timestamp, as ``versions[].created`` supplies it.

    Returns
    -------
    datetime.date | None
        Parsed date, or ``None`` when unparsable.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return parsedate_to_datetime(value).date()
    except (TypeError, ValueError):
        return None


def _iso_date(value: Any) -> datetime.date | None:
    """Parse a ``YYYY-MM-DD`` string into a date.

    Parameters
    ----------
    value : Any
        Raw value, as ``update_date`` supplies it.

    Returns
    -------
    datetime.date | None
        Parsed date, or ``None`` when unparsable.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.date.fromisoformat(value.strip())
    except ValueError:
        return None
