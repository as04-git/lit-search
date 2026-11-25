"""Utilities for resolving package version information."""

from __future__ import annotations

import tomllib
from importlib import metadata
from pathlib import Path


def package_version() -> str:
    """Resolve the package version for metadata.

    Returns
    -------
    str
        Version string.
    """
    try:
        return metadata.version("lit_search")
    except metadata.PackageNotFoundError:
        return version_from_pyproject()


def version_from_pyproject() -> str:
    """Read version from pyproject.toml when package metadata is unavailable.

    Returns
    -------
    str
        Version string or "unknown" if missing.
    """
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject_path.exists():
        return "unknown"
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)
    return str(data.get("project", {}).get("version", "unknown"))
