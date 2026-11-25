"""What the request log says about how each service refuses.

For each quota bucket: requests, refusals (HTTP 429), transport failures, and
the refusal rate broken down by the gap since the previous request to the same
bucket and by how many requests the bucket sent in the trailing 1, 10 and 60
seconds.  A refusal rate that climbs with the trailing count points to a
windowed limit; a flat one points to refusals the client cannot pace away.
"""

from __future__ import annotations

import bisect
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GAP_BINS = (0.5, 1.0, 1.5, 2.0, 5.0, 10.0, float("inf"))
COUNT_WINDOWS = (1.0, 10.0, 60.0)


@dataclass
class Bin:
    """Requests and refusals in one bin."""

    requests: int = 0
    refused: int = 0

    def rate(self) -> str:
        """Refusal share as text."""
        return f"{100 * self.refused / self.requests:.0f}%" if self.requests else "-"


@dataclass
class ScopeReport:
    """Refusal statistics for one quota bucket."""

    scope: str
    requests: int = 0
    refused: int = 0
    transport: int = 0
    first: float = 0.0
    last: float = 0.0
    by_gap: dict[float, Bin] = field(default_factory=lambda: defaultdict(Bin))
    by_count: dict[tuple[float, int], Bin] = field(default_factory=lambda: defaultdict(Bin))


def read_log(paths: Iterable[Path], since: float | None = None) -> list[dict[str, Any]]:
    """Read request-log lines, oldest first; bad lines are skipped."""
    entries = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if since is None or float(entry.get("t", 0)) >= since:
                entries.append(entry)
    entries.sort(key=lambda e: float(e.get("t", 0)))
    return entries


def _count_bucket(n: int) -> int:
    for edge in (0, 1, 2, 3, 5, 10):
        if n <= edge:
            return edge
    return 11


def analyse(entries: list[dict[str, Any]]) -> dict[str, ScopeReport]:
    """Per-bucket refusal statistics."""
    by_scope: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_scope[str(entry.get("scope"))].append(entry)
    reports = {}
    for scope, rows in by_scope.items():
        report = ScopeReport(scope)
        times = [float(r["t"]) for r in rows]
        report.first, report.last = times[0], times[-1]
        for i, row in enumerate(rows):
            status = row.get("status")
            report.requests += 1
            refused = status == 429
            report.refused += refused
            report.transport += isinstance(status, str)
            if i:
                gap = times[i] - times[i - 1]
                edge = GAP_BINS[bisect.bisect_left(GAP_BINS, gap)]
                report.by_gap[edge].requests += 1
                report.by_gap[edge].refused += refused
            for window in COUNT_WINDOWS:
                n = i - bisect.bisect_left(times, times[i] - window, 0, i)
                key = (window, _count_bucket(n))
                report.by_count[key].requests += 1
                report.by_count[key].refused += refused
        reports[scope] = report
    return reports


def render_requests(reports: dict[str, ScopeReport]) -> str:
    """Render per-bucket refusal statistics."""
    lines = []
    for scope, r in sorted(reports.items(), key=lambda kv: -kv[1].requests):
        span = max(r.last - r.first, 1.0)
        lines.append(
            f"{scope}: {r.requests} requests over {span / 60:.0f} min "
            f"({r.requests / span:.2f}/s average), {r.refused} refused (429, "
            f"{100 * r.refused / r.requests:.0f}%), {r.transport} transport failures"
        )
        if r.requests < 20:
            continue
        gaps = "  ".join(
            f"<{edge:g}s {r.by_gap[edge].rate()} (n={r.by_gap[edge].requests})"
            for edge in GAP_BINS
            if r.by_gap[edge].requests
        )
        lines.append(f"  refused by gap since previous request: {gaps}")
        for window in COUNT_WINDOWS:
            cells = "  ".join(
                f"{'>10' if n == 11 else n} {b.rate()} (n={b.requests})"
                for (w, n), b in sorted(r.by_count.items())
                if w == window and b.requests
            )
            lines.append(f"  refused by requests in prior {window:g}s: {cells}")
    return "\n".join(lines) or "no requests logged yet"
