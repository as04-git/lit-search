"""Cross-process contention tests for the credential-scoped rate limiter.

These spawn genuine operating-system processes, because the defect being fixed
is invisible to any single-process test: the old limiter was a
``threading.Lock`` plus an instance float, which is perfectly correct inside one
interpreter and gives every additional process its own independent clock.  A
test that only used threads would have passed against the broken code.

The falsifier each test carries is stated in its docstring: with per-process
limiting the observed elapsed time would be roughly half, and the assertion
would fail.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess  # nosec B404: fixed argv, no shell, test-only.
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Worker: reserve N slots against one scope and report a wall-clock stamp for
# each.  Deliberately standalone so it can be exec'd in a fresh interpreter with
# no pytest, no fixtures, and no shared memory with the parent.
_WORKER = """
import json
import sys
import time

from lit_search.utils.rate_limit import RateLimitScope, reserve_slot

service, credential, count, interval = sys.argv[1:5]
scope = RateLimitScope.build(service, [credential] if credential else [])

stamps = []
for _ in range(int(count)):
    reserve_slot(scope, float(interval))
    stamps.append(time.time())

sys.stdout.write(json.dumps({"scope": scope.key, "stamps": stamps}))
"""

_INTERVAL = 0.25
_PER_PROCESS = 4


@pytest.fixture
def worker_script(tmp_path: Path) -> Path:
    """Write the standalone worker to disk and return its path.

    Returns
    -------
    Path
        Location of the worker script.
    """
    path = tmp_path / "rate_limit_worker.py"
    path.write_text(_WORKER, encoding="utf-8")
    return path


def _spawn(
    script: Path, lock_dir: Path, service: str, credential: str, count: int, interval: float
) -> subprocess.Popen[str]:
    """Start one worker process.

    Parameters
    ----------
    script : Path
        Worker script to execute.
    lock_dir : Path
        Shared rate-limit state directory for this test.
    service : str
        Service name the worker reserves against.
    credential : str
        Credential material, or ``""`` for the anonymous bucket.
    count : int
        Number of slots to reserve.
    interval : float
        Declared minimum interval.

    Returns
    -------
    subprocess.Popen[str]
        The running process, with stdout captured.
    """
    env = {
        **os.environ,
        "LIT_SEARCH_RATELIMIT_DIR": str(lock_dir),
        "PYTHONPATH": str(_REPO_ROOT),
    }
    return subprocess.Popen(  # nosec B603: fixed argv from test constants, no shell.
        [sys.executable, str(script), service, credential, str(count), str(interval)],
        cwd=str(_REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _collect(process: subprocess.Popen[str]) -> dict:
    """Wait for a worker and parse its JSON report.

    Parameters
    ----------
    process : subprocess.Popen[str]
        A process started by :func:`_spawn`.

    Returns
    -------
    dict
        The worker's decoded report.
    """
    stdout, stderr = process.communicate(timeout=120)
    assert process.returncode == 0, f"worker failed: {stderr}"
    return json.loads(stdout)


class TestCrossProcessContention:
    """Two operating-system processes contending for one credential."""

    def test_two_processes_sharing_a_credential_are_serialised(
        self, worker_script: Path, tmp_path: Path
    ) -> None:
        """Eight reservations across two processes are spaced by the interval.

        Falsifier: with the previous per-instance, in-memory limiter each
        process would pace its own four requests independently, finishing the
        whole run in about ``3 * interval`` instead of ``7 * interval``.  The
        elapsed-span assertion below fails under that behaviour.
        """
        lock_dir = tmp_path / "shared"
        procs = [
            _spawn(worker_script, lock_dir, "s2", "one-key", _PER_PROCESS, _INTERVAL)
            for _ in range(2)
        ]
        reports = [_collect(p) for p in procs]

        assert reports[0]["scope"] == reports[1]["scope"]

        stamps = sorted(reports[0]["stamps"] + reports[1]["stamps"])
        assert len(stamps) == 2 * _PER_PROCESS

        total_slots = 2 * _PER_PROCESS
        span = stamps[-1] - stamps[0]
        # Serialised across processes: (total - 1) gaps of one interval.
        assert span >= (total_slots - 1) * _INTERVAL * 0.95
        # Per-process pacing would land near this much smaller span.
        assert span > (_PER_PROCESS - 1) * _INTERVAL * 1.5

        gaps = [b - a for a, b in itertools.pairwise(stamps)]
        # Tolerance covers process-side overhead between the slot instant and
        # the timestamp; no gap may collapse toward zero, which is what a
        # missing shared clock produces.
        assert min(gaps) >= _INTERVAL * 0.8

    def test_processes_with_different_credentials_run_concurrently(
        self, worker_script: Path, tmp_path: Path
    ) -> None:
        """Separate keys on one service must not throttle each other.

        The limit belongs to the credential, so two keys hold two independent
        allowances.  Sharing one clock would be safe but needlessly halve
        throughput, and would make the scope the connector class rather than the
        credential.
        """
        lock_dir = tmp_path / "shared"
        procs = [
            _spawn(worker_script, lock_dir, "s2", "key-a", _PER_PROCESS, _INTERVAL),
            _spawn(worker_script, lock_dir, "s2", "key-b", _PER_PROCESS, _INTERVAL),
        ]
        reports = [_collect(p) for p in procs]

        assert reports[0]["scope"] != reports[1]["scope"]
        assert len({p.name for p in lock_dir.iterdir()}) == 2

        stamps = sorted(reports[0]["stamps"] + reports[1]["stamps"])
        span = stamps[-1] - stamps[0]
        # Each process still paces itself: about 3 intervals of work.  If the
        # two credentials had been merged into one bucket the span would be
        # about 7 intervals.
        assert span < (2 * _PER_PROCESS - 1) * _INTERVAL * 0.75

    def test_anonymous_processes_share_the_host_bucket(
        self, worker_script: Path, tmp_path: Path
    ) -> None:
        """Unkeyed processes converge on one clock per service.

        An unauthenticated quota is enforced against this host's source IP, so
        every credential-free process on the machine must queue behind the same
        clock.  Two independent anonymous clocks would send at twice the
        permitted rate.
        """
        lock_dir = tmp_path / "shared"
        procs = [
            _spawn(worker_script, lock_dir, "arxiv", "", _PER_PROCESS, _INTERVAL) for _ in range(2)
        ]
        reports = [_collect(p) for p in procs]

        assert reports[0]["scope"] == reports[1]["scope"] == "arxiv:anon"
        stamps = sorted(reports[0]["stamps"] + reports[1]["stamps"])
        span = stamps[-1] - stamps[0]
        assert span >= (2 * _PER_PROCESS - 1) * _INTERVAL * 0.95
