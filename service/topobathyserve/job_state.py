"""
Persistent job state for hydration tasks.

Each job writes its state to a JSON file at:
    ~/.cache/topobathykit/hydration_jobs/{job_id}.json

Design constraints:
- Single writer (the hydration subprocess) per job file.
- Multiple readers (GET endpoint, WebSocket handler) may read concurrently.
- Writes are atomic: write to .tmp then os.replace() (POSIX-atomic on same filesystem).
- Readers never get a partial/corrupt file — they either see the old or new version.
- Dead subprocess detection: reader checks if the recorded PID is still alive.
"""

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from topobathykit.config import get_cache_root

logger = logging.getLogger(__name__)


def _ensure_jobs_dir() -> Path:
    jobs_dir = get_cache_root() / "hydration_jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    return jobs_dir


def job_path(job_id: str) -> Path:
    return _ensure_jobs_dir() / f"{job_id}.json"


def write_state(job_id: str, state: dict[str, Any]) -> None:
    """Atomically write job state to disk."""
    path = job_path(job_id)
    tmp = path.with_suffix(".json.tmp")
    data = json.dumps(state, default=str)
    tmp.write_text(data)
    os.replace(str(tmp), str(path))  # atomic on POSIX


def read_state(job_id: str) -> dict[str, Any] | None:
    """Read job state from disk, detecting dead subprocesses."""
    path = job_path(job_id)
    if not path.exists():
        return None
    try:
        state: dict[str, Any] = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    return _reconcile_dead_worker(job_id, state)


def _reconcile_dead_worker(job_id: str, state: dict[str, Any]) -> dict[str, Any]:
    """Mark a running/pending job as failed when its worker process is gone.

    A worker killed by the kernel OOM killer or by a container restart never
    writes a terminal state itself, so without this the job would stay
    `running` indefinitely.
    """
    if state.get("status") not in ("running", "pending"):
        return state
    pid = state.get("pid")
    if not pid:
        return state
    reason = _worker_dead_reason(int(pid), state.get("pid_start_ticks"))
    if reason is None:
        return state

    # Re-read before writing: the worker may have recorded completion between
    # our read and the liveness check, and that must not be overwritten.
    try:
        current: dict[str, Any] = json.loads(job_path(job_id).read_text())
    except (json.JSONDecodeError, OSError):
        current = state
    if current.get("status") not in ("running", "pending"):
        return current

    current["status"] = "failed"
    current["failure_reason"] = "worker_died"
    current["error"] = (
        f"Hydration worker process {pid} {reason} without recording completion "
        "(likely killed by the OOM killer or a container restart)"
    )
    current["finished_at"] = datetime.now(UTC).isoformat()
    # Update the file so subsequent reads don't re-check
    write_state(job_id, current)
    return current


def list_jobs(max_age_hours: float = 24) -> list[dict[str, Any]]:
    """List recent jobs, pruning expired ones."""
    _ensure_jobs_dir()
    now = datetime.now(UTC)
    jobs = []
    for f in _ensure_jobs_dir().glob("*.json"):
        try:
            state = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        state = _reconcile_dead_worker(state.get("id") or f.stem, state)
        # Prune old completed/failed jobs
        submitted = state.get("submitted_at", "")
        try:
            submitted_dt = datetime.fromisoformat(submitted)
            if submitted_dt.tzinfo is None:
                submitted_dt = submitted_dt.replace(tzinfo=UTC)
            age_hours = (now - submitted_dt).total_seconds() / 3600
            if age_hours > max_age_hours and state.get("status") in ("completed", "failed"):
                f.unlink(missing_ok=True)
                continue
        except (ValueError, TypeError):
            pass
        jobs.append(state)
    return jobs


def _pid_alive(pid: int) -> bool:
    """Check if a process is still running. Safe, no signals sent."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it — still alive
        return True


def _read_proc_stat(pid: int) -> tuple[str, int] | None:
    """Return (state letter, start time in clock ticks since boot) from /proc/<pid>/stat.

    Returns None where /proc is unavailable or the process does not exist.
    """
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # The command name (field 2) is parenthesised and may contain spaces, so split
    # after its closing parenthesis. The fields after it start at field 3 (state);
    # starttime is field 22.
    try:
        rest = raw[raw.rindex(")") + 2 :].split()
        return rest[0], int(rest[19])
    except (ValueError, IndexError):
        return None


def process_start_ticks(pid: int) -> int | None:
    """Start time of `pid` in clock ticks since boot, recorded to detect PID reuse."""
    stat = _read_proc_stat(pid)
    return stat[1] if stat else None


def _worker_dead_reason(pid: int, start_ticks: int | None = None) -> str | None:
    """Return why the worker `pid` should be treated as dead, or None if it is alive.

    Besides a missing PID, this catches two cases that `os.kill(pid, 0)` reports
    as alive: a zombie (the worker exited but the server has not reaped it yet),
    and a PID reused by an unrelated process after a container restart, detected
    by comparing the recorded start time.
    """
    if not _pid_alive(pid):
        return "is no longer running"
    stat = _read_proc_stat(pid)
    if stat is None:
        return None
    state_letter, ticks = stat
    if state_letter in ("Z", "X"):
        return "exited (zombie)"
    if start_ticks is not None and ticks != int(start_ticks):
        return "is no longer running (PID reused by another process)"
    return None
