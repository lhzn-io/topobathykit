"""Tests for hydration job terminal-state detection when the worker process dies."""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from topobathyserve import job_state

pytestmark = pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="requires Linux /proc")


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOPOBATHYSIM_CACHE_DIR", str(tmp_path))


def _running_state(job_id: str, pid: int, **extra: Any) -> dict[str, Any]:
    state = {
        "id": job_id,
        "status": "running",
        "submitted_at": "2026-09-28T15:00:00+00:00",
        "pid": pid,
        "failed_cells": 0,
    }
    state.update(extra)
    job_state.write_state(job_id, state)
    return state


def _exited_pid() -> int:
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def test_live_worker_stays_running() -> None:
    pid = os.getpid()
    _running_state("live", pid, pid_start_ticks=job_state.process_start_ticks(pid))

    state = job_state.read_state("live")

    assert state is not None
    assert state["status"] == "running"


def test_exited_worker_is_reported_failed_with_reason() -> None:
    _running_state("dead", _exited_pid())

    state = job_state.read_state("dead")

    assert state is not None
    assert state["status"] == "failed"
    assert state["failure_reason"] == "worker_died"
    assert "without recording completion" in state["error"]
    # The terminal state is persisted, not just reported.
    assert '"status": "failed"' in job_state.job_path("dead").read_text()


def test_zombie_worker_is_reported_failed() -> None:
    """A worker killed by the OOM killer stays a zombie until the server reaps it."""
    proc = subprocess.Popen(["true"])
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            stat = job_state._read_proc_stat(proc.pid)
            if stat is not None and stat[0] == "Z":
                break
            time.sleep(0.05)
        else:
            pytest.fail("child did not become a zombie")
        assert job_state._pid_alive(proc.pid), "os.kill(pid, 0) alone reports a zombie as alive"

        _running_state("zombie", proc.pid)
        state = job_state.read_state("zombie")

        assert state is not None
        assert state["status"] == "failed"
        assert "zombie" in state["error"]
    finally:
        proc.wait()


def test_reused_pid_is_reported_failed() -> None:
    """After a container restart the recorded PID can belong to an unrelated process."""
    pid = os.getpid()
    ticks = job_state.process_start_ticks(pid)
    assert ticks is not None
    _running_state("reused", pid, pid_start_ticks=ticks - 1)

    state = job_state.read_state("reused")

    assert state is not None
    assert state["status"] == "failed"
    assert "PID reused" in state["error"]


def test_list_jobs_reports_dead_workers() -> None:
    _running_state("listed", _exited_pid())

    jobs = {j["id"]: j for j in job_state.list_jobs(max_age_hours=24 * 365 * 10)}

    assert jobs["listed"]["status"] == "failed"


def test_terminal_states_are_left_alone() -> None:
    job_state.write_state("done", {"id": "done", "status": "completed", "pid": _exited_pid()})

    state = job_state.read_state("done")

    assert state is not None
    assert state["status"] == "completed"
    assert "error" not in state
