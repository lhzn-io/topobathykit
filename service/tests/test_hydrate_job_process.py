"""How POST /hydrate starts its job process."""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from topobathyserve import main


class _FakeProcess:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.started = False
        self.pid = 4242

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return self.started


class _RecordingContext:
    def __init__(self) -> None:
        self.processes: list[_FakeProcess] = []

    def Process(self, **kwargs: Any) -> _FakeProcess:  # noqa: N802 - mirrors multiprocessing
        proc = _FakeProcess(**kwargs)
        self.processes.append(proc)
        return proc


def test_hydration_context_is_spawn() -> None:
    assert main._HYDRATE_MP_CONTEXT.get_start_method() == "spawn"


def test_hydrate_starts_a_non_daemon_job_process_from_the_spawn_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TOPOBATHYKIT_CACHE_DIR", str(tmp_path / "cache"))
    ctx = _RecordingContext()
    monkeypatch.setattr(main, "_HYDRATE_MP_CONTEXT", ctx)
    monkeypatch.setattr(main, "_HYDRATE_PROCESSES", {})
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        "name: JobProcessTest\ncrs: EPSG:4326\nvariables:\n  - name: elevation\n    steps: []\n"
    )
    monkeypatch.setitem(main.app.dependency_overrides, main.get_policy_path, lambda: policy_path)

    response = TestClient(main.app).post(
        "/hydrate", json={"bbox": [-72.02, 41.22, -71.98, 41.26], "resolution": 30}
    )

    assert response.status_code == 200, response.text
    assert len(ctx.processes) == 1
    proc = ctx.processes[0]
    assert proc.started
    assert proc.kwargs["target"] is main._hydrate_subprocess
    assert proc.kwargs["daemon"] is False
