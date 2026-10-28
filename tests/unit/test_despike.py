"""
Tests for the canvas-resolution despike applied by `_run_cell` when a step sets
`filter.max_deviation` (or `max_depth_change`).

Background: the fused WLIS DEM (policy 5577efa5) carried a two-pixel spike of -193 m
and -161 m at the Mount Sinai Harbor inlet, amid water 5 to 10 m deep. The source COG
(New_England_Coned_Topobathy_DEM_2016_6194) holds a smooth bowl about 40 m across that
reaches -290 m at 1 m resolution, and nothing in the pipeline removed it.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
import yaml
from affine import Affine

import topobathykit.runtime as runtime
from topobathykit.providers.base import Provider, despike_median_deviation
from topobathykit.providers.registry import registry


def _da(values: np.ndarray) -> xr.DataArray:
    ny, nx = values.shape
    return xr.DataArray(
        values.astype(np.float32), coords={"y": np.arange(ny), "x": np.arange(nx)}, dims=("y", "x")
    )


def test_isolated_spike_is_masked() -> None:
    z = np.full((9, 9), -8.0)
    z[4, 4] = -193.0
    out, n = despike_median_deviation(_da(z), threshold=20.0)
    assert n == 1
    assert np.isnan(out.values[4, 4])
    assert np.isfinite(np.delete(out.values.ravel(), 4 * 9 + 4)).all()


def test_adjacent_pair_is_masked() -> None:
    # The Mount Sinai case at 30 m: two neighbouring deep pixels.
    z = np.full((9, 9), -8.0)
    z[4, 4] = -193.3
    z[3, 4] = -161.3
    out, n = despike_median_deviation(_da(z), threshold=20.0)
    assert n == 2
    assert np.isnan(out.values[4, 4]) and np.isnan(out.values[3, 4])


def test_broad_steep_feature_is_kept() -> None:
    # A channel three pixels wide, 30 m below its banks, and a straight cliff: real
    # relief that a 3x3 median follows, so nothing is removed.
    z = np.full((12, 12), -5.0)
    z[:, 4:7] = -35.0
    z[:, 9:] = 20.0
    out, n = despike_median_deviation(_da(z), threshold=20.0)
    assert n == 0
    np.testing.assert_array_equal(out.values, z.astype(np.float32))


def test_nan_is_ignored_and_sparse_windows_are_not_judged() -> None:
    z = np.full((7, 7), np.nan)
    z[3, 3] = -150.0
    z[3, 4] = -8.0
    z[2, 3] = -8.0
    # Three finite values in the centre window: too few to judge, so it stays.
    out, n = despike_median_deviation(_da(z), threshold=20.0)
    assert n == 0
    assert out.values[3, 3] == -150.0
    # With a full neighbourhood around it the same pixel goes, and NaN stays NaN.
    z[2:5, 2:5] = -8.0
    z[3, 3] = -150.0
    out, n = despike_median_deviation(_da(z), threshold=20.0)
    assert n == 1
    assert np.isnan(out.values[3, 3])
    assert np.isnan(out.values[0, 0])


def test_non_positive_threshold_is_a_no_op() -> None:
    z = np.full((5, 5), -8.0)
    z[2, 2] = -200.0
    out, n = despike_median_deviation(_da(z), threshold=0.0)
    assert n == 0
    assert out.values[2, 2] == -200.0


# --- _run_cell integration ---------------------------------------------------------

HYDRATE_BBOX = (-73.49, 40.91, -73.46, 40.94)
CELL_BBOX = runtime._get_grid_cells(HYDRATE_BBOX, "EPSG:4326")[0][0]
RES_M = 500.0
SPIKE = -193.0


class _SpikedSeabed(Provider):
    """A flat -8 m seabed with a two-pixel deep spike near the centre of the request.

    The grid is the runtime's EPSG:4326 canvas grid itself (resolution / 111320 degree
    cells snapped to multiples of the cell size), so canvas alignment is an identity
    whatever resampling it uses, and the spike reaches the canvas as exactly the two
    pixels it occupies here: the Mount Sinai case at 30 m.
    """

    def fetch_layer(self, bbox: Any, resolution: Any = None, crs: str = "EPSG:4326", **kw: Any) -> Any:
        min_lon, min_lat, max_lon, max_lat = bbox
        res = float(resolution) / 111320.0
        x0 = int(np.floor(min_lon / res))
        x1 = int(np.ceil(max_lon / res))
        y0 = int(np.floor(min_lat / res))
        y1 = int(np.ceil(max_lat / res))
        nx, ny = x1 - x0, y1 - y0
        z = np.full((ny, nx), -8.0, dtype=np.float32)
        z[ny // 2, nx // 2] = SPIKE
        z[ny // 2 - 1, nx // 2] = SPIKE + 32.0
        da = xr.DataArray(
            z,
            coords={
                "y": (np.arange(y1 - 1, y0 - 1, -1) + 0.5) * res,
                "x": (np.arange(x0, x1) + 0.5) * res,
            },
            dims=("y", "x"),
        )
        da.rio.write_crs(crs, inplace=True)
        da.rio.write_transform(Affine.translation(x0 * res, y1 * res) @ Affine.scale(res, -res), inplace=True)
        return da

    def get_metadata(self) -> dict[str, Any]:
        return {"name": "spiked"}


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOPOBATHYKIT_CACHE_DIR", str(tmp_path / "cache"))
    registry.register("it_spiked", _SpikedSeabed)


def _policy(tmp_path: Path, provider: str, step_filter: dict[str, Any] | None) -> Any:
    step: dict[str, Any] = {"provider": provider, "operation": "overwrite"}
    if step_filter is not None:
        step["filter"] = step_filter
    content = {
        "name": "DespikeTest",
        "crs": "EPSG:4326",
        "variables": [{"name": "elevation", "steps": [step]}],
    }
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.dump(content))
    return runtime._resolve_policy(str(path))


def _run(policy: Any) -> tuple[xr.Dataset, dict[str, Any]]:
    ds = runtime._run_cell(policy, CELL_BBOX, RES_M, None, "EPSG:4326")
    outcome = json.loads(ds.attrs["step_outcomes_json"])[0]
    return ds, outcome


def test_run_cell_keeps_the_spike_without_a_filter(tmp_path: Path) -> None:
    ds, outcome = _run(_policy(tmp_path, "it_spiked", None))
    assert float(ds["elevation"].min()) == pytest.approx(SPIKE)
    assert "despiked" not in outcome


def test_run_cell_despikes_when_max_deviation_is_set(tmp_path: Path) -> None:
    ds, outcome = _run(_policy(tmp_path, "it_spiked", {"max_deviation": 20.0}))
    assert float(ds["elevation"].min()) > -9.0
    assert outcome["status"] == "ok"
    assert outcome["despiked"] == 2


def test_run_cell_does_not_refilter_ncei_bag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # ncei_bag applies these thresholds itself at native resolution.
    monkeypatch.setitem(registry._providers, "ncei_bag", _SpikedSeabed)
    ds, outcome = _run(_policy(tmp_path, "ncei_bag", {"max_deviation": 20.0}))
    assert float(ds["elevation"].min()) == pytest.approx(SPIKE)
    assert "despiked" not in outcome
