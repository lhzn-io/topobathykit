"""
Unit test: BAG coverage void filled by BlueTopo base layer.

Regression guard for the W00206 scenario — a BAG survey with a spatial
coverage gap (NaN region wider than the runtime's interpolation limit)
that must be filled by the underlying BlueTopo layer rather than
propagating as a data void.

Note on design: the runtime applies interpolate_na(limit=2) at the mosaic
step to heal single-pixel cell-boundary artefacts.  A genuine survey gap
spans many more cells, so the mock BAG provider returns NaN over the entire
western half of the bbox — well beyond the 2-cell interpolation limit —
ensuring the base layer is the only possible fill source.
"""

from typing import Any

import numpy as np
import pytest
import xarray as xr
import yaml
from affine import Affine

from topobathykit.providers.base import Provider
from topobathykit.providers.registry import registry
from topobathykit.runtime import run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BBOX = (-70.620, 42.960, -70.580, 43.000)  # ~4 km patch — wide enough for a clear gap at 100 m res


def _make_grid(
    bbox: tuple[float, float, float, float],
    value: float,
    crs: str = "EPSG:4326",
    size: int = 10,
) -> xr.DataArray:
    """Return a uniform size x size DataArray covering bbox."""
    west, south, east, north = bbox
    res_x = (east - west) / size
    res_y = (north - south) / size
    transform = Affine.translation(west, north) * Affine.scale(res_x, -res_y)
    xs = np.linspace(west + res_x / 2, east - res_x / 2, size)
    ys = np.linspace(north - res_y / 2, south + res_y / 2, size)
    data = np.full((size, size), value, dtype=np.float32)
    da = xr.DataArray(data, coords={"y": ys, "x": xs}, dims=("y", "x"))
    da.rio.write_crs(crs, inplace=True)
    da.rio.write_transform(transform, inplace=True)
    return da


# ---------------------------------------------------------------------------
# Mock providers
# ---------------------------------------------------------------------------


class MockBlueTopo(Provider):
    """Solid base layer — valid bathymetry everywhere (-15 m)."""

    DEPTH = -15.0

    def fetch_layer(
        self,
        bbox: tuple[float, float, float, float],
        resolution: float | None = None,
        crs: str = "EPSG:4326",
        **kwargs: Any,
    ) -> xr.DataArray:
        return _make_grid(bbox, self.DEPTH, crs)

    def get_metadata(self) -> dict[str, Any]:
        return {"name": "MockBlueTopo"}


class MockBAGWithCoverageGap(Provider):
    """
    BAG survey with a large coverage void west of LON_SPLIT.

    The eastern sub-region has valid multibeam data (-20 m); everything
    west of LON_SPLIT is NaN — simulating a survey swath that simply has no
    coverage.  The split is anchored to an absolute longitude so the void
    spans multiple runtime cells regardless of how the bbox is tiled.
    """

    SURVEY_DEPTH = -20.0
    # Split at 40% into the bbox: -70.620 + 0.40 * 0.040 = -70.604
    LON_SPLIT = -70.604

    def fetch_layer(
        self,
        bbox: tuple[float, float, float, float],
        resolution: float | None = None,
        crs: str = "EPSG:4326",
        **kwargs: Any,
    ) -> xr.DataArray:
        da = _make_grid(bbox, self.SURVEY_DEPTH, crs)
        # Mask cells whose centre longitude is west of the split — absolute coords.
        da.values[:, da.x.values < self.LON_SPLIT] = np.nan
        return da

    def get_metadata(self) -> dict[str, Any]:
        return {"name": "MockBAGWithCoverageGap"}


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.fixture
def bag_hole_policy(tmp_path: Any) -> str:
    """Policy: BlueTopo as base, BAG overwrites — gap should fall back to BlueTopo."""
    policy = {
        "name": "BAG Coverage Gap Fill Test",
        "crs": "EPSG:4326",
        "variables": [
            {
                "name": "elevation",
                "steps": [
                    {"provider": "mock_bluetopo_hole_fill", "operation": "overwrite"},
                    {"provider": "mock_bag_coverage_gap", "operation": "overwrite"},
                ],
            }
        ],
    }
    path = tmp_path / "bag_hole_policy.yaml"
    path.write_text(yaml.dump(policy))
    return str(path)


def test_bag_coverage_gap_filled_by_bluetopo(bag_hole_policy: str) -> None:
    """
    When a BAG survey has a large coverage void (NaN region wider than the
    runtime's interpolate_na limit), the underlying BlueTopo base layer must
    fill those cells rather than leaving a NaN void that would form an
    infinite IBM wall in Oceananigans.
    """
    registry.register("mock_bluetopo_hole_fill", MockBlueTopo)
    registry.register("mock_bag_coverage_gap", MockBAGWithCoverageGap)

    ds = run(bag_hole_policy, BBOX, resolution=100.0, use_cache=False)

    assert "elevation" in ds
    elev = ds["elevation"]

    # No NaNs should survive in the output — BlueTopo fills the BAG gap
    nan_count = int(np.isnan(elev.values).sum())
    assert nan_count == 0, (
        f"Expected 0 NaN cells after gap fill, got {nan_count}. "
        "BlueTopo base layer did not cover the BAG coverage void."
    )

    centre_y = (BBOX[1] + BBOX[3]) / 2

    # West of LON_SPLIT (BAG gap) should carry the BlueTopo fill depth
    # Sample well inside the gap (1 km west of split) — clear of bilinear bleed zone (~100 m)
    gap_x = MockBAGWithCoverageGap.LON_SPLIT - 0.009
    gap_val = float(elev.sel(x=gap_x, y=centre_y, method="nearest").item())
    assert gap_val == pytest.approx(MockBlueTopo.DEPTH, abs=0.5), (
        f"GAP cell (x={gap_x}) value {gap_val:.2f} m does not match BlueTopo "
        f"base ({MockBlueTopo.DEPTH} m). Coverage void was not filled."
    )

    # Sample well inside the BAG coverage (1 km east of split)
    survey_x = MockBAGWithCoverageGap.LON_SPLIT + 0.009
    survey_val = float(elev.sel(x=survey_x, y=centre_y, method="nearest").item())
    assert survey_val == pytest.approx(MockBAGWithCoverageGap.SURVEY_DEPTH, abs=0.5), (
        f"SURVEY cell (x={survey_x}) value {survey_val:.2f} m should be BAG "
        f"survey depth ({MockBAGWithCoverageGap.SURVEY_DEPTH} m)."
    )
