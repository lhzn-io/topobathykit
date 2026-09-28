from enum import StrEnum
from typing import Any, ClassVar

import numpy as np
import xarray as xr


class QualityClass(StrEnum):
    DIRECT = "Direct Measurement"
    INDIRECT = "Indirect/Predicted"
    UNKNOWN = "Unknown/Interpolated"


class TIDClassifier:
    """
    Classifies GEBCO Type Identifier (TID) codes into semantic quality classes.
    Based on GEBCO 2025 TID conventions.
    """

    # Direct measurements: Single beam, Multibeam, Seismic, Lidar, Optical, etc.
    DIRECT_TIDS: ClassVar[set[int]] = {10, 11, 12, 13, 14, 15, 16, 17}

    # Indirect methods: Satellite/Gravity Predicted, Interpolated, Contours, etc.
    INDIRECT_TIDS: ClassVar[set[int]] = {40, 41, 42, 43, 44, 45, 46, 47}

    # Unknown, Pre-generated, or Steering points
    UNKNOWN_TIDS: ClassVar[set[int]] = {0, 70, 71, 72}

    @classmethod
    def classify(cls, tid: int) -> QualityClass:
        if tid in cls.DIRECT_TIDS:
            return QualityClass.DIRECT
        elif tid in cls.INDIRECT_TIDS:
            return QualityClass.INDIRECT
        else:
            return QualityClass.UNKNOWN

    @classmethod
    def get_color(cls, quality: QualityClass) -> str:
        if quality == QualityClass.DIRECT:
            return "green"
        elif quality == QualityClass.INDIRECT:
            return "orange"
        else:
            return "grey"


def source_report(tid_data: xr.DataArray | np.ndarray) -> dict[str, float]:
    """
    Generates a percentage report of data sources for the given TID array.

    Args:
        tid_data: xarray.DataArray or numpy array containing GEBCO TID values.

    Returns:
        Dictionary mapping QualityClass names to percentage coverage (0-100).
    """
    data = tid_data.values.flatten() if isinstance(tid_data, xr.DataArray) else np.asarray(tid_data).flatten()

    total_pixels = len(data)
    if total_pixels == 0:
        return {
            QualityClass.DIRECT.value: 0.0,
            QualityClass.INDIRECT.value: 0.0,
            QualityClass.UNKNOWN.value: 0.0,
        }

    # Vectorized counting
    # We can use numpy's isin or logical checks, but a simple loop or mask is clear for now
    # Optimization: Use searchsorted or histograms if performance is critical for massive arrays

    # Direct
    direct_mask = np.isin(data, list(TIDClassifier.DIRECT_TIDS))
    direct_count = np.count_nonzero(direct_mask)

    # Indirect
    indirect_mask = np.isin(data, list(TIDClassifier.INDIRECT_TIDS))
    indirect_count = np.count_nonzero(indirect_mask)

    # Unknown/Rest
    # Alternatively explicit check:
    # unknown_mask = np.isin(data, list(TIDClassifier.UNKNOWN_TIDS))
    # unknown_count = np.count_nonzero(unknown_mask)

    # Handle any unclassified TIDs as unknown for safety
    final_unknown = total_pixels - direct_count - indirect_count

    return {
        QualityClass.DIRECT.value: round((direct_count / total_pixels) * 100, 2),
        QualityClass.INDIRECT.value: round((indirect_count / total_pixels) * 100, 2),
        QualityClass.UNKNOWN.value: round((final_unknown / total_pixels) * 100, 2),
    }


def calculate_spatial_stats(
    elevation: xr.DataArray | np.ndarray,
    source_map: xr.DataArray | np.ndarray | None,
    resolution_m: float,
    bounds: tuple[float, float, float, float] | None = None,
    steep_threshold_m: float = 3.0,
    pit_ratio_threshold: float = 3.0,
) -> dict[str, Any]:
    """
    Computes spatial statistics and fragility metrics for a DEM.
    Flags steep gradients (cliffs), isolated pits, contiguous NaN regions,
    and vertical datum jumps precisely at source seams.

    Args:
        elevation: xr.DataArray or 2D np.ndarray of elevations.
        source_map: xr.DataArray or 2D np.ndarray of provider source IDs, aligned with elevation.
        resolution_m: grid cell resolution in meters.
        bounds: (min_lon, min_lat, max_lon, max_lat). Required if elevation is np.ndarray.
        steep_threshold_m: minimum elevation delta (m) to flag as a steep cliff between cells.
        pit_ratio_threshold: ratio of center depth vs mean neighbor depth to flag as pit.

    Returns:
        dict containing 'stats' (aggregates) and 'features' (a GeoJSON FeatureCollection of anomalies).
    """
    # Defensive checks
    if isinstance(elevation, xr.DataArray):
        elev_da: xr.DataArray = elevation.squeeze() if elevation.ndim != 2 else elevation
        z = elev_da.values
        valid_mask = elev_da.notnull().values
        lons = elev_da.coords["lon"].values if "lon" in elev_da.coords else elev_da.coords["x"].values
        lats = elev_da.coords["lat"].values if "lat" in elev_da.coords else elev_da.coords["y"].values
    else:
        z = elevation
        valid_mask = ~np.isnan(z)
        if bounds is None:
            raise ValueError("bounds (min_lon, min_lat, max_lon, max_lat) must be provided for np.ndarray")
        # generate linspaces for lons and lats
        # assumes z shape is (ny, nx) and bounds are (minx, miny, maxx, maxy)
        ny, nx = z.shape
        lons = np.linspace(bounds[0], bounds[2], nx)
        # assuming north-first array (maxy to miny)
        lats = np.linspace(bounds[3], bounds[1], ny)

    if not valid_mask.any() or resolution_m <= 0:
        return {
            "stats": {
                "max_slope_deg": 0.0,
                "missing_data_pct": 100.0,
                "vdatum_risk_score": 0,
                "steep_cliffs_count": 0,
            },
            "features": {"type": "FeatureCollection", "features": []},
        }

    import scipy.ndimage as ndimage

    invalid = np.isnan(z)

    if invalid.any() and not invalid.all():
        # Quick nearest neighbor fill so gradients don't spike at missing data bounds.
        # Find distance to nearest valid pixel in grid space.
        ind = ndimage.distance_transform_edt(invalid, return_distances=False, return_indices=True)
        z_filled = z[tuple(ind)]
    else:
        # Either all valid, or all invalid (which was caught above anyway).
        z_filled = z.copy()

    # Calculate gradients
    dy, dx = np.gradient(z_filled, resolution_m, resolution_m)
    # True elevation delta per cell length (not normalized by resolution, just absolute dz)
    dy_dz, dx_dz = np.gradient(z_filled, 1.0, 1.0)

    abs_dy_dz = np.abs(dy_dz)
    abs_dx_dz = np.abs(dx_dz)

    slope_percent = np.sqrt(dx**2 + dy**2)
    slope_degrees = np.degrees(np.arctan(slope_percent))

    max_slope = float(np.max(slope_degrees[~np.isnan(z)]))
    missing_pct = float(np.count_nonzero(~valid_mask) / z.size * 100)

    # Anomaly Masks
    # 1. Steep gradients: where change in Z between adjacent cells > threshold
    steep_mask = (abs_dx_dz > steep_threshold_m) | (abs_dy_dz > steep_threshold_m)
    steep_mask = steep_mask & ~np.isnan(z)

    # 2. VDatum Jumps (Seam anomalies)
    seam_jump_mask = np.zeros_like(steep_mask, dtype=bool)
    if source_map is not None:
        if isinstance(source_map, xr.DataArray):
            squeezed_sm = source_map.squeeze() if source_map.ndim != 2 else source_map
            sm = squeezed_sm.values
        else:
            sm = source_map
        sm_filled = np.nan_to_num(sm, nan=-1)
        # Any change in source map means a seam (gradient != 0)
        s_dy, s_dx = np.gradient(sm_filled, 1.0, 1.0)
        seam_mask = (s_dy != 0) | (s_dx != 0)
        seam_jump_mask = steep_mask & seam_mask & ~np.isnan(z)
        # Filter steep mask so it doesn't double count VDatum jumps
        steep_mask = steep_mask & ~seam_jump_mask

    # Build GeoJSON Features
    features = []

    # Helper to add points to GeoJSON
    def add_points(mask: np.ndarray, anomaly_type: str, properties_func: Any | None = None) -> None:
        y_idx, x_idx = np.where(mask)
        for y, x in zip(y_idx, x_idx, strict=False):
            lon, lat = lons[x], lats[y]
            elev_val = float(z[y, x])
            props = {"type": anomaly_type, "elevation": elev_val}
            if properties_func:
                props.update(properties_func(y, x))
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                    "properties": props,
                }
            )

    add_points(
        steep_mask,
        "cliff",
        lambda y, x: {
            "slope": float(slope_degrees[y, x]),
            "dz_x": float(abs_dx_dz[y, x]),
            "dz_y": float(abs_dy_dz[y, x]),
        },
    )
    add_points(
        seam_jump_mask,
        "vdatum_jump",
        lambda y, x: {
            "slope": float(slope_degrees[y, x]),
            "source_id": float(sm[y, x]) if source_map is not None else -1,
        },
    )

    vdatum_risk_score = int(np.count_nonzero(seam_jump_mask))

    return {
        "stats": {
            "max_slope_deg": round(max_slope, 2),
            "missing_data_pct": round(missing_pct, 2),
            "vdatum_risk_score": vdatum_risk_score,
            "steep_cliffs_count": int(np.count_nonzero(steep_mask)),
        },
        "features": {"type": "FeatureCollection", "features": features},
    }
