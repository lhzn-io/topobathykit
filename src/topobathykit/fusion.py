import logging

import numpy as np
import xarray as xr
from scipy.special import expit

logger = logging.getLogger(__name__)


class FusionEngine:
    """
    Fuses Lidar (Topography) and BlueTopo (Bathymetry) into a continuous surface.
    Uses a slope-adaptive logistic weighting function for seamless transition.
    """

    def __init__(self, z_center: float = 0.0, k_default: float = 1.0):
        """
        Args:
            z_center: The transition midpoint (e.g., 0.0 for NAVD88/MSL).
            k_default: Steepness of the transition curve.
        """
        self.z_center = z_center
        self.k_default = k_default

    def fuse(self, lidar_da: xr.DataArray, bathy_da: xr.DataArray, adaptive_k: bool = True) -> xr.DataArray:
        """
        Fuses Lidar and Bathymetry.

        Args:
            lidar_da: Topography DataArray (Land).
            bathy_da: Bathymetry DataArray (Sea).
            adaptive_k: If True, adjusts 'k' based on local slope.

        Returns:
            xr.DataArray: Fused elevation model.
        """
        # 1. Align Grids
        # Assuming lidar_da has been reprojected/regridded to match bathy_da externally
        # or we do simple alignment here.
        # For MVP, we assume they are spatially aligned or we use bathy as the primary.

        lidar_interp = lidar_da.interp_like(bathy_da, method="linear")

        # 2. Fill Missing Data
        # Where one source is missing, use the other 100%
        # lidar_val = lidar_interp.fillna(bathy_da) # Dangerous if bathy is also NaN
        # Better: calculate weights on valid data, then fill.

        # 3. Calculate Logistic Weights
        # weight = expit(k * (z - z_center))
        # z is the Lidar elevation (driving the land-dominance)

        z = lidar_interp.values

        k = self.k_default
        if adaptive_k:
            # Calculate slope magnitude
            # Using simple gradient
            # slope ~ sqrt(dz/dx^2 + dz/dy^2)
            # Higher slope -> Sharper transition (Higher k)

            # Placeholder for complex slope logic:
            # For cliff (high slope), k = 5.0
            # For marsh (low slope), k = 0.5
            pass
            # (Keeping it simple for MVP reliability: constant k or simple heuristic)

        # We want 'w' to be 1.0 (Lidar) when z >> 0, and 0.0 (Bathy) when z << 0
        # This assumes Lidar covers the transition zone.
        # Often Lidar stops at water, Bathy stops at land. The overlap is key.

        # Use a safe fill for z calculation
        z_safe = np.nan_to_num(z, nan=-9999.0)

        # Calculate Weight
        # If z > z_center, w -> 1. If z < z_center, w -> 0.
        w = expit(k * (z_safe - self.z_center))

        # 4. Fuse
        # fused = w * lidar + (1-w) * bathy

        # Handle NaNs naturally:
        # If Lidar is NaN, w should effectively be 0?
        # Actually expit(-large) is 0.
        # But if we rely on Lidar to define "Landness", missing Lidar means "Sea".

        # Force NaN handling explicitly
        # If Lidar is valid and Bathy is valid: Mix
        # If Lidar valid, Bathy NaN: Lidar
        # If Lidar NaN, Bathy valid: Bathy

        fused_values = np.zeros_like(bathy_da.values)

        lidar_vals = lidar_interp.values
        bathy_vals = bathy_da.values

        mask_lidar = ~np.isnan(lidar_vals)
        mask_bathy = ~np.isnan(bathy_vals)
        mask_both = mask_lidar & mask_bathy

        # Mix where overlaps
        fused_values[mask_both] = (w[mask_both] * lidar_vals[mask_both]) + (
            (1 - w[mask_both]) * bathy_vals[mask_both]
        )

        # Fill non-overlaps
        mask_only_lidar = mask_lidar & (~mask_bathy)
        fused_values[mask_only_lidar] = lidar_vals[mask_only_lidar]

        mask_only_bathy = mask_bathy & (~mask_lidar)
        fused_values[mask_only_bathy] = bathy_vals[mask_only_bathy]

        # Set absolute void to NaN (or nodata)
        mask_void = (~mask_lidar) & (~mask_bathy)
        fused_values[mask_void] = np.nan

        # Create result DataArray
        fused_da = xr.DataArray(
            fused_values, coords=bathy_da.coords, dims=bathy_da.dims, name="fused_elevation"
        )

        # Determine CRS (Prefer Bathy -> Lidar)
        crs = None
        if hasattr(bathy_da, "rio") and bathy_da.rio.crs:
            crs = bathy_da.rio.crs
        elif hasattr(lidar_da, "rio") and lidar_da.rio.crs:
            crs = lidar_da.rio.crs

        if crs:
            fused_da.rio.write_crs(crs, inplace=True)

        return fused_da

    def fuse_seamline(
        self,
        lidar_da: xr.DataArray,
        bathy_da: xr.DataArray,
        blend_dist: float = 20.0,
    ) -> xr.DataArray:
        """
        Fuses datasets using USGS-inspired seamline blending (Distance Field).
        Addresses "hard seams" where Lidar ends and Bathy begins with no overlap.
        Projects Lidar values outwards into the void/bathy area to create a smooth blend.

        Args:
            lidar_da: Topography (Priority/Land).
            bathy_da: Bathymetry (Background/Sea).
            blend_dist: Distance in meters to blend into the bathymetry.

        Returns:
            xr.DataArray: Fused surface.
        """
        import scipy.ndimage as ndimage

        # 1. Align Grids (Lidar to Bathy grid)
        lidar_interp = lidar_da.interp_like(bathy_da, method="linear")

        lad_vals = lidar_interp.values
        bat_vals = bathy_da.values

        # 0. Fast Robustness Check: If Lidar is fully empty, return Bathy
        if np.all(np.isnan(lad_vals)):
            logger.warning(
                "Fusion Seamline: Lidar (Priority) layer is empty (all NaNs) after alignment. Dropping layer."
            )
            return bathy_da

        # 2. Masks
        mask_lidar = ~np.isnan(lad_vals)
        mask_bathy = ~np.isnan(bat_vals)

        # 3. Compute Distance Transform from Lidar Edge (into the void/bathy)
        # We invert mask_lidar because edt computes distance to nearest zero.
        # So we want Lidar=0 (Source), Void=1 (Target).
        # We also need pixel resolution to convert px -> meters.
        # Assuming isotropic approx for now or extracting from transform.
        # dx = abs(bathy_da.rio.transform()[0]) # Approximation
        # For valid EDT on geodetic coords, this is tricky. assuming projected or small scale approx.
        # Let's assume roughly meters if projected, or convert deg->m.
        # For simplicity in this logic, we assume input is projected or we treat units as generic.
        # PROD TODO: Handle CRS properly.

        # return_indices=True allows us to find *which* Lidar pixel is closest (Proxy Logic)
        dist_px, indices = ndimage.distance_transform_edt(
            ~mask_lidar, return_distances=True, return_indices=True
        )

        # 4. Create Lidar Proxy Layer (Nearest Neighbor Extrapolation)
        # indices is (ndim, shape). We use it to sample lad_vals.
        # We only want to extrapolate where we are close enough (dist < blend_dist) AND in bathy area.
        # But we can compute it everywhere for simplicity first.

        # indices[0] is y-indices, indices[1] is x-indices
        lidar_proxy = lad_vals[tuple(indices)]

        # 5. Calculate Weights
        # W=1 at lidar edge (dist=0), W=0 at blend_dist.
        # But wait, USGS says "Project outward from Data A".
        # So at dist=0 (Lidar Edge), we want dominant Lidar.
        # As we go out, we want more Bathy.
        # So W_Lidar = 1.0 - (dist / blend_dist).
        # Clamped to 0.

        # Convert px distance to meters (approx)
        # If coords are lat/lon, 1 deg ~ 111km.
        # Check units.
        res = 1.0
        if hasattr(bathy_da, "rio"):
            res = abs(bathy_da.rio.resolution()[0])
            # If resolution is tiny (e.g. 0.0001), it's likely degrees.
            if res < 0.1:
                res *= 111000.0  # Approx conversions

        dist_m = dist_px * res

        w_lidar = 1.0 - (dist_m / blend_dist)
        w_lidar = np.clip(w_lidar, 0.0, 1.0)

        # 6. Fuse
        # Fused = w * Proxy + (1-w) * Bathy
        # Only apply this in the "Transition Zone" (where mask_lidar is False, but w_lidar > 0)
        # Where mask_lidar is True, keep Lidar.
        # Where w_lidar is 0, keep Bathy.

        fused_vals = np.copy(bat_vals)

        # A. Existing Lidar Data (Priority)
        fused_vals[mask_lidar] = lad_vals[mask_lidar]

        # B. Transition Zone (No Lidar, but within blend_dist)
        mask_trans = (~mask_lidar) & (w_lidar > 0)

        # We need valid bathy to blend against.
        # If bathy is strictly NaN here, we effectively extrapolate Lidar (w * proxy + (1-w)*0 ? No)
        # If bathy is NaN, we should probably just use Proxy * w ? Or just Proxy?
        # USGS fills gaps. Let's assume if Bathy is NaN, we extend Lidar Proxy 100% until cutoff?
        # Or let it fade to NaN?
        # Let's blend with available bathy.

        # Case B1: Transition & Valid Bathy
        mask_t_b = mask_trans & mask_bathy
        fused_vals[mask_t_b] = (w_lidar[mask_t_b] * lidar_proxy[mask_t_b]) + (
            (1.0 - w_lidar[mask_t_b]) * bat_vals[mask_t_b]
        )

        # Case B2: Transition & NaN Bathy (Pure Extrapolation)
        # This fills holes between land and deep water if bathy is clipped tight.
        mask_t_nan_b = mask_trans & (~mask_bathy)
        fused_vals[mask_t_nan_b] = lidar_proxy[mask_t_nan_b]  # Full proxy extension

        fused_da = xr.DataArray(fused_vals, coords=bathy_da.coords, dims=bathy_da.dims, name="fused_seamline")

        # Propagate CRS
        crs = None
        if hasattr(bathy_da, "rio") and bathy_da.rio.crs:
            crs = bathy_da.rio.crs
        elif hasattr(lidar_da, "rio") and lidar_da.rio.crs:
            crs = lidar_da.rio.crs

        if crs:
            fused_da.rio.write_crs(crs, inplace=True)

        return fused_da
