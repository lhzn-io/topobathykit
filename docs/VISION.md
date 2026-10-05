# Vision

topobathykit is a policy-driven **topobathymetric fusion runtime**: a Python library (and optional service) for turning heterogeneous elevation sources into **seamless, simulation- and analysis-ready grids** with **per-pixel provenance**.

## Mission

Make it straightforward to build *trustworthy digital terrain* anywhere on Earth by combining global basemaps with local high-resolution surveys—while keeping assumptions explicit and outputs reproducible.

## What topobathykit is

topobathykit focuses on the layer between *raw geospatial infrastructure* and *downstream modeling/simulation*.

Concretely, it provides:

- **Providers** that fetch and normalize data layers (global grids, regional compilations, local surveys/lidar), with caching for offline replay.
- **Policies** (YAML) that encode domain expertise as deterministic fusion strategies **per variable** (elevation now; time-varying covariates next).
- **Operators** for transition zones (e.g., seamline feathering), so outputs can be continuous without hiding where each dataset came from.
- **Provenance-first outputs** (`xarray.Dataset`) that include fused values and source masks for debugging, QA, and downstream trust.
- An optional **microservice** that serves the same fused products over HTTP for real-time clients (Unity/Unreal/Omniverse, robotics stacks, web viewers).

## What topobathykit is not

topobathykit is intentionally not:

- A hosted dataset, or an “authoritative” replacement for hydrographic offices.
- A single “one true fusion algorithm.” Fusion strategies are **use-case dependent** and belong in policies.
- A purely visualization-oriented tool. PNG/hillshade and viewers are **debugging tools**; data products are the core output.
- A guarantee of global vertical datum harmonization. Datum assumptions and transforms are surfaced explicitly in metadata/provenance, and regional transformations are pluggable.

## Where we fit in the open topo/bathy ecosystem

topobathykit complements existing tools and workflows:

- **CUDEM** and similar pipelines are excellent for systematic DEM production and gridding workflows.
- **GDAL / PDAL / xarray / rioxarray** provide foundational geospatial I/O and processing primitives.
- **bmi-topography** provides standardized access patterns to certain DEM products.

topobathykit’s niche is the **runtime + policy layer**:

- codifying *“what should win where”* (trusted overwrite vs transition blending),
- making results reproducible and inspectable (provenance masks),
- and delivering outputs in forms that simulation and modeling stacks can consume quickly (arrays, tiles, offline caches).

## Principles

1. **Overwrite in trusted zones; blend in transition zones**
   Survey “truth” should win where present; seams should be softened where coverage edges meet.

2. **Provenance is first-class**
   Every fused product should carry a source mask and enough metadata to explain how it was produced.

3. **Global-first, regional-extensible**
   A global basemap policy should work anywhere. Regional expertise should be added via provider plugins and policy presets.

4. **Deterministic policies**
   Policies are versionable artifacts (YAML) that can be reviewed, shared, and cited. Natural-language interfaces may exist later, but must compile to deterministic policies.

5. **Data-first outputs**
   The canonical output is an `xarray.Dataset`. Visualizations exist to validate the data and debug fusion logic.

## How to contribute (most valuable paths)

- Add a **provider** for a dataset relevant to your region (EU/Asia-Pacific/etc.).
- Add a **policy preset** that encodes best practices and local domain knowledge.
- Add **operators** or tests that improve seam handling, performance, or QA.
- Help expand the platform toward **time-varying covariates** (Sentinel-1/2/3 and beyond), with consistent caching and provenance.

See `CONTRIBUTING.md` and the `policies/` gallery for entry points.
