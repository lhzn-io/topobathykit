Runtime Architecture
====================

The TopoBathySim Runtime is a stateless engine that executes Fusion Policies.

Core Concept
------------

Unlike traditional pipelines that hardcode logic (e.g., "Always put Lidar over GEBCO"), the Runtime is agnostic. It simply executes the list of steps defined in the Policy YAML.

.. mermaid::

    graph TD
        subgraph Policy Loading
            Policy[Policy YAML] -->|Load & Validate| Schema[Pydantic Models]
            Schema --> Runtime[Runtime Engine]
        end

        subgraph Runtime Execution
            Runtime -->|1. Initialize| Canvas[Canvas: Elevation + Provenance]

            Runtime -->|2. Loop Steps| Registry[Provider Registry]
            Registry -->|Get Provider| Match

            Runtime -->|3. Fetch & Align| Match[Fetch Layer (Lazy/Cached) + Reproject]

            Match -->|4. Check Rules| Rules{Transition Rules?}

            Rules -- Yes --> TransOp[Apply Specific Operator (e.g. Feather)]
            Rules -- No --> DefOp[Apply Default Operator (e.g. Overwrite)]

            TransOp --> Blend[Blend into Canvas]
            DefOp --> Blend

            Blend -->|Update| Provenance[Update Source Mask]
        end

        subgraph Output
            Provenance -->|Finalize| Dataset[Xarray Dataset]
            Dataset -->|Save| Zarr[Fused Zarr Cache]
            Dataset -->|Write sidecars| Sidecars["_meta.json + _src.npz"]
        end

Components
----------

Runtime Engine (``topobathysim.runtime``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The entry point ``run(policy_path, bbox)`` orchestrates the entire process:

1.  **Canvas Initialization**: Creates a blank ``xarray.Dataset`` covering the requested BBox in the Policy's CRS.
2.  **Step Execution**: Iterates through policy steps.
3.  **Data Fetching**: Calls ``provider.fetch_layer()``.
4.  **Alignment**: Reprojects the fetched layer to match the Canvas pixel grid (using ``rio.reproject_match``).
5.  **Composition**: Applies the blening operator (Overwrite/Feather) to merge the aligned layer into the canvas.

Providers (``topobathysim.providers``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Providers are standardized adapters that fetch data from remote sources.

- **Lazy Loading**: Providers like ``gebco_2025`` only open network connections when a requested tile is not in the local Zarr cache.
- **Smart Caching**: Data is cached locally as Zarr (for dense grids) or COGs/LAZ (for source files).
- **Aliases**: Providers are registered with short names (e.g., ``usgs_3dep``, ``ncei_bag``) used in the Policy YAML.
- **No-Data Signalling**: When a provider has no coverage for a requested cell it raises
  ``ProviderNoDataError`` (a ``LookupError`` subclass).  This is a normal operating
  condition — the runtime skips the provider silently at ``DEBUG`` log level.  Only
  genuine unexpected failures are logged at ``ERROR`` with a traceback.
- **Fetch-Error Signalling**: When a provider has coverage but fails to deliver it
  (network timeout, VDatum outage, ``MemoryError``), it raises ``ProviderFetchError``.
  When it returns data but dropped some tiles or projects to errors, it sets an
  integer ``attrs["fetch_errors"]`` on the returned Dataset.  The runtime records a
  per-step outcome (``ok``, ``nodata``, ``partial`` or ``error``, with the pixel count
  each step contributed) in the cell attribute ``step_outcomes_json``.  ``hydrate``
  does not cache a cell with any ``partial`` or ``error`` step, or whose NaN fraction
  exceeds ``TOPOBATHY_MAX_CELL_NAN_FRACTION`` (default 0.01); such cells count as
  ``failed`` and are retried on the next run.  ``run`` (the ``/fuse`` and tile path)
  applies only the step check: it still returns an incomplete cell but does not
  cache it.

Provenance System
-----------------

Every tile carries a full record of which source dataset contributed each pixel.

**Provider → Dataset**

Each ``fetch_layer()`` call returns an ``xr.Dataset`` containing two arrays:

- ``elevation`` — float32 depth/height values
- ``source_id`` — uint32 pixel map where each value identifies the contributing
  survey or sub-tile (provider-assigned IDs for BAG/BlueTopo; hash-derived IDs
  for other providers)

The dataset's ``attrs["provenance_dict"]`` maps each ``source_id`` integer to
a ``{"name": "...", "provider": "..."}`` dict.

**Runtime → Fused Dataset**

As the runtime processes policy steps, it accumulates ``provenance_dict`` entries
from all providers into a single ``cell_provenance_dict``.  The final fused
``xr.Dataset`` carries the merged provenance across all cells.

**Service → Tile Sidecars**

When a rendered tile is written to the tile cache, two sidecar files are written
alongside it:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - File
     - Contents
   * - ``{y}_{hash}_meta.json``
     - ``provenance_dict`` filtered to source IDs present in this specific tile.
       Read by ``GET /tiles/{z}/{x}/{y}/metadata``.
   * - ``{y}_{hash}_src.npz``
     - Compressed uint32 ``source_elevation`` array (512 × 512, padding stripped).
       Read by ``GET /tiles/{z}/{x}/{y}/pixel`` for per-pixel source lookups.

**Viewer**

The Leaflet popup fetches both ``/metadata`` and ``/pixel`` for each clicked
location, displaying the pixel-level contributing survey (name + provider color
swatch) and a deduplicated, priority-sorted list of all surveys present in the
tile.

Caching Strategy
----------------

1.  **Source Cache**: Original files (TIFF, BAG, LAZ) downloaded from agencies (download providers only; streaming providers write directly to zarr).
2.  **Provider Zarr Cache**: Intermediate rasterised chunks per provider, stored for fast repeated access.
3.  **Fused Zarr Cache**: Multi-provider composite grids keyed by ``{policy, cell_bbox, resolution, crs}`` hash.
4.  **Tile Cache**: Final XYZ PNG/NPY/NPZ tiles at ``~/.cache/topobathysim/tiles/``, with ``_meta.json`` and ``_src.npz`` sidecars.

See :doc:`cli_tools` for cache tier management.
