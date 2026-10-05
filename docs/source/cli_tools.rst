CLI Tools
=========

topobathykit includes a unified command-line utility — ``cache_manager`` — for
inspecting, verifying integrity, and purging cached data.  A separate ``run_server``
script starts the tile server.

Understanding the Cache Hierarchy
----------------------------------

topobathykit maintains five cache tiers, ordered from the safest to the most
expensive to rebuild:

.. list-table:: Cache Tier Inventory
   :widths: 5 25 40 30
   :header-rows: 1

   * - #
     - Tier
     - Paths (relative to ``~/.cache/topobathykit/``)
     - Rebuild cost
   * - 1
     - Output tiles & metadata
     - ``tiles/visual/``, ``tiles/data/``, ``tiles/raw/`` (PNGs, NPY/NPZ, ``_meta.json``, ``_src.npz``)
     - Seconds — re-rendered on next tile request
   * - 2
     - Fused zarr
     - ``fused_zarr/`` (multi-provider composite grids)
     - Minutes — re-fused from provider zarr
   * - 3
     - Provider zarr
     - ``{provider}/zarr/*.zarr`` (rasterised per-provider grids)
     - Minutes–hours — re-rasterised from raw source files
   * - 4
     - Discovery caches
     - 5 JSON/GeoJSON metadata files (see below)
     - Seconds–minutes — re-queried from disk or network
   * - 5
     - Raw source files ⚠️
     - ``*.bag``, ``*.tiff``, ``*.tif``, ``*.laz`` in provider directories
     - Hours — full re-download

**Tier 4 metadata & discovery caches** (lightweight indices, fast to rebuild):

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - File (relative to ``~/.cache/topobathykit/``)
     - Stores
   * - ``ncei_bag/discovery_cache.json``
     - bbox → BAG download-URL list (NCEI ArcGIS API)
   * - ``metadata/ncei_bag_redirects.json``
     - NCEI landing-page URL → ``.bag`` file URL
   * - ``noaa_bluetopo/tile_url_cache.json``
     - tile_id → HTTPS asset URL (S3 glob lookup)
   * - ``noaa_bluetopo/BlueTopo_Tile_Scheme.gpkg``
     - tile footprint spatial index (downloaded from NOAA S3)
   * - ``noaa_bluetopo/sidecars/``
     - per-tile RAT ``.aux.xml`` files mapping pixel values to survey IDs;
       ``.failed`` sentinels mark tiles whose RAT_Link is stale (404)
   * - ``noaa_topobathy/*.zip``
     - per-project tile-index shapefiles (one per lidar project)
   * - ``metadata/noaa_coastal_lidar.json``
     - project ID → folder name (S3 index HTML)
   * - ``metadata/noaa_project_extents.geojson``
     - project spatial bbox index (GeoJSON)
   * - ``vdatum.sqlite``
     - Local SQLite WAL cache containing NOAA VDatum API vertical offsets

Discovery caches survive server restarts so tile generation is fully offline
once a region is hydrated.  Invalidate them when you expect newly published
survey data to be available.

.. note::

   ``noaa_bluetopo``, ``noaa_topobathy``, and ``usgs_3dep`` are *streaming
   providers* — they never download raw source files.  They read Cloud
   Optimized GeoTIFFs directly from S3 and write straight to provider zarr.
   Only ``ncei_bag``, ``usgs_lidar``, and ``ncei_cudem`` populate tier 5.


Cache Manager
-------------

Use ``cache_manager`` for all cache operations: status, integrity checks, and
selective or bulk purging.

Interactive Mode (default)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Run without arguments to enter the interactive menu:

.. code-block:: bash

   python -m topobathykit.scripts.cache_manager

The menu displays a live status table and loops after each operation::

   topobathykit Cache Manager
   ==========================
     Cache root : ~/.cache/topobathykit
     Total size : 4.2 GB

     #   Tier                            Size    Items
     ────────────────────────────────────────────────────────
   ✓  [1] Output tiles & metadata          23.4 MB      847
   ✓  [2] Fused zarr                        0.2 MB        3
   ✓  [3] Provider zarr                     3.8 GB       12
   ✓  [4] Discovery caches                 12.0 KB        5
   ✓  [5] Raw source files ⚠               0.4 GB       38

     ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
     [1–5] purge tier   [c] check integrity   [i] tier info   [q] quit

Available interactive commands:

- ``1``–``5`` — purge that tier (prompts for confirmation; tier 5 requires typing ``yes``)
- ``c`` — run a read-only integrity audit across all providers and discovery caches
- ``i`` — print detailed per-tier descriptions with when/why to purge
- ``q`` — quit

Non-Interactive Flags
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # 1. Integrity audit (read-only)
   python -m topobathykit.scripts.cache_manager --check

   # 2. Integrity check + delete corrupt files and stale locks
   python -m topobathykit.scripts.cache_manager --clean

   # 3. Purge a specific tier
   python -m topobathykit.scripts.cache_manager --purge 1

   # 4. Purge multiple tiers
   python -m topobathykit.scripts.cache_manager --purge 1 2 4

   # 5. Purge all tiers (prompts for tier 5 confirmation unless --yes)
   python -m topobathykit.scripts.cache_manager --purge all

   # 6. Dry run — preview what would be deleted without deleting
   python -m topobathykit.scripts.cache_manager --dry-run --purge all

   # 7. Skip confirmation prompts (use with care for tier 5)
   python -m topobathykit.scripts.cache_manager --yes --purge 1 2

Arguments
^^^^^^^^^

- ``--check`` — read-only integrity scan (corrupt files, stale locks, malformed JSON)
- ``--clean`` — same as ``--check`` but also deletes the corrupt/stale items
- ``--purge TIER [TIER ...]`` — tier number(s) 1–5, or ``all``
- ``--dry-run`` — show what would be deleted without deleting
- ``--yes`` — skip confirmation prompts
- ``--lock-timeout N`` — seconds before a ``.lock`` file is considered stale (default: 3600)


Common Scenarios
^^^^^^^^^^^^^^^^

**Re-render tiles after a style change**

.. code-block:: bash

   python -m topobathykit.scripts.cache_manager --purge 1 --yes

**Force re-fusion after updating the policy YAML**

.. code-block:: bash

   # Purge fused zarr so next tile request re-runs the fusion pipeline
   python -m topobathykit.scripts.cache_manager --purge 2 --yes

**NCEI has published new multibeam surveys for your area**

.. code-block:: bash

   python -m topobathykit.scripts.cache_manager --purge 4 --yes

**NOAA has updated BlueTopo tiles (filenames changed with a new date stamp)**

.. code-block:: bash

   # Stale tile_url_cache entries will 404; purge tier 4 to force re-discovery
   python -m topobathykit.scripts.cache_manager --purge 4 --yes

**BlueTopo sidecar RAT_Links returning 404 (stale tile scheme)**

When NOAA republishes a tile with a new date stamp, the ``RAT_Link`` URLs
baked into the cached ``BlueTopo_Tile_Scheme.gpkg`` go stale.  The provider
writes a ``.failed`` sentinel so downloads are not retried on every request,
but the underlying surveys can no longer be resolved to human-readable names
until the scheme is refreshed:

.. code-block:: bash

   # Purge tier 4 — clears the GPKG, sidecars, and .failed sentinels together
   python -m topobathykit.scripts.cache_manager --purge 4 --yes

**A new NOAA topobathy lidar project was added to the archive**

.. code-block:: bash

   python -m topobathykit.scripts.cache_manager --purge 4 --yes

**Start completely fresh (keep raw source files)**

.. code-block:: bash

   # Clears tiers 1–4; raw files are retained so re-rendering is fast
   python -m topobathykit.scripts.cache_manager --purge 1 2 3 4 --yes

**Check for corrupt or stale files after a crash**

.. code-block:: bash

   python -m topobathykit.scripts.cache_manager --check

   # If issues are found, run with --clean to delete them
   python -m topobathykit.scripts.cache_manager --clean

.. note::

   Purging a tier does **not** purge lower-numbered tiers.  For example, purging
   tier 4 (metadata & discovery caches) does not delete provider zarr or raw source files.
   If the underlying provider data is still cached, discovery caches will be
   re-populated from that local data on the next tile request — no network calls
   are needed for already-cached survey regions.


Server Management
-----------------

To run the tile server for local development or production usage:

.. code-block:: bash

   # Start the server on port 9595 with 8 worker processes
   micromamba run -n topobathykit python service/run_server.py --host 0.0.0.0 --port 9595 --workers 8

Arguments:
^^^^^^^^^^

- ``--host``: Bind address (default: 127.0.0.1)
- ``--port``: Port to listen on (default: 9595)
- ``--workers``: Number of worker processes (default: 4)
- ``--debug``: Enable debug logging (default: 0)
