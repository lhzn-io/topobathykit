.. _cache_viewer:

Cache Inventory Viewer & Manager
================================

topobathykit includes a web-based **Cache Inventory Viewer** for inspecting and managing the multi-tiered caching system. This tool provides visibility into storage usage, cache age, and allows safe maintenance operations.

Access
------

By default, the viewer is accessible at:

.. code-block:: text

   http://localhost:9595/viewer/cache-viewer.html

(Adjust host and port if running on a remote server or custom configuration).

Features
--------

- **Visual Dashboard**: View total cache size and breakdown by tier.
- **Tier Inspection**: Drill down into specific cache tiers to see file counts and size.
- **Project Provenance**: Track which source files (BAGs, COGs) correspond to cached tiles.
- **Safe Purging**: Selectively clear specific cache tiers to free up disk space or force data regeneration.

Cache Tiers
-----------

topobathykit organizes data into 5 tiers of increasing persistence and reconstruction cost:

+------+-------------------------+-----------------------------------------------+------------------------+---------------------+
| Tier | Name                    | Description                                   | Path                   | Rebuild Cost        |
+======+=========================+===============================================+========================+=====================+
| **1**| Output Tiles & Metadata | Final rendered products (PNGs, NPYs)          | ``tiles/``             | Seconds (re-render) |
+------+-------------------------+-----------------------------------------------+------------------------+---------------------+
| **2**| Fused Zarr              | Multi-provider composites (Intermediate)      | ``fused_zarr/``        | Minutes (re-fuse)   |
+------+-------------------------+-----------------------------------------------+------------------------+---------------------+
| **3**| Provider Zarr           | Optimized single-source caches (Rasterized)   | ``{provider}/zarr/``   | Minutes-Hours       |
+------+-------------------------+-----------------------------------------------+------------------------+---------------------+
| **4**| Discovery Metadata      | JSON/GeoJSON indexes of remote datasets       | ``*_cache.json``       | Seconds-Minutes     |
+------+-------------------------+-----------------------------------------------+------------------------+---------------------+
| **5**| Raw Source Files        | Original downloaded assets (BAG, LAZ, TIF)    | ``downloads/``         | Hours (Download)    |
+------+-------------------------+-----------------------------------------------+------------------------+---------------------+

.. warning::
   **Tier 5 (Raw Source Files)** is the most expensive to replace. Purging this tier requires re-downloading gigabytes of data from NOAA/USGS. Only purge if you suspect corrupted downloads.

Operations
----------

The viewer supports the following actions:

* **Refresh**: Updates the inventory statistics from disk.
* **Purge**: Opens a modal to select tiers for deletion.
    * **Dry Run**: Simulates the deletion to show what would be removed.
    * **Confirm**: Requires typing "purge" to prevent accidental data loss.

Common Workflows
~~~~~~~~~~~~~~~~

**Scenario: Policy Change**
If you update a fusion policy (e.g., changing blending logic), you typically only need to purge **Tier 2 (Fused Zarr)** and **Tier 1 (Output Tiles)**. The underlying provider data (Tier 3-5) can remain valid.

**Scenario: Bad Source Data**
If a source file (e.g., a specific BAG) is corrupted or updated by the provider:
1. Identify the file in the viewer.
2. Purge **Tier 3 (Provider Zarr)** for that provider to force re-reading the source.
3. If the source file itself is bad, purge **Tier 5** to re-download.
