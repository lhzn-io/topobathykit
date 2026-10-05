Data Sources & Acknowledgements
=============================

**topobathykit** relies on a federation of open-access geospatial datasets. We gratefully acknowledge the following agencies and specific surveys used for development, validation, and simulation.

The datasets are listed below in inverse order of priority (from global context to high-resolution survey truth).

Global Coverage (Tier 4)
------------------------

General Bathymetric Chart of the Oceans (GEBCO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*   **GEBCO_2025 Grid**

    *   **Description**: A continuous terrain model for ocean and land, providing global context at 15 arc-second intervals.
    *   **Citation**: GEBCO Compilation Group (2025) GEBCO 2025 Grid (doi:10.5285/1c44ce9e-0a53-431f-9780-3843464522c5).
    *   **Access**: `GEBCO Grid Download <https://www.gebco.net/data_and_products/gridded_bathymetry_data/>`_

Continental / Land (Tier 3.5)
-----------------------------

U.S. Geological Survey (USGS) - 3DEP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*   **3D Elevation Program (3DEP)**

    *   **Description**: The source for high-quality topographic data of the conterminous United States, Hawaii, and U.S. territories.
    *   **Citation**: U.S. Geological Survey, 2017, 3D Elevation Program (3DEP): U.S. Geological Survey Fact Sheet 2017–3081.
    *   **Access**: `3DEP Homepage <https://www.usgs.gov/core-science-systems/ngp/3dep>`_

Regional Bathymetry (Tier 3)
----------------------------

NOAA NCEI - CUDEM
~~~~~~~~~~~~~~~~~

*   **Continuously Updated Digital Elevation Model (CUDEM)**

    *   **Description**: Ninth Arc-Second Resolution Bathymetric-Topographic Tiles. Used as a critical gap-filler for coastal inundation zones.
    *   **Citation**: Cooperative Institute for Research in Environmental Sciences (CIRES) at the University of Colorado, Boulder. 2014. Continuously Updated Digital Elevation Model (CUDEM) - Ninth Arc-Second Resolution Bathymetric-Topographic Tiles. NOAA National Centers for Environmental Information.
    *   **Access**: `Dataset Homepage <https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ngdc.mgg.dem:999919/html>`_ | `Bulk Access <https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/NCEI_ninth_Topobathy_2014_8483/index.html>`_

Regional High-Resolution (Tier 2)
---------------------------------

NOAA Office of Coast Survey - BlueTopo™
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*   **BlueTopo™**

    *   **Description**: A compilation of the best available public bathymetric data of U.S. waters, merged into a simplified product.
    *   **Citation**: NOAA Office of Coast Survey. BlueTopo™.
    *   **Access**: `BlueTopo Portal <https://nauticalcharts.noaa.gov/data/bluetopo.html>`_

Local High-Resolution Lidar (Tier 1)
------------------------------------

NOAA National Geodetic Survey (NGS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*   **Topobathymetric Lidar**

    *   **Description**: High-resolution nearshore bathymetry captured via green-wavelength Lidar.
    *   **Citation**: National Geodetic Survey, 2026: 2023 NOAA NGS Lidar DEM. NOAA National Centers for Environmental Information.
    *   **Example (Long Island Sound)**: `Methodology Report (PDF) <https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/laz/geoid18/10273/supplemental/NOAA_NY_Long_Island_Topobathymetric_Lidar_Imagery_Report_Final.pdf>`_
    *   **Access**: `NOAA Digital Coast <https://coast.noaa.gov/digitalcoast/>`_

    **Coverage Maximization & Survey Selection Strategy**

    To manage the hundreds of overlapping proprietary surveys available via NOAA's Digital Coast, topobathykit employs a dynamic **Spatial Index** to select the optimal data source for any given area of interest.

    When a simulation region intersects multiple available surveys, the system prioritizes them using the following **Fusion Hierarchy**:

    1.  **Vertical Datum Quality**: Surveys referenced to **NAVD88 (Geoid18)** are strictly preferred over legacy NAVD88 or Ellipsoidal references, ensuring consistent vertical integration with water levels.
    2.  **Recency**: Among surveys with equal datum quality, the most recently completed survey (based on ``end_date``) is selected.
    3.  **Spatial Overlap**: The system identifies all surveys intersecting the requested bounding box. Currently, the single "best" survey is selected to provide a self-consistent foundation, rather than blending multiple surveys which may have slight vertical offsets or temporal discrepancies.

    Within a selected survey, individual data tiles are fused using a **Mosaic** operation, where valid data overwrites NoData/Void areas.

Survey "Truth" Data (Tier 0)
----------------------------

NOAA NCEI - Bathymetric Attributed Grid (BAG)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*   **Bathymetric Attributed Grids (BAG)**

    *   **Description**: The definitive format for multi-beam sonar surveys. When found, these overwrite all other sources.
    *   **Citation**: NOAA National Centers for Environmental Information. "Bathymetric Attributed Grid (BAG) File Format."
    *   **Access**: `NCEI Bathymetry Data Viewer <https://www.ncei.noaa.gov/maps/bathymetry/>`_
    *   **Programmatic Access**: `NOAA GeoPlatform Hydrographic Surveys <https://services2.arcgis.com/C8EMgrsFcRFL6LrL/arcgis/rest/services/NOS_Hydro_Surveys/FeatureServer>`_

Fusion Strategy: Redundancy & Spatial Overwrite
-----------------------------------------------

A frequent scenario in topobathykit is the overlap between **BlueTopo** (Tier 2, Regional) and **NCEI BAG** (Tier 0, High-Res Truth). BlueTopo is distributed as large geographic mosaic tiles (e.g., ``BH5245GP``) which contain compiled data from dozens of underlying surveys. Many of the newer sub-surveys inside these mosaics are also natively available via the NCEI BAG provider at much higher resolutions (e.g., 50cm).

Because BlueTopo macro-tiles aggregate both sparse historic data (which has no BAG equivalent) and modern high-resolution arrays, topobathykit **intentionally fetches both layers without attempting to pre-filter redundancies at the network level**.

The engine gracefully resolves this redundancy automatically in memory:

1. **Base Canvas**: The full BlueTopo mosaic (e.g., 4m resolution) is fetched and instantiated, establishing a continuous base elevation canvas that includes the older historical soundings.
2. **High-Res Overlay**: The engine simultaneously fetches the hyper-resolution 50cm or 1m BAG geometries for the modern surveys inside the bounding box.
3. **Spatial Overwrite**: Using the policy's defined transition operators (such as ``metric_feather`` blending), the native BAG arrays seamlessly overwrite their exact spatial footprints directly on top of the BlueTopo canvas in VRAM.

This top-to-bottom geographic override guarantees that older, non-BAG surveys inside the BlueTopo macro-tile are preserved, while the modern active swaths are dynamically upgraded to their highest possible fidelity with zero visual seams, duplicated points, or wasted network requests.

NoData Sentinel Evidence Matrix
-------------------------------

topobathykit treats NoData handling as a provider-specific ingest concern and a
merge-boundary safety concern. The table below records currently implemented,
evidence-backed handling:

.. list-table:: Provider NoData handling (evidence-backed)
     :header-rows: 1
     :widths: 20 22 34 24

     * - Provider
         - Observed/Declared NoData
         - Evidence
         - Runtime Handling
     * - NOAA Topobathy
         - ``-999999``
         - Confirmed cached leakage and downstream contamination (Issue #31)
         - Explicit sentinel + metadata masking at ingest, fetch, and merge boundaries
     * - NOAA BlueTopo
         - ``-999999`` (defensive)
         - Operational parity with NOAA topobathy COG family and observed coastal COG conventions
         - Metadata masking + explicit ``-999999`` fallback at ingest and merged output
     * - NCEI BAG
         - ``-1000000`` / ``1000000``
         - Existing BAG XML/GDAL leakage handling in provider comments and prior fixes
         - Explicit sentinel masking plus generic metadata-driven masking
     * - USGS Lidar (PDAL rasterization)
         - ``-9999``
         - Provider explicitly writes ``nodata=-9999`` in PDAL writer config
         - Explicit ``-9999`` + metadata masking before provenance and merge
     * - USGS 3DEP / CopDEM / NASADEM
         - metadata-dependent, plus ``-9999`` fallback
         - Mixed collection assets with tokenized COGs; robust safety needed at merge boundaries
         - Metadata-driven masking with explicit ``-9999`` defensive fallback
     * - NCEI CUDEM
         - metadata-dependent
         - Provider reads GeoTIFF nodata metadata from source/cache tiles
         - Shared metadata-driven masking at ingest and post-merge
     * - GEBCO 2025
         - metadata-dependent
         - OPeNDAP grid values are generally well-formed; masking still applied as safety net
         - Shared metadata-driven masking on tile ingest and final merged elevation

Implementation Note:

Shared sanitization is centralized in ``topobathykit.providers.base.sanitize_elevation_nodata``.
This keeps provider behavior consistent while still allowing provider-specific sentinel overrides.
