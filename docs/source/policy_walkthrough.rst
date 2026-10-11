Policy Walkthrough
==================

This guide walks through two real-world Fusion Policies: a simple regional overlay and a complex multi-tier coastal model.

1. Simple Fusion: Great Barrier Reef
------------------------------------

**Goal**: Enhance the global GEBCO bathymetry with high-resolution (100m) local survey data for the Great Barrier Reef (GBR100).

**Key Concepts**:
*   **Base Layer**: GEBCO 2025 (~450m resolution).
*   **Overlay**: GBR100 (~100m resolution).
*   **Metric Feathering**: Blending the edges over 500m to hide the seam.

.. code-block:: yaml

    name: "Great Barrier Reef Fusion"
    crs: "EPSG:4326"

    variables:
      - name: "elevation"
        steps:
          # 1. Base Layer (Global)
          - provider: "gebco_2025"
            operation: "overwrite"

          # 2. Local Overlay (Feathered)
          - provider: "jcu_gbr100"
            operation: "metric_feather"
            blend_distance: 500.0

**Outcome**:
The output is a seamless grid where GBR100 data dominates usage, but smoothly transitions to GEBCO in deep water or at the edge of the survey.

2. Complex Fusion: US Coastal (WLIS)
------------------------------------

**Goal**: Create a "Digital Twin" quality model for Western Long Island Sound (WLIS) using every available dataset priority.

**Key Concepts**:
*   **Priority Overwrite**: Using a strict hierarchy where better data strictly replaces worse data.
*   **7-Layer Stack**: Handling Terrestrial, Intertidal, and Bathymetric zones securely.
*   **Global Projection**: Using EPSG:3857 (Web Mercator) for compatibility with web viewers.

**The Stack (Bottom-Up)**:

1.  **Global Base**: GEBCO 2025 (Ensures no void pixels).
2.  **Regional Land**: USGS 3DEP (10m) for upland context.
3.  **Regional Bathy**: NOAA CUDEM (3m) gap-filling coastal zones.
4.  **Modern Bathy**: NOAA BlueTopo (2-8m) high-quality modeled bathymetry.
5.  **Shallow Water**: Topobathy Lidar (Green Laser) for the intertidal zone.
6.  **Upland Detail**: USGS 3DEP Lidar (Airborne) for structures and coastline.
7.  **Survey Truth**: NOAA BAGs (Multibeam) for precise channel depth.

.. code-block:: yaml

    name: "US Coastal Topobathy Fusion"
    crs: "EPSG:3857"

    variables:
      - name: "elevation"
        steps:
          # --- Base Layers ---
          - provider: "gebco_2025"
            operation: "overwrite"

          - provider: "usgs_3dep"
            product: "10m"
            operation: "overwrite"

          # --- Middle Tiers ---
          - provider: "ncei_cudem"
            operation: "overwrite"

          - provider: "noaa_bluetopo"
            operation: "overwrite"

          # --- High Res Lidar ---
          - provider: "noaa_topobathy"
            operation: "overwrite"

          - provider: "usgs_lidar"
            operation: "overwrite"

          # --- Survey Truth (Top Priority) ---
          - provider: "ncei_bag"
            operation: "overwrite"

**Outcome**:
A high-fidelity model that is "Best Available" at every pixel. If a ship survey (BAG) exists, it is used. If not, Lidar is used. If that fails, BlueTopo/CUDEM are used. Finally, GEBCO fills the deep ocean.
