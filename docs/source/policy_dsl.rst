Fusion Policy DSL
=================

topobathykit logic is defined in **Fusion Policy** files (YAML). A policy describes *what* data to fetch and *how* to combine it to create a variable (e.g., elevation).

Structure
---------

A policy file consists of metadata and a list of variables (layers) to generate.

.. code-block:: yaml

    name: "Western Long Island Sound Fusion"
    version: "1.1.0"
    crs: "EPSG:3857"  # Output Coordinate Reference System

    variables:
      - name: "elevation"
        description: "Topobathymetry (Meters)"
        steps:
          # List of composition steps (Bottom-Up order)
          - provider: "gebco"
            product: "sub_ice_topo_bathymetry"
            operator: "overwrite"

          - provider: "usgs_3dep"
            product: "1m"
            operator: "overwrite"

Variables
---------

Currently, the runtime supports generating a single variable: ``elevation``. Future versions may support uncertainty or source masks.

Composition Steps
-----------------

The ``steps`` list defines the fusion order. Steps are processed sequentially. Later steps are applied *on top of* previous steps using the specified **Operator**.

Fields
~~~~~~

- **provider**: The registered provider alias (e.g., ``gebco``, ``usgs_3dep``, ``ncei_bag``).
- **product**: (Optional) Product sub-type passed to the provider.
- **operator**: The blending operation to apply.
- **description**: (Optional) Human-readable note.
- **filter**: (Optional) FilterConfig options mapping to provider-specific arguments (e.g. ``include_patterns``, ``exclude_patterns``, ``max_interpolation_gap_m``).
- **transitions**: (Optional) A list of rules for smoothing edges based on what lies beneath.

Advanced Provider Filtering (FilterConfig)
------------------------------------------

Providers can accept arbitrary metadata or filtering instructions via the ``filter`` block. This is especially useful for excluding specific interpolations or natively smoothing sub-scale gaps before the data ever reaches the fusion compiler.

.. code-block:: yaml

    - provider: "noaa_bluetopo"
      operator: "metric_feather"
      filter:
        exclude_patterns: ["*.interpolated", "Chart*"]
        max_interpolation_gap_m: 15.0

- **include_patterns** / **exclude_patterns**: A list of strings (supporting ``fnmatch`` wildcards like ``*``) to whitelist or blacklist specific survey IDs or URLs (e.g. omitting historical or interpolated data from BlueTopo or BAGs).
- **max_interpolation_gap_m**: Float distance in meters. Any internal `NaN` gap (data holiday) smaller than this physical distance will be natively interpolated via distance-transform morphology by the provider before it is merged. This heals "screen-door" artifacts without blurring the actual survey boundaries.

Operators
---------

overwrite
~~~~~~~~~
Hard replacement. Where the new layer has data (valid), it replaces the underlying value.

.. code-block:: yaml

    operator: "overwrite"

metric_feather
~~~~~~~~~~~~~~
Smooths the transition between the new layer and the base layer over a specified distance.

.. code-block:: yaml

    operator: "metric_feather"
    blend_distance: 100.0  # Meters

Topologies (Transition Rules)
-----------------------------

For advanced fusion, you can specify **Transition Rules** to apply different operators depending on *which* provider is currently at a pixel in the base layer.

*Example*: Feather into GEBCO (Bathymetry), but Hard-Cut against USGS Lidar (Land).

.. code-block:: yaml

    - provider: "noaa_bluetopo"
      operator: "overwrite"  # Default
      transitions:
        - target_provider: "gebco"
          operator: "metric_feather"
          blend_distance: 500.0


Runtime Policy Override
-----------------------

Policies are normally loaded from a YAML file at server startup, but both the
``/fuse`` and ``/hydrate`` endpoints accept a ``policy_yaml`` field that lets
you pass raw YAML at request time. This is useful for:

- **Testing a policy change** before committing it to the server.
- **One-off fusion** with a region-specific provider stack.
- **Hydrating a cache** for a custom policy without restarting the server.

Custom policies produce **separate cache entries** (keyed by a SHA256 hash of
the policy content), so they never collide with the server default or with each
other.

See :doc:`fuse_endpoint` and :doc:`hydration` for API details and examples.
