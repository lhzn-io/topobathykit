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
            operation: "overwrite"

          - provider: "usgs_3dep"
            product: "1m"
            operation: "overwrite"

Variables
---------

Currently, the runtime supports generating a single variable: ``elevation``. Future versions may support uncertainty or source masks.

Composition Steps
-----------------

The ``steps`` list defines the fusion order. Steps are processed sequentially. Later steps are applied *on top of* previous steps using the specified **Operator**.

Fields
~~~~~~

- **provider**: The registered provider alias (e.g., ``gebco``, ``usgs_3dep``, ``ncei_bag``).
- **operation**: The blending operation to apply: ``overwrite`` or ``metric_feather`` (the default). ``metric_feather`` without a ``blend_distance`` behaves as ``overwrite``.
- **blend_distance**: (Optional) Feathering distance in meters for ``metric_feather``.
- **product**: (Optional) Informational label for the provider product. It is not passed to the provider.
- **description**: (Optional) Human-readable note.
- **filter**: (Optional) FilterConfig options mapping to provider-specific arguments (e.g. ``include_patterns``, ``exclude_patterns``, ``max_interpolation_gap_m``).
- **transitions**: (Optional) A list of rules for smoothing edges based on what lies beneath.

Unknown keys in a step, a transition rule or a variable are rejected when the
policy is loaded, so a misspelled field fails loudly instead of falling back to
a default. The step key ``operator`` is still accepted as a deprecated alias of
``operation`` and emits a ``FutureWarning``. Transition rules are the exception:
their key is ``operator`` (see below).

Advanced Provider Filtering (FilterConfig)
------------------------------------------

Providers can accept arbitrary metadata or filtering instructions via the ``filter`` block. This is especially useful for excluding specific interpolations or natively smoothing sub-scale gaps before the data ever reaches the fusion compiler.

.. code-block:: yaml

    - provider: "noaa_bluetopo"
      operation: "metric_feather"
      filter:
        exclude_patterns: ["*.interpolated", "Chart*"]
        max_interpolation_gap_m: 15.0

- **include_patterns** / **exclude_patterns**: A list of strings (supporting ``fnmatch`` wildcards like ``*``) to whitelist or blacklist specific survey IDs or URLs (e.g. omitting historical or interpolated data from BlueTopo or BAGs). For ``usgs_3dep`` the patterns match the STAC asset file name (e.g. ``USGS_13_n41w074.tif``), the same names ``/analyze`` reports.
- **max_interpolation_gap_m**: Float distance in meters. Any internal `NaN` gap (data holiday) smaller than this physical distance will be natively interpolated via distance-transform morphology by the provider before it is merged. This heals "screen-door" artifacts without blurring the actual survey boundaries.

Operators
---------

overwrite
~~~~~~~~~
Hard replacement. Where the new layer has data (valid), it replaces the underlying value.

.. code-block:: yaml

    operation: "overwrite"

metric_feather
~~~~~~~~~~~~~~
Smooths the transition between the new layer and the base layer over a specified distance.

.. code-block:: yaml

    operation: "metric_feather"
    blend_distance: 100.0  # Meters

Topologies (Transition Rules)
-----------------------------

For advanced fusion, you can specify **Transition Rules** to apply different operators depending on *which* provider supplied the current value of a pixel in the base layer. The runtime tracks this per pixel by provider, independent of the finer per-survey ids written to ``source_elevation``.

*Example*: Feather into GEBCO (Bathymetry), but Hard-Cut against USGS Lidar (Land).

.. code-block:: yaml

    - provider: "noaa_bluetopo"
      operation: "overwrite"  # Applied where no transition rule matches
      transitions:
        - target_provider: "gebco_2025"
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
