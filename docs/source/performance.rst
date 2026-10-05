Performance notes
=================

Reference development machines
------------------------------

topobathykit has primarily been developed and tested on:

- ASUS ROG Strix Scar 16 laptop (64 GB RAM)
- Mac Studio (M4 Max, 128 GB RAM)

These are **not** formal benchmarks. They are reference machines used to validate end-to-end workflows
(provider fetching/caching, fusion, and tile serving). Many use-cases (small regions, global basemap presets,
offline analysis) can run with substantially less memory, especially after caches are warmed.

Practical guidance
------------------

- Large regions and high-resolution presets can cache **multiple GB** of source assets and derived products.
- For server usage, start with a small worker count and scale up based on IO and memory behavior.
- When in doubt, validate your policy on a small bounding box first, then expand.
