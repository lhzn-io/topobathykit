# Contributing to topobathykit

topobathykit is a policy-driven **topobathymetric fusion runtime** (library + optional service). The core idea is to fuse heterogeneous geospatial layers into data-first products (`xarray.Dataset`) with **per-pixel provenance**.

We welcome contributions, especially **new dataset providers** and **policy presets** for regions outside our initial Northeast US focus.

## Project principles

1. **Data-first outputs**
   - The canonical output is an `xarray.Dataset` (not PNGs).
   - Visual tiles and the viewer exist primarily for QA/provenance debugging.

2. **Overwrite in trusted zones; blend in transition zones**
   - Survey “truth” should win where present.
   - Use operators (seamline feathering, logistic blend, etc.) to avoid hard seams at coverage boundaries.

3. **Provenance is first-class**
   - Every fused variable must have a corresponding source mask (e.g., `source_elevation`).
   - Assumptions must be explicit: CRS, vertical reference/datum, transforms, and caveats belong in metadata.

4. **Policy-driven behavior**
   - Fusion strategies are defined in YAML policy presets (selected by filename).
   - New fusion logic should generally be expressed as:
     - a new **operator**, and/or
     - a new **policy preset**, rather than hardcoding rules into the runtime.

5. **International-friendly by default**
   - Avoid US-specific assumptions in core logic.
   - Region-specific logic belongs in providers and policy presets.

## Ways to contribute

### 1) Add a dataset provider (high value)

Providers are the main extension mechanism. A provider is responsible for:

- fetching/streaming remote data
- caching for offline replay
- normalizing CRS/metadata
- returning an `xarray.DataArray` + provider metadata

When adding a provider:

- Include dataset citation + license/terms (and any attribution requirements).
- Prefer stable endpoints (STAC, COG, COPC, OPeNDAP).
- Handle “data missing” gracefully: return `None` (or an empty/NaN layer) rather than synthesizing data.

### 2) Add a policy preset (high value)

Policy presets are YAML files under `policies/` and act as shareable “domain knowledge”:

- provider ordering per zone
- transition blending operators
- variable-specific strategies (elevation vs covariates)

Good presets include:

- a short description and intended region/use case
- recommended bounding boxes for demos
- expectations about coverage and vertical reference

### 3) Add an operator or QA test

Operators are small, testable functions used by policies (e.g., seamline feathering in meters).
Operators should:

- be deterministic
- work globally (CRS-aware; distances in meters)
- come with unit tests using synthetic rasters (no network access)

### 4) Improve docs and onboarding

Especially valuable:

- provider authoring docs
- policy authoring docs
- “iconic presets” walkthroughs (e.g., Great Barrier Reef)

## Development setup

We recommend **Micromamba** (or Mamba/Conda) for development because topobathykit depends on compiled geospatial libraries such as GDAL/PDAL.

### Option A: Micromamba (recommended)

```bash
micromamba create -f environment.yml
micromamba activate topobathykit
pip install -e .
```

### Option B: Python venv (pip)

If you prefer standard Python tooling, you must install system-level dependencies for GDAL/PDAL yourself.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

## Troubleshooting & Known Issues

### "Too many open files" (Errno 24) on macOS

If you encounter `[Errno 24] Too many open files` when running large fusions or processing many tiles, you may be hitting the operating system's file descriptor limit. This is common when working with Zarr-backed datasets (like `ncei_bag`) which can open many small files simultaneously.

On macOS, the default limit is often low (256). To fix this, increase the limit in your shell:

```bash
ulimit -n 10240
```

For a persistent fix, you may need to configure `limit maxfiles` via `launchctl`.

## Code quality tooling

### Pre-commit hooks

We use [pre-commit](https://pre-commit.com/) for formatting, linting, and type checks.

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

### Type checking

- All new code must be type-hinted.
- Run:

```bash
mypy
```

### Tests

- Run unit tests:

```bash
pytest
```

#### Network and dataset tests

If you add tests that require network access or large external downloads, mark them clearly (e.g., `@pytest.mark.network`) and keep them **opt-in** so CI remains stable.

## Contribution workflow

1. Fork the repository.
2. Create a feature branch.
3. Make changes with small, reviewable commits.
4. Add/extend tests.
5. Update docs (README, policies/README, provider docs) as needed.
6. Open a Pull Request describing:
   - what changed
   - how to test
   - dataset citations/licenses (if you added a provider or preset)
   - any known limitations (CRS, datum, coverage)

## Guidance for provider PRs (checklist)

- [ ] Provider returns a valid `xarray.DataArray` for the requested bbox (or `None` if unavailable)
- [ ] Provider caches downloads locally for offline replay
- [ ] Provider metadata includes citation and license/terms
- [ ] Vertical reference/datum is declared (or explicitly `"Unknown"`)
- [ ] Policy preset added/updated to demonstrate usage
- [ ] Provenance/source mask reflects the provider where it contributes
- [ ] Unit tests added (synthetic where possible); network tests opt-in

## Community norms

We aim for:

- clear, technical discussions
- explicit assumptions and reproducibility
- regionally inclusive contributions (providers and presets from outside the US are especially welcome)

Thank you for helping build a long-lived open-source fusion framework.
