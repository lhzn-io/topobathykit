# Cache Hydration

## Overview

Hydration pre-warms the fused Zarr cache for a region so that subsequent
`/fuse` requests return instantly from cache instead of computing on the fly.

Unlike `/fuse` (which is synchronous and returns data), `/hydrate` is an
asynchronous workflow: it accepts a bounding box, splits it into 0.05° grid
cells, processes each cell through the full fusion pipeline, writes results to
`~/.cache/topobathykit/fused_zarr/`, and reports progress.

## Architecture

```{mermaid}
sequenceDiagram
    participant Client
    participant Server as FastAPI Server
    participant Sub as Hydration Subprocess
    participant FS as JSON State File
    participant Cache as Zarr Cache

    Client->>Server: POST /hydrate {bbox, resolution}
    Server->>FS: Write initial state (pending)
    Server->>Sub: Spawn subprocess
    Server-->>Client: {job_id, ws_url}

    loop For each grid cell (batched)
        Sub->>Cache: Check cell cache
        alt Cache miss
            Sub->>Cache: Fuse & write .zarr
        end
        Sub->>FS: Atomic write progress
    end
    Sub->>FS: Write final state (completed/failed)

    alt WebSocket monitoring
        Client->>Server: WS /hydrate/{job_id}/ws
        loop Until done
            Server->>FS: Read state
            Server-->>Client: JSON frame
        end
    else HTTP polling
        loop Until done
            Client->>Server: GET /hydrate/{job_id}
            Server->>FS: Read state
            Server-->>Client: JSON response
        end
    end
```

### Why a subprocess?

The hydration task loads large raster datasets (NCEI BAG tiles at 50cm–2m
resolution, CUDEM DEMs, etc.) that can consume gigabytes of memory. Running
this as a subprocess means:

- **OOM isolation** — if the subprocess is killed by the OS, the web server
  stays up and the job state file records the failure.
- **No shared mutable state** — all communication is via atomic JSON file
  writes (`write .tmp` then `os.replace`), eliminating race conditions.
- **Multi-worker safe** — any uvicorn worker can serve `GET /hydrate/{job_id}`
  by reading the state file from disk.

### Dead process detection

When a reader (GET or WebSocket) loads a state file that says `"status": "running"`,
it checks whether the recorded PID is still alive via `os.kill(pid, 0)`. If the
process is gone, the state is atomically updated to `"status": "failed"` with an
error message indicating the likely cause (OOM).

## Usage

### CLI script

The `hydrate_cache` script provides two subcommands: `fuse` (zarr cells) and
`tiles` (rendered PNGs).

```bash
# Hydrate fused zarr cells (30m default)
python -m topobathykit.scripts.hydrate_cache fuse -70.99 42.89 -70.49 43.26

# 15m resolution with custom policy
python -m topobathykit.scripts.hydrate_cache fuse -70.99 42.89 -70.49 43.26 \
  -r 15 \
  -c config/topobathykit/policies/great_bay_estuary.yaml

# Hydrate rendered PNG tiles at zoom 13
python -m topobathykit.scripts.hydrate_cache tiles -70.99 42.89 -70.49 43.26 -z 13

# HTTP polling only (skip WebSocket)
python -m topobathykit.scripts.hydrate_cache fuse -70.99 42.89 -70.49 43.26 --no-ws

# Custom service URL
python -m topobathykit.scripts.hydrate_cache fuse -70.99 42.89 -70.49 43.26 --url http://remote:9595
```

**Arguments:**

| Argument                 | Required | Default                  | Description                        |
| ------------------------ | -------- | ------------------------ | ---------------------------------- |
| `WEST SOUTH EAST NORTH`  | yes      | —                        | Bounding box (WGS84)               |
| `-c, --config`           | no       | server default           | Path to YAML policy file           |
| `-r, --resolution`       | no       | 30.0                     | Output resolution in meters        |
| `--url`                  | no       | `http://localhost:9595`  | Service base URL                   |
| `--no-ws`                | no       | false                    | Disable WebSocket, use HTTP polling|

### REST API

**Submit a job:**

```text
POST /hydrate
Content-Type: application/json

{
  "bbox": [-70.99, 42.89, -70.49, 43.26],
  "resolution": 15,
  "policy_yaml": "name: Custom\ncrs: EPSG:4326\nvariables: ..."
}
```

**Request fields:**

| Field          | Type            | Required | Default | Description                                        |
| -------------- | --------------- | -------- | ------- | -------------------------------------------------- |
| `bbox`         | `[w, s, e, n]`  | yes      | —       | Bounding box in WGS84 decimal degrees              |
| `resolution`   | float           | no       | 30.0    | Output resolution in meters                        |
| `policy_yaml`  | string          | no       | null    | Raw YAML policy to override server default         |
| `max_workers`  | int (1–16)      | no       | 2       | Thread pool parallelism for cell processing        |

When `policy_yaml` is provided, the custom policy is used instead of the
server's default. The cache key is a SHA256 hash of the policy content +
cell bbox + resolution, so custom policies produce **separate cache entries**
that never collide with the server default. See the "Cache behavior" section
in {doc}`fuse_endpoint` for details.

**Response:**

```json
{
  "status": "accepted",
  "job_id": "a1b2c3d4-...",
  "ws": "ws://localhost:9595/hydrate/a1b2c3d4-.../ws",
  "bbox": [-70.99, 42.89, -70.49, 43.26],
  "resolution": 15
}
```

**Poll status:**

```text
GET /hydrate/{job_id}
```

```json
{
  "id": "a1b2c3d4-...",
  "status": "running",
  "total_cells": 48,
  "processed_cells": 12,
  "cached_cells": 30,
  "failed_cells": 0
}
```

**List recent jobs:**

```text
GET /hydrate
```

### WebSocket

Connect to `ws://host:port/hydrate/{job_id}/ws` to receive real-time JSON
frames as each cell completes. The server only sends a frame when state
changes (deduplicated). The connection closes automatically when the job
reaches `completed` or `failed` status.

### Web UI

Navigate to the hydration page at `http://localhost:9595/static/hydrate.html`.
The UI provides a Leaflet map for bbox selection, resolution controls, optional
policy YAML upload, and live progress monitoring via WebSocket.

## Grid cell caching

Hydration splits the requested bbox into a fixed 0.05° grid. Each cell is
cached independently at `~/.cache/topobathykit/fused_zarr/{hash}.zarr`, where
the hash is derived from the policy content, cell bbox, and resolution.

If you request a second bbox that overlaps a previously hydrated region, the
overlapping cells are **cache hits** — only new cells are processed. This
makes incremental hydration of adjacent regions efficient.

## Memory management

The hydration subprocess uses several strategies to limit peak memory:

- **Batched cell processing** — cells are submitted to a `ThreadPoolExecutor`
  in small batches (default: `max_workers * 2`) rather than all at once.
  `gc.collect()` runs between batches.
- **Per-cell cleanup** — after each cell is written to Zarr, dataset
  references are explicitly deleted and garbage collected.
- **BAG LRU cache cap** — the NCEI BAG provider's in-memory LRU cache is
  capped at 8 entries (tunable via `TOPOBATHY_BAG_LRU_SIZE`).
- **Worker recycling** — uvicorn workers auto-restart after 200 requests
  (tunable via `--limit-max-requests`).

## Job state file format

State files are stored at `~/.cache/topobathykit/hydration_jobs/{job_id}.json`:

```json
{
  "id": "a1b2c3d4-...",
  "status": "running",
  "pid": 12345,
  "bbox": [-70.99, 42.89, -70.49, 43.26],
  "resolution": 15,
  "submitted_at": "2026-03-22T22:19:40+00:00",
  "total_cells": 48,
  "processed_cells": 12,
  "cached_cells": 30,
  "failed_cells": 0
}
```

Valid statuses: `pending`, `running`, `completed`, `failed`.

Completed and failed jobs are automatically pruned after 24 hours.
