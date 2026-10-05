# /fuse Endpoint — Zarr & GeoTIFF Output

## Overview

The `/fuse` endpoint fuses topobathymetric data on-demand and returns it in your choice of:

- **Zarr** (ZIP archive) — preferred for streaming and cross-platform use, especially Julia
- **GeoTIFF** — traditional raster format

## API

```http
GET /fuse?bbox=west,south,east,north&resolution=30&format=zarr
```

### Query Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `bbox` | string | * | — | Bounding box as `west,south,east,north` |
| `west` | float | * | — | Bounding box west (longitude) |
| `south` | float | * | — | Bounding box south (latitude) |
| `east` | float | * | — | Bounding box east (longitude) |
| `north` | float | * | — | Bounding box north (latitude) |
| `resolution` | float | no | 30.0 | Output resolution in meters |
| `format` | string | no | `geotiff` | Output format: `geotiff`, `tiff`, `tif`, `zarr`, `zip` |

**Note:** Provide **either** `bbox` **or** all four (`west`, `south`, `east`, `north`). West < East and South < North are enforced.

## Response

### Zarr (ZIP Archive)

```http
HTTP/1.1 200 OK
Content-Type: application/zip
Content-Disposition: attachment; filename=fused.zarr.zip

[binary zip file containing Zarr store]
```

The ZIP archive contains:

- `elevation/` — fused DEM array (float32)
- `source_elevation/` — per-pixel provider ID (uint32, optional)
- `.zattrs` — metadata (JSON)
- `.zgroup` — Zarr group metadata

### Zarr Metadata (`.zattrs`)

```json
{
  "crs": "EPSG:4326",
  "bbox": [-73.85, 40.78, -73.72, 40.82],
  "resolution_m": 10.0,
  "vertical_datum": "NAVD88",
  "policy": "wlis.yaml",
  "providers_used": ["gebco_2025", "usgs_3dep", "noaa_topobathy"],
  "created": "2026-03-08T14:23:45.123456+00:00",
  "provenance_dict": {
    "1": {"provider": "gebco_2025", "product": "sub_ice_topo_bathymetry"},
    "2": {"provider": "usgs_3dep", "product": "1m"},
    ...
  }
}
```

### GeoTIFF

```http
HTTP/1.1 200 OK
Content-Type: image/tiff

[binary GeoTIFF file]
```

Returns a single-band GeoTIFF with elevation data. CRS and geotransform are embedded.

## Examples

### Julia (Zarr)

```julia
using Zarr, Downloads

# Download fused elevation as Zarr
url = "http://localhost:9595/fuse?bbox=-73.85,40.78,-73.72,40.82&resolution=10&format=zarr"
filepath = "fused.zarr.zip"
Downloads.download(url, filepath)

# Open and read into memory
store = Zarr.zopen(filepath)
elevation = store["elevation"][:, :]
metadata = store.attrs
println("CRS: $(metadata["crs"])")
println("Providers: $(metadata["providers_used"])")
```

### Python

```python
import requests
import zarr
import xarray as xr
from pathlib import Path

url = "http://localhost:9595/fuse?bbox=-73.85,40.78,-73.72,40.82&format=zarr"
response = requests.get(url)

# Save and read Zarr
Path("fused.zarr.zip").write_bytes(response.content)
store = zarr.ZipStore("fused.zarr.zip", mode="r")
ds = xr.open_dataset(store, engine="zarr")
print(ds)
store.close()
```

### cURL (GeoTIFF)

```bash
curl "http://localhost:9595/fuse?bbox=-73.85,40.78,-73.72,40.82&format=geotiff" \
  -o fused.tif
gdalinfo fused.tif
```

## POST /fuse — Dynamic Policy Override

The `POST` variant accepts a JSON body and supports **dynamic policy switching**,
allowing you to override the server's default fusion policy per request.

```text
POST /fuse
Content-Type: application/json
```

### Request Body

| Field          | Type          | Required | Default    | Description                                          |
| -------------- | ------------- | -------- | ---------- | ---------------------------------------------------- |
| `bbox`         | `[w, s, e, n]` | yes    | —          | Bounding box in WGS84 decimal degrees                |
| `resolution`   | float         | no       | 30.0       | Output resolution in meters                          |
| `format`       | string        | no       | `zarr`     | Output format: `zarr` or `geotiff`                   |
| `policy_yaml`  | string        | no       | null       | Raw YAML policy content to override default fusion   |
| `policy_name`  | string        | no       | null       | Name of a server-side policy (reserved, not yet implemented) |

When `policy_yaml` is provided, it takes precedence over both `policy_name`
and the server's default policy. The raw YAML string is parsed and used as
the fusion policy for that request only.

### Example — custom policy

```python
import requests, yaml

# Define a minimal custom policy inline
policy = {
    "name": "Simple GEBCO+CUDEM",
    "crs": "EPSG:4326",
    "variables": [
        {
            "name": "elevation",
            "steps": [
                {"provider": "gebco_2025", "operator": "overwrite"},
                {"provider": "ncei_cudem", "operator": "overwrite"},
            ],
        }
    ],
}

resp = requests.post(
    "http://localhost:9595/fuse",
    json={
        "bbox": [-73.85, 40.78, -73.72, 40.82],
        "resolution": 30,
        "format": "zarr",
        "policy_yaml": yaml.dump(policy),
    },
)
```

### Cache behavior

Custom policies are **cached separately** from the server default. The cache
key is a SHA256 hash of the policy content + bbox + resolution, so:

- Two requests with the same custom YAML hit the same cache entry.
- A custom policy request never collides with a default policy request.
- Modifying a single line in the YAML produces a different cache key.

Cached results live at `~/.cache/topobathykit/fused_zarr/{hash}.zarr`.

## Error Handling

| Status | Reason                                                       |
| ------ | ------------------------------------------------------------ |
| `400`  | Invalid bbox parameters, ordering, format, or missing values |
| `404`  | Fusion returned no elevation data (bbox outside valid area)  |
| `500`  | Runtime fusion error                                         |

## Performance Notes

- Zarr is **chunked** and supports streaming access via fsspec (future enhancement)
- GeoTIFF returns a single monolithic file
- Default resolution is 30m; increase for faster response times
- Results are cached in `~/.cache/topobathykit/fused_zarr/` by policy, bbox, and resolution
