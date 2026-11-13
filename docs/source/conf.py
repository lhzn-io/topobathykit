# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = "topobathykit"
copyright = "2026, Long Horizon Observatory"
author = "Daniel Fry"
release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "lhzn"
html_theme_options = {
    "source_repository": "https://github.com/lhzn-io/topobathykit",
    "source_branch": "main",
    "source_directory": "docs/source/",
}
html_static_path = ["_static"]
html_extra_path = ["_extra"]

autodoc_mock_imports = [
    "osgeo",
    "xarray",
    "rioxarray",
    "numpy",
    "scipy",
    "pyproj",
    "affine",
    "rasterio",
    "h5py",
    "zarr",
    "fsspec",
    "s3fs",
    "geopandas",
    "pdal",
    "pydap",
    "matplotlib",
    "planetary_computer",
    "pystac_client",
    "requests",
    "httpx",
    "filelock",
    "pydantic",
    "uvicorn",
    "fastapi",
    "dotenv",
    "questionary",
    "rich",
    "lazrs",
    "yaml",
]
