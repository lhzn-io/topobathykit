"""
Provider registry and implementations for topobathykit.
"""

from .base import Provider
from .registry import registry

# Automatically discover and register all providers in this package
registry.auto_discover()

__all__ = [
    "Provider",
    "gebco_2025",
    "jcu_gbr100",
    "ncei_bag",
    "ncei_cudem",
    "noaa_bluetopo",
    "noaa_topobathy",
    "registry",
    "usgs_3dep",
    "usgs_lidar",
]
