"""
Operators module for topobathykit.

Exposes geospatial operations like blending and fusion algorithms.
"""

from .blend import metric_feather

__all__ = ["metric_feather"]
