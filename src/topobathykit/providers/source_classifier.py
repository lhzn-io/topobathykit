"""
Source Classifier Module for managing data duplication across providers.
"""

import re

# Rubric for identifying sources that are likely covered by other specific providers
# Regex patterns that match source IDs belonging to other integrations
PATTERNS = {
    # NCEI BAG (Tier 0)
    # Hxxxxx, Fxxxxx, Wxxxxx (standard hydrographic surveys)
    # or explicit "BAG" in name
    "ncei_bag": re.compile(r"(^[HFW]\d{5})|(\.bag$)", re.IGNORECASE),
    # NOAA Topobathy (Tier 1)
    # Explicit "topobathy" mention or specific NOAA/NGS naming conventions
    "noaa_topobathy": re.compile(r"(topobathy|ngs_topobathy)", re.IGNORECASE),
    # NCEI CUDEM (Tier 2)
    "ncei_cudem": re.compile(r"(cudem|topo_dem)", re.IGNORECASE),
    # USGS 3DEP (Tier 2/3)
    "usgs_3dep": re.compile(r"(usgs_3dep|usgs_dem|ned|3dep)", re.IGNORECASE),
    # GEBCO (Tier 4 - Fallback)
    "gebco_2025": re.compile(r"(gebco_202\d|gebco)", re.IGNORECASE),
}


def classify_source_provider(source_id: str) -> str | None:
    """
    Analyze a source identifier (string) and return the provider key
    if it matches a known pattern of another integration.

    Args:
        source_id: The raw identification string from the metadata (e.g. "H12345", "2023_ngs_topobathy...")

    Returns:
        Provider key (e.g. "ncei_bag") if matched, else None.
    """
    if not source_id:
        return None

    s = str(source_id).strip()

    for provider, pattern in PATTERNS.items():
        if pattern.search(s):
            return provider

    return None
