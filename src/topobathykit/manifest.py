import fcntl
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class OfflineManifest:
    """
    Thread-safe persistent manifest for tracking cached STAC assets.
    Maps spatial bounds/collections to Asset URLs, enabling offline lookup.
    """

    def __init__(self, cache_dir: Path, filename: str = "stac_manifest.json") -> None:
        self.manifest_path = cache_dir / filename
        self._ensure_manifest()

    def _ensure_manifest(self) -> None:
        if not self.manifest_path.exists():
            with open(self.manifest_path, "w") as f:
                json.dump({"items": [], "searches": []}, f)

    def add_item(
        self, collection_id: str, bbox: tuple, asset_href: str, properties: dict | None = None
    ) -> None:
        """
        Records a STAC item in the manifest.
        """
        entry = {
            "collection": collection_id,
            "bbox": bbox,  # [minx, miny, maxx, maxy]
            "href": asset_href,
            "properties": properties or {},
        }

        # Lock and Update
        try:
            with open(self.manifest_path, "r+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                data = json.load(f)

                # Check for duplicate (by Base URL - ignoring SAS tokens)
                # This prevents accumulation of the same asset with different expiring signatures.
                asset_base = asset_href.split("?")[0]

                existing_idx = None
                for i, x in enumerate(data["items"]):
                    if x["href"].split("?")[0] == asset_base:
                        existing_idx = i
                        break

                if existing_idx is not None:
                    data["items"][existing_idx] = entry
                else:
                    data["items"].append(entry)

                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            logger.error(f"Failed to update manifest: {e}")

    def find_items(self, collection_id: str, search_bbox: tuple) -> list[dict]:
        """
        Finds items in the manifest that intersect the search_bbox.
        search_bbox: (minx, miny, maxx, maxy)
        """
        try:
            # Shared lock for reading? Or just quick read.
            # "r" open doesn't blocking-wait on lock usually unless we flock it.
            # Let's be safe and use shared lock.
            with open(self.manifest_path) as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                data = json.load(f)
                fcntl.flock(f, fcntl.LOCK_UN)

            matches = []
            s_minx, s_miny, s_maxx, s_maxy = search_bbox

            for item in data.get("items", []):
                if item["collection"] != collection_id and collection_id != "*":
                    continue

                # Intersect Check
                i_minx, i_miny, i_maxx, i_maxy = item["bbox"]

                # Standard AABB intersection
                if i_minx <= s_maxx and i_maxx >= s_minx and i_miny <= s_maxy and i_maxy >= s_miny:
                    matches.append(item)

            return matches

        except Exception as e:
            logger.error(f"Failed to read manifest: {e}")
            return []

    def remove_item_by_href(self, href: str) -> None:
        """
        Removes an item from the manifest by matching the base URL (ignoring tokens).
        """
        try:
            target_base = href.split("?")[0]

            with open(self.manifest_path, "r+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                data = json.load(f)

                original_len = len(data.get("items", []))
                # Filter out matching items
                data["items"] = [x for x in data.get("items", []) if x["href"].split("?")[0] != target_base]

                if len(data["items"]) < original_len:
                    logger.debug(f"Removed {original_len - len(data['items'])} stale items from manifest.")
                    f.seek(0)
                    json.dump(data, f, indent=2)
                    f.truncate()

                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            logger.error(f"Failed to remove item from manifest: {e}")

    def record_search(self, collection_id: str, bbox: tuple, found_count: int) -> None:
        """
        Records a search result. Useful for Negative Caching (recording 0 results).
        """
        # Only record empty searches for now to save space/complexity
        if found_count > 0:
            return

        entry = {
            "collection": collection_id,
            "bbox": bbox,
            "found": 0,
        }

        try:
            with open(self.manifest_path, "r+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                data = json.load(f)

                if "searches" not in data:
                    data["searches"] = []

                # Naive append for now. Could implement spatially merging bboxes later.
                # Check for exact duplicate to avoid spam
                is_duplicate = False
                for s in data["searches"]:
                    if s["collection"] == collection_id and s["bbox"] == list(bbox) and s["found"] == 0:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    data["searches"].append(entry)
                    f.seek(0)
                    json.dump(data, f, indent=2)
                    f.truncate()

                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            logger.error(f"Failed to record search in manifest: {e}")

    def has_no_coverage(self, collection_id: str, search_bbox: tuple) -> bool:
        """
        Checks if the search_bbox is strictly contained within a previously searched
        region that returned 0 results.
        """
        try:
            with open(self.manifest_path) as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                data = json.load(f)
                fcntl.flock(f, fcntl.LOCK_UN)

            s_minx, s_miny, s_maxx, s_maxy = search_bbox

            for s in data.get("searches", []):
                if s["collection"] != collection_id:
                    continue
                if s["found"] != 0:
                    continue

                # Check Containment: Is search_bbox INSIDE the empty_bbox?
                e_minx, e_miny, e_maxx, e_maxy = s["bbox"]

                if s_minx >= e_minx and s_maxx <= e_maxx and s_miny >= e_miny and s_maxy <= e_maxy:
                    return True

            return False

        except Exception as e:
            logger.error(f"Failed to check negative cache: {e}")
            return False
