from topobathykit.policy.loader import hash_policy
from topobathykit.runtime import get_fused_cache_info


def test_hash_policy_stability() -> None:
    """Test that hash_policy produces stable hashes for the same content."""
    policy_dict = {
        "crs": "EPSG:3857",
        "variables": [
            {
                "name": "elevation",
                "steps": [
                    {"provider": "gebco_2025", "priority": 10},
                    {"provider": "noaa_bluetopo", "priority": 20},
                ],
            }
        ],
    }

    hash1 = hash_policy(policy_dict)

    # Reorder keys in dict (should not affect hash due to sort_keys=True)
    policy_dict_reordered = {
        "variables": policy_dict["variables"],
        "crs": "EPSG:3857",
    }
    hash2 = hash_policy(policy_dict_reordered)

    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 length


def test_hash_policy_change() -> None:
    """Test that changing content changes hash."""
    p1 = {"crs": "EPSG:3857"}
    p2 = {"crs": "EPSG:4326"}

    assert hash_policy(p1) != hash_policy(p2)


def test_get_fused_cache_info_hashing() -> None:
    """Test that cache key involves policy content."""
    bbox = (-74.0, 40.0, -73.0, 41.0)

    # Create two policies with same structure but different content
    policy_str1 = """
    name: test_policy_1
    crs: EPSG:4326
    variables:
      - name: elevation
        steps:
          - provider: gebco_2025
    """

    policy_str2 = """
    name: test_policy_2
    crs: EPSG:4326
    variables:
      - name: elevation
        steps:
          - provider: noaa_bluetopo
    """

    path1, hash1 = get_fused_cache_info(policy_str1, bbox)
    path2, hash2 = get_fused_cache_info(policy_str2, bbox)

    assert hash1 != hash2
    assert path1 != path2
    assert hash1 in str(path1)
