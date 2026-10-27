"""Tests that rendered tile cache keys change with the fusion algorithm version."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from topobathyserve import main


def test_tile_cache_key_includes_fusion_algorithm(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = Path("policies/examples/wlis.yaml")
    params_now, hash_now = main._tile_cache_key(policy, None, None, 512)
    assert f"fusion_algo={main.FUSION_ALGO_VERSION}" in params_now

    monkeypatch.setattr(main, "FUSION_ALGO_VERSION", main.FUSION_ALGO_VERSION + 1)
    _, hash_next = main._tile_cache_key(policy, None, None, 512)
    assert hash_next != hash_now
    assert len(hash_next) == 8


def test_tile_cache_key_still_separates_render_parameters() -> None:
    policy = Path("policies/examples/wlis.yaml")
    keys = {
        main._tile_cache_key(policy, vmin, vmax, size)[1]
        for vmin, vmax, size in [(None, None, 512), (-50.0, None, 512), (None, None, 256)]
    }
    assert len(keys) == 3
