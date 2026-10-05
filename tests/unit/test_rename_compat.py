"""The one-release compatibility paths left by the topobathysim -> topobathykit rename."""

import importlib
import sys
import warnings
from pathlib import Path

import pytest

from topobathykit import config


def test_old_package_aliases_new_modules_with_a_warning() -> None:
    for name in [m for m in sys.modules if m == "topobathysim" or m.startswith("topobathysim.")]:
        del sys.modules[name]
    with pytest.warns(DeprecationWarning, match="renamed topobathykit"):
        old = importlib.import_module("topobathysim")
    assert old.__name__ == "topobathysim"

    from topobathysim.config import get_cache_root

    import topobathykit.config

    assert get_cache_root is topobathykit.config.get_cache_root
    assert importlib.import_module("topobathysim.policy") is importlib.import_module("topobathykit.policy")


def test_unknown_old_submodule_still_fails() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        importlib.import_module("topobathysim")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("topobathysim.no_such_module")


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    for name in (
        "TOPOBATHYKIT_CACHE_DIR",
        "TOPOBATHYSIM_CACHE_DIR",
        "TOPOBATHYKIT_DEBUG",
        "TOPOBATHYSIM_DEBUG",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(config, "_warned", set())
    return tmp_path


def test_default_cache_root(clean_env: Path) -> None:
    assert config.get_cache_root() == clean_env / ".cache" / "topobathykit"


def test_new_cache_name_wins(clean_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOPOBATHYKIT_CACHE_DIR", "/new")
    monkeypatch.setenv("TOPOBATHYSIM_CACHE_DIR", "/old")
    assert config.get_cache_root() == Path("/new")


def test_legacy_env_names_still_read_with_a_warning(clean_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOPOBATHYSIM_CACHE_DIR", "/old")
    monkeypatch.setenv("TOPOBATHYSIM_DEBUG", "1")
    with pytest.warns(
        FutureWarning, match="TOPOBATHYSIM_CACHE_DIR is deprecated; use TOPOBATHYKIT_CACHE_DIR"
    ):
        assert config.get_cache_root() == Path("/old")
    with pytest.warns(FutureWarning, match="TOPOBATHYSIM_DEBUG"):
        assert config.env("TOPOBATHYKIT_DEBUG", "0") == "1"


def test_legacy_cache_directory_used_until_moved(clean_env: Path) -> None:
    legacy = clean_env / ".cache" / "topobathysim"
    legacy.mkdir(parents=True)
    with pytest.warns(FutureWarning, match="move the directory"):
        assert config.get_cache_root() == legacy
    (clean_env / ".cache" / "topobathykit").mkdir()
    assert config.get_cache_root() == clean_env / ".cache" / "topobathykit"
