"""Environment settings, read under the topobathykit names.

topobathykit was topobathysim until 2026-10-05. For one release the old environment variable names
(TOPOBATHYSIM_*) and the old default cache directory (~/.cache/topobathysim) still work, each with
a FutureWarning (shown by default, as it is meant for whoever runs the code) naming the
replacement. The next release drops them.
"""

import logging
import os
import warnings
from pathlib import Path

logger = logging.getLogger("topobathykit")

DEFAULT_CACHE_DIR = Path("~/.cache/topobathykit")
LEGACY_CACHE_DIR = Path("~/.cache/topobathysim")
LEGACY_PREFIX = ("TOPOBATHYKIT_", "TOPOBATHYSIM_")

_warned: set[str] = set()


def _deprecated(old: str, new: str) -> None:
    if old in _warned:
        return
    _warned.add(old)
    message = f"{old} is deprecated; use {new} (topobathykit was topobathysim)."
    logger.warning(message)
    warnings.warn(message, FutureWarning, stacklevel=3)


def env(name: str, default: str | None = None) -> str | None:
    """The value of `name` (a TOPOBATHYKIT_* variable), else of its TOPOBATHYSIM_* form with a
    warning, else `default`."""
    value = os.environ.get(name)
    if value is not None:
        return value
    new_prefix, old_prefix = LEGACY_PREFIX
    if name.startswith(new_prefix):
        old = old_prefix + name[len(new_prefix) :]
        value = os.environ.get(old)
        if value is not None:
            _deprecated(old, name)
            return value
    return default


def get_cache_root() -> Path:
    """The cache root: TOPOBATHYKIT_CACHE_DIR, default ~/.cache/topobathykit.

    While the default directory does not exist and ~/.cache/topobathysim does, the old directory
    is used, with a warning to move it.
    """
    configured = env("TOPOBATHYKIT_CACHE_DIR")
    if configured is not None:
        return Path(configured).expanduser()
    root = DEFAULT_CACHE_DIR.expanduser()
    legacy = LEGACY_CACHE_DIR.expanduser()
    if not root.exists() and legacy.exists():
        _deprecated(str(legacy), f"{root} (move the directory)")
        return legacy
    return root


CACHE_ROOT = get_cache_root()
