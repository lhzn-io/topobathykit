"""Compatibility alias: topobathysim was renamed topobathykit on 2026-10-05.

`import topobathysim` and `import topobathysim.<module>` still work for one release and give the
topobathykit modules themselves, so objects are shared, not copied. Import topobathykit instead;
the next release removes this package.
"""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import warnings
from collections.abc import Sequence
from types import ModuleType

warnings.warn(
    "topobathysim was renamed topobathykit on 2026-10-05; import topobathykit instead. "
    "The topobathysim alias is removed in the next release.",
    DeprecationWarning,
    stacklevel=2,
)

import topobathykit as _topobathykit  # noqa: E402
from topobathykit import *  # noqa: E402, F403

__version__ = getattr(_topobathykit, "__version__", None)

_OLD, _NEW = "topobathysim.", "topobathykit."


class _AliasLoader(importlib.abc.Loader):
    def create_module(self, spec: importlib.machinery.ModuleSpec) -> None:
        return None  # a throwaway module; exec_module swaps in the real one

    def exec_module(self, module: ModuleType) -> None:
        # importlib returns whatever sys.modules holds under the name after exec_module.
        sys.modules[module.__name__] = importlib.import_module(_NEW + module.__name__[len(_OLD) :])


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self, fullname: str, path: Sequence[str] | None = None, target: ModuleType | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        if not fullname.startswith(_OLD):
            return None
        if importlib.util.find_spec(_NEW + fullname[len(_OLD) :]) is None:
            return None
        return importlib.machinery.ModuleSpec(fullname, _AliasLoader())


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())
