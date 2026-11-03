"""Import-time bootstrap for the QGIS CPython 3.12 interpreter.

Must be imported *before* pandas/arviz by any script run in this environment
(``import src.runtime_shim`` immediately after putting the repository root on
``sys.path``).  It performs three environment-specific fixes:

* blocks ``pyarrow``: the QGIS build segfaults the interpreter when pandas
  probes it;
* appends the project virtualenv's pure-Python site-packages (ArviZ, PyMC,
  xarray) after the QGIS site-packages, so QGIS's binary wheels keep priority;
* redirects ``platformdirs`` caches into the workspace, because ``%LOCALAPPDATA%``
  is not writable here.

It is deliberately *not* a ``sitecustomize`` module: loading the same shims via
``sitecustomize``/``PYTHONPATH`` breaks the QGIS Pillow extension import.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR_SITE = ROOT / ".venv" / "Lib" / "site-packages"
CACHE_ROOT = ROOT / ".runtime_cache"

for directory in (CACHE_ROOT, CACHE_ROOT / "matplotlib", CACHE_ROOT / "arviz"):
    directory.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))

# GDAL/PROJ data shipped with the QGIS installation.  Without these, pyproj
# raises DataDirError when writing the GeoPackage products.
_QGIS_ROOT = Path(sys.executable).resolve().parents[2]
for _variable, _relative in (
    ("PROJ_DATA", Path("share") / "proj"),
    ("PROJ_LIB", Path("share") / "proj"),
    ("GDAL_DATA", Path("apps") / "gdal" / "share" / "gdal"),
):
    _candidate = _QGIS_ROOT / _relative
    if _candidate.is_dir():
        os.environ.setdefault(_variable, str(_candidate))

if VENDOR_SITE.is_dir() and str(VENDOR_SITE) not in sys.path:
    sys.path.append(str(VENDOR_SITE))


class _BlockPyArrow:
    def find_spec(self, name, path=None, target=None):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ImportError(
                "pyarrow is disabled by src/runtime_shim.py: the QGIS build "
                "crashes this interpreter on import."
            )
        return None


if not any(getattr(finder, "__class__", None).__name__ == "_BlockPyArrow" for finder in sys.meta_path):
    sys.meta_path.insert(0, _BlockPyArrow())

try:
    import platformdirs

    platformdirs.user_cache_dir = lambda *args, **kwargs: str(CACHE_ROOT / "arviz")
    platformdirs.user_data_dir = lambda *args, **kwargs: str(CACHE_ROOT / "arviz")
except Exception:  # platformdirs is optional at import time
    pass
