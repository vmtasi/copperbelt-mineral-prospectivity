# Run the repository test suite with the QGIS Python runtime.
#
# pytest is installed in the project's legacy virtualenv, but that virtualenv's
# native scikit-learn build targets CPython 3.10. Appending its pure-Python
# site-packages after the QGIS site-packages makes pytest available while the
# runtime shim and src.numerics provide compatible numerical fallbacks.

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.runtime_shim  # noqa: F401,E402

VENDOR_SITE = ROOT / ".venv" / "Lib" / "site-packages"
if VENDOR_SITE.is_dir() and str(VENDOR_SITE) not in sys.path:
    sys.path.append(str(VENDOR_SITE))

import pytest  # noqa: E402


if __name__ == "__main__":
    arguments = sys.argv[1:] or ["-q", "tests"]
    raise SystemExit(pytest.main(arguments))

