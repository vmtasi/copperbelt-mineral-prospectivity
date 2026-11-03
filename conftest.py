"""Pytest bootstrap: install the QGIS runtime shim before pandas is imported."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.runtime_shim  # noqa: F401,E402
