"""Preflight check that the V14 wrapper can build the frozen V11 runner."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.v14_centroid_spatial_stability as runner


function = runner.build_runner()
print("BUILT", function.__name__)
import pymc
import arviz

print("PYMC", pymc.__version__, "ARVIZ", arviz.__version__)
