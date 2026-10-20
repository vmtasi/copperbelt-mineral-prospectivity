"""Trace persistence with a NetCDF-first, pickle fallback.

ArviZ ``InferenceData.to_netcdf`` writes HDF5-backed groups. Some minimal
QGIS/Python runtimes provide ArviZ and PyMC but not an HDF5 backend. This
adapter keeps the canonical ``.nc`` path when that backend exists and writes a
sidecar ``.pkl`` only when it does not. Readers accept either form.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import arviz as az


def _fallback_path(path: Path) -> Path:
    return path.with_suffix(".pkl")


def save_trace(trace, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        trace.to_netcdf(path)
        return path
    except Exception as error:  # missing HDF5/netCDF4 backend
        print(f"[trace_io] NetCDF write unavailable ({error}); using pickle fallback.")
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        fallback = _fallback_path(path)
        with fallback.open("wb") as handle:
            pickle.dump(trace, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return fallback


def resolve_trace_path(path: Path) -> Path:
    path = Path(path)
    if path.exists():
        return path
    fallback = _fallback_path(path)
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No readable trace at {path} or {fallback}.")


def trace_exists(path: Path) -> bool:
    path = Path(path)
    return path.exists() or _fallback_path(path).exists()


def load_trace(path: Path):
    path = Path(path)
    if path.suffix.lower() != ".pkl" and path.exists():
        try:
            return az.from_netcdf(path)
        except Exception as error:
            print(f"[trace_io] Could not read {path} as NetCDF ({error}).")
    fallback = path if path.suffix.lower() == ".pkl" else _fallback_path(path)
    if fallback.exists():
        with fallback.open("rb") as handle:
            return pickle.load(handle)
    raise FileNotFoundError(
        f"No readable trace at {path} or {fallback}. Refit the affected model."
    )
