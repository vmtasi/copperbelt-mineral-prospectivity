"""Moran's I for V15 out-of-fold residuals using k-nearest spatial weights."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from src.v15 import config  # noqa: E402
from src.v15.common import ensure_v15_path  # noqa: E402


def moran(residual: np.ndarray, weights: np.ndarray, permutations: int, seed: int) -> tuple[float, float]:
    n = len(residual)
    s0 = weights.sum()
    observed = float((n / s0) * (residual @ weights @ residual) / (residual @ residual))
    rng = np.random.default_rng(seed)
    null = np.empty(permutations)
    for index in range(permutations):
        shuffled = rng.permutation(residual)
        null[index] = (n / s0) * (shuffled @ weights @ shuffled) / (shuffled @ shuffled)
    p_value = float((1 + np.sum(null >= observed)) / (1 + permutations))
    return observed, p_value


def weights_matrix(coords: np.ndarray, k: int = 8) -> np.ndarray:
    tree = cKDTree(coords)
    distances, indices = tree.query(coords, k=k + 1)
    weights = np.zeros((len(coords), len(coords)), dtype=float)
    rows = np.repeat(np.arange(len(coords)), k)
    weights[rows, indices[:, 1:].ravel()] = 1.0
    weights = weights / weights.sum(axis=1, keepdims=True)
    return weights


def main() -> None:
    rows = []
    for path in sorted((config.V15 / "predictions").glob("oof_*.csv")):
        model = path.stem.removeprefix("oof_")
        frame = pd.read_csv(path)
        if not {"centroid_x", "centroid_y", "y", "prob"}.issubset(frame.columns):
            continue
        coords = frame[["centroid_x", "centroid_y"]].to_numpy(float)
        residual = frame["y"].to_numpy(float) - frame["prob"].to_numpy(float)
        weights = weights_matrix(coords)
        value, p_value = moran(residual, weights, 999, config.BOOTSTRAP_SEED)
        rows.append(
            {
                "model": model,
                "n": int(len(frame)),
                "k_neighbors": 8,
                "weights": "row-standardised k-nearest neighbours",
                "morans_i": value,
                "permutations": 999,
                "p_value_one_sided_positive": p_value,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(ensure_v15_path(config.V15 / "diagnostics" / "moran_i.csv"), index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
