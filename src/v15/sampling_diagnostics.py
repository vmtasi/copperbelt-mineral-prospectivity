"""Summarise convergence diagnostics for every saved V15 trace."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import arviz as az  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.trace_io import load_trace, resolve_trace_path  # noqa: E402
from src.v15 import config  # noqa: E402
from src.v15.common import ensure_v15_path, write_json  # noqa: E402

RAW_OFFSETS = {
    "alpha_offset",
    "offset_f_lin",
    "offset_f_sq",
    "offset_l_lin",
    "offset_l_sq",
    "offset_beta_grav",
}


def main() -> None:
    rows = []
    parameter_tables = []
    trace_root = config.V15 / "traces"
    traces = sorted(
        path
        for pattern in ("fold_*.nc", "fold_*.pkl")
        for path in trace_root.rglob(pattern)
        if not any("pilot" in part.lower() for part in path.relative_to(trace_root).parts)
    )
    for requested in traces:
        path = resolve_trace_path(requested)
        trace = load_trace(path)
        variables = [name for name in trace.posterior.data_vars if name not in RAW_OFFSETS]
        summary = az.summary(trace, var_names=variables, round_to=8).reset_index(names="parameter")
        summary.insert(0, "fold", int(path.stem.split("_")[-1]))
        summary.insert(0, "model", str(path.parent.relative_to(trace_root)))
        parameter_tables.append(summary)
        diverging = int(np.asarray(trace.sample_stats["diverging"]).sum())
        try:
            bfmi = np.asarray(az.bfmi(trace), dtype=float)
        except Exception:
            bfmi = np.array([np.nan])
        rows.append(
            {
                "model": str(path.parent.relative_to(trace_root)),
                "fold": int(path.stem.split("_")[-1]),
                "trace": str(path.relative_to(config.V15)),
                "chains": int(trace.posterior.sizes["chain"]),
                "draws_per_chain": int(trace.posterior.sizes["draw"]),
                "divergences": diverging,
                "max_rhat": float(summary["r_hat"].max()),
                "min_ess_bulk": float(summary["ess_bulk"].min()),
                "min_ess_tail": float(summary["ess_tail"].min()),
                "max_mcse_mean": float(summary["mcse_mean"].max()),
                "max_mcse_sd": float(summary["mcse_sd"].max()),
                "min_bfmi": float(np.nanmin(bfmi)),
                "max_bfmi": float(np.nanmax(bfmi)),
                "converged": bool(
                    diverging == 0
                    and float(summary["r_hat"].max()) < 1.05
                    and float(summary["ess_bulk"].min()) >= 200
                    and float(summary["ess_tail"].min()) >= 200
                ),
            }
        )
    table = pd.DataFrame(rows).sort_values(["model", "fold"])
    table.to_csv(ensure_v15_path(config.V15 / "diagnostics" / "sampling_diagnostics.csv"), index=False)
    write_json(
        config.V15 / "models" / "fit_status_recomputed.json",
        table[["model", "fold", "divergences", "max_rhat", "min_ess_bulk", "min_ess_tail", "converged"]]
        .rename(columns={"divergences": "diverging"})
        .to_dict("records"),
    )
    pd.concat(parameter_tables, ignore_index=True).to_csv(
        ensure_v15_path(config.V15 / "diagnostics" / "sampling_parameter_summary.csv"), index=False
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
