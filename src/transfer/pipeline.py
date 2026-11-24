"""Command-line orchestration for the Paper 1 transfer study."""

from __future__ import annotations

import argparse
import hashlib

import pandas as pd

from src.transfer import config
from src.transfer.analysis import (
    analyse_a1,
    analyse_a2,
    analyse_a3,
    analyse_a4,
    analyse_a5,
    analyse_a6,
    analyse_a7,
    analyse_a8,
    make_figures,
    map_gate_passed,
    validate_map_mcmc,
    write_partition_outputs,
)
from src.transfer.analysis_v2 import (
    _all_new_fits_converged,
    _load_amendment_predictions,
    a2_mcmc_matrix,
    a3_mcmc_decomposition,
    a5_mcmc,
    a6_mcmc,
    a8_mcmc,
    sensitivity_metrics,
)
from src.transfer.data import ensure_output_path, load_transfer_frame
from src.transfer.reporting import write_blocked_report, write_report
from src.transfer.reporting_v2 import write_report_v2
from src.transfer.specification import write_specification, write_specification_v2
from src.transfer.training import (
    run_all_fits,
    run_map_validation,
    run_mcmc,
)
from src.transfer.training_mcmc import aggregate_status, run_batch


def _checksums() -> str:
    root = config.OUTPUT
    path = ensure_output_path(root / "checksums" / "SHA256SUMS")
    lines = []
    for item in sorted(root.rglob("*")):
        if not item.is_file() or item == path:
            continue
        digest = hashlib.sha256(item.read_bytes()).hexdigest()
        lines.append(f"{digest}  {item.relative_to(root).as_posix()}")
    # LF line endings so `sha256sum -c` works on every platform.
    path.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    return str(path)


def run_analysis() -> None:
    map_gate_passed()
    frame = load_transfer_frame()
    partitions = pd.read_csv(
        config.OUTPUT / "partitions" / "transfer_partitions.csv"
    )
    a1 = analyse_a1(frame)
    _, a2_wide = analyse_a2(frame)
    decomposition, _ = analyse_a3(frame)
    _, aoa_summary = analyse_a4(frame)
    coefficients, marginal, verdicts = analyse_a5(frame)
    a6 = analyse_a6(frame, a1)
    analyse_a7()
    analyse_a8(frame, partitions, a1, verdicts, a6)
    make_figures(
        frame,
        a2_wide,
        decomposition,
        pd.read_csv(config.OUTPUT / "metrics" / "a4_aoa_cells.csv"),
        coefficients,
        marginal,
    )
    print(str(config.OUTPUT / "metrics"))


def write_amendment_coefficients() -> pd.DataFrame:
    """Concatenate per-fit coefficient files, labelling every row by its fit.

    The A3 oracle files carry no ``fit_id`` column, so the file stem (which
    is the fit ID) fills it.
    """
    frames = []
    for path in sorted((config.OUTPUT / "coefficients" / "amendment1").glob("*.csv")):
        table = pd.read_csv(path)
        if "fit_id" not in table.columns:
            table.insert(0, "fit_id", path.stem)
        else:
            table["fit_id"] = table["fit_id"].fillna(path.stem)
        frames.append(table)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(
        ensure_output_path(config.OUTPUT / "coefficients" / "all_amendment1.csv"),
        index=False,
    )
    return combined


def run_mcmc_analysis() -> None:
    frame = load_transfer_frame()
    _all_new_fits_converged()
    aggregate_status()
    partition_predictions = _load_amendment_predictions("", batch=None)
    partition_predictions.to_csv(
        ensure_output_path(config.OUTPUT / "predictions" / "all_amendment1.csv"),
        index=False,
    )
    write_amendment_coefficients()
    a1 = analyse_a1(frame)
    a2, a2_wide = a2_mcmc_matrix(frame)
    decomposition, _ = a3_mcmc_decomposition(frame)
    _, aoa_summary = analyse_a4(frame)
    coefficients, marginal, verdicts = a5_mcmc(frame)
    a6 = a6_mcmc(frame, a1)
    sensitivity_metrics()
    analyse_a7()
    a8_mcmc(frame, a1, a6, verdicts)
    make_figures(
        frame,
        a2_wide,
        decomposition,
        pd.read_csv(config.OUTPUT / "metrics" / "a4_aoa_cells.csv"),
        coefficients,
        marginal,
    )
    print(str(config.OUTPUT / "metrics"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "spec",
            "partitions",
            "fit-mcmc",
            "validate-map",
            "fit-map",
            "analyze",
            "report",
            "checksums",
            "spec-v2",
            "fit-amendment",
            "analyze-mcmc",
            "report-v2",
            "all",
        ],
    )
    parser.add_argument(
        "--batch",
        choices=["a2", "a3", "a6", "a8", "sensitivities"],
    )
    parser.add_argument("--overwrite-mcmc", action="store_true")
    args = parser.parse_args()

    if args.command in {"spec", "all"}:
        write_specification()
    if args.command in {"partitions", "all"}:
        write_partition_outputs(load_transfer_frame())
    if args.command in {"fit-mcmc", "all"}:
        run_mcmc(load_transfer_frame(), overwrite=args.overwrite_mcmc)
    if args.command == "spec-v2":
        print(write_specification_v2())
    if args.command == "fit-amendment":
        if not args.batch:
            raise ValueError("--batch is required with fit-amendment")
        frame = load_transfer_frame()
        results = run_batch(frame, args.batch)
        aggregate_status()
        print(pd.DataFrame(results).to_string(index=False))
    if args.command in {"validate-map", "all"}:
        run_map_validation(load_transfer_frame())
        result = validate_map_mcmc()
        print(pd.Series(result).to_string())
    if args.command in {"fit-map", "all"}:
        map_gate_passed()
        run_all_fits(load_transfer_frame())
    if args.command in {"analyze", "all"}:
        run_analysis()
    if args.command == "analyze-mcmc":
        run_mcmc_analysis()
    if args.command in {"report", "all"}:
        validation_path = config.OUTPUT / "diagnostics" / "map_mcmc_validation.json"
        if validation_path.exists():
            validation = __import__("json").loads(
                validation_path.read_text(encoding="utf-8")
            )
            if not validation["pass"]:
                print(write_blocked_report())
            else:
                print(write_report())
        else:
            raise FileNotFoundError("MAP-MCMC validation has not been run")
    if args.command == "report-v2":
        print(write_report_v2())
    if args.command in {"checksums", "all"}:
        print(_checksums())


if __name__ == "__main__":
    main()

