#!/usr/bin/env python3
"""Fit per-cell I–B intensity windows to reference manual B (calibration step)."""

from __future__ import annotations

import argparse
from pathlib import Path

from nnb_pipeline.ib_window_fit import fit_session_windows
from nnb_pipeline.machine import apply_machine_defaults
from nnb_pipeline.pipeline import load_config


def main():
    parser = argparse.ArgumentParser(description="Fit I–B windows vs reference workbook B values")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "nnb_pipeline" / "config.yaml"),
    )
    parser.add_argument(
        "--session",
        nargs="+",
        default=["my_session"],
        help="Session key(s) from config.yaml",
    )
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument(
        "--cache",
        default=None,
        help="Output cache root (default: machine.ib_window_cache_root in config)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_machine_defaults(cfg)
    cache_root = Path(
        args.cache
        or cfg.get("machine", {}).get("ib_window_cache_root", "nnb_analysis_output/ib_window_cache")
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    print(f"I–B window fit → {cache_root}")
    print(f"Data: {cfg['raw_data_root']}\n")

    all_rows = []
    for session in args.session:
        print(f"Fitting {session}...")
        df = fit_session_windows(cfg, session, conditions=args.conditions, cache_root=cache_root)
        df["session"] = session
        all_rows.append(df)
        n_fit = int(df["fitted"].sum()) if "fitted" in df else 0
        nuc_mae = df["nuc_b_err"].mean() if "nuc_b_err" in df else None
        arr_mae = df["arr_b_err"].mean() if "arr_b_err" in df else None
        print(f"  cells fitted: {n_fit}/{len(df)}")
        if nuc_mae is not None:
            print(f"  nucleus B fit MAE: {nuc_mae:.4f}")
        if arr_mae is not None:
            print(f"  array B fit MAE: {arr_mae:.4f}")

    if all_rows:
        import pandas as pd

        summary = pd.concat(all_rows, ignore_index=True)
        out_csv = cache_root / "fit_summary.csv"
        summary.to_csv(out_csv, index=False)
        print(f"\nSummary: {out_csv}")


if __name__ == "__main__":
    main()