#!/usr/bin/env python3
"""
N&B Machine — one-command analysis for fluorescence N&B movies.

  python3 run_nnb_machine.py --session my_session
  python3 run_nnb_machine.py --session my_session --train-qc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nnb_pipeline.ib_window_fit import fit_session_windows
from nnb_pipeline.machine import apply_machine_defaults, run_machine_session
from nnb_pipeline.pipeline import load_config
from nnb_pipeline.qc_model import build_training_table, train_classifier, load_classifier


def main():
    parser = argparse.ArgumentParser(
        description="N&B Machine: movie QC → I–B gating → Excel Gaussian B",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "nnb_pipeline" / "config.yaml"),
    )
    parser.add_argument(
        "--session",
        required=True,
        help="Session key from config.yaml sessions (e.g. my_session)",
    )
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "nnb_analysis_output" / "machine"),
    )
    parser.add_argument(
        "--train-qc",
        action="store_true",
        help="Retrain movie QC classifier before running",
    )
    parser.add_argument(
        "--fit-windows",
        action="store_true",
        help="Fit I–B windows from reference B before running (writes ib_window_cache)",
    )
    parser.add_argument(
        "--no-fitted-windows",
        action="store_true",
        help="Blind I–B window search (ignore fitted cache)",
    )
    parser.add_argument(
        "--qc-model",
        default=None,
        help="Path to qc_classifier.joblib (default: nnb_pipeline/models/qc_classifier.joblib)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_machine_defaults(cfg)
    if args.no_fitted_windows:
        cfg.setdefault("machine", {})["use_fitted_windows"] = False
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    cache_root = Path(cfg.get("machine", {}).get("ib_window_cache_root", "nnb_analysis_output/ib_window_cache"))
    if args.fit_windows:
        print(f"Fitting I–B windows → {cache_root}")
        fit_session_windows(cfg, args.session, conditions=args.conditions, cache_root=cache_root)
        print("Window fit complete.\n")

    model_path = Path(
        args.qc_model
        or cfg.get("machine", {}).get(
            "qc_model_path",
            Path(__file__).parent / "nnb_pipeline" / "models" / "qc_classifier.joblib",
        )
    )

    if args.train_qc or not model_path.exists():
        print("Training movie QC classifier...")
        df_train = build_training_table(cfg)
        meta = train_classifier(df_train, model_path=model_path)
        print(f"  CV accuracy: {meta['cv_accuracy']:.1%}")
        print(f"  Train accuracy: {meta.get('train_accuracy', 0):.1%}")
        th = meta.get("thresholds", {})
        if th:
            print(f"  Thresholds: include≥{th.get('include_threshold', 0):.2f} exclude≤{th.get('exclude_threshold', 0):.2f}")
        print(f"  Saved: {model_path}")

    bundle = load_classifier(model_path)
    if bundle is None:
        print("Warning: no QC model — cells will be marked 'review'")

    print(f"\nN&B Machine — session {args.session}")
    print(f"  Data: {cfg['raw_data_root']}")
    print(f"  Output: {out}")
    print("  Pipeline: trim 10 → movie QC → I–B (Y≤2) → Excel Gauss B\n")

    df_cells, df_summary = run_machine_session(
        cfg,
        args.session,
        conditions=args.conditions,
        output_dir=out,
        qc_bundle=bundle,
    )

    print("Summary:")
    print(df_summary.to_string(index=False))

    cmp_report = out / f"compare_{args.session}_report.json"
    if cmp_report.exists():
        with open(cmp_report) as f:
            rep = json.load(f)
        print(f"\nVs reference workbook:")
        print(f"  Inclusion agreement: {rep.get('inclusion_agreement', 0):.1%}")
        if rep.get("included_epsilon_mae") is not None:
            print(f"  Included ε MAE: {rep['included_epsilon_mae']:.4f}")

    run_report = out / f"machine_run_{args.session}.json"
    if run_report.exists():
        with open(run_report) as f:
            mr = json.load(f)
        print(f"\nQC decisions: include={mr.get('n_include')} review={mr.get('n_review')} exclude={mr.get('n_exclude')}")
        print(f"Review queue: {out / f'review_queue_{args.session}.csv'}")

    manual = Path(__file__).parent / "NnB_Machine_Manual.md"
    print(f"\nDone. User manual: {manual}")


if __name__ == "__main__":
    main()