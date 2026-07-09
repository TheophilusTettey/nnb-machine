#!/usr/bin/env python3
"""Train movie QC classifier from expert inclusion labels in reference workbooks."""

from __future__ import annotations

import argparse
from pathlib import Path

from nnb_pipeline.pipeline import load_config
from nnb_pipeline.qc_model import build_training_table, train_classifier


def main():
    parser = argparse.ArgumentParser(description="Train N&B Machine movie QC model")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "nnb_pipeline" / "config.yaml"),
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "nnb_pipeline" / "models" / "qc_classifier.joblib"),
    )
    parser.add_argument(
        "--sessions",
        nargs="*",
        default=None,
        help="Session keys from config.yaml (e.g. my_session)",
    )
    parser.add_argument(
        "--from-csv",
        default=None,
        help="Skip LSM read; train from saved qc_classifier.training_data.csv",
    )
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        import pandas as pd
        df = pd.read_csv(args.from_csv)
        print(f"Loaded training data from {args.from_csv} ({len(df)} rows)")
    else:
        cfg = load_config(args.config)
        print("Building training features from LSM movies...")
        print("(External drive with LSM files must be connected.)")
        df = build_training_table(cfg, sessions=args.sessions)
        df.to_csv(out.parent / f"{out.stem}.training_data.csv", index=False)
    inc_col = "reference_included"
    if inc_col not in df.columns and "diego_included" in df.columns:
        inc_col = "diego_included"  # legacy training CSV
    print(f"Training samples: {len(df)} (included={df[inc_col].sum()}, excluded={(~df[inc_col]).sum()})")

    meta = train_classifier(df, model_path=out)
    print(f"CV accuracy: {meta['cv_accuracy']:.1%}")
    print(f"Train accuracy: {meta.get('train_accuracy', 0):.1%}")
    th = meta.get("thresholds", {})
    if th:
        print(f"Thresholds: include≥{th.get('include_threshold', 0):.2f} exclude≤{th.get('exclude_threshold', 0):.2f}")
    print(f"Model saved: {out}")
    print(f"Metadata: {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()