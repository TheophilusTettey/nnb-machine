"""
N&B Machine — automated movie QC → I–B gating → Excel Gaussian B.

Designed for: collect LSM data → run one command → get B/ε per cell.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .excel_gauss import brightness_from_mask
from .ib_gating import (
    build_machine_ib_masks,
    compute_pixel_maps,
    y_gate_mask,
)
from .ib_window_fit import load_fitted_windows, masks_from_fitted_windows
from .nb_core import apply_control_normalization, epsilon_from_roi
from .manual_reader import manual_file_index, read_manual_block
from .movie_qc_features import extract_movie_qc_features
from .pipeline import compare_to_manual, load_raw_stack, load_config
from .preprocess import preprocess_stack
from .qc_model import load_classifier, predict_qc


def _fallback_brightness(stack, nuc_mask, arr_mask, y_mask, nuc_res, arr_res, cfg):
    """If Excel Gauss fails, use gaussian_hist (still no manual calibration)."""
    b_max = cfg.get("ib_gating", {}).get("brightness_max", 2.0)
    kw = dict(min_mean=cfg["pixel_min_mean"], b_min=cfg["brightness_min"], b_max=b_max, method="gaussian_hist")
    if nuc_res.get("epsilon") == -1.0 and nuc_mask is not None:
        nuc_res = epsilon_from_roi(stack, nuc_mask & y_mask, **kw)
        nuc_res["method"] = "gaussian_hist_fallback"
    if arr_res.get("epsilon") == -1.0 and arr_mask is not None:
        arr_res = epsilon_from_roi(stack, arr_mask & y_mask, **kw)
        arr_res["method"] = "gaussian_hist_fallback"
    return nuc_res, arr_res


def apply_machine_defaults(cfg: dict) -> dict:
    """Production machine settings."""
    m = cfg.setdefault("machine", {})
    cfg = {**cfg}
    cfg["detrend"] = m.get("detrend", False)
    roi_mode = m.get("roi_mode", "ib_scatter")
    cfg["roi_mode"] = "ib_scatter" if roi_mode in ("diego_ib", "ib_scatter") else roi_mode
    cfg["use_manual_inclusion"] = False
    cfg["register_for_qc"] = False
    return cfg


def analyze_cell_machine(
    lsm_path: Path,
    cfg: dict,
    *,
    qc_bundle=None,
    session_key: str | None = None,
    condition_id: str | None = None,
    manual_record: dict | None = None,
) -> dict:
    """Full machine path for one LSM file."""
    mcfg = cfg.get("machine", {})
    raw = load_raw_stack(lsm_path)
    stack, prep_meta = preprocess_stack(
        raw,
        discard_first=cfg["discard_first_frames"],
        detrend=cfg.get("detrend", False),
    )

    movie_feats = extract_movie_qc_features(
        stack,
        nucleus_percentile=cfg["nucleus_percentile_threshold"],
        nucleus_box_half=cfg.get("nucleus_box_half_px", 10),
        array_radius_px=cfg["array_radius_px"],
    )

    qc_pred = predict_qc(
        movie_feats,
        qc_bundle,
        review_threshold=mcfg.get("review_threshold", 0.65),
    )

    cell_index = manual_file_index(lsm_path.name)
    fitted = None
    cache_root = mcfg.get("ib_window_cache_root")
    if (
        mcfg.get("use_fitted_windows", True)
        and cache_root
        and session_key
        and condition_id
        and cell_index is not None
    ):
        fitted = load_fitted_windows(Path(cache_root), session_key, condition_id, cell_index)

    array_only = bool(fitted and fitted.get("nucleus_excluded") and fitted.get("array"))
    skip_nb = mcfg.get("skip_nb_if_excluded", True) and qc_pred["decision"] == "exclude" and not array_only

    empty_nuc = {"brightness": np.nan, "epsilon": -1.0, "n_pixels": 0, "method": "excel_gauss"}
    empty_arr = dict(empty_nuc)

    ib_meta: dict = {}
    nuc_res = empty_nuc
    arr_res = empty_arr

    if not skip_nb:
        ib_cfg = cfg.get("ib_gating", {})
        b_max = ib_cfg.get("brightness_max", 2.0)
        mean_i, brightness, _ = compute_pixel_maps(stack)
        y_mask = y_gate_mask(
            brightness,
            mean_i,
            b_max=b_max,
            b_min=cfg["brightness_min"],
            min_mean=cfg["pixel_min_mean"],
        )

        use_fitted = (
            fitted
            and fitted.get("fitted")
            and (
                (fitted.get("nucleus") or {}).get("b_error", 1.0) < 0.05
                or (fitted.get("array") or {}).get("b_error", 1.0) < 0.05
            )
        )
        if use_fitted:
            nuc_mask, arr_mask = masks_from_fitted_windows(mean_i, y_mask, fitted)
            ib_meta = {
                "source": "fitted_windows",
                "nucleus_window": fitted.get("nucleus"),
                "array_window": fitted.get("array"),
            }
            if fitted.get("array_fit_failed") and manual_record and manual_record.get("array", {}).get("included"):
                arr_m = manual_record["array"]
                if arr_m.get("intensity_min") is not None and arr_m.get("intensity_max") is not None:
                    arr = y_mask & (mean_i >= float(arr_m["intensity_min"])) & (mean_i <= float(arr_m["intensity_max"]))
                    if int(arr.sum()) >= ib_cfg.get("min_array_pixels", 8):
                        arr_mask = arr
                        ib_meta["array_window"] = {
                            "intensity_min": float(arr_m["intensity_min"]),
                            "intensity_max": float(arr_m["intensity_max"]),
                            "source": "manual_intensity_fallback",
                        }
                        ib_meta["source"] = "fitted_windows_hybrid"
        else:
            nuc_mask, arr_mask, ib_meta = build_machine_ib_masks(
                stack,
                ib_brightness_max=b_max,
                pixel_min_mean=cfg["pixel_min_mean"],
                brightness_min=cfg["brightness_min"],
                nucleus_i_lo=tuple(ib_cfg.get("nucleus_i_lo_range", [6.0, 18.0])),
                nucleus_i_hi=tuple(ib_cfg.get("nucleus_i_hi_range", [14.0, 32.0])),
                array_i_lo=tuple(ib_cfg.get("array_i_lo_range", [12.0, 35.0])),
                array_i_hi=tuple(ib_cfg.get("array_i_hi_range", [18.0, 55.0])),
                search_step=ib_cfg.get("search_step", 0.5),
                min_nucleus_pixels=ib_cfg.get("min_nucleus_pixels", 40),
                min_array_pixels=ib_cfg.get("min_array_pixels", 8),
                umbral_start=cfg.get("umbral_1_start", 0.62601),
                umbral_step=cfg.get("umbral_x_step", 0.02282),
                array_umbral_start=mcfg.get("array_umbral_start"),
                umbral_n_bins=mcfg.get("umbral_n_bins", 45),
            )
        gauss_kw = dict(
            umbral_start=cfg.get("umbral_1_start", 0.62601),
            umbral_step=cfg.get("umbral_x_step", 0.02282),
            n_bins=mcfg.get("umbral_n_bins", 45),
            min_mean=cfg["pixel_min_mean"],
            b_min=cfg["brightness_min"],
            b_max=b_max,
        )
        nuc_res = brightness_from_mask(stack, nuc_mask, y_mask, **gauss_kw)
        arr_gauss_kw = {**gauss_kw, "umbral_start": mcfg.get("array_umbral_start") or gauss_kw["umbral_start"]}
        arr_res = brightness_from_mask(stack, arr_mask, y_mask, **arr_gauss_kw)
        nuc_res, arr_res = _fallback_brightness(stack, nuc_mask, arr_mask, y_mask, nuc_res, arr_res, cfg)
        if ib_meta.get("motion_ring_fraction") is not None:
            ib_meta["motion_ring_fraction"] = ib_meta.get("motion_ring_fraction")
        roi_source = "fitted_ib_excel_gauss" if fitted and fitted.get("fitted") else "machine_ib_excel_gauss"
    else:
        nuc_mask = arr_mask = None
        roi_source = "skipped_qc_exclude"

    # Pass/fail
    if qc_pred["decision"] == "exclude":
        qc_passed = False
        qc_reasons = [qc_pred["reason"]]
    elif qc_pred["decision"] == "review":
        qc_passed = nuc_res["epsilon"] != -1.0 and arr_res["epsilon"] != -1.0
        qc_reasons = ["needs_review"] if not qc_passed else ["review_but_computed"]
    else:
        qc_passed = nuc_res["epsilon"] != -1.0 and arr_res["epsilon"] != -1.0
        qc_reasons = [] if qc_passed else ["invalid_nucleus_or_array"]

    nucleus_qc_passed = qc_pred["decision"] != "exclude" and nuc_res["epsilon"] != -1.0
    if array_only:
        array_qc_passed = arr_res["epsilon"] != -1.0
    else:
        array_qc_passed = qc_pred["decision"] != "exclude" and arr_res["epsilon"] != -1.0

    manual_obs = manual_included = None
    if manual_record is not None:
        manual_obs = manual_record["nucleus"].get("obs")
        manual_included = manual_record["nucleus"]["included"]

    mean_roi = 0.0
    if not skip_nb and nuc_mask is not None and nuc_mask.any():
        mean_roi = float(stack.mean(axis=0)[nuc_mask].mean())

    return {
        "file": lsm_path.name,
        "cell_index": cell_index,
        "mean_roi_intensity": mean_roi,
        "nucleus": nuc_res,
        "array": arr_res,
        "nucleus_pixels": int(nuc_mask.sum()) if nuc_mask is not None else 0,
        "array_pixels": int(arr_mask.sum()) if arr_mask is not None else 0,
        "roi_source": roi_source,
        "roi_mode": "machine",
        "qc_passed": qc_passed,
        "nucleus_qc_passed": nucleus_qc_passed,
        "array_qc_passed": array_qc_passed,
        "qc_reasons": qc_reasons,
        "qc_decision": qc_pred["decision"],
        "qc_probability_included": qc_pred["probability_included"],
        "qc_suggested_obs": qc_pred.get("suggested_obs") or "",
        "qc_metrics": movie_feats,
        "ib_meta": ib_meta,
        "prep_meta": prep_meta,
        "manual_obs": manual_obs,
        "manual_included": manual_included,
        "session": session_key,
        "condition": condition_id,
    }


def run_machine_session(
    cfg: dict,
    session_key: str,
    *,
    conditions: list[str] | None = None,
    output_dir: Path,
    qc_bundle=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = apply_machine_defaults(cfg)
    session = cfg["sessions"][session_key]
    session_dir = Path(cfg["raw_data_root"]) / session["folder"]
    rows = []

    for cond in session["conditions"]:
        if conditions and cond["id"] not in conditions:
            continue
        manual_by_id = {}
        if session_key in cfg.get("manual_workbooks", {}):
            manual_by_id = {
                m["cell_id"]: m
                for m in read_manual_block(cfg["manual_workbooks"][session_key], cond["manual_sheet_block"])
            }
        cond_dir = session_dir / cond["subdir"]
        for f in sorted(cond_dir.glob(cond["file_glob"])):
            cid = manual_file_index(f.name)
            manual_rec = manual_by_id.get(cid) if cid else None
            rec = analyze_cell_machine(
                f,
                cfg,
                qc_bundle=qc_bundle,
                session_key=session_key,
                condition_id=cond["id"],
                manual_record=manual_rec,
            )
            rows.append(rec)

    df_cells = pd.json_normalize(rows, sep="_")
    output_dir.mkdir(parents=True, exist_ok=True)
    df_cells.to_csv(output_dir / f"cells_{session_key}.csv", index=False)

    # Review queue
    review = df_cells[df_cells["qc_decision"].isin(["review", "exclude"])].copy()
    if len(review):
        cols = [
            "cell_index", "condition", "file", "qc_decision",
            "qc_probability_included", "qc_suggested_obs", "qc_reasons",
        ]
        cols = [c for c in cols if c in review.columns]
        review[cols].to_csv(output_dir / f"review_queue_{session_key}.csv", index=False)

    summary_rows = []
    for cond in session["conditions"]:
        if conditions and cond["id"] not in conditions:
            continue
        sub = df_cells[df_cells["condition"] == cond["id"]]
        included = sub[sub["nucleus_qc_passed"] & (sub["nucleus_epsilon"] != -1)]
        nuc_mean = included["nucleus_epsilon"].mean() if len(included) else np.nan
        arr_sub = sub[sub["array_qc_passed"] & (sub["array_epsilon"] != -1)]
        arr_mean = arr_sub["array_epsilon"].mean() if len(arr_sub) else np.nan
        summary_rows.append({
            "session": session_key,
            "condition": cond["id"],
            "n_cells_total": len(sub),
            "n_cells_included": len(included),
            "n_review_or_exclude": int((sub["qc_decision"] != "include").sum()),
            "mean_epsilon_nucleus": nuc_mean,
            "mean_epsilon_array": arr_mean,
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary = apply_control_normalization(
        df_summary, session.get("control_condition")
    )

    df_summary.to_csv(output_dir / f"summary_{session_key}.csv", index=False)

    # Validation vs manual if workbooks configured
    if session_key in cfg.get("manual_workbooks", {}):
        compare_to_manual(cfg, session_key, df_cells, output_dir)

    report = {
        "session": session_key,
        "machine": True,
        "n_cells": int(len(df_cells)),
        "n_include": int((df_cells["qc_decision"] == "include").sum()),
        "n_review": int((df_cells["qc_decision"] == "review").sum()),
        "n_exclude": int((df_cells["qc_decision"] == "exclude").sum()),
    }
    with open(output_dir / f"machine_run_{session_key}.json", "w") as f:
        json.dump(report, f, indent=2)

    return df_cells, df_summary