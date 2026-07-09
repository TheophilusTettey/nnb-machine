"""End-to-end computational N&B pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import yaml

from .movie_exclusion import automated_movie_exclusion
from .ib_gating import build_ib_scatter_masks, epsilon_from_ib_mask
from .manual_reader import manual_file_index, read_manual_block
from .nb_core import apply_control_normalization, epsilon_from_roi, normalize_epsilon
from .preprocess import preprocess_stack
from .qc import automated_qc, manual_obs_excluded
from .register import register_stack_rigid
from .roi import auto_nucleus_mask, auto_array_mask, resolve_roi_masks
from .roi_fit import load_fitted_roi
from .motion_flags import (
    classify_motion_flags,
    compute_motion_metrics,
    match_manual_obs_category,
    suggested_qc_note,
)


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_raw_stack(lsm_path: Path) -> np.ndarray:
    stack = tifffile.imread(lsm_path).astype(np.float64)
    if stack.ndim != 3:
        raise ValueError(f"Expected TYX stack, got {stack.shape} for {lsm_path}")
    return stack


def analyze_cell(
    lsm_path: Path,
    cfg: dict,
    *,
    epsilon_method: str = "weighted_mean",
    roi_dir: Path | None = None,
    roi_cache: Path | None = None,
    session_key: str | None = None,
    condition_id: str | None = None,
    manual_record: dict | None = None,
) -> dict:
    raw = load_raw_stack(lsm_path)
    stack_nb, prep_meta = preprocess_stack(
        raw,
        discard_first=cfg["discard_first_frames"],
        detrend=cfg.get("detrend", True),
    )

    # Registration before N&B destroys per-pixel variance (ε becomes negative).
    # SimFCS detrend → N&B on the same stack; registration is visual/QC only.
    stack_qc = stack_nb
    reg_meta = {}
    if cfg.get("register_for_qc", False):
        stack_qc, reg_meta = register_stack_rigid(
            stack_nb,
            reference_frame=cfg.get("reference_frame", 0),
            upsample_factor=cfg.get("registration_upsample", 10),
        )

    mean_img = stack_nb.mean(axis=0)
    cell_index = manual_file_index(lsm_path.name)
    roi_mode = cfg.get("roi_mode", "geometric")
    if roi_mode in ("diego_ib", "ib_scatter"):
        roi_mode = "ib_scatter"
    ib_meta: dict = {}
    fitted_roi = None

    if roi_mode == "ib_scatter":
        ib_cfg = cfg.get("ib_gating", {})
        nuc_mask, arr_mask, ib_meta = build_ib_scatter_masks(
            stack_nb,
            ib_brightness_max=ib_cfg.get("brightness_max", 2.0),
            pixel_min_mean=cfg["pixel_min_mean"],
            brightness_min=cfg["brightness_min"],
            nucleus_i_lo=tuple(ib_cfg.get("nucleus_i_lo_range", [6.0, 18.0])),
            nucleus_i_hi=tuple(ib_cfg.get("nucleus_i_hi_range", [14.0, 32.0])),
            array_i_lo=tuple(ib_cfg.get("array_i_lo_range", [15.0, 35.0])),
            array_i_hi=tuple(ib_cfg.get("array_i_hi_range", [22.0, 55.0])),
            search_step=ib_cfg.get("search_step", 0.5),
            min_nucleus_pixels=ib_cfg.get("min_nucleus_pixels", 40),
            min_array_pixels=ib_cfg.get("min_array_pixels", 15),
        )
        roi_source = "ib_scatter"
        b_max = ib_cfg.get("brightness_max", 2.0)
        nuc_res = epsilon_from_ib_mask(
            stack_nb,
            nuc_mask,
            method=epsilon_method,
            min_mean=cfg["pixel_min_mean"],
            b_min=cfg["brightness_min"],
            b_max=b_max,
        )
        arr_res = epsilon_from_ib_mask(
            stack_nb,
            arr_mask,
            method=epsilon_method,
            min_mean=cfg["pixel_min_mean"],
            b_min=cfg["brightness_min"],
            b_max=b_max,
        )
    else:
        if roi_cache and session_key and condition_id and cell_index is not None:
            fitted_roi = load_fitted_roi(roi_cache, session_key, condition_id, cell_index)

        nuc_mask, arr_mask, roi_source = resolve_roi_masks(
            mean_img,
            roi_dir=roi_dir,
            fitted_roi=fitted_roi,
            cell_index=cell_index,
            nucleus_percentile=cfg["nucleus_percentile_threshold"],
            nucleus_box_half=cfg.get("nucleus_box_half_px", 10),
            array_radius_px=cfg["array_radius_px"],
        )

        nuc_res = epsilon_from_roi(
            stack_nb,
            nuc_mask,
            method=epsilon_method,
            min_mean=cfg["pixel_min_mean"],
            b_min=cfg["brightness_min"],
            b_max=cfg["brightness_max"],
        )
        arr_res = epsilon_from_roi(
            stack_nb,
            arr_mask,
            method=epsilon_method,
            min_mean=cfg["pixel_min_mean"],
            b_min=cfg["brightness_min"],
            b_max=cfg["brightness_max"],
        )

    mean_roi = float(mean_img[nuc_mask].mean()) if nuc_mask.any() else 0.0

    motion_review = None
    motion_flag_summary = ""
    motion_review_recommended = False
    motion_flag_codes: list[str] = []
    motion_suggested_obs = ""
    motion_nuc = auto_nucleus_mask(
        mean_img,
        percentile=cfg["nucleus_percentile_threshold"],
        box_half_height=cfg.get("nucleus_box_half_px", 10),
        box_half_width=cfg.get("nucleus_box_half_px", 10),
    )
    motion_arr = auto_array_mask(mean_img, motion_nuc, radius_px=cfg["array_radius_px"])
    if cfg.get("motion_flags", {}).get("enabled", True):
        motion_metrics = compute_motion_metrics(stack_nb, motion_nuc, motion_arr, mean_img)
        motion_review = classify_motion_flags(
            motion_metrics,
            thresholds=cfg.get("motion_flags"),
        )
        motion_flag_summary = motion_review.summary()
        motion_review_recommended = motion_review.review_recommended
        motion_flag_codes = motion_review.codes()
        motion_suggested_obs = suggested_qc_note(motion_review.flags)

    nucleus_qc_passed = None
    array_qc_passed = None
    if cfg.get("use_manual_inclusion") and manual_record is not None:
        qc_passed = manual_record["nucleus"]["included"]
        qc_reasons = [] if qc_passed else ["manual_excluded"]
        qc_metrics = {}
        nucleus_qc_passed = manual_record["nucleus"]["included"]
        array_qc_passed = manual_record["array"]["included"]
    elif roi_mode == "ib_scatter":
        sec_ratio = (
            motion_review.metrics.get("array_secondary_ratio") if motion_review else None
        )
        dq = automated_movie_exclusion(
            motion_review=motion_review,
            epsilon_nuc=nuc_res["epsilon"],
            epsilon_array=arr_res["epsilon"],
            ib_meta=ib_meta,
            qc_cfg=cfg.get("qc", {}),
            array_secondary_ratio=sec_ratio,
        )
        qc_passed = dq.passed
        qc_reasons = dq.reasons
        qc_metrics = dq.metrics
        nucleus_qc_passed = dq.nucleus_passed
        array_qc_passed = dq.array_passed
        if dq.suggested_obs:
            motion_suggested_obs = dq.suggested_obs
    else:
        qc = automated_qc(
            stack=stack_qc,
            nucleus_mask=nuc_mask,
            array_mask=arr_mask,
            mean_roi_intensity=mean_roi,
            epsilon_nuc=nuc_res["epsilon"],
            epsilon_array=arr_res["epsilon"],
            qc_cfg=cfg["qc"],
            intensity_range=tuple(cfg["intensity_range"]),
        )
        qc_passed = qc.passed
        qc_reasons = qc.reasons
        qc_metrics = qc.metrics
        nucleus_qc_passed = qc_passed and nuc_res["epsilon"] != -1.0
        array_qc_passed = qc_passed and arr_res["epsilon"] != -1.0

    manual_obs = None
    manual_included = None
    if manual_record is not None:
        manual_obs = manual_record["nucleus"].get("obs")
        manual_included = manual_record["nucleus"]["included"]

    return {
        "file": lsm_path.name,
        "cell_index": cell_index,
        "mean_roi_intensity": mean_roi,
        "nucleus": nuc_res,
        "array": arr_res,
        "nucleus_pixels": int(nuc_mask.sum()),
        "array_pixels": int(arr_mask.sum()),
        "roi_source": roi_source,
        "qc_passed": qc_passed,
        "nucleus_qc_passed": nucleus_qc_passed,
        "array_qc_passed": array_qc_passed,
        "qc_reasons": qc_reasons,
        "qc_metrics": qc_metrics,
        "motion_review_recommended": motion_review_recommended,
        "motion_flag_summary": motion_flag_summary,
        "motion_flag_codes": motion_flag_codes,
        "motion_suggested_obs": motion_suggested_obs,
        "motion_metrics": motion_review.metrics if motion_review else {},
        "manual_obs": manual_obs,
        "manual_included": manual_included,
        "prep_meta": prep_meta,
        "reg_meta": reg_meta,
        "ib_meta": ib_meta,
        "roi_mode": roi_mode,
    }


def run_session(
    cfg: dict,
    session_key: str,
    *,
    conditions: list[str] | None = None,
    epsilon_method: str = "weighted_mean",
    output_dir: Path,
) -> pd.DataFrame:
    session = cfg["sessions"][session_key]
    session_dir = Path(cfg["raw_data_root"]) / session["folder"]
    roi_root = cfg.get("roi_root")
    roi_cache = Path(cfg["roi_cache_root"]) if cfg.get("roi_cache_root") else None
    manual_path = cfg["manual_workbooks"][session_key]
    rows = []

    for cond in session["conditions"]:
        if conditions and cond["id"] not in conditions:
            continue
        cond_dir = session_dir / cond["subdir"]
        roi_dir = Path(roi_root) / session_key / cond["id"] if roi_root else None
        manual_by_id = {
            m["cell_id"]: m
            for m in read_manual_block(manual_path, cond["manual_sheet_block"])
        }
        files = sorted(cond_dir.glob(cond["file_glob"]))
        for f in files:
            cell_index = manual_file_index(f.name)
            manual_rec = manual_by_id.get(cell_index) if manual_by_id and cell_index else None
            rec = analyze_cell(
                f,
                cfg,
                epsilon_method=epsilon_method,
                roi_dir=roi_dir,
                roi_cache=roi_cache,
                session_key=session_key,
                condition_id=cond["id"],
                manual_record=manual_rec,
            )
            rec["condition"] = cond["id"]
            rec["session"] = session_key
            rows.append(rec)

    df_cells = pd.json_normalize(rows, sep="_")
    df_cells.to_csv(output_dir / f"cells_{session_key}.csv", index=False)

    summary_rows = []
    for cond in session["conditions"]:
        if conditions and cond["id"] not in conditions:
            continue
        sub = df_cells[df_cells["condition"] == cond["id"]]
        included = sub[sub["qc_passed"] & (sub["nucleus_epsilon"] != -1)]
        nuc_mean = included["nucleus_epsilon"].mean() if len(included) else np.nan
        arr_mean = included[included["array_epsilon"] != -1]["array_epsilon"].mean() if len(included) else np.nan
        summary_rows.append(
            {
                "session": session_key,
                "condition": cond["id"],
                "n_cells_total": len(sub),
                "n_cells_included": len(included),
                "mean_epsilon_nucleus": nuc_mean,
                "mean_epsilon_array": arr_mean,
            }
        )

    df_summary = pd.DataFrame(summary_rows)
    df_summary = apply_control_normalization(
        df_summary, session.get("control_condition")
    )

    df_summary.to_csv(output_dir / f"summary_{session_key}.csv", index=False)
    build_motion_review_report(cfg, session_key, df_cells, output_dir)
    return df_cells, df_summary


def build_motion_review_report(
    cfg: dict,
    session_key: str,
    df_cells: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Human-review queue: motion flags sorted by priority."""
    if not cfg.get("motion_flags", {}).get("enabled", True):
        return pd.DataFrame()

    rows = []
    for _, row in df_cells.iterrows():
        codes = row.get("motion_flag_codes") or []
        if isinstance(codes, str):
            codes = [c.strip() for c in codes.strip("[]").replace("'", "").split(",") if c.strip()]

        manual_obs = row.get("manual_obs")
        manual_inc = row.get("manual_included")
        if pd.isna(manual_inc):
            manual_inc = None
        else:
            manual_inc = bool(manual_inc)

        manual_cat = match_manual_obs_category(
            manual_obs if pd.notna(manual_obs) else None,
            manual_inc if manual_inc is not None else True,
        )

        # Flag / manual concordance
        analog_hits = {
            "likely_z_movement": manual_cat in ("hard_exclude_z_movement", "hard_exclude_rotation"),
            "likely_rotation": manual_cat == "hard_exclude_rotation",
            "likely_xy_movement": manual_cat == "hard_exclude_xy_movement",
            "possible_xy_movement": manual_cat == "hard_exclude_xy_movement",
            "possible_array_motion": manual_cat == "hard_exclude_array_motion",
            "possible_multiple_arrays": manual_cat == "soft_flag_multiple_arrays",
        }
        matched = [c for c in codes if analog_hits.get(c)]
        missed_hard = manual_cat.startswith("hard_exclude") and not matched
        false_alarm = row.get("motion_review_recommended") and manual_cat == "included_clean"

        priority = 0
        if missed_hard:
            priority = 3
        elif row.get("motion_review_recommended"):
            priority = 2
        elif codes:
            priority = 1

        rows.append(
            {
                "session": row["session"],
                "condition": row["condition"],
                "cell_id": row["cell_index"],
                "file": row["file"],
                "review_recommended": bool(row.get("motion_review_recommended")),
                "motion_flag_summary": row.get("motion_flag_summary"),
                "suggested_qc_note": row.get("motion_suggested_obs") or "",
                "motion_flag_codes": "|".join(codes) if codes else "",
                "primary_concern": row.get("motion_flag_summary", "").split(";")[0] if row.get("motion_flag_summary") else "",
                "manual_included": manual_inc,
                "manual_obs": manual_obs,
                "manual_obs_category": manual_cat,
                "flag_matches_manual": bool(matched),
                "missed_hard_exclude": missed_hard,
                "flagged_but_manually_clean": false_alarm,
                "max_rigid_shift_px": row.get("motion_metrics_max_rigid_shift_px"),
                "mean_rigid_shift_px": row.get("motion_metrics_mean_rigid_shift_px"),
                "min_frame_corr": row.get("motion_metrics_min_frame_corr"),
                "rotation_range_rad": row.get("motion_metrics_rotation_range_rad"),
                "nuclear_mean_cv": row.get("motion_metrics_nuclear_mean_cv"),
                "array_secondary_ratio": row.get("motion_metrics_array_secondary_ratio"),
                "review_priority": priority,
            }
        )

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["review_priority", "review_recommended"], ascending=[False, False])
    df.to_csv(output_dir / f"motion_review_{session_key}.csv", index=False)
    priority = df[df["review_recommended"] | df["missed_hard_exclude"]].copy()
    priority.to_csv(output_dir / f"motion_review_{session_key}_priority.csv", index=False)

    # Summary stats for calibration feedback
    hard = df[df.manual_obs_category.str.startswith("hard_exclude", na=False)]
    detected = hard["flag_matches_manual"].sum() if len(hard) else 0
    summary = {
        "session": session_key,
        "n_cells": int(len(df)),
        "n_review_recommended": int(df["review_recommended"].sum()),
        "n_hard_manual_excludes": int(len(hard)),
        "n_hard_excludes_flagged": int(detected),
        "hard_exclude_detection_rate": float(detected / len(hard)) if len(hard) else None,
        "n_false_alarms_on_clean": int(df["flagged_but_manually_clean"].sum()),
    }
    with open(output_dir / f"motion_review_{session_key}_report.json", "w") as f:
        json.dump(summary, f, indent=2)
    return df


def compare_to_manual(
    cfg: dict,
    session_key: str,
    df_cells: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    manual_path = cfg["manual_workbooks"][session_key]
    session = cfg["sessions"][session_key]
    cmp_rows = []

    for cond in session["conditions"]:
        manual = read_manual_block(manual_path, cond["manual_sheet_block"])
        manual_by_id = {m["cell_id"]: m for m in manual}
        sub = df_cells[df_cells["condition"] == cond["id"]]

        for _, row in sub.iterrows():
            cid = int(row["cell_index"]) if pd.notna(row["cell_index"]) else None
            if cid is None or cid not in manual_by_id:
                continue
            man = manual_by_id[cid]
            for compartment in ("nucleus", "array"):
                m = man[compartment]
                comp_prefix = compartment
                auto_e = row[f"{comp_prefix}_epsilon"]
                man_inc = m["included"]
                if cfg.get("use_manual_inclusion"):
                    auto_inc = man_inc and auto_e != -1
                elif compartment == "nucleus" and pd.notna(row.get("nucleus_qc_passed")):
                    auto_inc = bool(row["nucleus_qc_passed"]) and auto_e != -1
                elif compartment == "array" and pd.notna(row.get("array_qc_passed")):
                    auto_inc = bool(row["array_qc_passed"]) and auto_e != -1
                else:
                    auto_inc = bool(row["qc_passed"]) and auto_e != -1
                man_obs = m.get("obs")
                keyword_excl = manual_obs_excluded(man_obs, cfg["qc"]["exclude_obs_keywords"])

                cmp_rows.append(
                    {
                        "session": session_key,
                        "condition": cond["id"],
                        "cell_id": cid,
                        "compartment": compartment,
                        "manual_included": man_inc,
                        "manual_obs": man_obs,
                        "manual_epsilon": m["epsilon"],
                        "manual_norm": m["norm_epsilon"],
                        "auto_included": auto_inc,
                        "auto_epsilon": auto_e,
                        "auto_qc_reasons": row["qc_reasons"],
                        "epsilon_diff": (auto_e - m["epsilon"]) if man_inc and auto_inc and m["epsilon"] is not None else np.nan,
                        "inclusion_match": man_inc == auto_inc,
                        "keyword_would_exclude": keyword_excl,
                    }
                )

    df_cmp = pd.DataFrame(cmp_rows)
    df_cmp.to_csv(output_dir / f"compare_{session_key}.csv", index=False)

    # Per-condition breakdown
    by_cond = []
    for cond_id, grp in df_cmp.groupby("condition"):
        inc = grp["inclusion_match"].mean() if len(grp) else None
        mae = grp["epsilon_diff"].abs().mean() if grp["epsilon_diff"].notna().any() else None
        by_cond.append({"condition": cond_id, "inclusion_agreement": inc, "epsilon_mae": mae, "n": len(grp)})
    by_cond_df = pd.DataFrame(by_cond)
    by_cond_df.to_csv(output_dir / f"compare_{session_key}_by_condition.csv", index=False)

    both_included = df_cmp["manual_included"] & df_cmp["auto_included"]
    all_valid = df_cmp["manual_epsilon"].notna() & (df_cmp["auto_epsilon"] != -1)

    by_compartment = []
    for comp, grp in df_cmp.groupby("compartment"):
        bi = grp["manual_included"] & grp["auto_included"]
        av = grp["manual_epsilon"].notna() & (grp["auto_epsilon"] != -1)
        by_compartment.append(
            {
                "compartment": comp,
                "inclusion_agreement": float(grp["inclusion_match"].mean()) if len(grp) else None,
                "included_epsilon_mae": float(grp.loc[bi, "epsilon_diff"].abs().mean()) if bi.any() else None,
                "all_cells_epsilon_mae": float(
                    (grp.loc[av, "auto_epsilon"] - grp.loc[av, "manual_epsilon"]).abs().mean()
                )
                if av.any()
                else None,
            }
        )

    nuc_cmp = df_cmp[df_cmp["compartment"] == "nucleus"]
    hard_manual = nuc_cmp[~nuc_cmp["manual_included"]]
    hard_auto_excl = nuc_cmp[~nuc_cmp["auto_included"] & nuc_cmp["manual_included"]]
    hard_detected = hard_manual["auto_included"].eq(False).sum() if len(hard_manual) else 0
    keyword_excl = nuc_cmp["keyword_would_exclude"].sum() if "keyword_would_exclude" in nuc_cmp else 0

    report = {
        "session": session_key,
        "inclusion_agreement": float(df_cmp["inclusion_match"].mean()) if len(df_cmp) else None,
        "nucleus_inclusion_agreement": float(nuc_cmp["inclusion_match"].mean()) if len(nuc_cmp) else None,
        "motion_hard_excludes": {
            "n_manual_excluded": int(len(hard_manual)),
            "n_auto_excluded_matching": int(hard_detected),
            "detection_rate": float(hard_detected / len(hard_manual)) if len(hard_manual) else None,
            "n_false_excludes_on_included": int(len(hard_auto_excl)),
            "n_keyword_obs_would_exclude": int(keyword_excl),
        },
        "included_epsilon_mae": float(df_cmp.loc[both_included, "epsilon_diff"].abs().mean())
        if both_included.any()
        else None,
        "all_cells_epsilon_mae": float(
            (df_cmp.loc[all_valid, "auto_epsilon"] - df_cmp.loc[all_valid, "manual_epsilon"]).abs().mean()
        )
        if all_valid.any()
        else None,
        "n_compared": int(len(df_cmp)),
        "by_condition": by_cond_df.to_dict(orient="records"),
        "by_compartment": by_compartment,
        "workflow": {
            "discard_first_frames": cfg["discard_first_frames"],
            "detrend": cfg.get("detrend", True),
            "roi_mode": cfg.get("roi_mode", "geometric"),
            "use_manual_inclusion": cfg.get("use_manual_inclusion", False),
            "register_for_qc": cfg.get("register_for_qc", False),
            "ib_brightness_max": cfg.get("ib_gating", {}).get("brightness_max", 2.0),
            "note": "Trim 10 frames (bleaching), no detrend, I-B gating Y<=2, automated QC.",
        },
    }
    with open(output_dir / f"compare_{session_key}_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return df_cmp