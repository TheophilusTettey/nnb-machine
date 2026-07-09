"""QC rules for motion and acquisition quality (SimFCS Obs-style keywords)."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class QCResult:
    passed: bool
    reasons: list[str]
    metrics: dict


def focus_stability_metrics(stack: np.ndarray, nucleus_mask: np.ndarray) -> dict:
    """Proxy for Z-focus / shape drift (Z mov, cell rotates)."""
    if not nucleus_mask.any():
        return {
            "nuclear_mean_cv": np.nan,
            "nuclear_area_cv": np.nan,
            "laplacian_var_cv": np.nan,
        }

    means = []
    areas = []
    lap_vars = []
    for t in range(stack.shape[0]):
        roi = stack[t][nucleus_mask]
        if roi.size < 10:
            continue
        means.append(float(roi.mean()))
        areas.append(float(((stack[t] > np.percentile(stack[t], 90)) & nucleus_mask).sum()))
        lap = ndimage.laplace(stack[t].astype(np.float64))
        lap_vars.append(float(lap[nucleus_mask].var()))

    def cv(vals: list[float]) -> float:
        if len(vals) < 3:
            return np.nan
        arr = np.asarray(vals)
        m = arr.mean()
        return float(arr.std() / m) if m > 0 else np.nan

    return {
        "nuclear_mean_cv": cv(means),
        "nuclear_area_cv": cv(areas),
        "laplacian_var_cv": cv(lap_vars),
    }


def movement_metrics(stack: np.ndarray, nucleus_mask: np.ndarray) -> dict:
    ys, xs = np.where(nucleus_mask)
    if len(xs) == 0:
        return {"centroid_drift_y": np.nan, "centroid_drift_x": np.nan, "centroid_std_y": np.nan, "centroid_std_x": np.nan, "min_frame_corr": np.nan}

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    cents = []
    for t in range(stack.shape[0]):
        m = stack[t] > np.percentile(stack[t], 92)
        m &= nucleus_mask
        if m.sum() < 20:
            cents.append((np.nan, np.nan))
        else:
            cents.append(ndimage.center_of_mass(stack[t], m))
    c = np.array(cents)
    ref = stack[0, y0 : y1 + 1, x0 : x1 + 1].astype(np.float64)
    ref = (ref - ref.mean()) / (ref.std() + 1e-6)
    corrs = []
    for t in range(1, stack.shape[0]):
        im = stack[t, y0 : y1 + 1, x0 : x1 + 1].astype(np.float64)
        im = (im - im.mean()) / (im.std() + 1e-6)
        corrs.append(float((ref * im).mean()))

    return {
        "centroid_drift_y": float(np.nanmax(c[:, 0]) - np.nanmin(c[:, 0])),
        "centroid_drift_x": float(np.nanmax(c[:, 1]) - np.nanmin(c[:, 1])),
        "centroid_std_y": float(np.nanstd(c[:, 0])),
        "centroid_std_x": float(np.nanstd(c[:, 1])),
        "min_frame_corr": float(min(corrs)) if corrs else np.nan,
    }


def intensity_in_range(mean_roi: float, lo: float, hi: float) -> bool:
    return lo <= mean_roi <= hi


def automated_qc(
    *,
    stack: np.ndarray,
    nucleus_mask: np.ndarray,
    array_mask: np.ndarray,
    mean_roi_intensity: float,
    epsilon_nuc: float,
    epsilon_array: float,
    qc_cfg: dict,
    intensity_range: tuple[float, float],
) -> QCResult:
    reasons = []
    metrics = movement_metrics(stack, nucleus_mask)
    metrics.update(focus_stability_metrics(stack, nucleus_mask))
    metrics["mean_roi_intensity"] = mean_roi_intensity

    lo, hi = intensity_range
    if not intensity_in_range(mean_roi_intensity, lo, hi):
        reasons.append(f"intensity_out_of_range ({mean_roi_intensity:.1f} not in [{lo},{hi}])")

    if epsilon_nuc == -1.0:
        reasons.append("invalid_nucleus_epsilon")
    if epsilon_array == -1.0:
        reasons.append("invalid_array_epsilon")

    if qc_cfg.get("use_movement_qc", False):
        if metrics["centroid_drift_y"] > qc_cfg.get("max_centroid_drift_px", 12):
            reasons.append("centroid_drift_y")
        if metrics["centroid_drift_x"] > qc_cfg.get("max_centroid_drift_px", 12):
            reasons.append("centroid_drift_x")
        if metrics["centroid_std_y"] > qc_cfg.get("max_centroid_std_px", 2.5):
            reasons.append("centroid_std_y")
        if metrics["centroid_std_x"] > qc_cfg.get("max_centroid_std_px", 2.5):
            reasons.append("centroid_std_x")
        if metrics["min_frame_corr"] < qc_cfg.get("min_frame_correlation", 0.65):
            reasons.append("low_frame_correlation")
        if metrics.get("nuclear_mean_cv", 0) > qc_cfg.get("max_nuclear_mean_cv", 0.12):
            reasons.append("z_focus_drift")
        if metrics.get("laplacian_var_cv", 0) > qc_cfg.get("max_laplacian_var_cv", 0.35):
            reasons.append("focus_instability")

    # Two-array heuristic: secondary peak nearly as bright as primary
    if qc_cfg.get("use_multiple_array_qc", False):
        nuc_img = stack.mean(0) * nucleus_mask
        if array_mask.any() and nucleus_mask.any():
            blurred = ndimage.gaussian_filter(nuc_img, 1)
            primary = blurred[array_mask].max()
            work = blurred.copy()
            work[array_mask] = 0
            secondary = work[nucleus_mask].max() if nucleus_mask.any() else 0
            ratio = secondary / primary if primary > 0 else 0
            metrics["array_secondary_ratio"] = float(ratio)
            if ratio > qc_cfg.get("max_array_secondary_ratio", 0.85):
                reasons.append("multiple_arrays")

    return QCResult(passed=len(reasons) == 0, reasons=reasons, metrics=metrics)


def manual_obs_excluded(obs_text: str | None, keywords: list[str]) -> bool:
    if not obs_text:
        return False
    text = obs_text.lower().strip()
    for kw in keywords:
        if kw.lower() in text:
            # "little mov" kept manually — don't auto-exclude
            if "little" in text and "mov" in kw:
                continue
            return True
    return False


def parse_manual_obs(obs: str | None) -> list[str]:
    if not obs:
        return []
    return [s.strip() for s in re.split(r"[;,]", obs) if s.strip()]