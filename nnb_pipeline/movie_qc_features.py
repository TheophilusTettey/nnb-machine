"""
Movie QC features — computational equivalent of watching the stack in Fiji/SimFCS.

Extracted before N&B on the trimmed stack. Used by the QC classifier and audit reports.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.registration import phase_cross_correlation

from .qc import focus_stability_metrics, movement_metrics
from .roi import auto_array_mask, auto_nucleus_mask


def bleaching_features(stack: np.ndarray, nucleus_mask: np.ndarray) -> dict:
    """Bleaching: average intensity vs frame number (~5% loss typical)."""
    if not nucleus_mask.any():
        return {
            "bleach_slope_per_frame": np.nan,
            "bleach_pct_total": np.nan,
            "mean_intensity_first10": np.nan,
            "mean_intensity_last10": np.nan,
            "frames_below_half_peak": np.nan,
        }

    means = []
    for t in range(stack.shape[0]):
        roi = stack[t][nucleus_mask]
        means.append(float(roi.mean()) if roi.size else np.nan)
    m = np.array(means, dtype=float)
    valid = np.isfinite(m)
    if valid.sum() < 5:
        return {k: np.nan for k in (
            "bleach_slope_per_frame", "bleach_pct_total",
            "mean_intensity_first10", "mean_intensity_last10", "frames_below_half_peak",
        )}

    t_idx = np.arange(len(m))[valid]
    mm = m[valid]
    slope = float(np.polyfit(t_idx, mm, 1)[0])
    m0 = float(mm[: min(10, len(mm))].mean())
    m1 = float(mm[-min(10, len(mm)) :].mean())
    pct = 100.0 * (m1 - m0) / m0 if m0 > 0 else np.nan
    peak = float(mm.max())
    below_half = float((mm < 0.5 * peak).mean()) if peak > 0 else np.nan
    return {
        "bleach_slope_per_frame": slope,
        "bleach_pct_total": pct,
        "mean_intensity_first10": m0,
        "mean_intensity_last10": m1,
        "frames_below_half_peak": below_half,
    }


def rotation_proxy(stack: np.ndarray, nucleus_mask: np.ndarray) -> dict:
    """Proxy for 'cell rotates' from nuclear mask orientation drift."""
    if not nucleus_mask.any() or stack.shape[0] < 5:
        return {"orientation_std_rad": np.nan, "eccentricity_std": np.nan}

    angles = []
    ecc = []
    for t in range(stack.shape[0]):
        m = stack[t] > np.percentile(stack[t], 92)
        m &= nucleus_mask
        if m.sum() < 30:
            continue
        props = ndimage.center_of_mass(stack[t], m)
        ys, xs = np.where(m)
        cy, cx = props
        cov = np.cov(xs - cx, ys - cy)
        if cov.ndim == 2:
            evals, evecs = np.linalg.eigh(cov)
            angles.append(float(np.arctan2(evecs[1, 1], evecs[0, 1])))
            ecc.append(float(np.sqrt(evals.max() / (evals.min() + 1e-6))))

    if len(angles) < 3:
        return {"orientation_std_rad": np.nan, "eccentricity_std": np.nan}
    a = np.unwrap(np.array(angles))
    return {
        "orientation_std_rad": float(a.std()),
        "eccentricity_std": float(np.std(ecc)),
    }


def rigid_shift_series(stack: np.ndarray, nucleus_mask: np.ndarray) -> dict:
    ys, xs = np.where(nucleus_mask)
    if len(xs) == 0:
        return {"max_shift_px": np.nan, "mean_shift_px": np.nan, "shift_std_px": np.nan}

    pad = 8
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(stack.shape[1] - 1, int(ys.max()) + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(stack.shape[2] - 1, int(xs.max()) + pad)
    ref = stack[0, y0 : y1 + 1, x0 : x1 + 1]
    norms = []
    for t in range(1, stack.shape[0]):
        shift, _, _ = phase_cross_correlation(
            ref, stack[t, y0 : y1 + 1, x0 : x1 + 1], upsample_factor=1
        )
        norms.append(float(np.linalg.norm(shift)))
    if not norms:
        return {"max_shift_px": np.nan, "mean_shift_px": np.nan, "shift_std_px": np.nan}
    n = np.array(norms)
    return {
        "max_shift_px": float(n.max()),
        "mean_shift_px": float(n.mean()),
        "shift_std_px": float(n.std()),
    }


def extract_movie_qc_features(
    stack: np.ndarray,
    *,
    nucleus_percentile: float = 88.0,
    nucleus_box_half: int = 10,
    array_radius_px: int = 4,
) -> dict:
    """
    Full movie feature vector for one cell (trimmed stack, shape T×Y×X).
    """
    mean_img = stack.mean(axis=0)
    nuc = auto_nucleus_mask(
        mean_img,
        percentile=nucleus_percentile,
        box_half_height=nucleus_box_half,
        box_half_width=nucleus_box_half,
    )
    arr = auto_array_mask(mean_img, nuc, radius_px=array_radius_px)

    feats: dict = {}
    feats.update(movement_metrics(stack, nuc))
    feats.update(focus_stability_metrics(stack, nuc))
    feats.update(bleaching_features(stack, nuc))
    feats.update(rotation_proxy(stack, nuc))
    feats.update(rigid_shift_series(stack, nuc))

    # Array punctum stability
    if arr.any():
        cents = []
        for t in range(stack.shape[0]):
            m = stack[t] > np.percentile(stack[t], 95)
            m &= arr
            if m.sum() < 3:
                cents.append((np.nan, np.nan))
            else:
                cents.append(ndimage.center_of_mass(stack[t], m))
        c = np.array(cents)
        jumps = np.sqrt(np.diff(c[:, 0]) ** 2 + np.diff(c[:, 1]) ** 2)
        feats["array_max_jump_px"] = float(np.nanmax(jumps)) if len(jumps) else np.nan
        feats["array_centroid_std"] = float(np.nanstd(c[:, 0]) + np.nanstd(c[:, 1]))
    else:
        feats["array_max_jump_px"] = np.nan
        feats["array_centroid_std"] = np.nan

    feats["nuclear_mask_pixels"] = int(nuc.sum())
    feats["n_frames"] = int(stack.shape[0])
    return feats


FEATURE_NAMES: list[str] = [
    "centroid_drift_y",
    "centroid_drift_x",
    "centroid_std_y",
    "centroid_std_x",
    "min_frame_corr",
    "nuclear_mean_cv",
    "nuclear_area_cv",
    "laplacian_var_cv",
    "bleach_slope_per_frame",
    "bleach_pct_total",
    "mean_intensity_first10",
    "mean_intensity_last10",
    "frames_below_half_peak",
    "orientation_std_rad",
    "eccentricity_std",
    "max_shift_px",
    "mean_shift_px",
    "shift_std_px",
    "array_max_jump_px",
    "array_centroid_std",
    "nuclear_mask_pixels",
    "n_frames",
]


def features_to_vector(feats: dict) -> np.ndarray:
    return np.array([float(feats.get(k, np.nan)) for k in FEATURE_NAMES], dtype=np.float64)