"""
Flag cells with motion/quality patterns for human review (SimFCS Obs-style notes).

REVIEW flags for double-checking — not auto-exclusions in N&B Machine (ML QC handles that).

Common manual exclude reasons in N&B movies:
  Z mov / mov Z     — focal plane drift; cell brightness breathes, low XY shift
  Cell rotates      — nucleus reorients (often with some XY displacement)
  it moves          — large translational displacement frame-to-frame
  array moves       — bright punctum drifts independently of nucleus
  Two arrays        — secondary focus nearly as bright as primary
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from skimage.registration import phase_cross_correlation

from .qc import focus_stability_metrics, movement_metrics


@dataclass
class MotionFlag:
    code: str
    label: str
    severity: str  # high | medium | low | info
    detail: str
    qc_note: str = ""


@dataclass
class MotionReview:
    flags: list[MotionFlag] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    review_recommended: bool = False
    primary_concern: str | None = None

    def summary(self) -> str:
        if not self.flags:
            return ""
        high_med = [f for f in self.flags if f.severity in ("high", "medium")]
        show = high_med if high_med else self.flags[:2]
        return "; ".join(f.label for f in show)

    def codes(self) -> list[str]:
        return [f.code for f in self.flags]


def _rigid_shift_metrics(stack: np.ndarray, nucleus_mask: np.ndarray) -> dict:
    ys, xs = np.where(nucleus_mask)
    if len(xs) == 0:
        return {"max_rigid_shift_px": np.nan, "mean_rigid_shift_px": np.nan}

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
        return {"max_rigid_shift_px": np.nan, "mean_rigid_shift_px": np.nan}
    return {
        "max_rigid_shift_px": float(max(norms)),
        "mean_rigid_shift_px": float(np.mean(norms)),
    }


def _array_motion_metrics(stack: np.ndarray, array_mask: np.ndarray) -> dict:
    if not array_mask.any():
        return {"array_centroid_jitter": np.nan, "array_max_jump": np.nan}

    mean_arr = stack.mean(0) * array_mask
    peak = np.unravel_index(int(np.argmax(mean_arr)), mean_arr.shape)
    cents = []
    for t in range(stack.shape[0]):
        m = stack[t] > np.percentile(stack[t], 95)
        m &= array_mask
        if m.sum() < 3:
            cents.append(peak)
        else:
            cents.append(ndimage.center_of_mass(stack[t], m))
    c = np.array(cents)
    jumps = np.sqrt(np.diff(c[:, 0]) ** 2 + np.diff(c[:, 1]) ** 2)
    return {
        "array_centroid_jitter": float(c[:, 0].std() + c[:, 1].std()),
        "array_max_jump": float(jumps.max()) if len(jumps) else np.nan,
    }


def _multiple_array_ratio(mean_image: np.ndarray, nucleus_mask: np.ndarray, array_mask: np.ndarray) -> float:
    if not array_mask.any() or not nucleus_mask.any():
        return np.nan
    nuc_img = mean_image * nucleus_mask
    blurred = ndimage.gaussian_filter(nuc_img, 1)
    primary = blurred[array_mask].max()
    work = blurred.copy()
    work[array_mask] = 0
    secondary = work[nucleus_mask].max()
    return float(secondary / primary) if primary > 0 else np.nan


def compute_motion_metrics(
    stack: np.ndarray,
    nucleus_mask: np.ndarray,
    array_mask: np.ndarray,
    mean_image: np.ndarray,
) -> dict:
    metrics = movement_metrics(stack, nucleus_mask)
    metrics.update(focus_stability_metrics(stack, nucleus_mask))
    metrics.update(_rigid_shift_metrics(stack, nucleus_mask))
    metrics.update(_array_motion_metrics(stack, array_mask))
    metrics["array_secondary_ratio"] = _multiple_array_ratio(mean_image, nucleus_mask, array_mask)
    metrics["total_centroid_drift"] = metrics.get("centroid_drift_y", 0) + metrics.get("centroid_drift_x", 0)
    return metrics


def classify_motion_flags(metrics: dict, thresholds: dict | None = None) -> MotionReview:
    t = thresholds or {}
    flags: list[MotionFlag] = []

    shift_max = metrics.get("max_rigid_shift_px", np.nan)
    shift_mean = metrics.get("mean_rigid_shift_px", np.nan)
    min_corr = metrics.get("min_frame_corr", np.nan)
    mean_cv = metrics.get("nuclear_mean_cv", np.nan)
    area_cv = metrics.get("nuclear_area_cv", np.nan)
    drift = metrics.get("total_centroid_drift", np.nan)
    cstd = metrics.get("centroid_std_y", 0) + metrics.get("centroid_std_x", 0)
    arr_jump = metrics.get("array_max_jump", np.nan)
    sec_ratio = metrics.get("array_secondary_ratio", np.nan)

    # --- Z movement (manual QC: "Z mov") ---
    # Signature: low XY rigid shift, negative frame correlation,
    # low intensity CV, shrinking/tight nuclear footprint (low area_cv).
    z_focus_pattern = (
        np.isfinite(min_corr)
        and min_corr < t.get("z_mov_max_frame_corr", -0.07)
        and np.isfinite(mean_cv)
        and mean_cv < t.get("z_mov_max_mean_cv", 0.026)
    )
    z_mov = z_focus_pattern and (
        not np.isfinite(area_cv) or area_cv < t.get("z_mov_max_area_cv", 0.09)
    )
    if z_mov:
        flags.append(
            MotionFlag(
                code="likely_z_movement",
                label="Possible Z movement / focus drift",
                severity="high",
                detail=(
                    f"Frame corr {min_corr:.2f}, intensity CV {mean_cv:.3f} "
                    f"(mean shift {shift_mean:.1f} px) — typical of Z mov / cell rotates."
                ),
                qc_note="Z mov",
            )
        )

    # --- XY movement (manual QC: "it moves") ---
    if (
        np.isfinite(shift_mean)
        and shift_mean >= t.get("xy_shift_high_mean_px", 11.5)
        and np.isfinite(min_corr)
        and min_corr > t.get("xy_min_frame_corr", -0.03)
        and not z_focus_pattern
    ):
        flags.append(
            MotionFlag(
                code="likely_xy_movement",
                label="Likely XY movement",
                severity="high",
                detail=f"Sustained translational shift (mean {shift_mean:.1f} px/frame).",
                qc_note="it moves",
            )
        )
    elif (
        np.isfinite(shift_mean)
        and shift_mean >= t.get("xy_shift_medium_mean_px", 12.0)
        and np.isfinite(cstd)
        and cstd > t.get("centroid_std_px", 0.52)
        and not z_focus_pattern
    ):
        flags.append(
            MotionFlag(
                code="possible_xy_movement",
                label="Possible XY movement",
                severity="medium",
                detail=f"Mean shift {shift_mean:.1f} px with jittery centroid (σ={cstd:.2f}).",
                qc_note="it moves?",
            )
        )

    # --- Rotation (manual QC: "Cell rotates") ---
    if (
        np.isfinite(shift_max)
        and shift_max >= t.get("rotation_min_shift_px", 50.0)
        and np.isfinite(drift)
        and drift > t.get("rotation_min_drift_px", 5.0)
    ):
        flags.append(
            MotionFlag(
                code="likely_rotation",
                label="Possible cell rotation / large displacement",
                severity="high",
                detail=f"Very large shift ({shift_max:.0f} px) — check for rotation or translation.",
                qc_note="Cell rotates",
            )
        )

    # --- Array motion ---
    if np.isfinite(arr_jump) and arr_jump > t.get("array_max_jump_px", 2.8):
        flags.append(
            MotionFlag(
                code="possible_array_motion",
                label="Possible array punctum motion",
                severity="medium",
                detail=f"Array centroid jump up to {arr_jump:.1f} px.",
                qc_note="array moves",
            )
        )

    # --- Multiple arrays (info only — ratio is high in most GR cells) ---
    if np.isfinite(sec_ratio) and sec_ratio > t.get("multiple_array_ratio_info", 0.94):
        flags.append(
            MotionFlag(
                code="check_multiple_arrays",
                label="Check for multiple arrays (visual)",
                severity="info",
                detail=f"Secondary peak at {sec_ratio:.0%} of primary — confirm in movie.",
                qc_note="Two arrays?",
            )
        )

    # --- Soft: little movement (often kept in manual QC) ---
    if (
        np.isfinite(shift_max)
        and t.get("little_mov_shift_lo", 12.0) <= shift_max < t.get("xy_shift_medium_px", 18.0)
        and not z_mov
    ):
        flags.append(
            MotionFlag(
                code="little_movement",
                label="Little movement (often kept)",
                severity="low",
                detail=f"Moderate shift {shift_max:.1f} px.",
                qc_note="little mov",
            )
        )

    high = [f for f in flags if f.severity == "high"]
    medium = [f for f in flags if f.severity == "medium"]
    review = len(high) > 0
    primary = (high or medium or flags)[0].label if flags else None

    return MotionReview(
        flags=flags,
        metrics=metrics,
        review_recommended=review,
        primary_concern=primary,
    )


def suggested_qc_note(flags: list[MotionFlag]) -> str:
    """Short Obs-style hint for manual review in SimFCS/Fiji."""
    codes = {f.code for f in flags}
    if "likely_z_movement" in codes:
        return "Z mov? (auto)"
    if "likely_xy_movement" in codes:
        return "it moves? (auto)"
    if "likely_rotation" in codes:
        return "Cell rotates? (auto)"
    if "possible_array_motion" in codes:
        return "array moves? (auto)"
    if "check_multiple_arrays" in codes:
        return "Two arrays? (auto)"
    if "little_movement" in codes:
        return "little mov? (auto)"
    if "possible_xy_movement" in codes:
        return "it moves? (auto)"
    return ""


def match_manual_obs_category(obs: str | None, included: bool) -> str:
    if not included:
        if not obs:
            return "excluded_no_comment"
        o = obs.lower().strip()
        if "rotat" in o:
            return "hard_exclude_rotation"
        if "z mov" in o or "mov z" in o:
            return "hard_exclude_z_movement"
        if "it moves" in o or o == "mov":
            return "hard_exclude_xy_movement"
        if "no array" in o or "too small" in o or "too open" in o:
            return "hard_exclude_array_issue"
        if "array moves" in o or "array move" in o:
            return "hard_exclude_array_motion"
        return "hard_exclude_other"
    if not obs:
        return "included_clean"
    o = obs.lower()
    if "little" in o and "mov" in o:
        return "soft_flag_little_movement"
    if "two array" in o:
        return "soft_flag_multiple_arrays"
    if "ymax" in o:
        return "soft_flag_ymax_note"
    if "cyto" in o or "aggr" in o:
        return "soft_flag_cyto_aggregate"
    if "overlap" in o:
        return "soft_flag_overlap_foci"
    if "?" in o:
        return "soft_flag_uncertain"
    return "soft_flag_other_note"