"""
Automated whole-cell exclusion when motion or artifacts make I–B gating unreliable.

Used by the legacy geometric pipeline path; N&B Machine uses the ML QC model instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .motion_flags import MotionReview, suggested_qc_note


@dataclass
class MovieExclusionResult:
    passed: bool
    nucleus_passed: bool
    array_passed: bool
    reasons: list[str]
    suggested_obs: str
    metrics: dict


def _motion_hard_exclude(
    motion_review: MotionReview | None,
    ib_meta: dict | None,
    qc_cfg: dict | None,
) -> tuple[bool, list[str]]:
    """Skip cell when motion / I–B artifact patterns exceed thresholds."""
    reasons: list[str] = []
    if motion_review is None:
        return False, reasons

    m = motion_review.metrics
    ring_thr = (qc_cfg or {}).get("max_motion_ring_fraction", 0.38)
    ring = float((ib_meta or {}).get("motion_ring_fraction", 0.0))
    min_corr = m.get("min_frame_corr", np.nan)
    max_shift = m.get("max_rigid_shift_px", np.nan)
    mean_shift = m.get("mean_rigid_shift_px", np.nan)

    codes = set(motion_review.codes())
    if "likely_rotation" in codes:
        reasons.append("motion_cell_rotates")
        return True, reasons

    if np.isfinite(ring) and ring > ring_thr:
        reasons.append(f"ib_motion_ring ({ring:.2f})")
        return True, reasons

    if (
        np.isfinite(min_corr)
        and min_corr < (qc_cfg or {}).get("z_exclude_min_frame_corr", -0.10)
        and ring > (qc_cfg or {}).get("z_exclude_min_ring", 0.12)
    ):
        reasons.append("z_mov_pattern")
        return True, reasons

    if "likely_z_movement" in codes:
        reasons.append("motion_z_mov")
        return True, reasons

    if (
        np.isfinite(max_shift)
        and max_shift >= (qc_cfg or {}).get("xy_exclude_max_shift_px", 24.5)
        and np.isfinite(min_corr)
        and min_corr < (qc_cfg or {}).get("xy_exclude_min_frame_corr", -0.12)
    ):
        reasons.append("it_moves_pattern")
        return True, reasons

    if (
        "likely_xy_movement" in codes
        and np.isfinite(mean_shift)
        and mean_shift >= (qc_cfg or {}).get("xy_exclude_mean_shift_px", 15.0)
        and np.isfinite(min_corr)
        and min_corr < -0.05
    ):
        reasons.append("motion_it_moves")
        return True, reasons

    return False, reasons


def automated_movie_exclusion(
    *,
    motion_review: MotionReview | None,
    epsilon_nuc: float,
    epsilon_array: float,
    ib_meta: dict | None = None,
    qc_cfg: dict | None = None,
    array_secondary_ratio: float | None = None,
) -> MovieExclusionResult:
    metrics: dict = {}
    if ib_meta:
        metrics["motion_ring_fraction"] = ib_meta.get("motion_ring_fraction")
        metrics["nucleus_intensity_range"] = ib_meta.get("nucleus_intensity_range")
        metrics["array_intensity_range"] = ib_meta.get("array_intensity_range")

    motion_exclude, reasons = _motion_hard_exclude(motion_review, ib_meta, qc_cfg)

    if motion_review is not None:
        metrics.update(motion_review.metrics)

    if array_secondary_ratio is not None and np.isfinite(array_secondary_ratio):
        metrics["array_secondary_ratio"] = float(array_secondary_ratio)

    if epsilon_nuc == -1.0 and not motion_exclude:
        reasons.append("invalid_nucleus_epsilon")
    if epsilon_array == -1.0 and not motion_exclude:
        reasons.append("invalid_array_epsilon")

    nucleus_passed = (not motion_exclude) and epsilon_nuc != -1.0
    array_passed = (not motion_exclude) and epsilon_array != -1.0

    suggested = ""
    if motion_review is not None:
        suggested = suggested_qc_note(motion_review.flags)

    return MovieExclusionResult(
        passed=not motion_exclude,
        nucleus_passed=nucleus_passed,
        array_passed=array_passed,
        reasons=reasons,
        suggested_obs=suggested,
        metrics=metrics,
    )