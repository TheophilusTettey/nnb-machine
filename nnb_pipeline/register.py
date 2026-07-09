"""Rigid 2D registration to reference frame (pre-N&B)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.registration import phase_cross_correlation


def register_stack_rigid(
    stack: np.ndarray,
    *,
    reference_frame: int = 0,
    upsample_factor: int = 10,
) -> tuple[np.ndarray, dict]:
    """
    Align every frame to ``reference_frame`` via phase cross-correlation.
    Returns registered stack and shift metadata.
    """
    if stack.ndim != 3:
        raise ValueError(f"Expected TYX stack, got {stack.shape}")

    ref = stack[reference_frame].astype(np.float64)
    registered = np.empty_like(stack)
    registered[reference_frame] = ref
    shifts = []

    for t in range(stack.shape[0]):
        if t == reference_frame:
            shifts.append((0.0, 0.0))
            continue
        shift, _, _ = phase_cross_correlation(
            ref,
            stack[t],
            upsample_factor=upsample_factor,
        )
        # skimage returns (row, col) = (y, x)
        registered[t] = ndimage.shift(
            stack[t],
            shift=shift,
            order=1,
            mode="nearest",
            prefilter=False,
        )
        shifts.append((float(shift[0]), float(shift[1])))

    shift_arr = np.asarray(shifts)
    metrics = {
        "reference_frame": reference_frame,
        "max_shift_y": float(np.abs(shift_arr[:, 0]).max()),
        "max_shift_x": float(np.abs(shift_arr[:, 1]).max()),
        "mean_shift_y": float(np.abs(shift_arr[:, 0]).mean()),
        "mean_shift_x": float(np.abs(shift_arr[:, 1]).mean()),
    }
    return registered, metrics