"""Trim and detrend stacks (pysimfcs / SimFCS Tutorial order)."""

from __future__ import annotations

import numpy as np


def trim_stack(stack: np.ndarray, discard_first: int) -> np.ndarray:
    if discard_first <= 0:
        return stack.astype(np.float64, copy=False)
    return stack[discard_first:].astype(np.float64)


def detrend_stack_linear(stack: np.ndarray) -> np.ndarray:
    """
    Per-pixel linear detrend (pysimfcs ``detrendStackLinear``).
    Subtracts bleach/focus drift trend while preserving temporal mean.
    """
    if stack.ndim != 3:
        raise ValueError(f"Expected TYX stack, got {stack.shape}")

    t = np.arange(stack.shape[0], dtype=np.float64)
    t_mean = t.mean()
    t_centered = t - t_mean
    denom = float((t_centered**2).sum())
    if denom <= 0:
        return stack.copy()

    ts_mean = stack.mean(axis=0)
    ts_centered = stack - ts_mean
    slope = (t_centered[:, None, None] * ts_centered).sum(axis=0) / denom
    intercept = ts_mean - slope * t_mean
    trend = slope[None, :, :] * t[:, None, None] + intercept[None, :, :]
    detrended = stack - trend
    # Restore per-pixel mean so N&B brightness scale matches manual workflow.
    detrended += ts_mean[None, :, :]
    return detrended


def preprocess_stack(
    stack: np.ndarray,
    *,
    discard_first: int,
    detrend: bool = True,
) -> tuple[np.ndarray, dict]:
    """Trim leading frames and optionally linear-detrend."""
    meta = {"discard_first": discard_first, "detrended": detrend}
    trimmed = trim_stack(stack, discard_first)
    meta["n_frames"] = int(trimmed.shape[0])
    if detrend:
        trimmed = detrend_stack_linear(trimmed)
    return trimmed, meta