"""
SimFCS-style I–B scatter gating.

Per pixel: X = mean intensity, Y = brightness B = var/mean.
1. Y gate: B <= ib_brightness_max (default 2) — removes border/motion artifacts.
2. X gate: intensity window per compartment (nucleus vs array).
3. Mask = pixels satisfying both gates; ε from Gaussian histogram on masked pixels.
"""

from __future__ import annotations

import numpy as np

from .excel_gauss import excel_gauss_fit_score
from .nb_core import epsilon_from_roi, gaussian_hist_mean


def compute_pixel_maps(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-pixel mean intensity, brightness, and epsilon (full frame)."""
    mean_i = stack.mean(axis=0)
    var_i = stack.var(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        brightness = var_i / mean_i
    eps = brightness - 1.0
    return mean_i, brightness, eps


def y_gate_mask(
    brightness: np.ndarray,
    mean_i: np.ndarray,
    *,
    b_max: float = 2.0,
    b_min: float = 0.5,
    min_mean: float = 1.0,
) -> np.ndarray:
    """Y cutoff: discard pixels with B > b_max."""
    return (
        (mean_i >= min_mean)
        & np.isfinite(brightness)
        & (brightness >= b_min)
        & (brightness <= b_max)
    )


def _flat_ib_data(
    mean_i: np.ndarray,
    brightness: np.ndarray,
    base_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten Y-gated pixels for fast window search."""
    m = mean_i[base_mask]
    b = brightness[base_mask]
    e = b - 1.0
    idx = np.flatnonzero(base_mask)
    return m, b, e, idx


def _window_score(
    eps: np.ndarray,
    weights: np.ndarray,
    *,
    min_pixels: int,
    i_lo: float,
    i_hi: float,
    compartment: str,
) -> float:
    if eps.size < min_pixels:
        return 1e9
    hist, edges = np.histogram(eps, bins=min(40, max(10, eps.size // 2)), weights=weights)
    if hist.sum() <= 0 or (hist > 0).sum() < 3:
        return 1e8
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = hist > 0
    x = centers[mask]
    y = hist[mask].astype(float)
    y = y / y.sum()
    mu = float(np.average(eps, weights=weights))
    model = y.max() * np.exp(-0.5 * ((x - mu) / max(0.05, float(np.std(eps)))) ** 2)
    sse = float(np.sum((y - model) ** 2))
    mean_i = float(np.average(weights, weights=weights))
    width = i_hi - i_lo

    score = sse + 0.02 / eps.size
    if compartment == "nucleus":
        # Nucleoplasm cloud: moderate intensity, broader window.
        score += max(0.0, mean_i - 17.0) * 0.04
        score += max(0.0, i_lo - 14.0) * 0.08
        score += max(0.0, 8.0 - width) * 0.03
    else:
        # Array core: brighter and tighter than nucleus.
        score += max(0.0, 20.0 - mean_i) * 0.05
        score += max(0.0, 4.0 - width) * 0.02
    return score


def search_intensity_window(
    mean_i: np.ndarray,
    brightness: np.ndarray,
    base_mask: np.ndarray,
    *,
    i_lo_range: tuple[float, float],
    i_hi_range: tuple[float, float],
    step: float = 0.5,
    min_window: float = 2.0,
    min_pixels: int = 40,
    compartment: str = "nucleus",
) -> tuple[float, float, np.ndarray, dict]:
    """Trial-and-error X gate: search [i_min, i_max] minimizing histogram SSE."""
    flat_m, flat_b, flat_e, flat_idx = _flat_ib_data(mean_i, brightness, base_mask)
    if flat_m.size == 0:
        empty = np.zeros_like(base_mask, dtype=bool)
        return i_lo_range[0], i_hi_range[1], empty, {"compartment": compartment, "n_pixels": 0}

    lo_start, lo_end = i_lo_range
    hi_start, hi_end = i_hi_range
    best_score = 1e9
    best_lo, best_hi = lo_start, hi_end
    best_n = 0

    for i_lo in np.arange(lo_start, lo_end + step / 2, step):
        lo_ok = flat_m >= i_lo
        for i_hi in np.arange(max(i_lo + min_window, hi_start), hi_end + step / 2, step):
            sel = lo_ok & (flat_m <= i_hi)
            n = int(sel.sum())
            if n < min_pixels:
                continue
            score = _window_score(
                flat_e[sel],
                flat_m[sel],
                min_pixels=min_pixels,
                i_lo=float(i_lo),
                i_hi=float(i_hi),
                compartment=compartment,
            )
            if score < best_score:
                best_score = score
                best_lo, best_hi = float(i_lo), float(i_hi)
                best_n = n

    gate = base_mask & (mean_i >= best_lo) & (mean_i <= best_hi)
    meta = {
        "compartment": compartment,
        "intensity_min": best_lo,
        "intensity_max": best_hi,
        "score": best_score,
        "n_pixels": int(gate.sum()) if gate.any() else best_n,
    }
    return best_lo, best_hi, gate, meta


def search_intensity_window_excel(
    mean_i: np.ndarray,
    brightness: np.ndarray,
    base_mask: np.ndarray,
    *,
    i_lo_range: tuple[float, float],
    i_hi_range: tuple[float, float],
    step: float = 0.5,
    min_window: float = 2.0,
    min_pixels: int = 40,
    compartment: str = "nucleus",
    umbral_start: float = 0.626,
    umbral_step: float = 0.02282,
    n_bins: int = 45,
) -> tuple[float, float, np.ndarray, dict]:
    """X gate search scored by Excel Gaussian fit quality (machine accuracy mode)."""
    flat_m, flat_b, flat_e, _ = _flat_ib_data(mean_i, brightness, base_mask)
    if flat_m.size == 0:
        empty = np.zeros_like(base_mask, dtype=bool)
        return i_lo_range[0], i_hi_range[1], empty, {"compartment": compartment, "n_pixels": 0}

    lo_start, lo_end = i_lo_range
    hi_start, hi_end = i_hi_range
    best_score = 1e9
    best_lo, best_hi = lo_start, hi_end

    for i_lo in np.arange(lo_start, lo_end + step / 2, step):
        lo_ok = flat_m >= i_lo
        for i_hi in np.arange(max(i_lo + min_window, hi_start), hi_end + step / 2, step):
            sel = lo_ok & (flat_m <= i_hi)
            if int(sel.sum()) < min_pixels:
                continue
            score = excel_gauss_fit_score(
                flat_b[sel],
                flat_m[sel],
                umbral_start=umbral_start,
                umbral_step=umbral_step,
                n_bins=n_bins,
                min_pixels=min_pixels,
            )
            # Prior intensity search ranges from config
            width = float(i_hi - i_lo)
            mean_iwin = float(flat_m[sel].mean())
            if compartment == "nucleus":
                score += max(0.0, mean_iwin - 17.0) * 0.03
                score += max(0.0, float(i_lo) - 14.0) * 0.05
            else:
                score += max(0.0, 18.0 - mean_iwin) * 0.04
                score += max(0.0, 3.0 - width) * 0.02
            if score < best_score:
                best_score = score
                best_lo, best_hi = float(i_lo), float(i_hi)

    gate = base_mask & (mean_i >= best_lo) & (mean_i <= best_hi)
    meta = {
        "compartment": compartment,
        "intensity_min": best_lo,
        "intensity_max": best_hi,
        "score": best_score,
        "n_pixels": int(gate.sum()),
        "search_mode": "excel_gauss",
    }
    return best_lo, best_hi, gate, meta


def build_machine_ib_masks(
    stack: np.ndarray,
    *,
    ib_brightness_max: float = 2.0,
    pixel_min_mean: float = 1.0,
    brightness_min: float = 0.5,
    nucleus_i_lo: tuple[float, float] = (6.0, 18.0),
    nucleus_i_hi: tuple[float, float] = (14.0, 32.0),
    array_i_lo: tuple[float, float] = (12.0, 35.0),
    array_i_hi: tuple[float, float] = (18.0, 55.0),
    search_step: float = 0.5,
    min_nucleus_pixels: int = 40,
    min_array_pixels: int = 8,
    umbral_start: float = 0.626,
    umbral_step: float = 0.02282,
    array_umbral_start: float | None = None,
    umbral_n_bins: int = 45,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """I–B gating optimized for Excel Gaussian B (N&B Machine accuracy mode)."""
    mean_i, brightness, _ = compute_pixel_maps(stack)
    y_mask = y_gate_mask(
        brightness,
        mean_i,
        b_max=ib_brightness_max,
        b_min=brightness_min,
        min_mean=pixel_min_mean,
    )

    nuc_lo, nuc_hi, nuc_mask, nuc_meta = search_intensity_window_excel(
        mean_i,
        brightness,
        y_mask,
        i_lo_range=nucleus_i_lo,
        i_hi_range=nucleus_i_hi,
        step=search_step,
        min_pixels=min_nucleus_pixels,
        compartment="nucleus",
        umbral_start=umbral_start,
        umbral_step=umbral_step,
        n_bins=umbral_n_bins,
    )

    arr_lo_start = max(array_i_lo[0], nuc_hi - 1.0)
    arr_umbral = array_umbral_start if array_umbral_start is not None else umbral_start
    arr_lo, arr_hi, arr_mask, arr_meta = search_intensity_window_excel(
        mean_i,
        brightness,
        y_mask,
        i_lo_range=(arr_lo_start, array_i_lo[1]),
        i_hi_range=(max(array_i_hi[0], arr_lo_start + 2.0), array_i_hi[1]),
        step=search_step,
        min_pixels=min_array_pixels,
        compartment="array",
        umbral_start=arr_umbral,
        umbral_step=umbral_step,
        n_bins=umbral_n_bins,
    )

    meta = {
        "roi_mode": "machine_ib_excel",
        "ib_brightness_max": ib_brightness_max,
        "y_gated_pixels": int(y_mask.sum()),
        "motion_ring_fraction": motion_ring_fraction(brightness, y_mask),
        "nucleus": nuc_meta,
        "array": arr_meta,
        "nucleus_intensity_range": [nuc_lo, nuc_hi],
        "array_intensity_range": [arr_lo, arr_hi],
    }
    return nuc_mask, arr_mask, meta


def motion_ring_fraction(
    brightness: np.ndarray,
    base_mask: np.ndarray,
    *,
    b_lo: float = 1.5,
    b_hi: float = 2.0,
) -> float:
    if not base_mask.any():
        return 0.0
    ring = base_mask & (brightness >= b_lo) & (brightness <= b_hi)
    return float(ring.sum() / base_mask.sum())


def build_ib_scatter_masks(
    stack: np.ndarray,
    *,
    ib_brightness_max: float = 2.0,
    pixel_min_mean: float = 1.0,
    brightness_min: float = 0.5,
    nucleus_i_lo: tuple[float, float] = (6.0, 18.0),
    nucleus_i_hi: tuple[float, float] = (14.0, 32.0),
    array_i_lo: tuple[float, float] = (15.0, 35.0),
    array_i_hi: tuple[float, float] = (22.0, 55.0),
    search_step: float = 0.5,
    min_nucleus_pixels: int = 40,
    min_array_pixels: int = 15,
) -> tuple[np.ndarray, np.ndarray, dict]:
    mean_i, brightness, _ = compute_pixel_maps(stack)
    y_mask = y_gate_mask(
        brightness,
        mean_i,
        b_max=ib_brightness_max,
        b_min=brightness_min,
        min_mean=pixel_min_mean,
    )

    nuc_lo, nuc_hi, nuc_mask, nuc_meta = search_intensity_window(
        mean_i,
        brightness,
        y_mask,
        i_lo_range=nucleus_i_lo,
        i_hi_range=nucleus_i_hi,
        step=search_step,
        min_pixels=min_nucleus_pixels,
        compartment="nucleus",
    )

    arr_lo_start = max(array_i_lo[0], nuc_hi + 1.0)
    arr_lo, arr_hi, arr_mask, arr_meta = search_intensity_window(
        mean_i,
        brightness,
        y_mask,
        i_lo_range=(arr_lo_start, array_i_lo[1]),
        i_hi_range=(max(array_i_hi[0], arr_lo_start + 2.0), array_i_hi[1]),
        step=search_step,
        min_pixels=min_array_pixels,
        compartment="array",
    )

    meta = {
        "roi_mode": "ib_scatter",
        "ib_brightness_max": ib_brightness_max,
        "y_gated_pixels": int(y_mask.sum()),
        "motion_ring_fraction": motion_ring_fraction(brightness, y_mask),
        "nucleus": nuc_meta,
        "array": arr_meta,
        "nucleus_intensity_range": [nuc_lo, nuc_hi],
        "array_intensity_range": [arr_lo, arr_hi],
    }
    return nuc_mask, arr_mask, meta


def epsilon_from_ib_mask(
    stack: np.ndarray,
    mask: np.ndarray,
    *,
    method: str = "gaussian_hist",
    min_mean: float = 1.0,
    b_min: float = 0.5,
    b_max: float = 2.0,
) -> dict:
    mean_i, brightness, _ = compute_pixel_maps(stack)
    y_mask = y_gate_mask(brightness, mean_i, b_max=b_max, b_min=b_min, min_mean=min_mean)
    final = mask & y_mask
    return epsilon_from_roi(
        stack,
        final,
        method=method,
        min_mean=min_mean,
        b_min=b_min,
        b_max=b_max,
    )