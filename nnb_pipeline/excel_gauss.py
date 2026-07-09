"""
Excel Gauss sheet equivalent — BarSeries1 umbral grid + Gaussian fit → B.

Manual workflow: export histogram from SimFCS → Gauss sheet → Solver → mean x₀ = B.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .nb_core import pixel_brightness


def umbral_bin_edges(
    *,
    start: float,
    step: float,
    n_bins: int = 45,
) -> np.ndarray:
    return start + step * np.arange(n_bins + 1, dtype=np.float64)


def weighted_umbral_histogram(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    start: float,
    step: float,
    n_bins: int = 45,
) -> tuple[np.ndarray, np.ndarray]:
    edges = umbral_bin_edges(start=start, step=step, n_bins=n_bins)
    hist, _ = np.histogram(values, bins=edges, weights=weights)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return hist.astype(float), centers


def fit_gaussian_solver_style(
    centers: np.ndarray,
    counts: np.ndarray,
) -> tuple[float, float]:
    """
    Gaussian fit on histogram counts (Excel Solver analogue).
    Returns (mu, sigma) in same units as centers (brightness B).
    """
    mask = counts > 0
    if mask.sum() < 3:
        return float(np.nan), float(np.nan)

    x = centers[mask]
    y = counts[mask].astype(float)
    y = y / y.sum()
    mu0 = float(np.average(x, weights=y))
    a0 = float(y.max())
    s0 = max(step_estimate(x, y), 0.02)

    def loss(params):
        a, mu, sig = params
        if a <= 0 or sig <= 1e-4:
            return 1e9
        model = a * np.exp(-0.5 * ((x - mu) / sig) ** 2)
        return float(np.sum((y - model) ** 2))

    res = minimize(loss, [a0, mu0, s0], method="Nelder-Mead")
    if not res.success:
        return mu0, s0
    return float(res.x[1]), float(res.x[2])


def step_estimate(x: np.ndarray, y: np.ndarray) -> float:
    mu = float(np.average(x, weights=y))
    var = float(np.average((x - mu) ** 2, weights=y))
    return float(np.sqrt(max(var, 1e-6)))


def adaptive_umbral_start(
    values: np.ndarray,
    *,
    default_start: float,
    step: float,
) -> float:
    """Align umbral grid with data (fixed Excel Gauss sheet layout)."""
    if values.size == 0:
        return default_start
    lo = float(np.percentile(values, 3))
    start = min(default_start, max(0.5, lo - 2.0 * step))
    return start


def brightness_from_pixels(
    brightness_values: np.ndarray,
    weights: np.ndarray,
    *,
    umbral_start: float,
    umbral_step: float,
    n_bins: int = 45,
    min_pixels: int = 5,
    adaptive_umbral: bool = True,
) -> dict:
    """B from gated pixel brightness list using Excel umbral grid."""
    if brightness_values.size < min_pixels:
        return {"brightness": np.nan, "epsilon": -1.0, "n_pixels": int(brightness_values.size), "method": "excel_gauss"}

    start = (
        adaptive_umbral_start(brightness_values, default_start=umbral_start, step=umbral_step)
        if adaptive_umbral
        else umbral_start
    )

    hist, centers = weighted_umbral_histogram(
        brightness_values,
        weights,
        start=start,
        step=umbral_step,
        n_bins=n_bins,
    )
    mu, sigma = fit_gaussian_solver_style(centers, hist)
    if not np.isfinite(mu) or mu < 0:
        return {"brightness": np.nan, "epsilon": -1.0, "n_pixels": int(brightness_values.size), "method": "excel_gauss"}

    b = mu
    eps = b - 1.0
    if eps < 0:
        # Monomer / very dim array: fall back to weighted mean B
        b = float(np.average(brightness_values, weights=weights))
        eps = b - 1.0
        if eps < 0:
            return {"brightness": np.nan, "epsilon": -1.0, "n_pixels": int(brightness_values.size), "method": "excel_gauss"}
        return {
            "brightness": b,
            "epsilon": eps,
            "n_pixels": int(brightness_values.size),
            "method": "excel_gauss_fallback_mean",
            "gauss_mu": mu,
            "gauss_sigma": sigma,
        }

    return {
        "brightness": b,
        "epsilon": eps,
        "n_pixels": int(brightness_values.size),
        "method": "excel_gauss",
        "gauss_mu": mu,
        "gauss_sigma": sigma,
        "umbral_start_used": start,
    }


def excel_gauss_fit_score(
    brightness_values: np.ndarray,
    weights: np.ndarray,
    *,
    umbral_start: float,
    umbral_step: float,
    n_bins: int = 45,
    min_pixels: int = 8,
) -> float:
    """Lower is better — used to pick I–B X windows (machine accuracy mode)."""
    res = brightness_from_pixels(
        brightness_values,
        weights,
        umbral_start=umbral_start,
        umbral_step=umbral_step,
        n_bins=n_bins,
        min_pixels=min_pixels,
    )
    if res["epsilon"] == -1.0:
        return 1e9
    sigma = res.get("gauss_sigma", np.nan)
    penalty = 0.0 if np.isfinite(sigma) and 0.02 <= sigma <= 0.25 else 0.5
    return penalty + 0.1 / brightness_values.size


def brightness_from_mask(
    stack: np.ndarray,
    mask: np.ndarray,
    y_gate: np.ndarray,
    *,
    umbral_start: float,
    umbral_step: float,
    n_bins: int = 45,
    min_mean: float = 1.0,
    b_min: float = 0.5,
    b_max: float = 2.0,
) -> dict:
    """Collect per-pixel B inside mask ∩ y_gate, fit Excel-style Gaussian."""
    final = mask & y_gate
    b_list = []
    w_list = []
    ys, xs = np.where(final)
    for y, x in zip(ys, xs):
        b, e, m = pixel_brightness(stack[:, y, x])
        if m < min_mean or not np.isfinite(b) or b < b_min or b > b_max:
            continue
        b_list.append(b)
        w_list.append(m)
    if not b_list:
        return {"brightness": np.nan, "epsilon": -1.0, "n_pixels": 0, "method": "excel_gauss"}
    return brightness_from_pixels(
        np.asarray(b_list),
        np.asarray(w_list),
        umbral_start=umbral_start,
        umbral_step=umbral_step,
        n_bins=n_bins,
    )