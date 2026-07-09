"""Number & brightness (Digman 2008 / Globals for Images conventions)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize


def pixel_brightness(time_series: np.ndarray) -> tuple[float, float, float]:
    """Return (B, epsilon, mean_I) for one pixel time series."""
    mean_i = float(np.mean(time_series))
    if mean_i <= 0:
        return np.nan, -1.0, mean_i
    var_i = float(np.var(time_series, ddof=1))
    b = var_i / mean_i
    eps = b - 1.0
    return b, eps, mean_i


def roi_pixel_epsilons(
    stack: np.ndarray,
    mask: np.ndarray,
    *,
    min_mean: float = 1.0,
    b_min: float = 0.5,
    b_max: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel epsilon and weights (mean intensity) inside a 2D mask."""
    eps_list = []
    weight_list = []
    ys, xs = np.where(mask)
    for y, x in zip(ys, xs):
        b, e, m = pixel_brightness(stack[:, y, x])
        if m < min_mean or not np.isfinite(b):
            continue
        if b < b_min or b > b_max:
            continue
        eps_list.append(e)
        weight_list.append(m)
    if not eps_list:
        return np.array([]), np.array([])
    return np.asarray(eps_list), np.asarray(weight_list)


def epsilon_from_roi(
    stack: np.ndarray,
    mask: np.ndarray,
    *,
    method: str = "weighted_mean",
    min_mean: float = 1.0,
    b_min: float = 0.5,
    b_max: float = 20.0,
) -> dict:
    """
    Summarize ROI brightness.
    method: 'weighted_mean' (SimFCS export style) or 'gaussian_hist' (Presman figure 1B).
    """
    eps, weights = roi_pixel_epsilons(
        stack, mask, min_mean=min_mean, b_min=b_min, b_max=b_max
    )
    if eps.size == 0:
        return {"epsilon": -1.0, "brightness": np.nan, "n_pixels": 0, "method": method}

    if method == "gaussian_hist":
        e_roi = gaussian_hist_mean(eps, weights)
    else:
        e_roi = float(np.average(eps, weights=weights))

    if not np.isfinite(e_roi) or e_roi < 0:
        return {"epsilon": -1.0, "brightness": np.nan, "n_pixels": int(eps.size), "method": method}

    return {
        "epsilon": e_roi,
        "brightness": e_roi + 1.0,
        "n_pixels": int(eps.size),
        "method": method,
    }


def gaussian_hist_mean(eps: np.ndarray, weights: np.ndarray) -> float:
    """Gaussian fit to weighted histogram of pixel epsilons (Presman 2014 panel B)."""
    if eps.size < 5:
        return float(np.average(eps, weights=weights))

    hist, edges = np.histogram(eps, bins=min(40, max(10, eps.size // 2)), weights=weights)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = hist > 0
    if mask.sum() < 3:
        return float(np.average(eps, weights=weights))

    x = centers[mask]
    y = hist[mask].astype(float)
    y = y / y.sum()

    x0 = float(np.average(eps, weights=weights))
    a0 = float(y.max())
    s0 = max(0.05, float(np.std(eps)))

    def loss(params):
        a, mu, sig = params
        if a <= 0 or sig <= 1e-3:
            return 1e9
        model = a * np.exp(-0.5 * ((x - mu) / sig) ** 2)
        return float(np.sum((y - model) ** 2))

    res = minimize(loss, [a0, x0, s0], method="Nelder-Mead")
    if not res.success:
        return float(np.average(eps, weights=weights))
    return float(res.x[1])


def normalize_epsilon(epsilon: float, control_mean: float) -> float:
    if epsilon == -1.0 or not np.isfinite(epsilon) or control_mean <= 0:
        return np.nan
    return epsilon / control_mean


def apply_control_normalization(
    df_summary,
    control_condition: str | None,
    *,
    nucleus_col: str = "mean_epsilon_nucleus",
    array_col: str = "mean_epsilon_array",
):
    """Add norm_nucleus / norm_array columns relative to one monomer control condition."""
    if not control_condition:
        return df_summary
    ctrl = df_summary[df_summary["condition"] == control_condition]
    if len(ctrl) == 0:
        return df_summary
    out = df_summary.copy()
    ctrl_n = float(ctrl[nucleus_col].iloc[0])
    ctrl_a = float(ctrl[array_col].iloc[0])
    if np.isfinite(ctrl_n) and ctrl_n > 0:
        out["norm_nucleus"] = out[nucleus_col] / ctrl_n
    if np.isfinite(ctrl_a) and ctrl_a > 0:
        out["norm_array"] = out[array_col] / ctrl_a
    return out


def detect_nucleus_mask(mean_image: np.ndarray, percentile: float = 60.0) -> np.ndarray:
    """
    Tight nuclear ROI — manual SimFCS draws a small box/ellipse on the nucleus,
    not the whole cell footprint. Use high percentile + area cap.
    """
    pos = mean_image[mean_image > 0]
    if pos.size == 0:
        return np.zeros_like(mean_image, dtype=bool)

    # Start strict; relax only if mask is empty
    for pct in (92, 88, percentile):
        thr = max(float(np.percentile(pos, pct)), 4.0)
        mask = mean_image > thr
        mask = ndimage.binary_opening(mask, iterations=1)
        mask = ndimage.binary_closing(mask, iterations=1)
        labeled, n = ndimage.label(mask)
        if n == 0:
            continue
        sizes = [(labeled == i).sum() for i in range(1, n + 1)]
        nuc = labeled == (int(np.argmax(sizes)) + 1)
        # Nucleus should be a compact ROI, not most of the 256×256 field
        if 80 <= nuc.sum() <= 6000:
            cy, cx = ndimage.center_of_mass(mean_image, nuc)
            y_grid, x_grid = np.ogrid[: mean_image.shape[0], : mean_image.shape[1]]
            dist = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)
            # ~25 px radius ≈ 2 µm — typical manual SimFCS nuclear box
            tight = nuc & (dist <= 25)
            if tight.sum() >= 50:
                return tight
            return nuc
    return np.zeros_like(mean_image, dtype=bool)


def detect_array_mask(mean_image: np.ndarray, nucleus_mask: np.ndarray, radius_px: int = 6) -> np.ndarray:
    nuc_img = mean_image * nucleus_mask
    if nuc_img.max() <= 0:
        return np.zeros_like(nucleus_mask, dtype=bool)
    peak = np.unravel_index(int(np.argmax(nuc_img)), nuc_img.shape)
    yy, xx = peak
    y_grid, x_grid = np.ogrid[: mean_image.shape[0], : mean_image.shape[1]]
    dist = np.sqrt((y_grid - yy) ** 2 + (x_grid - xx) ** 2)
    return (dist <= radius_px) & nucleus_mask