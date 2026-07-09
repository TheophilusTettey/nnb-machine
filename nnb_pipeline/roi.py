"""ROI detection — SimFCS-style small boxes + optional ImageJ .roi import."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage


def rectangular_mask(
    shape: tuple[int, int],
    center_y: float,
    center_x: float,
    half_height: int,
    half_width: int,
) -> np.ndarray:
    h, w = shape
    cy, cx = int(round(center_y)), int(round(center_x))
    y0 = max(0, cy - half_height)
    y1 = min(h, cy + half_height + 1)
    x0 = max(0, cx - half_width)
    x1 = min(w, cx + half_width + 1)
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def circular_mask(
    shape: tuple[int, int],
    center_y: float,
    center_x: float,
    radius_px: int,
) -> np.ndarray:
    y_grid, x_grid = np.ogrid[: shape[0], : shape[1]]
    dist = np.sqrt((y_grid - center_y) ** 2 + (x_grid - center_x) ** 2)
    return dist <= radius_px


def _nuclear_seed(mean_image: np.ndarray, percentile: float) -> tuple[float, float] | None:
    pos = mean_image[mean_image > 0]
    if pos.size == 0:
        return None

    for pct in (92, 88, percentile, 80):
        thr = max(float(np.percentile(pos, pct)), 4.0)
        mask = mean_image > thr
        mask = ndimage.binary_opening(mask, iterations=1)
        mask = ndimage.binary_closing(mask, iterations=1)
        labeled, n = ndimage.label(mask)
        if n == 0:
            continue
        sizes = [(labeled == i).sum() for i in range(1, n + 1)]
        nuc = labeled == (int(np.argmax(sizes)) + 1)
        if nuc.sum() < 30:
            continue
        cy, cx = ndimage.center_of_mass(mean_image, nuc)
        return float(cy), float(cx)
    return None


def auto_nucleus_mask(
    mean_image: np.ndarray,
    *,
    percentile: float = 88.0,
    box_half_height: int = 10,
    box_half_width: int = 10,
) -> np.ndarray:
    """Small rectangular nucleus ROI centered on brightest nuclear region."""
    seed = _nuclear_seed(mean_image, percentile)
    if seed is None:
        return np.zeros(mean_image.shape, dtype=bool)
    return rectangular_mask(
        mean_image.shape,
        seed[0],
        seed[1],
        box_half_height,
        box_half_width,
    )


def auto_array_mask(
    mean_image: np.ndarray,
    nucleus_mask: np.ndarray,
    *,
    radius_px: int = 4,
) -> np.ndarray:
    """Tight circular ROI on brightest punctum inside nucleus mask."""
    nuc_img = mean_image * nucleus_mask
    if nuc_img.max() <= 0:
        return np.zeros_like(nucleus_mask, dtype=bool)
    peak = np.unravel_index(int(np.argmax(nuc_img)), nuc_img.shape)
    return circular_mask(mean_image.shape, peak[0], peak[1], radius_px)


def load_imagej_roi_mask(roi_path: Path, shape: tuple[int, int]) -> np.ndarray | None:
    """
    Load ImageJ .roi into boolean mask. Returns None if file missing or unreadable.
    """
    try:
        import roifile
    except ImportError:
        return None

    if not roi_path.exists():
        return None

    roi = roifile.roiread(str(roi_path))
    mask = np.zeros(shape, dtype=bool)
    if hasattr(roi, "coordinates"):
        coords = np.asarray(roi.coordinates, dtype=int)
        if coords.ndim == 2 and coords.shape[1] >= 2:
            ys = np.clip(coords[:, 1], 0, shape[0] - 1)
            xs = np.clip(coords[:, 0], 0, shape[1] - 1)
            mask[ys, xs] = True
            mask = ndimage.binary_fill_holes(mask)
            mask = ndimage.binary_dilation(mask, iterations=1)
            return mask

    if hasattr(roi, "top") and hasattr(roi, "left"):
        top, left = int(roi.top), int(roi.left)
        bottom = int(getattr(roi, "bottom", top))
        right = int(getattr(roi, "right", left))
        y0, y1 = max(0, top), min(shape[0], bottom + 1)
        x0, x1 = max(0, left), min(shape[1], right + 1)
        mask[y0:y1, x0:x1] = True
        return mask
    return None


def masks_from_fitted_params(fitted: dict, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    nuc = fitted["nucleus"]
    nucleus_mask = rectangular_mask(
        shape, nuc["center_y"], nuc["center_x"], nuc["half_px"], nuc["half_px"]
    )
    array_mask = np.zeros(shape, dtype=bool)
    if fitted.get("array"):
        arr = fitted["array"]
        array_mask = circular_mask(shape, arr["center_y"], arr["center_x"], arr["radius_px"])
    return nucleus_mask, array_mask


def resolve_roi_masks(
    mean_image: np.ndarray,
    *,
    roi_dir: Path | None,
    fitted_roi: dict | None,
    cell_index: int | None,
    nucleus_percentile: float,
    nucleus_box_half: int,
    array_radius_px: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Priority: fitted cache → ImageJ ROIs → auto boxes.
    """
    shape = mean_image.shape[:2]
    if fitted_roi is not None and fitted_roi.get("nucleus"):
        nuc, arr = masks_from_fitted_params(fitted_roi, shape)
        return nuc, arr, "fitted"

    if roi_dir is not None and cell_index is not None:
        nuc_path = roi_dir / f"{cell_index:02d}_nucleus.roi"
        arr_path = roi_dir / f"{cell_index:02d}_array.roi"
        nuc = load_imagej_roi_mask(nuc_path, shape)
        arr = load_imagej_roi_mask(arr_path, shape)
        if nuc is not None and arr is not None and nuc.any() and arr.any():
            return nuc, arr, "imagej"

    nuc = auto_nucleus_mask(
        mean_image,
        percentile=nucleus_percentile,
        box_half_height=nucleus_box_half,
        box_half_width=nucleus_box_half,
    )
    arr = auto_array_mask(mean_image, nuc, radius_px=array_radius_px)
    return nuc, arr, "auto"