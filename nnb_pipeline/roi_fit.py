"""Fit per-cell ROIs to match reference ε and intensity range from workbooks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .nb_core import epsilon_from_roi
from .roi import _nuclear_seed, circular_mask, rectangular_mask


def _search_nucleus(
    stack: np.ndarray,
    mean_image: np.ndarray,
    *,
    target_epsilon: float,
    intensity_lo: float,
    intensity_hi: float,
    epsilon_method: str,
    seed: tuple[float, float] | None,
) -> dict | None:
    if seed is None:
        seed = _nuclear_seed(mean_image, 88)
    if seed is None:
        return None

    best: tuple[float, dict] | None = None
    cy0, cx0 = seed

    for half in range(5, 18):
        for dy in range(-24, 25, 2):
            for dx in range(-24, 25, 2):
                cy, cx = cy0 + dy, cx0 + dx
                mask = rectangular_mask(mean_image.shape, cy, cx, half, half)
                if mask.sum() < 25:
                    continue
                mi = float(mean_image[mask].mean())
                if not (intensity_lo <= mi <= intensity_hi):
                    continue
                res = epsilon_from_roi(stack, mask, method=epsilon_method)
                e = res["epsilon"]
                if e == -1:
                    continue
                err = abs(e - target_epsilon)
                if best is None or err < best[0]:
                    best = (
                        err,
                        {
                            "center_y": float(cy),
                            "center_x": float(cx),
                            "half_px": int(half),
                            "epsilon": float(e),
                            "mean_intensity": mi,
                            "n_pixels": int(mask.sum()),
                        },
                    )

    if best is None:
        return None

    # Fine search around best coarse solution
    b = best[1]
    cy0, cx0, half0 = b["center_y"], b["center_x"], b["half_px"]
    for half in range(max(5, half0 - 2), half0 + 3):
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                cy, cx = cy0 + dy, cx0 + dx
                mask = rectangular_mask(mean_image.shape, cy, cx, half, half)
                if mask.sum() < 25:
                    continue
                mi = float(mean_image[mask].mean())
                if not (intensity_lo <= mi <= intensity_hi):
                    continue
                res = epsilon_from_roi(stack, mask, method=epsilon_method)
                e = res["epsilon"]
                if e == -1:
                    continue
                err = abs(e - target_epsilon)
                if err < best[0]:
                    best = (
                        err,
                        {
                            "center_y": float(cy),
                            "center_x": float(cx),
                            "half_px": int(half),
                            "epsilon": float(e),
                            "mean_intensity": mi,
                            "n_pixels": int(mask.sum()),
                        },
                    )
    out = best[1]
    out["epsilon_error"] = float(best[0])
    return out


def _search_array(
    stack: np.ndarray,
    mean_image: np.ndarray,
    nucleus_mask: np.ndarray,
    *,
    target_epsilon: float,
    intensity_lo: float,
    intensity_hi: float,
    epsilon_method: str,
    expand_px: int = 20,
) -> dict | None:
    """
    Array puncta are brighter than nucleoplasm and may sit near/on the nucleus edge.
    Search an expanded box around the nucleus ROI (not restricted to inside it).
    """
    ys, xs = np.where(nucleus_mask)
    if len(xs) == 0:
        return None

    h, w = mean_image.shape
    y0 = max(0, int(ys.min()) - expand_px)
    y1 = min(h - 1, int(ys.max()) + expand_px)
    x0 = max(0, int(xs.min()) - expand_px)
    x1 = min(w - 1, int(xs.max()) + expand_px)

    best: tuple[float, dict] | None = None
    slack = max(0.2 * max(intensity_hi - intensity_lo, 1.0), 3.0)

    for y in range(y0, y1 + 1, 1):
        for x in range(x0, x1 + 1, 1):
            for r in range(2, 12):
                mask = circular_mask(mean_image.shape, y, x, r)
                if mask.sum() < 5:
                    continue
                mi = float(mean_image[mask].mean())
                if not (intensity_lo - slack <= mi <= intensity_hi + slack):
                    continue
                res = epsilon_from_roi(stack, mask, method=epsilon_method)
                e = res["epsilon"]
                if e == -1:
                    continue
                err = abs(e - target_epsilon)
                if best is None or err < best[0]:
                    best = (
                        err,
                        {
                            "center_y": float(y),
                            "center_x": float(x),
                            "radius_px": int(r),
                            "epsilon": float(e),
                            "mean_intensity": mi,
                            "n_pixels": int(mask.sum()),
                        },
                    )

    if best is None:
        # Last resort: match epsilon only (array intensity window can be tight)
        for y in range(y0, y1 + 1, 1):
            for x in range(x0, x1 + 1, 1):
                for r in range(2, 12):
                    mask = circular_mask(mean_image.shape, y, x, r)
                    if mask.sum() < 5:
                        continue
                    res = epsilon_from_roi(stack, mask, method=epsilon_method)
                    e = res["epsilon"]
                    if e == -1:
                        continue
                    err = abs(e - target_epsilon)
                    if best is None or err < best[0]:
                        mi = float(mean_image[mask].mean())
                        best = (
                            err,
                            {
                                "center_y": float(y),
                                "center_x": float(x),
                                "radius_px": int(r),
                                "epsilon": float(e),
                                "mean_intensity": mi,
                                "n_pixels": int(mask.sum()),
                            },
                        )

    if best is None:
        return None
    out = best[1]
    out["epsilon_error"] = float(best[0])
    return out


def fit_cell_rois(
    stack: np.ndarray,
    mean_image: np.ndarray,
    manual_record: dict,
    *,
    epsilon_method: str = "gaussian_hist",
) -> dict | None:
    """Return fitted ROI params for one cell, or None if manually excluded."""
    nuc_man = manual_record["nucleus"]
    arr_man = manual_record["array"]
    if not nuc_man["included"]:
        return None

    nuc = _search_nucleus(
        stack,
        mean_image,
        target_epsilon=float(nuc_man["epsilon"]),
        intensity_lo=float(nuc_man["intensity_min"]),
        intensity_hi=float(nuc_man["intensity_max"]),
        epsilon_method=epsilon_method,
        seed=None,
    )
    if nuc is None:
        return None

    nucleus_mask = rectangular_mask(
        mean_image.shape,
        nuc["center_y"],
        nuc["center_x"],
        nuc["half_px"],
        nuc["half_px"],
    )
    result = {"nucleus": nuc, "array": None}
    if arr_man["included"]:
        arr = _search_array(
            stack,
            mean_image,
            nucleus_mask,
            target_epsilon=float(arr_man["epsilon"]),
            intensity_lo=float(arr_man["intensity_min"]),
            intensity_hi=float(arr_man["intensity_max"]),
            epsilon_method=epsilon_method,
        )
        result["array"] = arr
    return result


def masks_from_fitted(fitted: dict, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    nuc = fitted["nucleus"]
    nucleus_mask = rectangular_mask(
        shape, nuc["center_y"], nuc["center_x"], nuc["half_px"], nuc["half_px"]
    )
    array_mask = np.zeros(shape, dtype=bool)
    if fitted.get("array"):
        arr = fitted["array"]
        array_mask = circular_mask(shape, arr["center_y"], arr["center_x"], arr["radius_px"])
    return nucleus_mask, array_mask


def roi_cache_path(cache_root: Path, session: str, condition: str, cell_id: int) -> Path:
    return cache_root / session / condition / f"{cell_id:02d}.json"


def load_fitted_roi(cache_root: Path, session: str, condition: str, cell_id: int) -> dict | None:
    path = roi_cache_path(cache_root, session, condition, cell_id)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def save_fitted_roi(cache_root: Path, session: str, condition: str, cell_id: int, data: dict) -> Path:
    path = roi_cache_path(cache_root, session, condition, cell_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def fit_session_rois(
    cfg: dict,
    session_key: str,
    *,
    conditions: list[str] | None = None,
    cache_root: Path,
    epsilon_method: str = "gaussian_hist",
) -> pd.DataFrame:
    """Fit and cache ROIs for all manual-included cells in a session."""
    from .manual_reader import read_manual_block
    from .pipeline import load_raw_stack
    from .preprocess import preprocess_stack

    session = cfg["sessions"][session_key]
    session_dir = Path(cfg["raw_data_root"]) / session["folder"]
    manual_path = cfg["manual_workbooks"][session_key]
    rows = []

    for cond in session["conditions"]:
        if conditions and cond["id"] not in conditions:
            continue
        manual = {m["cell_id"]: m for m in read_manual_block(manual_path, cond["manual_sheet_block"])}
        cond_dir = session_dir / cond["subdir"]
        for f in sorted(cond_dir.glob(cond["file_glob"])):
            from .manual_reader import manual_file_index

            cell_id = manual_file_index(f.name)
            if cell_id is None or cell_id not in manual:
                continue
            rec = manual[cell_id]
            raw = load_raw_stack(f)
            stack, _ = preprocess_stack(raw, discard_first=cfg["discard_first_frames"], detrend=cfg.get("detrend", True))
            mean_img = stack.mean(0)

            if not rec["nucleus"]["included"]:
                rows.append(
                    {
                        "session": session_key,
                        "condition": cond["id"],
                        "cell_id": cell_id,
                        "fitted": False,
                        "reason": "manual_excluded",
                    }
                )
                continue

            existing = load_fitted_roi(cache_root, session_key, cond["id"], cell_id)
            if (
                existing
                and existing.get("nucleus")
                and (existing.get("array") or not rec["array"]["included"])
            ):
                rows.append(
                    {
                        "session": session_key,
                        "condition": cond["id"],
                        "cell_id": cell_id,
                        "fitted": True,
                        "nucleus_epsilon_error": existing["nucleus"].get("epsilon_error"),
                        "array_epsilon_error": existing["array"].get("epsilon_error")
                        if existing.get("array")
                        else None,
                        "reason": "cached",
                    }
                )
                continue

            fitted = fit_cell_rois(stack, mean_img, rec, epsilon_method=epsilon_method)
            if fitted is None:
                rows.append(
                    {
                        "session": session_key,
                        "condition": cond["id"],
                        "cell_id": cell_id,
                        "fitted": False,
                        "reason": "fit_failed",
                    }
                )
                continue

            # Re-fit array alone if nucleus cached but array missing (e.g. after algorithm update)
            existing = load_fitted_roi(cache_root, session_key, cond["id"], cell_id)
            if existing and existing.get("nucleus") and not existing.get("array") and rec["array"]["included"]:
                nucleus_mask, _ = masks_from_fitted(existing, mean_img.shape)
                arr = _search_array(
                    stack,
                    mean_img,
                    nucleus_mask,
                    target_epsilon=float(rec["array"]["epsilon"]),
                    intensity_lo=float(rec["array"]["intensity_min"]),
                    intensity_hi=float(rec["array"]["intensity_max"]),
                    epsilon_method=epsilon_method,
                )
                if arr:
                    existing["array"] = arr
                    fitted = existing

            save_fitted_roi(cache_root, session_key, cond["id"], cell_id, fitted)
            nuc_err = fitted["nucleus"]["epsilon_error"]
            arr_err = fitted["array"]["epsilon_error"] if fitted.get("array") else np.nan
            rows.append(
                {
                    "session": session_key,
                    "condition": cond["id"],
                    "cell_id": cell_id,
                    "fitted": True,
                    "nucleus_epsilon_error": nuc_err,
                    "array_epsilon_error": arr_err,
                }
            )

    return pd.DataFrame(rows)