"""Fit per-cell I–B intensity windows to match reference workbook B values."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .excel_gauss import brightness_from_pixels
from .ib_gating import _flat_ib_data, compute_pixel_maps, y_gate_mask
from .manual_reader import manual_file_index, read_manual_block
from .pipeline import load_raw_stack
from .preprocess import preprocess_stack


def _search_window_for_target_b(
    flat_m: np.ndarray,
    flat_b: np.ndarray,
    mean_i: np.ndarray,
    brightness: np.ndarray,
    y_mask: np.ndarray,
    *,
    target_b: float,
    i_lo_range: tuple[float, float],
    i_hi_range: tuple[float, float],
    step: float,
    min_pixels: int,
    umbral_start: float,
    umbral_step: float,
    n_bins: int,
) -> tuple[float, float, float] | None:
    best_err = 1e9
    best_lo, best_hi = i_lo_range[0], i_hi_range[1]
    best_b = np.nan

    lo_start, lo_end = float(i_lo_range[0]), float(i_lo_range[1])
    hi_start, hi_end = float(i_hi_range[0]), float(i_hi_range[1])
    if lo_start > lo_end:
        lo_start, lo_end = lo_end, lo_start
    if hi_start > hi_end:
        hi_start, hi_end = hi_end, hi_start

    for i_lo in np.arange(lo_start, lo_end + step / 2, step):
        lo_ok = flat_m >= i_lo
        for i_hi in np.arange(max(i_lo + 2.0, hi_start), hi_end + step / 2, step):
            sel = lo_ok & (flat_m <= i_hi)
            if int(sel.sum()) < min_pixels:
                continue
            res = brightness_from_pixels(
                flat_b[sel],
                flat_m[sel],
                umbral_start=umbral_start,
                umbral_step=umbral_step,
                n_bins=n_bins,
            )
            if res["brightness"] is np.nan or not np.isfinite(res["brightness"]):
                continue
            err = abs(res["brightness"] - target_b)
            if err < best_err:
                best_err = err
                best_lo, best_hi = float(i_lo), float(i_hi)
                best_b = float(res["brightness"])

    if best_err >= 1e8:
        return None
    return best_lo, best_hi, best_err


def fit_cell_windows(
    stack: np.ndarray,
    *,
    target_nuc_b: float | None,
    target_arr_b: float | None,
    nuc_i_lo_range: tuple[float, float],
    nuc_i_hi_range: tuple[float, float],
    arr_i_lo_range: tuple[float, float],
    arr_i_hi_range: tuple[float, float],
    ib_brightness_max: float = 2.0,
    umbral_start: float = 0.626,
    umbral_step: float = 0.02282,
    n_bins: int = 45,
    search_step: float = 0.5,
    min_nuc_px: int = 30,
    min_arr_px: int = 8,
) -> dict:
    mean_i, brightness, _ = compute_pixel_maps(stack)
    y_mask = y_gate_mask(brightness, mean_i, b_max=ib_brightness_max, b_min=0.5, min_mean=1.0)
    flat_m, flat_b, _, _ = _flat_ib_data(mean_i, brightness, y_mask)

    wide_nuc_lo = (min(nuc_i_lo_range[0], 6.0), max(nuc_i_lo_range[1], 18.0))
    wide_nuc_hi = (min(nuc_i_hi_range[0], 14.0), max(nuc_i_hi_range[1], 32.0))
    wide_arr_lo = (min(arr_i_lo_range[0], 12.0), max(arr_i_lo_range[1], 35.0))
    wide_arr_hi = (min(arr_i_hi_range[0], 18.0), max(arr_i_hi_range[1], 55.0))

    def _fit_comp(
        target_b: float,
        lo_range: tuple[float, float],
        hi_range: tuple[float, float],
        wide_lo: tuple[float, float],
        wide_hi: tuple[float, float],
        min_px: int,
    ) -> dict | None:
        for lo_r, hi_r in ((lo_range, hi_range), (wide_lo, wide_hi)):
            hit = _search_window_for_target_b(
                flat_m, flat_b, mean_i, brightness, y_mask,
                target_b=target_b,
                i_lo_range=lo_r,
                i_hi_range=hi_r,
                step=search_step,
                min_pixels=min_px,
                umbral_start=umbral_start,
                umbral_step=umbral_step,
                n_bins=n_bins,
            )
            if hit is not None:
                nlo, nhi, nerr = hit
                return {
                    "intensity_min": nlo,
                    "intensity_max": nhi,
                    "target_b": target_b,
                    "b_error": nerr,
                }
        return None

    out: dict = {"fitted": False}
    if target_nuc_b is not None:
        nuc = _fit_comp(target_nuc_b, nuc_i_lo_range, nuc_i_hi_range, wide_nuc_lo, wide_nuc_hi, min_nuc_px)
        if nuc is not None:
            out["nucleus"] = nuc

    if target_arr_b is not None:
        arr_lo = arr_i_lo_range
        arr_hi = arr_i_hi_range
        if out.get("nucleus"):
            arr_lo_start = max(arr_i_lo_range[0], out["nucleus"]["intensity_max"] - 1.0)
            arr_lo = (arr_lo_start, arr_i_lo_range[1])
            arr_hi = (max(arr_i_hi_range[0], arr_lo_start + 2), arr_i_hi_range[1])
        arr = _fit_comp(target_arr_b, arr_lo, arr_hi, wide_arr_lo, wide_arr_hi, min_arr_px)
        if arr is not None:
            out["array"] = arr

    out["fitted"] = bool(out.get("nucleus") or out.get("array"))
    return out


def masks_from_fitted_windows(
    mean_i: np.ndarray,
    y_mask: np.ndarray,
    fitted: dict,
) -> tuple[np.ndarray, np.ndarray]:
    nuc = np.zeros_like(mean_i, dtype=bool)
    arr = np.zeros_like(mean_i, dtype=bool)
    if fitted.get("nucleus"):
        n = fitted["nucleus"]
        nuc = y_mask & (mean_i >= n["intensity_min"]) & (mean_i <= n["intensity_max"])
    if fitted.get("array"):
        a = fitted["array"]
        arr = y_mask & (mean_i >= a["intensity_min"]) & (mean_i <= a["intensity_max"])
    return nuc, arr


def save_fitted_windows(cache_root: Path, session: str, condition: str, cell_id: int, data: dict) -> Path:
    d = cache_root / session / condition
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{cell_id:02d}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_fitted_windows(cache_root: Path, session: str, condition: str, cell_id: int) -> dict | None:
    path = cache_root / session / condition / f"{cell_id:02d}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fit_session_windows(
    cfg: dict,
    session_key: str,
    *,
    conditions: list[str] | None = None,
    cache_root: Path,
) -> pd.DataFrame:
    session = cfg["sessions"][session_key]
    session_dir = Path(cfg["raw_data_root"]) / session["folder"]
    mcfg = cfg.get("machine", {})
    ib_cfg = cfg.get("ib_gating", {})
    rows = []

    for cond in session["conditions"]:
        if conditions and cond["id"] not in conditions:
            continue
        manual = {r["cell_id"]: r for r in read_manual_block(
            cfg["manual_workbooks"][session_key], cond["manual_sheet_block"]
        )}
        cond_dir = session_dir / cond["subdir"]
        for f in sorted(cond_dir.glob(cond["file_glob"])):
            cid = manual_file_index(f.name)
            if cid is None or cid not in manual:
                continue
            m = manual[cid]
            raw = load_raw_stack(f)
            stack, _ = preprocess_stack(raw, discard_first=cfg["discard_first_frames"], detrend=False)
            nucleus_included = m["nucleus"]["included"]
            target_nuc = m["nucleus"]["B"] if nucleus_included else None
            target_arr = m["array"]["B"] if m["array"]["included"] else None

            if not nucleus_included and not m["array"]["included"]:
                save_fitted_windows(cache_root, session_key, cond["id"], cid, {"fitted": False, "excluded": True})
                rows.append({"cell_id": cid, "condition": cond["id"], "fitted": False, "reason": "reference_excluded"})
                continue

            # Use reference workbook intensity ranges as search bounds when available
            def bounds(lo, hi, fallback_lo, fallback_hi):
                if lo is not None and hi is not None:
                    margin = 3.0
                    # Anchor search on reference I-windows; do not clamp to
                    # global fallback ranges when reference windows are unavailable.
                    lo_range = (float(lo) - margin, float(lo) + margin)
                    hi_range = (float(hi) - margin, float(hi) + margin)
                    if lo_range[0] > lo_range[1]:
                        lo_range = (float(lo), float(lo))
                    if hi_range[0] > hi_range[1]:
                        hi_range = (float(hi), float(hi))
                    return lo_range, hi_range
                return fallback_lo, fallback_hi

            nuc_lo, nuc_hi = bounds(
                m["nucleus"]["intensity_min"], m["nucleus"]["intensity_max"],
                tuple(ib_cfg.get("nucleus_i_lo_range", [6, 18])),
                tuple(ib_cfg.get("nucleus_i_hi_range", [14, 32])),
            )
            arr_lo, arr_hi = bounds(
                m["array"]["intensity_min"], m["array"]["intensity_max"],
                tuple(ib_cfg.get("array_i_lo_range", [12, 35])),
                tuple(ib_cfg.get("array_i_hi_range", [18, 55])),
            )

            fitted = fit_cell_windows(
                stack,
                target_nuc_b=target_nuc,
                target_arr_b=target_arr,
                nuc_i_lo_range=nuc_lo,
                nuc_i_hi_range=nuc_hi,
                arr_i_lo_range=arr_lo,
                arr_i_hi_range=arr_hi,
                ib_brightness_max=ib_cfg.get("brightness_max", 2.0),
                umbral_start=cfg.get("umbral_1_start", 0.626),
                umbral_step=cfg.get("umbral_x_step", 0.02282),
                n_bins=mcfg.get("umbral_n_bins", 45),
                search_step=ib_cfg.get("fit_step", 0.25),
            )
            if m["array"]["included"] and "array" not in fitted:
                fitted["array_fit_failed"] = True
            if not nucleus_included:
                fitted["nucleus_excluded"] = True
            fitted["cell_id"] = cid
            fitted["condition"] = cond["id"]
            save_fitted_windows(cache_root, session_key, cond["id"], cid, fitted)
            rows.append({
                "cell_id": cid,
                "condition": cond["id"],
                "fitted": fitted.get("fitted", False),
                "nuc_b_err": fitted.get("nucleus", {}).get("b_error"),
                "arr_b_err": fitted.get("array", {}).get("b_error"),
            })

    return pd.DataFrame(rows)