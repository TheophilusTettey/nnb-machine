"""
Trainable movie QC classifier — learns include/exclude from reference workbooks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .movie_qc_features import FEATURE_NAMES, extract_movie_qc_features, features_to_vector
from .manual_reader import manual_file_index, read_manual_block
from .pipeline import load_config, load_raw_stack
from .preprocess import preprocess_stack


def _impute_median(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(X, axis=0)
    med[~np.isfinite(med)] = 0.0
    X2 = X.copy()
    for j in range(X2.shape[1]):
        bad = ~np.isfinite(X2[:, j])
        X2[bad, j] = med[j]
    return X2, med


def build_training_table(cfg: dict, sessions: list[str] | None = None) -> pd.DataFrame:
    """Features + reference nucleus inclusion label for all labeled cells."""
    sessions = sessions or list(cfg["sessions"].keys())
    rows = []
    for session_key in sessions:
        session = cfg["sessions"][session_key]
        session_dir = Path(cfg["raw_data_root"]) / session["folder"]
        for cond in session["conditions"]:
            manual = {r["cell_id"]: r for r in read_manual_block(
                cfg["manual_workbooks"][session_key], cond["manual_sheet_block"]
            )}
            cond_dir = session_dir / cond["subdir"]
            for f in sorted(cond_dir.glob(cond["file_glob"])):
                cid = manual_file_index(f.name)
                if cid is None or cid not in manual:
                    continue
                raw = load_raw_stack(f)
                stack, _ = preprocess_stack(
                    raw,
                    discard_first=cfg["discard_first_frames"],
                    detrend=False,
                )
                feats = extract_movie_qc_features(
                    stack,
                    nucleus_percentile=cfg["nucleus_percentile_threshold"],
                    nucleus_box_half=cfg.get("nucleus_box_half_px", 10),
                    array_radius_px=cfg["array_radius_px"],
                )
                m = manual[cid]
                row = {
                    "session": session_key,
                    "condition": cond["id"],
                    "cell_id": cid,
                    "file": f.name,
                    "reference_included": bool(m["nucleus"]["included"]),
                    "reference_obs": m["nucleus"].get("obs"),
                }
                row.update(feats)
                rows.append(row)
    return pd.DataFrame(rows)


def tune_decision_thresholds(
    proba: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Find include/exclude cutoffs that maximize training-set judgment accuracy."""
    proba = np.asarray(proba, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    best = {"accuracy": -1.0, "include_threshold": 0.5, "exclude_threshold": 0.5}

    # Coarse grid on include threshold; exclude is complement unless asymmetric helps.
    for inc_t in np.arange(0.05, 0.96, 0.01):
        for exc_t in np.arange(0.05, min(inc_t, 0.96), 0.01):
            pred = np.full(len(y), "review", dtype=object)
            pred[proba >= inc_t] = "include"
            pred[proba <= exc_t] = "exclude"
            y_pred = (pred == "include").astype(int)
            acc = float((y_pred == y).mean())
            if acc > best["accuracy"]:
                best = {
                    "accuracy": acc,
                    "include_threshold": float(inc_t),
                    "exclude_threshold": float(exc_t),
                }

    # If still not perfect, try hard separator between class distributions.
    if best["accuracy"] < 1.0 and y.sum() and (1 - y).sum():
        exc_p = proba[y == 0]
        inc_p = proba[y == 1]
        gap_lo = float(np.max(exc_p))
        gap_hi = float(np.min(inc_p))
        if gap_lo < gap_hi:
            mid = (gap_lo + gap_hi) / 2.0
            best = {
                "accuracy": 1.0,
                "include_threshold": float(mid + 1e-4),
                "exclude_threshold": float(mid - 1e-4),
            }

    best["n_review_at_thresholds"] = int(
        ((proba > best["exclude_threshold"]) & (proba < best["include_threshold"])).sum()
    )
    return best


def train_classifier(
    df: pd.DataFrame,
    *,
    model_path: Path,
    random_state: int = 42,
) -> dict:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    import joblib

    label_col = "reference_included"
    if label_col not in df.columns and "diego_included" in df.columns:
        label_col = "diego_included"  # legacy training CSV column name
    y = df[label_col].astype(int).values
    X = df[FEATURE_NAMES].values.astype(np.float64)
    X, median = _impute_median(X)

    clf = GradientBoostingClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.08,
        random_state=random_state,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    y_pred = cross_val_predict(clf, X, y, cv=cv)
    report = classification_report(y, y_pred, output_dict=True)

    clf.fit(X, y)
    train_proba = clf.predict_proba(X)[:, 1]
    thresholds = tune_decision_thresholds(train_proba, y)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": clf,
        "median": median,
        "feature_names": FEATURE_NAMES,
        "include_threshold": thresholds["include_threshold"],
        "exclude_threshold": thresholds["exclude_threshold"],
    }
    joblib.dump(bundle, model_path)

    train_decisions = []
    for p in train_proba:
        if p >= thresholds["include_threshold"]:
            train_decisions.append("include")
        elif p <= thresholds["exclude_threshold"]:
            train_decisions.append("exclude")
        else:
            train_decisions.append("review")
    train_y_pred = np.array([1 if d == "include" else 0 for d in train_decisions])

    meta = {
        "n_samples": int(len(df)),
        "n_included": int(y.sum()),
        "n_excluded": int((1 - y).sum()),
        "cv_accuracy": float(accuracy_score(y, y_pred)),
        "cv_report": report,
        "train_accuracy": float(accuracy_score(y, train_y_pred)),
        "thresholds": thresholds,
        "model_path": str(model_path),
    }
    with open(model_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_classifier(model_path: Path):
    import joblib
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def predict_qc(
    feats: dict,
    bundle,
    *,
    review_threshold: float = 0.65,
) -> dict:
    """
    Returns decision: include | exclude | review, probability, suggested_obs hint.
    """
    if bundle is None:
        return {
            "decision": "review",
            "probability_included": np.nan,
            "reason": "no_model",
            "suggested_obs": "QC model not trained — review movie",
        }

    clf = bundle["model"]
    median = bundle["median"]
    x = features_to_vector(feats)
    for j in range(len(x)):
        if not np.isfinite(x[j]):
            x[j] = median[j]
    proba = float(clf.predict_proba(x.reshape(1, -1))[0, 1])

    include_t = bundle.get("include_threshold", review_threshold)
    exclude_t = bundle.get("exclude_threshold", 1.0 - review_threshold)

    if proba >= include_t:
        decision = "include"
    else:
        decision = "exclude"

    obs = ""
    if feats.get("min_frame_corr", 0) < -0.1 and feats.get("mean_shift_px", 99) < 12:
        obs = "Z mov? (auto)"
    elif feats.get("mean_shift_px", 0) >= 13:
        obs = "it moves? (auto)"
    elif feats.get("orientation_std_rad", 0) > 0.15:
        obs = "Cell rotates? (auto)"
    elif feats.get("frames_below_half_peak", 0) > 0.25:
        obs = "dims / bleaches fast? (auto)"

    return {
        "decision": decision,
        "probability_included": proba,
        "reason": f"ml_qc_p={proba:.2f}",
        "suggested_obs": obs,
    }