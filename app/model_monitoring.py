import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ks_2samp

from app.logging_config import get_logger
from app.clv_model import load_clv_model
from app.data_loader import load_feature_data

LOG = get_logger(__name__, filename="monitoring.log")

BASELINE_PATH = Path("data/processed/baseline_feature_store.csv")


# ============================
# SAFE LOADERS
# ============================
def safe_load_feature_store():
    df = load_feature_data()
    if isinstance(df, tuple):   # ensure DF
        df = df[0]
    return df


def load_baseline_snapshot():
    if not BASELINE_PATH.exists():
        LOG.warning("Baseline missing → auto-creating baseline snapshot")
        df = safe_load_feature_store()
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(BASELINE_PATH, index=True)

    df = pd.read_csv(BASELINE_PATH, index_col=0)
    return df


def save_baseline_snapshot():
    df = safe_load_feature_store()
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(BASELINE_PATH, index=True)


# ============================
# FEATURE ALIGNMENT
# ============================
def clean_columns(df):
    """Remove unwanted extra columns."""
    bad_cols = [c for c in df.columns if "customer_id" in c and c != "customer_id"]
    if bad_cols:
        df = df.drop(columns=bad_cols, errors="ignore")
    return df


def align(df, feature_cols):
    df = clean_columns(df)

    # Add missing
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    # Drop extra
    df = df[feature_cols].copy()

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    return df


# ============================
# PSI
# ============================
def calculate_psi(base, curr, buckets=10):
    base = np.array(base)
    curr = np.array(curr)

    breakpoints = np.linspace(0, 100, buckets + 1)

    psi_val = 0
    for i in range(buckets):
        br = np.percentile(base, breakpoints[i:i+2])
        cr = np.percentile(curr, breakpoints[i:i+2])

        b_perc = ((base >= br[0]) & (base <= br[1])).mean()
        c_perc = ((curr >= cr[0]) & (curr <= cr[1])).mean()

        b_perc = max(b_perc, 0.0001)
        c_perc = max(c_perc, 0.0001)

        psi_val += (b_perc - c_perc) * np.log(b_perc / c_perc)

    return psi_val


# ============================
# PREDICTION DRIFT
# ============================
def prediction_drift():
    try:
        baseline = clean_columns(load_baseline_snapshot())
        current = clean_columns(safe_load_feature_store())

        model, scaler = load_clv_model()

        # feature names from scaler (MOST RELIABLE)
        if hasattr(scaler, "feature_names_in_"):
            feature_cols = list(scaler.feature_names_in_)
        else:
            exclude = {"customer_id", "future_clv", "persona", "subscription_plan"}
            feature_cols = [c for c in baseline.columns if c not in exclude]

        # align
        bX = align(baseline, feature_cols)
        cX = align(current, feature_cols)

        # scale + predict
        b_pred = model.predict(scaler.transform(bX))
        c_pred = model.predict(scaler.transform(cX))

        psi = calculate_psi(b_pred, c_pred)
        ks_stat, ks_p = ks_2samp(b_pred, c_pred)

        drift_state = "Drift" if psi > 0.2 or ks_p < 0.05 else "No Drift"

        return {
            "psi": float(psi),
            "ks_p_value": float(ks_p),
            "drift": drift_state,
        }

    except Exception as e:
        LOG.exception("Prediction drift failed")
        return {
            "psi": None,
            "ks_p_value": None,
            "drift": "Error",
            "error": str(e)
        }


# ============================
# FEATURE DRIFT
# ============================
def monitor_drift():
    try:
        baseline = clean_columns(load_baseline_snapshot())
        current = clean_columns(safe_load_feature_store())

        rows = []
        exclude = {"customer_id", "future_clv", "persona", "subscription_plan"}

        for col in baseline.columns:
            if col in exclude:
                continue
            if col not in current:
                continue

            b = pd.to_numeric(baseline[col], errors="coerce").fillna(0)
            c = pd.to_numeric(current[col], errors="coerce").fillna(0)

            psi = calculate_psi(b, c)
            ks_stat, ks_p = ks_2samp(b, c)

            drift_flag = (psi > 0.2) or (ks_p < 0.05)

            rows.append({
                "feature": col,
                "psi": float(psi),
                "ks_stat": float(ks_stat),
                "ks_p_value": float(ks_p),
                "drift": "Drift" if drift_flag else "No Drift",
            })

        return pd.DataFrame(rows)

    except Exception as e:
        LOG.exception("Feature drift failed")
        return pd.DataFrame({
            "feature": ["Error"],
            "psi": [None],
            "ks_stat": [None],
            "ks_p_value": [None],
            "drift": [str(e)]
        })