
from pathlib import Path
import logging
from datetime import datetime

from app.model_monitoring import prediction_drift, save_baseline_snapshot, safe_load_feature_store
from app.clv_model import train_clv_model, load_clv_model
from app.model_registry import register_model, list_versions, get_active_version
from app.logging_config import get_logger as _get_logger

"""
Automated retraining orchestration:
- checks drift using model_monitoring.prediction_drift()
- retrains if thresholds exceeded
- registers new model & updates baseline snapshot
"""

# thresholds (tunable)
PSI_THRESHOLD = 0.25
KS_PVALUE_THRESHOLD = 0.05

LOG = logging.getLogger("auto_retrain")
LOG.setLevel(logging.INFO)

# ===========================================
# ADDED — FULL DIAGNOSTIC LOGGING (Depth C)
# ===========================================
LOG = _get_logger(__name__, filename="retrain.log")
LOG.info("auto_retrain.py loaded (retrain system initialized)")
# ===========================================


def should_retrain():
    """
    Returns (bool, reason, drift_report)
    """
    LOG.info("Running drift check (should_retrain)")

    try:
        drift = prediction_drift()
        LOG.debug(f"Drift report: {drift}")
    except Exception as e:
        LOG.exception(f"Prediction drift check failed: {e}")
        return True, f"drift_check_error:{e}", {}

    psi = drift.get("psi", 0.0)
    ks_p = drift.get("ks_p_value", 1.0)

    reasons = []
    retrain = False

    if psi > PSI_THRESHOLD:
        retrain = True
        reasons.append(f"psi={psi:.4f} > {PSI_THRESHOLD}")

    if ks_p < KS_PVALUE_THRESHOLD:
        retrain = True
        reasons.append(f"ks_p={ks_p:.6f} < {KS_PVALUE_THRESHOLD}")

    reason_text = "; ".join(reasons) if reasons else "No significant drift"

    LOG.info(f"Retrain decision: {retrain}, Reason: {reason_text}")
    return retrain, reason_text, drift


def retrain_if_needed(force: bool = False):
    """
    Main entry:
    - checks drift
    - retrains if needed or forced
    - registers new model and saves baseline snapshot
    """
    LOG.info(f"Retrain check triggered (force={force})")

    try:
        retrain, reason, drift = should_retrain()
    except Exception as e:
        LOG.exception("Failed to compute drift", exc_info=True)
        retrain, reason, drift = (True, f"drift_check_error:{e}", {})

    if not retrain and not force:
        LOG.info(f"No retrain needed: {reason}")
        return {"retrained": False, "reason": reason, "drift": drift}

    LOG.warning(f"Retraining triggered → Reason: {reason}")

    # train model (returns model, scaler, mae, rmse, r2) — allow flexible returns
    try:
        result = train_clv_model()
        LOG.info("Model training completed inside auto_retrain")
    except Exception as e:
        LOG.exception(f"Model training failed: {e}")
        raise

    # flexible unpacking
    model = result[0]
    scaler = result[1]
    metrics = {}

    if len(result) >= 4:
        try:
            if len(result) == 4:
                metrics = {"MAE": float(result[2]), "R2": float(result[3])}
            elif len(result) >= 5:
                metrics = {
                    "MAE": float(result[2]),
                    "RMSE": float(result[3]),
                    "R2": float(result[4]),
                }
            LOG.info(f"Captured training metrics: {metrics}")
        except Exception as e:
            LOG.warning(f"Could not parse metrics: {e}")
            metrics = {}

    # register model
    try:
        version = register_model(
            model,
            scaler,
            metrics=metrics,
            tags={"auto_retrain": True, "trigger": reason, "timestamp": datetime.utcnow().isoformat() + "Z"},
        )
        LOG.info(f"Registered new model version: v{version}")
    except Exception as e:
        LOG.exception(f"Failed to register model: {e}")
        raise

    # update baseline snapshot to latest
    try:
        save_baseline_snapshot()
        LOG.info("Baseline snapshot updated successfully")
    except Exception as e:
        LOG.exception("Failed to save baseline snapshot", exc_info=True)

    LOG.info(f"Retrain complete → New model version: v{version}")
    return {
        "retrained": True,
        "version": version,
        "metrics": metrics,
        "reason": reason,
        "drift": drift
    }