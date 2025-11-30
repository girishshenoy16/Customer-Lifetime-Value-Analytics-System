from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from .clv_model import (
    build_clv_features,
    load_clv_model,
    FEATURE_COLS,
)

from app.logging_config import get_logger


# ==================================
# ADDED: Logging Integration
# ==================================
LOG = get_logger(__name__, filename="monitoring.log")
# ==================================


def evaluate_clv_model(
    shap_sample_frac: float = 0.3,
    random_state: int = 42,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full evaluation pipeline for the CLV model.

    The CLV model was trained on log-transformed CLV.
    Here we:
        - Predict CLV in log-space
        - Convert back to ₹ currency using expm1()
        - Compute MAE, RMSE, R² on the real currency values
        - Build evaluation dataframe (actual/predicted/residual)
        - Return model-based feature importance
        - Return SHAP-based importance (if SHAP is available)

    Returns:
        metrics     : dict with MAE, RMSE, R²
        eval_df     : DataFrame with y_true, y_pred, residual
        fi_df       : feature importances from the model
        shap_df     : SHAP mean(|impact|) values per feature (empty if SHAP unavailable)
    """

    LOG.info("Starting CLV model evaluation...")

    # --------------------
    # Load data & model
    # --------------------
    LOG.info("Loading CLV features and trained model...")
    X, y_true = build_clv_features()
    model, scaler = load_clv_model()

    X_scaled = scaler.transform(X)

    # --------------------
    # Predict: log-space → ₹ real scale
    # --------------------
    LOG.info("Generating CLV predictions...")
    y_pred_log = model.predict(X_scaled)
    y_pred = np.expm1(y_pred_log)

    # Ensure no negative CLV
    y_pred = np.maximum(y_pred, 0.0)

    # --------------------
    # Metrics on original scale
    # --------------------
    LOG.info("Computing evaluation metrics...")
    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }

    LOG.info(
        f"Evaluation metrics → MAE={metrics['MAE']:.4f}, "
        f"RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}"
    )

    # --------------------
    # Evaluation DataFrame
    # --------------------
    eval_df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    eval_df["residual"] = eval_df["y_true"] - eval_df["y_pred"]

    LOG.info("Built evaluation dataframe with residuals.")

    # -------------------------
    # Model-based Feature Importance
    # -------------------------
    LOG.info("Extracting model feature importances...")
    try:
        importances = model.feature_importances_
        fi_df = pd.DataFrame(
            {"feature": FEATURE_COLS, "importance": importances}
        )
        # Normalize importance
        fi_df["importance"] = fi_df["importance"] / fi_df["importance"].sum()
        fi_df = fi_df.sort_values("importance", ascending=False)
        LOG.info("Model feature importance extracted successfully.")
    except Exception as e:
        LOG.warning("Model does not provide feature_importances_", exc_info=True)
        fi_df = pd.DataFrame(columns=["feature", "importance"])

    # -------------------------
    # SHAP Explainability (Optional)
    # -------------------------
    LOG.info("Attempting SHAP explainability...")
    try:
        import shap  # type: ignore

        n = X_scaled.shape[0]
        sample_size = max(100, int(n * shap_sample_frac))
        sample_size = min(sample_size, n)

        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=sample_size, replace=False)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled[idx])
        shap_mean_abs = np.abs(shap_values).mean(axis=0)

        shap_df = pd.DataFrame(
            {"feature": FEATURE_COLS, "mean_abs_shap": shap_mean_abs}
        ).sort_values("mean_abs_shap", ascending=False)

        LOG.info("SHAP values computed successfully.")

    except Exception:
        LOG.warning("SHAP unavailable, returning empty shap_df.", exc_info=True)
        shap_df = pd.DataFrame(columns=["feature", "mean_abs_shap"])

    LOG.info("CLV model evaluation completed.")
    return metrics, eval_df, fi_df, shap_df


if __name__ == "__main__":
    metrics, eval_df, fi_df, shap_df = evaluate_clv_model()

    print("\n=== MODEL EVALUATION ===")
    print(metrics)

    print("\n=== FEATURE IMPORTANCE ===")
    print(fi_df)

    print("\n=== SHAP SUMMARY (TOP 10) ===")
    print(shap_df.head(10))