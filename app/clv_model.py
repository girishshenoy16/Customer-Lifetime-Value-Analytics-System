from typing import Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.data_loader import load_feature_data
from app.utils import get_models_path 
try:
    from xgboost import XGBRegressor
except ImportError as e:
    raise ImportError("xgboost is not installed. Install it via `pip install xgboost`.") from e

# ====================================
# ADDED: Logging Integration
# ====================================
from app.logging_config import get_logger
LOG = get_logger(__name__, filename="retrain.log")
# ====================================

# Features from feature_store used as CLV predictors
FEATURE_COLS = [
    "recency",
    "frequency",
    "monetary",
    "tenure",
    "total_revenue",
    "category_diversity",
    "channel_diversity",
    "cross_sell_rate",
    "cart_conversion_rate",
    "is_subscriber",
    "is_churned",
]


def build_clv_features() -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Build X, y for CLV model from the processed feature_store.
    Assumes feature_store.csv has `future_clv`.
    """
    LOG.info("Building CLV features from feature_store...")

    feature_df = load_feature_data()

    missing = [c for c in FEATURE_COLS if c not in feature_df.columns]
    if missing:
        LOG.error(f"Missing CLV feature columns: {missing}")
        raise ValueError(f"Missing CLV feature columns in feature_store: {missing}")

    if "future_clv" not in feature_df.columns:
        LOG.error("`future_clv` missing from feature_store.csv")
        raise ValueError("Missing target column `future_clv`. Run preprocess_all().")

    X = feature_df[FEATURE_COLS].copy()
    y_raw = feature_df["future_clv"].values.astype(float)

    LOG.info(f"Prepared CLV features for {len(feature_df):,} customers.")
    return X, y_raw


def train_clv_model(
    test_size: float = 0.2,
    random_state: int = 42,
    save: bool = True,
):
    """
    Train XGBoost CLV model on log-transformed target.
    Returns model, scaler, mae, rmse, r2
    """
    LOG.info("Starting CLV model training...")

    X, y_raw = build_clv_features()

    y_log = np.log1p(y_raw)  # log-transform target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split happens on SCALED numpy arrays
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X_scaled, y_raw, test_size=test_size, random_state=random_state
    )

    y_train_log = np.log1p(y_train_raw)

    # ================================================
    # 👇 NEW: Convert scaled arrays → DataFrames
    # Ensures XGBoost stores feature_names_in_
    # ================================================
    X_train_df = pd.DataFrame(X_train, columns=FEATURE_COLS)
    X_test_df = pd.DataFrame(X_test, columns=FEATURE_COLS)

    model = XGBRegressor(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.07,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        objective="reg:squarederror",
    )

    LOG.info("Fitting XGBoost model on DataFrame (feature names preserved)...")

    # 👇 USE DATAFRAME, NOT NUMPY ARRAY
    model.fit(X_train_df, y_train_log)

    # Predict ALSO with DataFrame
    y_pred_log = model.predict(X_test_df)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.maximum(y_pred, 0.0)


    mae = mean_absolute_error(y_test_raw, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred))
    r2 = r2_score(y_test_raw, y_pred)

    LOG.info(f"CLV model metrics → MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

    print("\n=== CLV MODEL METRICS ===")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}")

    if save:
        models_dir = get_models_path()
        models_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, models_dir / "clv_xgboost.pkl")
        joblib.dump(scaler, models_dir / "scaler.pkl")

        LOG.info(f"Saved model → {models_dir / 'clv_xgboost.pkl'}")
        LOG.info(f"Saved scaler → {models_dir / 'scaler.pkl'}")

        print(f"[OK] Saved model  → {models_dir / 'clv_xgboost.pkl'}")
        print(f"[OK] Saved scaler → {models_dir / 'scaler.pkl'}")

    return model, scaler, mae, rmse, r2


def load_clv_model():
    """Load trained CLV model and scaler."""
    LOG.info("Loading CLV model and scaler...")

    models_dir = get_models_path()
    model_path = models_dir / "clv_xgboost.pkl"
    scaler_path = models_dir / "scaler.pkl"

    if not model_path.exists() or not scaler_path.exists():
        LOG.error("Model or scaler missing in models directory.")
        raise FileNotFoundError(
            f"Missing CLV model or scaler.\n"
            f"Expected: {model_path} and {scaler_path}"
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    LOG.info("CLV model and scaler loaded successfully.")
    return model, scaler


def score_customers(feature_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Score customers using trained CLV model.
    """
    LOG.info("Scoring customers for CLV prediction...")

    if feature_df is None:
        LOG.info("feature_df not provided → loading feature_store...")
        feature_df = load_feature_data().copy()
    else:
        feature_df = feature_df.copy()

    missing = [c for c in FEATURE_COLS if c not in feature_df.columns]
    if missing:
        LOG.error(f"Missing feature columns for scoring: {missing}")
        raise ValueError(f"Missing CLV feature columns in provided data: {missing}")

    model, scaler = load_clv_model()

    X_scaled = scaler.transform(feature_df[FEATURE_COLS])
    X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLS)

    clv_log = model.predict(X_scaled_df)
    clv_raw = np.expm1(clv_log)
    clv_raw = np.maximum(clv_raw, 0.0)

    feature_df["predicted_clv"] = clv_raw

    LOG.info(f"Finished scoring {len(feature_df):,} customers.")
    return feature_df


if __name__ == "__main__":
    train_clv_model()