import pandas as pd
import numpy as np
from pathlib import Path
from app.logging_config import get_logger
from app.data_loader import load_feature_data

LOG = get_logger(__name__, filename="drift.log")

FEATURE_STORE = Path("data/processed/feature_store.csv")
BACKUP_PATH = Path("data/processed/feature_store_backup.csv")


def load_store():
    df = load_feature_data()
    if isinstance(df, tuple):
        df = df[0]
    return df


def save_store(df):
    df.to_csv(FEATURE_STORE, index=True)


def backup_store():
    df = load_store()
    BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(BACKUP_PATH, index=True)
    LOG.info("Backup feature store saved.")


def restore_backup():
    if not BACKUP_PATH.exists():
        raise FileNotFoundError("Backup missing. Simulate drift first.")
    df = pd.read_csv(BACKUP_PATH, index_col=0)
    save_store(df)
    LOG.info("Feature store restored from backup.")
    return df


# ----------------------------------------------
# 🔥 INSTANT DRIFT SIMULATION (for demos)
# ----------------------------------------------
def simulate_drift(intensity: float = 0.2):
    """
    intensity: 0.2 = 20% drift
    """
    backup_store()

    df = load_store()

    # 1) Monetary Drift
    df["monetary"] *= (1 + intensity)

    # 2) Total Revenue Drift
    df["total_revenue"] *= (1 + intensity * 0.8)

    # 3) Recency Drift (more recent)
    df["recency"] = df["recency"] * (1 - intensity * 0.5)

    # 4) Reduce channel diversity
    df["channel_diversity"] *= (1 - intensity)

    # 5) Increase churn
    df["is_churned"] = np.where(
        np.random.rand(len(df)) < (0.1 + intensity), 1, df["is_churned"]
    )

    # 6) Modify cart conversion
    df["cart_conversion_rate"] *= (1 + intensity)

    save_store(df)
    LOG.info(f"Simulated drift with intensity {intensity}.")
    return df


# ----------------------------------------------
# 🌀 LONG-TERM NATURAL DRIFT (Option B)
# ----------------------------------------------
def evolve_synthetic_data():
    df = load_store()

    # monthly upward revenue drift
    df["monetary"] *= np.random.normal(1.01, 0.02)

    # seasonality
    df["total_revenue"] *= np.random.normal(1.02, 0.03)

    # churn evolution
    df["is_churned"] = np.where(
        np.random.rand(len(df)) < 0.05, 1, df["is_churned"]
    )

    # product category shift
    df["category_diversity"] *= np.random.normal(0.95, 0.03)

    save_store(df)
    LOG.info("Synthetic data evolved.")
    return df