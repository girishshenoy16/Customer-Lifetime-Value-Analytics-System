from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from app.data_loader import load_transactions
from app.logging_config import get_logger


# ========================================
# ADDED: Logging Integration
# ========================================
LOG = get_logger(__name__, filename="app.log")
# ========================================

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


def build_rfm_dataset(
    tx: Optional[pd.DataFrame] = None,
    snapshot_date: Optional[pd.Timestamp] = None,
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build RFM dataset from transactions.

    Returns:
        rfm_df: one row per customer with R, F, M, tenure, total_revenue, etc.
        tx:    the transactions used.
    """
    LOG.info("Starting build_rfm_dataset()")

    if tx is None:
        LOG.info("No transactions passed. Loading from data_loader...")
        tx = load_transactions().copy()

    if snapshot_date is None:
        snapshot_date = tx["order_date"].max() + pd.Timedelta(days=1)
        LOG.info(f"Snapshot date automatically chosen as: {snapshot_date}")

    # aggregate
    LOG.info("Computing R, F, M aggregates...")

    recency = (snapshot_date - tx.groupby("customer_id")["order_date"].max()).dt.days
    frequency = tx.groupby("customer_id")["order_id"].nunique()
    monetary = tx.groupby("customer_id")["amount"].mean()
    tenure = (
        tx.groupby("customer_id")["order_date"].max()
        - tx.groupby("customer_id")["order_date"].min()
    ).dt.days
    total_revenue = tx.groupby("customer_id")["amount"].sum()

    rfm_df = pd.DataFrame(
        {
            "customer_id": recency.index,
            "recency": recency.values,
            "frequency": frequency.values,
            "monetary": monetary.values,
            "tenure_days": tenure.values,
            "total_revenue": total_revenue.values,
        }
    )

    LOG.info(f"RFM dataset built successfully with {len(rfm_df):,} customers.")

    if save:
        out = PROC_DIR / "rfm_dataset.csv"
        rfm_df.to_csv(out, index=False)
        LOG.info(f"Saved RFM dataset → {out}")
        print(f"[OK] Saved RFM dataset → {out}")

    return rfm_df, tx


def build_rfm_segments(rfm_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Create RFM scores and assign rule-based segments.
    Handles duplicate quantile edges using 'duplicates=drop'.
    Ensures scoring always works even if distributions are skewed.
    """
    LOG.info("Starting build_rfm_segments()")

    if rfm_df is None:
        LOG.info("No RFM dataframe passed, generating a new one...")
        rfm_df, _ = build_rfm_dataset()

    df = rfm_df.copy()

    def safe_qcut(series, q):
        """Safe qcut that guarantees valid scoring."""
        try:
            return pd.qcut(series, q, labels=False, duplicates="drop") + 1
        except ValueError:
            LOG.warning(f"Duplicate quantile edges detected for {series.name}. Falling back to rank-based bins.")
            return pd.cut(series.rank(method="first"), q, labels=False) + 1

    LOG.info("Assigning R, F, M scores...")

    # Safe quantile binning (automatically handled)
    df["R_score"] = safe_qcut(df["recency"], 5)
    df["F_score"] = safe_qcut(df["frequency"], 5)
    df["M_score"] = safe_qcut(df["monetary"], 5)

    # Convert to integers
    df["R_score"] = df["R_score"].astype(int)
    df["F_score"] = df["F_score"].astype(int)
    df["M_score"] = df["M_score"].astype(int)

    df["RFM_score"] = df["R_score"] * 100 + df["F_score"] * 10 + df["M_score"]

    def assign_segment(row):
        r, f, m = row["R_score"], row["F_score"], row["M_score"]

        if r >= 4 and f >= 4:
            return "Champions"
        
        if r >= 3 and f >= 3:
            return "Loyal"
        
        if r >= 4 and f <= 2:
            return "At Risk"
        
        if r <= 2 and f >= 3:
            return "Potential Loyalist"
        
        if r <= 2 and f <= 2:
            return "Hibernating"
        
        return "Others"

    LOG.info("Assigning high-level RFM segments...")

    df["rfm_segment"] = df.apply(assign_segment, axis=1)

    LOG.info(f"RFM segmentation complete. Total customers segmented: {len(df):,}")

    return df