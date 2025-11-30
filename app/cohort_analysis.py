import pandas as pd
from typing import Tuple, Optional

from app.data_loader import load_transactions
from app.logging_config import get_logger

# ==================================
# ADDED: Logging Integration
# ==================================
LOG = get_logger(__name__, filename="app.log")
# ==================================


def build_cohort_retention(
    tx: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build cohort size and retention matrix.

    Returns:
        cohort_sizes: DataFrame with cohort_month as index and size in first column
        retention:    DataFrame (heatmap matrix) with retention rates
    """

    LOG.info("Starting build_cohort_retention()")

    if tx is None:
        LOG.info("No transactions passed to cohort function. Loading via data_loader...")
        tx = load_transactions().copy()

    LOG.info(f"Computing cohort metrics for {len(tx):,} rows")

    tx["order_month"] = tx["order_date"].values.astype("datetime64[M]")

    # first month for each customer
    LOG.info("Assigning cohort_month per customer...")
    cohort_month = tx.groupby("customer_id")["order_month"].min()
    cohort_month.name = "cohort_month"

    tx = tx.join(cohort_month, on="customer_id")

    # cohort index: months since first purchase
    LOG.info("Computing cohort_index (months since first purchase)...")
    tx["cohort_index"] = (
        (tx["order_month"].dt.year - tx["cohort_month"].dt.year) * 12
        + (tx["order_month"].dt.month - tx["cohort_month"].dt.month)
    )

    # count active customers per cohort & cohort_index
    LOG.info("Building cohort pivot table...")
    cohort_data = (
        tx.groupby(["cohort_month", "cohort_index"])["customer_id"]
        .nunique()
        .reset_index()
    )

    cohort_pivot = cohort_data.pivot_table(
        index="cohort_month",
        columns="cohort_index",
        values="customer_id",
    ).fillna(0)

    # base sizes in period 0
    cohort_sizes = cohort_pivot.iloc[:, 0].to_frame(name="cohort_size")

    retention = cohort_pivot.divide(cohort_pivot.iloc[:, 0], axis=0).round(3)

    LOG.info("Cohort analysis completed successfully.")

    return cohort_sizes, retention