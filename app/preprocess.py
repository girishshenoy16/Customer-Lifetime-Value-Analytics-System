import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

LOG = logging.getLogger("preprocess")

DATA_RAW = Path("data/raw/customers_transactions.csv")
STATUS_RAW = Path("data/raw/customer_status.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)

FEATURE_STORE = PROCESSED_DIR / "feature_store.csv"


def preprocess_all():
    """
    Full preprocessing pipeline:
    - Load raw transaction data
    - Compute RFM + behavioral features
    - Merge customer status safely
    - Create stable, numeric-ready feature_store
    """

    LOG.info("Starting preprocessing...")

    if not DATA_RAW.exists():
        raise FileNotFoundError(f"Missing raw data file: {DATA_RAW}")

    df = pd.read_csv(DATA_RAW)
    df["order_date"] = pd.to_datetime(df["order_date"])

    # -------------------------
    # BASIC CLEANING
    # -------------------------
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    df["category"] = df["category"].fillna("Unknown")
    df["channel"] = df["channel"].fillna("Unknown")

    LOG.info(f"Loaded {len(df):,} transactions.")

    # -------------------------
    # RFM CORE METRICS
    # -------------------------
    today = df["order_date"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("customer_id").agg(
        recency=("order_date", lambda x: (today - x.max()).days),
        frequency=("order_id", "count"),
        monetary=("amount", "sum"),
        tenure=("order_date", lambda x: (x.max() - x.min()).days + 1),
        total_revenue=("amount", "sum"),
    )

    rfm["recency"] = rfm["recency"].astype(float)
    rfm["frequency"] = rfm["frequency"].astype(float)
    rfm["monetary"] = rfm["monetary"].astype(float)

    # -------------------------
    # CATEGORY DIVERSITY
    # -------------------------
    cat_div = df.groupby("customer_id")["category"].nunique()
    chan_div = df.groupby("customer_id")["channel"].nunique()

    # CROSS-SELL RATE = (#unique categories) / (#orders)
    cross_rate = cat_div / df.groupby("customer_id")["order_id"].count()

    # CART CONVERSION RATE (if available)
    if "cart_events" in df.columns:
        cart_conv = df.groupby("customer_id")["cart_events"].sum() / df.groupby("customer_id")["order_id"].count()
    else:
        cart_conv = pd.Series(0, index=rfm.index)

    features = rfm.copy()
    features["category_diversity"] = cat_div
    features["channel_diversity"] = chan_div
    features["cross_sell_rate"] = cross_rate.fillna(0.0)
    features["cart_conversion_rate"] = cart_conv.fillna(0.0)

    # -------------------------
    # SAFE NUMERIC COERCION
    # -------------------------
    numeric_cols = [
        "recency", "frequency", "monetary", "tenure", "total_revenue",
        "category_diversity", "channel_diversity",
        "cross_sell_rate", "cart_conversion_rate"
    ]
    for c in numeric_cols:
        features[c] = pd.to_numeric(features[c], errors="coerce").fillna(0)

    # -------------------------
    # MERGE CUSTOMER STATUS (persona, churn flags, subscription)
    # -------------------------
    if STATUS_RAW.exists():
        cust = pd.read_csv(STATUS_RAW)

        required_cols = ["customer_id", "persona", "is_subscriber", "subscription_plan", "is_churned"]
        for col in required_cols:
            if col not in cust.columns:
                LOG.warning(f"Column missing in customer_status: {col}. Filling default.")
                if col in ["is_subscriber", "is_churned"]:
                    cust[col] = 0
                else:
                    cust[col] = "Unknown"

        # CLEAN dtypes
        cust["is_subscriber"] = cust["is_subscriber"].astype(int).fillna(0)
        cust["is_churned"] = cust["is_churned"].astype(int).fillna(0)
        cust["persona"] = cust["persona"].fillna("Unknown")
        cust["subscription_plan"] = cust["subscription_plan"].fillna("None")

        # -------------------------
        # PREVENT COLUMN OVERLAP 🚨
        # -------------------------
        overlap_cols = ["persona", "is_subscriber", "subscription_plan", "is_churned"]
        features = features.drop(columns=[c for c in overlap_cols if c in features.columns], errors="ignore")

        # SAFE JOIN (NO OVERLAP)
        features = features.join(
            cust.set_index("customer_id")[overlap_cols],
            how="left"
        )

    else:
        LOG.warning("customer_status.csv not found! Filling defaults.")

        features["persona"] = "Unknown"
        features["subscription_plan"] = "None"
        features["is_subscriber"] = 0
        features["is_churned"] = 0

    # -------------------------
    # ENSURE FINAL DTYPE CONSISTENCY
    # -------------------------
    features["is_subscriber"] = features["is_subscriber"].astype(int)
    features["is_churned"] = features["is_churned"].astype(int)
    features["persona"] = features["persona"].astype(str)
    features["subscription_plan"] = features["subscription_plan"].astype(str)

    # ---------------------------------------------------------
    # ADD CLV TARGET: future_clv = revenue AFTER a cutoff date
    # ---------------------------------------------------------
    cutoff_date = df["order_date"].quantile(0.70)  # Use 70% of timeline as cut
    future_df = df[df["order_date"] > cutoff_date]

    future_revenue = (
        future_df.groupby("customer_id")["amount"].sum()
        .reindex(features.index)
        .fillna(0.0)
    )

    features["future_clv"] = future_revenue
    features["future_clv"] = pd.to_numeric(features["future_clv"], errors="coerce").fillna(0.0)


    # -------------------------
    # SAVE FEATURE STORE
    # -------------------------
    features.to_csv(FEATURE_STORE)
    LOG.info(f"feature_store saved: {FEATURE_STORE} ({len(features):,} rows)")

    LOG.info("Preprocessing complete.")
    return features
