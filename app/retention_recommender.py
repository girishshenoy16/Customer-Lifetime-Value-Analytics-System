import numpy as np
import pandas as pd

from app.data_loader import load_feature_data, load_customer_status
from app.clv_model import score_customers
from app.rfm_segmentation import build_rfm_dataset, build_rfm_segments
from app.logging_config import get_logger


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# ========================================
# ADDED: Logging Integration
# ========================================
LOG = get_logger(__name__, filename="app.log")
# ========================================


def build_clv_with_segments() -> pd.DataFrame:
    """
    Build unified customer table:
    - RFM features + segment
    - CLV features + predicted CLV
    - Customer status (persona, churn, subscription)
    Handles overlapping columns safely.
    """
    LOG.info("Building CLV + RFM + persona unified table...")

    # RFM from transactions
    LOG.info("Computing RFM dataset...")
    rfm_df, tx = build_rfm_dataset()
    rfm_segments = build_rfm_segments(rfm_df).set_index("customer_id")

    # Feature store → includes churn, persona, subscription for some pipelines
    LOG.info("Loading feature_store for CLV scoring...")
    feat = load_feature_data().set_index("customer_id")

    # Score CLV (model loads automatically)
    LOG.info("Scoring customers with CLV model...")
    feat_scored = score_customers(feat).copy()

    # Customer status file (optional)
    LOG.info("Loading customer_status (persona/churn/subscription)...")
    cust_status = load_customer_status()
    if cust_status is not None and not cust_status.empty:
        cust_status = cust_status.set_index("customer_id")
        LOG.info(f"Customer status loaded: {len(cust_status):,} rows")
    else:
        cust_status = pd.DataFrame()
        LOG.warning("Customer status file empty or missing.")

    # -------------------
    # 1) MERGE RFM + FEATURES
    # -------------------
    LOG.info("Merging RFM segments + CLV features...")

    overlapping_cols = [
        c for c in feat_scored.columns
        if c in rfm_segments.columns
    ]
    if overlapping_cols:
        LOG.warning(f"Dropping overlapping columns before merge: {overlapping_cols}")
        feat_scored = feat_scored.drop(columns=overlapping_cols)

    combined = rfm_segments.join(feat_scored, how="inner")

    # -------------------
    # 2) MERGE CUSTOMER STATUS SAFELY
    # -------------------
    if not cust_status.empty:
        LOG.info("Merging customer status data...")

        status_cols = ["persona", "is_subscriber", "subscription_plan", "is_churned"]
        status_cols = [c for c in status_cols if c in cust_status.columns]

        duplicates = [c for c in status_cols if c in combined.columns]
        if duplicates:
            LOG.warning(f"Dropping duplicate status columns: {duplicates}")
            combined = combined.drop(columns=duplicates)

        combined = combined.join(cust_status[status_cols], how="left")

    combined.reset_index(inplace=True)

    if "predicted_clv" not in combined.columns:
        LOG.error("predicted_clv missing after scoring. Model may not be trained.")
        raise ValueError("predicted_clv missing. Train model first.")

    LOG.info(f"Unified customer dataset built with {len(combined):,} rows.")
    return combined


def _assign_value_tier(clv_series: pd.Series) -> pd.Series:
    LOG.info("Assigning customer value tiers (High / Medium / Low)...")

    q_low = clv_series.quantile(0.33)
    q_mid = clv_series.quantile(0.66)

    def tier(x):
        if x >= q_mid:
            return "High"
        if x >= q_low:
            return "Medium"
        return "Low"

    return clv_series.apply(tier)


def generate_recommendations(clv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate retention actions and ROI estimates.
    """
    LOG.info("Generating retention recommendations...")

    df = clv_df.copy()

    # value tier based on predicted CLV
    df["value_tier"] = _assign_value_tier(df["predicted_clv"])

    LOG.info("Computing campaign_cost per customer...")
    df["campaign_cost"] = df["predicted_clv"].apply(lambda v: max(50.0, v * 0.05))
    df.loc[df["value_tier"] == "High", "campaign_cost"] = df["predicted_clv"] * 0.08
    df.loc[df["value_tier"] == "Low", "campaign_cost"] = df["predicted_clv"] * 0.02
    df["campaign_cost"] = df["campaign_cost"].clip(lower=50.0)

    # uplift assumptions
    LOG.info("Computing expected uplift + ROI...")

    def uplift(row):
        base = row["predicted_clv"]
        persona = row.get("persona", "Unknown")
        is_churned = row.get("is_churned", 0)

        factor = 0.12
        if persona == "HighSpender":
            factor = 0.18
        elif persona == "Loyalist":
            factor = 0.15
        elif persona == "DealSeeker":
            factor = 0.1

        if is_churned == 1:
            factor *= 0.6

        return base * factor

    df["expected_uplift"] = df.apply(uplift, axis=1)
    df["expected_roi"] = df["expected_uplift"] - df["campaign_cost"]

    LOG.info("Assigning recommended actions + offers...")

    def action(row):
        seg = row.get("rfm_segment", "Others")
        persona = row.get("persona", "Unknown")
        value_tier = row.get("value_tier", "Medium")
        is_churned = row.get("is_churned", 0)

        if is_churned == 1 and value_tier == "High":
            return "Win-back premium churned customer"
        if seg == "Champions":
            return "Maintain loyalty & surprise-and-delight"
        if seg == "Loyal":
            return "Upsell / cross-sell personalized bundles"
        if seg == "At Risk":
            return "Targeted win-back with strong incentive"
        if seg == "Hibernating":
            return "Reactivation email + soft discount"
        return "Standard lifecycle nurturing"

    def offer(row):
        persona = row.get("persona", "Unknown")
        value_tier = row.get("value_tier", "Medium")
        is_sub = row.get("is_subscriber", 0)

        if persona == "HighSpender":
            return "Exclusive early-access + premium support"
        if persona == "Loyalist":
            return "Tier upgrade / loyalty points booster"
        if persona == "DealSeeker":
            return "Limited-time discount on frequently browsed category"
        if is_sub:
            return "Bundle add-on offer for subscription users"
        if value_tier == "Low":
            return "Light discount + content-driven engagement"
        return "Personalized recommendation email"

    df["recommended_action"] = df.apply(action, axis=1)
    df["suggested_offer"] = df.apply(offer, axis=1)

    LOG.info("Retention recommendations generated successfully.")

    cols = [
        "customer_id",
        "rfm_segment",
        "value_tier",
        "persona",
        "is_subscriber",
        "is_churned",
        "predicted_clv",
        "campaign_cost",
        "expected_uplift",
        "expected_roi",
        "recommended_action",
        "suggested_offer",
    ]

    cols = [c for c in cols if c in df.columns]
    return df[cols].sort_values("predicted_clv", ascending=False)