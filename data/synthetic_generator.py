
import sys
from pathlib import Path

# Project-root-safe paths
ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

"""
Advanced Synthetic Transaction Generator with Personas, Cross-sell, Churn & Subscription
----------------------------------------------------------------------------------------

Outputs (in data/raw/):

1) customers_transactions.csv
   - customer_id, order_id, order_date, amount, category, channel
   - persona, is_repeat, is_cross_sell, is_churned, is_subscriber, subscription_plan

2) cart_events.csv
   - customer_id, event_id, event_date, cart_value, category, channel, persona, converted

3) customer_status.csv
   - customer_id, persona, is_subscriber, subscription_plan, is_churned,
     first_order_date, last_order_date, total_orders, total_revenue,
     distinct_categories, distinct_channels
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from app.logging_config import get_logger

LOG = get_logger(__name__, filename="synthetic_data.log")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CONFIG_PATH = ROOT / "data" / "config" / "synthetic_config.csv"

# ----------------------------------------------------------
# DEFAULT PERSONAS (fallback if CSV is missing/incomplete)
# ----------------------------------------------------------
DEFAULT_PERSONAS = {
    "loyalist": {
        "purchase_freq": 0.8,
        "avg_amount": 1200,
        "cross_sell_rate": 0.45,
        "cart_conversion_rate": 0.75,
        "churn_prob": 0.05,
        "subscriber_prob": 0.65,
        "seasonality_high": 1.15,
        "seasonality_low": 0.90,
        "monthly_drift": 1.02,
    },
    "high_spender": {
        "purchase_freq": 0.6,
        "avg_amount": 2500,
        "cross_sell_rate": 0.55,
        "cart_conversion_rate": 0.82,
        "churn_prob": 0.08,
        "subscriber_prob": 0.45,
        "seasonality_high": 1.15,
        "seasonality_low": 0.90,
        "monthly_drift": 1.02,
    },
    "bargain_hunter": {
        "purchase_freq": 0.45,
        "avg_amount": 650,
        "cross_sell_rate": 0.20,
        "cart_conversion_rate": 0.40,
        "churn_prob": 0.20,
        "subscriber_prob": 0.10,
        "seasonality_high": 1.15,
        "seasonality_low": 0.90,
        "monthly_drift": 1.02,
    },
    "one_timer": {
        "purchase_freq": 0.15,
        "avg_amount": 500,
        "cross_sell_rate": 0.05,
        "cart_conversion_rate": 0.10,
        "churn_prob": 0.70,
        "subscriber_prob": 0.02,
        "seasonality_high": 1.15,
        "seasonality_low": 0.90,
        "monthly_drift": 1.02,
    },
}

CHANNELS = ["App", "Website", "Referral"]
CATEGORIES = ["Electronics", "Fashion", "Beauty", "Groceries", "Fitness", "Books"]


# ----------------------------------------------------------
# LOAD CONFIG WITH ERROR-TOLERANCE
# ----------------------------------------------------------
def load_config():
    if not CONFIG_PATH.exists():
        LOG.warning("Config missing → using default personas.")
        return DEFAULT_PERSONAS

    df = pd.read_csv(CONFIG_PATH)

    required = [
        "persona", "purchase_freq", "avg_amount",
        "cross_sell_rate", "cart_conversion_rate", "churn_prob",
        "subscriber_prob", "seasonality_high", "seasonality_low",
        "monthly_drift",
    ]

    if not all(col in df.columns for col in required):
        LOG.error(
            f"Config missing required columns.\n"
            f"Found: {df.columns.tolist()}\n"
            f"Expected: {required}\n"
            "→ Using default persona config."
        )
        return DEFAULT_PERSONAS

    LOG.info("Loaded synthetic_config.csv successfully.")

    return {
        row["persona"]: {
            "purchase_freq": row["purchase_freq"],
            "avg_amount": row["avg_amount"],
            "cross_sell_rate": row["cross_sell_rate"],
            "cart_conversion_rate": row["cart_conversion_rate"],
            "churn_prob": row["churn_prob"],
            "subscriber_prob": row["subscriber_prob"],
            "seasonality_high": row["seasonality_high"],
            "seasonality_low": row["seasonality_low"],
            "monthly_drift": row["monthly_drift"],
        }
        for _, row in df.iterrows()
    }


# ----------------------------------------------------------
# MAIN GENERATION FUNCTION
# ----------------------------------------------------------
def generate_synthetic_data(
    n_customers: int = 5000,
    months: int = 18,
    drift_enabled: bool = True,
):
    LOG.info(f"Generating synthetic dataset for {n_customers} customers…")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    config = load_config()

    personas = list(config.keys())
    customer_persona = rng.choice(personas, size=n_customers)

    start_date = datetime.today() - timedelta(days=months * 30)

    all_txn_rows = []
    all_cart_rows = []
    all_status_rows = []

    for cid in range(1, n_customers + 1):

        persona = customer_persona[cid - 1]
        pconfig = config[persona]

        is_churned = rng.random() < pconfig["churn_prob"]
        is_subscriber = rng.random() < pconfig["subscriber_prob"]
        subscription_plan = rng.choice(["None", "Basic", "Pro"]) if is_subscriber else "None"

        freq = max(1, np.random.poisson(5 * pconfig["purchase_freq"]))

        full_days = pd.date_range(start=start_date, periods=months * 30)
        purchase_days = rng.choice(full_days, size=freq, replace=False)

        total_revenue = 0
        categories_set = set()
        channels_set = set()

        for day in purchase_days:

            # 🔧 FIX: Convert numpy.datetime64 → pandas Timestamp
            day = pd.Timestamp(day)

            month = day.month
            if month in [11, 12]:
                seasonal_factor = pconfig["seasonality_high"]
            else:
                seasonal_factor = pconfig["seasonality_low"]

            amount = np.random.normal(pconfig["avg_amount"], pconfig["avg_amount"] * 0.25)
            amount = max(200, amount * seasonal_factor)

            if drift_enabled:
                amount *= pconfig["monthly_drift"]

            category = rng.choice(CATEGORIES)
            channel = rng.choice(CHANNELS)

            all_txn_rows.append({
                "customer_id": cid,
                "order_id": f"ORD-{cid}-{rng.integers(999999)}",
                "order_date": day,
                "amount": float(amount),
                "category": category,
                "channel": channel,
                "is_repeat": 1,
                "persona": persona,
                "is_subscriber": int(is_subscriber),
                "subscription_plan": subscription_plan,
                "is_churned": int(is_churned),
            })

            total_revenue += amount
            categories_set.add(category)
            channels_set.add(channel)

            # CART EVENTS
            cart_val = amount * rng.uniform(0.3, 1.0)
            converted = rng.random() < pconfig["cart_conversion_rate"]

            all_cart_rows.append({
                "customer_id": cid,
                "event_id": f"CART-{cid}-{rng.integers(999999)}",
                "event_date": day,
                "cart_value": float(cart_val),
                "category": category,
                "channel": channel,
                "persona": persona,
                "converted": int(converted),
            })

        all_status_rows.append({
            "customer_id": cid,
            "persona": persona,
            "is_subscriber": int(is_subscriber),
            "subscription_plan": subscription_plan,
            "is_churned": int(is_churned),
            "first_order_date": min(purchase_days),
            "last_order_date": max(purchase_days),
            "total_orders": freq,
            "total_revenue": float(total_revenue),
            "distinct_categories": len(categories_set),
            "distinct_channels": len(channels_set),
        })

    df_txn = pd.DataFrame(all_txn_rows)
    df_cart = pd.DataFrame(all_cart_rows)
    df_status = pd.DataFrame(all_status_rows)

    df_txn.to_csv(RAW_DIR / "customers_transactions.csv", index=False)
    df_cart.to_csv(RAW_DIR / "cart_events.csv", index=False)
    df_status.to_csv(RAW_DIR / "customer_status.csv", index=False)

    LOG.info("Generated 3 datasets → transactions, carts, status")

    return df_txn, df_cart, df_status


# ----------------------------------------------------------
# RUN DIRECTLY
# ----------------------------------------------------------
if __name__ == "__main__":
    generate_synthetic_data()