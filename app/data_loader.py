import pandas as pd
from pathlib import Path
from app.logging_config import get_logger

"""
Unified loader for:
- Raw synthetic data
- Processed feature store
- RFM datasets
- Customer status data

Automatically falls back to raw data if processed not available.
"""

# ============================
# ADDED: Logging Integration
# ============================
LOG = get_logger(__name__, filename="app.log")
# ============================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"


# ==========================================================
# 1. Load RAW FILES
# ==========================================================
def load_raw_transactions():
    path = RAW_DIR / "customers_transactions.csv"
    LOG.info(f"Loading RAW transactions from: {path}")
    df = pd.read_csv(path, parse_dates=["order_date"])
    LOG.info(f"Loaded {len(df):,} raw transaction rows")
    return df


def load_raw_cart_events():
    path = RAW_DIR / "cart_events.csv"
    LOG.info(f"Loading RAW cart events from: {path}")

    if path.exists():
        df = pd.read_csv(path, parse_dates=["event_date"])
        LOG.info(f"Loaded {len(df):,} cart event rows")
        return df

    LOG.warning("cart_events.csv not found, returning empty DataFrame")
    return pd.DataFrame()


def load_raw_customer_status():
    path = RAW_DIR / "customer_status.csv"
    LOG.info(f"Loading RAW customer status from: {path}")

    if path.exists():
        df = pd.read_csv(
            path,
            parse_dates=["first_order_date", "last_order_date"],
        )
        LOG.info(f"Loaded {len(df):,} customer status rows")
        return df

    LOG.warning("customer_status.csv not found, returning empty DataFrame")
    return pd.DataFrame()


# ==========================================================
# 2. Load PROCESSED FILES
# ==========================================================
def load_feature_store():
    path = PROC_DIR / "feature_store.csv"
    LOG.info(f"Loading processed feature store: {path}")

    if path.exists():
        df = pd.read_csv(path)
        df.index.name = "customer_id"
        LOG.info(f"Loaded feature store with {len(df):,} customers")
        return df

    LOG.warning("feature_store.csv not found")
    return None


def load_cleaned_transactions():
    path = PROC_DIR / "cleaned_transactions.csv"
    LOG.info(f"Loading cleaned transactions: {path}")

    if path.exists():
        df = pd.read_csv(path, parse_dates=["order_date"])
        LOG.info(f"Loaded {len(df):,} cleaned transactions")
        return df

    LOG.warning("cleaned_transactions.csv not found")
    return None


# ==========================================================
# 3. UNIFIED ACCESS POINTS
# ==========================================================
def load_transactions():
    """Load cleaned transactions if available, else raw."""
    LOG.info("Request: load_transactions()")

    cleaned = load_cleaned_transactions()
    if cleaned is not None:
        LOG.info("Using cleaned_transactions.csv")
        return cleaned

    LOG.info("Falling back to raw customers_transactions.csv")
    return load_raw_transactions()


def load_cart_events():
    LOG.info("Request: load_cart_events()")
    return load_raw_cart_events()


def load_customer_status():
    LOG.info("Request: load_customer_status()")
    return load_raw_customer_status()


def load_feature_data():
    """Main dataset used for CLV model training."""
    LOG.info("Request: load_feature_data() → feature_store.csv")

    feature_store = load_feature_store()
    if feature_store is not None:
        return feature_store

    LOG.error(
        "feature_store.csv missing. User must run preprocessing.",
        exc_info=False,
    )
    raise FileNotFoundError(
        "Processed feature_store.csv not found. Run: python app/preprocess.py"
    )