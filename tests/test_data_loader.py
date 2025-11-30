import sys
from pathlib import Path

# Add project root to PYTHONPATH at runtime
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from app.data_loader import load_transactions
import pandas as pd

def test_load_transactions_runs():
    df = load_transactions()
    assert isinstance(df, pd.DataFrame)
    assert "customer_id" in df.columns
    assert "order_date" in df.columns