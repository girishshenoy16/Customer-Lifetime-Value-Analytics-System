import sys
from pathlib import Path

# Add project root to PYTHONPATH at runtime
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


import pandas as pd
import pytest
from pathlib import Path


@pytest.fixture
def dummy_transactions():
    return pd.DataFrame({
        "customer_id": [1,1,2,2,3],
        "order_id": [10,11,20,21,30],
        "order_date": pd.to_datetime([
            "2024-01-05", "2024-01-10", "2024-02-01",
            "2024-02-10", "2024-03-01"
        ]),
        "amount": [100,150,70,90,200],
        "category": ["A","B","A","C","B"],
        "channel": ["web","web","app","ref","web"]
    })
