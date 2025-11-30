import sys
from pathlib import Path

# Add project root to PYTHONPATH at runtime
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from app.clv_model import train_clv_model, build_clv_features
import pandas as pd

def test_clv_feature_builder():
    df = pd.DataFrame({
        "customer_id": [1,1,2],
        "order_id": [10,11,20],
        "order_date": pd.to_datetime(["2024-01-01","2024-02-01","2024-01-15"]),
        "amount": [100,120,80],
        "category": ["A","B","A"],
        "channel": ["web","app","web"]
    })

    X, y = build_clv_features()
    assert len(X) == len(y)


def test_clv_model_train():
    model, scaler, mae, rmse, r2 = train_clv_model()
    assert model is not None
    assert scaler is not None