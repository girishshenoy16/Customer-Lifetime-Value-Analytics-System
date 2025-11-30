import sys
from pathlib import Path


# Add project root to PYTHONPATH at runtime
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))



from app.rfm_segmentation import build_rfm_dataset, build_rfm_segments
import pandas as pd

def test_rfm_pipeline():
    df = pd.DataFrame({
        "customer_id": [1,1,2,3],
        "order_id": [10,11,20,30],
        "order_date": pd.to_datetime([
            "2024-01-01", "2024-02-01", "2024-01-10", "2024-03-01"
        ]),
        "amount": [100,120,60,80]
    })

    rfm, _ = build_rfm_dataset(df)
    assert "recency" in rfm.columns

    seg = build_rfm_segments(rfm)
    assert "rfm_segment" in seg.columns