import sys
from pathlib import Path

# Add project root to PYTHONPATH at runtime
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from app.retention_recommender import generate_recommendations, build_clv_with_segments
import pandas as pd

def test_recommendations_run():
    df = build_clv_with_segments()
    rec = generate_recommendations(df)
    assert isinstance(rec, pd.DataFrame)
    assert "customer_id" in rec.columns
    assert "recommended_action" in rec.columns