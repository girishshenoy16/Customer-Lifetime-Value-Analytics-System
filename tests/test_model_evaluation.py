import sys
from pathlib import Path

# Add project root to PYTHONPATH at runtime
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

    
from app.model_evaluation import evaluate_clv_model

def test_model_evaluation_runs():
    metrics, eval_df, fi_df, shap_df = evaluate_clv_model(shap_sample_frac=0.1)
    assert "MAE" in metrics
    assert eval_df.shape[0] > 0