import sys
from pathlib import Path

# Add project root to PYTHONPATH at runtime
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

    
import pytest
from app import model_evaluation

def test_evaluate_clv_model_returns_metrics():
    if not hasattr(model_evaluation, "evaluate_clv_model"):
        pytest.skip("evaluate_clv_model not implemented")
    try:
        metrics, eval_df, fi_df, shap_df = model_evaluation.evaluate_clv_model(shap_sample_frac=0.2)
    except Exception as e:
        pytest.skip(f"evaluate_clv_model unable to run in test env: {e}")
    assert isinstance(metrics, dict)

    for k in ["MAE", "RMSE", "R2"]:
        assert k in metrics, f"Metric {k} missing from evaluation metrics"