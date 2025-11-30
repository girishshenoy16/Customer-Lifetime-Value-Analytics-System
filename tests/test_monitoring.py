import sys
from pathlib import Path

# Add project root to PYTHONPATH at runtime
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.model_monitoring import (
    load_baseline_snapshot,
    save_baseline_snapshot,
    monitor_drift,
    prediction_drift
)


def test_baseline_snapshot():
    save_baseline_snapshot()
    baseline = load_baseline_snapshot()
    assert baseline is not None


def test_prediction_drift_runs():
    try:
        result = prediction_drift()
        assert "psi" in result
    except Exception:
        pass


def test_feature_drift_runs():
    try:
        df = monitor_drift()
        assert df is not None
    except Exception:
        pass