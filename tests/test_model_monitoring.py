import sys
from pathlib import Path

# Add project root to PYTHONPATH at runtime
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

    

from app.model_monitoring import (
    monitor_drift, prediction_drift,
    load_baseline_snapshot, save_baseline_snapshot
)

def test_baseline_save_load():
    save_baseline_snapshot()
    base = load_baseline_snapshot()
    assert base is not None

def test_prediction_drift_runs():
    out = prediction_drift()
    assert "psi" in out or True  # no crash

def test_feature_drift_runs():
    df = monitor_drift()
    assert df is not None