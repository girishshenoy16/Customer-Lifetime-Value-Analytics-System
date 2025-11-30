import sys
from pathlib import Path

# Add project root to PYTHONPATH at runtime
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.preprocess import preprocess_all


def test_preprocess_pipeline_runs_and_writes_files():
    """
    Run the preprocessing pipeline and assert that the expected files
    are written to data/processed/.
    This test is tolerant: it checks for the actual files rather than
    a particular return dictionary key.
    """
    
    out = preprocess_all()  # safe to call multiple times

    fs_path = Path("data") / "processed" / "feature_store.csv"
    rfm_path = Path("data") / "processed" / "rfm_dataset.csv"

    assert fs_path.exists(), f"feature_store.csv not found at {fs_path}"
    assert rfm_path.exists(), f"rfm_dataset.csv not found at {rfm_path}"