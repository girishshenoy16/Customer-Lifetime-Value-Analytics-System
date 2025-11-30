import sys
from pathlib import Path

# Add project root to PYTHONPATH at runtime
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from app.generate_reports import main
from pathlib import Path

def test_generate_reports():
    main()
    assert Path("reports/executive_summary.md").exists()
    assert Path("reports/executive_summary.pdf").exists()
