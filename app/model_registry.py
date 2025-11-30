import json
from pathlib import Path
from datetime import datetime
import joblib

"""
Model Registry utilities:
- saves models/scalers as model_v{N}.pkl / scaler_v{N}.pkl
- maintains models/registry.json with metadata (version, timestamp, metrics, tags)
- supports listing and rollback (set 'active' flag)
"""

MODELS_DIR = Path("models")
REGISTRY_FILE = MODELS_DIR / "registry.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _load_registry():
    if not REGISTRY_FILE.exists():
        return {"versions": []}
    return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))


def _save_registry(reg):
    REGISTRY_FILE.write_text(json.dumps(reg, indent=2), encoding="utf-8")


def _next_version(reg):
    vs = [v["version"] for v in reg.get("versions", [])]
    if not vs:
        return 1
    return max(vs) + 1


def register_model(model, scaler, metrics: dict = None, tags: dict = None):
    """
    Save model + scaler and register metadata.
    Returns new version number.
    """
    reg = _load_registry()
    version = _next_version(reg)
    timestamp = datetime.utcnow().isoformat() + "Z"

    model_path = MODELS_DIR / f"model_v{version}.pkl"
    scaler_path = MODELS_DIR / f"scaler_v{version}.pkl"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    entry = {
        "version": version,
        "timestamp": timestamp,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "metrics": metrics or {},
        "tags": tags or {},
        "active": True,
    }

    # mark previous active -> False
    for v in reg.get("versions", []):
        v["active"] = False

    reg.setdefault("versions", []).append(entry)
    _save_registry(reg)
    return version


def list_versions():
    reg = _load_registry()
    return reg.get("versions", [])


def get_active_version():
    reg = _load_registry()
    for v in reg.get("versions", []):
        if v.get("active"):
            return v
    return None


def load_active_model_and_scaler():
    active = get_active_version()
    if not active:
        raise FileNotFoundError("No active model in registry.")
    model = joblib.load(Path(active["model_path"]))
    scaler = joblib.load(Path(active["scaler_path"]))
    return model, scaler


def rollback_to_version(version: int):
    reg = _load_registry()
    found = False
    for v in reg.get("versions", []):
        v["active"] = (v["version"] == version)
        if v["version"] == version:
            found = True
    if not found:
        raise ValueError(f"Version {version} not found in registry.")
    _save_registry(reg)
    return True