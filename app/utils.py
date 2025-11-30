from pathlib import Path


def get_base_path() -> Path:
    """
    Returns the root folder path of the project.
    Assumes this file is in: project_root/app/utils.py
    """

    return Path(__file__).resolve().parents[1]


def get_data_paths():
    base = get_base_path()

    data_dir = base / "data"
    raw_dir = data_dir / "raw"

    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    return {
        "base": base,
        "data": data_dir,
        "raw": raw_dir,
        "processed": processed_dir,
    }


def get_models_path() -> Path:
    base = get_base_path()
    
    models_dir = base / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    return models_dir