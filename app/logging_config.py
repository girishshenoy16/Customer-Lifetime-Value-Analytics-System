import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import zipfile

"""
Central logging configuration for the CLV platform.
Provides:
- Rotating file handlers
- Unified logger factory
- Log directory auto-creation
"""

# ================================
# LOG DIRECTORY
# ================================
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


# ================================
# GLOBAL LOG FORMAT
# ================================
LOG_FORMAT = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"


def get_rotating_handler(filename: str, max_mb: int = 5):
    """Create rotating handler for a log file."""
    file_path = LOG_DIR / filename
    handler = RotatingFileHandler(
        file_path,
        maxBytes=max_mb * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    return handler


def get_logger(name: str, filename: str = "app.log", level=logging.INFO):
    """Create or retrieve a logger with rotating file output."""
    logger = logging.getLogger(name)

    if not logger.handlers:  # Avoid duplicate handlers
        handler = get_rotating_handler(filename)
        logger.setLevel(level)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


def list_log_files():
    """List all log files in logs/ directory."""
    return [p.name for p in LOG_DIR.glob("*.log")]


def zip_all_logs():
    """Create a ZIP of all logs for download in Streamlit."""
    zip_path = LOG_DIR / "all_logs.zip"

    with zipfile.ZipFile(zip_path, "w") as z:
        for f in LOG_DIR.glob("*.log"):
            z.write(f, arcname=f.name)

    return zip_path