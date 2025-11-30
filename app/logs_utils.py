from pathlib import Path
from app.logging_config import LOG_DIR
import re


def read_last_lines(path: Path, n_lines: int = 300):
    """Efficient log tail (no full file load)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            block = 1024
            data = []

            while size > 0 and len(data) < n_lines:
                step = min(block, size)
                f.seek(size - step)
                chunk = f.read(step).decode(errors="ignore")
                data.extend(chunk.splitlines())
                size -= step

            return data[-n_lines:]

    except Exception:
        return ["ERROR: Could not read log file."]


def filter_log_lines(lines: list, keyword: str, level: str):
    """Filter logs by keyword and severity."""
    output = []
    for line in lines:
        if keyword and keyword.lower() not in line.lower():
            continue
        if level != "ALL" and level.upper() not in line.upper():
            continue
        output.append(line)
    return output


def highlight_line(line: str):
    """Convert a log line to styled HTML."""
    if "ERROR" in line:
        color = "#ff4c4c"
    elif "CRITICAL" in line:
        color = "#ff0000"
    elif "WARNING" in line:
        color = "#ffaa00"
    elif "INFO" in line:
        color = "#00c6ff"
    else:
        color = "#d3d3d3"

    return f"<div style='color:{color}; font-family:Monospace; font-size:14px;'>{line}</div>"


def highlight_block(lines: list):
    """Return all highlighted lines as HTML."""
    return "\n".join([highlight_line(l) for l in lines])