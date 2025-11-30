import pandas as pd
import re
from pathlib import Path
from app.logging_config import LOG_DIR


LOG_PATTERN = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}.*?\d{2}:\d{2}:\d{2}).*?— (?P<logger>.*?) — (?P<level>INFO|WARNING|ERROR|CRITICAL) — (?P<msg>.*)"
)


def parse_log_file(path: Path):
    """Parse .log file lines into a DataFrame."""
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = LOG_PATTERN.match(line)
            if m:
                d = m.groupdict()
                d["timestamp"] = pd.to_datetime(d["timestamp"])
                rows.append(d)

    return pd.DataFrame(rows)


def load_all_logs():
    """Load all logs in logs/ as a combined DataFrame."""
    dfs = []

    for path in LOG_DIR.glob("*.log"):
        df = parse_log_file(path)
        if not df.empty:
            df["file"] = path.name
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)