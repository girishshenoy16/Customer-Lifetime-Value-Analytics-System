import psutil
import platform
from datetime import datetime


def get_system_metrics():
    """Return system-wide CPU, RAM, DISK, platform info."""
    return {
        "timestamp": datetime.now(),
        "cpu_percent": psutil.cpu_percent(interval=0.5),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage("/").percent,
        "active_processes": len(psutil.pids()),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }


def get_process_table(limit: int = 50):
    """Top processes sorted by CPU%."""
    processes = []

    for p in psutil.process_iter(attrs=["pid", "name", "cpu_percent", "memory_percent"]):
        processes.append(p.info)

    return sorted(processes, key=lambda x: x["cpu_percent"], reverse=True)[:limit]