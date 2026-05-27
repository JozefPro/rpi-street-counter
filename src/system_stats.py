from pathlib import Path

import psutil


PI_TEMP_PATH = Path("/sys/class/thermal/thermal_zone0/temp")


def get_temperature_c():
    if not PI_TEMP_PATH.exists():
        return None

    try:
        raw_value = PI_TEMP_PATH.read_text(encoding="utf-8").strip()
        return round(int(raw_value) / 1000.0, 1)
    except (OSError, ValueError):
        return None


def get_system_stats():
    return {
        "cpu_percent": round(psutil.cpu_percent(interval=None), 1),
        "ram_percent": round(psutil.virtual_memory().percent, 1),
        "temperature_c": get_temperature_c(),
    }
