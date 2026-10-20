"""Where aiVibe Studio keeps the user's data: %LOCALAPPDATA%\\aiVibe Studio, outside the install folder, so an
update or reinstall never touches settings, trained subjects or downloaded models."""
import os
from pathlib import Path

APP_DIR = "aiVibe Studio"


def data_root(env=os.environ) -> Path:
    """AIVIBE_DATA wins (tests, portable use); otherwise <LOCALAPPDATA>\\aiVibe Studio."""
    if env.get("AIVIBE_DATA"):
        return Path(env["AIVIBE_DATA"])
    base = env.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Local")
    return Path(base) / APP_DIR


def default_hf_home(root: Path) -> Path:
    return Path(root) / "models"


def hf_home(setting: str, root: Path) -> Path:
    """The model download folder: the Advanced setting, else <data root>\\models. It always wins over an
    HF_HOME the user may have set for other tools, so uninstall and failed-download cleanup stay ours."""
    return Path(setting) if setting and str(setting).strip() else default_hf_home(root)
