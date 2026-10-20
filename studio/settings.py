"""App settings, one JSON file in the data folder."""
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path


def default_output_root() -> str:
    return str(Path.home() / "Pictures" / "aiVibe Studio")


@dataclass
class Settings:
    output_root: str = field(default_factory=default_output_root)
    idle_minutes: int = 5
    theme: str = "dark"
    last: dict = field(default_factory=dict)
    # Advanced (blank = off / default)
    extra_model_folders: list = field(default_factory=list)   # read-only folders with diffusers models
    gpu_marker_file: str = ""       # busy marker another GPU job runner on this PC reads
    gpu_jobs_folder: str = ""       # that runner's jobs folder: wait while one of its jobs runs
    hf_cache_dir: str = ""          # model download folder; blank = <data>\models
    update_check: bool = True       # look for a new release when the app starts


def load_settings(path: Path) -> Settings:
    path = Path(path)
    if not path.exists():
        return Settings()
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return Settings(**{k: v for k, v in d.items() if k in Settings.__dataclass_fields__})
    except (ValueError, TypeError):
        os.replace(path, path.with_name(path.name + ".bad"))
        return Settings()


def save_settings(path: Path, s: Settings) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(asdict(s), indent=2), encoding="utf-8")
    os.replace(tmp, path)
