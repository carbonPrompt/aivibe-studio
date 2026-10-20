"""What graphics card this PC has, and whether a model or subject training will fit on it."""
import re
import subprocess
import sys
from typing import NamedTuple

TRAIN_MIN_VRAM_GB = 12
NO_WINDOW = 0x08000000      # CREATE_NO_WINDOW: pythonw would otherwise flash a console


class GpuInfo(NamedTuple):
    name: str
    vram_gb: float
    capability: tuple
    driver: str


def _smi_driver() -> str:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10,
                           creationflags=NO_WINDOW if sys.platform == "win32" else 0)
        return r.stdout.strip().splitlines()[0].strip() if r.returncode == 0 and r.stdout.strip() else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def detect(torch_mod=None, smi=None) -> GpuInfo | None:
    """The first CUDA card, or None when torch sees none."""
    try:
        if torch_mod is None:
            import torch as torch_mod
        if not torch_mod.cuda.is_available():
            return None
        p = torch_mod.cuda.get_device_properties(0)
        cap = tuple(torch_mod.cuda.get_device_capability(0))
    except Exception:
        return None
    return GpuInfo(p.name, round(p.total_memory / 1024 ** 3, 1), cap, (smi or _smi_driver)())


def support(info: GpuInfo | None) -> str:
    """'ok', 'unsupported_old' (GTX 16-series and older: fp16 gives black pictures) or 'no_gpu'."""
    if info is None:
        return "no_gpu"
    if tuple(info.capability) < (7, 5) or re.search(r"GTX\s*16", info.name, re.I):
        return "unsupported_old"
    return "ok"


def fits(min_vram_gb: float, info: GpuInfo | None) -> str:
    """'ok' when the card has the memory, 'tight' within 80% (offload, slower), else 'too_big'."""
    if info is None:
        return "too_big"
    if info.vram_gb >= min_vram_gb:
        return "ok"
    return "tight" if info.vram_gb >= min_vram_gb * 0.8 else "too_big"
