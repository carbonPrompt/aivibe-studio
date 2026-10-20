import json
from pathlib import Path
from types import SimpleNamespace

from studio import hardware
from studio.hardware import GpuInfo

CATALOG = Path(__file__).parent.parent / "studio" / "catalog.json"


def fake_torch(name, vram_gb, cap, available=True):
    props = SimpleNamespace(name=name, total_memory=int(vram_gb * 1024 ** 3))
    cuda = SimpleNamespace(is_available=lambda: available, get_device_properties=lambda i: props,
                           get_device_capability=lambda i: cap)
    return SimpleNamespace(cuda=cuda)


def test_detect_none_without_cuda():
    assert hardware.detect(torch_mod=fake_torch("x", 8, (8, 6), available=False), smi=lambda: "") is None


def test_detect_reads_card_and_driver():
    info = hardware.detect(torch_mod=fake_torch("NVIDIA GeForce RTX 4070", 12, (8, 9)), smi=lambda: "566.36")
    assert info == GpuInfo("NVIDIA GeForce RTX 4070", 12.0, (8, 9), "566.36")


def test_gtx1660_unsupported():
    assert hardware.support(GpuInfo("NVIDIA GeForce GTX 1660", 6.0, (7, 5), "")) == "unsupported_old"


def test_old_capability_unsupported():
    assert hardware.support(GpuInfo("NVIDIA GeForce GTX 1080", 8.0, (6, 1), "")) == "unsupported_old"


def test_rtx4070_ok():
    assert hardware.support(GpuInfo("NVIDIA GeForce RTX 4070", 12.0, (8, 9), "")) == "ok"


def test_no_gpu():
    assert hardware.support(None) == "no_gpu"


def test_fits_thresholds():
    info = GpuInfo("RTX", 12.0, (8, 6), "")
    assert hardware.fits(11, info) == "ok"
    assert hardware.fits(9, info) == "ok"
    assert hardware.fits(13, info) == "tight"
    assert hardware.fits(16, info) == "too_big"
    assert hardware.fits(4, None) == "too_big"


def test_catalog_every_model_has_min_vram():
    for m in json.loads(CATALOG.read_text(encoding="utf-8"))["models"]:
        assert m.get("min_vram_gb", 0) > 0, m["id"]
