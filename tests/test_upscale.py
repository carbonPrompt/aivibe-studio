import math
import threading

import numpy as np
import pytest
from PIL import Image

from studio.naming import output_stem
from studio.upscale import parse_choice, target_size, suffix, model_passes, tile_upscale, upscale_image


def test_parse_choice():
    assert parse_choice("2x") == "2x" and parse_choice("4K") == "4k" and parse_choice("1080P") == "1080p"
    assert parse_choice(2400) == "2400px" and parse_choice("2400") == "2400px" and parse_choice("2400px") == "2400px"
    for bad in ("3x", "8193px", "100", "", None, "big"):
        with pytest.raises(ValueError):
            parse_choice(bad)


def test_target_size():
    assert target_size(1024, 1024, "4k") == (3840, 3840)
    assert target_size(1344, 768, "1080p") == (1920, 1097)
    assert target_size(1344, 768, "2x") == (2688, 1536)
    assert target_size(768, 1344, "2400px") == (1371, 2400)


def test_suffix():
    assert [suffix(c) for c in ("2x", "4x", "1080p", "4k", "2400px")] == ["2x", "4x", "1080p", "4k", "2400px"]


def test_model_passes():
    assert model_passes(1344, 768, 5376, 3072) == 1
    assert model_passes(1344, 768, 8192, 4681) == 2
    assert model_passes(1024, 1024, 800, 800) == 0
    assert model_passes(1024, 1024, 1024, 1024) == 0


def test_upscale_of_upscale_name():
    from datetime import datetime
    stem = "generated_x_2026-10-18_1652_4x"
    assert f"{stem}_{suffix('2x')}" == "generated_x_2026-10-18_1652_4x_2x"


def nn4(a):
    return np.repeat(np.repeat(a, 4, axis=0), 4, axis=1)


def rgb(w, h, seed=1):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), "RGB")


def test_tiles_stitch_without_seams():
    img = rgb(1000, 700)
    calls = []
    out = tile_upscale(img, nn4, scale=4, tile=256, overlap=32, on_tile=lambda i, n: calls.append((i, n)))
    whole = Image.fromarray((np.clip(nn4(np.asarray(img, np.float32) / 255), 0, 1) * 255 + 0.5).astype(np.uint8))
    assert out.size == (4000, 2800)
    assert np.abs(np.asarray(out, int) - np.asarray(whole, int)).max() == 0
    assert len(calls) == math.ceil(1000 / 256) * math.ceil(700 / 256)


def test_upscale_image_exact_size_and_modes():
    for mode in ("RGB", "RGBA", "P", "L"):
        img = rgb(300, 200).convert(mode)
        out = upscale_image(img, "1080p", nn4)
        assert out.size == (1920, 1280)
        assert ("A" in out.getbands()) == (mode == "RGBA")


def test_downscale_target_uses_no_model():
    calls = []
    def fn(a):
        calls.append(1); return nn4(a)
    out = upscale_image(rgb(1024, 1024), "800px", fn)
    assert out.size == (800, 800) and calls == []


def test_two_passes_for_huge_target():
    calls = []
    def fn(a):
        calls.append(a.shape); return nn4(a)
    out = upscale_image(rgb(64, 32), "1024px", fn, tile=512)
    assert out.size == (1024, 512) and len({c for c in calls}) >= 1 and len(calls) == 2


def test_cancel_before_start():
    ev = threading.Event(); ev.set()
    assert upscale_image(rgb(64, 64), "4x", nn4, cancel=ev) is None


def test_second_pass_starts_from_a_quarter_of_the_target():
    shapes = []
    def fn(a):
        shapes.append(a.shape[:2]); return nn4(a)
    big = rgb(1344, 768)
    out = upscale_image(big, "8192px", fn, tile=4096)
    assert out.size == (8192, 4681)
    assert shapes[0] == (768, 1344) and shapes[1] == (1171, 2048)


def test_target_over_8192_raises():
    from studio.upscale import check_target
    check_target(1344, 768, "4x")
    with pytest.raises(ValueError, match="8192"):
        check_target(5376, 3072, "4x")
