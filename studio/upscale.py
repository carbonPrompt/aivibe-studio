"""AI upscaling without the GPU parts: size choices, names, and a tiler that runs any 4x model over big images."""
import math
import re

import numpy as np
from PIL import Image

LONG_EDGE = {"1080p": 1920, "4k": 3840}
MIN_PX, MAX_PX = 512, 8192


def parse_choice(choice) -> str:
    """'2x', '4x', '1080p', '4k', or a long edge in pixels (512-8192) as '<N>px'."""
    c = str(choice if choice is not None else "").strip().lower()
    if c in ("2x", "4x", "1080p", "4k"):
        return c
    m = re.fullmatch(r"(\d+)(px)?", c)
    if m and MIN_PX <= int(m.group(1)) <= MAX_PX:
        return f"{int(m.group(1))}px"
    raise ValueError(f"upscale size must be 2x, 4x, 1080p, 4k or a long edge of {MIN_PX}-{MAX_PX} px, not {choice!r}")


def target_size(w: int, h: int, choice: str) -> tuple[int, int]:
    c = parse_choice(choice)
    if c in ("2x", "4x"):
        f = int(c[0])
        return w * f, h * f
    edge = LONG_EDGE.get(c) or int(c[:-2])
    f = edge / max(w, h)
    return (edge, int(h * f + 0.5)) if w >= h else (int(w * f + 0.5), edge)


def check_target(w: int, h: int, choice: str) -> tuple[int, int]:
    """The target size, or ValueError if its long edge passes 8192 (a 4x of a 4x would need tens of GB)."""
    tw, th = target_size(w, h, choice)
    if max(tw, th) > MAX_PX:
        raise ValueError(f"{choice} of a {w}x{h} image would be {tw}x{th}; the limit is {MAX_PX} px on the long edge")
    return tw, th


def suffix(choice: str) -> str:
    return parse_choice(choice)


def model_passes(w: int, h: int, tw: int, th: int) -> int:
    f = max(tw / w, th / h)
    if f <= 1:
        return 0
    return 1 if f <= 4 else 2


def tile_upscale(img: Image.Image, model_fn, scale: int = 4, tile: int = 512, overlap: int = 32,
                 on_tile=None, cancel=None):
    """Upscale an RGB image tile by tile. Each tile is fed with `overlap` px of context on every side and only its
    interior is kept, so tile edges never show. Returns None if cancelled."""
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    h, w = a.shape[:2]
    out = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)   # uint8: an 8192 px result stays ~0.2 GB
    nx, ny = math.ceil(w / tile), math.ceil(h / tile)
    n = 0
    for ty in range(ny):
        for tx in range(nx):
            if cancel is not None and cancel.is_set():
                return None
            x0, y0 = tx * tile, ty * tile
            x1, y1 = min(x0 + tile, w), min(y0 + tile, h)
            cx0, cy0 = max(x0 - overlap, 0), max(y0 - overlap, 0)
            cx1, cy1 = min(x1 + overlap, w), min(y1 + overlap, h)
            res = model_fn(a[cy0:cy1, cx0:cx1])
            ox, oy = (x0 - cx0) * scale, (y0 - cy0) * scale
            part = res[oy:oy + (y1 - y0) * scale, ox:ox + (x1 - x0) * scale]
            out[y0 * scale:y1 * scale, x0 * scale:x1 * scale] = (np.clip(part, 0, 1) * 255 + 0.5).astype(np.uint8)
            n += 1
            if on_tile:
                on_tile(n, nx * ny)
    return Image.fromarray(out, "RGB")


def upscale_image(img: Image.Image, choice: str, model_fn, on_tile=None, cancel=None, tile: int = 512):
    """Upscale to the exact target size: up to two 4x model passes, then a Lanczos resize. Alpha is resized
    with Lanczos. Returns None if cancelled."""
    if cancel is not None and cancel.is_set():
        return None
    if img.mode == "P":
        img = img.convert("RGBA" if "transparency" in img.info else "RGB")
    alpha = img.getchannel("A") if "A" in img.getbands() else None
    rgb = img.convert("RGB")
    tw, th = target_size(rgb.width, rgb.height, choice)
    passes = model_passes(rgb.width, rgb.height, tw, th)
    for i in range(passes):
        if i == 1:
            # shrink the first result so the second 4x pass lands right on the target: 16x less work than
            # upscaling the full intermediate and then throwing most of it away
            rgb = rgb.resize((math.ceil(tw / 4), math.ceil(th / 4)), Image.LANCZOS)
        rgb = tile_upscale(rgb, model_fn, tile=tile, on_tile=on_tile, cancel=cancel)
        if rgb is None:
            return None
    if rgb.size != (tw, th):
        rgb = rgb.resize((tw, th), Image.LANCZOS)
    if alpha is not None:
        rgb.putalpha(alpha.resize((tw, th), Image.LANCZOS))
    return rgb
