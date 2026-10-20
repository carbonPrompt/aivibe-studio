"""Face fix: find small faces in a finished picture and redraw each one at full resolution with the subject's
likeness, then blend it back. Wide shots keep the person recognisable this way."""
import os

from PIL import Image, ImageDraw, ImageFilter

MAX_FACE_FRACTION = 0.3     # faces wider than this share of the picture already get enough pixels


def face_region(face, img_w: int, img_h: int, expand: float = 2.5) -> tuple[int, int, int, int]:
    """A square around the face (face box grown by `expand`), shifted to stay inside the picture."""
    x, y, w, h = face
    side = int(min(max(w, h) * expand, img_w, img_h))
    cx, cy = x + w / 2, y + h / 2
    left = int(min(max(cx - side / 2, 0), img_w - side))
    top = int(min(max(cy - side / 2, 0), img_h - side))
    return left, top, left + side, top + side


def needs_fix(face, img_w: int, img_h: int) -> bool:
    return max(face[2] / img_w, face[3] / img_h) < MAX_FACE_FRACTION


def choose_faces(faces, img_w: int, img_h: int, people: int) -> list:
    """The faces to redraw: only when it is unambiguous whose face is whose. One person subject and exactly one
    face in the picture -> that face (if small). Anything else -> none, so a bystander never gets the subject's face."""
    if people != 1 or len(faces) != 1:
        return []
    return [f for f in faces if needs_fix(f, img_w, img_h)]


def click_region(faces, x: int, y: int, img_w: int, img_h: int, default_frac: float = 0.3):
    """The square to redraw for a click: the clicked face's region, or (a dog, a car, a face the detector
    missed) a square of `default_frac` of the short side centred on the click."""
    for f in faces:
        fx, fy, fw, fh = f
        if fx - fw * 0.25 <= x <= fx + fw * 1.25 and fy - fh * 0.25 <= y <= fy + fh * 1.25:
            return face_region(f, img_w, img_h)
    side = int(min(img_w, img_h) * default_frac)
    left = int(min(max(x - side / 2, 0), img_w - side))
    top = int(min(max(y - side / 2, 0), img_h - side))
    return left, top, left + side, top + side


def feather_mask(side: int, margin: float = 0.12) -> Image.Image:
    """White ellipse fading to black at the edges, so the redrawn square leaves no seam."""
    m = Image.new("L", (side, side), 0)
    pad = int(side * margin)
    ImageDraw.Draw(m).ellipse((pad, pad, side - pad, side - pad), fill=255)
    return m.filter(ImageFilter.GaussianBlur(max(1, pad / 2)))


def match_colour(patch: Image.Image, ref: Image.Image, amount: float = 0.6) -> Image.Image:
    """Move the patch's per-channel mean and spread part of the way toward the original area's, so a redrawn
    face keeps the scene's light and does not leave a tone step at the neck or hairline."""
    import numpy as np
    p = np.asarray(patch, dtype=np.float32)
    r = np.asarray(ref.resize(patch.size), dtype=np.float32)
    pm, ps = p.mean(axis=(0, 1)), p.std(axis=(0, 1)) + 1e-3
    rm, rs = r.mean(axis=(0, 1)), r.std(axis=(0, 1)) + 1e-3
    moved = (p - pm) / ps * rs + rm
    out = p + (moved - p) * amount
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def paste_region(base: Image.Image, patch: Image.Image, region, match: bool = True) -> Image.Image:
    l, t, r, b = region
    side = r - l
    out = base.copy()
    patch = patch.convert("RGB").resize((side, side), Image.LANCZOS)
    if match:
        patch = match_colour(patch, base.crop(region).convert("RGB"))
    out.paste(patch, (l, t), feather_mask(side))
    return out


def cpu_cv2():
    """OpenCV with OpenCL off. Left on, the face detector loads NVIDIA's OpenCL driver next to torch's CUDA, and
    a thread of that driver crashed the whole app a few seconds later (nvopencl64.dll in the 2026-10-19 dumps)."""
    os.environ.setdefault("OPENCV_OPENCL_DEVICE", "disabled")      # only read at the first import
    import cv2
    cv2.ocl.setUseOpenCL(False)
    return cv2


def find_faces(img: Image.Image) -> list[tuple[int, int, int, int]]:
    """All frontal faces (x, y, w, h), largest first. OpenCV's bundled Haar cascade."""
    import numpy as np
    cv2 = cpu_cv2()
    gray = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=6, minSize=(24, 24))
    return sorted((tuple(int(v) for v in f) for f in faces), key=lambda f: -f[2] * f[3])
