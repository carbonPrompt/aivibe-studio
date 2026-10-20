from PIL import Image

from studio.facefix import face_region, needs_fix, feather_mask, paste_region


def test_region_is_square_expanded_and_inside():
    l, t, r, b = face_region((500, 300, 40, 40), 1024, 1024, expand=2.5)
    assert r - l == b - t == 100 and (l, t) == (470, 270)
    l, t, r, b = face_region((5, 5, 40, 40), 1024, 768)        # near the corner: shifted, not cut
    assert l == 0 and t == 0 and r - l == b - t == 100
    l, t, r, b = face_region((0, 0, 900, 700), 1024, 768)      # huge face: capped by the image
    assert r - l == b - t == 768 and 0 <= l and r <= 1024


def test_needs_fix_only_small_faces():
    assert needs_fix((0, 0, 100, 100), 1024, 1024)
    assert not needs_fix((0, 0, 400, 400), 1024, 1024)


def test_feather_mask_soft_edges():
    m = feather_mask(100)
    assert m.size == (100, 100) and m.getpixel((50, 50)) == 255 and m.getpixel((0, 0)) == 0
    assert 0 < m.getpixel((50, 3)) < 255


def test_paste_region_changes_only_inside():
    base = Image.new("RGB", (200, 200), "black")
    patch = Image.new("RGB", (512, 512), "white")
    out = paste_region(base, patch, (50, 50, 150, 150), match=False)     # geometry only
    assert out.getpixel((100, 100)) == (255, 255, 255) and out.getpixel((10, 10)) == (0, 0, 0)
    assert out.getpixel((52, 100))[0] < 128        # feathered edge, no hard seam
    assert base.getpixel((100, 100)) == (0, 0, 0)  # original untouched


def test_choose_faces_only_when_unambiguous():
    from studio.facefix import choose_faces
    small = [(10, 10, 50, 50)]
    assert choose_faces(small, 1024, 1024, people=1) == small
    assert choose_faces(small + [(500, 10, 50, 50)], 1024, 1024, people=1) == []   # who is Alex? skip
    assert choose_faces([(10, 10, 600, 600)], 1024, 1024, people=1) == []           # close-up: no need
    assert choose_faces(small, 1024, 1024, people=0) == []


def test_click_region_uses_the_clicked_face_or_a_default_square():
    from studio.facefix import click_region
    faces = [(100, 100, 60, 60), (600, 120, 50, 50)]
    assert click_region(faces, 620, 140, 1024, 1024) == face_region((600, 120, 50, 50), 1024, 1024)
    l, t, r, b = click_region(faces, 400, 800, 1024, 1024)           # no face there (a dog, say)
    assert r - l == b - t == int(1024 * 0.3) and l <= 400 <= r and t <= 800 <= b
    l, t, r, b = click_region([], 1020, 1020, 1344, 768)              # corner: square stays inside
    assert r <= 1344 and b <= 768 and r - l == int(768 * 0.3)


def test_paste_matches_colour_to_the_surroundings():
    from studio.facefix import paste_region
    import numpy as np
    base = Image.new("RGB", (200, 200), (120, 80, 60))
    patch = Image.new("RGB", (512, 512), (200, 170, 150))            # much paler redraw
    plain = np.asarray(paste_region(base, patch, (50, 50, 150, 150), match=False))[100, 100]
    matched = np.asarray(paste_region(base, patch, (50, 50, 150, 150)))[100, 100]
    assert abs(int(matched[0]) - 120) < abs(int(plain[0]) - 120)      # pulled toward the scene's tones


def test_face_detectors_turn_opencl_off_first(monkeypatch):
    """OpenCV's OpenCL loads NVIDIA's OpenCL driver next to CUDA, and that crashed the app (2026-10-19)."""
    import sys
    import types
    calls = []

    class Cascade:
        def __init__(self, path):
            pass

        def detectMultiScale(self, gray, **kw):
            calls.append("detect")
            return [(10, 10, 30, 30)]

    fake = types.SimpleNamespace(
        ocl=types.SimpleNamespace(setUseOpenCL=lambda on: calls.append(("opencl", on))),
        data=types.SimpleNamespace(haarcascades=""), CascadeClassifier=Cascade, COLOR_RGB2GRAY=0,
        cvtColor=lambda a, code: a[..., 0], resize=lambda a, *x, **k: a)
    monkeypatch.setitem(sys.modules, "cv2", fake)
    monkeypatch.delenv("OPENCV_OPENCL_DEVICE", raising=False)
    from studio.facefix import find_faces
    from studio.subjects import find_face
    assert find_faces(Image.new("RGB", (64, 64))) == [(10, 10, 30, 30)]
    assert find_face(Image.new("RGB", (64, 64))) is not None
    assert calls == [("opencl", False), "detect", ("opencl", False), "detect"]
    import os
    assert os.environ["OPENCV_OPENCL_DEVICE"] == "disabled"
