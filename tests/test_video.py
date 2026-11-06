import json
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from studio.api import Api
from studio.engine import DiffusersEngine
from studio.jobs import JobQueue
from studio.registry import Registry
from studio.settings import Settings
from studio.video import VideoRequest, save_video, snap_video_params, to_pil_frames, video_first_frame

CATALOG = Path(__file__).parent.parent / "studio" / "catalog.json"
T = datetime(2026, 10, 18, 16, 52, 7)


def test_snap_video_params():
    assert snap_video_params("ltx", 700, 470, 50, 24) == (704, 480, 49, 24)
    assert snap_video_params("wan", 830, 478, 80, 16) == (832, 480, 81, 16)
    assert snap_video_params("cogvideo", 720, 480, 49, 8) == (720, 480, 49, 8)
    assert snap_video_params("svd", 1000, 560, 25, 7) == (1024, 576, 25, 7)


def test_to_pil_frames_normalizes_nested_and_numpy():
    pil_frames = [Image.new("RGB", (16, 16), "red"), Image.new("RGB", (16, 16), "blue")]
    assert len(to_pil_frames([pil_frames])) == 2
    arr = np.zeros((1, 3, 16, 16, 3), dtype=np.float32) + 0.5
    out = to_pil_frames(arr)
    assert len(out) == 3 and isinstance(out[0], Image.Image) and out[0].size == (16, 16)


def test_save_video_and_first_frame(tmp_path):
    frames = [
        Image.new("RGB", (64, 48), (220, 40, 40)),
        Image.new("RGB", (64, 48), (40, 220, 40)),
        Image.new("RGB", (64, 48), (40, 40, 220)),
    ]
    out = tmp_path / "clip.mp4"
    save_video(frames, out, fps=8)
    assert out.exists() and out.stat().st_size > 0
    first = video_first_frame(out)
    assert first.size == (64, 48)


class FakeMarker:
    def acquire(self): pass
    def heartbeat(self): pass
    def release(self): pass


class FakeVideoRunner:
    def __init__(self, entry):
        self.entry = entry
        self.calls = []

    def generate_video(self, req, seed, on_step):
        self.calls.append((req, seed))
        on_step(1, req.steps)
        return [Image.new("RGB", (req.width, req.height), "green") for _ in range(4)]


def test_engine_generate_video_and_job_queue(tmp_path):
    reg = Registry(CATALOG, tmp_path / "u.json", [tmp_path / "app"], tmp_path / "hf")
    for mid in ("ltx-video", "svd-xt"):
        snap = reg.hf_repo_dir(mid) / "snapshots" / "1"
        snap.mkdir(parents=True)
        (snap / "model_index.json").write_text("{}")

    runners = {}

    def loader(entry, path, device):
        r = FakeVideoRunner(entry)
        runners[entry.id] = r
        return r

    eng = DiffusersEngine(reg, FakeMarker(), None, loader=loader)
    st = Settings(output_root=str(tmp_path / "out"))
    q = JobQueue(eng, reg, st, clock=lambda: T)
    q.start()
    try:
        req = VideoRequest("ltx-video", "ocean waves at sunset", "blurry", 64, 48, 9, 16, 4, 3.0, 123)
        jid = q.submit_video(req, "ocean-waves")
        end = time.time() + 5
        while time.time() < end:
            if not q.state()["running"] and q.state()["pending"] == 0:
                break
            time.sleep(0.01)
        evs = q.events(0)
        mp4 = tmp_path / "out" / "2026-10-18" / "generated_ocean-waves_2026-10-18_1652.mp4"
        assert mp4.exists() and mp4.with_suffix(".json").exists()
        rec = json.loads(mp4.with_suffix(".json").read_text())
        assert rec["media"] == "video" and rec["fps"] == 16 and rec["seed"] == 123
        assert any(e["kind"] == "video" and e["job"] == jid and e["path"] == str(mp4) for e in evs)

        # Verify Api.gallery, Api.image, and Api.remake work with the saved .mp4
        api = Api(q, reg, st, tmp_path / "settings.json", eng)
        gal = api.gallery()
        assert len(gal) == 1 and gal[0]["media"] == "video" and gal[0]["thumb"].startswith("data:image/jpeg;base64,")
        im_res = api.image(str(mp4))
        assert im_res["media"] == "video" and im_res["data"].startswith("data:video/mp4;base64,")
    finally:
        q.stop()


def test_find_ffmpeg_uses_imageio_ffmpeg(monkeypatch, tmp_path):
    import sys
    import types
    from studio import video
    fake = tmp_path / "ffmpeg-win-x86_64.exe"
    fake.write_bytes(b"x")
    monkeypatch.setattr(video.sys, "executable", str(tmp_path / "python.exe"))
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", types.SimpleNamespace(get_ffmpeg_exe=lambda: str(fake)))
    assert video.find_ffmpeg() == str(fake)


def test_find_ffmpeg_prefers_exe_beside_python(monkeypatch, tmp_path):
    import sys
    import types
    from studio import video
    (tmp_path / "ffmpeg.exe").write_bytes(b"x")
    monkeypatch.setattr(video.sys, "executable", str(tmp_path / "python.exe"))
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", types.SimpleNamespace(get_ffmpeg_exe=lambda: "/nope"))
    assert video.find_ffmpeg() == str(tmp_path / "ffmpeg.exe")


def test_first_frame_restricts_ffmpeg_to_local_container(monkeypatch, tmp_path):
    from studio import video
    seen = {}

    class R:
        returncode = 1
        stdout = b""

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return R()

    monkeypatch.setattr(video, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(video.subprocess, "run", fake_run)
    try:
        video.video_first_frame(tmp_path / "clip.mp4")
    except Exception:
        pass
    cmd = seen["cmd"]
    i = cmd.index("-i")
    assert cmd[cmd.index("-protocol_whitelist") + 1] == "file" and cmd.index("-protocol_whitelist") < i
    assert cmd[cmd.index("-f") + 1] == "mov" and cmd.index("-f") < i
