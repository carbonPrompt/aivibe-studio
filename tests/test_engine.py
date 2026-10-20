import threading
import time
import pytest
from pathlib import Path
from PIL import Image
from studio.engine import DiffusersEngine, GenRequest, ModelLoadError, GpuBusyError
from studio.registry import Registry

CATALOG = Path(__file__).parent.parent / "studio" / "catalog.json"


class FakeMarker:
    def __init__(self): self.log = []
    def acquire(self): self.log.append("acquire")
    def heartbeat(self): self.log.append("hb")
    def release(self): self.log.append("release")


class FakePipe:
    def __init__(self, entry): self.entry, self.calls, self.configs = entry, [], []
    def configure(self, subjects, reference, strength): self.configs.append((list(subjects), reference, strength))
    def generate(self, req, seed, on_step):
        if req.prompt == "oom" and self.calls:
            raise RuntimeError("CUDA out of memory")
        self.calls.append(seed)
        for s in range(req.steps):
            on_step(s + 1, req.steps)
        return Image.new("RGB", (req.width, req.height))


@pytest.fixture
def env(tmp_path, monkeypatch):
    reg = Registry(CATALOG, tmp_path / "u.json", [tmp_path / "app"], tmp_path / "hf")
    loads = []
    def loader(entry, path, device):
        loads.append(entry.id if device == "cuda" else f"{entry.id}@{device}")
        if entry.id == "sd15":
            raise RuntimeError("weights missing")
        return FakePipe(entry)
    marker = FakeMarker()
    clock = Clock()
    eng = DiffusersEngine(reg, marker, tmp_path / "jobs", idle_minutes=5, loader=loader, clock=lambda: clock.now)
    return eng, loads, marker, clock


class Clock:
    now = 1000.0


def req(model="realvis-v5", count=1, seed=7, steps=2):
    return GenRequest(model, "a dinosaur", "", 64, 64, steps, 6.0, seed, count)


def test_same_model_loads_once(env):
    eng, loads, _, _ = env
    eng.generate(req(), lambda a, b: None, threading.Event())
    eng.generate(req(), lambda a, b: None, threading.Event())
    assert loads == ["realvis-v5"] and eng.loaded_model() == "realvis-v5"


def test_switch_model_unloads_first(env):
    eng, loads, marker, _ = env
    eng.generate(req(), lambda a, b: None, threading.Event())
    eng.generate(req("flux-schnell"), lambda a, b: None, threading.Event())
    assert loads == ["realvis-v5", "flux-schnell"]
    assert marker.log.count("release") == 1 and marker.log.count("acquire") == 2


def test_seeds_and_steps(env):
    eng, _, marker, _ = env
    steps = []
    imgs = eng.generate(req(count=3, seed=10), lambda a, b: steps.append((a, b)), threading.Event())
    assert len(imgs) == 3 and eng._pipe.calls == [10, 11, 12]
    assert steps[:2] == [(1, 2), (2, 2)] and "hb" in marker.log


def test_idle_unload(env):
    eng, _, marker, clock = env
    eng.generate(req(), lambda a, b: None, threading.Event())
    clock.now += 299; eng.idle_tick(clock.now)
    assert eng.loaded_model() == "realvis-v5"
    clock.now += 2; eng.idle_tick(clock.now)
    assert eng.loaded_model() is None and marker.log[-1] == "release"


def test_load_error(env):
    eng, _, _, _ = env
    with pytest.raises(ModelLoadError, match="sd15"):
        eng.generate(req("sd15"), lambda a, b: None, threading.Event())
    assert eng.loaded_model() is None


def test_waits_for_other_jobs_then_gives_up(env, monkeypatch):
    import studio.engine as e
    eng, loads, _, _ = env
    monkeypatch.setattr(e, "other_job_running", lambda root: True)
    eng.busy_poll, eng.busy_timeout = 0.0, 0.01
    with pytest.raises(GpuBusyError):
        eng.generate(req(), lambda a, b: None, threading.Event())
    assert loads == []


def test_cancel_between_images(env):
    eng, _, _, _ = env
    cancel = threading.Event()
    def on_step(a, b):
        if a == b: cancel.set()
    imgs = eng.generate(req(count=3), on_step, cancel)
    assert len(imgs) == 1


def test_unload_without_model_is_noop(env):
    eng, _, marker, _ = env
    eng.unload()
    assert marker.log == []


def test_set_device_unloads_and_reloads_on_cpu(env):
    eng, loads, marker, _ = env
    eng.generate(req(), lambda a, b: None, threading.Event())
    eng.set_device("cpu")
    assert eng.loaded_model() is None
    eng.generate(req(), lambda a, b: None, threading.Event())
    assert loads == ["realvis-v5", "realvis-v5@cpu"]
    eng.set_device("cpu")
    assert eng.loaded_model() == "realvis-v5"


def test_cancel_while_waiting_for_other_jobs(env, monkeypatch):
    import studio.engine as e
    eng, loads, _, _ = env
    monkeypatch.setattr(e, "other_job_running", lambda root: True)
    eng.busy_poll, eng.busy_timeout = 0.01, 60
    cancel = threading.Event(); cancel.set()
    t0 = time.time()
    assert eng.generate(req(), lambda a, b: None, cancel) == [] and time.time() - t0 < 2 and loads == []


def test_on_image_gets_images_before_a_failure(env):
    eng, _, _, _ = env
    got = []
    r = GenRequest("realvis-v5", "oom", "", 64, 64, 1, 6.0, 1, 3)
    with pytest.raises(RuntimeError):
        eng.generate(r, lambda a, b: None, threading.Event(), on_image=lambda img, k: got.append(k))
    assert got == [0]


def test_heartbeat_runs_while_model_loads(tmp_path):
    reg = Registry(CATALOG, tmp_path / "u.json", [tmp_path / "app"], tmp_path / "hf")
    marker = FakeMarker()
    def slow(entry, path, device):
        time.sleep(0.4)
        return FakePipe(entry)
    eng = DiffusersEngine(reg, marker, tmp_path / "jobs", loader=slow, heartbeat_interval=0.05)
    eng.generate(req(steps=0), lambda a, b: None, threading.Event())
    assert marker.log.count("hb") >= 3
    eng.unload()


# ---- phase 2: upscaler slot ----
import numpy as np


def _up_env(tmp_path, loads):
    reg = Registry(CATALOG, tmp_path / "u.json", [tmp_path / "app"], tmp_path / "hf")
    marker = FakeMarker()
    def up_loader(device):
        loads.append(device)
        return lambda a: np.repeat(np.repeat(a, 4, axis=0), 4, axis=1)
    eng = DiffusersEngine(reg, marker, tmp_path / "jobs", loader=lambda e, p, d: FakePipe(e),
                          upscaler_loader=up_loader, heartbeat_interval=60)
    return eng, marker


def test_upscale_loads_once_and_holds_marker(tmp_path):
    loads = []
    eng, marker = _up_env(tmp_path, loads)
    img = Image.new("RGB", (64, 32))
    a = eng.upscale(img, "4x", lambda i, n: None, threading.Event())
    b = eng.upscale(img, "2x", lambda i, n: None, threading.Event())
    assert a.size == (256, 128) and b.size == (128, 64) and loads == ["cuda"]
    assert "acquire" in marker.log
    eng.unload()


def test_idle_unloads_upscaler(tmp_path):
    loads = []
    eng, marker = _up_env(tmp_path, loads)
    eng.upscale(Image.new("RGB", (32, 32)), "4x", lambda i, n: None, threading.Event())
    eng.idle_tick(time.time() + 3600)
    assert eng._upscaler is None and marker.log[-1] == "release"
    eng.upscale(Image.new("RGB", (32, 32)), "4x", lambda i, n: None, threading.Event())
    assert loads == ["cuda", "cuda"]


def test_upscale_waits_for_other_jobs_and_cancel_returns_none(tmp_path, monkeypatch):
    import studio.engine as e
    loads = []
    eng, _ = _up_env(tmp_path, loads)
    monkeypatch.setattr(e, "other_job_running", lambda root: True)
    eng.busy_poll = 0.01
    c = threading.Event(); c.set()
    assert eng.upscale(Image.new("RGB", (32, 32)), "4x", lambda i, n: None, c) is None and loads == []


def test_generate_after_upscale_still_works(tmp_path):
    loads = []
    eng, _ = _up_env(tmp_path, loads)
    eng.upscale(Image.new("RGB", (32, 32)), "4x", lambda i, n: None, threading.Event())
    assert len(eng.generate(req(), lambda a, b: None, threading.Event())) == 1
    assert eng._upscaler is not None
    eng.unload()


def test_set_device_drops_upscaler(tmp_path):
    loads = []
    eng, _ = _up_env(tmp_path, loads)
    eng.upscale(Image.new("RGB", (32, 32)), "4x", lambda i, n: None, threading.Event())
    eng.set_device("cpu")
    assert eng._upscaler is None
    eng.upscale(Image.new("RGB", (32, 32)), "4x", lambda i, n: None, threading.Event())
    assert loads == ["cuda", "cpu"]


def test_failed_model_load_keeps_marker_while_upscaler_loaded(tmp_path):
    loads = []
    eng, marker = _up_env(tmp_path, loads)
    eng.loader = lambda e, p, d: (_ for _ in ()).throw(RuntimeError("missing weights"))
    eng.upscale(Image.new("RGB", (32, 32)), "4x", lambda i, n: None, threading.Event())
    with pytest.raises(ModelLoadError):
        eng.generate(req(), lambda a, b: None, threading.Event())
    assert marker.log[-1] != "release" and eng._beat_stop is not None
    eng.unload()


def test_exclusive_unloads_and_holds_marker(tmp_path):
    loads = []
    eng, marker = _up_env(tmp_path, loads)
    eng.upscale(Image.new("RGB", (32, 32)), "4x", lambda i, n: None, threading.Event())
    with eng.exclusive(threading.Event()) as ok:
        assert ok and eng._upscaler is None and eng._pipe is None
        assert marker.log[-1] == "acquire" and eng._beat_stop is not None
    assert marker.log[-1] == "release" and eng._beat_stop is None


def test_exclusive_cancelled_while_waiting(tmp_path, monkeypatch):
    import studio.engine as e
    eng, marker = _up_env(tmp_path, [])
    monkeypatch.setattr(e, "other_job_running", lambda root: True)
    eng.busy_poll = 0.01
    c = threading.Event(); c.set()
    with eng.exclusive(c) as ok:
        assert ok is False
    assert "acquire" not in marker.log



# ---- phase 3: subjects + reference ----
def sreq(model="realvis-v5", subjects=(), reference=None, strength=0.5):
    r = GenRequest(model, "a man", "", 64, 64, 1, 6.0, 1, 1)
    r.subjects = list(subjects); r.reference = reference; r.reference_strength = strength
    return r


def test_subjects_and_reference_passed_to_runner(env):
    eng, _, _, _ = env
    eng.generate(sreq(subjects=[("alex", "/s/alex.safetensors", 0.9)], reference="/r.png", strength=0.6),
                 lambda a, b: None, threading.Event())
    assert eng._pipe.configs[-1] == ([("alex", "/s/alex.safetensors", 0.9)], "/r.png", 0.6)
    eng.generate(sreq(), lambda a, b: None, threading.Event())
    assert eng._pipe.configs[-1] == ([], None, 0.5)


def test_subjects_refused_on_non_sdxl(env):
    eng, _, _, _ = env
    with pytest.raises(ValueError, match="SDXL"):
        eng.generate(sreq("z-image-turbo", subjects=[("alex", "/x", 0.9)]), lambda a, b: None, threading.Event())
    with pytest.raises(ValueError, match="SDXL"):
        eng.generate(sreq("flux-schnell", reference="/r.png"), lambda a, b: None, threading.Event())


def test_more_than_three_subjects_refused(env):
    eng, _, _, _ = env
    with pytest.raises(ValueError, match="3"):
        eng.generate(sreq(subjects=[(f"s{i}", "/x", 0.9) for i in range(4)]), lambda a, b: None, threading.Event())


def test_face_fix_runs_only_with_subjects_when_asked(env):
    eng, _, _, _ = env
    calls = []
    def ff(img, req, seed, cancel=None):
        calls.append(seed); return img.convert("L").convert("RGB")
    eng.generate(sreq(), lambda a, b: None, threading.Event())          # no subjects: loads the pipe
    eng._pipe.face_fix = ff
    eng.generate(sreq(), lambda a, b: None, threading.Event())
    r = sreq(subjects=[("alex", "/s", 1.0)]); r.face_fix = False
    eng.generate(r, lambda a, b: None, threading.Event())
    assert calls == []
    r.face_fix = True
    eng.generate(r, lambda a, b: None, threading.Event())
    assert calls == []                  # no face prompt (queue sets it only for one person subject)
    r.face_prompt = "photo of zqabc man, face"
    got = []
    eng.generate(r, lambda a, b: None, threading.Event(), on_image=lambda img, k: got.append(img.mode))
    assert calls == [1] and got == ["RGB"]


def test_cancel_stops_inside_an_image(env):
    """Cancel must stop the denoising loop at the next step, not after the picture."""
    eng, _, _, _ = env
    cancel, seen = threading.Event(), []
    def on_step(a, b):
        seen.append(a)
        if a == 3: cancel.set()
    imgs = eng.generate(req(count=2, steps=30), on_step, cancel)
    assert imgs == [] and seen == [1, 2, 3]
    assert eng.generate(req(), lambda a, b: None, threading.Event())     # engine still usable afterwards


def test_engine_no_jobs_root_never_waits(tmp_path, monkeypatch):
    import studio.engine as e
    reg = Registry(CATALOG, tmp_path / "u.json", [], tmp_path / "hf")
    monkeypatch.setattr(e, "other_job_running", lambda root: (_ for _ in ()).throw(AssertionError("checked")))
    eng = DiffusersEngine(reg, FakeMarker(), None, loader=lambda entry, path, device: FakePipe(entry))
    assert eng._wait_for_other_jobs(threading.Event()) is True
