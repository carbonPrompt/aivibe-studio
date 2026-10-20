import json
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image

from studio.batch import BatchItem
from studio.engine import GenRequest, ModelLoadError, GpuBusyError
from studio.jobs import JobQueue
from studio.registry import Registry
from studio.settings import Settings

CATALOG = Path(__file__).parent.parent / "studio" / "catalog.json"
T = datetime(2026, 10, 18, 16, 52, 7)


class FakeEngine:
    def __init__(self, fail_load=(), fail_prompt=(), busy=()):
        self.fail_load, self.fail_prompt, self.loads, self.model = set(fail_load), set(fail_prompt), [], None
        self.busy = set(busy)
        self.ticks = 0
    def generate(self, req, on_step, cancel, on_image=None):
        if req.model_id != self.model:
            if req.model_id in self.busy:
                raise GpuBusyError("busy")
            if req.model_id in self.fail_load:
                raise ModelLoadError(f"could not load {req.model_id}: boom")
            self.loads.append(req.model_id); self.model = req.model_id
        if req.prompt in self.fail_prompt:
            raise RuntimeError("CUDA out of memory")
        out = []
        for k in range(req.count):
            if cancel.is_set():
                break
            if req.prompt == "oom-on-second" and k == 1:
                raise RuntimeError("CUDA out of memory")
            on_step(1, 1)
            img = Image.new("RGB", (8, 8))
            out.append(img)
            if on_image:
                on_image(img, k)
            if req.prompt == "cancel-after-first":
                cancel.set()
        return out
    def upscale(self, img, choice, on_tile, cancel):
        if cancel.is_set():
            return None
        from studio.upscale import target_size
        on_tile(1, 1)
        out = img.convert("RGB").resize(target_size(img.width, img.height, choice))
        if getattr(self, "cancel_after_upscale", False):
            cancel.set()
        return out
    def unload(self): self.model = None
    def loaded_model(self): return self.model
    def idle_tick(self, now): self.ticks += 1


@pytest.fixture
def make(tmp_path):
    reg = Registry(CATALOG, tmp_path / "u.json", [tmp_path / "app"], tmp_path / "hf")
    def mk(engine=None, batch_subfolder=False):     # older tests read batch output from the day folder
        st = Settings(output_root=str(tmp_path / "out")); st.last["batch_subfolder"] = batch_subfolder
        q = JobQueue(engine or FakeEngine(), reg, st, clock=lambda: T)
        q.start()
        return q
    yield mk


def wait(q, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        if not q.state()["running"] and q.state()["pending"] == 0:
            return q.events(0)
        time.sleep(0.01)
    raise AssertionError("queue did not finish")


def item(n, prompt, model="realvis-v5", count=1, subject=None, seed=5):
    return BatchItem(n, prompt, "", model, 64, 64, count, seed, 2, 6.0, subject or prompt.split()[0])


def day(tmp_path):
    return tmp_path / "out" / "2026-10-18"


def test_single_job_writes_png_and_recipe(make, tmp_path):
    q = make()
    q.submit_single(GenRequest("realvis-v5", "a dinosaur", "blurry", 64, 64, 2, 6.0, 42, 1), "dino")
    ev = wait(q)
    png = day(tmp_path) / "generated_dino_2026-10-18_1652.png"
    assert png.exists()
    r = json.loads(png.with_suffix(".json").read_text())
    assert (r["seed"], r["model_name"], r["license"], r["negative"]) == (42, "RealVisXL V5.0", "commercial", "blurry")
    assert [e["kind"] for e in ev][-1] == "done" and any(e["kind"] == "image" and e["path"] == str(png) for e in ev)


def test_random_seed_recorded(make, tmp_path):
    q = make()
    q.submit_single(GenRequest("realvis-v5", "a cat", "", 64, 64, 2, 6.0, None, 1), "cat")
    wait(q)
    r = json.loads((day(tmp_path) / "generated_cat_2026-10-18_1652.json").read_text())
    assert isinstance(r["seed"], int) and 0 <= r["seed"] < 2**32


def test_count_numbers_files(make, tmp_path):
    q = make()
    q.submit_single(GenRequest("realvis-v5", "a cat", "", 64, 64, 2, 6.0, 1, 3), "cat")
    wait(q)
    names = sorted(p.name for p in day(tmp_path).glob("*.png"))
    assert names == [f"generated_cat_2026-10-18_1652_{i}.png" for i in (1, 2, 3)]
    seeds = [json.loads((day(tmp_path) / n.replace(".png", ".json")).read_text())["seed"] for n in names]
    assert seeds == [1, 2, 3]


def test_batch_groups_models(make, tmp_path):
    eng = FakeEngine()
    q = make(eng)
    q.submit_batch([item(1, "a one"), item(2, "b two", "sd15"), item(3, "c three")], "space.json")
    wait(q)
    assert eng.loads == ["realvis-v5", "sd15"] and len(list(day(tmp_path).glob("*.png"))) == 3
    r = json.loads(next(day(tmp_path).glob("generated_b_*.json")).read_text())
    assert r["batch_file"] == "space.json"


def test_model_load_failure_skips_only_that_model(make, tmp_path):
    q = make(FakeEngine(fail_load={"sd15"}))
    q.submit_batch([item(1, "a one", "sd15"), item(2, "b two", "sd15"), item(3, "c three")], "x.json")
    ev = wait(q)
    assert [e["index"] for e in ev if e["kind"] == "model_failed"] == [1, 2]
    assert [p.name.split("_")[1] for p in day(tmp_path).glob("*.png")] == ["c"]
    assert ev[-1]["kind"] == "done"


def test_item_failure_continues(make, tmp_path):
    q = make(FakeEngine(fail_prompt={"b two"}))
    q.submit_batch([item(1, "a one"), item(2, "b two"), item(3, "c three")], "x.json")
    ev = wait(q)
    failed = [e for e in ev if e["kind"] == "item_failed"]
    assert len(failed) == 1 and failed[0]["index"] == 2 and "out of memory" in failed[0]["error"]
    assert len(list(day(tmp_path).glob("*.png"))) == 2


def test_cancel_stops_after_current_image(make, tmp_path):
    q = make()
    q.submit_batch([item(1, "cancel-after-first", count=3, subject="x"), item(2, "b two")], "x.json")
    ev = wait(q)
    assert len(list(day(tmp_path).glob("*.png"))) == 1 and ev[-1]["kind"] == "cancelled"


def test_cancel_call_drops_queue(make, tmp_path):
    q = JobQueue(FakeEngine(), Registry(CATALOG, tmp_path / "u.json", [tmp_path / "a"], tmp_path / "h"),
                 Settings(output_root=str(tmp_path / "out")), clock=lambda: T)
    q.submit_batch([item(1, "a one"), item(2, "b two")], "x.json")
    q.cancel()
    q.start()
    ev = wait(q)
    assert not list((tmp_path / "out").rglob("*.png")) and any(e["kind"] == "cancelled" for e in ev)


def test_same_minute_never_overwrites(make, tmp_path):
    q = make()
    for _ in range(2):
        q.submit_single(GenRequest("realvis-v5", "a cat", "", 64, 64, 2, 6.0, 1, 1), "cat")
    wait(q)
    assert len(list(day(tmp_path).glob("*.png"))) == 2 and len(list(day(tmp_path).glob("*.json"))) == 2


def test_events_since(make):
    q = make()
    q.submit_single(GenRequest("realvis-v5", "a cat", "", 64, 64, 2, 6.0, 1, 1), "cat")
    ev = wait(q)
    last = ev[-1]["seq"]
    assert q.events(last) == [] and q.events(ev[0]["seq"])[0]["seq"] == ev[1]["seq"]


def test_note_adds_event(make):
    q = make()
    q.note("download", model="flux-schnell", percent=40)
    assert q.events(0)[-1]["kind"] == "download" and q.events(0)[-1]["percent"] == 40


def test_removed_model_does_not_kill_worker(make, tmp_path):
    q = make()
    m = q.registry.add_hf("a/b", "B", "sdxl", "nc")
    q.registry.remove_user(m.id)
    q.submit_batch([item(1, "a one", m.id)], "x.json")
    q.submit_single(GenRequest("realvis-v5", "a cat", "", 64, 64, 2, 6.0, 1, 1), "cat")
    ev = wait(q)
    assert any(e["kind"] in ("item_failed", "model_failed") for e in ev)
    assert (day(tmp_path) / "generated_cat_2026-10-18_1652.png").exists()


def test_unwritable_output_does_not_kill_worker(make, tmp_path):
    q = make()
    (tmp_path / "out").write_text("i am a file, not a folder")
    q.submit_single(GenRequest("realvis-v5", "a cat", "", 64, 64, 2, 6.0, 1, 1), "cat")
    ev = wait(q)
    assert any(e["kind"] == "item_failed" and "save" in e["error"] for e in ev)
    (tmp_path / "out").unlink()
    q.submit_single(GenRequest("realvis-v5", "a dog", "", 64, 64, 2, 6.0, 1, 1), "dog")
    wait(q)
    assert (day(tmp_path) / "generated_dog_2026-10-18_1652.png").exists()


def test_gpu_busy_skips_rest_of_model_group(make, tmp_path):
    q = make(FakeEngine(busy={"sd15"}))
    q.submit_batch([item(1, "a one", "sd15"), item(2, "b two", "sd15"), item(3, "c three")], "x.json")
    ev = wait(q)
    assert [e["index"] for e in ev if e["kind"] == "model_failed"] == [1, 2]
    assert len(list(day(tmp_path).glob("*.png"))) == 1


def test_images_before_a_failure_are_kept(make, tmp_path):
    q = make()
    q.submit_batch([item(1, "oom-on-second", count=3, subject="x")], "x.json")
    ev = wait(q)
    assert len(list(day(tmp_path).glob("*.png"))) == 1 and any(e["kind"] == "item_failed" for e in ev)


def test_save_never_overwrites_a_file_that_appears(make, tmp_path, monkeypatch):
    import studio.jobs as j
    q = make()
    d = day(tmp_path); d.mkdir(parents=True)
    taken = d / "generated_cat_2026-10-18_1652.png"; taken.write_bytes(b"keep me")
    real = j.unique_path
    calls = []
    def racy(folder, stem, ext=".png"):
        calls.append(1)
        return taken if len(calls) == 1 else real(folder, stem, ext)
    monkeypatch.setattr(j, "unique_path", racy)
    q.submit_single(GenRequest("realvis-v5", "a cat", "", 64, 64, 2, 6.0, 1, 1), "cat")
    wait(q)
    assert taken.read_bytes() == b"keep me" and len(list(d.glob("*.png"))) == 2



# ---- phase 2: upscale jobs ----
def _src(folder, name="a.png", recipe=True):
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name; Image.new("RGB", (16, 8), "blue").save(p)
    if recipe:
        from studio.recipe import Recipe, write_recipe
        write_recipe(p, Recipe("a blue thing", "", "realvis-v5", "RealVisXL V5.0", "commercial", 9, 30, 6.0, 16, 8,
                               "blue", "0.1.0", "2026-10-18T16:52:00"))
    return p


def test_upscale_writes_beside_original_with_recipe(make, tmp_path):
    q = make()
    src = _src(day(tmp_path))
    q.submit_upscale([str(src)], "4x")
    ev = wait(q)
    out = day(tmp_path) / "a_4x.png"
    assert Image.open(out).size == (64, 32) and Image.open(src).size == (16, 8)
    r = json.loads(out.with_suffix(".json").read_text())
    assert (r["prompt"], r["upscaled_from"], r["upscale"], r["upscaler"], r["out_width"], r["width"]) == \
        ("a blue thing", "a.png", "4x", "realesrgan-x4plus", 64, 16)
    assert any(e["kind"] == "upscaled" and e["path"] == str(out) and e["source"] == str(src) for e in ev)


def test_upscale_of_upscale_and_no_overwrite(make, tmp_path):
    q = make()
    src = _src(day(tmp_path))
    q.submit_upscale([str(src)], "4x"); wait(q)
    q.submit_upscale([str(day(tmp_path) / "a_4x.png")], "2x"); wait(q)
    q.submit_upscale([str(src)], "4x"); wait(q)
    names = sorted(p.name for p in day(tmp_path).glob("*.png"))
    assert "a_4x_2x.png" in names and len([n for n in names if n.startswith("a_4x") and n != "a_4x_2x.png"]) == 2


def test_upscale_without_recipe(make, tmp_path):
    q = make()
    src = _src(day(tmp_path), "dropped.png", recipe=False)
    q.submit_upscale([str(src)], "1080p"); wait(q)
    r = json.loads((day(tmp_path) / "dropped_1080p.json").read_text())
    assert r["prompt"] == "" and r["upscaled_from"] == "dropped.png" and r["out_width"] == 1920


def test_missing_source_fails_and_next_continues(make, tmp_path):
    q = make()
    good = _src(day(tmp_path))
    q.submit_upscale([str(day(tmp_path) / "gone.png"), str(good)], "2x")
    ev = wait(q)
    assert any(e["kind"] == "item_failed" for e in ev) and (day(tmp_path) / "a_2x.png").exists()


def test_cancel_during_multi_upscale(make, tmp_path):
    eng = FakeEngine(); eng.cancel_after_upscale = True
    q = make(eng)
    srcs = [_src(day(tmp_path), f"s{i}.png") for i in range(3)]
    q.submit_upscale([str(p) for p in srcs], "2x")
    ev = wait(q)
    assert len(list(day(tmp_path).glob("*_2x.png"))) == 1 and ev[-1]["kind"] == "cancelled"


def test_batch_item_upscale(make, tmp_path):
    q = make()
    it = item(1, "a cat", subject="cat"); it.upscale = "1080p"
    q.submit_batch([it], "x.json")
    wait(q)
    names = sorted(p.name for p in day(tmp_path).glob("*.png"))
    assert names == ["generated_cat_2026-10-18_1652.png", "generated_cat_2026-10-18_1652_1080p.png"]
    assert q.state()["done"] == 2


def test_upscale_over_8192_is_refused(make, tmp_path):
    q = make()
    d = day(tmp_path); d.mkdir(parents=True)
    big = d / "big.png"; Image.new("RGB", (5376, 3072)).save(big)
    q.submit_upscale([str(big)], "4x")
    ev = wait(q)
    assert any(e["kind"] == "item_failed" and "8192" in e["error"] for e in ev)
    assert not list(d.glob("big_4x*"))


# ---- phase 3: training jobs ----
from contextlib import contextmanager
from studio.trainer import TrainError


class FakeTrainer:
    def __init__(self, mode="ok"): self.mode, self.calls = mode, []
    def train(self, sid, cancel, on_progress):
        self.calls.append(sid)
        on_progress(1, 2, 10.0); on_progress(2, 2, 0.0)
        if self.mode == "fail":
            raise TrainError("training failed (exit 1): CUDA out of memory")
        if self.mode == "cancel":
            cancel.set(); raise TrainError("training cancelled")
        return "lora"


def _train_queue(tmp_path, mode):
    eng = FakeEngine()
    eng.excl = []
    @contextmanager
    def exclusive(cancel):
        eng.excl.append("in"); yield True; eng.excl.append("out")
    eng.exclusive = exclusive
    reg = Registry(CATALOG, tmp_path / "u.json", [tmp_path / "app"], tmp_path / "hf")
    tr = FakeTrainer(mode)
    q = JobQueue(eng, reg, Settings(output_root=str(tmp_path / "out")), clock=lambda: T, trainer=tr)
    q.start()
    return q, eng, tr


def test_train_job_success(tmp_path):
    q, eng, tr = _train_queue(tmp_path, "ok")
    q.submit_train("alex")
    ev = wait(q)
    kinds = [e["kind"] for e in ev]
    assert tr.calls == ["alex"] and eng.excl == ["in", "out"]
    assert "train_step" in kinds and "trained" in kinds and kinds[-1] == "done"


def test_train_job_failure(tmp_path):
    q, eng, tr = _train_queue(tmp_path, "fail")
    q.submit_train("alex")
    ev = wait(q)
    assert any(e["kind"] == "train_failed" and "out of memory" in e["error"] for e in ev) and eng.excl == ["in", "out"]


def test_train_job_cancel(tmp_path):
    q, eng, tr = _train_queue(tmp_path, "cancel")
    q.submit_train("alex")
    ev = wait(q)
    assert ev[-1]["kind"] == "cancelled" and not any(e["kind"] == "train_failed" for e in ev)


# ---- phase 3: subjects in generation ----
from studio.subjects import SubjectStore


class RecEngine(FakeEngine):
    def __init__(self): super().__init__(); self.reqs = []
    def generate(self, req, on_step, cancel, on_image=None):
        self.reqs.append(req); return super().generate(req, on_step, cancel, on_image)


def _subj_queue(tmp_path):
    st = SubjectStore(tmp_path / "subjects")
    alex = st.create("Alex", "person", "man"); st.lora_path(alex.id).write_bytes(b"x")
    rex = st.create("Rex", "pet", "dog"); st.lora_path(rex.id).write_bytes(b"x")
    raw = st.create("Raw", "thing", "object")      # not trained
    eng = RecEngine()
    reg = Registry(CATALOG, tmp_path / "u.json", [tmp_path / "app"], tmp_path / "hf")
    q = JobQueue(eng, reg, Settings(output_root=str(tmp_path / "out")), clock=lambda: T, subjects=st)
    q.start()
    return q, eng, st, alex, rex, raw


def test_subjects_rewrite_prompt_and_reach_engine(tmp_path):
    q, eng, st, alex, rex, _ = _subj_queue(tmp_path)
    it = item(1, "Alex walking Rex on the beach", subject="beach")
    it.subjects = [(alex.id, 0.9), (rex.id, 0.8)]; it.reference = str(tmp_path / "r.png"); it.reference_strength = 0.4
    q.submit_batch([it], None)
    wait(q)
    req = eng.reqs[0]
    assert req.prompt == f"{alex.token} man walking {rex.token} dog on the beach"
    assert req.subjects == [(alex.id, st.lora_path(alex.id), 0.9), (rex.id, st.lora_path(rex.id), 0.8)]
    assert (req.reference, req.reference_strength) == (str(tmp_path / "r.png"), 0.4)
    r = json.loads(next(day(tmp_path).glob("*.json")).read_text())
    assert r["prompt"] == "Alex walking Rex on the beach" and r["prompt_sent"] == req.prompt
    assert r["subjects"] == [{"id": alex.id, "name": "Alex", "weight": 0.9}, {"id": rex.id, "name": "Rex", "weight": 0.8}]
    assert r["reference"] == "r.png"


def test_untrained_or_deleted_subject_fails_item(tmp_path):
    q, eng, st, alex, rex, raw = _subj_queue(tmp_path)
    a = item(1, "x one", subject="a"); a.subjects = [(raw.id, 0.9)]
    b = item(2, "y two", subject="b"); b.subjects = [("ghost", 0.9)]
    c = item(3, "z three", subject="c")
    q.submit_batch([a, b, c], None)
    ev = wait(q)
    fails = {e["index"]: e["error"] for e in ev if e["kind"] == "item_failed"}
    assert "not trained" in fails[1] and "ghost" in fails[2] and len(eng.reqs) == 1


def test_face_fix_flag_reaches_engine(tmp_path):
    q, eng, st, alex, rex, _ = _subj_queue(tmp_path)
    a = item(1, "Alex at a wedding", subject="w"); a.subjects = [(alex.id, 1.0)]
    b = item(2, "Alex at a wedding", subject="w2"); b.subjects = [(alex.id, 1.0)]; b.face_fix = False
    q.submit_batch([a, b], None)
    wait(q)
    assert [r.face_fix for r in eng.reqs] == [True, False]


def test_face_prompt_set_for_one_person_subject(tmp_path):
    q, eng, st, alex, rex, _ = _subj_queue(tmp_path)
    a = item(1, "Alex walking Rex", subject="w"); a.subjects = [(alex.id, 1.0), (rex.id, 1.0)]
    b = item(2, "Rex in a field", subject="f"); b.subjects = [(rex.id, 1.0)]
    q.submit_batch([a, b], None)
    wait(q)
    assert eng.reqs[0].face_prompt.startswith(f"photo of {alex.token} man") and eng.reqs[1].face_prompt == ""


class FixEngine(RecEngine):
    def __init__(self): super().__init__(); self.fixes = []
    def fix_region(self, img, model_id, region, subject, prompt, negative, seed, steps, guidance, cancel):
        self.fixes.append((model_id, region, subject, prompt))
        return img.convert("RGB")


def test_fix_job_writes_new_file_with_one_subject(tmp_path):
    st = SubjectStore(tmp_path / "subjects")
    alex = st.create("Alex", "person", "man"); st.lora_path(alex.id).write_bytes(b"x")
    rex = st.create("Rex", "pet", "dog"); st.lora_path(rex.id).write_bytes(b"x")
    eng = FixEngine()
    reg = Registry(CATALOG, tmp_path / "u.json", [tmp_path / "app"], tmp_path / "hf")
    q = JobQueue(eng, reg, Settings(output_root=str(tmp_path / "out")), clock=lambda: T, subjects=st)
    q.start()
    src = _src(day(tmp_path), "scene.png")
    q.submit_fix(str(src), 8, 4, rex.id)
    ev = wait(q)
    model, region, subject, prompt = eng.fixes[0]
    assert model == "realvis-v5" and subject == (rex.id, st.lora_path(rex.id), 1.0)
    assert prompt.startswith(f"photo of {rex.token} dog")
    out = day(tmp_path) / "scene_fix.png"
    r = json.loads(out.with_suffix(".json").read_text())
    assert r["fixed_from"] == "scene.png" and r["fixed_subject"] == "Rex" and src.exists()
    assert any(e["kind"] == "fixed" and e["path"] == str(out) for e in ev)


def test_fix_job_on_non_sdxl_recipe_uses_realvis(tmp_path):
    st = SubjectStore(tmp_path / "subjects")
    alex = st.create("Alex", "person", "man"); st.lora_path(alex.id).write_bytes(b"x")
    eng = FixEngine()
    reg = Registry(CATALOG, tmp_path / "u.json", [tmp_path / "app"], tmp_path / "hf")
    q = JobQueue(eng, reg, Settings(output_root=str(tmp_path / "out")), clock=lambda: T, subjects=st)
    q.start()
    from studio.recipe import read_recipe, write_recipe
    src = _src(day(tmp_path), "z.png"); r = read_recipe(src); r.model_id = "z-image-turbo"; write_recipe(src, r)
    q.submit_fix(str(src), 8, 4, alex.id)
    wait(q)
    assert eng.fixes[0][0] == "realvis-v5"


def test_batch_outputs_go_in_their_own_folder(make, tmp_path):
    q = make(batch_subfolder=True)
    q.submit_batch([item(1, "a one"), item(2, "b two")], "Space Set.json")
    q.submit_single(GenRequest("realvis-v5", "a cat", "", 64, 64, 2, 6.0, 1, 1), "cat")
    wait(q)
    assert sorted(p.name for p in (day(tmp_path) / "space-set").glob("*.png")) == \
        ["generated_a_2026-10-18_1652.png", "generated_b_2026-10-18_1652.png"]
    assert (day(tmp_path) / "generated_cat_2026-10-18_1652.png").exists()


def test_batch_folder_setting_off(tmp_path):
    reg = Registry(CATALOG, tmp_path / "u.json", [tmp_path / "app"], tmp_path / "hf")
    s = Settings(output_root=str(tmp_path / "out")); s.last["batch_subfolder"] = False
    q = JobQueue(FakeEngine(), reg, s, clock=lambda: T); q.start()
    q.submit_batch([item(1, "a one")], "space.json")
    wait(q)
    assert (day(tmp_path) / "generated_a_2026-10-18_1652.png").exists()


def test_state_reports_kind_label_and_start(make):
    q = make()
    seen = []
    class Slow(FakeEngine):
        def generate(self, req, on_step, cancel, on_image=None):
            seen.append(q.state()); return super().generate(req, on_step, cancel, on_image)
    q.engine = Slow()
    q.submit_batch([item(1, "a one")], "space.json")
    wait(q)
    st = seen[0]
    assert st["running"] and st["kind"] == "gen" and st["label"] == "Batch: space.json" and st["started"] > 0
    assert q.state()["kind"] is None


@pytest.mark.parametrize("fmt,ext,pil", [("jpg", ".jpg", "JPEG"), ("webp", ".webp", "WEBP"), ("png", ".png", "PNG")])
def test_pictures_saved_in_the_chosen_format(make, tmp_path, fmt, ext, pil):
    q = make()
    q.settings.last.update(output_format=fmt, output_quality=80)
    q.submit_single(GenRequest("realvis-v5", "a cat", "", 64, 64, 2, 6.0, 1, 1), "cat")
    wait(q)
    files = {p.name for p in day(tmp_path).iterdir()}
    assert files == {f"generated_cat_2026-10-18_1652{ext}", "generated_cat_2026-10-18_1652.json"}
    with Image.open(day(tmp_path) / f"generated_cat_2026-10-18_1652{ext}") as im:
        assert im.format == pil


def test_assistant_meta_goes_into_the_recipe(make, tmp_path):
    from studio.recipe import read_recipe
    q = make()
    it = item(1, "a fox")
    it.meta = {"assistant": {"provider": "gemini", "model": "m"}}
    q.submit_batch([it], "assistant-fox.json")
    wait(q)
    (png,) = [p for p in day(tmp_path).rglob("*.png")]
    assert read_recipe(png).extra["assistant"] == {"provider": "gemini", "model": "m"}
