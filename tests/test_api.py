import re
import os
import base64
import json
from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image

from studio.api import Api
from studio.recipe import Recipe, write_recipe
from studio.registry import Registry
from studio.settings import Settings, load_settings

CATALOG = Path(__file__).parent.parent / "studio" / "catalog.json"


class FakeQueue:
    def __init__(self): self.singles, self.batches, self.cancelled = [], [], 0
    def submit_single(self, req, subject): self.singles.append((req, subject)); return "job1"
    def submit_batch(self, items, batch_file): self.batches.append((items, batch_file)); return "job2"
    def cancel(self): self.cancelled += 1
    def events(self, since): return []
    def state(self): return {"running": False, "current": None, "done": 0, "total": 0, "pending": 0}
    def note(self, kind, **kw): pass


class FakeEngine:
    device = "cuda"
    def set_device(self, d): self.device = d
    def loaded_model(self): return None


@pytest.fixture
def api(tmp_path):
    app = tmp_path / "app"; d = app / "realvisxl_v5.0-1"; d.mkdir(parents=True); (d / "model_index.json").write_text("{}")
    reg = Registry(CATALOG, tmp_path / "data" / "u.json", [app], tmp_path / "hf")
    s = Settings(output_root=str(tmp_path / "out"))
    a = Api(FakeQueue(), reg, s, tmp_path / "data" / "settings.json", FakeEngine())
    return a


def params(**kw):
    p = dict(model="realvis-v5", prompt="a red fox in the snow", negative="blurry", name="", width=1024, height=1024,
             steps=30, guidance=6.0, count=1, seed=-1)
    p.update(kw)
    return p


def test_generate_submits(api):
    r = api.generate(params(name="My Fox", seed=7))
    req, subject = api._queue.singles[0]
    assert r == {"job": "job1"} and subject == "my-fox" and req.seed == 7 and req.model_id == "realvis-v5"


def test_generate_random_seed_and_auto_subject(api):
    api.generate(params())
    req, subject = api._queue.singles[0]
    assert req.seed is None and subject == "red-fox-snow"


@pytest.mark.parametrize("bad", [dict(width=1000 + 4), dict(prompt="  "), dict(count=0), dict(model="nope"),
                                 dict(steps=0), dict(height="x")])
def test_generate_rejects(api, bad):
    r = api.generate(params(**bad))
    assert "error" in r and not api._queue.singles


def test_generate_needs_download(api):
    r = api.generate(params(model="flux-schnell"))
    assert r["needs_download"]["id"] == "flux-schnell" and not api._queue.singles


def test_load_bad_batch_submits_nothing(api, tmp_path):
    f = tmp_path / "b.json"; f.write_text('[{"prompt": "a", "model": "fluxx"}]')
    r = api.load_batch(str(f))
    assert r["problems"] and not api._queue.batches


def test_load_batch_ok(api, tmp_path):
    f = tmp_path / "b.txt"; f.write_text("a fox\na cat\n")
    r = api.load_batch(str(f))
    assert r == {"job": "job2", "count": 2} and api._queue.batches[0][1] == "b.txt"


def test_load_batch_with_undownloaded_model_reports_it(api, tmp_path):
    f = tmp_path / "b.json"; f.write_text('[{"prompt": "a", "model": "flux-schnell"}]')
    r = api.load_batch(str(f))
    assert r["needs_download"][0]["id"] == "flux-schnell" and not api._queue.batches


def _img(folder, stem, seed=5, when="2026-10-18T16:52:00"):
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"{stem}.png"; Image.new("RGB", (64, 64), "red").save(p)
    write_recipe(p, Recipe("a fox", "", "realvis-v5", "RealVisXL V5.0", "commercial", seed, 30, 6.0, 64, 64, "fox",
                           "0.1.0", when))
    return p


def test_gallery_newest_first_and_only_under_root(api, tmp_path):
    day = tmp_path / "out" / "2026-10-18"
    a = _img(day, "generated_fox_2026-10-18_1652")
    b = _img(day, "generated_fox_2026-10-18_1653")
    import os; os.utime(a, (1, 1))
    g = api.gallery(None)
    assert [x["path"] for x in g] == [str(b), str(a)] and g[0]["seed"] == 5
    assert g[0]["thumb"].startswith("data:image/jpeg;base64,")


def test_image_outside_root_refused(api, tmp_path):
    outside = _img(tmp_path / "elsewhere", "x")
    assert "error" in api.image(str(outside))
    assert "error" in api.remake(str(outside), False)


def test_image_data_url(api, tmp_path):
    p = _img(tmp_path / "out" / "2026-10-18", "generated_fox_2026-10-18_1652")
    d = api.image(str(p))["data"]
    assert base64.b64decode(d.split(",", 1)[1])[:4] == b"\x89PNG"


def test_remake_keeps_or_changes_seed(api, tmp_path):
    p = _img(tmp_path / "out" / "2026-10-18", "generated_fox_2026-10-18_1652", seed=99)
    api.remake(str(p), False)
    api.remake(str(p), True)
    (r1, s1), (r2, _) = api._queue.singles
    assert r1.seed == 99 and r2.seed is None and s1 == "fox"


def test_delete_model_refuses_app_folder(api):
    assert "error" in api.delete_model("realvis-v5")


def test_add_model_errors_are_returned(api):
    assert "error" in api.add_model("hf", "bad id", "X", "sdxl", "nc")
    assert api.add_model("hf", "a/b", "B", "sdxl", "nc")["id"] == "b"


def test_save_settings_persists_and_flags_unbacked_folder(api, tmp_path):
    r = api.save_settings({"output_root": str(tmp_path / "somewhere"), "theme": "light"})
    assert r["backed_up"] is False
    assert load_settings(tmp_path / "data" / "settings.json").theme == "light"
    home_pics = str(Path.home() / "Pictures" / "aiVibe Studio")
    assert api.save_settings({"output_root": home_pics})["backed_up"] is True


def test_device_choice_reaches_engine(api):
    api.save_settings({"device": "CPU"})
    assert api._engine.device == "cpu"


def test_delete_refuses_user_file_inside_hf_home(api, tmp_path):
    f = tmp_path / "hf" / "mine.safetensors"; f.parent.mkdir(parents=True, exist_ok=True); f.write_bytes(b"0")
    m = api._registry.add_file(f, "Mine", "sdxl", "nc")
    assert "error" in api.delete_model(m.id) and f.exists() and (tmp_path / "hf").exists()


def test_delete_hf_model_removes_only_its_repo_folder(api, tmp_path):
    repo = tmp_path / "hf" / "hub" / "models--Lykon--dreamshaper-8"
    snap = repo / "snapshots" / "abc"; snap.mkdir(parents=True); (snap / "model_index.json").write_text('{}')
    other = tmp_path / "hf" / "hub" / "models--x--y"; other.mkdir(parents=True)
    assert api.delete_model("dreamshaper-8") == {"ok": True}
    assert not repo.exists() and other.exists()


def test_download_patterns_pick_one_variant_and_skip_root_checkpoints():
    from studio.api import download_patterns
    files = ["model_index.json", "flux1-schnell.safetensors", "transformer/config.json",
             "transformer/diffusion_pytorch_model.safetensors", "transformer/diffusion_pytorch_model.fp16.safetensors",
             "tokenizer/vocab.json", "tokenizer/spiece.model", "scheduler/scheduler_config.json"]
    pats = download_patterns(files)
    import fnmatch
    picked = {f for f in files if any(fnmatch.fnmatch(f, p) for p in pats)}
    assert "flux1-schnell.safetensors" not in picked
    assert "transformer/diffusion_pytorch_model.fp16.safetensors" in picked
    assert "transformer/diffusion_pytorch_model.safetensors" not in picked
    assert {"model_index.json", "tokenizer/spiece.model", "scheduler/scheduler_config.json"} <= picked
    plain = download_patterns(["model_index.json", "unet/diffusion_pytorch_model.safetensors"])
    assert any(fnmatch.fnmatch("unet/diffusion_pytorch_model.safetensors", p) for p in plain)


# ---- phase 2 ----
class UpQueue(FakeQueue):
    def __init__(self):
        super().__init__(); self.ups = []
    def submit_upscale(self, paths, choice): self.ups.append((paths, choice)); return "job3"


def test_upscale_refuses_paths_outside_root(api, tmp_path):
    api._queue = UpQueue()
    inside = _img(tmp_path / "out" / "2026-10-18", "generated_fox_2026-10-18_1652")
    outside = _img(tmp_path / "elsewhere", "x")
    assert "error" in api.upscale([str(inside), str(outside)], "4x") and not api._queue.ups


def test_upscale_bad_choice(api, tmp_path):
    api._queue = UpQueue()
    p = _img(tmp_path / "out" / "2026-10-18", "generated_fox_2026-10-18_1652")
    assert "error" in api.upscale([str(p)], "3x") and not api._queue.ups
    assert "error" in api.upscale([], "4x")


def test_upscale_submits_one_job(api, tmp_path):
    api._queue = UpQueue()
    a = _img(tmp_path / "out" / "2026-10-18", "generated_fox_2026-10-18_1652")
    b = _img(tmp_path / "out" / "2026-10-18", "generated_fox_2026-10-18_1653")
    assert api.upscale([str(a), str(b)], "4K") == {"job": "job3", "count": 2}
    assert api._queue.ups == [([str(a), str(b)], "4k")]


def test_gallery_marks_upscales(api, tmp_path):
    from studio.recipe import read_recipe, write_recipe
    day = tmp_path / "out" / "2026-10-18"
    a = _img(day, "generated_fox_2026-10-18_1652")
    up = _img(day, "generated_fox_2026-10-18_1652_4x")
    r = read_recipe(up); r.extra.update(upscaled_from=a.name, upscale="4x"); write_recipe(up, r)
    g = {x["name"]: x for x in api.gallery(None)}
    assert g[up.name]["upscaled"] == "4x" and g[up.name]["source"] == a.name
    assert g[a.name]["upscaled"] is None


def test_gallery_survives_a_bad_image(api, tmp_path):
    day = tmp_path / "out" / "2026-10-18"
    good = _img(day, "generated_fox_2026-10-18_1652")
    (day / "broken.png").write_bytes(b"not a png")
    names = [x["name"] for x in api.gallery(None)]
    assert good.name in names


def test_remake_of_upscale_uses_generation_size_and_refuses_no_model(api, tmp_path):
    from studio.recipe import read_recipe, write_recipe
    day = tmp_path / "out" / "2026-10-18"
    up = _img(day, "generated_fox_2026-10-18_1652_4k")
    r = read_recipe(up); r.extra.update(out_width=3840, out_height=3840, upscale="4k"); write_recipe(up, r)
    api.remake(str(up), False)
    req, _ = api._queue.singles[-1]
    assert (req.width, req.height) == (64, 64)
    r.model_id = ""; write_recipe(up, r)
    assert "error" in api.remake(str(up), False)


# ---- phase 3 ----
from studio.subjects import SubjectStore


@pytest.fixture
def sapi(api, tmp_path):
    st = SubjectStore(tmp_path / "subjects")
    api._subjects = st
    alex = st.create("Alex", "person", "man"); st.lora_path(alex.id).write_bytes(b"x")
    st.create("Casey", "person", "woman")
    return api, st


def test_subjects_list(sapi):
    api, st = sapi
    s = {x["name"]: x for x in api.subjects()}
    assert s["Alex"]["trained"] and not s["Casey"]["trained"] and s["Alex"]["class_word"] == "man"
    assert "token" not in s["Alex"]          # the made-up trigger stays internal


def test_generate_with_subjects(sapi):
    api, st = sapi
    r = api.generate(params(prompt="Alex on a boat", subjects=[{"id": "alex", "weight": 0.8}]))
    req, _ = api._queue.singles[-1]
    assert r == {"job": "job1"} and req.subjects == [("alex", 0.8)]


def test_generate_untrained_subject_refused(sapi):
    api, st = sapi
    assert "not trained" in api.generate(params(subjects=[{"id": "casey", "weight": 0.9}]))["error"]


def test_generate_subjects_need_sdxl(sapi):
    api, st = sapi
    assert "SDXL" in api.generate(params(model="z-image-turbo", subjects=[{"id": "alex", "weight": 0.9}]))["error"]


def test_new_subject_import_and_train(sapi, tmp_path, monkeypatch):
    api, st = sapi
    r = api.new_subject("Rex", "pet", "dog")
    inbox = tmp_path / "inbox"; inbox.mkdir()
    for i in range(6):
        Image.new("RGB", (800, 800)).save(inbox / f"{i}.jpg")
    imp = api.import_photos(r["id"], str(inbox))
    assert imp["added"] == 6 and imp["total"] == 6 and imp["warning"]
    trains = []
    api._queue.submit_train = lambda sid: trains.append(sid) or "job9"
    assert api.train(r["id"]) == {"job": "job9", "steps": 800, "minutes": 37}
    assert trains == ["rex"]


def test_train_refuses_too_few_photos(sapi):
    api, st = sapi
    assert "photos" in api.train("casey")["error"]


def test_remake_with_deleted_subject_names_it(sapi, tmp_path):
    from studio.recipe import read_recipe, write_recipe
    api, st = sapi
    p = _img(tmp_path / "out" / "2026-10-18", "generated_x_2026-10-18_1652")
    r = read_recipe(p); r.extra["subjects"] = [{"id": "gone", "name": "Old Dog", "weight": 0.9}]; write_recipe(p, r)
    assert "Old Dog" in api.remake(str(p), False)["error"]


def test_generate_passes_face_fix(sapi):
    api, st = sapi
    api.generate(params(prompt="Alex", subjects=[{"id": "alex", "weight": 1.0}], face_fix=False))
    req, _ = api._queue.singles[-1]
    assert req.face_fix is False


def test_fix_face_checks_path_and_subject(sapi, tmp_path):
    api, st = sapi
    fixes = []
    api._queue.submit_fix = lambda p, x, y, sid: fixes.append((p, x, y, sid)) or "job7"
    inside = _img(tmp_path / "out" / "2026-10-18", "generated_x_2026-10-18_1652")
    outside = _img(tmp_path / "elsewhere", "x")
    assert "error" in api.fix_face(str(outside), 10, 10, "alex")
    assert "not trained" in api.fix_face(str(inside), 10, 10, "casey")["error"]
    assert api.fix_face(str(inside), 10, 20, "alex") == {"job": "job7"} and fixes == [(str(inside), 10, 20, "alex")]


def test_subjects_report_and_set_default_weight(sapi):
    api, st = sapi
    assert {x["name"]: x["default_weight"] for x in api.subjects()}["Alex"] == 1.0
    assert api.set_default_weight("alex", 1.2) == {"ok": True}
    assert {x["name"]: x["default_weight"] for x in api.subjects()}["Alex"] == 1.2
    assert "error" in api.set_default_weight("alex", 3)


def test_subject_photos_list_and_remove(sapi, tmp_path):
    api, st = sapi
    inbox = tmp_path / "in2"; inbox.mkdir()
    for i in range(3):
        Image.new("RGB", (800, 800)).save(inbox / f"{i}.jpg")
    st.import_photos("casey", inbox, face_finder=lambda im: None)
    photos = api.subject_photos("casey")
    assert [p["name"] for p in photos] == ["0.png", "1.png", "2.png"] and photos[0]["thumb"].startswith("data:image/jpeg")
    assert api.remove_subject_photo("casey", "1.png") == {"ok": True, "total": 2}
    assert "error" in api.remove_subject_photo("casey", "..\\subject.json")


def test_gpu_reports_temperature_and_load(api, monkeypatch):
    import studio.api as a
    class R: stdout = "5120, 12288, 40, 61\n"
    monkeypatch.setattr(a.subprocess, "run", lambda *x, **k: R())
    g = api.gpu()
    assert (g["used_mb"], g["total_mb"], g["util"], g["temp_c"]) == (5120.0, 12288.0, 40.0, 61.0)


def test_gallery_includes_batch_subfolders(api, tmp_path):
    day = tmp_path / "out" / "2026-10-18"
    a = _img(day / "space-set", "generated_a_2026-10-18_1652")
    assert [x["name"] for x in api.gallery(None)] == [a.name]


def test_import_and_upscale_copies_into_output_then_queues(api, tmp_path):
    api._queue = UpQueue()
    src = tmp_path / "phone" / "IMG 0001.JPG"; src.parent.mkdir()
    Image.new("RGB", (40, 30), "green").save(src, format="JPEG")
    r = api.import_and_upscale(str(src), "4x")
    assert r["count"] == 1
    (paths, choice), = api._queue.ups
    p = Path(paths[0])
    assert choice == "4x" and p.parent.name == "imported" and p.suffix == ".png" and p.exists()
    assert str(p).startswith(str(tmp_path / "out")) and src.exists()


def test_import_refuses_network_and_non_images(api, tmp_path):
    api._queue = UpQueue()
    assert "error" in api.import_and_upscale(r"\\host\share\a.png", "4x")
    bad = tmp_path / "a.txt"; bad.write_text("x")
    assert "error" in api.import_and_upscale(str(bad), "4x")
    assert not api._queue.ups


def test_gallery_follows_the_recent_settings(api, tmp_path):
    """Settings: how many pictures, and one chosen folder shown without its subfolders."""
    import os
    day = tmp_path / "out" / "2026-10-18"
    for i in range(5):
        os.utime(_img(day, f"generated_fox_2026-10-18_165{i}"), (i + 1, i + 1))
    api.save_settings({"recent_limit": 2})
    assert [x["name"] for x in api.gallery(None, None)] == ["generated_fox_2026-10-18_1654.png", "generated_fox_2026-10-18_1653.png"]
    mine = tmp_path / "keepers"
    keep = _img(mine, "generated_best_2026-10-18_1700")
    _img(mine / "older", "generated_old_2026-10-18_1701")           # a subfolder: not shown
    api.save_settings({"recent_folder": str(mine), "recent_limit": 60})
    g = api.gallery(None, None)
    assert [x["path"] for x in g] == [str(keep)]
    assert "data" in api.image(str(keep))                              # outside the output folder, but chosen
    assert "error" in api.image(str(mine / "older" / "generated_old_2026-10-18_1701.png"))
    api.save_settings({"recent_folder": ""})
    assert len(api.gallery(None, None)) == 5


def test_gallery_shows_and_opens_jpg_and_webp(api, tmp_path):
    day = tmp_path / "out" / "2026-10-18"
    p = _img(day, "generated_fox_2026-10-18_1652")
    j = p.with_suffix(".jpg"); Image.open(p).convert("RGB").save(j, "JPEG"); p.unlink()
    w = day / "other.webp"; Image.new("RGB", (8, 8)).save(w, "WEBP")
    assert sorted(x["name"] for x in api.gallery(None)) == ["generated_fox_2026-10-18_1652.jpg", "other.webp"]
    assert api.image(str(j))["data"].startswith("data:image/jpeg;base64,")
    assert api.image(str(w))["data"].startswith("data:image/webp;base64,")


# ---- phase 4: assistant ----
@pytest.fixture
def aapi(api, tmp_path):
    from studio.keys import KeyStore
    from studio.subjects import SubjectStore
    api._keys = KeyStore(tmp_path / "data" / "assistant_keys.json")
    api._subjects = SubjectStore(tmp_path / "data" / "subjects")
    return api


def _draft(items, problems=()):
    from studio.assistant import Draft
    return Draft({"defaults": {}, "items": items}, list(problems), "gemini", "gemini-x", 123, 4.2)


def test_assistant_no_key_and_empty_request_messages(aapi):
    assert aapi.assistant_write("") == {"error": "Type what you want first."}
    assert aapi.assistant_write("a fox") == {"error": "Add your Gemini (Google) API key in Settings > Assistant first."}
    assert aapi.assistant_models("gemini") == {"error": "Add your Gemini (Google) API key in Settings > Assistant first."}


def test_assistant_key_is_masked_and_never_logged_or_reported(aapi, tmp_path, caplog):
    import logging
    with caplog.at_level(logging.DEBUG):
        r = aapi.set_assistant_key("gemini", "AIzaSySECRETSECRET9876")
        st = aapi.assistant_status()
        name = aapi.report_problem("test", None)["name"]
    assert r == {"ok": True, "key": "saved, ends in 9876"}
    assert "SECRET" not in caplog.text and "SECRET" not in json.dumps(st)
    folder = tmp_path / "data" / "reports" / name
    assert folder.is_dir() and not any("SECRET" in p.read_text(errors="ignore") for p in folder.rglob("*") if p.is_file())
    g = [p for p in st["providers"] if p["id"] == "gemini"][0]
    assert g["key"] == "saved, ends in 9876" and st["provider"] == "gemini" and g["label"] == "Gemini (Google)"
    assert aapi.set_assistant_key("bard", "x")["error"] and aapi.set_assistant_key("gemini", "  ")["error"]
    assert aapi.remove_assistant_key("gemini") == {"ok": True} and aapi.assistant_status()["providers"][1]["key"] is None


def test_assistant_provider_and_model_choice_saved(aapi):
    aapi.save_settings({"assistant_provider": "claude", "assistant_model_claude": "claude-opus-5-5"})
    st = aapi.assistant_status()
    assert st["provider"] == "claude" and st["model"] == "claude-opus-5-5"


def test_assistant_write_returns_draft_fields(aapi, monkeypatch):
    from studio import assistant
    seen = {}

    def fake(pid, key, model, request, registry, subjects, fallback):
        seen.update(pid=pid, key=key, model=model, request=request); return _draft([{"prompt": "a fox"}])
    monkeypatch.setattr(assistant, "write_batch", fake)
    aapi.set_assistant_key("gemini", "KEY1234")
    r = aapi.assistant_write("a fox please")
    assert r == {"batch": {"defaults": {}, "items": [{"prompt": "a fox"}]}, "problems": [], "provider": "gemini",
                 "label": "Gemini (Google)", "model": "gemini-x", "tokens": 123, "seconds": 4.2}
    assert seen == {"pid": "gemini", "key": "KEY1234", "model": "gemini-flash-latest", "request": "a fox please"}


def test_assistant_provider_error_is_a_message(aapi, monkeypatch):
    from studio import assistant
    from studio.providers import ProviderError

    def boom(*a):
        raise ProviderError("Gemini (Google) rejected the API key. Check it in Settings > Assistant.")
    monkeypatch.setattr(assistant, "write_batch", boom)
    aapi.set_assistant_key("gemini", "KEY1234")
    assert aapi.assistant_write("x") == {"error": "Gemini (Google) rejected the API key. Check it in Settings > Assistant."}


def test_assistant_run_saves_file_tags_items_and_submits(aapi, tmp_path):
    batch = {"defaults": {"upscale": "1080p"}, "items": [{"prompt": "a red fox in the snow", "width": 1344, "height": 768}]}
    r = aapi.assistant_run("Three space foxes, please!", batch, "gemini", "gemini-x")
    assert r["job"] == "job2" and r["count"] == 1
    items, batch_file = aapi._queue.batches[0]
    assert batch_file == "assistant-three-space-foxes-please.json"
    assert items[0].meta == {"assistant": {"provider": "gemini", "model": "gemini-x"}} and items[0].upscale == "1080p"
    saved = json.loads(Path(r["file"]).read_text(encoding="utf-8"))
    assert Path(r["file"]).parent == tmp_path / "data" / "assistant"
    assert saved["request"] == "Three space foxes, please!" and saved["provider"] == "gemini" and saved["items"] == batch["items"]


def test_assistant_run_refuses_an_edited_invalid_batch(aapi):
    r = aapi.assistant_run("x", {"defaults": {}, "items": [{"prompt": ""}]}, "gemini", "m")
    assert r["problems"] and not aapi._queue.batches
    r = aapi.assistant_run("x", {"defaults": {}, "items": []}, "gemini", "m")
    assert r["problems"] and not aapi._queue.batches


def test_assistant_run_refuses_reference_added_in_review(aapi, tmp_path):
    ref = tmp_path / "r.png"; Image.new("RGB", (8, 8)).save(ref)
    r = aapi.assistant_run("x", {"defaults": {}, "items": [{"prompt": "a", "reference": str(ref)}]}, "gemini", "m")
    assert "item 1: the assistant may not add reference images" in r["problems"] and not aapi._queue.batches


def test_assistant_revise_passes_batch_and_instruction(aapi, monkeypatch):
    from studio import assistant
    seen = {}

    def fake(pid, key, model, request, batch, instruction, registry, subjects, fallback):
        seen.update(request=request, batch=batch, instruction=instruction); return _draft([{"prompt": "b"}])
    monkeypatch.setattr(assistant, "revise_batch", fake)
    aapi.set_assistant_key("gemini", "KEY1234")
    b = {"defaults": {}, "items": [{"prompt": "a"}]}
    assert aapi.assistant_revise("req", b, "")["error"] == "Type how to change it first."
    assert aapi.assistant_revise("req", b, "night")["batch"]["items"] == [{"prompt": "b"}]
    assert seen == {"request": "req", "batch": b, "instruction": "night"}


def test_assistant_unexpected_errors_become_messages(aapi, monkeypatch):
    from studio import assistant

    def boom(*a):
        raise RuntimeError("weird")
    monkeypatch.setattr(assistant, "write_batch", boom)
    aapi.set_assistant_key("gemini", "KEY1234")
    r = aapi.assistant_write("x")
    assert r["error"].startswith("Something went wrong (RuntimeError: weird)")
    monkeypatch.setattr(assistant, "check_assistant_batch", boom)
    r = aapi.assistant_run("x", {"defaults": {}, "items": [{"prompt": "a"}]}, "gemini", "m")
    assert r["error"].startswith("Something went wrong (RuntimeError: weird)") and not aapi._queue.batches


def test_save_advanced_settings_top_level_and_restart_flag(api, tmp_path):
    r = api.save_settings({"extra_model_folders": ["D:\\models", "  ", "E:\\more"], "hf_cache_dir": "D:\\hf",
                           "gpu_marker_file": "C:\\x\\busy.json", "gpu_jobs_folder": "", "update_check": False})
    s = load_settings(tmp_path / "data" / "settings.json")
    assert s.extra_model_folders == ["D:\\models", "E:\\more"] and s.hf_cache_dir == "D:\\hf"
    assert s.gpu_marker_file == "C:\\x\\busy.json" and s.update_check is False
    assert "extra_model_folders" not in s.last and r["restart"] is True
    assert api.save_settings({"update_check": True})["restart"] is False


def _api_with_gpu(tmp_path, gpu):
    reg = Registry(CATALOG, tmp_path / "data" / "u.json", [], tmp_path / "hf")
    return Api(FakeQueue(), reg, Settings(output_root=str(tmp_path / "out")), tmp_path / "data" / "settings.json",
               FakeEngine(), gpu=gpu)


def test_settings_report_the_card(tmp_path):
    from studio.hardware import GpuInfo
    a = _api_with_gpu(tmp_path, GpuInfo("NVIDIA GeForce RTX 4070", 12.0, (8, 9), "566.36"))
    hw = a.get_settings()["hardware"]
    assert hw == {"support": "ok", "name": "NVIDIA GeForce RTX 4070", "vram_gb": 12.0, "driver": "566.36",
                  "train_ok": True}


def test_models_carry_fit_for_this_card(tmp_path):
    from studio.hardware import GpuInfo
    a = _api_with_gpu(tmp_path, GpuInfo("RTX 2070", 8.0, (7, 5), ""))
    fit = {m["id"]: (m["fit"], m["min_vram_gb"]) for m in a.models()}
    assert fit["sd15"] == ("ok", 4) and fit["flux2-klein-4b"] == ("tight", 10) and fit["flux-schnell"] == ("too_big", 12)
    assert a.get_settings()["hardware"]["train_ok"] is False


def test_no_card_reported(tmp_path):
    hw = _api_with_gpu(tmp_path, None).get_settings()["hardware"]
    assert hw["support"] == "no_gpu" and hw["train_ok"] is False


# ---- downloads: consent, space, token ----
class FakeFetch:
    """Stands in for the download child process: records the token, writes a partial file, then fails, waits
    for Cancel, or finishes."""
    def __init__(self, status=None, wait_cancel=False):
        self.status, self.wait_cancel, self.tokens, self.revisions = status, wait_cancel, [], []
    def __call__(self, repo, cache_dir, token, cancel, revision=None):
        from studio.hfget import FetchError
        self.tokens.append(token); self.revisions.append(revision)
        d = Path(cache_dir) / ("models--" + repo.replace("/", "--")) / "blobs"; d.mkdir(parents=True, exist_ok=True)
        (d / "x.incomplete").write_bytes(b"0" * 10)
        if self.wait_cancel:
            assert cancel.wait(5)
            raise FetchError("cancelled", None)
        if self.status is not None:
            raise FetchError(f"{self.status} Client Error" if self.status else "No space left on device", self.status or None)
        snap = d.parent / "snapshots" / "abc"; snap.mkdir(parents=True, exist_ok=True)


class Events(FakeQueue):
    def __init__(self): super().__init__(); self.notes = []
    def note(self, kind, **kw): self.notes.append((kind, kw))


def _dl_api(tmp_path, monkeypatch, free_gb=500.0, token=None, fetch=None):
    from studio.keys import KeyStore
    import studio.api as api_mod
    monkeypatch.setattr(api_mod.shutil, "disk_usage", lambda p: (0, 0, int(free_gb * 1e9)))
    keys = KeyStore(tmp_path / "keys.json")
    if token:
        keys.set("huggingface", token)
    reg = Registry(CATALOG, tmp_path / "data" / "u.json", [], tmp_path / "hf")
    a = Api(Events(), reg, Settings(output_root=str(tmp_path / "out")), tmp_path / "data" / "settings.json",
            FakeEngine(), keys=keys, gpu=None)
    a._fetch = fetch or FakeFetch()
    return a


def test_download_info_fields(tmp_path, monkeypatch):
    info = _dl_api(tmp_path, monkeypatch, free_gb=100).download_info("sd35-medium")
    assert info["license"] == "conditions" and info["gated"] is True and info["has_token"] is False
    assert info["license_text"] == "Free for personal use; commercial use has conditions. Read the model card."
    assert info["card_url"] == "https://huggingface.co/stabilityai/stable-diffusion-3.5-medium"
    assert info["size_gb"] == 16.0 and info["free_gb"] == 100.0 and info["enough_space"] is True and info["fit"] == "too_big"
    assert _dl_api(tmp_path, monkeypatch).download_info("juggernaut-x")["license_text"] == "Non-commercial use only."
    assert _dl_api(tmp_path, monkeypatch).download_info("realvis-v5")["license_text"] == "Free to use, including commercially."


def test_download_refuses_no_space(tmp_path, monkeypatch):
    a = _dl_api(tmp_path, monkeypatch, free_gb=5)
    r = a.download("realvis-v5")
    assert r["error"] == "no_space" and r["free_gb"] == 5.0 and r["size_gb"] == 6.5 and "realvis-v5" not in a._downloads


def test_gated_without_token_needs_token(tmp_path, monkeypatch):
    assert _dl_api(tmp_path, monkeypatch).download("sd35-large")["error"] == "needs_token"
    assert _dl_api(tmp_path, monkeypatch).download("flux-schnell")["error"] == "needs_token"


def test_train_without_base_needs_download(tmp_path, monkeypatch):
    from studio.subjects import SubjectStore
    a = _dl_api(tmp_path, monkeypatch)
    a._subjects = SubjectStore(tmp_path / "subjects")
    s = a._subjects.create("Alex", "person", "man")
    for i in range(12):
        (a._subjects.root / s.id / "photos" / f"p{i}.png").write_bytes(b"x")
    r = a.train(s.id)
    assert r["needs_download"] == {"id": "realvis-v5", "name": "RealVisXL V5.0", "size_gb": 6.5}


def test_hf_token_saved_masked_and_never_logged(tmp_path, monkeypatch, caplog):
    import logging
    a = _dl_api(tmp_path, monkeypatch)
    with caplog.at_level(logging.DEBUG):
        r = a.set_hf_token("  hf_SECRETSECRET4321 ")
        st = a.hf_token_status()
    assert r == {"ok": True, "key": "saved, ends in 4321"} and st == {"key": "saved, ends in 4321"}
    assert "SECRET" not in caplog.text and a.download_info("sd35-large")["has_token"] is True
    assert a.remove_hf_token() == {"ok": True} and a.hf_token_status() == {"key": None}


# ---- updates ----
def test_check_update_offline_silent_at_start_visible_when_manual(api, monkeypatch):
    import studio.update as up
    monkeypatch.setattr(up, "latest", lambda **kw: None)
    assert api.check_update(False) == {"update": False}
    assert api.check_update(True) == {"error": "offline", "message": "Couldn't reach GitHub to check for updates."}


def test_check_update_reports_newer(api, monkeypatch):
    import studio.update as up
    monkeypatch.setattr(up, "latest", lambda **kw: {"version": "9.0.0", "url": "https://x/s.exe", "notes": "n", "size": 3})
    assert api.check_update(False) == {"update": True, "version": "9.0.0", "notes": "n"}
    monkeypatch.setattr(up, "latest", lambda **kw: {"version": "0.0.1", "url": "https://x/s.exe", "notes": "", "size": 3})
    assert api.check_update(True) == {"update": False}


def test_get_update_size_mismatch_deletes_file(api, monkeypatch, tmp_path):
    import studio.update as up
    monkeypatch.setattr(up, "latest", lambda **kw: {"version": "9.0.0", "url": "https://x/s.exe", "notes": "", "size": 999, "sha256": "0" * 64})
    started = []
    api._downloads_dir = tmp_path / "dl"
    api._fetch_file = lambda url, dest, on_progress: dest.write_bytes(b"short")
    api._start_installer = lambda path: started.append(path)
    r = api.get_update()
    assert r["error"].startswith("The download was incomplete") and started == []
    assert not any((tmp_path / "dl").glob("*.exe"))


def test_get_update_runs_installer(api, monkeypatch, tmp_path):
    import studio.update as up
    import hashlib
    monkeypatch.setattr(up, "latest", lambda **kw: {"version": "9.0.0", "url": "https://x/s.exe", "notes": "", "size": 4,
                                                    "sha256": hashlib.sha256(b"1234").hexdigest()})
    started = []
    api._downloads_dir = tmp_path / "dl"
    api._fetch_file = lambda url, dest, on_progress: dest.write_bytes(b"1234")
    api._start_installer = lambda path: started.append(path)
    assert api.get_update() == {"ok": True}
    assert started == [tmp_path / "dl" / "aiVibeStudio-Setup-9.0.0.exe"]


def test_get_update_hash_mismatch_never_runs(api, monkeypatch, tmp_path):
    import studio.update as up
    monkeypatch.setattr(up, "latest", lambda **kw: {"version": "9.0.0", "url": "https://x/s.exe", "notes": "", "size": 4,
                                                    "sha256": "0" * 64})
    started = []
    api._downloads_dir = tmp_path / "dl"
    api._fetch_file = lambda url, dest, on_progress: dest.write_bytes(b"1234")
    api._start_installer = lambda path: started.append(path)
    assert api.get_update()["error"].startswith("The download did not match") and started == []
    assert not any((tmp_path / "dl").glob("*.exe"))


def test_token_only_sent_for_gated_models(tmp_path, monkeypatch):
    f = FakeFetch()
    a = _dl_api(tmp_path, monkeypatch, token="hf_SECRET", fetch=f)
    a._download(a._registry.get("sd35-medium")); a._download(a._registry.get("realvis-v5"))
    assert f.tokens == ["hf_SECRET", None] and [n[0] for n in a._queue.notes][-1] == "download_done"


def test_401_without_token_needs_token(tmp_path, monkeypatch):
    a = _dl_api(tmp_path, monkeypatch, fetch=FakeFetch(401))
    a._download(a._registry.get("realvis-v5"))
    kind, kw = a._queue.notes[-1]
    assert kind == "download_failed" and kw["code"] == "needs_token"


def test_401_with_saved_token_says_the_token_is_bad(tmp_path, monkeypatch):
    a = _dl_api(tmp_path, monkeypatch, token="hf_old", fetch=FakeFetch(401))
    a._download(a._registry.get("sd35-medium"))
    kind, kw = a._queue.notes[-1]
    assert kw["code"] == "bad_token" and "refused your token" in kw["error"]


def test_403_maps_to_needs_terms(tmp_path, monkeypatch):
    a = _dl_api(tmp_path, monkeypatch, token="hf_ok", fetch=FakeFetch(403))
    a._download(a._registry.get("sd35-medium"))
    assert a._queue.notes[-1][1]["code"] == "needs_terms"


def test_failed_new_download_removes_its_folder(tmp_path, monkeypatch):
    a = _dl_api(tmp_path, monkeypatch, fetch=FakeFetch(0))
    a._download(a._registry.get("realvis-v5"))
    assert a._queue.notes[-1][0] == "download_failed"
    assert not a._registry.hf_repo_dir("realvis-v5").exists() and not a._registry.is_downloaded("realvis-v5")


def test_failed_download_keeps_a_folder_that_was_already_there(tmp_path, monkeypatch):
    a = _dl_api(tmp_path, monkeypatch, fetch=FakeFetch(0))
    repo = a._registry.hf_repo_dir("z-image-turbo")
    keep = repo / "snapshots" / "old" / "model_index.json"; keep.parent.mkdir(parents=True); keep.write_text("{}")
    a._download(a._registry.get("z-image-turbo"))
    assert keep.exists() and not list(repo.rglob("*.incomplete")) and not (repo / ".studio-downloading").exists()


def test_cancel_download(tmp_path, monkeypatch):
    import threading, time
    a = _dl_api(tmp_path, monkeypatch, fetch=FakeFetch(wait_cancel=True))
    a._downloads.add("realvis-v5")
    t = threading.Thread(target=a._download, args=(a._registry.get("realvis-v5"),)); t.start()
    for _ in range(200):
        if "realvis-v5" in a._cancels: break
        time.sleep(0.01)
    assert a.cancel_download("realvis-v5") == {"ok": True}
    t.join(5)
    assert a._queue.notes[-1][0] == "download_cancelled" and not a._registry.hf_repo_dir("realvis-v5").exists()
    assert a.cancel_download("realvis-v5") == {"error": "not downloading"}


def _fake_python(tmp_path, body):
    f = tmp_path / "fakepy"; f.write_text("#!/bin/sh\n" + body + "\n"); f.chmod(0o755)
    return str(f)


@pytest.mark.skipif(os.name == "nt", reason="uses a shell script as the child")
def test_fetch_child_reports_status_from_json(api, tmp_path):
    import threading
    from studio.hfget import FetchError
    api._python = _fake_python(tmp_path, """echo '{"error": "401 Client Error", "status": 401}'; exit 1""")
    with pytest.raises(FetchError) as x:
        api._fetch_child("a/b", str(tmp_path), "hf_x", threading.Event())
    assert x.value.status == 401


@pytest.mark.skipif(os.name == "nt", reason="uses a shell script as the child")
def test_fetch_child_cancel_kills_it_fast(api, tmp_path):
    import threading, time
    from studio.hfget import FetchError
    api._python = _fake_python(tmp_path, "sleep 30")
    cancel = threading.Event(); threading.Timer(0.3, cancel.set).start()
    t0 = time.time()
    with pytest.raises(FetchError, match="cancelled"):
        api._fetch_child("a/b", str(tmp_path), None, cancel)
    assert time.time() - t0 < 5


def test_download_passes_the_pinned_revision(tmp_path, monkeypatch):
    f = FakeFetch()
    a = _dl_api(tmp_path, monkeypatch, fetch=f)
    a._download(a._registry.get("sd-compact")); a._download(a._registry.get("realvis-v5"))
    assert re.fullmatch(r"[0-9a-f]{40}", f.revisions[0] or "") and f.revisions[1] is None
