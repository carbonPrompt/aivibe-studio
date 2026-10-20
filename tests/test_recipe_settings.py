import json
from studio.recipe import Recipe, write_recipe, read_recipe
from studio.settings import Settings, load_settings, save_settings


def _r(**kw):
    base = dict(prompt="a dinosaur", negative="blurry", model_id="realvis-v5", model_name="RealVisXL V5.0",
                license="commercial", seed=42, steps=30, guidance=6.0, width=1024, height=1024,
                subject="dinosaur", app_version="0.1.0", created="2026-10-18T16:52:00")
    base.update(kw)
    return Recipe(**base)


def test_recipe_round_trip(tmp_path):
    png = tmp_path / "generated_dinosaur_2026-10-18_1652.png"
    p = write_recipe(png, _r())
    assert p == png.with_suffix(".json")
    assert read_recipe(png) == _r()
    assert read_recipe(p) == _r()


def test_recipe_unknown_keys_go_to_extra(tmp_path):
    p = tmp_path / "x.json"
    d = json.loads(json.dumps(_r().__dict__)); d["upscale"] = "4x"
    p.write_text(json.dumps(d))
    assert read_recipe(p).extra == {"upscale": "4x"}


def test_settings_defaults_when_missing(tmp_path):
    s = load_settings(tmp_path / "settings.json")
    assert s.idle_minutes == 5 and s.theme == "dark" and s.output_root.endswith("aiVibe Studio")


def test_settings_corrupt_file_renamed(tmp_path):
    p = tmp_path / "settings.json"; p.write_text("{not json")
    s = load_settings(p)
    assert s.idle_minutes == 5 and (tmp_path / "settings.json.bad").exists()


def test_settings_round_trip(tmp_path):
    p = tmp_path / "settings.json"
    save_settings(p, Settings(output_root="X", idle_minutes=9, theme="light", last={"model": "sd15"}))
    assert load_settings(p) == Settings(output_root="X", idle_minutes=9, theme="light", last={"model": "sd15"})


def test_settings_old_file_gets_new_defaults(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"output_root": "D:\\pics", "idle_minutes": 7, "theme": "light", "last": {}}))
    s = load_settings(p)
    assert s.output_root == "D:\\pics" and s.idle_minutes == 7
    assert s.update_check is True and s.extra_model_folders == []
    assert s.gpu_marker_file == "" and s.gpu_jobs_folder == "" and s.hf_cache_dir == ""
