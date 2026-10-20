import re
import pytest
from pathlib import Path
from studio.registry import Registry, ModelEntry

CATALOG = Path(__file__).parent.parent / "studio" / "catalog.json"


@pytest.fixture
def reg(tmp_path):
    app = tmp_path / "app"; app.mkdir()
    hf = tmp_path / "hf"; (hf / "hub").mkdir(parents=True)
    return Registry(CATALOG, tmp_path / "data" / "user_models.json", [app], hf)


def test_catalog_has_day_one_models(reg):
    ids = {m.id for m in reg.all()}
    assert {"sd35-large", "sd35-large-turbo", "sd35-medium", "juggernaut-x", "realvis-v5", "dreamshaper-8",
            "playground-25", "sd15", "flux-schnell", "sd-compact", "z-image-turbo", "flux2-klein-4b"} <= ids
    assert reg.get("juggernaut-x").license == "nc"
    assert isinstance(reg.get("realvis-v5"), ModelEntry)


def test_get_unknown_raises(reg):
    with pytest.raises(KeyError):
        reg.get("nope")


def test_local_path_prefers_app_folder(reg, tmp_path):
    d = tmp_path / "app" / "realvisxl_v5.0-7061e3dd"; d.mkdir()
    (d / "model_index.json").write_text("{}")
    assert reg.local_path("realvis-v5") == d
    assert reg.is_downloaded("realvis-v5")


def test_local_path_finds_hf_snapshot(reg, tmp_path):
    snap = tmp_path / "hf" / "hub" / "models--Lykon--dreamshaper-8" / "snapshots" / "abc"; snap.mkdir(parents=True)
    (snap / "model_index.json").write_text("{}")
    assert reg.local_path("dreamshaper-8") == snap


def test_not_downloaded(reg):
    assert reg.local_path("flux-schnell") is None
    assert not reg.is_downloaded("flux-schnell")


def test_incomplete_hf_snapshot_is_not_downloaded(reg, tmp_path):
    (tmp_path / "hf" / "hub" / "models--black-forest-labs--FLUX.1-schnell" / "snapshots" / "abc").mkdir(parents=True)
    assert not reg.is_downloaded("flux-schnell")


def test_add_hf_validates_and_persists(reg, tmp_path):
    with pytest.raises(ValueError):
        reg.add_hf("bad id", "Bad", "sdxl", "nc")
    m = reg.add_hf("someone/cool-xl", "Cool XL", "sdxl", "nc")
    assert m.user and m.source == {"hf": "someone/cool-xl"} and m.pipeline == "StableDiffusionXLPipeline"
    again = Registry(CATALOG, tmp_path / "data" / "user_models.json", [tmp_path / "app"], tmp_path / "hf")
    assert again.get(m.id).name == "Cool XL"


def test_add_file(reg, tmp_path):
    bad = tmp_path / "x.ckpt"; bad.write_bytes(b"0")
    with pytest.raises(ValueError):
        reg.add_file(bad, "X", "sdxl", "nc")
    with pytest.raises(ValueError):
        reg.add_file(tmp_path / "missing.safetensors", "X", "sdxl", "nc")
    f = tmp_path / "mine.safetensors"; f.write_bytes(b"0")
    m = reg.add_file(f, "Mine", "sd15", "commercial")
    assert reg.local_path(m.id) == f and reg.get(m.id).defaults["width"] == 512


def test_add_hf_rejects_unknown_family(reg):
    with pytest.raises(ValueError):
        reg.add_hf("a/b", "B", "dalle", "nc")


def test_remove_user_only(reg):
    with pytest.raises(ValueError):
        reg.remove_user("realvis-v5")
    m = reg.add_hf("a/b", "B", "sdxl", "nc")
    reg.remove_user(m.id)
    with pytest.raises(KeyError):
        reg.get(m.id)


def test_catalog_entries_have_descriptions(reg):
    assert all(m.description for m in reg.all())


def _snap(tmp_path, repo, comps=("unet",), with_files=True):
    root = tmp_path / "hf" / "hub" / ("models--" + repo.replace("/", "--"))
    snap = root / "snapshots" / "abc"; snap.mkdir(parents=True)
    import json as _j
    (snap / "model_index.json").write_text(_j.dumps({"_class_name": "X", **{c: ["diffusers", "Y"] for c in comps},
                                                      "safety_checker": [None, None]}))
    for c in comps:
        (snap / c).mkdir()
        if with_files:
            (snap / c / "w.safetensors").write_bytes(b"0")
    (root / "blobs").mkdir()
    return root, snap


def test_hf_snapshot_missing_component_is_not_downloaded(reg, tmp_path):
    _snap(tmp_path, "Lykon/dreamshaper-8", comps=("unet", "vae"), with_files=False)
    assert not reg.is_downloaded("dreamshaper-8")


def test_hf_snapshot_with_incomplete_blob_is_not_downloaded(reg, tmp_path):
    root, _ = _snap(tmp_path, "Lykon/dreamshaper-8")
    (root / "blobs" / "abc.incomplete").write_bytes(b"0")
    assert not reg.is_downloaded("dreamshaper-8")


def test_hf_snapshot_marked_downloading_is_not_downloaded(reg, tmp_path):
    root, _ = _snap(tmp_path, "Lykon/dreamshaper-8")
    (root / ".studio-downloading").write_text("")
    assert not reg.is_downloaded("dreamshaper-8")


def test_complete_hf_snapshot_is_downloaded(reg, tmp_path):
    _snap(tmp_path, "Lykon/dreamshaper-8", comps=("unet", "vae"))
    assert reg.is_downloaded("dreamshaper-8")


def test_registry_extra_folders_in_order(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        d = root / "realvisxl_v5.0-x"; d.mkdir(parents=True)
        (d / "model_index.json").write_text("{}")
    r = Registry(CATALOG, tmp_path / "u.json", [tmp_path / "missing", a, b], tmp_path / "hf")
    assert r.local_path("realvis-v5") == a / "realvisxl_v5.0-x"
    assert r.extra_model_dirs == [tmp_path / "missing", a, b]


def test_registry_no_extra_folders(tmp_path):
    r = Registry(CATALOG, tmp_path / "u.json", [], tmp_path / "hf")
    assert r.local_path("realvis-v5") is None


def test_component_with_config_but_no_weights_is_not_downloaded(reg, tmp_path):
    root, snap = _snap(tmp_path, "segmind/small-sd", comps=("unet", "tokenizer"), with_files=False)
    (snap / "unet" / "config.json").write_text("{}")
    (snap / "tokenizer" / "vocab.json").write_text("{}")
    assert not reg.is_downloaded("sd-compact")
    (snap / "unet" / "diffusion_pytorch_model.bin").write_bytes(b"0")
    assert reg.is_downloaded("sd-compact")


def test_bin_only_models_are_pinned_to_a_tested_commit(reg):
    # SD Compact ships .bin (pickle) weights: pin the exact commit so a changed repo can never be pulled
    m = reg.get("sd-compact")
    assert re.fullmatch(r"[0-9a-f]{40}", m.source.get("revision", ""))
