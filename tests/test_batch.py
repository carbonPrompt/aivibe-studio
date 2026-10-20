import json
import pytest
from pathlib import Path
from studio.batch import parse_batch, group_by_model, BatchError, BatchItem
from studio.registry import Registry

CATALOG = Path(__file__).parent.parent / "studio" / "catalog.json"
EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture
def reg(tmp_path):
    return Registry(CATALOG, tmp_path / "u.json", [tmp_path / "app"], tmp_path / "hf")


def P(text, reg, name="b.json"):
    return parse_batch(text, name, reg, fallback_model="realvis-v5")


def test_object_form_with_defaults(reg):
    items = P(json.dumps({"defaults": {"model": "flux-schnell", "count": 2},
                          "items": [{"prompt": "a dinosaur in a swamp", "name": "Dino"},
                                    {"prompt": "a lighthouse in a storm", "model": "realvis-v5", "seed": 42}]}), reg)
    a, b = items
    assert (a.index, a.model_id, a.count, a.subject, a.steps, a.guidance) == (1, "flux-schnell", 2, "dino", 4, 0.0)
    assert (b.model_id, b.seed, b.steps, b.subject, b.width) == ("realvis-v5", 42, 30, "lighthouse-storm", 1024)
    assert a.seed is None


def test_array_form_uses_fallback_model(reg):
    (a,) = P('[{"prompt": "a red sports car"}]', reg)
    assert a.model_id == "realvis-v5" and a.count == 1 and a.negative == ""


def test_txt_form(reg):
    items = P("# space set\nan astronaut on mars\n\n  a nebula  \n", reg, "list.txt")
    assert [i.prompt for i in items] == ["an astronaut on mars", "a nebula"]
    assert [i.index for i in items] == [1, 2]


def test_bom_accepted(reg):
    assert len(P("﻿" + '[{"prompt": "x y"}]', reg)) == 1


def test_all_problems_collected(reg):
    text = json.dumps([{"name": "no prompt"}, {"prompt": "a", "model": "fluxx"},
                       {"prompt": "b", "width": 1004}, {"prompt": "c", "count": 9}, {"promt": "d", "prompt": "d"}])
    with pytest.raises(BatchError) as e:
        P(text, reg)
    probs = e.value.problems
    assert any(p.startswith("item 1:") and "prompt" in p for p in probs)
    assert any(p.startswith("item 2:") and "fluxx" in p for p in probs)
    assert any(p.startswith("item 3:") and "multiple of 8" in p for p in probs)
    assert any(p.startswith("item 4:") and "count" in p for p in probs)
    assert any(p.startswith("item 5:") and "promt" in p for p in probs)


def test_invalid_json_reports_line(reg):
    with pytest.raises(BatchError) as e:
        P('[\n{"prompt": "a"},\n]', reg)
    assert "line 2" in e.value.problems[0]


def test_empty_batch_rejected(reg):
    with pytest.raises(BatchError):
        P("# nothing\n\n", reg, "x.txt")


def test_group_by_model_keeps_order(reg):
    items = P(json.dumps([{"prompt": "a", "model": "sd15"}, {"prompt": "b"}, {"prompt": "c", "model": "sd15"}]), reg)
    groups = group_by_model(items)
    assert [(m, [i.prompt for i in its]) for m, its in groups] == [("sd15", ["a", "c"]), ("realvis-v5", ["b"])]


def test_examples_parse(reg):
    assert len(parse_batch((EXAMPLES / "space.json").read_text(), "space.json", reg, "realvis-v5")) == 20
    assert len(parse_batch((EXAMPLES / "simple.txt").read_text(), "simple.txt", reg, "realvis-v5")) >= 3


def test_upscale_key(reg):
    (a,) = P('[{"prompt": "x y", "upscale": "4K"}]', reg)
    assert a.upscale == "4k"
    (b,) = P('[{"prompt": "x y"}]', reg)
    assert b.upscale is None
    with pytest.raises(BatchError) as e:
        P('[{"prompt": "x y", "upscale": "3x"}]', reg)
    assert e.value.problems[0].startswith("item 1:") and "upscale" in e.value.problems[0]


def test_subjects_and_reference_keys(reg, tmp_path):
    from studio.subjects import SubjectStore
    st = SubjectStore(tmp_path / "s")
    alex = st.create("Alex", "person", "man"); st.lora_path(alex.id).write_bytes(b"x")
    rex = st.create("Rex", "pet", "dog"); st.lora_path(rex.id).write_bytes(b"x")
    ref = tmp_path / "r.png"; ref.write_bytes(b"x")
    text = json.dumps([{"prompt": "a", "subjects": ["alex", {"name": "Rex", "weight": 0.7}], "reference": str(ref),
                        "reference_strength": 0.3}])
    (a,) = parse_batch(text, "b.json", reg, "realvis-v5", subjects=st)
    assert a.subjects == [(alex.id, 1.0), (rex.id, 0.7)] and a.reference == str(ref) and a.reference_strength == 0.3
    with pytest.raises(BatchError) as e:
        parse_batch(json.dumps([{"prompt": "a", "subjects": ["Nobody"]}, {"prompt": "b", "reference": "/no/such.png"},
                                {"prompt": "c", "subjects": ["a", "b", "c", "d"]}]), "b.json", reg, "realvis-v5", subjects=st)
    p = e.value.problems
    assert any("item 1" in x and "Nobody" in x for x in p) and any("item 2" in x and "reference" in x for x in p)
    assert any("item 3" in x and "3" in x for x in p)


def test_reference_rejects_network_paths_and_non_images(reg, tmp_path):
    from studio.batch import safe_reference
    img = tmp_path / "r.png"; img.write_bytes(b"x")
    assert safe_reference(str(img)) is None
    assert "image" in safe_reference(str(tmp_path / "r.exe"))
    # refused as paths, never checked on disk (on Windows a check of a network path sends credentials)
    import studio.batch as b
    touched = []
    real = b.Path.is_file
    b.Path.is_file = lambda self: touched.append(str(self)) or real(self)
    try:
        for bad in (r"\\host\share\r.png", "//host/share/r.png", r"\/host/share/r.png", r"/\host\share\r.png",
                    r"\\?\UNC\host\share\r.png", r"\\?\C:\r.png", "relative/r.png", r"C:r.png"):
            assert "not allowed" in safe_reference(bad), bad
        assert touched == []
    finally:
        b.Path.is_file = real
    with pytest.raises(BatchError) as e:
        parse_batch(json.dumps([{"prompt": "a", "reference": r"\\evil\share\x.png"}]), "b.json", reg, "realvis-v5")
    assert "network" in e.value.problems[0]


def test_reference_windows_rules():
    from studio.batch import safe_reference
    for bad in ("/share/r.png", r"C:\a\?\r.png", r"C:\a:b\r.png", r"\\.\C:\r.png", r"\??\C:\r.png"):
        assert "not allowed" in safe_reference(bad, windows=True), bad
    assert "not found" in safe_reference(r"C:\Users\someone\Pictures\nope.png", windows=True)
    assert "not allowed" in safe_reference(r"C:\x.png", windows=False)       # drive paths mean nothing on Linux


def test_batch_stores_the_stripped_reference(reg, tmp_path):
    img = tmp_path / "r.png"; img.write_bytes(b"x")
    (a,) = parse_batch(json.dumps([{"prompt": "a", "reference": f"  {img}  "}]), "b.json", reg, "realvis-v5")
    assert a.reference == str(img)


def test_subject_default_weight_used_when_not_given(reg, tmp_path):
    from studio.subjects import SubjectStore
    st = SubjectStore(tmp_path / "s2")
    car = st.create("Car", "car", "car"); st.lora_path(car.id).write_bytes(b"x"); st.set_default_weight(car.id, 1.1)
    (a,) = parse_batch(json.dumps([{"prompt": "the Car", "subjects": ["Car"]}]), "b.json", reg, "realvis-v5", subjects=st)
    assert a.subjects == [(car.id, 1.1)]
