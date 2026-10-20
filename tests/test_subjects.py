import json
from pathlib import Path

import pytest
from PIL import Image

from studio.subjects import SubjectStore, prep_photo, rewrite_prompt


@pytest.fixture
def store(tmp_path):
    return SubjectStore(tmp_path / "subjects")


def photo(path, size=(1600, 1200), color="gray", exif_orientation=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size, color)
    if exif_orientation:
        ex = Image.Exif(); ex[0x0112] = exif_orientation
        im.save(path, exif=ex)
    else:
        im.save(path)
    return path


def test_create_unique_ids_and_tokens(store):
    subs = [store.create("Alex", "person", "man") for _ in range(50)]
    assert len({s.id for s in subs}) == 50 and len({s.token for s in subs}) == 50
    assert all(s.token.startswith("zq") and len(s.token) == 5 for s in subs)
    assert subs[0].id == "alex" and subs[1].id == "alex-2"
    assert json.loads((store.root / "alex" / "subject.json").read_text())["class_word"] == "man"


def test_import_skips_small_and_non_images(store, tmp_path):
    s = store.create("Car", "car", "car")
    inbox = tmp_path / "inbox"
    photo(inbox / "a.jpg"); photo(inbox / "b.jpg", (400, 300)); (inbox / "notes.txt").write_text("hi")
    r = store.import_photos(s.id, inbox)
    assert r["added"] == 1 and {f for f, _ in r["skipped"]} == {"b.jpg", "notes.txt"}
    assert all(Image.open(p).size == (1024, 1024) for p in store.photo_paths(s.id))
    assert (inbox / "a.jpg").exists()          # originals untouched


def test_exif_rotation_applied(tmp_path):
    src = photo(tmp_path / "r.jpg", (1600, 1200), exif_orientation=6)   # stored landscape, displayed portrait
    dest = tmp_path / "out.png"
    seen = {}
    def finder(img):
        seen["size"] = img.size; return None
    assert prep_photo(src, dest, "person", face_finder=finder) is None
    assert seen["size"] == (1200, 1600) and Image.open(dest).size == (1024, 1024)


def test_person_crop_centres_on_face(tmp_path):
    im = Image.new("RGB", (2000, 1000), "black")
    for x in range(1700, 1800):
        for y in range(400, 500):
            im.putpixel((x, y), (255, 255, 255))
    src = tmp_path / "f.png"; im.save(src)
    dest = tmp_path / "o.png"
    prep_photo(src, dest, "person", face_finder=lambda img: (1700, 400, 100, 100))
    out = Image.open(dest)
    assert out.getpixel((768, 460)) == (255, 255, 255)   # face kept: crop is the right half (x 1000-2000)
    assert out.getpixel((100, 460)) == (0, 0, 0)


def test_delete_and_rename(store):
    s = store.create("Casey", "person", "woman")
    store.rename(s.id, "Tam")
    t = store.get(s.id)
    assert t.name == "Tam" and t.token == s.token
    store.delete(s.id)
    assert not (store.root / s.id).exists() and store.all() == []


def test_trained_flag(store):
    s = store.create("Alex", "person", "man")
    assert not store.get(s.id).trained
    store.lora_path(s.id).write_bytes(b"x")
    assert store.get(s.id).trained


# ---- prompt rewriting ----
def subj(store, name, cls):
    return store.create(name, "person" if cls in ("man", "woman", "person") else "car", cls)


def test_rewrite_names(store):
    alex, tam = subj(store, "Alex", "man"), subj(store, "Casey", "woman")
    out = rewrite_prompt("Alex and Casey on a beach", [alex, tam])
    assert out == f"{alex.token} man and {tam.token} woman on a beach"


def test_rewrite_whole_word_only(store):
    alex = subj(store, "Alex", "man")
    out = rewrite_prompt("Samantha waves at alex", [alex])
    assert out == f"Samantha waves at {alex.token} man"


def test_rewrite_prepends_unnamed(store):
    alex, car = subj(store, "Alex", "man"), subj(store, "Car", "car")
    assert rewrite_prompt("the car at night", [car]) == f"the {car.token} car at night"
    assert rewrite_prompt("a mountain road", [alex, car]) == f"{alex.token} man, {car.token} car, a mountain road"
    assert rewrite_prompt("Alex on a mountain road", [alex, car]) == f"{car.token} car, {alex.token} man on a mountain road"


def test_broken_exif_still_rotates(tmp_path, monkeypatch):
    # live finding: Pillow's exif_transpose raised TypeError on one of Alex's phone photos and stopped the import
    import studio.subjects as sm
    src = photo(tmp_path / "r.jpg", (1600, 1200), exif_orientation=6)
    def boom(im): raise TypeError("bad operand type for abs(): 'bytes'")
    monkeypatch.setattr(sm.ImageOps, "exif_transpose", boom)
    seen = {}
    assert prep_photo(src, tmp_path / "o.png", "person", face_finder=lambda im: seen.setdefault("s", im.size) and None) is None
    assert seen["s"] == (1200, 1600)


def test_one_bad_photo_does_not_stop_import(store, tmp_path, monkeypatch):
    import studio.subjects as sm
    s = store.create("Alex", "person", "man")
    inbox = tmp_path / "in"; photo(inbox / "a.jpg"); photo(inbox / "b.jpg")
    real = sm.prep_photo
    def flaky(src, dest, kind, face_finder=None):
        if src.name == "a.jpg":
            raise RuntimeError("weird file")
        return real(src, dest, kind, face_finder=lambda im: None)
    monkeypatch.setattr(sm, "prep_photo", flaky)
    r = store.import_photos(s.id, inbox)
    assert r["added"] == 1 and r["skipped"][0][0] == "a.jpg"


def test_default_weight_per_subject(store):
    s = store.create("Car", "car", "car")
    assert store.get(s.id).default_weight == 1.0
    store.set_default_weight(s.id, 1.1)
    assert store.get(s.id).default_weight == 1.1


def test_remove_photo(store, tmp_path):
    s = store.create("Alex", "person", "man")
    inbox = tmp_path / "in"; photo(inbox / "a.jpg"); photo(inbox / "b.jpg")
    store.import_photos(s.id, inbox, face_finder=lambda im: None)
    store.remove_photo(s.id, "a.png")
    assert [p.name for p in store.photo_paths(s.id)] == ["b.png"] and store.get(s.id).photo_count == 1
    import pytest as _p
    with _p.raises(ValueError):
        store.remove_photo(s.id, "../subject.json")       # only files inside photos\


@pytest.mark.parametrize("bad", ["../x", "..\\x", "\\\\host\\share", "//host/share", "C:\\x", "a/b", "", "UPPER"])
def test_subject_ids_from_outside_are_never_paths(store, bad):
    with pytest.raises(KeyError):
        store.get(bad)
    with pytest.raises(KeyError):
        store.lora_path(bad)
    with pytest.raises(KeyError):
        store.delete(bad)


def test_subject_id_comes_from_its_folder_not_the_file(store):
    s = store.create("Alex", "person", "man")
    f = store.root / s.id / "subject.json"
    data = json.loads(f.read_text()); data["id"] = "../evil"; f.write_text(json.dumps(data))
    assert store.get(s.id).id == s.id and [x.id for x in store.all()] == [s.id]
