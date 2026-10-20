from datetime import datetime
from studio.naming import subject_from_prompt, slugify, output_stem, unique_path, day_folder


def test_subject_from_prompt_keeps_leading_nouns():
    assert subject_from_prompt("A dinosaur in its natural swamp environment") == "dinosaur-swamp"


def test_subject_drops_style_words():
    assert subject_from_prompt("ultra detailed cinematic photo of a red sports car, 8k") == "red-sports-car"


def test_subject_never_empty():
    assert subject_from_prompt("!!! 🌟 the of and") == "image"
    assert subject_from_prompt("") == "image"


def test_slugify_name_box():
    assert slugify("My  Space Station!") == "my-space-station"
    assert slugify("one two three four") == "one-two-three"
    assert slugify("   ") == "image"


def test_output_stem():
    t = datetime(2026, 10, 18, 16, 52, 7)
    assert output_stem("dinosaur-swamp", t) == "generated_dinosaur-swamp_2026-10-18_1652"
    assert output_stem("dinosaur-swamp", t, index=2) == "generated_dinosaur-swamp_2026-10-18_1652_2"
    assert output_stem("dinosaur-swamp", t, scale="4x") == "generated_dinosaur-swamp_2026-10-18_1652_4x"


def test_unique_path_never_overwrites(tmp_path):
    a = unique_path(tmp_path, "generated_x_2026-10-18_1652"); a.write_bytes(b"1")
    b = unique_path(tmp_path, "generated_x_2026-10-18_1652"); b.write_bytes(b"2")
    c = unique_path(tmp_path, "generated_x_2026-10-18_1652")
    assert len({a, b, c}) == 3 and not c.exists()
    assert all(p.name.startswith("generated_x_2026-10-18_1652") and p.suffix == ".png" for p in (a, b, c))


def test_day_folder_created(tmp_path):
    d = day_folder(tmp_path, datetime(2026, 10, 18, 9, 0))
    assert d == tmp_path / "2026-10-18" and d.is_dir()


def test_unique_path_does_not_share_a_recipe_across_formats(tmp_path):
    (tmp_path / "generated_x_2026-10-18_1652.png").write_bytes(b"1")
    j = unique_path(tmp_path, "generated_x_2026-10-18_1652", ".jpg")
    assert j.suffix == ".jpg" and j.stem != "generated_x_2026-10-18_1652"      # x.json would be shared
