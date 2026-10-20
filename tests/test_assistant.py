import json
from pathlib import Path

import pytest

from studio import assistant, providers
from studio.assistant import MAX_ITEMS, brief, check_assistant_batch, extract_json, revise_batch, write_batch
from studio.providers import ProviderError, Reply
from studio.registry import Registry
from studio.subjects import SubjectStore

CATALOG = Path(__file__).parent.parent / "studio" / "catalog.json"


@pytest.fixture
def reg(tmp_path):
    app = tmp_path / "app"; d = app / "realvisxl_v5.0-1"; d.mkdir(parents=True); (d / "model_index.json").write_text("{}")
    return Registry(CATALOG, tmp_path / "u.json", [app], tmp_path / "hf")


@pytest.fixture
def store(tmp_path):
    st = SubjectStore(tmp_path / "subjects")
    c = st.create("Jordan", "person", "woman"); st.lora_path(c.id).write_bytes(b"x")
    st.create("Casey", "person", "woman")                  # not trained
    return st


class FakeProvider(providers.Provider):
    id, label, default_model = "fake", "Fake AI", "m"

    def __init__(self):
        self.replies, self.calls = [], []

    def queue(self, *texts):
        self.replies.extend(texts)

    def complete(self, key, model, system, user, timeout=120.0):
        self.calls.append({"system": system, "user": user, "key": key, "model": model})
        return Reply(self.replies.pop(0), model, 10)


@pytest.fixture
def fake(monkeypatch):
    f = FakeProvider()
    monkeypatch.setitem(providers.PROVIDERS, "fake", f)
    return f


def test_brief_lists_installed_models_and_trained_subjects_only_text(reg, store):
    b = brief(reg, store)
    assert "realvis-v5" in b and "installed yes" in b and "installed no" in b
    assert "Jordan (person, woman)" in b and "Casey" not in b
    tok = store.get("jordan").token
    assert tok not in b and str(store.root) not in b and "C:\\" not in b
    keys_line = [l for l in b.splitlines() if l.startswith("Allowed keys")][0]
    assert "reference" not in keys_line and "prompt" in keys_line and "upscale" in keys_line
    assert "1344x768" in b and "50 words" in b and "JSON only" in b


def test_brief_without_subjects(reg, tmp_path):
    assert "Trained subjects: none" in brief(reg, SubjectStore(tmp_path / "empty"))


@pytest.mark.parametrize("text", ['{"items":[{"prompt":"a"}]}', '```json\n{"items":[{"prompt":"a"}]}\n```',
                                  'Sure! Here it is: {"items":[{"prompt":"a"}]} Enjoy', '[{"prompt":"a"}]'])
def test_extract_json_finds_the_batch(text):
    d = extract_json(text)
    assert d["items"] == [{"prompt": "a"}] and d["defaults"] == {}


def test_extract_json_keeps_defaults():
    assert extract_json('{"defaults":{"count":2},"items":[{"prompt":"a"}]}')["defaults"] == {"count": 2}


def test_extract_json_explains_a_refusal():
    with pytest.raises(ValueError, match="did not send a batch. It said: I can't help"):
        extract_json("I can't help with that.")


def test_check_flags_reference_size_model_and_subject_rules(reg, store):
    probs = check_assistant_batch({"defaults": {}, "items": [
        {"prompt": "x", "reference": "C:/a.png"},
        {"prompt": "x", "width": 1001, "height": 1024},
        {"prompt": "x", "model": "flux2-klein-4b"},
        {"prompt": "Jordan", "model": "sd35-large", "subjects": ["Jordan"]}]}, reg, store, "realvis-v5")
    text = "\n".join(probs)
    assert "item 1: the assistant may not add reference images" in probs
    assert "item 2: width 1001 must be a multiple of 8" in text
    assert "item 3: model 'flux2-klein-4b' is not downloaded" in probs
    assert "item 4: subjects need an SDXL model" in probs


def test_check_passes_a_good_batch(reg, store):
    assert check_assistant_batch({"defaults": {"upscale": "1080p"}, "items": [
        {"prompt": "Jordan in a garden", "subjects": ["Jordan"], "width": 1344, "height": 768}]}, reg, store, "realvis-v5") == []


def test_too_many_items(reg, store):
    probs = check_assistant_batch({"defaults": {}, "items": [{"prompt": "x"}] * (MAX_ITEMS + 1)}, reg, store, "realvis-v5")
    assert "too many lines (51); the limit is 50" in probs


def test_write_batch_fixes_once_then_returns(reg, store, fake):
    fake.queue('{"items":[{"prompt":"a","width":1001}]}', '{"items":[{"prompt":"a","width":1024}]}')
    d = write_batch("fake", "KEY", "m", "one picture", reg, store, "realvis-v5")
    assert d.problems == [] and d.batch["items"] == [{"prompt": "a", "width": 1024}]
    assert len(fake.calls) == 2 and "Fix these problems" in fake.calls[1]["user"] and "width 1001" in fake.calls[1]["user"]
    assert fake.calls[0]["user"] == "one picture" and "JSON only" in fake.calls[0]["system"]
    assert (d.provider, d.model, d.tokens) == ("fake", "m", 20)


def test_write_batch_stops_after_one_fix(reg, store, fake):
    fake.queue('{"items":[{"prompt":"a","width":1001}]}', '{"items":[{"prompt":"a","width":1003}]}')
    d = write_batch("fake", "KEY", "m", "one picture", reg, store, "realvis-v5")
    assert len(fake.calls) == 2 and any("width 1003" in p for p in d.problems)


def test_write_batch_good_first_time_makes_one_call(reg, store, fake):
    fake.queue('{"items":[{"prompt":"a"}]}')
    assert write_batch("fake", "KEY", "m", "x", reg, store, "realvis-v5").problems == [] and len(fake.calls) == 1


def test_write_batch_no_json_twice_is_an_error(reg, store, fake):
    fake.queue("no.", "still no.")
    with pytest.raises(ProviderError, match="did not send a batch"):
        write_batch("fake", "KEY", "m", "x", reg, store, "realvis-v5")


def test_revise_sends_current_batch_and_instruction(reg, store, fake):
    fake.queue('{"items":[{"prompt":"a at night"}]}')
    d = revise_batch("fake", "KEY", "m", "one picture", {"defaults": {}, "items": [{"prompt": "a"}]},
                     "make it night", reg, store, "realvis-v5")
    u = fake.calls[0]["user"]
    assert u.startswith("one picture") and "Current batch:" in u and '"prompt": "a"' in u and u.endswith("Change it like this: make it night")
    assert d.batch["items"][0]["prompt"] == "a at night"


@pytest.mark.parametrize("batch,expect", [
    ({"defaults": [], "items": [{"prompt": "a"}]}, "defaults must be an object of shared settings"),
    ({"defaults": "none", "items": [{"prompt": "a"}]}, "defaults must be an object of shared settings"),
    ({"defaults": {}, "items": [{"prompt": "a", "model": ["realvis-v5"]}]}, "item 1: model must be one model id"),
    ({"defaults": {}, "items": [{"prompt": ["a"]}]}, "item 1: prompt must be text"),
    ({"defaults": {}, "items": [{"prompt": "a", "count": 1e400}]}, "item 1: count must be a number"),
])
def test_check_turns_odd_ai_values_into_problems(reg, store, batch, expect):
    assert expect in check_assistant_batch(batch, reg, store, "realvis-v5")


def test_extract_json_turns_plain_string_items_into_prompts():
    assert extract_json('{"items": ["a fox", {"prompt": "b"}]}')["items"] == [{"prompt": "a fox"}, {"prompt": "b"}]


def test_brief_names_the_fallback_model_and_list_shapes(reg, store):
    b = brief(reg, store, "realvis-v5")
    assert "Items without a model use realvis-v5 (RealVisXL V5.0, family sdxl)." in b
    assert '"subjects": ["Name"]' in b
