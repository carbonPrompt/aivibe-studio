import os

import pytest

from studio.keys import KeyStore


def test_keys_round_trip_and_mask(tmp_path):
    f = tmp_path / "assistant_keys.json"
    ks = KeyStore(f)
    assert ks.get("gemini") is None and ks.masked("gemini") is None
    ks.set("gemini", "  AIzaSyEXAMPLE1234  ")
    assert ks.get("gemini") == "AIzaSyEXAMPLE1234"
    assert ks.masked("gemini") == "saved, ends in 1234"
    assert os.name != "nt" or "AIzaSyEXAMPLE1234" not in f.read_text()     # DPAPI-protected on Windows
    assert KeyStore(f).get("gemini") == "AIzaSyEXAMPLE1234"                  # survives a restart
    ks.remove("gemini")
    assert ks.get("gemini") is None and KeyStore(f).get("gemini") is None


def test_unknown_provider_and_empty_key_refused(tmp_path):
    ks = KeyStore(tmp_path / "k.json")
    with pytest.raises(ValueError):
        ks.set("bard", "x")
    with pytest.raises(ValueError):
        ks.set("claude", "   ")


def test_keys_never_land_in_settings(tmp_path):
    KeyStore(tmp_path / "assistant_keys.json").set("claude", "sk-ant-SECRET")
    assert [p.name for p in tmp_path.iterdir()] == ["assistant_keys.json"]


def test_corrupt_file_reads_as_empty(tmp_path):
    f = tmp_path / "k.json"; f.write_text("{not json")
    assert KeyStore(f).get("openai") is None


def test_keystore_accepts_huggingface(tmp_path):
    ks = KeyStore(tmp_path / "k.json")
    ks.set("huggingface", "hf_EXAMPLE9876")
    assert ks.get("huggingface") == "hf_EXAMPLE9876" and ks.masked("huggingface") == "saved, ends in 9876"
