from types import SimpleNamespace

import pytest

from studio import hfget


class Hub:
    def __init__(self, error=None):
        self.error, self.calls, self.revisions = error, [], []
        hub = self
        class HfApi:
            def list_repo_files(self, repo, revision=None, token=None):
                hub.calls.append(("list", repo, token)); hub.revisions.append(revision); return ["model_index.json", "unet/x.safetensors"]
        self.HfApi = HfApi
    def snapshot_download(self, repo, cache_dir, allow_patterns, token=None, revision=None):
        self.revisions.append(revision)
        self.calls.append(("get", repo, cache_dir, token))
        if self.error:
            raise self.error


def test_fetch_ok_passes_token_and_patterns():
    hub = Hub()
    assert hfget.fetch("a/b", "C:\\hf\\hub", "hf_x", hub=hub) is None
    assert hub.calls == [("list", "a/b", "hf_x"), ("get", "a/b", "C:\\hf\\hub", "hf_x")]


def test_fetch_http_error_carries_status():
    e = Exception("403 Client Error"); e.response = SimpleNamespace(status_code=403)
    with pytest.raises(hfget.FetchError) as x:
        hfget.fetch("a/b", "c", None, hub=Hub(e))
    assert x.value.status == 403


def test_main_reports_json_line(capsys, monkeypatch):
    monkeypatch.setattr(hfget, "fetch", lambda repo, cache, token, hub=None, revision=None: (_ for _ in ()).throw(hfget.FetchError("401 nope", 401)))
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert hfget.main(["a/b", "c"]) == 1
    assert capsys.readouterr().out.strip().splitlines()[-1] == '{"error": "401 nope", "status": 401}'


def test_patterns_take_bin_weights_for_components_without_safetensors():
    import fnmatch
    files = ["model_index.json", "unet/config.json", "unet/diffusion_pytorch_model.bin",
             "vae/config.json", "vae/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.bin",
             "text_encoder/config.json", "text_encoder/pytorch_model.bin", "pytorch_model.bin"]
    pats = hfget.download_patterns(files)
    picked = {f for f in files if any(fnmatch.fnmatch(f, p) for p in pats)}
    assert {"unet/diffusion_pytorch_model.bin", "text_encoder/pytorch_model.bin",
            "vae/diffusion_pytorch_model.safetensors"} <= picked
    assert "vae/diffusion_pytorch_model.bin" not in picked and "pytorch_model.bin" not in picked


def test_fetch_pins_the_revision_when_given():
    hub = Hub()
    hfget.fetch("a/b", "c", None, hub=hub, revision="f" * 40)
    assert hub.revisions == ["f" * 40, "f" * 40]
    hub = Hub()
    hfget.fetch("a/b", "c", None, hub=hub)
    assert hub.revisions == [None, None]


def test_main_takes_an_optional_revision(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(hfget, "fetch", lambda repo, cache, token, hub=None, revision=None: seen.append(revision))
    hfget.main(["a/b", "c", "abc123"]); hfget.main(["a/b", "c"])
    assert seen == ["abc123", None]
