"""One model download from Hugging Face, run as a child process so Cancel can stop it at once:
python -m studio.hfget <repo> <cache_dir> [revision]   (token, if any, in the HF_TOKEN environment variable)
The last stdout line is JSON: {"ok": true} or {"error": "...", "status": 401}."""
import json
import os
import sys


class FetchError(Exception):
    def __init__(self, message: str, status: int | None):
        super().__init__(message)
        self.status = status


def download_patterns(files: list[str]) -> list[str]:
    """What to fetch from a diffusers repo: the component folders only (root single-file checkpoints duplicate
    them), only the fp16 weights when the repo has them, which the loader then asks for, and .bin weights
    only for components that have no .safetensors at all."""
    pats = ["model_index.json", "*/*.json", "*/*.txt", "*/*.model", "*/*.jinja"]
    has_fp16 = any("/" in f and f.endswith(".fp16.safetensors") for f in files)
    weights = [f for f in files if "/" in f and f.endswith(".safetensors")]
    if has_fp16:
        # fp16 variant where a component has one, plain weights for components that do not
        comps_fp16 = {f.split("/")[0] for f in weights if f.endswith(".fp16.safetensors")}
        pats.append("*/*.fp16.safetensors")
        pats += [f for f in weights if f.split("/")[0] not in comps_fp16]
    else:
        pats.append("*/*.safetensors")
    # older repos ship some components only as .bin (torch.load in weights-only mode reads them safely)
    with_st = {f.split("/")[0] for f in weights}
    pats += [f for f in files if f.count("/") == 1 and f.endswith(".bin") and f.split("/")[0] not in with_st]
    return pats


def fetch(repo: str, cache_dir: str, token: str | None, hub=None, revision: str | None = None) -> None:
    if hub is None:
        import huggingface_hub as hub
    try:
        files = hub.HfApi().list_repo_files(repo, revision=revision, token=token)
        hub.snapshot_download(repo, cache_dir=cache_dir, allow_patterns=download_patterns(files), token=token,
                              revision=revision)
    except Exception as e:
        raise FetchError(str(e), getattr(getattr(e, "response", None), "status_code", None)) from e


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    repo, cache_dir, revision = args[0], args[1], (args[2] if len(args) > 2 else None)
    try:
        fetch(repo, cache_dir, os.environ.get("HF_TOKEN") or None, revision=revision)
    except FetchError as e:
        print(json.dumps({"error": str(e), "status": e.status}), flush=True)
        return 1
    print(json.dumps({"ok": True}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
