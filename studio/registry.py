"""Model registry: the built-in catalog plus models the user adds, and where each one's weights are on disk."""
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

FAMILIES = {
    # family: (pipeline class, default steps, guidance, size)
    "sd15": ("StableDiffusionPipeline", 25, 7.0, 512),
    "sdxl": ("StableDiffusionXLPipeline", 30, 6.0, 1024),
    "playground": ("StableDiffusionXLPipeline", 30, 3.0, 1024),
    "sd3": ("StableDiffusion3Pipeline", 28, 4.5, 1024),
    "flux": ("FluxPipeline", 4, 0.0, 1024),
    "flux2": ("Flux2KleinPipeline", 4, 1.0, 1024),
    "zimage": ("ZImagePipeline", 8, 0.0, 1024),
}
LICENSES = {"commercial", "conditions", "nc"}


@dataclass
class ModelEntry:
    id: str
    name: str
    family: str
    pipeline: str
    source: dict
    license: str
    defaults: dict
    vram_mb: int = 10000
    offload: bool = False
    size_gb: float = 0.0
    gated: bool = False          # Hugging Face wants an account, accepted terms and a token
    min_vram_gb: float = 0.0     # below this the model offloads (slow) or does not run
    app_folder_prefix: str | None = None
    user: bool = False
    description: str = ""


def _complete_snapshot(snap: Path) -> bool:
    """model_index.json present and every component it names has a folder with at least one file in it, and
    model components (config.json) have weights: model_index.json and the configs are among the first files
    a download writes, so on their own they prove nothing."""
    try:
        index = json.loads((snap / "model_index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    for name, v in index.items():
        if name.startswith("_") or not isinstance(v, list) or v[0] is None:
            continue
        d = snap / name
        if not d.is_dir() or not any(d.iterdir()):
            return False
        # a model component (it has config.json) needs its weights too, not just the config
        if (d / "config.json").exists() and not any(d.glob("*.safetensors")) and not any(d.glob("*.bin")):
            return False
    return True


class Registry:
    def __init__(self, catalog_path: Path, user_path: Path, extra_model_dirs: list, hf_home: Path):
        self.user_path = Path(user_path)
        self.extra_model_dirs = [Path(d) for d in extra_model_dirs]     # searched in order, read-only
        self.hf_home = Path(hf_home)
        self._models = {}
        for d in json.loads(Path(catalog_path).read_text(encoding="utf-8"))["models"]:
            self._models[d["id"]] = ModelEntry(**d)
        if self.user_path.exists():
            for d in json.loads(self.user_path.read_text(encoding="utf-8"))["models"]:
                self._models[d["id"]] = ModelEntry(**d)

    def all(self) -> list[ModelEntry]:
        return list(self._models.values())

    def get(self, id: str) -> ModelEntry:
        return self._models[id]

    def local_path(self, id: str) -> Path | None:
        m = self.get(id)
        if "file" in m.source:
            p = Path(m.source["file"])
            return p if p.exists() else None
        if m.app_folder_prefix:
            for root in self.extra_model_dirs:
                if not root.is_dir():
                    continue
                for d in sorted(root.iterdir()):
                    if d.is_dir() and d.name.startswith(m.app_folder_prefix) and (d / "model_index.json").exists():
                        return d
        root = self.hf_repo_dir(id)
        snaps = root / "snapshots"
        if snaps.is_dir() and not (root / ".studio-downloading").exists() \
                and not any((root / "blobs").glob("*.incomplete")):
            for s in sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if _complete_snapshot(s):
                    return s
        return None

    def hf_repo_dir(self, id: str) -> Path:
        """The Hugging Face cache folder of an hf-sourced model: <hf_home>/hub/models--<org>--<name>."""
        return self.hf_home / "hub" / ("models--" + self.get(id).source["hf"].replace("/", "--"))

    def is_downloaded(self, id: str) -> bool:
        return self.local_path(id) is not None

    def _new_entry(self, name, family, license, source) -> ModelEntry:
        if family not in FAMILIES:
            raise ValueError(f"unknown family {family!r}, expected one of {sorted(FAMILIES)}")
        if license not in LICENSES:
            raise ValueError(f"unknown license class {license!r}")
        pipe, steps, guidance, size = FAMILIES[family]
        base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "model"
        id, k = base, 1
        while id in self._models:
            k += 1
            id = f"{base}-{k}"
        m = ModelEntry(id=id, name=name, family=family, pipeline=pipe, source=source, license=license,
                       defaults=dict(steps=steps, guidance=guidance, width=size, height=size),
                       offload=family in ("sd3", "flux", "flux2", "zimage"), user=True)
        self._models[id] = m
        self._save_user()
        return m

    def add_hf(self, repo_id: str, name: str, family: str, license: str) -> ModelEntry:
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo_id or ""):
            raise ValueError(f"not a Hugging Face id (expected owner/name): {repo_id!r}")
        return self._new_entry(name, family, license, {"hf": repo_id})

    def add_file(self, path: Path, name: str, family: str, license: str) -> ModelEntry:
        path = Path(path)
        if path.suffix.lower() != ".safetensors":
            raise ValueError("only .safetensors files can be added")
        if not path.exists():
            raise ValueError(f"file not found: {path}")
        return self._new_entry(name, family, license, {"file": str(path)})

    def remove_user(self, id: str) -> None:
        if not self.get(id).user:
            raise ValueError("built-in models cannot be removed")
        del self._models[id]
        self._save_user()

    def _save_user(self):
        self.user_path.parent.mkdir(parents=True, exist_ok=True)
        users = [asdict(m) for m in self._models.values() if m.user]
        self.user_path.write_text(json.dumps({"models": users}, indent=1), encoding="utf-8")
