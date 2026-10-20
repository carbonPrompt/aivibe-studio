"""The recipe note saved beside every image: everything needed to re-make or tweak it."""
import json
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path


@dataclass
class Recipe:
    prompt: str
    negative: str
    model_id: str
    model_name: str
    license: str
    seed: int
    steps: int
    guidance: float
    width: int
    height: int
    subject: str
    app_version: str
    created: str
    batch_file: str | None = None
    extra: dict = field(default_factory=dict)


def write_recipe(png_path: Path, r: Recipe) -> Path:
    p = Path(png_path).with_suffix(".json")
    d = asdict(r)
    d.update(d.pop("extra"))
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def read_recipe(path: Path) -> Recipe:
    d = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
    names = {f.name for f in fields(Recipe)} - {"extra"}
    known = {k: v for k, v in d.items() if k in names}
    extra = {k: v for k, v in d.items() if k not in names and k != "extra"}
    extra.update(d.get("extra") or {})
    return Recipe(**known, extra=extra)
