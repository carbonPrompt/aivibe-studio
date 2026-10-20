"""Batch files: a JSON list of prompts (or {defaults, items}), or a .txt file with one prompt per line."""
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from studio.naming import slugify, subject_from_prompt
from studio.upscale import parse_choice

KEYS = {"prompt", "name", "negative", "model", "width", "height", "count", "seed", "steps", "guidance", "upscale",
        "subjects", "reference", "reference_strength", "face_fix"}
DEFAULT_LIKENESS = 1.0      # full strength unless the user lowers it


@dataclass
class BatchItem:
    index: int
    prompt: str
    negative: str
    model_id: str
    width: int
    height: int
    count: int
    seed: int | None
    steps: int
    guidance: float
    subject: str
    upscale: str | None = None
    subjects: list = field(default_factory=list)     # [(subject id, weight)]
    reference: str | None = None
    reference_strength: float = 0.5
    face_fix: bool = True
    meta: dict = field(default_factory=dict)         # extra recipe fields, e.g. which assistant wrote the batch


class BatchError(ValueError):
    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def _raw_items(text: str, filename: str) -> tuple[dict, list]:
    text = text.lstrip("﻿")
    if filename.lower().endswith(".txt"):
        lines = [l.strip() for l in text.splitlines()]
        return {}, [{"prompt": l} for l in lines if l and not l.startswith("#")]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise BatchError([f"not valid JSON: {e.msg} at line {e.lineno}, column {e.colno}"])
    if isinstance(data, list):
        return {}, data
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data.get("defaults") or {}, data["items"]
    raise BatchError(["expected a list of items, or an object with \"items\""])


REF_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


_WIN_DRIVE = re.compile(r"^[A-Za-z]:[\\/](?![\\/])")


def safe_reference(path: str, windows: bool = os.name == "nt") -> str | None:
    """None if `path` is an acceptable local reference image, else the problem. Decided from the text alone,
    before any disk access: on Windows even an existence check of a network or device path sends the user's
    credentials. Windows: a drive-letter path only, no '?', no second ':', no doubled separators. Elsewhere: an
    absolute path with a single leading '/'."""
    p = str(path).strip()
    if windows:
        ok = bool(_WIN_DRIVE.match(p)) and "?" not in p and ":" not in p[2:] \
            and not re.search(r"[\\/]{2}", p[2:])
    else:
        ok = p.startswith("/") and not re.search(r"[\\/]{2}", p) and "\\" not in p
    if not ok:
        return "only local image files are allowed as references (network, device and relative paths are not allowed)"
    if Path(p).suffix.lower() not in REF_EXT:
        return "reference must be a .png, .jpg, .webp or .bmp image"
    if not Path(p).is_file():
        return f"reference image not found: {p}"
    return None


def _resolve_subjects(raw, store, n, problems) -> list:
    if raw in (None, []):
        return []
    if store is None:
        problems.append(f"item {n}: subjects are not available"); return []
    if not isinstance(raw, list):
        raw = [raw]
    if len(raw) > 3:
        problems.append(f"item {n}: at most 3 subjects in one picture"); return []
    by_key = {}
    for s in store.all():
        by_key[s.id.lower()] = s; by_key[s.name.lower()] = s
    out = []
    for r in raw:
        name, weight = (r.get("name") or r.get("id"), r.get("weight")) if isinstance(r, dict) else (r, None)
        s = by_key.get(str(name).lower())
        if s is not None and weight is None:
            weight = s.default_weight
        if s is None:
            problems.append(f"item {n}: unknown subject {name!r}"); continue
        if not s.trained:
            problems.append(f"item {n}: subject {s.name!r} is not trained yet"); continue
        try:
            out.append((s.id, float(weight)))
        except (TypeError, ValueError):
            problems.append(f"item {n}: bad weight for {s.name!r}")
    return out


def parse_batch(text: str, filename: str, registry, fallback_model: str, subjects=None) -> list[BatchItem]:
    defaults, raw = _raw_items(text, filename)
    if not raw:
        raise BatchError(["the batch has no prompts"])
    problems, items = [], []
    for n, d in enumerate(raw, 1):
        if not isinstance(d, dict):
            problems.append(f"item {n}: expected an object with a prompt"); continue
        bad = (set(d) | set(defaults)) - KEYS
        for k in sorted(bad):
            problems.append(f"item {n}: unknown key {k!r}")
        v = {**defaults, **d}
        prompt = str(v.get("prompt") or "").strip()
        if not prompt:
            problems.append(f"item {n}: missing prompt"); continue
        model_id = v.get("model") or fallback_model
        try:
            md = registry.get(model_id).defaults
        except KeyError:
            problems.append(f"item {n}: unknown model {model_id!r}"); continue
        try:
            width, height = int(v.get("width", md["width"])), int(v.get("height", md["height"]))
            count = int(v.get("count", 1))
            steps = int(v.get("steps", md["steps"]))
            guidance = float(v.get("guidance", md["guidance"]))
            seed = None if v.get("seed") in (None, -1) else int(v["seed"])
        except (TypeError, ValueError) as e:
            problems.append(f"item {n}: bad number ({e})"); continue
        for label, x in (("width", width), ("height", height)):
            if x % 8 or not 256 <= x <= 2048:
                problems.append(f"item {n}: {label} {x} must be a multiple of 8 between 256 and 2048")
        if not 1 <= count <= 8:
            problems.append(f"item {n}: count {count} must be 1 to 8")
        if not 1 <= steps <= 150:
            problems.append(f"item {n}: steps {steps} must be 1 to 150")
        if seed is not None and not 0 <= seed < 2**32:
            problems.append(f"item {n}: seed must be 0 to 4294967295")
        upscale = None
        if v.get("upscale") not in (None, ""):
            try:
                upscale = parse_choice(v["upscale"])
            except ValueError as e:
                problems.append(f"item {n}: {e}")
        subs = _resolve_subjects(v.get("subjects"), subjects, n, problems)
        ref = str(v.get("reference") or "").strip() or None
        ref_problem = safe_reference(ref) if ref else None
        if ref_problem:
            problems.append(f"item {n}: {ref_problem}")
        try:
            ref_strength = float(v.get("reference_strength", 0.5))
        except (TypeError, ValueError):
            problems.append(f"item {n}: bad reference_strength"); ref_strength = 0.5
        subject = slugify(str(d["name"])) if d.get("name") else subject_from_prompt(prompt)
        items.append(BatchItem(n, prompt, str(v.get("negative") or ""), model_id, width, height, count, seed,
                               steps, guidance, subject, upscale, subs, ref, ref_strength,
                               bool(v.get("face_fix", True))))
    if problems:
        raise BatchError(problems)
    return items


def group_by_model(items: list[BatchItem]) -> list[tuple[str, list[BatchItem]]]:
    groups: dict[str, list[BatchItem]] = {}
    for i in items:
        groups.setdefault(i.model_id, []).append(i)
    return list(groups.items())
