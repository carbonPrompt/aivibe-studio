"""The assistant: a cloud AI writes a batch from the user's request. It gets text only (the batch rules, installed
models, trained subject names) and never runs anything: the batch goes to the review screen first."""
import json
import logging
import re
import time
from dataclasses import dataclass

from studio.batch import KEYS, BatchError, parse_batch
from studio.providers import PROVIDERS, ProviderError

log = logging.getLogger("studio.assistant")
MAX_ITEMS = 50
NO_KEYS = {"reference", "reference_strength"}     # a cloud AI never names files on this PC
SIZES = "1024x1024 (square), 1344x768 (16:9), 768x1344 (9:16), 1216x832 (3:2), 832x1216 (2:3)"


@dataclass
class Draft:
    batch: dict
    problems: list[str]
    provider: str
    model: str
    tokens: int
    seconds: float


def brief(registry, subjects, fallback_model: str | None = None) -> str:
    """The system text: batch rules, the model table and the trained subjects. No tokens, paths or photos."""
    models = []
    for m in registry.all():
        d = m.defaults
        models.append(f"{m.id} | {m.name} | {m.family} | installed {'yes' if registry.local_path(m.id) else 'no'} | "
                      f"{d.get('steps')} steps, guidance {d.get('guidance')}, {d.get('width')}x{d.get('height')} | "
                      f"{m.description}")
    subs = [f"{s.name} ({s.kind}, {s.class_word})" for s in (subjects.all() if subjects else []) if s.trained]
    try:
        fb = registry.get(fallback_model) if fallback_model else None
    except KeyError:
        fb = None
    fallback = f"Items without a model use {fb.id} ({fb.name}, family {fb.family})." if fb else ""
    return "\n".join([
        "You write picture batches for aiVibe Studio, a local image generator. You never make pictures yourself.",
        "Reply with JSON only, no other text: {\"defaults\": {...}, \"items\": [{...}, ...]}.",
        "Allowed keys: " + ", ".join(sorted(KEYS - NO_KEYS)) + ". Each item needs a prompt; defaults apply to every item.",
        "- prompt: a vivid description under 50 words (the models read about 75 tokens; the rest is lost).",
        "- negative: things to avoid, short.",
        f"- width and height: multiples of 8 between 256 and 2048. Good sizes: {SIZES}.",
        f"- count: 1 to 8 pictures per item. At most {MAX_ITEMS} items.",
        "- upscale: 2x, 4x, 1080p, 4k, or a long edge in px. For a 1080p or 4K request use 1344x768 with that upscale.",
        "- model: one id from the model list, only models with installed yes. Leave steps and guidance out unless asked.",
        *([f"  {fallback}"] if fallback else []),
        "- subjects: a list of up to 3 names from the trained subjects list, like \"subjects\": [\"Name\"], only with "
        "models whose family is sdxl. Use the subject's name in that item's prompt too.",
        "- name: a short label used in the file name (optional).",
        "- Never add file paths or reference images.",
        "",
        "Models (id | name | family | installed | defaults | notes):",
        *models,
        "",
        "Trained subjects: " + (", ".join(subs) if subs else "none"),
    ])


def extract_json(text: str) -> dict:
    """The batch from a reply: plain JSON, fenced JSON, or the JSON inside some prose."""
    t = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    cands = [t] + ([fence.group(1)] if fence else [])
    for a, b in (("{", "}"), ("[", "]")):
        i, j = t.find(a), t.rfind(b)
        if 0 <= i < j:
            cands.append(t[i:j + 1])
    for c in cands:
        try:
            d = json.loads(c)
        except ValueError:
            continue
        if isinstance(d, list):
            d = {"items": d}
        if isinstance(d, dict) and isinstance(d.get("items"), list):
            items = [{"prompt": x} if isinstance(x, str) else x for x in d["items"]]     # a bare prompt line
            return {"defaults": d.get("defaults") or {}, "items": items}
    raise ValueError(f"The AI did not send a batch. It said: {t[:200]}")


def check_assistant_batch(batch: dict, registry, subjects, fallback_model: str) -> list[str]:
    """Everything wrong with a batch, in plain words; empty when it can run."""
    defaults, items = batch.get("defaults"), batch.get("items")
    defaults, items = {} if defaults is None else defaults, [] if items is None else items
    problems, odd = [], False
    if not isinstance(defaults, dict):
        return ["defaults must be an object of shared settings"]
    if not isinstance(items, list):
        return ["items must be a list of lines"]
    if len(items) > MAX_ITEMS:
        problems.append(f"too many lines ({len(items)}); the limit is {MAX_ITEMS}")
    installed = {m.id for m in registry.all() if registry.local_path(m.id)}
    families = {m.id: m.family for m in registry.all()}
    for n, it in enumerate(items, 1):
        if not isinstance(it, dict):
            continue        # parse_batch says so
        v = {**defaults, **it}
        if NO_KEYS & set(v):
            problems.append(f"item {n}: the assistant may not add reference images")
        if not isinstance(v.get("prompt", ""), str):
            problems.append(f"item {n}: prompt must be text"); odd = True
        for k in ("width", "height", "count", "steps", "guidance", "seed"):
            x = v.get(k)
            if x is not None and (isinstance(x, bool) or not isinstance(x, (int, float)) or x != x or abs(x) > 1e9):
                problems.append(f"item {n}: {k} must be a number"); odd = True
        model = v.get("model") or fallback_model
        if not isinstance(model, str):
            problems.append(f"item {n}: model must be one model id"); odd = True; continue
        if model in families and model not in installed:
            problems.append(f"item {n}: model '{model}' is not downloaded")
        if v.get("subjects") and families.get(model) not in (None, "sdxl"):
            problems.append(f"item {n}: subjects need an SDXL model")
    if odd:
        return problems     # parse_batch would trip over the same odd values
    try:
        parse_batch(json.dumps({"defaults": defaults, "items": items}), "assistant.json", registry, fallback_model,
                    subjects=subjects)
    except BatchError as e:
        problems += [p for p in e.problems if p not in problems]
    except (TypeError, ValueError, OverflowError, AttributeError) as e:
        problems.append(f"the batch has a value the app cannot read ({e})")
    return problems


def _ask(provider_id, key, model, system, user, registry, subjects, fallback_model) -> Draft:
    """One request, plus one fix round if the batch breaks the rules or is not JSON."""
    p = PROVIDERS[provider_id]
    t, tokens = time.monotonic(), 0
    r = p.complete(key, model, system, user)
    tokens += r.tokens or 0
    try:
        batch = extract_json(r.text)
        problems = check_assistant_batch(batch, registry, subjects, fallback_model)
        sent = json.dumps(batch, indent=1)
    except ValueError as e:
        batch, problems, sent = None, [str(e)], r.text[:2000]
    if problems:
        r = p.complete(key, model, system, user + "\n\nYour batch:\n" + sent +
                       "\n\nFix these problems and send the whole batch again:\n- " + "\n- ".join(problems))
        tokens += r.tokens or 0
        try:
            batch = extract_json(r.text)
        except ValueError as e:
            raise ProviderError(str(e)) from None
        problems = check_assistant_batch(batch, registry, subjects, fallback_model)
    return Draft(batch, problems, provider_id, model, tokens, round(time.monotonic() - t, 1))


def write_batch(provider_id, key, model, request, registry, subjects, fallback_model) -> Draft:
    return _ask(provider_id, key, model, brief(registry, subjects, fallback_model), request, registry, subjects,
                fallback_model)


def revise_batch(provider_id, key, model, request, batch, instruction, registry, subjects, fallback_model) -> Draft:
    user = f"{request}\n\nCurrent batch:\n{json.dumps(batch, indent=1)}\n\nChange it like this: {instruction}"
    return _ask(provider_id, key, model, brief(registry, subjects, fallback_model), user, registry, subjects,
                fallback_model)
