"""File names: generated_<subject>_<YYYY-MM-DD>_<HHMM>[_<n>][_<scale>].<png|jpg|webp>"""
import re
from datetime import datetime
from pathlib import Path

STOP = set("""a an the of in on at with and or its it their this that to for from by into over under near
is are as be his her my our your some very""".split())
STYLE = set("""photo photograph image picture render ultra hyper detailed highly cinematic realistic photorealistic
4k 8k hd uhd hdr masterpiece best quality style art artwork illustration beautiful stunning epic natural
environment lighting shot closeup close portrait view background""".split())


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def subject_from_prompt(prompt: str, max_words: int = 3) -> str:
    keep = [w for w in _words(prompt) if w not in STOP and w not in STYLE]
    return "-".join(keep[:max_words]) or "image"


def slugify(name: str, max_words: int = 3) -> str:
    return "-".join(_words(name)[:max_words]) or "image"


def output_stem(subject: str, when: datetime, index: int | None = None, scale: str | None = None) -> str:
    stem = f"generated_{subject}_{when:%Y-%m-%d_%H%M}"
    if index is not None:
        stem += f"_{index}"
    if scale:
        stem += f"_{scale}"
    return stem


# What a picture can be saved as (Settings > Output): name -> (extension, PIL format)
FORMATS = {"png": (".png", "PNG"), "jpg": (".jpg", "JPEG"), "webp": (".webp", "WEBP")}
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")      # what the gallery shows and opens


def unique_path(folder: Path, stem: str, ext: str = ".png") -> Path:
    """The first free name: stem, then stem_<SS> (seconds now), then stem_<SS>_<k>. Checks the recipe and the
    other image types too: x.png and x.jpg would share the recipe x.json."""
    def free(s):
        return not any((folder / (s + e)).exists() for e in (*IMAGE_EXT, ".json"))
    if free(stem):
        return folder / (stem + ext)
    alt = f"{stem}_{datetime.now():%S}"
    k = 1
    cand = alt
    while not free(cand):
        k += 1
        cand = f"{alt}_{k}"
    return folder / (cand + ext)


def day_folder(root: Path, when: datetime) -> Path:
    d = Path(root) / f"{when:%Y-%m-%d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_image(img, f, fmt: str = "png", quality: int = 92) -> None:
    """Write a picture to an open file in one of FORMATS. JPG has no transparency, so it is flattened to RGB."""
    _, pil = FORMATS.get(fmt, FORMATS["png"])
    if pil == "JPEG":
        img.convert("RGB").save(f, format="JPEG", quality=quality, subsampling=0, optimize=True)
    elif pil == "WEBP":
        img.save(f, format="WEBP", quality=quality, method=4)
    else:
        img.save(f, format="PNG")
