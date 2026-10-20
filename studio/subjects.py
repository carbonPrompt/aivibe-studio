"""Named subjects (people, a car, things) trained from the user's photos. Stored only on this PC, in the data folder."""
import json
import random
import re
import shutil
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps

ID_RE = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")   # what create() makes: 'alex', 'the-truck', 'alex-2'

PHOTO_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MIN_SHORT_SIDE = 512
SIZE = 1024
_LETTERS = "bcdfghjkmnpqrtvwxz"     # no vowels: the made-up token should never be a real word


@dataclass
class Subject:
    id: str
    name: str
    kind: str               # person | car | thing
    class_word: str         # man | woman | person | car | object
    token: str
    base_model: str = "realvis-v5"
    steps: int = 0
    trained_at: str | None = None
    photo_count: int = 0
    version: int = 1
    default_weight: float = 1.0     # the likeness a pick starts at (set per subject; things like cars often like 1.1)
    root: Path | None = field(default=None, repr=False, compare=False)

    @property
    def trained(self) -> bool:
        return self.root is not None and (self.root / "lora.safetensors").exists()

    @property
    def phrase(self) -> str:
        return f"{self.token} {self.class_word}"


class SubjectStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, id) -> Path:
        """A subject's folder. Ids come from recipes and batch files other people may share, so only the ids
        create() makes are accepted: never a path, drive or network share."""
        if not isinstance(id, str) or not ID_RE.fullmatch(id):
            raise KeyError(id)
        return self.root / id

    def _load(self, d: Path) -> Subject:
        data = json.loads((d / "subject.json").read_text(encoding="utf-8"))
        data["id"] = d.name                 # the folder is the identity, not what the file says
        return Subject(**{k: v for k, v in data.items() if k in Subject.__dataclass_fields__ and k != "root"}, root=d)

    def _save(self, s: Subject):
        d = self.root / s.id
        d.mkdir(parents=True, exist_ok=True)
        data = asdict(s); data.pop("root")
        tmp = d / "subject.json.tmp"
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(d / "subject.json")

    def all(self) -> list[Subject]:
        return [self._load(d) for d in sorted(self.root.iterdir()) if (d / "subject.json").exists()]

    def get(self, id: str) -> Subject:
        d = self._dir(id)
        if not (d / "subject.json").exists():
            raise KeyError(id)
        return self._load(d)

    def create(self, name: str, kind: str, class_word: str) -> Subject:
        base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "subject"
        id, k = base, 1
        while (self.root / id).exists():
            k += 1
            id = f"{base}-{k}"
        taken = {s.token for s in self.all()}
        rng = random.SystemRandom()
        token = "zq" + "".join(rng.choice(_LETTERS) for _ in range(3))
        while token in taken:
            token = "zq" + "".join(rng.choice(_LETTERS) for _ in range(3))
        s = Subject(id=id, name=name.strip() or id, kind=kind, class_word=class_word, token=token, root=self.root / id)
        self._save(s)
        (self.root / id / "photos").mkdir(exist_ok=True)
        return s

    def rename(self, id: str, name: str) -> None:
        s = self.get(id)
        s.name = name.strip() or s.name
        self._save(s)

    def set_default_weight(self, id: str, weight: float) -> None:
        s = self.get(id)
        s.default_weight = round(float(weight), 2)
        self._save(s)

    def update(self, s: Subject) -> None:
        self._save(s)

    def delete(self, id: str) -> None:
        shutil.rmtree(self._dir(id))

    def photo_paths(self, id: str) -> list[Path]:
        return sorted((self._dir(id) / "photos").glob("*.png"))

    def remove_photo(self, id: str, name: str) -> None:
        """Delete one prepared training photo (the original in the user's folder is never touched)."""
        folder = (self._dir(id) / "photos").resolve()
        target = (folder / name).resolve()
        if target.parent != folder or target.suffix.lower() != ".png" or not target.exists():
            raise ValueError(f"not a training photo of this subject: {name!r}")
        target.unlink()
        s = self.get(id)
        s.photo_count = len(self.photo_paths(id))
        self._save(s)

    def lora_path(self, id: str) -> Path:
        return self._dir(id) / "lora.safetensors"

    def import_photos(self, id: str, folder: Path, face_finder=None) -> dict:
        s = self.get(id)
        dest = self._dir(id) / "photos"
        dest.mkdir(exist_ok=True)
        added, skipped = 0, []
        for f in sorted(Path(folder).iterdir()):
            if not f.is_file():
                continue
            if f.suffix.lower() not in PHOTO_EXT:
                skipped.append((f.name, "not a photo"))
                continue
            out = dest / (re.sub(r"[^A-Za-z0-9_-]+", "_", f.stem) + ".png")
            try:
                reason = prep_photo(f, out, s.kind, face_finder=face_finder)
            except Exception as e:      # one odd file must not stop the whole import
                reason = f"could not process ({e.__class__.__name__}: {e})"
            if reason:
                skipped.append((f.name, reason))
            else:
                added += 1
        s.photo_count = len(self.photo_paths(id))
        self._save(s)
        return {"added": added, "skipped": skipped, "total": s.photo_count}


def find_face(img: Image.Image):
    """Largest frontal face as (x, y, w, h), or None. OpenCV's bundled Haar cascade."""
    import numpy as np
    from .facefix import cpu_cv2
    cv2 = cpu_cv2()
    gray = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    scale = 800 / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1 else gray
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(small, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    k = 1 / scale if scale < 1 else 1
    return int(x * k), int(y * k), int(w * k), int(h * k)


_ORIENT = {2: [Image.FLIP_LEFT_RIGHT], 3: [Image.ROTATE_180], 4: [Image.FLIP_TOP_BOTTOM],
           5: [Image.FLIP_LEFT_RIGHT, Image.ROTATE_90], 6: [Image.ROTATE_270], 7: [Image.FLIP_LEFT_RIGHT, Image.ROTATE_270],
           8: [Image.ROTATE_90]}


def _upright(im: Image.Image) -> Image.Image:
    """Apply the EXIF orientation. Pillow's exif_transpose chokes on some phones' EXIF; then rotate by hand."""
    try:
        return ImageOps.exif_transpose(im)
    except Exception:
        out = im
        for op in _ORIENT.get(im.getexif().get(0x0112, 1), []):
            out = out.transpose(op)
        return out


def prep_photo(src: Path, dest: Path, kind: str, face_finder=None) -> str | None:
    """Square 1024 px training copy. People: the square is centred on the face (upper third if none found).
    Returns None, or the reason the photo was skipped."""
    try:
        with Image.open(src) as im:
            img = _upright(im).convert("RGB")
    except (OSError, ValueError) as e:
        return f"unreadable ({e.__class__.__name__})"
    w, h = img.size
    if min(w, h) < MIN_SHORT_SIDE:
        return f"too small ({w}x{h})"
    side = min(w, h)
    cx, cy = w / 2, h / 2
    if kind == "person":
        face = (face_finder or find_face)(img)
        if face:
            fx, fy, fw, fh = face
            cx, cy = fx + fw / 2, fy + fh / 2
        else:
            cy = h / 3
    left = int(min(max(cx - side / 2, 0), w - side))
    top = int(min(max(cy - side / 2, 0), h - side))
    crop = img.crop((left, top, left + side, top + side))
    if side > SIZE:
        crop = crop.resize((SIZE, SIZE), Image.LANCZOS)
    elif side < SIZE:
        crop = crop.resize((SIZE, SIZE), Image.LANCZOS)   # training wants 1024; mild upscale for 512-1023 sources
    dest.parent.mkdir(parents=True, exist_ok=True)
    crop.save(dest)
    return None


def rewrite_prompt(prompt: str, subjects: list[Subject]) -> str:
    """Replace each subject's name (whole word, any case) with its trigger phrase; prepend subjects not named."""
    out, missing = prompt, []
    for s in subjects:
        pat = re.compile(rf"\b{re.escape(s.name)}\b", re.IGNORECASE)
        if pat.search(out):
            out = pat.sub(s.phrase if s.name.lower() != s.class_word.lower() else s.token + " " + s.class_word, out)
        else:
            missing.append(s.phrase)
    return ", ".join(missing + [out]) if missing else out
