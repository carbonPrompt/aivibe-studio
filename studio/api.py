"""What the window can call (window.pywebview.api.*). Every method returns plain JSON-able data, never raises."""
import base64
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from PIL import Image

from studio import VERSION
from studio import assistant, diag, hardware, update
from studio.batch import DEFAULT_LIKENESS, BatchError, parse_batch, safe_reference
from studio.hfget import FetchError, download_patterns  # noqa: F401 (download_patterns re-exported)
from studio.engine import GenRequest, LIKENESS_FAMILIES, MAX_SUBJECTS
from studio.naming import IMAGE_EXT, slugify, subject_from_prompt
from studio.recipe import read_recipe
from studio.settings import save_settings
from studio.providers import PROVIDERS, ProviderError
from studio.upscale import parse_choice

log = logging.getLogger("studio.api")

NO_WINDOW = 0x08000000  # CREATE_NO_WINDOW: pythonw would otherwise flash a console for every nvidia-smi
DEFAULT_STYLES = [
    {"name": "Cinematic", "text": "cinematic lighting, dramatic composition, film still"},
    {"name": "Product", "text": "studio product photo, clean background, soft box lighting"},
    {"name": "Illustration", "text": "detailed digital illustration, vibrant colours"},
]
DEFAULT_NEGATIVES = [
    {"name": "Basic", "text": "low quality, blurry, distorted"},
    {"name": "Photo", "text": "text, watermark, signature, cartoon, illustration, plastic skin, blurry, deformed hands"},
]


def backed_up_roots() -> list[Path]:
    h = Path.home()
    return [h / "Pictures", h / "Documents", h / "Downloads"]


def _under(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _unexpected(e: Exception) -> str:
    return (f"Something went wrong ({type(e).__name__}: {e}). Nothing was started. It is in the log, and "
            "⚑ Report can send it.")


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


_DETECT = object()
APP_ROOT = Path(__file__).resolve().parent.parent     # the folder holding the studio package
LICENSE_TEXT = {"commercial": "Free to use, including commercially.",
                "conditions": "Free for personal use; commercial use has conditions. Read the model card.",
                "nc": "Non-commercial use only."}
TOKEN_HELP = ("This model needs a free Hugging Face account. 1. Make an account at huggingface.co. 2. Open the model "
              "page and accept its terms. 3. Make a Read token (Settings, Access Tokens) and paste it in Settings, "
              "Hugging Face token.")
TOP_LEVEL = ("output_root", "idle_minutes", "theme", "extra_model_folders", "gpu_marker_file", "gpu_jobs_folder",
             "hf_cache_dir", "update_check")
RESTART_KEYS = {"extra_model_folders", "gpu_marker_file", "gpu_jobs_folder", "hf_cache_dir"}   # read at start


# set_assistant_key is quiet so a pasted API key never reaches studio.log
@diag.logged_api(quiet={"events", "state", "gpu", "gallery", "storage", "log_client", "set_assistant_key",
                        "set_hf_token"})
class Api:
    def __init__(self, queue, registry, settings, settings_path, engine, window=None, subjects=None, keys=None,
                 gpu=_DETECT, python_exe=None):
        # underscores: pywebview exposes every public attribute of js_api to the page
        self._queue, self._registry, self._settings, self._engine = queue, registry, settings, engine
        self._settings_path = Path(settings_path)
        self._window = window
        self._subjects = subjects
        self._keys = keys       # KeyStore: the assistant's API keys (data\assistant_keys.json, not settings)
        self._thumbs = {}
        self._downloads = set()
        self._python = python_exe or sys.executable     # runs the download child (studio.hfget)
        self._fetch = self._fetch_child
        self._cancels = {}          # model id -> Event while its download runs
        self._downloads_dir = Path.home() / "Downloads"    # where a new version's installer is saved
        self._gpu_info = gpu        # hardware.GpuInfo, None (no card) or _DETECT (look on first use)
        self._apply_device(settings.last.get("device", "Auto"))

    # ---- settings ----
    def get_settings(self) -> dict:
        s = asdict(self._settings)
        s["last"].setdefault("styles", DEFAULT_STYLES)
        s["last"].setdefault("negatives", DEFAULT_NEGATIVES)
        s["backed_up"] = self._backed_up(self._settings.output_root)
        s["version"] = VERSION
        g = self._gpu()
        s["hardware"] = {"support": hardware.support(g), "name": g.name if g else "", "vram_gb": g.vram_gb if g else 0,
                         "driver": g.driver if g else "", "train_ok": bool(g) and g.vram_gb >= hardware.TRAIN_MIN_VRAM_GB}
        return s

    def save_settings(self, d: dict) -> dict:
        restart = False
        for k in TOP_LEVEL:
            if k not in d:
                continue
            v = d[k]
            if k == "extra_model_folders":
                v = [str(x).strip() for x in (v or []) if str(x).strip()]
            elif k in ("gpu_marker_file", "gpu_jobs_folder", "hf_cache_dir"):
                v = str(v or "").strip()
            elif k == "update_check":
                v = bool(v)
            restart |= k in RESTART_KEYS and v != getattr(self._settings, k)
            setattr(self._settings, k, v)
        for k, v in d.items():
            if k not in TOP_LEVEL:
                self._settings.last[k] = v
        if "device" in d:
            self._apply_device(d["device"])
        if "idle_minutes" in d and hasattr(self._engine, "idle_minutes"):
            self._engine.idle_minutes = int(d["idle_minutes"])
        save_settings(self._settings_path, self._settings)
        return {"ok": True, "backed_up": self._backed_up(self._settings.output_root), "restart": restart}

    def _apply_device(self, choice: str):
        if hasattr(self._engine, "set_device"):
            self._engine.set_device("cpu" if str(choice).upper() == "CPU" else "cuda")

    @staticmethod
    def _backed_up(folder) -> bool:
        return any(_under(Path(folder), r) for r in backed_up_roots())

    # ---- models ----
    def models(self) -> list[dict]:
        out = []
        for m in self._registry.all():
            out.append({"id": m.id, "name": m.name, "family": m.family, "license": m.license,
                        "description": m.description, "defaults": m.defaults, "size_gb": m.size_gb,
                        "downloaded": self._registry.is_downloaded(m.id), "user": m.user,
                        "guide": f"https://huggingface.co/{m.source['hf']}" if "hf" in m.source else None,
                        "min_vram_gb": m.min_vram_gb, "fit": hardware.fits(m.min_vram_gb, self._gpu())})
        return out

    def _gpu(self):
        if self._gpu_info is _DETECT:
            self._gpu_info = hardware.detect()
        return self._gpu_info

    def model_guide(self, model_id: str) -> None:
        m = self._registry.get(model_id)
        if "hf" in m.source:
            webbrowser.open(f"https://huggingface.co/{m.source['hf']}")

    def add_model(self, kind: str, value: str, name: str, family: str, license: str) -> dict:
        try:
            m = (self._registry.add_hf(value.strip(), name, family, license) if kind == "hf"
                 else self._registry.add_file(Path(value.strip().strip('"')), name, family, license))
            return {"id": m.id}
        except (ValueError, KeyError) as e:
            return {"error": str(e)}

    def remove_model(self, model_id: str) -> dict:
        try:
            self._registry.remove_user(model_id)
            return {"ok": True}
        except (ValueError, KeyError) as e:
            return {"error": str(e)}

    def _needs_download(self, model_id) -> dict | None:
        m = self._registry.get(model_id)
        if self._registry.is_downloaded(model_id) or "hf" not in m.source:
            return None
        return {"id": m.id, "name": m.name, "size_gb": m.size_gb}

    def download_info(self, model_id: str) -> dict:
        """What the download dialog shows before anything is fetched: size, license, space, token, fit."""
        m = self._registry.get(model_id)
        free = self._free_gb()
        return {"id": m.id, "name": m.name, "size_gb": m.size_gb, "license": m.license,
                "license_text": LICENSE_TEXT.get(m.license, ""),
                "card_url": f"https://huggingface.co/{m.source['hf']}" if "hf" in m.source else None,
                "gated": m.gated, "has_token": bool(self._hf_token()), "free_gb": free,
                "enough_space": free >= m.size_gb * 1.1, "fit": hardware.fits(m.min_vram_gb, self._gpu())}

    def download(self, model_id: str) -> dict:
        """Start downloading a model into the model folder; progress arrives as 'download' events."""
        m = self._registry.get(model_id)
        if "hf" not in m.source or model_id in self._downloads:
            return {"error": "nothing to download" if "hf" not in m.source else "already downloading"}
        free = self._free_gb()
        if free < m.size_gb * 1.1:
            return {"error": "no_space", "free_gb": free, "size_gb": m.size_gb,
                    "message": f"{m.name} needs about {m.size_gb} GB and the model folder's drive has {free} GB free."}
        if m.gated and not self._hf_token():
            return {"error": "needs_token", "message": TOKEN_HELP}
        self._downloads.add(model_id)
        threading.Thread(target=self._download, args=(m,), daemon=True).start()
        return {"ok": True}

    def _hf_token(self) -> str | None:
        return self._keys.get("huggingface") if self._keys is not None else None

    def _free_gb(self) -> float:
        d = self._registry.hf_home
        while not d.exists() and d.parent != d:     # the model folder may not exist yet: its drive counts
            d = d.parent
        return round(shutil.disk_usage(d)[2] / 1e9, 1)

    def cancel_download(self, model_id: str) -> dict:
        ev = self._cancels.get(model_id)
        if ev is None:
            return {"error": "not downloading"}
        ev.set()
        return {"ok": True}

    def _fetch_child(self, repo: str, cache_dir: str, token, cancel, revision=None) -> None:
        """Download in a child process (python -m studio.hfget) so Cancel stops it at once; the token goes in
        its environment, never on the command line."""
        from studio.trainer import NO_WINDOW, _kill_tree
        env = dict(os.environ, HF_HUB_DISABLE_TELEMETRY="1")
        env.pop("HF_TOKEN", None)
        if token:
            env["HF_TOKEN"] = token
        proc = subprocess.Popen([self._python, "-m", "studio.hfget", repo, cache_dir] + ([revision] if revision else []), cwd=str(APP_ROOT), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                creationflags=NO_WINDOW if os.name == "nt" else 0)
        stop = threading.Event()

        def watch():
            while not stop.wait(0.5):
                if cancel.is_set():
                    _kill_tree(proc)
                    return

        threading.Thread(target=watch, daemon=True).start()
        try:
            out = proc.communicate()[0].decode("utf-8", "replace").strip().splitlines()
        finally:
            stop.set()
        if cancel.is_set():
            raise FetchError("cancelled", None)
        try:
            result = json.loads(out[-1]) if out else {}
        except ValueError:
            result = {}
        if proc.returncode != 0 or not result.get("ok"):
            raise FetchError(result.get("error") or f"the download stopped (exit {proc.returncode})",
                             result.get("status"))

    def _download(self, m):
        repo = m.source["hf"]
        cache = self._registry.hf_repo_dir(m.id)
        existed = cache.exists()        # a shared model folder may already hold this repo (another program's)
        token = self._hf_token() if m.gated else None   # a stale token makes even public repos answer 401
        cancel = self._cancels[m.id] = threading.Event()
        done = threading.Event()

        def watch():
            while not done.wait(3):
                size = sum(f.stat().st_size for f in cache.rglob("*") if f.is_file()) if cache.exists() else 0
                pct = min(99, int(size / (m.size_gb * 1e9) * 100)) if m.size_gb else None
                self._queue.note("download", model=m.id, name=m.name, percent=pct, gb=round(size / 1e9, 1))

        threading.Thread(target=watch, daemon=True).start()
        marker = cache / ".studio-downloading"   # until removed, the registry treats the model as not downloaded
        try:
            cache.mkdir(parents=True, exist_ok=True)
            marker.write_text(str(time.time()))
            self._fetch(repo, str(self._registry.hf_home / "hub"), token, cancel, m.source.get("revision"))
            marker.unlink(missing_ok=True)
            self._queue.note("download_done", model=m.id, name=m.name)
        except FetchError as e:
            self._drop_partial(cache, existed)
            if cancel.is_set():
                self._queue.note("download_cancelled", model=m.id, name=m.name)
                return
            if e.status == 401:
                kind, msg = ("bad_token", "Hugging Face refused your token; check or replace it in Settings, "
                             "Hugging Face.") if token else ("needs_token", TOKEN_HELP)
            elif e.status == 403 or "gated" in str(e).lower():
                kind, msg = "needs_terms", ("Hugging Face refused the download. Open the model page while signed "
                                            "in, accept its terms, then try again.")
            else:
                kind, msg = "failed", str(e)
            self._queue.note("download_failed", model=m.id, name=m.name, code=kind, error=msg)
        except OSError as e:
            self._drop_partial(cache, existed)
            self._queue.note("download_failed", model=m.id, name=m.name, code="failed", error=str(e))
        finally:
            done.set()
            self._cancels.pop(m.id, None)
            self._downloads.discard(m.id)

    @staticmethod
    def _drop_partial(cache: Path, existed: bool) -> None:
        """A folder this attempt created goes entirely; one that was already there (maybe another program's
        model in a shared cache) loses only the unfinished pieces and the marker."""
        if not existed:
            shutil.rmtree(cache, ignore_errors=True)
            return
        for f in cache.rglob("*.incomplete"):
            f.unlink(missing_ok=True)
        (cache / ".studio-downloading").unlink(missing_ok=True)

    def storage(self) -> list[dict]:
        out = []
        for m in self._registry.all():
            p = self._registry.local_path(m.id)
            if p is None:
                continue
            where = ("app" if any(_under(p, d) for d in self._registry.extra_model_dirs)
                     else "cache" if _under(p, self._registry.hf_home) else "file")
            root = p.parent.parent if where == "cache" else p
            size = (p.stat().st_size if p.is_file()
                    else sum(f.stat().st_size for f in root.rglob("*") if f.is_file()))
            out.append({"id": m.id, "name": m.name, "where": where, "path": str(root),
                        "size_gb": round(size / 1e9, 1), "deletable": where == "cache"})
        return out

    def delete_model(self, model_id: str) -> dict:
        m = self._registry.get(model_id)
        p = self._registry.local_path(model_id)
        if p is None:
            return {"error": "not downloaded"}
        if "hf" not in m.source or not _under(p, self._registry.hf_home / "hub"):
            return {"error": "only models downloaded by aiVibe Studio can be deleted here (this one belongs to "
                             "another app or is your own file)"}
        target = self._registry.hf_repo_dir(model_id)
        if not _under(p, target):
            return {"error": "unexpected model location; not deleting"}
        if self._engine.loaded_model() == model_id:
            return {"error": "this model is loaded; wait until the GPU is released"}
        shutil.rmtree(target)
        return {"ok": True}

    # ---- generating ----
    def generate(self, p: dict) -> dict:
        try:
            model_id = p["model"]
            self._registry.get(model_id)
            prompt = str(p.get("prompt") or "").strip()
            if not prompt:
                return {"error": "write a prompt first"}
            w, h, steps, count = int(p["width"]), int(p["height"]), int(p["steps"]), int(p["count"])
            guidance = float(p["guidance"])
            seed = int(p.get("seed", -1))
        except KeyError as e:
            return {"error": f"unknown model or missing setting: {e}"}
        except (TypeError, ValueError) as e:
            return {"error": f"bad number: {e}"}
        for label, x in (("width", w), ("height", h)):
            if x % 8 or not 256 <= x <= 2048:
                return {"error": f"{label} must be a multiple of 8 between 256 and 2048"}
        if not 1 <= count <= 8 or not 1 <= steps <= 150:
            return {"error": "images must be 1 to 8 and steps 1 to 150"}
        subs, err = self._check_subjects(p.get("subjects") or [], model_id)
        if err:
            return {"error": err}
        need = self._needs_download(model_id)
        if need:
            return {"needs_download": need}
        ref = str(p.get("reference") or "").strip() or None
        if ref:
            problem = safe_reference(ref)
            if problem:
                return {"error": problem}
            if self._registry.get(model_id).family not in LIKENESS_FAMILIES:
                return {"error": "reference images work with SDXL models only (RealVisXL, Juggernaut)"}
        name = str(p.get("name") or "").strip()
        subject = slugify(name) if name else subject_from_prompt(prompt)
        req = GenRequest(model_id, prompt, str(p.get("negative") or ""), w, h, steps, guidance,
                         None if seed < 0 else seed, count, subs, ref, float(p.get("reference_strength", 0.5)),
                         bool(p.get("face_fix", True)))
        return {"job": self._queue.submit_single(req, subject)}

    def _check_subjects(self, picks, model_id):
        """[(id, weight)] from [{id, weight}], or an error message."""
        if not picks:
            return [], None
        if self._subjects is None:
            return [], "subjects are not available"
        if self._registry.get(model_id).family not in LIKENESS_FAMILIES:
            return [], "subjects work with SDXL models only (RealVisXL, Juggernaut)"
        if len(picks) > MAX_SUBJECTS:
            return [], f"at most {MAX_SUBJECTS} subjects in one picture"
        out = []
        for pk in picks:
            try:
                s = self._subjects.get(pk["id"])
            except KeyError:
                return [], f"subject {pk.get('id')!r} no longer exists"
            if not s.trained:
                return [], f"{s.name} is not trained yet"
            out.append((s.id, float(pk.get("weight", DEFAULT_LIKENESS))))
        return out, None

    # ---- subjects ----
    def subjects(self) -> list[dict]:
        if self._subjects is None:
            return []
        return [{"id": s.id, "name": s.name, "kind": s.kind, "class_word": s.class_word, "trained": s.trained,
                 "photos": len(self._subjects.photo_paths(s.id)), "trained_at": s.trained_at,
                 "default_weight": s.default_weight}
                for s in self._subjects.all()]

    def new_subject(self, name: str, kind: str, class_word: str) -> dict:
        if not str(name).strip():
            return {"error": "give the subject a name"}
        if kind not in ("person", "pet", "car", "thing"):
            return {"error": f"unknown kind {kind!r}"}
        s = self._subjects.create(name, kind, class_word)
        return {"id": s.id, "name": s.name}

    def import_photos(self, subject_id: str, folder: str | None = None) -> dict:
        if folder is None:
            if self._window is None:
                return {"error": "no window"}
            import webview
            inbox = self._subjects.root / "inbox"
            picked = self._window.create_file_dialog(webview.FileDialog.FOLDER, directory=str(inbox))
            if not picked:
                return {"cancelled": True}
            folder = picked[0] if isinstance(picked, (list, tuple)) else picked
        try:
            r = self._subjects.import_photos(subject_id, Path(folder))
        except (KeyError, OSError) as e:
            return {"error": str(e)}
        r["skipped"] = [f"{f}: {why}" for f, why in r["skipped"]]
        r["warning"] = (f"{r['total']} photos: training works from 5, but 12-30 gives a much better likeness"
                        if r["total"] < 12 else None)
        return r

    def train(self, subject_id: str) -> dict:
        from studio.trainer import MIN_PHOTOS, steps_for
        try:
            s = self._subjects.get(subject_id)
        except KeyError:
            return {"error": "no such subject"}
        n = len(self._subjects.photo_paths(subject_id))
        if n < MIN_PHOTOS:
            return {"error": f"{s.name} has {n} photos; add at least {MIN_PHOTOS}"}
        if not self._registry.is_downloaded("realvis-v5"):        # training's base model
            need = self._needs_download("realvis-v5")
            if need:
                return {"needs_download": need}
        steps = steps_for(n)
        return {"job": self._queue.submit_train(subject_id), "steps": steps, "minutes": round(steps * 2.8 / 60)}

    def set_default_weight(self, subject_id: str, weight) -> dict:
        try:
            w = float(weight)
        except (TypeError, ValueError):
            return {"error": "likeness must be a number"}
        if not 0 <= w <= 1.3:
            return {"error": "likeness must be between 0 and 1.3"}
        try:
            self._subjects.set_default_weight(subject_id, w)
            return {"ok": True}
        except KeyError:
            return {"error": "no such subject"}

    def subject_photos(self, subject_id: str) -> list[dict]:
        try:
            return [{"name": p.name, "thumb": self._thumb(p)} for p in self._subjects.photo_paths(subject_id)]
        except (KeyError, OSError):
            return []

    def remove_subject_photo(self, subject_id: str, name: str) -> dict:
        try:
            self._subjects.remove_photo(subject_id, name)
            return {"ok": True, "total": len(self._subjects.photo_paths(subject_id))}
        except (KeyError, ValueError) as e:
            return {"error": str(e)}

    def rename_subject(self, subject_id: str, name: str) -> dict:
        try:
            self._subjects.rename(subject_id, name)
            return {"ok": True}
        except KeyError:
            return {"error": "no such subject"}

    def delete_subject(self, subject_id: str) -> dict:
        try:
            self._subjects.delete(subject_id)
            return {"ok": True}
        except (KeyError, OSError) as e:
            return {"error": str(e)}

    def pick_reference(self) -> dict:
        if self._window is None:
            return {"error": "no window"}
        import webview
        picked = self._window.create_file_dialog(webview.FileDialog.OPEN, allow_multiple=False,
                                                 file_types=("Images (*.png;*.jpg;*.jpeg;*.webp)", "All files (*.*)"))
        if not picked:
            return {"cancelled": True}
        p = Path(picked[0])
        try:
            return {"path": str(p), "thumb": self._thumb(p), "name": p.name}
        except Exception as e:
            return {"error": f"cannot open {p.name}: {e}"}

    def load_batch(self, path: str | None = None) -> dict:
        if path is None:
            if self._window is None:
                return {"error": "no window"}
            import webview
            picked = self._window.create_file_dialog(webview.FileDialog.OPEN, allow_multiple=False,
                                                    file_types=("Batch files (*.json;*.txt)", "All files (*.*)"))
            if not picked:
                return {"cancelled": True}
            path = picked[0]
        try:
            text = Path(path).read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as e:
            return {"problems": [f"could not read the file: {e}"]}
        try:
            items = parse_batch(text, Path(path).name, self._registry, self._settings.last.get("model", "realvis-v5"),
                                subjects=self._subjects)
        except BatchError as e:
            return {"problems": e.problems}
        need = [n for n in {self._needs_download(i.model_id) and i.model_id for i in items} if n]
        if need:
            return {"needs_download": [self._needs_download(m) for m in sorted(need)]}
        return {"job": self._queue.submit_batch(items, Path(path).name), "count": len(items)}

    def cancel(self) -> None:
        self._queue.cancel()

    def events(self, since: int) -> list[dict]:
        return self._queue.events(since)

    def state(self) -> dict:
        return self._queue.state()

    # ---- images ----
    def _root(self) -> Path:
        return Path(self._settings.output_root)

    def _recent_folder(self) -> Path | None:
        """The one folder the Recent strip shows (Settings), or None for the latest days of the output folder."""
        f = str(self._settings.last.get("recent_folder") or "").strip()
        return Path(f) if f else None

    def gallery(self, day: str | None = None, limit: int | None = None) -> list[dict]:
        limit = int(limit or self._settings.last.get("recent_limit") or 60)
        chosen = None if day else self._recent_folder()
        if chosen:      # that folder only, not its subfolders
            pngs = [p for p in chosen.iterdir() if p.suffix.lower() in IMAGE_EXT] if chosen.is_dir() else []
        else:
            root = self._root()
            if not root.exists():
                return []
            folders = [root / day] if day else sorted((d for d in root.iterdir() if d.is_dir()), reverse=True)[:7]
            pngs = [p for d in folders if d.is_dir() for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXT]
        pngs = sorted(pngs, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
        out = []
        for p in pngs:
            try:
                thumb = self._thumb(p)
            except Exception:       # one unreadable or oversized file must not take the whole gallery down
                continue
            item = {"path": str(p), "name": p.name, "thumb": thumb, "upscaled": None, "source": None}
            try:
                r = read_recipe(p)
                item.update(seed=r.seed, model=r.model_name, prompt=r.prompt, width=r.width, height=r.height,
                            upscaled=r.extra.get("upscale"), source=r.extra.get("upscaled_from"))
            except (OSError, ValueError, TypeError):
                pass
            out.append(item)
        return out

    def _thumb(self, p: Path) -> str:
        key = (str(p), p.stat().st_mtime)
        if key not in self._thumbs:
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((256, 256))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=82)
            self._thumbs[key] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        return self._thumbs[key]

    def _checked(self, path: str) -> Path | None:
        p = Path(path)
        if p.suffix.lower() not in IMAGE_EXT or not p.is_file():
            return None
        chosen = self._recent_folder()
        return p if _under(p, self._root()) or (chosen and p.parent.resolve() == chosen.resolve()) else None

    def image(self, path: str) -> dict:
        p = self._checked(path)
        if p is None:
            return {"error": "not an image in the output folder"}
        mime = {".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp"}.get(p.suffix.lower(), "png")
        data = f"data:image/{mime};base64," + base64.b64encode(p.read_bytes()).decode()
        try:
            recipe = asdict(read_recipe(p))
        except (OSError, ValueError, TypeError):
            recipe = None
        return {"data": data, "recipe": recipe, "name": p.name}

    def remake(self, path: str, new_seed: bool) -> dict:
        p = self._checked(path)
        if p is None:
            return {"error": "not an image in the output folder"}
        try:
            r = read_recipe(p)
        except (OSError, ValueError, TypeError):
            return {"error": "this image has no recipe"}
        if not r.model_id:
            return {"error": "this image has no recipe to re-make from"}
        need = self._needs_download(r.model_id) if r.model_id in {m.id for m in self._registry.all()} else None
        if need:
            return {"needs_download": need}
        subs = []
        for x in r.extra.get("subjects") or []:
            if self._subjects is None:
                return {"error": "subjects are not available"}
            try:
                self._subjects.get(x["id"])
            except KeyError:
                return {"error": f"this picture used the subject {x.get('name', x['id'])!r}, which no longer exists"}
            subs.append((x["id"], float(x.get("weight", DEFAULT_LIKENESS))))
        req = GenRequest(r.model_id, r.prompt, r.negative, r.width, r.height, r.steps, r.guidance,
                         None if new_seed else r.seed, 1, subs)
        return {"job": self._queue.submit_single(req, r.subject)}

    def upscale(self, paths: list, choice) -> dict:
        try:
            c = parse_choice(choice)
        except ValueError as e:
            return {"error": str(e)}
        if not paths:
            return {"error": "pick at least one image"}
        checked = [self._checked(p) for p in paths]
        if any(p is None for p in checked):
            return {"error": "only images in the output folder can be upscaled"}
        return {"job": self._queue.submit_upscale([str(p) for p in checked], c), "count": len(checked)}

    def fix_face(self, path: str, x: int, y: int, subject_id: str) -> dict:
        """Click-to-fix: redraw the spot at (x, y) (image pixels) of a saved picture as one subject."""
        p = self._checked(path)
        if p is None:
            return {"error": "not an image in the output folder"}
        try:
            s = self._subjects.get(subject_id)
        except (KeyError, AttributeError):
            return {"error": "no such subject"}
        if not s.trained:
            return {"error": f"{s.name} is not trained yet"}
        return {"job": self._queue.submit_fix(str(p), int(x), int(y), s.id)}

    def import_and_upscale(self, path: str | None, choice) -> dict:
        """Upscale a picture from anywhere on this PC: a PNG copy goes into <output>/<today>/imported first, so
        the original is untouched and the result is backed up like everything else."""
        if path is None:
            if self._window is None:
                return {"error": "no window"}
            import webview
            picked = self._window.create_file_dialog(webview.FileDialog.OPEN, allow_multiple=False,
                                                     file_types=("Images (*.png;*.jpg;*.jpeg;*.webp;*.bmp)", "All files (*.*)"))
            if not picked:
                return {"cancelled": True}
            path = picked[0]
        path = str(path).strip()
        problem = safe_reference(path)
        if problem:
            return {"error": problem.replace("reference", "image")}
        try:
            c = parse_choice(choice)
            folder = self._root() / f"{datetime.now():%Y-%m-%d}" / "imported"
            folder.mkdir(parents=True, exist_ok=True)
            from studio.naming import unique_path
            with Image.open(path) as im:
                img = im.convert("RGBA" if "A" in im.getbands() else "RGB")
            dest = unique_path(folder, slugify(Path(path).stem, max_words=6))
            img.save(dest)
        except ValueError as e:
            return {"error": str(e)}
        except OSError as e:
            return {"error": f"could not import {Path(path).name}: {e}"}
        return {"job": self._queue.submit_upscale([str(dest)], c), "count": 1, "imported": str(dest)}

    def copy_prompt(self, path: str) -> str:
        p = self._checked(path)
        try:
            return read_recipe(p).prompt if p else ""
        except (OSError, ValueError, TypeError):
            return ""

    def open_output(self, path: str | None = None) -> None:
        target = Path(path).parent if path and self._checked(path) else self._root() / f"{datetime.now():%Y-%m-%d}"
        if not target.exists():
            target = self._root()
        target.mkdir(parents=True, exist_ok=True)
        if hasattr(os, "startfile"):
            os.startfile(str(target))

    def choose_folder(self) -> dict:
        if self._window is None:
            return {"error": "no window"}
        import webview
        picked = self._window.create_file_dialog(webview.FileDialog.FOLDER, directory=self._settings.output_root)
        if not picked:
            return {"cancelled": True}
        return self.save_settings({"output_root": picked[0] if isinstance(picked, (list, tuple)) else picked})

    def choose_recent_folder(self) -> dict:
        """Pick the one folder the Recent strip shows."""
        if self._window is None:
            return {"error": "no window"}
        import webview
        start = self._recent_folder() or self._root()
        picked = self._window.create_file_dialog(webview.FileDialog.FOLDER, directory=str(start))
        if not picked:
            return {"cancelled": True}
        return self.save_settings({"recent_folder": picked[0] if isinstance(picked, (list, tuple)) else picked})

    # ---- assistant (phase 4): a cloud AI writes a batch, the user reviews it, Go runs it ----
    def _assistant(self) -> tuple[str, str]:
        pid = self._settings.last.get("assistant_provider", "gemini")
        pid = pid if pid in PROVIDERS else "gemini"
        return pid, self._settings.last.get(f"assistant_model_{pid}") or PROVIDERS[pid].default_model

    def _key_or_error(self, pid: str) -> tuple[str | None, dict | None]:
        key = self._keys.get(pid) if self._keys else None
        if not key:
            return None, {"error": f"Add your {PROVIDERS[pid].label} API key in Settings > Assistant first."}
        return key, None

    def assistant_status(self) -> dict:
        pid, model = self._assistant()
        return {"provider": pid, "model": model, "providers": [
            {"id": p.id, "label": p.label, "key": self._keys.masked(p.id) if self._keys else None,
             "model": self._settings.last.get(f"assistant_model_{p.id}") or p.default_model} for p in PROVIDERS.values()]}

    # ---- updates: GitHub releases; the user clicks before anything is downloaded or run ----
    def check_update(self, manual: bool = False) -> dict:
        info = update.latest()
        if info is None:
            return {"error": "offline", "message": "Couldn't reach GitHub to check for updates."} if manual \
                else {"update": False}
        if not update.newer(VERSION, info):
            return {"update": False}
        return {"update": True, "version": info["version"], "notes": info["notes"]}

    def get_update(self) -> dict:
        info = update.latest()
        if info is None:
            return {"error": "Couldn't reach GitHub to download the update."}
        self._downloads_dir.mkdir(parents=True, exist_ok=True)
        dest = self._downloads_dir / f"aiVibeStudio-Setup-{info['version']}.exe"
        try:
            self._fetch_file(info["url"], dest, lambda pct: self._queue.note("update_progress", percent=pct))
        except OSError as e:
            dest.unlink(missing_ok=True)
            return {"error": f"The download failed ({e}). Try again later."}
        if info["size"] and dest.stat().st_size != info["size"]:
            dest.unlink(missing_ok=True)
            return {"error": "The download was incomplete. Try again."}
        if _sha256(dest) != info["sha256"]:
            dest.unlink(missing_ok=True)
            log.warning("update %s: sha256 mismatch, installer deleted", info["version"])
            return {"error": "The download did not match the release's checksum, so it was deleted. Try again later."}
        log.info("update %s downloaded to %s; starting the installer", info["version"], dest)
        self._start_installer(dest)
        if self._window is not None:
            self._window.destroy()          # the installer asks for the app to be closed
        return {"ok": True}

    @staticmethod
    def _fetch_file(url, dest: Path, on_progress) -> None:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": f"aiVibe-Studio/{VERSION}"})
        with urllib.request.urlopen(req, timeout=30) as r, open(dest, "wb") as f:
            total, got, last = int(r.headers.get("Content-Length") or 0), 0, -1
            while chunk := r.read(1 << 20):
                f.write(chunk)
                got += len(chunk)
                pct = int(got * 100 / total) if total else None
                if pct != last:
                    on_progress(pct)
                    last = pct

    @staticmethod
    def _start_installer(path: Path) -> None:
        os.startfile(str(path))     # Windows only: the installer is a Windows program

    # ---- Hugging Face token (gated models), stored like the assistant keys ----
    def set_hf_token(self, token: str) -> dict:
        try:
            self._keys.set("huggingface", token)
        except (ValueError, OSError, AttributeError) as e:
            return {"error": f"could not save the token: {e}"}
        log.info("Hugging Face token saved")
        return {"ok": True, "key": self._keys.masked("huggingface")}

    def hf_token_status(self) -> dict:
        return {"key": self._keys.masked("huggingface") if self._keys is not None else None}

    def remove_hf_token(self) -> dict:
        try:
            self._keys.remove("huggingface")
        except (OSError, AttributeError) as e:
            return {"error": f"could not remove the token: {e}"}
        return {"ok": True}

    def set_assistant_key(self, provider: str, key: str) -> dict:
        try:
            self._keys.set(provider, key)
        except (ValueError, OSError, AttributeError) as e:
            return {"error": f"could not save the key: {e}"}
        log.info("assistant key saved for %s", provider)
        return {"ok": True, "key": self._keys.masked(provider)}

    def remove_assistant_key(self, provider: str) -> dict:
        try:
            self._keys.remove(provider)
        except (ValueError, OSError) as e:
            return {"error": str(e)}
        return {"ok": True}

    def assistant_models(self, provider: str) -> dict:
        if provider not in PROVIDERS:
            return {"error": f"unknown provider {provider!r}"}
        key, err = self._key_or_error(provider)
        if err:
            return err
        try:
            return {"models": PROVIDERS[provider].list_models(key)}
        except ProviderError as e:
            return {"error": str(e)}

    def _draft(self, fn, *args) -> dict:
        pid, model = self._assistant()
        key, err = self._key_or_error(pid)
        if err:
            return err
        try:
            d = fn(pid, key, model, *args, self._registry, self._subjects, self._settings.last.get("model", "realvis-v5"))
        except ProviderError as e:
            log.warning("assistant %s %s failed: %s", pid, model, e)
            return {"error": str(e)}
        except Exception as e:      # never leave the window waiting on an answer that will not come
            log.exception("assistant %s %s failed", pid, model)
            return {"error": _unexpected(e)}
        log.info("assistant %s %s: %d lines, problems %s, %s tokens, %.1f s", pid, model,
                 len(d.batch.get("items") or []), d.problems, d.tokens, d.seconds)
        return {"batch": d.batch, "problems": d.problems, "provider": d.provider, "label": PROVIDERS[pid].label,
                "model": d.model, "tokens": d.tokens, "seconds": d.seconds}

    def assistant_write(self, request: str) -> dict:
        request = str(request or "").strip()
        if not request:
            return {"error": "Type what you want first."}
        return self._draft(lambda *a: assistant.write_batch(*a), request)

    def assistant_revise(self, request: str, batch: dict, instruction: str) -> dict:
        instruction = str(instruction or "").strip()
        if not instruction:
            return {"error": "Type how to change it first."}
        return self._draft(lambda pid, key, model, *rest: assistant.revise_batch(pid, key, model, str(request or ""), batch,
                                                                                instruction, *rest))

    def assistant_run(self, request: str, batch: dict, provider: str, model: str) -> dict:
        """Go: check the (possibly edited) batch again, keep a copy, run it."""
        try:
            return self._assistant_run(request, batch, provider, model)
        except Exception as e:
            log.exception("assistant run failed")
            return {"error": _unexpected(e)}

    def _assistant_run(self, request, batch, provider, model) -> dict:
        fallback = self._settings.last.get("model", "realvis-v5")
        batch = batch if isinstance(batch, dict) else {}
        batch = {"defaults": batch.get("defaults") or {}, "items": batch.get("items") or []}
        problems = assistant.check_assistant_batch(batch, self._registry, self._subjects, fallback)
        if problems:
            return {"problems": problems}
        items = parse_batch(json.dumps(batch), "assistant.json", self._registry, fallback, subjects=self._subjects)
        need = [n for n in {self._needs_download(i.model_id) and i.model_id for i in items} if n]
        if need:
            return {"needs_download": [self._needs_download(m) for m in sorted(need)]}
        slug = slugify(str(request or "batch"), max_words=6) or "batch"
        folder = self._data() / "assistant"
        folder.mkdir(parents=True, exist_ok=True)
        f = folder / f"{datetime.now():%Y-%m-%d_%H%M%S}_{slug}.json"
        f.write_text(json.dumps({"request": request, "provider": provider, "model": model, **batch}, indent=2,
                                ensure_ascii=False), encoding="utf-8")
        for it in items:
            it.meta = {"assistant": {"provider": provider, "model": model}}
        job = self._queue.submit_batch(items, f"assistant-{slug}.json")
        log.info("assistant batch %s: %d lines from %s %s", f.name, len(items), provider, model)
        return {"job": job, "count": len(items), "file": str(f)}

    # ---- logging and problem reports ----
    def _data(self) -> Path:
        return self._settings_path.parent

    def log_client(self, level: str, message: str) -> None:
        """The window's own errors and clicks, into the same log."""
        lvl = {"error": logging.ERROR, "warn": logging.WARNING}.get(str(level), logging.INFO)
        logging.getLogger("studio.ui").log(lvl, "%s", str(message)[:4000])

    def report_start(self) -> dict:
        """Screenshot the window as it is now, before the report box covers it."""
        shot = self._data() / "reports" / "_pending.png"
        ok = diag.grab_window(f"aiVibe Studio {VERSION}", shot)
        return {"screenshot": ok}

    def report_problem(self, note: str, selected: str | None = None) -> dict:
        try:
            image = self._checked(selected) if selected else None
            folder = diag.write_report(self._data() / "reports", self._data() / "logs", str(note or ""),
                                       version=VERSION, settings=asdict(self._settings), state=self._queue.state(),
                                       events=self._queue.events(0), image=image,
                                       screenshot=self._data() / "reports" / "_pending.png")
            return {"ok": True, "name": folder.name}
        except OSError as e:
            return {"error": f"could not save the report: {e}"}

    def reports(self) -> list[dict]:
        return diag.list_reports(self._data() / "reports")

    def open_reports(self) -> None:
        d = self._data() / "reports"
        d.mkdir(parents=True, exist_ok=True)
        if hasattr(os, "startfile"):
            os.startfile(str(d))

    def open_logs(self) -> None:
        d = self._data() / "logs"
        d.mkdir(parents=True, exist_ok=True)
        if hasattr(os, "startfile"):
            os.startfile(str(d))

    # ---- GPU ----
    def gpu(self) -> dict:
        st = self._queue.state()
        out = {"state": "working" if st["running"] else "ready", "loaded": self._engine.loaded_model()}
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
                               creationflags=NO_WINDOW if os.name == "nt" else 0)
            used, total, util, temp = [float(x) for x in r.stdout.strip().splitlines()[0].split(",")]
            out.update(used_mb=used, total_mb=total, util=util, temp_c=temp)
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            out.update(used_mb=None, total_mb=None, util=None, temp_c=None)
        return out
