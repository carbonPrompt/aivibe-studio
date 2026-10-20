"""The work queue: one worker thread runs single jobs and batches in order and saves every image."""
import itertools
import logging
import secrets
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from studio import VERSION
from studio.batch import BatchItem, group_by_model
from studio.engine import GenRequest, GpuBusyError, ModelLoadError
from studio.naming import FORMATS, day_folder, output_stem, save_image, slugify, unique_path
from studio.recipe import Recipe, read_recipe, write_recipe
from studio.upscale import check_target, suffix

UPSCALER_NAME = "realesrgan-x4plus"
STEP_KINDS = {"step", "upscale_step", "train_step"}
WARN_KINDS = {"item_failed", "model_failed", "train_failed", "download_failed"}
log = logging.getLogger("studio.jobs")


class JobQueue:
    def __init__(self, engine, registry, settings, clock=datetime.now, trainer=None, subjects=None):
        self.engine, self.registry, self.settings, self.clock = engine, registry, settings, clock
        self.trainer, self.subjects = trainer, subjects
        self._jobs = deque()
        self._events = []
        self._seq = itertools.count(1)
        self._ids = itertools.count(1)
        self._cv = threading.Condition()
        self._cancel = threading.Event()
        self._state = {"running": False, "current": None, "done": 0, "total": 0, "kind": None, "label": None,
                       "started": None}
        self._thread = None
        self._stop = False
        self._cancel_at = None

    # ---- public ----
    def submit_single(self, req: GenRequest, subject: str) -> str:
        item = BatchItem(1, req.prompt, req.negative, req.model_id, req.width, req.height, req.count, req.seed,
                         req.steps, req.guidance, subject, None, [(x[0], x[-1]) for x in req.subjects],
                         req.reference, req.reference_strength, req.face_fix)   # subjects: (id, weight)
        return self._submit("gen", [item], None)

    def submit_batch(self, items: list[BatchItem], batch_file: str) -> str:
        return self._submit("gen", list(items), batch_file)

    def submit_upscale(self, paths: list[str], choice: str) -> str:
        return self._submit("up", [str(p) for p in paths], choice)

    def submit_fix(self, path: str, x: int, y: int, subject_id: str) -> str:
        return self._submit("fix", [str(path)], (int(x), int(y), subject_id))

    def submit_train(self, subject_id: str) -> str:
        return self._submit("train", [subject_id], None)

    def cancel(self) -> None:
        with self._cv:
            dropped = len(self._jobs)
            self._jobs.clear()
            last = next((e for e in reversed(self._events) if e["kind"] in STEP_KINDS), None)
            log.info("Cancel pressed: running=%s job=%s kind=%s label=%r, dropped %d waiting, last progress %s",
                     self._state["running"], self._state["current"], self._state["kind"], self._state["label"],
                     dropped, {k: last[k] for k in ("kind", "index", "step", "steps") if k in last} if last else None)
            if self._state["running"]:
                self._cancel_at = time.time()
                self._cancel.set()
            elif dropped:
                self._emit("cancelled", dropped=dropped)

    def note(self, kind: str, **kw) -> None:
        """An event from outside the queue (downloads, the window), shown in the same log."""
        with self._cv:
            self._emit(kind, **kw)

    def events(self, since: int) -> list[dict]:
        with self._cv:
            return [e for e in self._events if e["seq"] > since]

    def state(self) -> dict:
        with self._cv:
            return dict(self._state, pending=len(self._jobs))

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True, name="jobqueue")
        self._thread.start()

    def stop(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if self._thread:
            self._thread.join(timeout=10)

    # ---- worker ----
    def _submit(self, kind, items, extra) -> str:
        with self._cv:
            job_id = f"job{next(self._ids)}"
            self._jobs.append((job_id, kind, items, extra))
            self._emit("queued", job=job_id, items=len(items))
            self._cv.notify_all()
            return job_id

    def _emit(self, kind, **kw):
        if kind not in STEP_KINDS:
            log.log(logging.WARNING if kind in WARN_KINDS else logging.INFO, "event %s %s", kind,
                    " ".join(f"{k}={v!r}" for k, v in kw.items()))
        self._events.append({"seq": next(self._seq), "kind": kind, "time": time.time(), **kw})
        del self._events[:-2000]

    def _loop(self):
        while True:
            with self._cv:
                while not self._jobs and not self._stop:
                    if not self._cv.wait(timeout=10):
                        self.engine.idle_tick(time.time())
                if self._stop:
                    return
                job_id, kind, items, extra = self._jobs.popleft()
                total = len(items) if kind in ("up", "train", "fix") else sum(i.count * (2 if i.upscale else 1) for i in items)
                self._state.update(running=True, current=job_id, done=0, total=total, kind=kind,
                                   label=self._label(kind, items, extra), started=time.time())
            log.info("job %s start: kind=%s label=%r items=%d total=%d extra=%r", job_id, kind, self._state["label"],
                     len(items), total, extra)
            t0 = time.time()
            try:
                if kind == "up":
                    self._run_upscale(job_id, items, extra)
                elif kind == "train":
                    self._run_train(job_id, items[0])
                elif kind == "fix":
                    self._run_fix(job_id, Path(items[0]), *extra)
                else:
                    self._run(job_id, items, extra)
            except Exception as e:      # never let one job end the worker: later jobs would queue forever
                log.exception("job %s stopped by an error", job_id)
                with self._cv:
                    self._emit("item_failed", job=job_id, index=None, error=f"job stopped: {e}")
                    self._emit("done", job=job_id)
            finally:
                with self._cv:
                    if self._cancel.is_set():
                        log.info("job %s cancelled: stopped %.1f s after Cancel was pressed", job_id,
                                 time.time() - (self._cancel_at or time.time()))
                    log.info("job %s end after %.1f s (%d of %d done)", job_id, time.time() - t0,
                             self._state["done"], self._state["total"])
                    self._state.update(running=False, current=None, kind=None, label=None, started=None)
                    self._cancel.clear()
                    self._cancel_at = None

    def _run(self, job_id, items, batch_file):
        for model_id, group in group_by_model(items):
            try:
                entry = self.registry.get(model_id)
            except KeyError:
                with self._cv:
                    for it in group:
                        self._emit("model_failed", job=job_id, index=it.index, model=model_id,
                                   error=f"model {model_id!r} is no longer installed")
                continue
            with self._cv:
                self._emit("model_loading", job=job_id, model=model_id)
            for it in group:
                if self._cancel.is_set():
                    break
                seed = it.seed if it.seed is not None else secrets.randbelow(2**32 - it.count)
                try:
                    prompt, subj = self._subjects_for(it)
                except ValueError as e:
                    with self._cv:
                        self._emit("item_failed", job=job_id, index=it.index, error=str(e))
                    continue
                req = GenRequest(model_id, prompt, it.negative, it.width, it.height, it.steps, it.guidance,
                                 seed, it.count, subj, it.reference, it.reference_strength, it.face_fix,
                                 self._face_prompt(it))
                log.info("job %s item %s: model=%s %dx%d steps=%d guidance=%s seed=%d count=%d subjects=%s "
                         "reference=%s strength=%s face_fix=%s face_prompt=%r prompt_sent=%r negative=%r",
                         job_id, it.index, model_id, it.width, it.height, it.steps, it.guidance, seed, it.count,
                         [(Path(x[1]).parent.name, x[2]) for x in subj], it.reference, it.reference_strength,
                         it.face_fix, req.face_prompt, prompt, it.negative)

                def on_step(i, n, it=it):
                    self._emit_step(job_id, it.index, i, n)

                saved = []

                def on_image(img, k, it=it, seed=seed, saved=saved):
                    # saved as each image finishes, so a failure on a later image keeps the earlier ones
                    try:
                        path = self._save(img, it, entry, seed + k, k + 1 if it.count > 1 else None, batch_file)
                    except Exception as e:
                        log.exception("saving image %d of item %s failed", k + 1, it.index)
                        with self._cv:
                            self._emit("item_failed", job=job_id, index=it.index, error=f"could not save image: {e}")
                        return
                    saved.append(path)
                    with self._cv:
                        self._state["done"] += 1
                        self._emit("image", job=job_id, index=it.index, path=str(path), seed=seed + k)

                try:
                    self.engine.generate(req, on_step, self._cancel, on_image=on_image)
                except (ModelLoadError, GpuBusyError) as e:
                    with self._cv:
                        for rest in group[group.index(it):]:
                            self._emit("model_failed", job=job_id, index=rest.index, model=model_id, error=str(e))
                    break
                except Exception as e:
                    log.exception("job %s item %s failed", job_id, it.index)
                    with self._cv:
                        self._emit("item_failed", job=job_id, index=it.index, error=str(e))
                finally:
                    if it.upscale:
                        for path in saved:
                            if self._cancel.is_set():
                                break
                            self._upscale_one(job_id, path, it.upscale, it.index)
            if self._cancel.is_set():
                break
        with self._cv:
            self._emit("cancelled" if self._cancel.is_set() else "done", job=job_id)

    def _label(self, kind, items, extra) -> str:
        """What the busy window says is happening."""
        if kind == "up":
            return f"Upscaling {len(items)} picture{'s' if len(items) > 1 else ''} to {extra}"
        if kind == "train":
            try:
                return f"Training {self.subjects.get(items[0]).name}"
            except Exception:
                return "Training"
        if kind == "fix":
            try:
                return f"Fixing a spot as {self.subjects.get(extra[2]).name}"
            except Exception:
                return "Fixing a spot"
        return f"Batch: {extra}" if extra else "Generating"

    def _run_upscale(self, job_id, paths, choice):
        for p in paths:
            if self._cancel.is_set():
                break
            self._upscale_one(job_id, Path(p), choice, None)
        with self._cv:
            self._emit("cancelled" if self._cancel.is_set() else "done", job=job_id)

    FIX_PROMPTS = {"person": "photo of {p}, face close-up, relaxed natural expression, eyes open, sharp focus, "
                             "detailed natural skin",
                   "pet": "photo of {p}, close-up, detailed fur, sharp focus",
                   "car": "photo of {p}, detailed, sharp focus", "thing": "photo of {p}, detailed, sharp focus"}

    def _run_fix(self, job_id, src: Path, x: int, y: int, subject_id: str):
        """Redraw the clicked spot of a saved picture as one subject; saved as <stem>_fix beside it."""
        from PIL import Image
        from studio.facefix import click_region, find_faces
        try:
            s = self.subjects.get(subject_id)
            with Image.open(src) as im:
                img = im.convert("RGB")
            try:
                r = read_recipe(src)
            except (OSError, ValueError, TypeError):
                r = None
            model = r.model_id if r and r.model_id in {m.id for m in self.registry.all()} \
                and self.registry.get(r.model_id).family == "sdxl" else "realvis-v5"
            try:
                faces = find_faces(img) if s.kind == "person" else []
            except ImportError:
                faces = []
            region = click_region(faces, x, y, img.width, img.height)
            log.info("fix %s: click (%d, %d) on %s (%dx%d) as %s (%s); faces found %s; region %s; model %s",
                     job_id, x, y, src.name, img.width, img.height, s.name, s.kind, faces, region, model)
            prompt = self.FIX_PROMPTS.get(s.kind, self.FIX_PROMPTS["thing"]).format(p=s.phrase)
            out = self.engine.fix_region(img, model, region, (s.id, self.subjects.lora_path(s.id), 1.0), prompt,
                                         r.negative if r else "", (r.seed if r else 0) + 7,
                                         r.steps if r and r.steps else 30, r.guidance if r and r.guidance else 5.5,
                                         self._cancel)
            if out is None:
                raise RuntimeError("cancelled")
            path = self._write_image(out, src.parent, f"{src.stem}_fix")
            rec = r or Recipe(prompt="", negative="", model_id=model, model_name="", license="", seed=0, steps=0,
                              guidance=0.0, width=img.width, height=img.height, subject="", app_version=VERSION,
                              created="")
            rec.extra.update(fixed_from=src.name, fixed_subject=s.name, fix_region=list(region))
            write_recipe(path, rec)
            with self._cv:
                self._state["done"] = 1
                self._emit("fixed", job=job_id, path=str(path), source=str(src), subject=s.name)
        except Exception as e:
            if not self._cancel.is_set():      # Cancel is not a failure: the "cancelled" event below says it
                log.exception("fix %s failed", job_id)
                with self._cv:
                    self._emit("item_failed", job=job_id, index=None, error=f"fix failed: {e}")
        with self._cv:
            self._emit("cancelled" if self._cancel.is_set() else "done", job=job_id)

    def _run_train(self, job_id, subject_id):
        from studio.trainer import TrainError

        def on_progress(step, steps, eta):
            self._emit_step(job_id, None, step, steps, kind="train_step", subject=subject_id,
                            eta_s=None if eta is None else int(eta))

        with self.engine.exclusive(self._cancel) as ok:
            if ok:
                with self._cv:
                    self._emit("training", job=job_id, subject=subject_id)
                try:
                    self.trainer.train(subject_id, self._cancel, on_progress)
                    with self._cv:
                        self._state["done"] = 1
                        self._emit("trained", job=job_id, subject=subject_id)
                except TrainError as e:
                    log.warning("training %s ended: %s", subject_id, e)
                    if not self._cancel.is_set():
                        with self._cv:
                            self._emit("train_failed", job=job_id, subject=subject_id, error=str(e))
        with self._cv:
            self._emit("cancelled" if self._cancel.is_set() else "done", job=job_id)

    def _upscale_one(self, job_id, src: Path, choice: str, index):
        """Upscale one saved image into <stem>_<choice> beside it, with a recipe saying where it came from."""
        from PIL import Image
        src = Path(src)
        try:
            with Image.open(src) as im:
                im.load()
                img = im.copy()
        except (OSError, ValueError) as e:
            with self._cv:
                self._emit("item_failed", job=job_id, index=index, error=f"cannot open {src.name}: {e}")
            return
        try:
            check_target(img.width, img.height, choice)
        except ValueError as e:
            with self._cv:
                self._emit("item_failed", job=job_id, index=index, error=str(e))
            return
        t0 = time.time()

        def on_tile(i, n):
            self._emit_step(job_id, index, i, n, kind="upscale_step", name=src.name)

        try:
            out = self.engine.upscale(img, choice, on_tile, self._cancel)
        except Exception as e:
            log.exception("upscale of %s failed", src.name)
            with self._cv:
                self._emit("item_failed", job=job_id, index=index, error=f"upscale of {src.name} failed: {e}")
            return
        if out is None:
            return      # cancelled
        try:
            path = self._write_image(out, src.parent, f"{src.stem}_{suffix(choice)}")
            try:
                r = read_recipe(src)
            except (OSError, ValueError, TypeError):
                r = Recipe(prompt="", negative="", model_id="", model_name="", license="", seed=0, steps=0,
                           guidance=0.0, width=img.width, height=img.height, subject="", app_version=VERSION,
                           created="")
            # width/height stay the generation size (Re-make renders at that size); the file's size goes in extra
            r.app_version = VERSION
            r.extra.update(out_width=out.width, out_height=out.height, upscaled_from=src.name, upscaler=UPSCALER_NAME, upscale=choice,
                           upscale_seconds=round(time.time() - t0, 1),
                           upscaled=self.clock().isoformat(timespec="seconds"))
            write_recipe(path, r)
        except Exception as e:
            log.exception("saving the upscale of %s failed", src.name)
            with self._cv:
                self._emit("item_failed", job=job_id, index=index, error=f"could not save upscale: {e}")
            return
        with self._cv:
            self._state["done"] += 1
            self._emit("upscaled", job=job_id, index=index, path=str(path), source=str(src),
                       seconds=round(time.time() - t0, 1))

    def _emit_step(self, job_id, index, i, n, kind="step", **kw):
        with self._cv:
            # keep only the newest progress event so the log does not fill with them
            if self._events and self._events[-1]["kind"] == kind:
                self._events.pop()
            self._emit(kind, job=job_id, index=index, step=i, steps=n, **kw)

    def _face_prompt(self, it: BatchItem) -> str:
        """Face-fix prompt when exactly one person subject is in the picture (pets have no face fix)."""
        if not it.subjects or self.subjects is None:
            return ""
        people = [s for s in (self.subjects.get(sid) for sid, _ in it.subjects) if s.kind == "person"]
        if len(people) != 1:
            return ""
        return (f"photo of {people[0].phrase}, face close-up, relaxed natural expression, eyes open, sharp focus, "
                "detailed natural skin")

    def _subjects_for(self, it: BatchItem):
        """The prompt with subject names turned into trigger phrases, and [(id, lora path, weight)]."""
        if not it.subjects:
            return it.prompt, []
        from studio.subjects import rewrite_prompt
        if self.subjects is None:
            raise ValueError("subjects are not available")
        subs = []
        for sid, w in it.subjects:
            try:
                s = self.subjects.get(sid)
            except KeyError:
                raise ValueError(f"subject {sid!r} no longer exists")
            if not s.trained:
                raise ValueError(f"subject {s.name!r} is not trained yet")
            subs.append((s, w))
        return (rewrite_prompt(it.prompt, [s for s, _ in subs]),
                [(s.id, self.subjects.lora_path(s.id), w) for s, w in subs])

    def _save(self, img, it: BatchItem, entry, seed, index, batch_file) -> Path:
        now = self.clock()
        folder = day_folder(Path(self.settings.output_root), now)
        if batch_file and self.settings.last.get("batch_subfolder", True):
            # a batch's pictures together in one folder named after the batch file
            folder = folder / slugify(Path(batch_file).stem, max_words=6)
            folder.mkdir(parents=True, exist_ok=True)
        path = self._write_image(img, folder, output_stem(it.subject, now, index))
        write_recipe(path, Recipe(prompt=it.prompt, negative=it.negative, model_id=entry.id, model_name=entry.name,
                                  license=entry.license, seed=seed, steps=it.steps, guidance=it.guidance,
                                  width=it.width, height=it.height, subject=it.subject, app_version=VERSION,
                                  created=now.isoformat(timespec="seconds"), batch_file=batch_file,
                                  extra=self._likeness_extra(it)))
        return path

    def _likeness_extra(self, it: BatchItem) -> dict:
        extra = dict(it.meta)       # e.g. {"assistant": {provider, model}}; the likeness keys below win a clash
        if it.subjects and self.subjects is not None:
            from studio.subjects import rewrite_prompt
            subs = [self.subjects.get(sid) for sid, _ in it.subjects]
            extra["subjects"] = [{"id": s.id, "name": s.name, "weight": w} for s, (_, w) in zip(subs, it.subjects)]
            extra["prompt_sent"] = rewrite_prompt(it.prompt, subs)
        if it.reference:
            extra["reference"] = Path(it.reference).name
            extra["reference_strength"] = it.reference_strength
        return extra

    def _write_image(self, img, folder: Path, stem: str) -> Path:
        """Save in the format chosen in Settings (PNG, JPG or WebP)."""
        fmt = self.settings.last.get("output_format", "png")
        fmt = fmt if fmt in FORMATS else "png"
        quality = int(self.settings.last.get("output_quality", 92))
        while True:
            path = unique_path(folder, stem, FORMATS[fmt][0])
            try:
                with open(path, "xb") as f:     # exclusive create: never overwrite, even if a name was just taken
                    save_image(img, f, fmt, quality)
                return path
            except FileExistsError:
                continue
