"""The GPU side. The only module that imports torch/diffusers, and only inside the real loader."""
import gc
import logging
from contextlib import contextmanager
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from studio.gpu import other_job_running
from studio.upscale import upscale_image

UPSCALER_REPO, UPSCALER_FILE = "Comfy-Org/Real-ESRGAN_repackaged", "RealESRGAN_x4plus.safetensors"   # BSD-3-Clause

NEGATIVE_FAMILIES = {"sd15", "sdxl", "playground", "sd3"}   # FLUX and Z-Image pipelines take no negative prompt
BF16_FAMILIES = {"sd3", "flux", "flux2", "zimage"}
LIKENESS_FAMILIES = {"sdxl"}        # subject LoRAs are trained on RealVisXL (SDXL); IP-Adapter Plus is the SDXL one
MAX_SUBJECTS = 3
log = logging.getLogger("studio.engine")


class Cancelled(Exception):
    """Raised from a step callback when Cancel is pressed, so the pipeline stops at the next step."""


def _stopper(cancel):
    """callback_on_step_end that stops an img2img pass when Cancel is pressed."""
    def cb(pipe, i, t, kw):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        return kw
    return cb


def warn_truncation(pipe, prompt: str, what: str = "prompt") -> None:
    """CLIP reads 77 tokens; diffusers drops the rest with only a console message. Say so in the log."""
    tok = getattr(pipe, "tokenizer", None)
    if tok is None or not prompt:
        return
    try:
        ids = tok(prompt, truncation=False).input_ids
        limit = getattr(tok, "model_max_length", 77)
        if len(ids) > limit:
            dropped = tok.decode(ids[limit - 1:-1])
            log.warning("%s is %d tokens; this model reads %d, so this end is ignored: %r", what, len(ids), limit, dropped)
    except Exception:
        log.debug("token count failed", exc_info=True)


@dataclass
class GenRequest:
    model_id: str
    prompt: str
    negative: str
    width: int
    height: int
    steps: int
    guidance: float
    seed: int
    count: int
    subjects: list = field(default_factory=list)      # [(adapter name, lora path, weight)]
    reference: str | None = None                      # image path for IP-Adapter
    reference_strength: float = 0.5
    face_fix: bool = True                             # redraw small faces with the subjects' likeness
    face_prompt: str = ""                             # set by the queue when exactly one person subject is used


class ModelLoadError(RuntimeError):
    pass


class GpuBusyError(RuntimeError):
    pass


class Runner:
    """A loaded diffusers pipeline plus how to call it for its family."""

    def __init__(self, entry, pipe):
        self.entry, self.pipe = entry, pipe
        self._adapters, self._ip, self._reference = set(), False, None

    def configure(self, subjects, reference, strength):
        """Subject LoRAs as named adapters (loaded once per file version) and the IP-Adapter reference."""
        names = []
        log.info("adapters: subjects %s, reference %s (strength %s)",
                 [(Path(p).parent.name, w) for _, p, w in subjects], reference, strength)
        for name, path, weight in subjects:
            key = f"{name}_{int(Path(path).stat().st_mtime)}"      # retrained subject = new adapter
            if key not in self._adapters:
                self.pipe.load_lora_weights(str(path), adapter_name=key)
                self._adapters.add(key)
            names.append((key, weight))
        if names:
            self.pipe.enable_lora()
            self.pipe.set_adapters([n for n, _ in names], adapter_weights=[w for _, w in names])
        elif self._adapters:
            self.pipe.disable_lora()
        if reference and not self._ip:
            self.pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                                      weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
                                      image_encoder_folder="models/image_encoder")
            self._ip = True
        if self._ip:
            self.pipe.set_ip_adapter_scale(strength if reference else 0.0)
        self._reference = reference

    def redraw(self, img, region, prompt, negative, seed, steps, guidance, strength=0.6, cancel=None):
        """Redraw one square of the picture at 1024 px with the active adapters, blended back in."""
        import torch
        from diffusers import AutoPipelineForImage2Image
        from studio.facefix import paste_region
        if getattr(self, "_i2i", None) is None:
            self._i2i = AutoPipelineForImage2Image.from_pipe(self.pipe)
        crop = img.crop(region).resize((1024, 1024), Image.LANCZOS)
        log.info("redraw region %s of %dx%d (crop %dx%d -> 1024), strength %s, seed %d, prompt %r",
                 region, img.width, img.height, region[2] - region[0], region[3] - region[1], strength, seed, prompt)
        warn_truncation(self.pipe, prompt, "fix prompt")
        kw = dict(prompt=prompt, image=crop, strength=strength, num_inference_steps=max(steps, 25),
                  guidance_scale=guidance, generator=torch.Generator("cpu").manual_seed(seed),
                  callback_on_step_end=_stopper(cancel))
        if negative:
            kw["negative_prompt"] = negative
        if self._ip:
            kw["ip_adapter_image"] = Image.new("RGB", (224, 224))
        return paste_region(img, self._i2i(**kw).images[0], region)

    def face_fix(self, img, req: GenRequest, seed: int, cancel=None):
        """Redraw each small face at 1024 px with the active subject adapters (img2img, low strength)."""
        import torch
        from diffusers import AutoPipelineForImage2Image
        from studio.facefix import choose_faces, face_region, find_faces, paste_region
        found = find_faces(img)
        faces = choose_faces(found, img.width, img.height, people=1)
        log.info("face fix on %dx%d: found %s, chose %s", img.width, img.height, found, faces)
        if not faces:
            return img
        if getattr(self, "_i2i", None) is None:
            self._i2i = AutoPipelineForImage2Image.from_pipe(self.pipe)     # shares weights and LoRA layers
        for f in faces:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            region = face_region(f, img.width, img.height)
            log.info("face fix: face %s -> region %s", f, region)
            crop = img.crop(region).resize((1024, 1024), Image.LANCZOS)
            kw = dict(prompt=req.face_prompt, image=crop, strength=0.6,
                      num_inference_steps=max(req.steps, 25), guidance_scale=req.guidance,
                      generator=torch.Generator("cpu").manual_seed(seed), callback_on_step_end=_stopper(cancel))
            if req.negative:
                kw["negative_prompt"] = req.negative
            if self._ip:
                kw["ip_adapter_image"] = (Image.open(self._reference).convert("RGB") if self._reference
                                          else Image.new("RGB", (224, 224)))
            img = paste_region(img, self._i2i(**kw).images[0], region)
        return img

    def generate(self, req: GenRequest, seed: int, on_step):
        import torch
        g = torch.Generator("cpu").manual_seed(seed)   # cpu generator: same seed, same image, offloaded or not

        def cb(pipe, i, t, kw):
            on_step(i + 1, req.steps)
            return kw

        kw = dict(prompt=req.prompt, width=req.width, height=req.height, num_inference_steps=req.steps,
                  guidance_scale=req.guidance, generator=g, callback_on_step_end=cb)
        if req.negative and self.entry.family in NEGATIVE_FAMILIES:
            kw["negative_prompt"] = req.negative
        warn_truncation(self.pipe, req.prompt)
        if req.negative and "negative_prompt" in kw:
            warn_truncation(self.pipe, req.negative, "negative prompt")
        if self._ip:
            from PIL import Image
            ref = Image.open(self._reference).convert("RGB") if self._reference else Image.new("RGB", (224, 224))
            kw["ip_adapter_image"] = ref
        return self.pipe(**kw).images[0]


def _variant(path: Path):
    """Some model folders store only *.fp16.safetensors, which diffusers only loads when asked for variant fp16."""
    if path is None or not path.is_dir():
        return None
    return "fp16" if any(path.glob("*/*.fp16.safetensors")) else None


def load_runner(entry, path: Path | None, device: str = "cuda") -> Runner:
    import torch
    import diffusers
    try:    # their warnings (token truncation, deprecations) into studio.log as well as the console
        import transformers
        transformers.utils.logging.enable_propagation()
        diffusers.utils.logging.enable_propagation()
    except Exception:
        pass
    cls = getattr(diffusers, entry.pipeline)
    if device == "cpu":
        dtype = torch.float32     # half precision is not usable on the CPU
    else:
        dtype = torch.bfloat16 if entry.family in BF16_FAMILIES else torch.float16
    if "file" in entry.source:
        pipe = cls.from_single_file(str(path), dtype=dtype)
    else:
        src = str(path) if path else entry.source["hf"]
        pipe = cls.from_pretrained(src, dtype=dtype, variant=_variant(path))
    if device == "cpu":
        pipe.to("cpu")
    elif entry.offload:
        # Whole-model cpu offload spills into Windows shared memory on 12 GB (Z-Image: 40 s/step, 2026-10-15).
        # Stream the big modules through the card in blocks instead; small ones stay resident.
        from diffusers.hooks import apply_group_offloading
        cuda, cpu = torch.device("cuda"), torch.device("cpu")
        for name, m in pipe.components.items():
            if not isinstance(m, torch.nn.Module):
                continue
            if name in ("transformer", "text_encoder", "text_encoder_2", "text_encoder_3", "unet"):
                apply_group_offloading(m, onload_device=cuda, offload_device=cpu, offload_type="block_level",
                                       num_blocks_per_group=1, use_stream=True, low_cpu_mem_usage=True)
            else:
                m.to(cuda)
    else:
        pipe.to("cuda")
    return Runner(entry, pipe)


def load_realesrgan(device: str = "cuda"):
    """Real-ESRGAN x4plus as a numpy function: float32 HxWx3 (0..1) in, 4x larger out."""
    import torch
    from huggingface_hub import hf_hub_download
    from spandrel import ModelLoader
    try:    # no network once the 64 MB file is cached (spec: no calls beyond downloads)
        path = hf_hub_download(UPSCALER_REPO, UPSCALER_FILE, local_files_only=True)
    except Exception:
        path = hf_hub_download(UPSCALER_REPO, UPSCALER_FILE)
    desc = ModelLoader().load_from_file(path)
    half = device == "cuda" and desc.supports_half
    desc.to(device).eval()
    if half:
        desc.half()
    dtype = torch.float16 if half else torch.float32

    def fn(a):
        with torch.inference_mode():
            t = torch.from_numpy(a.copy()).permute(2, 0, 1)[None].to(device, dtype)
            return desc(t)[0].permute(1, 2, 0).float().clamp(0, 1).cpu().numpy()

    return fn


class DiffusersEngine:
    def __init__(self, registry, marker, jobs_root: Path, idle_minutes: int = 5, loader=None, clock=time.time,
                 heartbeat_interval: float = 30.0, upscaler_loader=None):
        self.registry, self.marker = registry, marker
        self.jobs_root = Path(jobs_root) if jobs_root else None
        self.idle_minutes = idle_minutes
        self.loader = loader or load_runner
        self.clock = clock
        self.busy_poll, self.busy_timeout = 5.0, 600.0
        self._pipe, self._model, self._last = None, None, clock()
        self.upscaler_loader = upscaler_loader or load_realesrgan
        self._upscaler = None
        self._lock = threading.RLock()
        self.device = "cuda"
        self.heartbeat_interval = heartbeat_interval
        self._beat_stop = None

    def _hold_marker(self):
        """Acquire the busy marker and keep it fresh from a thread until _drop_marker: a 30 GB model can take
        longer than the busy marker's 120 s staleness window to load, before any CUDA context exists."""
        if self._beat_stop is not None:     # already held by the other slot
            return
        self.marker.acquire()
        stop = self._beat_stop = threading.Event()

        def beat():
            while not stop.wait(self.heartbeat_interval):
                self.marker.heartbeat()

        threading.Thread(target=beat, daemon=True, name="studio-heartbeat").start()

    def _drop_marker(self):
        if self._beat_stop is not None:
            self._beat_stop.set()
            self._beat_stop = None
        self.marker.release()

    def set_device(self, device: str) -> None:
        """'cuda' or 'cpu'. A change unloads the current model; the next request loads on the new device."""
        with self._lock:
            if device != self.device:
                self.unload()
                self.device = device

    def loaded_model(self) -> str | None:
        return self._model

    def unload(self) -> None:
        """Free both slots (diffusion pipeline and upscaler) and mark the card idle."""
        with self._lock:
            if self._pipe is None and self._upscaler is None:
                return
            self._pipe, self._model, self._upscaler = None, None, None
            self._free()
            self._drop_marker()

    def _unload_pipe(self) -> None:
        """Free only the diffusion pipeline (switching models); the small upscaler may stay."""
        if self._pipe is None:
            return
        self._pipe, self._model = None, None
        self._free()
        if self._upscaler is None:
            self._drop_marker()

    @staticmethod
    def _free():
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

    def _wait_for_other_jobs(self, cancel) -> bool:
        if self.jobs_root is None:      # GPU sharing off (Settings > Advanced)
            return True
        waited = 0.0
        while other_job_running(self.jobs_root):
            if waited == 0.0:
                log.info("waiting for another GPU job to finish")
            if cancel.is_set():
                return False
            if waited >= self.busy_timeout:
                raise GpuBusyError("the GPU is busy with another job; try again when it finishes")
            time.sleep(self.busy_poll)
            waited += max(self.busy_poll, 0.001)
        return True

    def idle_tick(self, now: float) -> None:
        with self._lock:
            if (self._pipe is not None or self._upscaler is not None) and now - self._last >= self.idle_minutes * 60:
                self.unload()

    def _ensure(self, model_id: str, cancel: threading.Event) -> bool:
        """Load model_id unless it is loaded. False if cancelled while waiting for another GPU job to finish."""
        if self._model == model_id:
            return True
        self._unload_pipe()
        if not self._wait_for_other_jobs(cancel):
            return False
        entry = self.registry.get(model_id)
        self._hold_marker()
        t0 = time.time()
        log.info("loading %s on %s from %s", model_id, self.device, self.registry.local_path(model_id) or entry.source)
        try:
            self._pipe = self.loader(entry, self.registry.local_path(model_id), self.device)
        except Exception as e:
            log.exception("loading %s failed", model_id)
            if self._upscaler is None:      # the upscaler slot still needs the marker
                self._drop_marker()
            raise ModelLoadError(f"could not load {model_id}: {e}") from e
        log.info("loaded %s in %.1f s", model_id, time.time() - t0)
        self._model = model_id
        return True

    def generate(self, req: GenRequest, on_step, cancel: threading.Event, on_image=None) -> list:
        """Render req.count images. on_image(img, k) is called as each one finishes, so images made before a
        later failure (out of memory on image 3) are not lost."""
        family = self.registry.get(req.model_id).family
        if (req.subjects or req.reference) and family not in LIKENESS_FAMILIES:
            raise ValueError("subjects and reference images work with SDXL models only (RealVisXL, Juggernaut)")
        if len(req.subjects) > MAX_SUBJECTS:
            raise ValueError(f"at most {MAX_SUBJECTS} subjects in one picture")
        with self._lock:
            if not self._ensure(req.model_id, cancel):
                return []
            if hasattr(self._pipe, "configure"):
                self._pipe.configure(req.subjects, req.reference, req.reference_strength)
            images = []
            for k in range(req.count):
                if cancel.is_set():
                    break

                def step(i, n):
                    if cancel.is_set():
                        log.info("cancel seen at step %d of %d (image %d)", i, n, k + 1)
                        raise Cancelled()
                    self.marker.heartbeat()
                    on_step(i, n)

                t0 = time.time()
                try:
                    img = self._pipe.generate(req, req.seed + k, step)
                    log.info("image %d of %d drawn in %.1f s (seed %d)", k + 1, req.count, time.time() - t0, req.seed + k)
                    if req.face_fix and req.face_prompt and hasattr(self._pipe, "face_fix"):
                        t1 = time.time()
                        img = self._pipe.face_fix(img, req, req.seed + k, cancel)
                        log.info("face fix took %.1f s", time.time() - t1)
                except Cancelled:
                    log.info("image %d stopped by Cancel; nothing saved for it", k + 1)
                    break
                images.append(img)
                self._last = self.clock()
                if on_image:
                    on_image(img, k)
            self._last = self.clock()
            return images

    def upscale(self, img, choice: str, on_tile, cancel: threading.Event):
        """AI-upscale one image to the choice ('2x', '4x', '1080p', '4k', '<N>px'). None if cancelled."""
        with self._lock:
            if self._upscaler is None:
                if not self._wait_for_other_jobs(cancel):
                    return None
                self._hold_marker()
                try:
                    self._upscaler = self.upscaler_loader(self.device)
                except Exception as e:
                    if self._pipe is None:
                        self._drop_marker()
                    raise ModelLoadError(f"could not load the upscaler: {e}") from e

            def tile(i, n):
                self.marker.heartbeat()
                on_tile(i, n)

            out = upscale_image(img, choice, self._upscaler, on_tile=tile, cancel=cancel)
            self._last = self.clock()
            return out

    @contextmanager
    def exclusive(self, cancel: threading.Event):
        """The whole card for outside work (training): both slots unloaded, wait for other GPU jobs, busy marker held
        with heartbeats. Yields False if cancelled while waiting."""
        with self._lock:
            self.unload()
            if not self._wait_for_other_jobs(cancel):
                yield False
                return
            self._hold_marker()
            try:
                yield True
            finally:
                self._drop_marker()
                self._last = self.clock()

    def fix_region(self, img, model_id, region, subject, prompt, negative, seed, steps, guidance, cancel):
        """Click-to-fix: redraw `region` with only `subject` (id, lora path, weight) active."""
        if self.registry.get(model_id).family not in LIKENESS_FAMILIES:
            raise ValueError("click-to-fix needs an SDXL model")
        with self._lock:
            if not self._ensure(model_id, cancel):
                return None
            self._pipe.configure([subject], None, 0.0)
            self.marker.heartbeat()
            t0 = time.time()
            try:
                out = self._pipe.redraw(img, region, prompt, negative, seed, steps, guidance, cancel=cancel)
            except Cancelled:
                log.info("click-to-fix stopped by Cancel")
                return None
            log.info("click-to-fix redraw took %.1f s", time.time() - t0)
            self._last = self.clock()
            return out
