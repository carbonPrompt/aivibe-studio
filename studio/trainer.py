"""Train a subject's LoRA by running the vendored diffusers SDXL DreamBooth-LoRA script in a child process."""
import os
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "train" / "train_dreambooth_lora_sdxl.py"
MIN_PHOTOS = 5
NO_WINDOW = 0x08000000
_PROGRESS = re.compile(r"Steps:.*?(\d+)/(\d+) \[")


class TrainError(RuntimeError):
    pass


def steps_for(photo_count: int) -> int:
    return max(800, min(1600, photo_count * 40))


def parse_progress(line: str):
    m = _PROGRESS.search(line)
    return (int(m.group(1)), int(m.group(2))) if m else None


def build_command(subject, store, base_path: Path, python_exe: str, steps: int, variant: str | None = None) -> list[str]:
    d = store.root / subject.id
    cmd = [python_exe, str(SCRIPT),
           "--pretrained_model_name_or_path", str(base_path),
           "--instance_data_dir", str(d / "photos"),
           "--instance_prompt", f"photo of {subject.phrase}",
           "--output_dir", str(d / "training"),
           "--resolution", "1024", "--train_batch_size", "1", "--gradient_accumulation_steps", "1",
           "--gradient_checkpointing", "--use_8bit_adam", "--mixed_precision", "fp16",
           "--learning_rate", "1e-4", "--lr_scheduler", "constant", "--lr_warmup_steps", "0",
           "--max_train_steps", str(steps), "--rank", "16", "--checkpointing_steps", "100000",
           "--seed", "7", "--dataloader_num_workers", "0"]
    if variant:
        cmd += ["--variant", variant]
    return cmd


def _kill_tree(proc):
    import psutil
    try:
        parent = psutil.Process(proc.pid)
        for c in parent.children(recursive=True):
            c.kill()
        parent.kill()
    except psutil.Error:
        pass


class Trainer:
    def __init__(self, store, python_exe: str, base_resolver, run=subprocess.Popen, kill_tree=_kill_tree):
        """base_resolver() -> the RealVisXL V5 folder, or None while it is not downloaded (looked up per run)."""
        self.store, self.python_exe, self.base_resolver = store, python_exe, base_resolver
        self.run, self.kill_tree = run, kill_tree

    def train(self, subject_id: str, cancel: threading.Event, on_progress) -> Path:
        """Blocks until done. on_progress(step, steps, eta_seconds). The new LoRA replaces the old one only on
        success; a failed or cancelled run leaves the subject as it was."""
        s = self.store.get(subject_id)
        photos = len(self.store.photo_paths(subject_id))
        if photos < MIN_PHOTOS:
            raise TrainError(f"{s.name} has {photos} photos; at least {MIN_PHOTOS} are needed")
        steps = steps_for(photos)
        base = self.base_resolver()
        if base is None:
            raise TrainError("training needs the RealVisXL V5.0 model; download it first")
        base = Path(base)
        variant = "fp16" if any(base.glob("*/*.fp16.safetensors")) else None   # some folders hold only fp16
        work = self.store.root / subject_id / "training"
        shutil.rmtree(work, ignore_errors=True)
        log_path = self.store.root / subject_id / "training.log"
        env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_DISABLE_TELEMETRY="1")
        cmd = build_command(s, self.store, base, self.python_exe, steps, variant)
        proc = self.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                        creationflags=NO_WINDOW if os.name == "nt" else 0)
        stop = threading.Event()

        def watch_cancel():
            while not stop.wait(0.5):
                if cancel.is_set():
                    self.kill_tree(proc)
                    return

        threading.Thread(target=watch_cancel, daemon=True).start()
        tail, t0, buf = [], time.time(), b""
        try:
            with open(log_path, "w", encoding="utf-8", errors="replace") as log:
                while True:
                    chunk = proc.stdout.read(4096) if hasattr(proc.stdout, "read") else b""
                    if not chunk:
                        break
                    buf += chunk
                    parts = re.split(rb"[\r\n]", buf)
                    buf = parts.pop()
                    for raw in parts:
                        line = raw.decode("utf-8", "replace").strip()
                        if not line:
                            continue
                        log.write(line + "\n")
                        tail = (tail + [line])[-40:]
                        p = parse_progress(line)
                        if p:
                            done, total = p
                            eta = (time.time() - t0) / done * (total - done) if done else None
                            on_progress(done, total, eta)
            rc = proc.wait()
        finally:
            stop.set()
        result = work / "pytorch_lora_weights.safetensors"
        try:
            if cancel.is_set():
                raise TrainError("training cancelled")
            if rc != 0 or not result.exists():
                errs = [l for l in tail if "Error" in l or "error" in l or "memory" in l] or tail[-3:]
                raise TrainError(f"training failed (exit {rc}): {' | '.join(errs[-3:])}")
            os.replace(result, self.store.lora_path(subject_id))
            s = self.store.get(subject_id)
            s.steps, s.trained_at, s.photo_count = steps, datetime.now().isoformat(timespec="seconds"), photos
            self.store.update(s)
            return self.store.lora_path(subject_id)
        finally:
            shutil.rmtree(work, ignore_errors=True)
