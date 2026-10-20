"""Optional GPU sharing with another job runner on this PC: a busy marker file it can read, and a check for one
of its jobs running (both set in Settings > Advanced; off when blank)."""
import json
import os
import time
from pathlib import Path

import psutil

STALE_SECONDS = 120


def _pid_alive(pid: int) -> bool:
    return psutil.pid_exists(pid)


class BusyMarker:
    def __init__(self, path: Path, pid_alive=None):
        self.path = Path(path)
        self.pid_alive = pid_alive or _pid_alive

    def _write(self, state: str):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps({"pid": os.getpid(), "heartbeat": time.time(), "state": state}))
        os.replace(tmp, self.path)

    def acquire(self):
        self._write("busy")

    def heartbeat(self):
        # best effort: on Windows the probe, Defender or the indexer can hold the file and os.replace then fails;
        # a missed beat is harmless, an exception here would abort the image being rendered
        try:
            self._write("busy")
        except OSError:
            pass

    def release(self):
        """Model unloaded. The process still holds a CUDA context (torch cannot drop it), so nvidia-smi keeps
        listing it; the 'idle' state tells the other runner to ignore it."""
        try:
            self._write("idle")
        except OSError:
            pass

    def clear(self):
        """App closing: remove the marker if it is ours."""
        try:
            if json.loads(self.path.read_text()).get("pid") == os.getpid():
                self.path.unlink()
        except (OSError, ValueError):
            pass

    def is_held_by_other(self, now: float | None = None) -> bool:
        try:
            d = json.loads(self.path.read_text())
            pid, hb = int(d["pid"]), float(d["heartbeat"])
        except (OSError, ValueError, KeyError, TypeError):
            return False
        if d.get("state") == "idle":
            return False
        now = time.time() if now is None else now
        return pid != os.getpid() and now - hb <= STALE_SECONDS and self.pid_alive(pid)


def _is_job_process(pid: int) -> bool:
    """Alive AND a PowerShell: old job folders keep their pid file, and Windows reuses PIDs."""
    try:
        return psutil.Process(pid).name().lower().startswith("powershell")
    except (psutil.Error, OSError):
        return False


def _started(pid: int) -> float:
    try:
        return psutil.Process(pid).create_time()
    except (psutil.Error, OSError):
        return float("inf")


def other_job_running(jobs_root: Path) -> bool:
    """A job of the other runner writes its PID to <jobs_root>/<id>/pid while it runs."""
    root = Path(jobs_root)
    if not root.is_dir():
        return False
    for pidf in root.glob("*/pid"):
        try:
            pid = int(pidf.read_text().strip())
            # the job's PowerShell writes its pid file within seconds of starting; a process that started after
            # the file was written is a different program that inherited a reused PID
            if _is_job_process(pid) and _started(pid) <= pidf.stat().st_mtime + 5:
                return True
        except (OSError, ValueError):
            continue
    return False


class NullMarker:
    """GPU sharing off: the same calls as BusyMarker, doing nothing."""
    path = None

    def acquire(self):
        pass

    def heartbeat(self):
        pass

    def release(self):
        pass

    def clear(self):
        pass


def make_marker(path: str):
    """A BusyMarker at path, or a NullMarker when the setting is blank."""
    return BusyMarker(Path(path)) if path and str(path).strip() else NullMarker()
