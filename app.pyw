"""aiVibe Studio: double-click entry point (pythonw, no console)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from studio.paths import data_root, hf_home  # noqa: E402
from studio.settings import load_settings  # noqa: E402

LOCAL = data_root()                       # %LOCALAPPDATA%\aiVibe Studio: survives updates and reinstalls
DATA = LOCAL / "data"
SETTINGS = load_settings(DATA / "settings.json")     # plain JSON, safe before the heavy imports
# the model download folder must be set before huggingface_hub/diffusers are imported (restart to change)
os.environ["HF_HOME"] = str(hf_home(SETTINGS.hf_cache_dir, LOCAL))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DISABLE_TELEMETRY", "1")
os.environ.setdefault("OPENCV_OPENCL_DEVICE", "disabled")   # OpenCV's OpenCL next to CUDA crashed the app (facefix.cpu_cv2)

import logging  # noqa: E402
import shutil  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402

(DATA / "logs").mkdir(parents=True, exist_ok=True)
from studio.diag import setup_logging  # noqa: E402
setup_logging(DATA / "logs")        # studio.log, one file per day, 14 days
log = logging.getLogger("studio")
import faulthandler  # noqa: E402
# a native crash (CUDA, OpenCV, the webview) kills the process before logging can run; this writes the Python
# stack of every thread to crash.log first
_crash_file = open(DATA / "logs" / "crash.log", "a", encoding="utf-8", buffering=1)
# faulthandler stamps nothing, and on Windows it also prints exceptions a library catches itself, so mark each run
_crash_file.write(f"=== run started {datetime.now():%Y-%m-%d %H:%M:%S} pid {os.getpid()}\n")
faulthandler.enable(_crash_file, all_threads=True)
threading.excepthook = lambda a: log.error("thread %s crashed", a.thread.name if a.thread else "?",
                                           exc_info=(a.exc_type, a.exc_value, a.exc_traceback))
if sys.stdout is None:   # pythonw has no console; libraries that print must not crash
    sys.stdout = sys.stderr = open(DATA / "logs" / "console.log", "a", encoding="utf-8", buffering=1)


MUTEX = "Local\\aiVibeStudioSingleInstance"


def _show_open_window() -> bool:
    """Bring an open aiVibe Studio window (any version) to the front. False when there is none."""
    import ctypes
    import re
    from ctypes import wintypes
    user32, found = ctypes.windll.user32, []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def each(hwnd, _):
        buf = ctypes.create_unicode_buffer(64)
        user32.GetWindowTextW(hwnd, buf, 64)
        if re.match(r"aiVibe Studio \d", buf.value) and user32.IsWindowVisible(hwnd):   # not an Explorer folder
            found.append(hwnd)
        return True

    user32.EnumWindows(each, 0)
    for hwnd in found:
        user32.ShowWindow(hwnd, 9)          # SW_RESTORE, in case it is minimised
        user32.SetForegroundWindow(hwnd)
    return bool(found)


def single_instance(wait: float = 30.0) -> bool:
    """A second double-click would start a second queue on the same card and folder. One window only: if one is
    open, bring it to the front; if the last one is still closing (unloading models), wait for it."""
    if os.name != "nt":
        return True
    import ctypes
    import time
    global _mutex
    k32 = ctypes.windll.kernel32
    deadline = time.monotonic() + wait
    while True:
        _mutex = k32.CreateMutexW(None, False, MUTEX)
        if k32.GetLastError() != 183:      # not ERROR_ALREADY_EXISTS: we are the only one
            return True
        k32.CloseHandle(_mutex)
        if _show_open_window():
            log.info("second start: brought the open window to the front")
            return False
        if time.monotonic() > deadline:
            log.warning("second start: the last window is gone but its process did not finish closing in %d s", wait)
            ctypes.windll.user32.MessageBoxW(None, "aiVibe Studio is still closing. Try again in a moment.",
                                             "aiVibe Studio", 0x40)
            return False
        time.sleep(0.5)


def main():
    if not single_instance():
        return
    import webview
    from studio import VERSION
    from studio.api import Api
    from studio.engine import DiffusersEngine
    from studio.gpu import make_marker
    from studio.jobs import JobQueue
    from studio.keys import KeyStore
    from studio.registry import Registry
    from studio.subjects import SubjectStore
    from studio.trainer import Trainer

    settings_path = DATA / "settings.json"
    settings = SETTINGS
    # Settings > Advanced (blank = off); folder changes take effect after a restart
    registry = Registry(ROOT / "studio" / "catalog.json", DATA / "user_models.json",
                        [Path(d) for d in settings.extra_model_folders if str(d).strip()], Path(os.environ["HF_HOME"]))
    marker = make_marker(settings.gpu_marker_file)
    engine = DiffusersEngine(registry, marker, Path(settings.gpu_jobs_folder) if settings.gpu_jobs_folder.strip() else None,
                             idle_minutes=settings.idle_minutes)
    subjects = SubjectStore(DATA / "subjects")
    # the venv's python.exe (console build, run hidden): sys.executable under pythonw is the base interpreter,
    # which would not see the venv's packages
    trainer = Trainer(subjects, python_exe=str(ROOT / "venv" / "Scripts" / "python.exe"),
                      base_resolver=lambda: registry.local_path("realvis-v5"))     # subjects train on RealVisXL V5
    queue = JobQueue(engine, registry, settings, trainer=trainer, subjects=subjects)
    queue.start()
    api = Api(queue, registry, settings, settings_path, engine, subjects=subjects,
              keys=KeyStore(DATA / "assistant_keys.json"), python_exe=str(ROOT / "venv" / "Scripts" / "python.exe"))
    window = webview.create_window(f"aiVibe Studio {VERSION}", str(ROOT / "ui" / "index.html"), js_api=api,
                                   width=1600, height=1000, min_size=(1280, 800), background_color="#0d0f14")
    api._window = window

    closed = threading.Event()

    def closing():
        import time
        t = time.monotonic()
        log.info("closing")
        try:
            queue.cancel()
            queue.stop()
            engine.unload()
            marker.clear()
            log.info("closed in %.1f s", time.monotonic() - t)
        finally:
            closed.set()

    window.events.closing += closing
    # The window keeps an HTTP cache of the page and served a stale index.html/app.js after an update, so the
    # new buttons never showed (2026-10-19). Drop it at every start; saved storage (Local Storage) is kept.
    for cache in ("Cache", "Code Cache", "GPUCache"):
        shutil.rmtree(DATA / "webview" / "EBWebView" / "Default" / cache, ignore_errors=True)
    import platform
    log.info("aiVibe Studio %s starting (Python %s, %s)", VERSION, platform.python_version(), platform.platform())
    webview.start(gui="edgechromium", private_mode=False, storage_path=str(DATA / "webview"))
    # Everything is saved and the card is free. A normal interpreter exit then spends a long time tearing down
    # torch and the loaded models, with no window and the single-instance mutex still held: double-clicking the
    # icon in that time said "already open" (2026-10-19). The OS frees all of it at once.
    if not closed.wait(60):
        log.warning("cleanup did not finish in 60 s; exiting anyway")
    log.info("window gone, exiting")
    logging.shutdown()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.error("crash:\n%s", traceback.format_exc())
        raise
