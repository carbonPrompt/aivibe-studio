"""Logging and problem reports. Everything stays on this PC: data\\logs (kept 14 days) and data\\reports."""
import functools
import inspect
import json
import logging
import logging.handlers
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path

LOG_FILE = "studio.log"
FORMAT = "%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s"
_STAMP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
log = logging.getLogger("studio.api")


def setup_logging(logs_dir: Path, days: int = 14) -> logging.Handler:
    """One file per day (studio.log, then studio.log.YYYY-MM-DD), the oldest dropped after `days`."""
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    h = logging.handlers.TimedRotatingFileHandler(logs_dir / LOG_FILE, when="midnight", backupCount=days,
                                                  encoding="utf-8", delay=True)
    h.setFormatter(logging.Formatter(FORMAT))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(h)
    logging.captureWarnings(True)
    return h


def short(v, limit: int = 300) -> str:
    """repr for the log, cut short: prompts and paths yes, base64 images no."""
    s = repr(v)
    if isinstance(v, str) and v.startswith("data:"):
        return f"<data {len(v)} chars>"
    return s if len(s) <= limit else s[:limit] + f"...(+{len(s) - limit})"


def logged_api(quiet=()):
    """Class decorator for the window API: every public call is logged (except `quiet` polling calls), and any
    exception is logged with its trace before it goes back to the window as before."""
    def wrap(cls):
        for name, fn in list(vars(cls).items()):
            if name.startswith("_") or not inspect.isfunction(fn):
                continue
            setattr(cls, name, _logged(fn, name in quiet))
        return cls
    return wrap


def _logged(fn, quiet):
    @functools.wraps(fn)
    def call(self, *a, **kw):
        if not quiet:
            log.info("call %s(%s)", fn.__name__, ", ".join([short(x) for x in a] + [f"{k}={short(v)}" for k, v in kw.items()]))
        try:
            return fn(self, *a, **kw)
        except Exception:
            log.exception("call %s failed", fn.__name__)
            raise
    call.__signature__ = inspect.signature(fn)     # pywebview reads the parameter names with getfullargspec
    return call


def log_files(logs_dir: Path) -> list[Path]:
    """studio.log and its rotated days, oldest first."""
    logs_dir = Path(logs_dir)
    old = sorted(logs_dir.glob(LOG_FILE + ".*"))
    cur = logs_dir / LOG_FILE
    return old + ([cur] if cur.exists() else [])


def log_tail(logs_dir: Path, minutes: int, now: datetime) -> str:
    """The log lines of the last `minutes`, traces included (their lines carry no time of their own)."""
    cutoff = now - timedelta(minutes=minutes)
    out, keep = [], False
    for f in log_files(logs_dir):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            m = _STAMP.match(line)
            if m:
                keep = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") >= cutoff
            if keep:
                out.append(line)
    return "\n".join(out) + ("\n" if out else "")


def write_report(reports_dir: Path, logs_dir: Path, note: str, *, version: str, settings: dict, state: dict,
                 events: list, image: Path | None = None, screenshot: Path | None = None,
                 now: datetime | None = None, minutes: int = 30) -> Path:
    """A folder with everything needed to work a bug without asking: note, recent log, queue state and events,
    settings, the selected picture with its recipe, and a screenshot of the window."""
    now = now or datetime.now()
    reports_dir = Path(reports_dir)
    base = reports_dir / f"{now:%Y-%m-%d_%H%M%S}"
    folder, n = base, 2
    while folder.exists():
        folder = base.with_name(f"{base.name}_{n}")
        n += 1
    folder.mkdir(parents=True)
    (folder / "note.txt").write_text(note.strip() + "\n", encoding="utf-8")
    (folder / "log.txt").write_text(log_tail(logs_dir, minutes, now), encoding="utf-8")
    (folder / "settings.json").write_text(json.dumps(settings, indent=2, default=str), encoding="utf-8")
    info = {"version": version, "time": now.isoformat(timespec="seconds"), "note": note.strip(), "state": state,
            "selected": str(image) if image else None, "events": events[-300:]}
    if image and Path(image).is_file():
        image = Path(image)
        shutil.copy2(image, folder / image.name)
        recipe = image.with_suffix(".json")
        if recipe.is_file():
            shutil.copy2(recipe, folder / recipe.name)
    if screenshot and Path(screenshot).is_file():
        shutil.move(str(screenshot), folder / "screen.png")
    (folder / "report.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    logging.getLogger("studio.report").info("problem report saved: %s (%s)", folder.name, short(note.strip(), 120))
    return folder


def list_reports(reports_dir: Path, limit: int = 20) -> list[dict]:
    reports_dir = Path(reports_dir)
    if not reports_dir.is_dir():
        return []
    out = []
    for d in sorted((d for d in reports_dir.iterdir() if d.is_dir()), reverse=True)[:limit]:
        try:
            note = (d / "note.txt").read_text(encoding="utf-8").strip().splitlines()[0]
        except (OSError, IndexError):
            note = ""
        out.append({"name": d.name, "note": note})
    return out


def grab_window(title: str, dest: Path) -> bool:
    """Screenshot of the window with this exact title (Windows). False when it cannot be found or grabbed."""
    try:
        import ctypes
        from ctypes import wintypes
        from PIL import ImageGrab
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, title)
        if not hwnd:
            return False
        rect = wintypes.RECT()
        dwm = ctypes.windll.dwmapi     # the visible frame; GetWindowRect includes the invisible resize border
        if dwm.DwmGetWindowAttribute(hwnd, 9, ctypes.byref(rect), ctypes.sizeof(rect)) != 0:
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
        img = ImageGrab.grab(bbox=(rect.left, rect.top, rect.right, rect.bottom), all_screens=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest)
        return True
    except Exception:
        logging.getLogger("studio.report").exception("window screenshot failed")
        return False
