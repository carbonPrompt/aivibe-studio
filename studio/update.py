"""New-version check against the project's GitHub releases. Nothing installs by itself: the app offers the
installer and the user decides."""
import json
import re
import urllib.request

from studio import VERSION

REPO = "carbonPrompt/aivibe-studio"
ASSET_RE = r"^aiVibeStudio-Setup-(\d+\.\d+\.\d+)\.exe$"
API_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
# only an installer attached to this repo's own releases is ever offered, over https, with GitHub's sha256
DOWNLOAD_PREFIX = f"https://github.com/{REPO}/releases/download/"


def parse_version(s: str) -> tuple:
    return tuple(int(x) for x in str(s).strip().lstrip("vV").split(".")[:3])


def _fetch(url, headers, timeout) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
        return r.read()


def latest(fetch=None, timeout=10) -> dict | None:
    """{"version", "url", "notes", "size", "sha256"} of the newest release's installer, or None (offline, no
    release, rate-limited, no installer attached, or one that is not from this repo's releases)."""
    try:
        raw = (fetch or _fetch)(API_URL, {"User-Agent": f"aiVibe-Studio/{VERSION}",
                                          "Accept": "application/vnd.github+json"}, timeout)
        d = json.loads(raw)
        for a in d.get("assets") or []:
            m = re.match(ASSET_RE, a.get("name", ""))
            digest = str(a.get("digest") or "")
            if m and a["browser_download_url"].startswith(DOWNLOAD_PREFIX) \
                    and re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
                return {"version": m.group(1), "url": a["browser_download_url"], "notes": d.get("body") or "",
                        "size": int(a.get("size") or 0), "sha256": digest.split(":", 1)[1]}
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def newer(current: str, info: dict | None) -> bool:
    try:
        return bool(info) and parse_version(info["version"]) > parse_version(current)
    except (ValueError, KeyError):
        return False
