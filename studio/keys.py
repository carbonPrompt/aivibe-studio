"""The assistant's API keys, pasted by the user. Kept only on this PC in data\\assistant_keys.json, never in
settings.json (problem reports copy that) and never in the log. On Windows each key is encrypted with DPAPI for
the logged-in user, so a copied file is useless elsewhere."""
import base64
import json
import os
from pathlib import Path

PROVIDER_IDS = ("claude", "gemini", "openai", "grok")
KEY_IDS = PROVIDER_IDS + ("huggingface",)      # the Hugging Face read token for gated models


def _blob(data: bytes):
    import ctypes
    from ctypes import wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    buf = ctypes.create_string_buffer(data, len(data))
    return DATA_BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))), buf, DATA_BLOB


def _dpapi(data: bytes, protect: bool) -> bytes:
    import ctypes
    blob_in, _keep, DATA_BLOB = _blob(data)
    blob_out = DATA_BLOB()
    crypt32, kernel32 = ctypes.windll.crypt32, ctypes.windll.kernel32
    fn = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    if not fn(ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)):
        raise OSError("Windows could not " + ("encrypt" if protect else "decrypt") + " the key")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        kernel32.LocalFree(blob_out.pbData)


def _protect(data: bytes) -> bytes:
    return _dpapi(data, True) if os.name == "nt" else data


def _unprotect(data: bytes) -> bytes:
    return _dpapi(data, False) if os.name == "nt" else data


class KeyStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def _read(self) -> dict:
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write(self, d: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(d, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    @staticmethod
    def _check(provider: str) -> None:
        if provider not in KEY_IDS:
            raise ValueError(f"unknown provider {provider!r}")

    def set(self, provider: str, key: str) -> None:
        self._check(provider)
        key = str(key or "").strip()
        if not key:
            raise ValueError("the key is empty")
        d = self._read()
        d[provider] = base64.b64encode(_protect(key.encode("utf-8"))).decode("ascii")
        self._write(d)

    def get(self, provider: str) -> str | None:
        self._check(provider)
        v = self._read().get(provider)
        if not v:
            return None
        try:
            return _unprotect(base64.b64decode(v)).decode("utf-8")
        except (ValueError, OSError):
            return None

    def remove(self, provider: str) -> None:
        self._check(provider)
        d = self._read()
        if d.pop(provider, None) is not None:
            self._write(d)

    def masked(self, provider: str) -> str | None:
        k = self.get(provider)
        return f"saved, ends in {k[-4:]}" if k else None
