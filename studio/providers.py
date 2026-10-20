"""Cloud AI providers for the assistant: one small adapter per service, plain HTTPS through urllib (no SDKs).
Only text goes out: the system brief and the user's request. Keys go in headers and are never logged."""
import http.client
import json
import logging
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

log = logging.getLogger("studio.assistant")


class ProviderError(Exception):
    """A sentence the user can act on."""


@dataclass
class Reply:
    text: str
    model: str
    tokens: int | None


class _HttpError(Exception):
    def __init__(self, code, message=""):
        super().__init__(code)
        self.code, self.message = code, message


def _send(req, timeout) -> dict:
    """The only network call. HTTP and network failures come back as _HttpError(code), code 0 = unreachable,
    -1 = timeout, so each provider can word them with its own name."""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8", "replace") or "{}")
            err = body.get("error")
            msg = err.get("message") if isinstance(err, dict) else (err or body.get("message") or "")
        except (ValueError, AttributeError):
            msg = ""
        raise _HttpError(e.code, str(msg)[:200])
    except (socket.timeout, TimeoutError):
        raise _HttpError(-1)
    except urllib.error.URLError as e:
        raise _HttpError(-1 if isinstance(e.reason, (socket.timeout, TimeoutError)) else 0)
    except http.client.InvalidURL:
        raise _HttpError(404)       # a model name with a space or other character no URL can hold
    except (OSError, http.client.HTTPException):
        raise _HttpError(0)         # dropped mid-answer (reset, remote closed, SSL): urllib only wraps the send
    except ValueError:
        raise _HttpError(200, "the answer was not JSON")


def _post(url: str, headers: dict, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    return _send(req, timeout)


def _get(url: str, headers: dict, timeout: float) -> dict:
    return _send(urllib.request.Request(url, headers=headers, method="GET"), timeout)


class Provider:
    id = label = default_model = ""

    # ---- per provider ----
    def _headers(self, key: str) -> dict:
        raise NotImplementedError

    def _request(self, model: str, system: str, user: str) -> tuple[str, dict]:
        raise NotImplementedError

    def _parse(self, d: dict) -> tuple[str, int | None, str]:
        """(text, tokens, finish reason) from a reply."""
        raise NotImplementedError

    def _models_url(self) -> str:
        raise NotImplementedError

    def _models(self, d: dict) -> list[str]:
        return sorted(m["id"] for m in d.get("data", []) if m.get("id"))

    # ---- shared ----
    def _call(self, fn, model: str, timeout: float):
        try:
            return fn()
        except _HttpError as e:
            raise ProviderError(self._explain(e, model, timeout)) from None

    def _explain(self, e: _HttpError, model: str, timeout: float) -> str:
        if e.code in (401, 403):
            return f"{self.label} rejected the API key. Check it in Settings > Assistant."
        if e.code == 404:
            return f"{self.label} has no model named '{model}'. Pick one in Settings > Assistant."
        if e.code == 429:
            return f"{self.label} says the quota or rate limit is used up. Try again later or check your plan."
        if e.code == 0:
            return f"Could not reach {self.label}. Is the internet up?"
        if e.code == -1:
            return f"{self.label} did not answer in {timeout:g} s."
        return f"{self.label} returned an error (HTTP {e.code}): {e.message}"

    def complete(self, key: str, model: str, system: str, user: str, timeout: float = 120.0) -> Reply:
        url, body = self._request(model, system, user)
        t = time.monotonic()
        d = self._call(lambda: _post(url, self._headers(key), body, timeout), model, timeout)
        text, tokens, reason = self._parse(d)
        log.info("%s %s answered in %.1f s, %s tokens", self.id, model, time.monotonic() - t, tokens)
        if not text.strip():
            raise ProviderError(f"{self.label} sent back an empty answer (finish reason: {reason or 'not given'}).")
        return Reply(text, model, tokens)

    def list_models(self, key: str, timeout: float = 30.0) -> list[str]:
        return self._models(self._call(lambda: _get(self._models_url(), self._headers(key), timeout), "", timeout))


class Claude(Provider):
    id, label, default_model = "claude", "Claude (Anthropic)", "claude-sonnet-5-5"
    base = "https://api.anthropic.com/v1"

    def _headers(self, key):
        return {"x-api-key": key, "anthropic-version": "2023-06-01"}

    def _request(self, model, system, user):
        return f"{self.base}/messages", {"model": model, "max_tokens": 8000, "system": system,
                                         "messages": [{"role": "user", "content": user}]}

    def _parse(self, d):
        text = "".join(c.get("text", "") for c in d.get("content", []) if c.get("type") == "text")
        u = d.get("usage") or {}
        tokens = (u.get("input_tokens") or 0) + (u.get("output_tokens") or 0) if u else None
        return text, tokens, d.get("stop_reason")

    def _models_url(self):
        return f"{self.base}/models"


class Gemini(Provider):
    id, label, default_model = "gemini", "Gemini (Google)", "gemini-flash-latest"
    base = "https://generativelanguage.googleapis.com/v1beta"

    def _headers(self, key):
        return {"x-goog-api-key": key}

    def _request(self, model, system, user):
        return f"{self.base}/models/{model}:generateContent", {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"responseMimeType": "application/json"}}

    def _parse(self, d):
        cands = d.get("candidates") or [{}]
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        reason = cands[0].get("finishReason") or (d.get("promptFeedback") or {}).get("blockReason")
        return text, (d.get("usageMetadata") or {}).get("totalTokenCount"), reason

    def _models_url(self):
        return f"{self.base}/models?pageSize=1000"

    def _models(self, d):
        return sorted(m["name"].removeprefix("models/") for m in d.get("models", [])
                      if "generateContent" in (m.get("supportedGenerationMethods") or []))


class OpenAI(Provider):
    id, label, default_model = "openai", "OpenAI", "gpt-5-mini"
    base = "https://api.openai.com/v1"

    def _headers(self, key):
        return {"Authorization": f"Bearer {key}"}

    def _request(self, model, system, user):
        return f"{self.base}/chat/completions", {
            "model": model, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}

    def _parse(self, d):
        ch = (d.get("choices") or [{}])[0]
        return (ch.get("message") or {}).get("content") or "", (d.get("usage") or {}).get("total_tokens"), \
            ch.get("finish_reason")

    def _models_url(self):
        return f"{self.base}/models"


class Grok(OpenAI):
    id, label, default_model = "grok", "Grok (xAI)", "grok-4"
    base = "https://api.x.ai/v1"


PROVIDERS: dict[str, Provider] = {p.id: p for p in (Claude(), Gemini(), OpenAI(), Grok())}
