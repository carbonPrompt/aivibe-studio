import json
import socket
import urllib.error
from io import BytesIO

import pytest

from studio import providers
from studio.providers import PROVIDERS, ProviderError

KEY = "KEY-abc-123"
JSON = '{"items":[]}'


class Recorder:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def post(self, url, headers, body, timeout):
        self.calls.append(("post", url, headers, body)); return self.reply

    def get(self, url, headers, timeout):
        self.calls.append(("get", url, headers, None)); return self.reply


def patch(monkeypatch, reply):
    r = Recorder(reply)
    monkeypatch.setattr(providers, "_post", r.post)
    monkeypatch.setattr(providers, "_get", r.get)
    return r


@pytest.mark.parametrize("pid,canned,tokens,host,header", [
    ("claude", {"content": [{"type": "text", "text": JSON}], "usage": {"input_tokens": 10, "output_tokens": 5}}, 15,
     "api.anthropic.com", "x-api-key"),
    ("gemini", {"candidates": [{"content": {"parts": [{"text": "thinking", "thought": True}, {"text": JSON}]}}],
                "usageMetadata": {"totalTokenCount": 9}}, 9, "generativelanguage.googleapis.com", "x-goog-api-key"),
    ("openai", {"choices": [{"message": {"content": JSON}}], "usage": {"total_tokens": 7}}, 7, "api.openai.com", "Authorization"),
    ("grok", {"choices": [{"message": {"content": JSON}}], "usage": {"total_tokens": 7}}, 7, "api.x.ai", "Authorization"),
])
def test_complete_parses_each_provider(monkeypatch, pid, canned, tokens, host, header):
    r = patch(monkeypatch, canned)
    reply = PROVIDERS[pid].complete(KEY, "m-1", "SYSTEM TEXT", "USER TEXT")
    assert reply.text == JSON and reply.tokens == tokens and reply.model == "m-1"
    _, url, headers, body = r.calls[0]
    assert host in url and KEY in headers[header]
    sent = json.dumps(body)
    assert "SYSTEM TEXT" in sent and "USER TEXT" in sent and KEY not in sent and KEY not in url


def test_labels_and_defaults():
    assert {p: PROVIDERS[p].label for p in PROVIDERS} == {"claude": "Claude (Anthropic)", "gemini": "Gemini (Google)",
                                                          "openai": "OpenAI", "grok": "Grok (xAI)"}
    assert PROVIDERS["gemini"].default_model and PROVIDERS["claude"].default_model == "claude-sonnet-5-5"


def http_error(code, body=b'{"error": {"message": "bad thing happened"}}'):
    return urllib.error.HTTPError("https://x", code, "err", {}, BytesIO(body))


@pytest.mark.parametrize("exc,expect", [
    (http_error(401), "Gemini (Google) rejected the API key. Check it in Settings > Assistant."),
    (http_error(403), "Gemini (Google) rejected the API key. Check it in Settings > Assistant."),
    (http_error(404), "Gemini (Google) has no model named 'm-1'. Pick one in Settings > Assistant."),
    (http_error(429), "Gemini (Google) says the quota or rate limit is used up. Try again later or check your plan."),
    (http_error(500), "Gemini (Google) returned an error (HTTP 500): bad thing happened"),
    (urllib.error.URLError("no route"), "Could not reach Gemini (Google). Is the internet up?"),
    (socket.timeout("timed out"), "Gemini (Google) did not answer in 120 s."),
])
def test_http_errors_become_sentences(monkeypatch, exc, expect):
    calls = []

    def boom(req, timeout):
        calls.append(1); raise exc
    monkeypatch.setattr(providers.urllib.request, "urlopen", boom)
    with pytest.raises(ProviderError) as e:
        PROVIDERS["gemini"].complete(KEY, "m-1", "s", "u")
    assert str(e.value) == expect and len(calls) == 1          # no automatic retry


def test_empty_or_blocked_reply_is_an_error(monkeypatch):
    patch(monkeypatch, {"promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(ProviderError, match="empty answer.*SAFETY"):
        PROVIDERS["gemini"].complete(KEY, "m", "s", "u")
    patch(monkeypatch, {"choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}]})
    with pytest.raises(ProviderError, match="empty answer.*content_filter"):
        PROVIDERS["openai"].complete(KEY, "m", "s", "u")


def test_list_models_gemini_keeps_generate_content_only(monkeypatch):
    patch(monkeypatch, {"models": [{"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
                                   {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]}]})
    assert PROVIDERS["gemini"].list_models(KEY) == ["gemini-2.5-flash"]


def test_list_models_openai_style(monkeypatch):
    patch(monkeypatch, {"data": [{"id": "b"}, {"id": "a"}]})
    assert PROVIDERS["grok"].list_models(KEY) == ["a", "b"] and PROVIDERS["claude"].list_models(KEY) == ["a", "b"]


@pytest.mark.parametrize("exc,expect", [
    (ConnectionResetError("reset"), "Could not reach Gemini (Google). Is the internet up?"),
    (__import__("http.client").client.RemoteDisconnected("gone"), "Could not reach Gemini (Google). Is the internet up?"),
    (__import__("http.client").client.InvalidURL("bad url"), "Gemini (Google) has no model named 'm-1'. Pick one in Settings > Assistant."),
])
def test_connection_drops_and_bad_model_names_become_sentences(monkeypatch, exc, expect):
    def boom(req, timeout):
        raise exc
    monkeypatch.setattr(providers.urllib.request, "urlopen", boom)
    with pytest.raises(ProviderError) as e:
        PROVIDERS["gemini"].complete(KEY, "m-1", "s", "u")
    assert str(e.value) == expect
