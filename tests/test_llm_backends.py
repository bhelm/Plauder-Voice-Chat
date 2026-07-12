"""LLM backends (openai_compat + openclaw), mocked — no real HTTP calls."""
import asyncio

import pytest

from plauder.backends.base import UpstreamTimeoutError
from plauder.backends.llm.base import LLMBackend
from plauder.backends.llm.openai_compat import OpenAICompatLLMBackend
from plauder.backends.llm.openclaw import OpenClawLLMBackend


class _FakeResp:
    def __init__(self, status, payload):
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        return str(self._payload)


class _FakeSession:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status
        self.calls = []

    def post(self, url, headers=None, json=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return _FakeResp(self.status, self.payload)


def _ok(text="Hallo, ich bin Antonia."):
    return {
        "id": "cmpl-1",
        "choices": [{"message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"total_tokens": 12},
    }


class _CompatCfg:
    llm_backend = "openai_compat"
    llm_api_key = "fw-test"
    llm_base_url = "https://api.fireworks.ai/inference/v1"
    llm_model = "accounts/fireworks/models/glm-5p2"
    llm_max_tokens = 256
    llm_timeout = 10


class _ClawCfg:
    llm_backend = "openclaw"
    openclaw_gateway_url = "http://localhost:8080"
    openclaw_gateway_token = "tok"
    openclaw_agent_id = "antonia"
    openclaw_user_id = "voice-user"
    llm_max_tokens = 256
    llm_timeout = 10
    llm_model = "antonia"


def _compat(payload, status=200):
    b = OpenAICompatLLMBackend.from_config(_CompatCfg())
    b._session = _FakeSession(payload, status)
    return b


def _claw(payload, status=200):
    b = OpenClawLLMBackend.from_config(_ClawCfg())
    b._session = _FakeSession(payload, status)
    return b


# --- factory ----------------------------------------------------------------
def test_factory_selects_compat():
    assert isinstance(LLMBackend.from_config(_CompatCfg()), OpenAICompatLLMBackend)


def test_factory_selects_openclaw():
    assert isinstance(LLMBackend.from_config(_ClawCfg()), OpenClawLLMBackend)


# --- openai_compat ----------------------------------------------------------
def test_compat_returns_text_and_meta():
    b = _compat(_ok("Antwort A"))
    text = asyncio.run(b.chat([{"role": "user", "content": "Hi"}], system_hint="du bist Antonia"))
    assert text == "Antwort A"
    assert b.last_meta["finish_reason"] == "stop"
    assert b.last_meta["usage"]["total_tokens"] == 12


def test_compat_url_single_v1():
    b = _compat(_ok())
    asyncio.run(b.chat([{"role": "user", "content": "Hi"}]))
    url = b._session.calls[0]["url"]
    assert url == "https://api.fireworks.ai/inference/v1/chat/completions"
    assert "/v1/v1/" not in url


def test_compat_prepends_system_hint():
    b = _compat(_ok())
    asyncio.run(b.chat([{"role": "user", "content": "Frage?"}], system_hint="SYSTEM"))
    body = b._session.calls[0]["json"]
    assert body["messages"][0] == {"role": "system", "content": "SYSTEM"}
    assert body["messages"][-1]["content"] == "Frage?"
    assert body["model"] == "accounts/fireworks/models/glm-5p2"
    assert b._session.calls[0]["headers"]["Authorization"] == "Bearer fw-test"


def test_compat_passes_full_message_list():
    b = _compat(_ok())
    msgs = [
        {"role": "user", "content": "eins"},
        {"role": "assistant", "content": "antwort eins"},
        {"role": "user", "content": "zwei"},
    ]
    asyncio.run(b.chat(msgs))
    contents = [m.get("content") for m in b._session.calls[0]["json"]["messages"]]
    assert contents == ["eins", "antwort eins", "zwei"]


def test_compat_raises_on_http_error():
    b = _compat({"error": "boom"}, status=500)
    with pytest.raises(RuntimeError):
        asyncio.run(b.chat([{"role": "user", "content": "Hi"}]))


def test_compat_raises_upstream_timeout_on_408():
    b = _compat({"error": "upstream provider timeout"}, status=408)
    with pytest.raises(UpstreamTimeoutError):
        asyncio.run(b.chat([{"role": "user", "content": "Hi"}]))


def test_compat_requires_load():
    b = OpenAICompatLLMBackend.from_config(_CompatCfg())
    b._session = None
    with pytest.raises(RuntimeError):
        asyncio.run(b.chat([{"role": "user", "content": "x"}]))


# --- openclaw ---------------------------------------------------------------
def test_openclaw_returns_text():
    b = _claw(_ok("claw antwort"))
    text = asyncio.run(b.chat([{"role": "user", "content": "Hi"}]))
    assert text == "claw antwort"


def test_openclaw_sends_session_key_as_user():
    b = _claw(_ok())
    asyncio.run(b.chat([{"role": "user", "content": "Hi"}]))
    body = b._session.calls[0]["json"]
    assert body["user"] == "agent:antonia:openai-user:voice-user"
    assert b._session.calls[0]["url"] == "http://localhost:8080/v1/chat/completions"


def test_openclaw_upstream_timeout():
    b = _claw("upstream provider timeout", status=408)
    with pytest.raises(UpstreamTimeoutError):
        asyncio.run(b.chat([{"role": "user", "content": "Hi"}]))


# --- Streaming (chat_stream, SSE) -------------------------------------------
class _FakeContent:
    """Async-iterable SSE body: yields lines as bytes (like aiohttp)."""
    def __init__(self, lines):
        self._lines = [(l + "\n").encode("utf-8") for l in lines]

    def __aiter__(self):
        async def gen():
            for l in self._lines:
                yield l
        return gen()


class _FakeStreamResp:
    def __init__(self, status, lines, text=""):
        self.status = status
        self.content = _FakeContent(lines)
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def text(self):
        return self._text


class _FakeStreamSession:
    def __init__(self, lines, status=200, text=""):
        self.lines = lines
        self.status = status
        self.text = text
        self.calls = []

    def post(self, url, headers=None, json=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return _FakeStreamResp(self.status, self.lines, self.text)


def _collect(agen):
    async def run():
        out = []
        async for d in agen:
            out.append(d)
        return out
    return asyncio.run(run())


_SSE_OK = [
    'data: {"id":"cmpl-9","choices":[{"delta":{"content":"Hallo"}}]}',
    'data: {"choices":[{"delta":{"content":" Welt"}}]}',
    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
    'data: {"usage":{"total_tokens":7},"choices":[]}',
    "data: [DONE]",
]


def test_compat_chat_stream_yields_deltas_and_meta():
    b = OpenAICompatLLMBackend.from_config(_CompatCfg())
    b._session = _FakeStreamSession(_SSE_OK)
    deltas = _collect(b.chat_stream([{"role": "user", "content": "Hi"}], system_hint="S"))
    assert deltas == ["Hallo", " Welt"]
    assert "".join(deltas) == "Hallo Welt"
    assert b.last_meta["finish_reason"] == "stop"
    assert b.last_meta["usage"]["total_tokens"] == 7
    body = b._session.calls[0]["json"]
    assert body["stream"] is True


def test_compat_chat_stream_raises_upstream_timeout():
    b = OpenAICompatLLMBackend.from_config(_CompatCfg())
    b._session = _FakeStreamSession([], status=408, text="upstream provider timeout")
    with pytest.raises(UpstreamTimeoutError):
        _collect(b.chat_stream([{"role": "user", "content": "Hi"}]))


def test_openclaw_chat_stream_yields_deltas():
    b = OpenClawLLMBackend.from_config(_ClawCfg())
    b._session = _FakeStreamSession(_SSE_OK)
    deltas = _collect(b.chat_stream([{"role": "user", "content": "Hi"}]))
    assert "".join(deltas) == "Hallo Welt"
    assert b._session.calls[0]["json"]["stream"] is True
    assert b._session.calls[0]["json"]["user"] == "agent:antonia:openai-user:voice-user"


# --------------------------------------------------------------------------- #
# Per-turn hint injection (gateway drops client system messages)
# --------------------------------------------------------------------------- #
def _hint_backend(hint="[Hinweis]"):
    b = OpenAICompatLLMBackend(api_key="k", base_url="http://x/v1", model="m")
    b.turn_hint = hint
    return b


def test_turn_hint_appended_to_last_user_message():
    b = _hint_backend()
    history = [
        {"role": "user", "content": "erste Frage"},
        {"role": "assistant", "content": "Antwort"},
        {"role": "user", "content": "zweite Frage"},
    ]
    out = b._inject_turn_hint(history)
    assert out[-1]["content"] == "zweite Frage\n\n[Hinweis]"
    assert out[0]["content"] == "erste Frage"  # only the LAST user turn
    # caller's history must stay untouched (no accumulation across turns)
    assert history[-1]["content"] == "zweite Frage"


def test_turn_hint_multimodal_content_gets_text_part():
    b = _hint_backend()
    history = [{"role": "user", "content": [
        {"type": "text", "text": "schau mal"},
        {"type": "image_url", "image_url": {"url": "data:..."}},
    ]}]
    out = b._inject_turn_hint(history)
    assert out[0]["content"][-1] == {"type": "text", "text": "[Hinweis]"}
    assert len(history[0]["content"]) == 2  # original unchanged


def test_turn_hint_empty_or_no_user_message_is_noop():
    b = _hint_backend(hint="")
    history = [{"role": "user", "content": "hi"}]
    assert b._inject_turn_hint(history) is history
    b2 = _hint_backend()
    only_assistant = [{"role": "assistant", "content": "hi"}]
    assert b2._inject_turn_hint(only_assistant) == only_assistant


def test_turn_hint_lands_in_request_body():
    b = _hint_backend()
    _url, _headers, body = b._build_request(
        [{"role": "user", "content": "hi"}], None, stream=False)
    assert body["messages"][-1]["content"].endswith("[Hinweis]")


def test_config_resolved_voice_turn_hint(monkeypatch):
    from plauder.config import Config
    cfg = Config.from_env()
    # language default is non-empty and bracketed
    assert cfg.resolved_voice_turn_hint().startswith("[")
    # explicit override wins
    object.__setattr__(cfg, "voice_turn_hint", "custom")
    assert cfg.resolved_voice_turn_hint() == "custom"
    # "-" disables
    object.__setattr__(cfg, "voice_turn_hint", "-")
    assert cfg.resolved_voice_turn_hint() == ""
