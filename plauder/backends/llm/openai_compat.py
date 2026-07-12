"""LLM backend: OpenAI-compatible /chat/completions (Fireworks, OpenAI, …).

Stateless: receives the finished message list (the Session handles history) and
returns the reply text. Raises UpstreamTimeoutError on 408.

The OpenClaw gateway backend subclasses this and only overrides the few hooks
that differ (endpoint suffix, error label, the extra ``user`` body field).
"""

from __future__ import annotations

import json

from aiohttp import ClientSession, ClientTimeout

from ..base import UpstreamTimeoutError
from .base import LLMBackend


class OpenAICompatLLMBackend(LLMBackend):
    #: Prefix used in HTTP-error messages (subclasses override).
    _error_label = "LLM"

    def __init__(self, *, api_key: str, base_url: str, model: str,
                 max_tokens: int = 4096, timeout: int = 300,
                 session_key: str = "", session_id: str = ""):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.session_key = session_key
        self.session_id = session_id
        # Per-turn voice rule line (set via from_config; see config.py).
        self.turn_hint = ""
        self._session: ClientSession | None = None
        self.last_meta: dict = {}

    @classmethod
    def from_config(cls, cfg) -> "OpenAICompatLLMBackend":
        backend = cls(
            api_key=cfg.llm_api_key,
            base_url=cfg.llm_base_url,
            model=cfg.llm_model,
            max_tokens=cfg.llm_max_tokens,
            timeout=cfg.llm_timeout,
        )
        # getattr: test doubles are duck-typed and may omit the method.
        _hint = getattr(cfg, "resolved_voice_turn_hint", None)
        backend.turn_hint = _hint() if callable(_hint) else ""
        return backend

    async def load(self) -> None:
        if not self.api_key:
            from ..base import BackendError
            raise BackendError(
                "openai_compat needs an API key (LLM_API_KEY / FIREWORKS_API_KEY).")
        # sock_connect: a dead/unreachable upstream must fail fast (and hit the
        # silent retry) instead of stalling the turn for the full total budget.
        self._session = ClientSession(
            timeout=ClientTimeout(total=self.timeout, sock_connect=10))

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def describe(self) -> dict:
        return {
            "engine": "openai_compat",
            "base_url": self.base_url,
            "model": self.model,
            "ready": self.loaded,
        }

    # --- request-shaping hooks (subclasses override) ----------------------- #
    @property
    def _endpoint_url(self) -> str:
        # base_url ends in .../v1 → only append the endpoint (no /v1/v1).
        return f"{self.base_url}/chat/completions"

    @property
    def _auth_token(self) -> str:
        return self.api_key

    def _extra_body(self, *, stream: bool) -> dict:
        """Provider-specific body fields. Cloud OpenAI-compatible endpoints want
        usage in the stream; subclasses (OpenClaw) inject a ``user`` instead."""
        return {"stream_options": {"include_usage": True}} if stream else {}

    def _inject_turn_hint(self, messages: list[dict]) -> list[dict]:
        """Append ``self.turn_hint`` to the LAST user message.

        The Hermes gateway rebuilds its own system prompt and drops client
        system messages, so voice-mode rules only reach the model inside a
        user turn (which also gives them recency). Copies the affected
        message — the caller's history is never mutated, so hints cannot
        accumulate across turns.
        """
        hint = getattr(self, "turn_hint", "") or ""
        if not hint:
            return messages
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") != "user":
                continue
            msg = dict(messages[i])
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = f"{content}\n\n{hint}"
            elif isinstance(content, list):
                msg["content"] = list(content) + [{"type": "text", "text": hint}]
            else:
                return messages
            return messages[:i] + [msg] + messages[i + 1:]
        return messages

    def _build_request(self, messages: list[dict], system_hint: str | None,
                       *, stream: bool) -> tuple[str, dict, dict]:
        full_messages: list[dict] = []
        if system_hint:
            full_messages.append({"role": "system", "content": system_hint})
        full_messages.extend(self._inject_turn_hint(messages))
        headers = {
            "Authorization": f"Bearer {self._auth_token}",
            "Content-Type": "application/json",
        }
        if self.session_key:
            headers["X-Hermes-Session-Key"] = self.session_key
        if self.session_id:
            headers["X-Hermes-Session-Id"] = self.session_id
        body = {
            "model": self.model,
            "messages": full_messages,
            "max_tokens": self.max_tokens,
            "stream": stream,
            **self._extra_body(stream=stream),
        }
        return self._endpoint_url, headers, body

    def _raise_for_status(self, status: int, err: str) -> None:
        if status == 408 or "upstream provider timeout" in err.lower():
            raise UpstreamTimeoutError(f"{self._error_label} HTTP {status}: {err[:300]}")
        raise RuntimeError(f"{self._error_label} HTTP {status}: {err[:300]}")

    # --- chat -------------------------------------------------------------- #
    async def chat(self, messages: list[dict], system_hint: str | None = None) -> str:
        if self._session is None:
            raise RuntimeError("LLM not initialized (load() did not run)")
        url, headers, body = self._build_request(messages, system_hint, stream=False)
        async with self._session.post(url, headers=headers, json=body) as resp:
            if resp.status != 200:
                self._raise_for_status(resp.status, await resp.text())
            data = await resp.json()
        choice = (data.get("choices") or [{}])[0]
        text = ((choice.get("message") or {}).get("content") or "").strip()
        self.last_meta = {
            "id": data.get("id"),
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage") or {},
        }
        return text

    async def chat_stream(self, messages: list[dict], system_hint: str | None = None):
        if self._session is None:
            raise RuntimeError("LLM not initialized (load() did not run)")
        url, headers, body = self._build_request(messages, system_hint, stream=True)
        finish_reason = None
        usage: dict = {}
        resp_id = None
        async with self._session.post(url, headers=headers, json=body) as resp:
            if resp.status != 200:
                self._raise_for_status(resp.status, await resp.text())
            async for raw in resp.content:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if obj.get("id"):
                    resp_id = obj["id"]
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = ((choice.get("delta") or {}).get("content")) or ""
                if delta:
                    yield delta
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
        self.last_meta = {"id": resp_id, "finish_reason": finish_reason, "usage": usage}
