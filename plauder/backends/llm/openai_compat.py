"""LLM backend: OpenAI-compatible /chat/completions (Fireworks, OpenAI, …).

Stateless: receives the finished message list (the Session handles history) and
returns the reply text. Raises UpstreamTimeoutError on 408.
"""

from __future__ import annotations

import json

from aiohttp import ClientSession, ClientTimeout

from ..base import UpstreamTimeoutError
from .base import LLMBackend


class OpenAICompatLLMBackend(LLMBackend):
    def __init__(self, *, api_key: str, base_url: str, model: str,
                 max_tokens: int = 4096, timeout: int = 300):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._session: ClientSession | None = None
        self.last_meta: dict = {}

    @classmethod
    def from_config(cls, cfg) -> "OpenAICompatLLMBackend":
        return cls(
            api_key=cfg.llm_api_key,
            base_url=cfg.llm_base_url,
            model=cfg.llm_model,
            max_tokens=cfg.llm_max_tokens,
            timeout=cfg.llm_timeout,
        )

    async def load(self) -> None:
        if not self.api_key:
            from ..base import BackendError
            raise BackendError(
                "openai_compat needs an API key (LLM_API_KEY / FIREWORKS_API_KEY).")
        self._session = ClientSession(timeout=ClientTimeout(total=self.timeout))

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

    async def chat(self, messages: list[dict], system_hint: str | None = None) -> str:
        if self._session is None:
            raise RuntimeError("LLM not initialized (load() did not run)")
        # base_url ends in .../v1 → only append the endpoint (no /v1/v1).
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        full_messages: list[dict] = []
        if system_hint:
            full_messages.append({"role": "system", "content": system_hint})
        full_messages.extend(messages)
        body = {
            "model": self.model,
            "messages": full_messages,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        async with self._session.post(url, headers=headers, json=body) as resp:
            if resp.status != 200:
                err = await resp.text()
                if resp.status == 408 or "upstream provider timeout" in err.lower():
                    raise UpstreamTimeoutError(f"LLM HTTP {resp.status}: {err[:300]}")
                raise RuntimeError(f"LLM HTTP {resp.status}: {err[:300]}")
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
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        full_messages: list[dict] = []
        if system_hint:
            full_messages.append({"role": "system", "content": system_hint})
        full_messages.extend(messages)
        body = {
            "model": self.model,
            "messages": full_messages,
            "max_tokens": self.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        finish_reason = None
        usage: dict = {}
        resp_id = None
        async with self._session.post(url, headers=headers, json=body) as resp:
            if resp.status != 200:
                err = await resp.text()
                if resp.status == 408 or "upstream provider timeout" in err.lower():
                    raise UpstreamTimeoutError(f"LLM HTTP {resp.status}: {err[:300]}")
                raise RuntimeError(f"LLM HTTP {resp.status}: {err[:300]}")
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
