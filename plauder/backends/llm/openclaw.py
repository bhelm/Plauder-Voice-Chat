"""LLM backend: OpenClaw gateway (legacy).

The OpenClaw gateway speaks an OpenAI-compatible /chat/completions endpoint
(openai-http adapter), but routes per agent/user via a session key. We send
the history along (the session is deliberately managed by the Session class
here) and set the ``user`` field to the derived session key.

Raises UpstreamTimeoutError on 408 so the server can use the retry path.
"""

from __future__ import annotations

import json

from aiohttp import ClientSession, ClientTimeout

from ..base import UpstreamTimeoutError
from .base import LLMBackend


class OpenClawLLMBackend(LLMBackend):
    def __init__(self, *, gateway_url: str, token: str, agent_id: str = "antonia",
                 user_id: str = "voice-user", max_tokens: int = 4096,
                 timeout: int = 300, model: str | None = None):
        self.gateway_url = gateway_url.rstrip("/")
        self.token = token
        self.agent_id = agent_id
        self.user_id = user_id
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.model = model or agent_id
        self._session: ClientSession | None = None
        self.last_meta: dict = {}

    @classmethod
    def from_config(cls, cfg) -> "OpenClawLLMBackend":
        return cls(
            gateway_url=cfg.openclaw_gateway_url,
            token=cfg.openclaw_gateway_token,
            agent_id=cfg.openclaw_agent_id,
            user_id=cfg.openclaw_user_id,
            max_tokens=cfg.llm_max_tokens,
            timeout=cfg.llm_timeout,
            model=cfg.llm_model,
        )

    @property
    def session_key(self) -> str:
        return f"agent:{self.agent_id}:openai-user:{self.user_id}"

    async def load(self) -> None:
        if not self.token:
            from ..base import BackendError
            raise BackendError("openclaw needs OPENCLAW_GATEWAY_TOKEN.")
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
            "engine": "openclaw",
            "gateway": self.gateway_url,
            "agent_id": self.agent_id,
            "session_key": self.session_key,
            "ready": self.loaded,
        }

    async def chat(self, messages: list[dict], system_hint: str | None = None) -> str:
        if self._session is None:
            raise RuntimeError("LLM not initialized (load() did not run)")
        url = f"{self.gateway_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.token}",
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
            "user": self.session_key,
        }
        async with self._session.post(url, headers=headers, json=body) as resp:
            if resp.status != 200:
                err = await resp.text()
                if resp.status == 408 or "upstream provider timeout" in err.lower():
                    raise UpstreamTimeoutError(f"OpenClaw HTTP {resp.status}: {err[:300]}")
                raise RuntimeError(f"OpenClaw HTTP {resp.status}: {err[:300]}")
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
        url = f"{self.gateway_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.token}",
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
            "user": self.session_key,
        }
        finish_reason = None
        usage: dict = {}
        resp_id = None
        async with self._session.post(url, headers=headers, json=body) as resp:
            if resp.status != 200:
                err = await resp.text()
                if resp.status == 408 or "upstream provider timeout" in err.lower():
                    raise UpstreamTimeoutError(f"OpenClaw HTTP {resp.status}: {err[:300]}")
                raise RuntimeError(f"OpenClaw HTTP {resp.status}: {err[:300]}")
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
