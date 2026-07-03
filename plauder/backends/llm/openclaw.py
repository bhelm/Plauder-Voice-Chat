"""LLM backend: OpenClaw gateway (legacy).

The OpenClaw gateway speaks an OpenAI-compatible /chat/completions endpoint
(openai-http adapter), but routes per agent/user via a session key. So it is a
thin subclass of OpenAICompatLLMBackend: same request/SSE handling, only the
endpoint suffix (``/v1`` prefix on the gateway), the ``user`` body field (the
derived session key) and the error label differ.

The history is sent along deliberately (the Session class manages it here).
Raises UpstreamTimeoutError on 408 so the server can use the retry path.
"""

from __future__ import annotations

from aiohttp import ClientSession, ClientTimeout

from .openai_compat import OpenAICompatLLMBackend


class OpenClawLLMBackend(OpenAICompatLLMBackend):
    _error_label = "OpenClaw"

    def __init__(self, *, gateway_url: str, token: str, agent_id: str = "antonia",
                 user_id: str = "voice-user", max_tokens: int = 4096,
                 timeout: int = 300, model: str | None = None):
        # The gateway exposes the OpenAI-compatible API under /v1, so the shared
        # endpoint builder (base_url + "/chat/completions") yields the right URL.
        super().__init__(
            api_key=token,
            base_url=f"{gateway_url.rstrip('/')}/v1",
            model=model or agent_id,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        self.gateway_url = gateway_url.rstrip("/")
        self.token = token
        self.agent_id = agent_id
        self.user_id = user_id
        # Default the (settable) Hermes session_key to the derived routing key, so
        # the gateway routes correctly even when nothing overrides it. The gateway
        # actually routes on the ``user`` body field — always the derived
        # route_key (see _extra_body) — so a Hermes override of session_key only
        # affects the X-Hermes-Session-Key header, not routing.
        self.session_key = self.route_key

    @property
    def route_key(self) -> str:
        """Derived agent/user routing key (the gateway's ``user`` field)."""
        return f"agent:{self.agent_id}:openai-user:{self.user_id}"

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

    async def load(self) -> None:
        if not self.token:
            from ..base import BackendError
            raise BackendError("openclaw needs OPENCLAW_GATEWAY_TOKEN.")
        self._session = ClientSession(timeout=ClientTimeout(total=self.timeout))

    def describe(self) -> dict:
        return {
            "engine": "openclaw",
            "gateway": self.gateway_url,
            "agent_id": self.agent_id,
            "session_key": self.route_key,
            "ready": self.loaded,
        }

    def _extra_body(self, *, stream: bool) -> dict:
        # The gateway routes by the OpenAI ``user`` field (no stream_options).
        return {"user": self.route_key}
