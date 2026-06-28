"""LLM backend abstraction."""

from __future__ import annotations

import abc


class LLMBackend(abc.ABC):
    """Chat completion backend. The conversation history is NOT kept here
    (that is done by plauder.session.Session); ``chat`` receives the finished
    message list and returns only the reply text.
    """

    #: Metadata of the last chat() call (finish_reason, usage, id). Read
    #: directly after ``await chat()`` (race-free thanks to cooperative scheduling).
    last_meta: dict

    @abc.abstractmethod
    async def load(self) -> None:
        """Initializes the backend (HTTP session/client)."""

    @abc.abstractmethod
    async def chat(self, messages: list[dict], system_hint: str | None = None) -> str:
        """Sends ``messages`` (+ optional system hint) to the model and
        returns the reply text. Raises UpstreamTimeoutError on 408."""

    async def chat_stream(self, messages: list[dict], system_hint: str | None = None):
        """Async generator: yields reply token deltas as they arrive.

        Sets ``last_meta`` (finish_reason/usage/id) after the last delta.
        Default fallback for backends without real streaming: a single delta
        with the full text. Raises UpstreamTimeoutError on 408.
        """
        text = await self.chat(messages, system_hint=system_hint)
        if text:
            yield text

    async def close(self) -> None:  # pragma: no cover - optional
        """Cleans up resources (close HTTP session)."""

    def describe(self) -> dict:
        return {"engine": self.__class__.__name__}

    @staticmethod
    def from_config(cfg) -> "LLMBackend":
        name = cfg.llm_backend
        if name == "openai_compat":
            from .openai_compat import OpenAICompatLLMBackend
            return OpenAICompatLLMBackend.from_config(cfg)
        if name == "openclaw":
            from .openclaw import OpenClawLLMBackend
            return OpenClawLLMBackend.from_config(cfg)
        from ..base import BackendError
        raise BackendError(f"Unknown LLM_BACKEND: {name!r}")
