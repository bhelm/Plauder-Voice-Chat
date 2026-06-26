"""LLM-Backend-Abstraktion."""

from __future__ import annotations

import abc


class LLMBackend(abc.ABC):
    """Chat-Completion-Backend. Der Gesprächsverlauf wird NICHT hier gehalten
    (das macht plauder.session.Session); ``chat`` bekommt die fertige
    Nachrichtenliste und gibt nur den Antworttext zurück.
    """

    #: Metadaten des letzten chat()-Calls (finish_reason, usage, id). Direkt
    #: nach ``await chat()`` lesen (race-frei dank kooperativem Scheduling).
    last_meta: dict

    @abc.abstractmethod
    async def load(self) -> None:
        """Initialisiert das Backend (HTTP-Session/Client)."""

    @abc.abstractmethod
    async def chat(self, messages: list[dict], system_hint: str | None = None) -> str:
        """Schickt ``messages`` (+ optionalen System-Hint) an das Modell und
        gibt den Antworttext zurück. Wirft UpstreamTimeoutError bei 408."""

    async def chat_stream(self, messages: list[dict], system_hint: str | None = None):
        """Async-Generator: liefert Antwort-Token-Deltas, sobald sie eintreffen.

        Setzt ``last_meta`` (finish_reason/usage/id) nach dem letzten Delta.
        Default-Fallback für Backends ohne echtes Streaming: ein einzelnes Delta
        mit dem vollständigen Text. Wirft UpstreamTimeoutError bei 408.
        """
        text = await self.chat(messages, system_hint=system_hint)
        if text:
            yield text

    async def close(self) -> None:  # pragma: no cover - optional
        """Räumt Ressourcen auf (HTTP-Session schließen)."""

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
        raise BackendError(f"Unbekanntes LLM_BACKEND: {name!r}")
