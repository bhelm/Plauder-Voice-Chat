"""Voice-Session-State: Gesprächsverlauf pro Session-Key + LLM-Orchestrierung.

Die LLM-Backends sind stateless (sie bekommen die fertige Nachrichtenliste).
Den Verlauf hält diese Klasse — pro ``user_key`` getrennt, damit der „Neue
Session"-Button (rotiert den Key) frischen Kontext bekommt, ohne andere
Connections zu stören.
"""

from __future__ import annotations


class ConversationManager:
    def __init__(self, llm, *, system_prompt: str = "", history_turns: int = 20):
        self.llm = llm
        self.system_prompt = system_prompt
        self.history_turns = max(1, int(history_turns))
        self._history: dict[str, list[dict]] = {}
        #: Meta (finish_reason/usage/id) des letzten chat_stream()-Durchlaufs.
        self.last_stream_meta: dict = {}

    def reset(self, user_key: str) -> None:
        """Verwirft den Verlauf eines Session-Keys ("Neue Session")."""
        self._history.pop(user_key, None)

    def history_for(self, user_key: str) -> list[dict]:
        return self._history.setdefault(user_key, [])

    @staticmethod
    def _build_user_message(user_text: str, image_urls=None) -> dict:
        if image_urls:
            content = [{"type": "text", "text": user_text or ""}]
            for u in image_urls:
                content.append({"type": "image_url", "image_url": {"url": u}})
            return {"role": "user", "content": content}
        return {"role": "user", "content": user_text}

    async def chat(self, user_text: str, *, user_key: str, image_urls=None):
        """Hängt die User-Message an den Verlauf, ruft das LLM, schreibt die
        Antwort fort. Gibt (reply_text, meta) zurück.
        """
        history = self.history_for(user_key)
        user_msg = self._build_user_message(user_text, image_urls)
        messages = list(history) + [user_msg]

        reply = await self.llm.chat(messages, system_hint=self.system_prompt)
        meta = dict(getattr(self.llm, "last_meta", {}) or {})

        if reply:
            history.append(user_msg)
            history.append({"role": "assistant", "content": reply})
            max_msgs = self.history_turns * 2
            if len(history) > max_msgs:
                del history[: len(history) - max_msgs]
        return reply, meta

    async def chat_stream(self, user_text: str, *, user_key: str, image_urls=None):
        """Wie ``chat``, aber als Async-Generator: liefert Antwort-Deltas live.

        Hängt User-Message + vollständige Antwort erst NACH dem letzten Delta an
        den Verlauf an. ``last_stream_meta`` enthält danach finish_reason/usage.
        """
        history = self.history_for(user_key)
        user_msg = self._build_user_message(user_text, image_urls)
        messages = list(history) + [user_msg]

        parts: list[str] = []
        llm_stream = getattr(self.llm, "chat_stream", None)
        if llm_stream is not None:
            async for delta in llm_stream(messages, system_hint=self.system_prompt):
                parts.append(delta)
                yield delta
        else:
            # Backend ohne Streaming → einmal komplett, als ein Delta.
            text = await self.llm.chat(messages, system_hint=self.system_prompt)
            if text:
                parts.append(text)
                yield text

        reply = "".join(parts).strip()
        self.last_stream_meta = dict(getattr(self.llm, "last_meta", {}) or {})
        if reply:
            history.append(user_msg)
            history.append({"role": "assistant", "content": reply})
            max_msgs = self.history_turns * 2
            if len(history) > max_msgs:
                del history[: len(history) - max_msgs]
