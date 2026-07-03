"""Voice session state: conversation history per session key + LLM orchestration.

The LLM backends are stateless (they receive the finished message list).
This class holds the history — kept separate per ``user_key`` so that the "New
Session" button (rotates the key) gets fresh context without disturbing other
connections.
"""

from __future__ import annotations


class ConversationManager:
    def __init__(self, llm, *, system_prompt: str = "", history_turns: int = 20):
        self.llm = llm
        self.system_prompt = system_prompt
        self.history_turns = max(1, int(history_turns))
        self._history: dict[str, list[dict]] = {}
        #: Meta (finish_reason/usage/id) of the last chat_stream() run.
        self.last_stream_meta: dict = {}

    def reset(self, user_key: str) -> None:
        """Discards the history of a session key ("New Session")."""
        self._history.pop(user_key, None)

    def history_for(self, user_key: str) -> list[dict]:
        return self._history.setdefault(user_key, [])

    def seed_history(self, user_key: str, messages: list[dict]) -> None:
        """Pre-fill the local history from an external source (e.g. Hermes
        backend) — but only when the local history for this key is still empty.
        Respects ``history_turns`` to cap the seeded context window."""
        if self._history.get(user_key):
            return  # already has local turns → don't overwrite
        max_msgs = self.history_turns * 2
        trimmed = messages[-max_msgs:] if len(messages) > max_msgs else list(messages)
        self._history[user_key] = trimmed

    @staticmethod
    def _build_user_message(user_text: str, image_urls=None) -> dict:
        if image_urls:
            content = [{"type": "text", "text": user_text or ""}]
            for u in image_urls:
                content.append({"type": "image_url", "image_url": {"url": u}})
            return {"role": "user", "content": content}
        return {"role": "user", "content": user_text}

    async def chat(self, user_text: str, *, user_key: str, image_urls=None):
        """Appends the user message to the history, calls the LLM, records the
        reply. Returns (reply_text, meta).
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
        """Like ``chat``, but as an async generator: yields reply deltas live.

        Appends user message + full reply to the history only AFTER the last
        delta. ``last_stream_meta`` then contains finish_reason/usage.
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
            # Backend without streaming → once in full, as a single delta.
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
