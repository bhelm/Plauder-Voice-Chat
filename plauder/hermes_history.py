"""Fetch conversation history from the Hermes API server.

The Hermes gateway (api_server.py) persists every turn in a SQLite-backed
SessionDB. Each voice session is identified by a stable session ID derived from
HERMES_SESSION_KEY_SEPARATE. This module fetches the message history via
``GET /api/sessions/{session_id}/messages`` so the voice-chat can restore
context across devices/reconnects without local persistence.

Only user/assistant messages are returned (tool calls, system prompts etc. are
filtered out) — that is what the UI displays and the ConversationManager needs.
"""

from __future__ import annotations

import logging
from typing import Optional

import aiohttp

LOG = logging.getLogger("voice-chat")


async def fetch_history(
    *,
    base_url: str,
    api_key: str,
    session_id: str,
    max_messages: int = 40,
    timeout: int = 10,
) -> list[dict]:
    """Fetch user/assistant message history from the Hermes API server.

    Returns a list of ``{"role": "user"|"assistant", "content": "..."}`` dicts,
    oldest first, capped to the last *max_messages* chat turns.  Returns ``[]``
    on any error (network, auth, missing session) — never raises.
    """
    if not base_url or not api_key or not session_id:
        return []

    # base_url is e.g. "http://127.0.0.1:8642/v1" — strip the /v1 suffix to
    # reach the /api/ endpoints which sit at the server root.
    api_root = base_url.rstrip("/")
    if api_root.endswith("/v1"):
        api_root = api_root[:-3]

    url = f"{api_root}/api/sessions/{session_id}/messages"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    LOG.warning(
                        "hermes history: HTTP %d for session %s",
                        resp.status, session_id[:16],
                    )
                    return []
                data = await resp.json()
    except Exception as exc:
        LOG.warning("hermes history: fetch failed: %s", exc)
        return []

    # The API returns {"data": [...]} or {"messages": [...]}.
    raw = data.get("data") or data.get("messages") or []

    # Filter to user/assistant only (skip tool, system, etc.).
    chat = [
        {"role": m["role"], "content": m.get("content") or ""}
        for m in raw
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]

    # Cap to the last max_messages entries.
    if len(chat) > max_messages:
        chat = chat[-max_messages:]

    LOG.info(
        "hermes history: loaded %d messages for session %s",
        len(chat), session_id[:16],
    )
    return chat
