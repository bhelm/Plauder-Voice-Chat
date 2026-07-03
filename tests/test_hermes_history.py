"""Tests for plauder.hermes_history (Hermes backend history retrieval)."""
import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from plauder.hermes_history import fetch_history


# --------------------------------------------------------------------------- #
# Helpers: tiny aiohttp server that mocks the Hermes /api/sessions/{id}/messages
# --------------------------------------------------------------------------- #
def _mock_hermes_app(messages, *, status=200, require_auth="test-key"):
    """Returns an aiohttp app that serves a single /api/sessions/{id}/messages."""
    async def handler(request):
        if require_auth:
            auth = request.headers.get("Authorization", "")
            if auth != f"Bearer {require_auth}":
                return web.json_response(
                    {"error": "unauthorized"}, status=401)
        return web.json_response({"data": messages}, status=status)

    app = web.Application()
    app.router.add_get("/api/sessions/{session_id}/messages", handler)
    return app


SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "I am fine."},
    {"role": "tool", "content": '{"result": 42}', "tool_call_id": "tc_1"},
]


def test_fetch_history_filters_user_assistant_only():
    """Only user/assistant messages are returned; system/tool are filtered."""
    async def run():
        app = _mock_hermes_app(SAMPLE_MESSAGES)
        async with TestServer(app) as server:
            base = f"http://localhost:{server.port}"
            result = await fetch_history(
                base_url=base, api_key="test-key",
                session_id="test-session-123",
            )
        assert len(result) == 4
        assert all(m["role"] in ("user", "assistant") for m in result)
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[-1] == {"role": "assistant", "content": "I am fine."}

    asyncio.run(run())


def test_fetch_history_respects_max_messages():
    """max_messages caps the returned list (tail)."""
    async def run():
        app = _mock_hermes_app(SAMPLE_MESSAGES)
        async with TestServer(app) as server:
            base = f"http://localhost:{server.port}"
            result = await fetch_history(
                base_url=base, api_key="test-key",
                session_id="s1", max_messages=2,
            )
        assert len(result) == 2
        assert result[0]["content"] == "How are you?"
        assert result[1]["content"] == "I am fine."

    asyncio.run(run())


def test_fetch_history_strips_v1_suffix():
    """base_url ending in /v1 (like LLM_BASE_URL) is handled correctly."""
    async def run():
        app = _mock_hermes_app(SAMPLE_MESSAGES)
        async with TestServer(app) as server:
            base = f"http://localhost:{server.port}/v1"
            result = await fetch_history(
                base_url=base, api_key="test-key",
                session_id="s1",
            )
        assert len(result) == 4

    asyncio.run(run())


def test_fetch_history_returns_empty_on_http_error():
    """Non-200 response → empty list, no exception."""
    async def run():
        app = _mock_hermes_app([], status=404)
        async with TestServer(app) as server:
            base = f"http://localhost:{server.port}"
            result = await fetch_history(
                base_url=base, api_key="test-key",
                session_id="nonexistent",
            )
        assert result == []

    asyncio.run(run())


def test_fetch_history_returns_empty_on_auth_failure():
    """Wrong API key → 401 → empty list."""
    async def run():
        app = _mock_hermes_app(SAMPLE_MESSAGES)
        async with TestServer(app) as server:
            base = f"http://localhost:{server.port}"
            result = await fetch_history(
                base_url=base, api_key="wrong-key",
                session_id="s1",
            )
        assert result == []

    asyncio.run(run())


def test_fetch_history_returns_empty_on_missing_params():
    """Missing base_url/api_key/session_id → empty list immediately."""
    async def run():
        assert await fetch_history(base_url="", api_key="k", session_id="s") == []
        assert await fetch_history(base_url="http://x", api_key="", session_id="s") == []
        assert await fetch_history(base_url="http://x", api_key="k", session_id="") == []

    asyncio.run(run())


def test_fetch_history_returns_empty_on_network_error():
    """Unreachable host → empty list, no exception."""
    async def run():
        result = await fetch_history(
            base_url="http://127.0.0.1:1",  # nothing listens here
            api_key="k", session_id="s", timeout=1,
        )
        assert result == []

    asyncio.run(run())


def test_fetch_history_handles_messages_key():
    """The API may return {'messages': [...]} instead of {'data': [...]}."""
    async def handler(request):
        return web.json_response({
            "session_id": "s1",
            "messages": [
                {"role": "user", "content": "A"},
                {"role": "assistant", "content": "B"},
            ],
        })

    async def run():
        app = web.Application()
        app.router.add_get("/api/sessions/{session_id}/messages", handler)
        async with TestServer(app) as server:
            base = f"http://localhost:{server.port}"
            result = await fetch_history(
                base_url=base, api_key="test-key",
                session_id="s1",
            )
        assert len(result) == 2
        assert result[0]["content"] == "A"

    asyncio.run(run())
