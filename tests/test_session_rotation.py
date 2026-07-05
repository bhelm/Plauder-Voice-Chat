"""Tests for the persisted Hermes session-ID rotation (turn_state).

Regression for: "New session" only cleared the UI — the rotation lived in the
per-connection TurnState, so any WS reconnect or page reload re-derived the
stable key-hash ID and silently continued the OLD Hermes session.
"""
import hashlib

from plauder.turn_state import (
    TurnState,
    current_hermes_session_id,
    rotate_hermes_session_id,
)

KEY = "agent:antonia:test-key"
KEY_HASH = hashlib.sha256(KEY.encode()).hexdigest()


def _use_state_file(monkeypatch, tmp_path, *, key=KEY):
    monkeypatch.setenv("HERMES_SESSION_KEY_SEPARATE", key)
    state_file = tmp_path / "hermes_session_id"
    monkeypatch.setenv("HERMES_SESSION_STATE_PATH", str(state_file))
    return state_file


def test_without_key_random_per_connection(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_SESSION_KEY_SEPARATE", raising=False)
    monkeypatch.setenv("HERMES_SESSION_STATE_PATH", str(tmp_path / "sid"))
    assert current_hermes_session_id() != current_hermes_session_id()


def test_with_key_derives_stable_id(monkeypatch, tmp_path):
    _use_state_file(monkeypatch, tmp_path)
    assert current_hermes_session_id() == KEY_HASH
    assert current_hermes_session_id() == KEY_HASH


def test_rotation_persists_across_new_connections(monkeypatch, tmp_path):
    state_file = _use_state_file(monkeypatch, tmp_path)
    assert TurnState().hermes_session_id_separate == KEY_HASH

    rotated = rotate_hermes_session_id()
    assert rotated != KEY_HASH
    assert state_file.read_text().strip() == rotated
    # A fresh connection (new TurnState = reconnect/reload) picks up the
    # rotated ID instead of snapping back to the pre-reset hash.
    assert TurnState().hermes_session_id_separate == rotated

    # A second reset rotates again and wins over the first rotation.
    rotated2 = rotate_hermes_session_id()
    assert rotated2 != rotated
    assert TurnState().hermes_session_id_separate == rotated2


def test_garbage_state_file_falls_back_to_derived_id(monkeypatch, tmp_path):
    state_file = _use_state_file(monkeypatch, tmp_path)
    state_file.write_text("../../etc/passwd\r\nX-Injected: 1")
    assert current_hermes_session_id() == KEY_HASH


def test_rotation_without_key_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_SESSION_KEY_SEPARATE", raising=False)
    state_file = tmp_path / "sid"
    monkeypatch.setenv("HERMES_SESSION_STATE_PATH", str(state_file))
    sid = rotate_hermes_session_id()
    assert sid
    assert not state_file.exists()


def test_apply_headers_picks_up_rotation_immediately(monkeypatch, tmp_path):
    """A reset on one device must apply to OTHER live connections' next LLM
    call at once: _apply_hermes_headers re-reads the persisted ID per call."""
    from types import SimpleNamespace
    from plauder import server as srv

    _use_state_file(monkeypatch, tmp_path)
    llm = SimpleNamespace(session_key="", session_id="")
    monkeypatch.setattr(srv, "CONV", SimpleNamespace(llm=llm))
    monkeypatch.setattr(srv, "CFG", SimpleNamespace(hermes_session_key_separate=KEY))

    state = TurnState()   # connection opened before the reset
    assert state.hermes_session_id_separate == KEY_HASH
    rotated = rotate_hermes_session_id()   # reset happens on another device

    srv._apply_hermes_headers(state)
    assert llm.session_id == rotated
    assert state.hermes_session_id_separate == rotated
