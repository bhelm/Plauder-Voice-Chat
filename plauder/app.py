"""Application boot & wiring: build the aiohttp app, load backends, run the
server process.

Kept separate from request handling. The route handlers and the runtime state
(CFG/STT/… set by ``configure``) live in ``plauder.server``; this module reads
them at call time via ``server.<name>`` to avoid an import cycle.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time

from aiohttp import web

from . import server
from .backends import LLMBackend, STTBackend, TTSBackend
from .config import SAMPLE_RATE, Config, load_config
from .images import UPLOAD_DIR, upload_image
from .session import ConversationManager

LOG = logging.getLogger("voice-chat")


@web.middleware
async def _security_headers_mw(request, handler):
    resp = await handler(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    return resp


def build_app() -> web.Application:
    app = web.Application(client_max_size=server.WS_MAX_MSG_BYTES,
                          middlewares=[_security_headers_mw])
    # Everything is mounted under BASE_PATH ('' = root, '/voice' = sub-path), so
    # the app works behind a reverse proxy that does NOT strip the prefix.
    base = server.CFG.base_path if server.CFG else ""

    def p(path: str) -> str:
        return f"{base}{path}"

    app.router.add_get(p("/"), server.index)
    if base:                       # also serve the prefix without a trailing slash
        app.router.add_get(base, server.index)
    app.router.add_get(p("/healthz"), server.healthz)
    app.router.add_get(p("/ws"), server.ws_handler)
    app.router.add_post(p("/upload"), upload_image)
    if server.STATIC_DIR.exists():
        app.router.add_static(p("/static/"), server.STATIC_DIR, show_index=False)
    app.router.add_static(p("/uploads/"), UPLOAD_DIR, show_index=False)
    return app


async def init_backends(cfg: Config):
    """Builds and loads the chosen backends. Raises on error (caller exits)."""
    stt = STTBackend.from_config(cfg)
    tts = TTSBackend.from_config(cfg)
    llm = LLMBackend.from_config(cfg)

    await stt.load()
    await tts.load()
    await llm.load()

    if cfg.stt_warmup:
        try:
            import numpy as np
            t0 = time.time()
            await stt.transcribe(np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32).tobytes(),
                                 SAMPLE_RATE)
            LOG.info("STT Warm-up: %.2fs", time.time() - t0)
        except Exception as exc:
            LOG.warning("STT Warm-up: %s", exc)
    if cfg.tts_warmup:
        try:
            t0 = time.time()
            await tts.synth("Hello.", speed=cfg.tts_speed)
            LOG.info("TTS Warm-up: %.2fs", time.time() - t0)
        except Exception as exc:
            LOG.warning("TTS Warm-up: %s", exc)

    conv = ConversationManager(llm, system_prompt=cfg.resolved_voice_system(),
                               history_turns=cfg.llm_history_turns)

    # Speaker lock (voice gate). Optional; fail-open: any load problem disables
    # the gate rather than blocking the mic.
    speaker = None
    if cfg.speaker_lock_enabled:
        from .speaker_verify import SpeakerVerifier
        speaker = SpeakerVerifier.from_config(cfg)
        try:
            speaker.load()
        except Exception as exc:
            LOG.warning("Speaker lock disabled (load failed): %s", exc)
            speaker = None

    return stt, tts, conv, speaker


async def main():
    cfg = load_config()
    logging.basicConfig(
        level=getattr(logging, cfg.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    try:
        cfg.validate()
    except Exception:
        LOG.exception("Configuration invalid")
        sys.exit(2)

    if cfg.house_mode:
        LOG.info("🏠 HOUSE_MODE active — speaker_id=%d wake_word=%d auth=%d",
                 cfg.house_speaker_id, cfg.house_wake_word, cfg.house_auth)

    try:
        stt, tts, conv, speaker = await init_backends(cfg)
    except Exception:
        LOG.exception("Backend initialization failed")
        sys.exit(3)

    bridge = None  # Telegram bridge is legacy/optional; off by default.

    server.configure(cfg, stt=stt, tts=tts, conv=conv, bridge=bridge, speaker=speaker)
    if speaker is not None:
        LOG.info("🔒 Speaker lock active (enrolled=%s, threshold=%.2f)",
                 speaker.has_profile(), speaker.threshold)

    LOG.info("STT-Backend: %s · %s", cfg.stt_backend, stt.describe())
    LOG.info("TTS-Backend: %s · %s", cfg.tts_backend, tts.describe())
    LOG.info("LLM-Backend: %s · %s", cfg.llm_backend, conv.llm.describe())

    app = build_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, cfg.host, cfg.port)
    await site.start()

    LOG.info("🎙️  Voice-Chat server running at http://%s:%s (agent: %s)",
             cfg.host, cfg.port, cfg.agent_name)
    LOG.info("    Debounce: %d ms · TTS speed: %.2f", cfg.debounce_ms, cfg.tts_speed)

    stop_event = asyncio.Event()

    def _request_stop(*_):
        LOG.info("Shutdown requested.")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            signal.signal(sig, _request_stop)
    await stop_event.wait()

    LOG.info("Stopping server …")
    if server.BRIDGE is not None:
        await server.BRIDGE.stop()
    await runner.cleanup()
    if conv is not None and hasattr(conv.llm, "close"):
        await conv.llm.close()


def run():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    run()
