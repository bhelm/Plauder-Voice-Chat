#!/usr/bin/env python3
"""Entrypoint shim.

The monolithic server was split into the ``plauder`` package. This file stays
as an entry point so ``python server.py`` (and start.sh) keep working
unchanged. All logic lives in ``plauder/`` (config, server, app, images, audio,
sanitizer, turn_state, session, backends).
"""

from plauder.server import run

if __name__ == "__main__":
    run()
