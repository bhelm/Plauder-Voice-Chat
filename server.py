#!/usr/bin/env python3
"""Entrypoint-Shim.

Der monolithische Server wurde in das Package ``plauder`` zerlegt. Diese
Datei bleibt als Einstiegspunkt erhalten, damit ``python server.py`` (und
start.sh) unverändert funktionieren. Die gesamte Logik liegt in
``plauder/`` (config, server, audio, sanitizer, turn_state, session, backends).
"""

from plauder.server import run

if __name__ == "__main__":
    run()
