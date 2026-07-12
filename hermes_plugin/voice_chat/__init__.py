"""Hermes gateway platform plugin: voice_chat.

``register`` is imported lazily so this package can be imported (and its
``bridge`` module unit-tested) outside the gateway venv, where the
``gateway.*`` modules that ``adapter.py`` needs do not exist.
"""


def register(ctx):
    from .adapter import register as _register
    return _register(ctx)


__all__ = ["register"]
