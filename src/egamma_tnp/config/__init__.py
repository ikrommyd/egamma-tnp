from __future__ import annotations

from egamma_tnp.config.manager import Config

config = Config()

__all__ = ("config",)


def dir():
    return __all__
