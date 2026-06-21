"""Safeway: extracción estructurada de direcciones imprecisas."""

from . import api
from .api import app
from . import external

__all__ = ["api", "app", "external"]
