import sys
import os

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "lib", "synergynet")
)

from .app import router
from .blazeface import *

__all__ = [
    "router",
]
