import sys
import os

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "lib", "synergynet")
)

from .app import app
from .all import *
from .info import *
from .devices import *
from .blazeface import *
from .hopenet import *
from .hsemotion import *
from .synergynet import *
from .fece_alignment import *

__all__ = [
    "app",
]
