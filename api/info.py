from .app import app
from .devices import cuda_devices
from torch import __version__ as torch_version


@app.get("/info")
async def info():
    return {
        "devices": {
            "cuda": cuda_devices(),
        },
        "pytorch": {
            "version": torch_version,
        },
    }


__all__ = []
