from time import perf_counter
import cv2
import numpy as np
from fastapi import UploadFile, Query
from .app import app
from .devices import cuda_devices
from models.synergynet import synergynet, load_model


@app.post("/synergynet")
async def synergynet_process(
    file: UploadFile,
    cuda: str = Query("cpu", enum=cuda_devices()),
):
    buffer = await file.read()
    img = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return synergynet(img, cuda, landmaraks=True, pose=True)


@app.post("/synergynet/prepare")
async def synergynet_prepare(
    cuda: str = Query("cpu", enum=cuda_devices()),
):
    start = perf_counter()
    load_model(cuda)
    return {
        "duration": perf_counter() - start,
    }


__all__ = []
