from time import perf_counter
from fastapi import UploadFile, Query
import cv2
import numpy as np
from .app import app
from .devices import cuda_devices
from models.hopenet import models, hopenet_single, load_model


@app.get("/hopenet/models")
def hopenet_models():
    return models()


@app.post("/hopenet")
async def hopenet_process(
    file: UploadFile,
    model: str = Query(hopenet_models()[0], enum=hopenet_models()),
    cuda: str = Query("cpu", enum=cuda_devices()),
):
    buffer = await file.read()
    img = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return hopenet_single(img, model, cuda)


@app.post("/hopenet/prepare")
def hopenet_prepare(
    cuda: str = Query("cpu", enum=cuda_devices()),
    model: str = Query(hopenet_models()[0], enum=hopenet_models()),
):
    start = perf_counter()
    load_model(model, cuda)
    return {
        "duration": perf_counter() - start,
    }


__all__ = []
