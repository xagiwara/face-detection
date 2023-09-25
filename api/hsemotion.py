from time import perf_counter
from fastapi import UploadFile, Query
import numpy as np
import cv2
from .app import app
from .devices import cuda_devices
from models.hsemotion import models, load_model, hsemotion


@app.get("/hsemotion/models")
def hsemotion_models():
    return models()


@app.post("/hsemotion")
async def hsemotion_process(
    file: UploadFile,
    cuda: str = Query("cpu", enum=cuda_devices()),
    model: str = Query(hsemotion_models()[0], enum=hsemotion_models()),
):
    buffer = await file.read()
    img = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return hsemotion([img], model, cuda)[0]


@app.post("/hsemotion/prepare")
async def hsemotion_prepare(
    cuda: str = Query("cpu", enum=cuda_devices()),
    model: str = Query(hsemotion_models()[0], enum=hsemotion_models()),
):
    start = perf_counter()
    load_model(model, cuda)
    return {
        "duration": perf_counter() - start,
    }


__all__ = []
