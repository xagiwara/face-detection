from .app import app
from models.hopenet import hopenet, hopenet_single, models
from fastapi import UploadFile, Query, HTTPException
from fastapi.responses import Response
from util.cuda import cuda_devices
import numpy as np
import cv2


__all__ = []


@app.post("/hopenet", tags=["Hopenet"])
async def _(
    files: list[UploadFile],
    cuda: str = Query("cpu", enum=cuda_devices()),
    model: str = Query(models()[0], enum=models()),
):
    names = []
    images = []
    for file in files:
        buffer = await file.read()
        img = cv2.imdecode(
            np.frombuffer(buffer, dtype=np.uint8), flags=cv2.IMREAD_COLOR
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images += [img]
        names += [file.filename]
    result = hopenet(images, model, cuda)

    return {names[i]: result[i] for i in range(len(files))}


@app.post("/hopenet/prepare", tags=["Hopenet"])
async def _(
    cuda: str = Query("cpu", enum=cuda_devices()),
    model: str = Query(models()[0], enum=models()),
):
    hopenet_single(np.zeros((224, 224, 3), dtype=np.uint8), model, cuda)
    return {"ok": True}


@app.get("/hopenet/models", tags=["Hopenet"])
def _():
    return models()


@app.get("/hopenet/models/{model}", tags=["Hopenet"])
def _(model: str = models()[0]):
    _models = models()
    if model not in _models:
        raise HTTPException(404)
    return {
        "input": {
            "width": 224,
            "height": 224,
        }
    }
