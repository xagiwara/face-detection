from typing import Optional
from fastapi import UploadFile, Query
import cv2
import numpy as np
from .app import app
from models.hopenet import hopenet, models as hopenet_models
from models.blazeface import blazeface, models as blazeface_models
from models.hsemotion import hsemotion, models as hsemotion_models
from .devices import cuda_devices


@app.get("/models")
async def models():
    return {
        "blazeface": blazeface_models(),
        "hopenet": hopenet_models(),
        "hsemotion": hsemotion_models(),
    }


@app.post("/")
async def all(
    file: UploadFile,
    cuda: str = Query("cpu", enum=cuda_devices()),
    blazeface_model: str = Query(blazeface_models()[0], enum=blazeface_models()),
    hopenet_model: Optional[str] = Query(None, enum=hopenet_models()),
    hsemotion_model: Optional[str] = Query(None, enum=hsemotion_models()),
    face_limit: Optional[int] = None,
):
    img = cv2.imdecode(
        np.frombuffer(await file.read(), dtype=np.uint8), flags=cv2.IMREAD_COLOR
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = blazeface(img, blazeface_model, cuda)
    if face_limit is not None:
        faces = faces[:face_limit]
    cropped = [face.crop(img) for face in faces]

    hopenet_results = None
    if hopenet_model is not None:
        hopenet_results = hopenet(cropped, hopenet_model, cuda)

    hsemotion_results = None
    if hsemotion_model is not None:
        hsemotion_results = hsemotion(cropped, hsemotion_model, cuda)

    def _item(i: int):
        data = {
            "blazeface": faces[i],
        }

        if hopenet_results is not None:
            data["hopenet"] = hopenet_results[i]
        if hsemotion_results is not None:
            data["hsemotion"] = hsemotion_results[i]
        return data

    return [_item(i) for i in range(len(faces))]


__all__ = []
