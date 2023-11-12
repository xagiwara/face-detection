from .app import app
from fastapi import UploadFile, Query
import numpy as np
import cv2
from util.cuda import cuda_devices
from models.blazeface import (
    blazeface,
    blazeface_batch,
    blazeface_prepare_image,
    blazeface_image_size,
    BlazeFaceResult,
)
from .interfaces.face import BlazeFaceResultModel

__all__ = []


@app.post(
    "/blazeface",
    tags=["Face Detection"],
    response_model=dict[str, list[BlazeFaceResultModel]],
)
async def _(
    files: list[UploadFile],
    cuda: str = Query("cpu", enum=cuda_devices()),
    model: str = Query("front", enum=["front", "back"]),
):
    images = []
    src_images = []
    names = []
    sizes = []
    lefts = []
    tops = []
    for file in files:
        buffer = await file.read()
        img = cv2.imdecode(
            np.frombuffer(buffer, dtype=np.uint8), flags=cv2.IMREAD_COLOR
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        src_images += [img]
        image, size, left, top = blazeface_prepare_image(img, model)
        images += [image]
        names += [file.filename]
        sizes += [size]
        lefts += [left]
        tops += [top]

    images = np.array(images)
    result = blazeface_batch(images, model, cuda)
    return {
        names[file]: [
            BlazeFaceResult(face, sizes[file], lefts[file], tops[file])
            for face in result[file]
        ]
        for file in range(len(images))
    }


@app.post("/blazeface/prepare", tags=["Face Detection"])
async def _(
    cuda: str = Query("cpu", enum=cuda_devices()),
    model: str = Query("front", enum=["front", "back"]),
):
    image = np.zeros((*blazeface_image_size(model), 3), dtype=np.float32)
    blazeface(image, model, cuda)
    return {"ok": True}
