from .app import app
from fastapi import UploadFile, Query
from fastapi.responses import JSONResponse, Response
import numpy as np
import cv2
from time import perf_counter
from util.cuda import cuda_devices
from models.blazeface import (
    load_model,
    blazeface,
    models,
    blazeface_batch,
    blazeface_prepare_image,
    BlazeFaceResult,
)

__all__ = []


@app.post("/blazeface", tags=["Face Detection"])
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
