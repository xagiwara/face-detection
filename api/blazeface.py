from .app import app
from fastapi import UploadFile, Query
from fastapi.responses import JSONResponse, Response
import numpy as np
import cv2
from time import perf_counter
from .devices import cuda_devices
from models.blazeface import load_model, blazeface, models


@app.post("/blazeface")
async def blazeface_process(
    file: UploadFile,
    cuda: str = Query("cpu", enum=cuda_devices()),
    model: str = Query("front", enum=["front", "back"]),
):
    buffer = await file.read()
    img = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return blazeface(img, model, cuda)


@app.post("/blazeface/prepare")
async def blazeface_prepare(
    cuda: str = Query("cpu", enum=cuda_devices()),
    model: str = Query("front", enum=["front", "back"]),
):
    start = perf_counter()
    load_model(model, cuda)
    return {
        "duration": perf_counter() - start,
    }


@app.post("/blazeface/crop")
async def blazeface_crop(
    file: UploadFile,
    cuda: str = Query("cpu", enum=cuda_devices()),
    model: str = Query("front", enum=["front", "back"]),
):
    buffer = await file.read()
    img = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = blazeface(img, model, cuda)

    if len(faces) < 1:
        return JSONResponse("", 204)

    img = faces[0].crop(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    _, img = cv2.imencode(".png", img)

    return Response(img.tobytes(), 200, None, "image/png")


@app.get("/blazeface/models")
def blazeface_models():
    return models()


__all__ = []
