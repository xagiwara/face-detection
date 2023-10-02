from time import perf_counter
import cv2
import numpy as np
from fastapi import UploadFile, Query, Response
from .app import app
from .devices import cuda_devices
from models.synergynet import synergynet_single, load_model, models
from time import perf_counter


@app.get("/synergynet/models")
def synergynet_models():
    return models()


@app.post("/synergynet")
async def synergynet_process(
    res: Response,
    file: UploadFile,
    model: str = Query(models()[0], enum=models()),
    cuda: str = Query("cpu", enum=cuda_devices()),
):
    synergynet_prepare(model, cuda)

    start = perf_counter()

    buffer = await file.read()
    img = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = synergynet_single(img, model, cuda, landmaraks=True, pose=True)
    duration = perf_counter() - start
    res.headers["X-Duration"] = "%f" % duration
    return result


@app.post("/synergynet/prepare")
def synergynet_prepare(
    model: str = Query(models()[0], enum=models()),
    cuda: str = Query("cpu", enum=cuda_devices()),
):
    start = perf_counter()
    load_model(model, cuda)
    return {
        "duration": perf_counter() - start,
    }


__all__ = []
