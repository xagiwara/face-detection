from time import perf_counter

from fastapi import Query, Response, UploadFile
import cv2
import numpy as np
from PIL import Image

from .app import app
from util import cuda_devices
from models.opengraphau import opengraphau_single, models, load_model, AU_ids, AU_names

__all__ = []


@app.get("/opengraphau/models", summary="OpenGraphAU Models")
def opengraphau_models():
    return models()


@app.get("/opengraphau/labels", summary="OpenGraphAU Labels")
def opengraphau_labels():
    return {AU_ids[i]: AU_names[i] for i in range(len(AU_ids))}


@app.post("/opengraphau/prepare", summary="OpenGraphAU Prepare")
def opengraphau_prepare(
    model: str = Query(models()[0], enum=models()),
    cuda: str = Query("cpu", enum=cuda_devices()),
):
    load_model(model, cuda)


@app.post("/opengraphau", summary="OpenGraphAU Process")
async def opengraphau_process(
    res: Response,
    file: UploadFile,
    model: str = Query(models()[0], enum=models()),
    cuda: str = Query("cpu", enum=cuda_devices()),
):
    buffer = await file.read()
    img = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = Image.fromarray(img)

    start = perf_counter()

    result = opengraphau_single(img, model, cuda)

    res.headers.append("X-Duration", "%f" % (perf_counter() - start))

    return {AU_ids[i]: float(result[i]) for i in range(result.shape[0])}
