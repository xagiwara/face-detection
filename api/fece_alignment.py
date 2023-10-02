from time import perf_counter
from .app import app
from fastapi import UploadFile, Query
from .devices import cuda_devices
import cv2
import numpy as np
from models.fece_alignment import load_model, face_alignment_single

__all__ = []


@app.post("/face-alignment")
async def face_alignment_process(
    file: UploadFile,
    cuda: str = Query("cpu", enum=cuda_devices()),
):
    buffer = await file.read()
    img = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return face_alignment_single(img, cuda)


@app.post("/face-alignment/prepare")
def face_alignment_prepare(cuda: str = Query("cpu", enum=cuda_devices())):
    start = perf_counter()
    load_model(cuda)
    return {
        "duration": perf_counter() - start,
    }
