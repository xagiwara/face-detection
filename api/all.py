from typing import Optional
from fastapi import UploadFile, Query
import cv2
import numpy as np
from .app import app
from models.hopenet import hopenet, models as hopenet_models
from models.blazeface import blazeface, models as blazeface_models
from models.hsemotion import hsemotion, models as hsemotion_models
from models.synergynet import synergynet, models as synergynet_models
from models.fece_alignment import face_alignment as face_alignment_run
from .devices import cuda_devices


@app.get("/models")
async def models():
    return {
        "blazeface": blazeface_models(),
        "hopenet": hopenet_models(),
        "hsemotion": hsemotion_models(),
        "synergynet": synergynet_models(),
    }


@app.post("/")
async def all(
    file: UploadFile,
    cuda: str = Query("cpu", enum=cuda_devices()),
    face_limit: Optional[int] = None,
    blazeface_model: str = Query(blazeface_models()[0], enum=blazeface_models()),
    hopenet_model: Optional[str] = Query(None, enum=hopenet_models()),
    hsemotion_model: Optional[str] = Query(None, enum=hsemotion_models()),
    synergynet_model: Optional[str] = Query(None, enum=synergynet_models()),
    synergynet_landmarks: bool = False,
    synergynet_vertices: bool = False,
    synergynet_pose: bool = False,
    face_alignment: bool = False,
):
    img = cv2.imdecode(
        np.frombuffer(await file.read(), dtype=np.uint8), flags=cv2.IMREAD_COLOR
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = blazeface(img, blazeface_model, cuda)
    if face_limit is not None:
        faces = faces[:face_limit]

    if len(faces) < 1:
        return []

    cropped = [face.crop(img) for face in faces]

    hopenet_results = None
    if hopenet_model is not None:
        hopenet_results = hopenet(cropped, hopenet_model, cuda)

    hsemotion_results = None
    if hsemotion_model is not None:
        hsemotion_results = hsemotion(cropped, hsemotion_model, cuda)

    synergynet_results = None
    if synergynet_model is not None and (
        synergynet_landmarks or synergynet_vertices or synergynet_pose
    ):
        synergynet_results = synergynet(
            cropped,
            synergynet_model,
            cuda,
            faces,
            landmaraks=synergynet_landmarks,
            vertices=synergynet_vertices,
            pose=synergynet_pose,
        )

    face_alignment_results = None
    if face_alignment:
        face_alignment_results = face_alignment_run([img], cuda, [faces])[0]

    def _item(i: int):
        data = {
            "blazeface": faces[i],
        }

        if hopenet_results is not None:
            data["hopenet"] = hopenet_results[i]
        if hsemotion_results is not None:
            data["hsemotion"] = hsemotion_results[i]
        if synergynet_results is not None:
            data["synergynet"] = synergynet_results[i]
        if face_alignment_results is not None:
            data["face_alignment"] = face_alignment_results[i]
        return data

    return [_item(i) for i in range(len(faces))]


__all__ = []
