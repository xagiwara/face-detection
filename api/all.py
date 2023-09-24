from math import floor, ceil
from typing import Optional
from fastapi import UploadFile, Query
from torchvision import transforms
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import numpy as np
from PIL import Image
from .app import app
from .hopenet import hopenet_models, load_model as load_hopenet_model
from .blazeface import load_model as load_blazeface_model
from .devices import cuda_devices

@app.post('/')
async def all(
        file: UploadFile,
        cuda: str = Query('cpu', enum=cuda_devices()),
        blazeface_model: str = Query('front', enum=['front', 'back']),
        hopenet_model: str = Query(hopenet_models()[0], enum=hopenet_models()),
        face_limit: Optional[int] = None):

    blazeface = load_blazeface_model(blazeface_model, cuda)
    hopenet = load_hopenet_model(hopenet_model, cuda)

    img = cv2.imdecode(np.frombuffer(await file.read(), dtype=np.uint8), flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width = img.shape[:2]
    size = max(height, width)
    top = int((size - height) / 2)
    left = int((size - width) / 2)
    img = cv2.copyMakeBorder(img, top, size - height - top, left, size - width - left, cv2.BORDER_CONSTANT, (0, 0, 0))

    if blazeface_model == 'front':
        img_blazeface = cv2.resize(img, (128, 128))

    if blazeface_model == 'back':
        img_blazeface = cv2.resize(img, (256, 256))

    detections = blazeface.predict_on_image(img_blazeface)
    detections = detections.cpu().numpy()

    if len(detections) < 1:
        return []

    if face_limit is not None:
        detections = detections[0:face_limit]

    faces = [
        {
            'top': face[0] * size - top,
            'left': face[1] * size - left,
            'bottom': face[2] * size - top,
            'right': face[3] * size - left,
            'keypoints': [[
                face[i * 2 + 4] * size - left,
                face[i * 2 + 5] * size - top,
            ] for i in range(6)],
            'confidence': float(face[16]),
        }
        for face
        in detections
    ]

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(cuda)

    transformations = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    hopenet_results = []
    for face in faces:
        face_top = floor(face['top'])
        face_left = floor(face['left'])
        face_bottom = ceil(face['bottom'])
        face_right = ceil(face['right'])

        img_hopenet = img

        if face_bottom > size or face_right > size:
            img_hopenet = cv2.copyMakeBorder(img_hopenet, 0, max(face_bottom - size, 0), 0, max(face_right - size, 0), cv2.BORDER_CONSTANT, (0, 0, 0))

        if face_top < 0:
            img_hopenet = cv2.copyMakeBorder(img_hopenet, -face_top, 0, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
            face_top = 0
            face_bottom += -face_top

        if face_left < 0:
            img_hopenet = cv2.copyMakeBorder(img_hopenet, 0, 0, -face_left, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
            face_left = 0
            face_right += -face_left

        img_hopenet = img_hopenet[face_top:face_bottom,face_left:face_right]
        img_hopenet = cv2.resize(img_hopenet, (224, 224))
        img_hopenet = Image.fromarray(img_hopenet)

        img_hopenet = transformations(img_hopenet)
        img_hopenet = img_hopenet.view(1, 3, 224, 224)
        img_hopenet = Variable(img_hopenet).to(cuda)

        yaw, pitch, roll = hopenet(img_hopenet)

        yaw_predicted = F.softmax(yaw, dim=1)
        pitch_predicted = F.softmax(pitch, dim=1)
        roll_predicted = F.softmax(roll, dim=1)

        yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor).cpu().item() * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor).cpu().item() * 3 - 99
        roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor).cpu().item() * 3 - 99

        hopenet_results += [{
            'roll': roll_predicted,
            'pitch': pitch_predicted,
            'yaw': yaw_predicted,
        }]

    return [
        {
            'blazeface': faces[i],
            'hopenet': hopenet_results[i],
        }
        for i in range(len(faces))
    ]


__all__ = []
