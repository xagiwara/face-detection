from .app import app
from fastapi import UploadFile, Query
import numpy as np
import cv2
from blazeface.blazeface import BlazeFace
from typing import Optional
from env import DATA_BLAZEFACE
from os.path import join
from device import cudaDevice
from time import perf_counter

model_cached: dict[str, BlazeFace] = {}

def load_model(model: str, device: str):
    key = '%s/%s' % (model, device)

    if key in model_cached:
        return model_cached[key]

    if model == 'front':
        blazeface = BlazeFace().to(device)
        blazeface.load_weights(join(DATA_BLAZEFACE, 'blazeface.pth'))
        blazeface.load_anchors(join(DATA_BLAZEFACE, 'anchors.npy'))

    if model == 'back':
        blazeface = BlazeFace(back_model=True).to(device)
        blazeface.load_weights(join(DATA_BLAZEFACE, 'blazefaceback.pth'))
        blazeface.load_anchors(join(DATA_BLAZEFACE, 'anchorsback.npy'))

    print('blazeface: %s model loaded with %s.' % (model, device))
    model_cached[key] = blazeface
    return blazeface

@app.post('/blazeface/prepare')
async def blazeface (cuda: Optional[int] = None, model: str = Query('front', enum=['front', 'back'])):
    start = perf_counter()
    load_model(model, cudaDevice(cuda))
    return {
        'duration': perf_counter() - start,
    }

@app.post('/blazeface')
async def blazeface (file: UploadFile, cuda: Optional[int] = None, model: str = Query('front', enum=['front', 'back'])):
    cuda = cudaDevice(cuda)
    front_net = load_model(model, cuda)

    buffer = await file.read()
    img = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
    height, width = img.shape[:2]
    size = max(height, width)
    top = int((size - height) / 2)
    left = int((size - width) / 2)

    img = cv2.copyMakeBorder(img, top, size - height - top, left, size - width - left, cv2.BORDER_CONSTANT, (0, 0, 0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if model == 'front':
        img = cv2.resize(img, (128, 128))

    if model == 'back':
        img = cv2.resize(img, (256, 256))

    # front_net.min_score_thresh = 0.75
    # front_net.min_suppression_threshold = 0.3

    front_detections = front_net.predict_on_image(img)
    front_detections = front_detections.cpu().numpy()

    return {
        'faces': [
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
            in front_detections
        ]
    }
