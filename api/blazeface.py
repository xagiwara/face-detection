from .app import app
from fastapi import UploadFile, Query
from fastapi.responses import JSONResponse, Response
import numpy as np
import cv2
from lib.blazeface.blazeface import BlazeFace
from env import DATA_BLAZEFACE
from os.path import join
from time import perf_counter
from math import floor, ceil
from .devices import cuda_devices

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


@app.post('/blazeface')
async def blazeface_process (file: UploadFile, cuda: str = Query('cpu', enum=cuda_devices()), model: str = Query('front', enum=['front', 'back'])):
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

@app.post('/blazeface/prepare')
async def blazeface_prepare (cuda: str = Query('cpu', enum=cuda_devices()), model: str = Query('front', enum=['front', 'back'])):
    start = perf_counter()
    load_model(model, cuda)
    return {
        'duration': perf_counter() - start,
    }

@app.post('/blazeface/crop')
async def blazeface_crop (file: UploadFile, cuda: str = Query('cpu', enum=cuda_devices()), model: str = Query('front', enum=['front', 'back'])):
    front_net = load_model(model, cuda)

    buffer = await file.read()
    img_orig = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
    height, width = img_orig.shape[:2]
    size = max(height, width)
    top = int((size - height) / 2)
    left = int((size - width) / 2)

    img_padded = cv2.copyMakeBorder(img_orig, top, size - height - top, left, size - width - left, cv2.BORDER_CONSTANT, (0, 0, 0))
    img = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)

    if model == 'front':
        img = cv2.resize(img, (128, 128))

    if model == 'back':
        img = cv2.resize(img, (256, 256))

    # front_net.min_score_thresh = 0.75
    # front_net.min_suppression_threshold = 0.3

    front_detections = front_net.predict_on_image(img)
    front_detections = front_detections.cpu().numpy()

    if len(front_detections) < 1:
        return JSONResponse('', 204)

    face = front_detections[0]
    face_top = floor(face[0] * size - top)
    face_left = floor(face[1] * size - left)
    face_bottom = ceil(face[2] * size - top)
    face_right = ceil(face[3] * size - left)

    if face_bottom > size or face_right > size:
        img_padded = cv2.copyMakeBorder(img_padded, 0, max(face_bottom - size, 0), 0, max(face_right - size, 0), cv2.BORDER_CONSTANT, (0, 0, 0))

    if face_top < 0:
        img_padded = cv2.copyMakeBorder(img_padded, -face_top, 0, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
        face_top = 0
        face_bottom += -face_top

    if face_left < 0:
        img_padded = cv2.copyMakeBorder(img_padded, 0, 0, -face_left, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
        face_left = 0
        face_right += -face_left

    img_padded = img_padded[face_top:face_bottom,face_left:face_right]

    _, image = cv2.imencode('.png', img_padded)

    return Response(image.tobytes(), 200, None, 'image/png')

__all__ = []
