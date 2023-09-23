from .app import app
from fastapi import UploadFile
import numpy as np
import cv2
from blazeface.blazeface import BlazeFace
from typing import Optional
from env import DATA_BLAZEFACE
from os.path import join
from device import cudaDevice


@app.post('/blazeface')
async def blazeface (cuda: Optional[int], file: UploadFile):
    cuda = cudaDevice(cuda)
    buffer = await file.read()
    img = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
    height, width = img.shape[:2]
    size = max(height, width)
    top = int((size - height) / 2)
    left = int((size - width) / 2)

    img = cv2.copyMakeBorder(img, top, size - height - top, left, size - width - left, cv2.BORDER_CONSTANT, (0, 0, 0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img128 = cv2.resize(img, (128, 128))

    print('blazeface: image loaded.')

    front_net = BlazeFace().to(cuda)
    front_net.load_weights(join(DATA_BLAZEFACE, 'blazeface.pth'))
    front_net.load_anchors(join(DATA_BLAZEFACE, 'anchors.npy'))

    print('blazeface: front model loaded.')

    # back_net = BlazeFace(back_model=True).to(cuda)
    # back_net.load_weights(join(DATA_BLAZEFACE, 'blazefaceback.pth'))
    # back_net.load_anchors(join(DATA_BLAZEFACE, 'anchorsback.npy'))

    # print('blazeface: back model loaded.')

    # front_net.min_score_thresh = 0.75
    # front_net.min_suppression_threshold = 0.3

    front_detections = front_net.predict_on_image(img128)
    front_detections = front_detections.cpu().numpy()

    return {
        'front': [
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
