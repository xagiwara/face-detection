from os import listdir, path
from os.path import isfile
from time import perf_counter
from fastapi import UploadFile, Query
from torchvision import transforms
from torchvision.models.resnet import Bottleneck
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import numpy as np
from PIL import Image
from hopenet.code.hopenet import Hopenet
from .app import app
from env import DATA_HOPENET
from .devices import cuda_devices

model_cached: dict[str, Hopenet] = {}

def load_model(model_name: str, device: str):
    key = '%s/%s' % (model_name, device)

    if key in model_cached:
        return model_cached[key]

    model = Hopenet(Bottleneck, [3, 4, 6, 3], 66)
    model.load_state_dict(torch.load(path.join(DATA_HOPENET, model_name)))
    model.to(device)
    model.eval()

    print('hopenet: %s model loaded with %s.' % (model_name, device))
    model_cached[key] = model
    return model


@app.get('/hopenet/models')
def hopenet_models ():
    return [f for f in listdir(DATA_HOPENET) if isfile(path.join(DATA_HOPENET, f)) and path.splitext(f)[1] == '.pkl']

@app.post('/hopenet')
async def hopenet_process (file: UploadFile, model: str = Query(hopenet_models()[0], enum=hopenet_models()), cuda: str = Query('cpu', enum=cuda_devices())):
    model = model or [f for f in listdir(DATA_HOPENET) if isfile(path.join(DATA_HOPENET, f)) and path.splitext(f)[1] == '.pkl'][0]
    model = load_model(model, cuda)

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(cuda)

    transformations = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    buffer = await file.read()
    img = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    img = Image.fromarray(img)
    img = transformations(img)
    img_shape = img.size()
    img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
    img = Variable(img).to(cuda)

    yaw, pitch, roll = model(img)

    yaw_predicted = F.softmax(yaw, dim=1)
    pitch_predicted = F.softmax(pitch, dim=1)
    roll_predicted = F.softmax(roll, dim=1)

    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor).cpu().item() * 3 - 99
    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor).cpu().item() * 3 - 99
    roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor).cpu().item() * 3 - 99

    return {
        'roll': roll_predicted,
        'pitch': pitch_predicted,
        'yaw': yaw_predicted,
    }

@app.post('/hopenet/prepare')
async def hopenet_prepare (cuda: str = Query('cpu', enum=cuda_devices()), model: str = Query(hopenet_models()[0], enum=hopenet_models())):
    start = perf_counter()
    load_model(model, cuda)
    return {
        'duration': perf_counter() - start,
    }

__all__ = []
