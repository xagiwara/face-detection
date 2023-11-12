from os import listdir, path
from os.path import isfile
from torchvision import transforms
from torchvision.models.resnet import Bottleneck
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
from lib.hopenet.code.hopenet import Hopenet
from env import DATA_HOPENET
from util import EulerAngle


_model_cached: dict[str, Hopenet] = {}


def load_model(model_name: str, device: str):
    key = "%s/%s" % (model_name, device)

    if key in _model_cached:
        return _model_cached[key]

    model = Hopenet(Bottleneck, [3, 4, 6, 3], 66)
    model.load_state_dict(
        torch.load(
            path.join(DATA_HOPENET, model_name), map_location=torch.device("cpu")
        )
    )
    model.to(device)
    model.eval()

    print("hopenet: %s model loaded with %s." % (model_name, device))
    _model_cached[key] = model
    return model


def models():
    return [
        f
        for f in listdir(DATA_HOPENET)
        if isfile(path.join(DATA_HOPENET, f)) and path.splitext(f)[1] == ".pkl"
    ]


def hopenet_single(image, model: str, cuda: str):
    return hopenet([image], model, cuda)[0]


def hopenet(images: list, model: str, cuda: str):
    model = load_model(model, cuda)

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(cuda)

    transformations = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    images = [transformations(Image.fromarray(image)) for image in images]

    images = Variable(torch.stack(images)).to(cuda)

    yaw, pitch, roll = model(images)

    yaw_predicted = F.softmax(yaw, dim=1) * idx_tensor
    pitch_predicted = F.softmax(pitch, dim=1) * idx_tensor
    roll_predicted = F.softmax(roll, dim=1) * idx_tensor

    yaw_predicted = (
        torch.sum(yaw_predicted, dim=1).cpu().detach().numpy() * 3 - 99
    ).astype(float)
    pitch_predicted = (
        torch.sum(pitch_predicted, dim=1).cpu().detach().numpy() * 3 - 99
    ).astype(float)
    roll_predicted = (
        torch.sum(roll_predicted, dim=1).cpu().detach().numpy() * 3 - 99
    ).astype(float)

    return [
        EulerAngle(roll_predicted[i], pitch_predicted[i], yaw_predicted[i])
        for i in range(len(images))
    ]
