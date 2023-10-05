from os import listdir, path
from os.path import isfile

from torchvision import transforms
import torch
import numpy as np

from lib.OpenGraphAU.model import swin_transformer
from lib.OpenGraphAU.model.ANFL import MEFARG
from lib.OpenGraphAU.utils import load_state_dict
from env import DATA_SWIN, DATA_OPENGRAPHAU

__all__ = ["load_model", "opengraphau", "opengraphau_single", "AU_ids", "AU_names"]

model_cached: dict[str, MEFARG] = {}

swin_transformer.models_dir = path.realpath(DATA_SWIN)


def models():
    return [
        f
        for f in listdir(DATA_OPENGRAPHAU)
        if isfile(path.join(DATA_OPENGRAPHAU, f)) and path.splitext(f)[1] == ".pth"
    ]


def load_model(model_name: str, device: str):
    key = "%s/%s" % (model_name, device)

    if key in model_cached:
        return model_cached[key]

    backbone = None
    if "SwinS" in model_name:
        backbone = "swin_transformer_small"

    model = MEFARG(backbone=backbone)
    model = load_state_dict(
        model,
        path.realpath(path.join(DATA_OPENGRAPHAU, model_name)),
    )
    model.to(device)
    model.eval()

    print("OpenGraphAU: %s model loaded with %s." % (model_name, device))
    model_cached[key] = model
    return model


def opengraphau_single(image, model: str, cuda: str) -> np.ndarray:
    return opengraphau([image], model, cuda)[0]


def opengraphau(images: list, model: str, cuda: str) -> np.ndarray:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model: MEFARG = load_model(model, cuda)

    images = [transform(image) for image in images]
    images: torch.Tensor = torch.stack(images)
    images = images.to(cuda)

    with torch.no_grad():
        pred = model(images)
        return pred.cpu().numpy()


AU_ids = [
    "1",
    "2",
    "4",
    "5",
    "6",
    "7",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "32",
    "38",
    "39",
    "L1",
    "R1",
    "L2",
    "R2",
    "L4",
    "R4",
    "L6",
    "R6",
    "L10",
    "R10",
    "L12",
    "R12",
    "L14",
    "R14",
]

AU_names = [
    "Inner brow raiser",
    "Outer brow raiser",
    "Brow lowerer",
    "Upper lid raiser",
    "Cheek raiser",
    "Lid tightener",
    "Nose wrinkler",
    "Upper lip raiser",
    "Nasolabial deepener",
    "Lip corner puller",
    "Sharp lip puller",
    "Dimpler",
    "Lip corner depressor",
    "Lower lip depressor",
    "Chin raiser",
    "Lip pucker",
    "Tongue show",
    "Lip stretcher",
    "Lip funneler",
    "Lip tightener",
    "Lip pressor",
    "Lips part",
    "Jaw drop",
    "Mouth stretch",
    "Lip bite",
    "Nostril dilator",
    "Nostril compressor",
    "Left Inner brow raiser",
    "Right Inner brow raiser",
    "Left Outer brow raiser",
    "Right Outer brow raiser",
    "Left Brow lowerer",
    "Right Brow lowerer",
    "Left Cheek raiser",
    "Right Cheek raiser",
    "Left Upper lip raiser",
    "Right Upper lip raiser",
    "Left Nasolabial deepener",
    "Right Nasolabial deepener",
    "Left Dimpler",
    "Right Dimpler",
]
