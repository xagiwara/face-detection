import torch
from os import path
from .lib.model_building import SynergyNet
from .lib.inference import (
    predict_sparseVert,
    predict_denseVert,
    predict_pose,
)

import numpy as np
from torchvision import transforms
from argparse import Namespace
from PIL import Image
from typing import Optional

from env import DATA_SYNERGYNET
from util import EulerAngle, CropRect
from .lib.params import ParamsPack
import cv2

__all__ = [
    "SynergyNetResult",
    "synergynet",
    "synergynet_single",
    "load_model",
    "models",
]


class SynergyNetResult:
    landmarks: Optional[list[list[float]]]
    vertices: Optional[list[list[float]]]
    pose: Optional[EulerAngle]
    translation: Optional[list[float]]

    def __init__(
        self, param, roi_box, landmaraks=False, vertices=False, pose=False
    ) -> None:
        if landmaraks:
            data = predict_sparseVert(param, roi_box=roi_box, transform=True)
            self.landmarks = [[data[0][i], data[1][i], data[2][i]] for i in range(68)]

        if vertices:
            data = predict_denseVert(param, roi_box=roi_box, transform=True)
            self.vertices = [[data[0][i], data[1][i], data[2][i]] for i in range(53215)]

        if pose:
            pose, translation = predict_pose(param, roi_box)
            self.pose = EulerAngle(pose[2], pose[1], pose[0])
            self.translation = [translation[0], translation[1], translation[2]]


class SynergyNetWrapper:
    def __init__(self, model: str, cuda: str) -> None:
        checkpoint = torch.load(
            path.join(DATA_SYNERGYNET, "best.pth.tar"),
            map_location=cuda,
        )["state_dict"]

        args = Namespace()
        args.arch = model
        args.img_size = 120

        model = SynergyNet(args)
        model_dict = model.state_dict()

        # because the model is trained by multiple gpus, prefix 'module' should be removed
        for k in checkpoint.keys():
            model_dict[k.replace("module.", "")] = checkpoint[k]

        model.load_state_dict(model_dict, strict=False)
        model = model.to(cuda)
        model.eval()
        self.model = model


_model_cached: dict[str, SynergyNetWrapper] = {}


def load_model(model_name: str, cuda: str):
    ParamsPack().load()
    key = "%s/%s" % (model_name, cuda)

    if key in _model_cached:
        return _model_cached[key]
    model = SynergyNetWrapper(model_name, cuda)

    print("synergynet: %s model loaded with %s." % (model_name, cuda))
    _model_cached[key] = model
    return model


def models():
    return [
        "mobilenet_v2",
        # "mobilenet",
        # "resnet",
        # "ghostnet",
        "resnest",
    ]


def synergynet_single(
    img,
    model_name: str,
    cuda: str,
    original_rect: Optional[CropRect] = None,
    landmaraks=False,
    vertices=False,
    pose=False,
):
    model = load_model(model_name, cuda)

    input = cv2.resize(img, dsize=(120, 120), interpolation=cv2.INTER_LANCZOS4)
    input = torch.from_numpy(input)
    input = (input - 127.5) / 128.0
    input = input.permute(2, 0, 1)
    input = input.unsqueeze(0)

    with torch.no_grad():
        input = input.to(cuda)
        param = model.model.forward_test(input)
        param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

    if original_rect is None:
        height, width = img.shape[:2]
        roi_box = [0, 0, width, height, 0]
    else:
        roi_box = [
            original_rect.left,
            original_rect.top,
            original_rect.right,
            original_rect.bottom,
            0,
        ]
    return SynergyNetResult(param, roi_box, landmaraks, vertices, pose)


def synergynet(
    images: list,
    model_name: str,
    cuda: str,
    original_rects: Optional[list[CropRect]] = None,
    landmaraks=False,
    vertices=False,
    pose=False,
):
    if original_rects is None:
        original_rects = [None for _ in images]

    return [
        synergynet_single(
            images[i], model_name, cuda, original_rects[i], landmaraks, vertices, pose
        )
        for i in range(len(images))
    ]
