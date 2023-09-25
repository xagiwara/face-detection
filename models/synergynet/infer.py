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

__all__ = ["SynergyNetResult", "synergynet", "load_model"]


class SynergyNetResult:
    landmarks: Optional[list[list[float]]]
    vertices: Optional[list[list[float]]]
    angle: Optional[EulerAngle]
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
            angle, translation = predict_pose(param, roi_box)
            self.angle = EulerAngle(angle[2], angle[1], angle[0])
            self.translation = [translation[0], translation[1], translation[2]]


class SynergyNetWrapper:
    def __init__(self, cuda: str) -> None:
        checkpoint = torch.load(
            path.join(DATA_SYNERGYNET, "best.pth.tar"),
            map_location=lambda storage, loc: storage,
        )["state_dict"]

        args = Namespace()
        args.arch = "mobilenet_v2"
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


def load_model(cuda: str):
    ParamsPack().load()
    if cuda in _model_cached:
        return _model_cached[cuda]
    model = SynergyNetWrapper(cuda)

    print("blazeface: model loaded with %s." % (cuda,))
    _model_cached[cuda] = model
    return model


def synergynet_single(
    img,
    cuda: str,
    original_rect: Optional[CropRect] = None,
    landmaraks=False,
    vertices=False,
    pose=False,
):
    model = load_model(cuda)

    transform = transforms.Compose(
        [
            transforms.Resize((120, 120)),
            transforms.ToTensor(),
            transforms.Normalize(mean=127.5, std=128),
        ]
    )

    input = transform(Image.fromarray(img)).unsqueeze(0)
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
            images[i], cuda, original_rects[i], landmaraks, vertices, pose
        )
        for i in range(len(images))
    ]
