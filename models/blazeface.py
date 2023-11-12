from os.path import join
from math import floor, ceil
import cv2
from env import DATA_BLAZEFACE
from lib.blazeface.blazeface import BlazeFace
from util import CropRect
from typing import Literal

_model_cached: dict[str, BlazeFace] = {}


class BlazeFaceResult(CropRect):
    keypoints: list[list[float]]
    confidence: float

    def __init__(
        self, result: list[float], size: float, xoffset: float, yoffset: float
    ) -> None:
        super().__init__(
            result[0] * size - yoffset,
            result[1] * size - xoffset,
            result[2] * size - yoffset,
            result[3] * size - xoffset,
        )

        self.keypoints = (
            [
                [
                    result[i * 2 + 4] * size - xoffset,
                    result[i * 2 + 5] * size - yoffset,
                ]
                for i in range(6)
            ],
        )
        self.confidence = float(result[16])

    def crop(self, image):
        height, width = image.shape[:2]
        top = floor(self.top)
        left = floor(self.left)
        bottom = ceil(self.bottom)
        right = ceil(self.right)

        if bottom > height or right > width:
            image = cv2.copyMakeBorder(
                image,
                0,
                max(bottom - height, 0),
                0,
                max(right - width, 0),
                cv2.BORDER_CONSTANT,
                (0, 0, 0),
            )

        if top < 0:
            image = cv2.copyMakeBorder(
                image, -top, 0, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0)
            )
            top = 0
            bottom += -top

        if left < 0:
            image = cv2.copyMakeBorder(
                image, 0, 0, -left, 0, cv2.BORDER_CONSTANT, (0, 0, 0)
            )
            left = 0
            right += -left

        return image[top:bottom, left:right]


def models():
    return ["front", "back"]


def load_model(model: str, device: str):
    key = "%s/%s" % (model, device)

    if key in _model_cached:
        return _model_cached[key]

    if model == "front":
        blazeface = BlazeFace().to(device)
        blazeface.load_weights(join(DATA_BLAZEFACE, "blazeface.pth"))
        blazeface.load_anchors(join(DATA_BLAZEFACE, "anchors.npy"))

    if model == "back":
        blazeface = BlazeFace(back_model=True).to(device)
        blazeface.load_weights(join(DATA_BLAZEFACE, "blazefaceback.pth"))
        blazeface.load_anchors(join(DATA_BLAZEFACE, "anchorsback.npy"))

    print("blazeface: %s model loaded with %s." % (model, device))
    _model_cached[key] = blazeface
    return blazeface


def blazeface_image_size(
    model_name: str,
) -> tuple[Literal[128, 256], Literal[128, 256]]:
    if model_name == "front":
        return (128, 128)
    if model_name == "back":
        return (256, 256)


def blazeface(image, model_name: str, device: str):
    model = load_model(model_name, device)
    image, size, left, top = blazeface_prepare_image(image, model_name)

    # front_net.min_score_thresh = 0.75
    # front_net.min_suppression_threshold = 0.3

    detections = model.predict_on_image(image)
    detections = detections.cpu().numpy()

    return [BlazeFaceResult(face, size, left, top) for face in detections]


def blazeface_prepare_image(image, model_name):
    height, width = image.shape[:2]
    height_t, width_t = blazeface_image_size(model_name)
    if height != height_t or width != width_t:
        size: int = max(height, width)
        top = int((size - height) / 2)
        left = int((size - width) / 2)

        image = cv2.copyMakeBorder(
            image,
            top,
            size - height - top,
            left,
            size - width - left,
            cv2.BORDER_CONSTANT,
            (0, 0, 0),
        )

        if model_name == "front":
            image = cv2.resize(image, (128, 128))

        if model_name == "back":
            image = cv2.resize(image, (256, 256))

        return image, size, left, top
    return image, height_t, 0, 0


def blazeface_batch(images, model_name, device):
    model = load_model(model_name, device)
    detections = model.predict_on_batch(images)
    return [detection.cpu().numpy() for detection in detections]


__all__ = [
    "models",
    "load_model",
]
