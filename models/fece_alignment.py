from face_alignment import FaceAlignment, LandmarksType
from util import CropRect
from typing import Optional

__all__ = ["load_model", "face_alignment_single", "face_alignment"]

_model_cached: dict[str, FaceAlignment] = {}


def load_model(device: str):
    key = device

    if key in _model_cached:
        return _model_cached[key]

    model = FaceAlignment(LandmarksType.THREE_D, device=device)

    print("face_alignment: model loaded with %s." % (device,))
    _model_cached[key] = model
    return model


def face_alignment_single(image, cuda: str, faces: Optional[list[CropRect]] = None):
    model = load_model(cuda)
    _faces = faces
    if faces is None:
        height, width = image.shape[:2]
        _faces = [0, 0, width, height]
    result = model.get_landmarks_from_image(
        image, [[face.left, face.top, face.right, face.bottom] for face in _faces]
    )

    if faces is None:
        return result[0].tolist()
    else:
        return [item.tolist() for item in result]


def face_alignment(
    images: list, cuda: str, faces: Optional[list[list[CropRect]]] = None
):
    if faces is None:
        return [face_alignment_single(images[i], cuda) for i in range(len(images))]
    else:
        return [
            face_alignment_single(images[i], cuda, faces[i]) for i in range(len(images))
        ]
