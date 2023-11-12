from pydantic import BaseModel


class FaceDetectionResultModel(BaseModel):
    top: float
    left: float
    bottom: float
    right: float


class BlazeFaceResultModel(FaceDetectionResultModel):
    keypoints: list[list[float]]
    confidence: float
