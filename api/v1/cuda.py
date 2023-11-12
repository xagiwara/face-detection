import torch
from .app import app
from pydantic import BaseModel

__all__ = []


class CUDADeviceData(BaseModel):
    device_name: str
    capability: str
    processors: int
    memory: int
    integrated: int
    multi_gpu_board: int


class CUDADevice(BaseModel):
    cpu: bool | None
    cuda: CUDADeviceData | None


class CUDAVersion(BaseModel):
    cuda: str | None


@app.get(
    "/cuda",
    tags=["CUDA Devices"],
    response_model=CUDAVersion,
    summary="Cuda Version",
)
def _():
    if torch.cuda.is_available():
        return {
            "cuda": torch.version.cuda,
        }
    else:
        return {"cuda": None}


@app.get(
    "/cuda/devices",
    tags=["CUDA Devices"],
    response_model=list[str],
    summary="Cuda Devices",
)
def _():
    items = ["cpu"]
    if torch.cuda.is_available():
        items += ["cuda:%d" % i for i in range(torch.cuda.device_count())]
    return items


@app.get(
    "/cuda/devices/{device}",
    tags=["CUDA Devices"],
    response_model=CUDADevice,
    summary="Cuda Device",
)
def _(device: str):
    if device == "cpu":
        return {
            "cpu": True,
            "cuda": None,
        }
    props = torch.cuda.get_device_properties(device)
    return {
        "cpu": False,
        "cuda": {
            "device_name": props.name,
            "capability": "%d.%d" % (props.major, props.minor),
            "processors": props.multi_processor_count,
            "memory": props.total_memory,
            "integrated": props.is_integrated,
            "multi_gpu_board": props.is_multi_gpu_board,
        },
    }
