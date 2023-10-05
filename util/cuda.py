from torch import cuda

__all__ = ["cuda_devices"]


def cuda_devices():
    items = ["cpu"]
    if cuda.is_available():
        items += ["cuda:%d" % i for i in range(cuda.device_count())]
    return items
