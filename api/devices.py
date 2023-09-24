from .app import app
from torch import cuda

@app.get('/devices/cuda')
def cuda_devices():
    items = ['cpu']
    if cuda.is_available():
        items += ['cuda:%d' % i for i in range(cuda.device_count())]
    return items

@app.get('/devices/cuda/{device}')
def cuda_device(device: str):
    if device == 'cpu':
        return {
            'cpu': True,
        }
    props = cuda.get_device_properties(device)
    return {
        'cuda': {
            'device_name': props.name,
            'capability': '%d.%d' % (props.major, props.minor),
            'processors': props.multi_processor_count,
            'memory': props.total_memory,
            'integrated': props.is_integrated,
            'multi_gpu_board': props.is_multi_gpu_board,
        }
    }

__all__ = []
