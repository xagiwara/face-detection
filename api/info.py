from .app import app
from torch import cuda, __version__

@app.get('/info')
async def info():
    devices = ['cpu']
    if cuda.is_available():
        for i in range(cuda.device_count()):
            devices += ['cuda:%d' % i]

    return {
        'devices': {
            'cuda': list(range(cuda.device_count())),
        },
        'version': __version__,
    }

@app.get('/devices/cuda/{device}')
async def cuda_device(device: int):
    props = cuda.get_device_properties('cuda:%d' % device)
    return {
        'device_name': props.name,
        'capability': '%d.%d' % (props.major, props.minor),
        'processors': props.multi_processor_count,
        'memory': props.total_memory,
        'integrated': props.is_integrated,
        'multi_gpu_board': props.is_multi_gpu_board,
    }

@app.get('/devices/cuda')
async def cuda_devices():
    return [await cuda_device(i) for i in range(cuda.device_count())]
