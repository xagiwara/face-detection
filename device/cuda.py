from typing import Optional

def cudaDevice(id: Optional[int]):
    if id is None:
        return 'cpu'
    else:
        return 'cuda:%d' % id
