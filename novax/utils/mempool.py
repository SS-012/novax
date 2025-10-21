import pycuda.driver as cuda

_pool = []

class _Buffer:
    def __init__(self, ptr, size):
        self.ptr = ptr
        self.size = size

def alloc(size):
    for buf in _pool:
        if buf.size >= size:
            _pool.remove(buf)
            return buf.ptr
    ptr = cuda.mem_alloc(size)
    return ptr

def free(ptr, size]):
    _pool.append(_Buffer(ptr, size))