try:
    import pycuda.driver as cuda
except Exception:  # ImportError or CUDA runtime missing
    cuda = None

_pool = []


class _Buffer:
    def __init__(self, ptr, size):
        self.ptr = ptr
        self.size = size


def alloc(size):
    """
    Allocate GPU memory of at least `size` bytes.
    If a suitable cached buffer exists, reuse it.
    """
    if cuda is None:
        raise RuntimeError("CUDA not available: cannot allocate GPU buffer")

    # Try to reuse from pool (best fit)
    for buf in _pool:
        if buf.size >= size:
            _pool.remove(buf)
            return buf.ptr

    # Allocate new memory on GPU
    ptr = cuda.mem_alloc(size)
    return ptr


def free(ptr, size, release=False):
    """
    Free a GPU buffer.
    If release=False (default), store it in the pool for reuse.
    If release=True, actually release the memory to CUDA driver.
    """
    if cuda is None or ptr is None:
        return

    if release:
        try:
            ptr.free()  # actually release to driver
        except Exception:
            pass
    else:
        _pool.append(_Buffer(ptr, size))
        

def clear_pool(release: bool = False):
    """
    Clear all cached GPU buffers in the pool.
    If release=True, actually release them to CUDA driver.
    """
    global _pool
    for buf in list(_pool):
        try:
            if release:
                buf.ptr.free()
        except Exception:
            pass
    _pool.clear()


def pool_size():
    """Return total bytes currently stored in the pool."""
    return sum(buf.size for buf in _pool)