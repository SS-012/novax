try:
    import pycuda.driver as cuda
except Exception:
    cuda = None

# Bucketed free-list: maps power-of-2 size → list of free device pointers.
# Allocations are rounded up to the next power of 2 so lookups are O(1).
_pool: dict = {}


def _bucket(size: int) -> int:
    """Round size up to the next power of 2 (minimum 64 bytes)."""
    if size <= 64:
        return 64
    return 1 << (size - 1).bit_length()


def alloc(size: int):
    """
    Allocate GPU memory of at least `size` bytes.
    Reuses a cached buffer of the same bucket size when available.
    """
    if cuda is None:
        raise RuntimeError("CUDA not available: cannot allocate GPU buffer")
    b = _bucket(size)
    bucket_list = _pool.get(b)
    if bucket_list:
        return bucket_list.pop()
    return cuda.mem_alloc(b)


def free(ptr, size: int, release: bool = False):
    """
    Return a GPU buffer to the pool (default) or release it to the CUDA driver.
    """
    if cuda is None or ptr is None:
        return
    if release:
        try:
            ptr.free()
        except Exception:
            pass
        return
    b = _bucket(size)
    if b not in _pool:
        _pool[b] = []
    _pool[b].append(ptr)


def clear_pool(release: bool = False):
    """Clear all cached GPU buffers."""
    global _pool
    for bucket_list in _pool.values():
        for ptr in bucket_list:
            try:
                if release:
                    ptr.free()
            except Exception:
                pass
    _pool.clear()


def pool_size() -> int:
    """Return total bytes currently held in the pool."""
    return sum(b * len(lst) for b, lst in _pool.items())
