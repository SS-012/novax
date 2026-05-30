import ctypes
import os


class CUDAGraph:
    """Minimal CUDA graph capture/replay wrapper for NovaX GPU callables."""

    def __init__(self):
        self._graph = ctypes.c_void_p()
        self._exec = ctypes.c_void_p()
        self._stream = None
        self._buffers = []
        self._cudart = _load_cudart()

    def capture(self, fn):
        if self._exec.value:
            raise RuntimeError("CUDAGraph instances can only capture once")

        from novax.ops import launcher
        from novax.utils import mempool
        import pycuda.driver as cuda

        self._stream = launcher._get_stream()
        if self._stream is None:
            raise RuntimeError("CUDA stream is not available for graph capture")

        # Warm once so kernels and cuBLAS handles are compiled/initialized before capture.
        warm_output = fn()
        cuda.Context.synchronize()
        del warm_output

        sizes = []
        original_alloc = mempool.alloc

        def recording_alloc(size):
            sizes.append(size)
            return original_alloc(size)

        mempool.alloc = recording_alloc
        try:
            traced_output = fn()
            cuda.Context.synchronize()
            del traced_output
        finally:
            mempool.alloc = original_alloc

        self._buffers = [original_alloc(size) for size in sizes]
        index = 0

        def graph_alloc(size):
            nonlocal index
            if index >= len(self._buffers):
                raise RuntimeError("Capture allocated more buffers than the warm trace")
            ptr = self._buffers[index]
            index += 1
            return ptr

        stream_handle = ctypes.c_void_p(int(self._stream.handle))
        mempool.alloc = graph_alloc
        captured_output = None
        try:
            _check(self._cudart.cudaStreamBeginCapture(stream_handle, ctypes.c_int(0)), "cudaStreamBeginCapture")
            captured_output = fn()
            _check(self._cudart.cudaStreamEndCapture(stream_handle, ctypes.byref(self._graph)), "cudaStreamEndCapture")
        except Exception:
            try:
                self._cudart.cudaStreamEndCapture(stream_handle, ctypes.byref(self._graph))
            except Exception:
                pass
            raise
        finally:
            mempool.alloc = original_alloc
            del captured_output

        if index != len(self._buffers):
            raise RuntimeError("Capture allocated fewer buffers than the warm trace")
        _check(self._cudart.cudaGraphInstantiate(ctypes.byref(self._exec), self._graph, ctypes.c_ulonglong(0)),
               "cudaGraphInstantiate")

    def capture_many(self, fn, count: int):
        if count <= 0:
            raise ValueError("count must be positive")

        def repeated():
            outputs = []
            for _ in range(count):
                outputs.append(fn())
            return outputs

        self.capture(repeated)

    def replay(self):
        if not self._exec.value:
            raise RuntimeError("CUDAGraph.capture() must be called before replay()")
        stream_handle = ctypes.c_void_p(int(self._stream.handle))
        _check(self._cudart.cudaGraphLaunch(self._exec, stream_handle), "cudaGraphLaunch")

    def __del__(self):
        try:
            if self._exec.value:
                self._cudart.cudaGraphExecDestroy(self._exec)
            if self._graph.value:
                self._cudart.cudaGraphDestroy(self._graph)
        except Exception:
            pass


def _load_cudart():
    names = ("cudart64_13.dll", "cudart64_12.dll", "cudart64_11.dll")
    roots = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        roots.extend([
            os.path.join(cuda_path, "bin", "x64"),
            os.path.join(cuda_path, "bin"),
        ])
    roots.extend(os.environ.get("PATH", "").split(os.pathsep))

    for root in roots:
        for name in names:
            path = os.path.join(root, name)
            if not os.path.exists(path):
                continue
            lib = ctypes.CDLL(path)
            lib.cudaStreamBeginCapture.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.cudaStreamBeginCapture.restype = ctypes.c_int
            lib.cudaStreamEndCapture.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            lib.cudaStreamEndCapture.restype = ctypes.c_int
            lib.cudaGraphInstantiate.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_void_p,
                ctypes.c_ulonglong,
            ]
            lib.cudaGraphInstantiate.restype = ctypes.c_int
            lib.cudaGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            lib.cudaGraphLaunch.restype = ctypes.c_int
            lib.cudaGraphDestroy.argtypes = [ctypes.c_void_p]
            lib.cudaGraphDestroy.restype = ctypes.c_int
            lib.cudaGraphExecDestroy.argtypes = [ctypes.c_void_p]
            lib.cudaGraphExecDestroy.restype = ctypes.c_int
            return lib
    raise RuntimeError("Could not locate cudart64 DLL for CUDA graph support")


def _check(status: int, name: str):
    if status != 0:
        raise RuntimeError(f"{name} failed with CUDA runtime status {status}")
