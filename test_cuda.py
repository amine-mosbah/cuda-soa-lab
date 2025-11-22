# test_cuda.py
from numba import cuda
import numpy as np


@cuda.jit
def add_one(x):
    i = cuda.grid(1)
    if i < x.size:
        x[i] += 1


def main():
    if not cuda.is_available():
        raise SystemExit("CUDA not available on this machine")

    a = np.arange(10, dtype=np.float32)
    d_a = cuda.to_device(a)

    threadsperblock = 32
    blockspergrid = (a.size + threadsperblock - 1) // threadsperblock

    add_one[blockspergrid, threadsperblock](d_a)
    cuda.synchronize()

    result = d_a.copy_to_host()
    if not np.allclose(result, a + 1):
        raise SystemExit("CUDA kernel did not produce expected result")
    print("CUDA test OK")


if __name__ == "__main__":
    main()
