# main.py
import io
import os
import time
import subprocess
from typing import List

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from numba import cuda
from prometheus_client import Counter, Histogram, Gauge, start_http_server

app = FastAPI(title="GPU Matrix Addition Service")

# ===== Prometheus metrics =====
REQUEST_COUNT = Counter(
    "gpu_service_requests_total", "Total number of requests", ["endpoint"]
)
REQUEST_LATENCY = Histogram(
    "gpu_service_request_latency_seconds", "Request latency", ["endpoint"]
)
GPU_MEMORY_USED = Gauge(
    "gpu_memory_used_mb", "GPU memory used in MB", ["gpu_index"]
)


# ===== CUDA KERNEL: matrix addition on GPU =====
@cuda.jit
def mat_add_kernel(A, B, C):
    i, j = cuda.grid(2)
    if i < A.shape[0] and j < A.shape[1]:
        C[i, j] = A[i, j] + B[i, j]


def gpu_matrix_add(A: np.ndarray, B: np.ndarray):
    """
    Perform C = A + B on the GPU using Numba.
    Returns (C, elapsed_time_seconds).
    """
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available on this machine")

    if A.shape != B.shape:
        raise ValueError("Matrices must have the same shape")

    # Ensure float32 for GPU
    A = A.astype(np.float32)
    B = B.astype(np.float32)

    rows, cols = A.shape

    # Host -> Device
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.device_array_like(d_A)

    # Configure grid / block
    threadsperblock = (16, 16)
    blockspergrid_x = (rows + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (cols + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    start = time.perf_counter()
    mat_add_kernel[blockspergrid, threadsperblock](d_A, d_B, d_C)
    cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Device -> Host
    C = d_C.copy_to_host()
    return C, elapsed


# ===== Helper: parse .npz uploaded file =====
def load_npz_from_upload(file: UploadFile) -> np.ndarray:
    content = file.file.read()
    try:
        npz = np.load(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail=f"File {file.filename} is not a valid npz")

    # convention: take first array inside
    if isinstance(npz, np.lib.npyio.NpzFile):
        keys = list(npz.files)
        if not keys:
            raise HTTPException(status_code=400, detail=f"No arrays in {file.filename}")
        arr = npz[keys[0]]
    else:
        arr = npz
    if arr.ndim != 2:
        raise HTTPException(status_code=400, detail="Matrix must be 2D")
    return arr


# ===== Endpoints =====
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/add")
def add_matrices(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
):
    endpoint = "/add"
    REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.perf_counter()

    try:
        A = load_npz_from_upload(file_a)
        B = load_npz_from_upload(file_b)

        if A.shape != B.shape:
            raise HTTPException(
                status_code=400,
                detail=f"Shape mismatch: {A.shape} vs {B.shape}",
            )

        # 👉 Decide CPU vs GPU
        if cuda.is_available():
            _, elapsed = gpu_matrix_add(A, B)
            device_str = "GPU"
        else:
            # CPU fallback so you can test locally
            t0 = time.perf_counter()
            _ = A + B  # we don't actually need the result for the response
            elapsed = time.perf_counter() - t0
            device_str = "CPU"

        rows, cols = A.shape

        return {
            "matrix_shape": [int(rows), int(cols)],
            "elapsed_time": elapsed,
            "device": device_str,
        }
    finally:
        elapsed_req = time.perf_counter() - start_time
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed_req)


@app.get("/gpu-info")
def gpu_info():
    endpoint = "/gpu-info"
    REQUEST_COUNT.labels(endpoint=endpoint).inc()
    start_time = time.perf_counter()

    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        try:
            output = subprocess.check_output(cmd, encoding="utf-8")
        except (subprocess.CalledProcessError, FileNotFoundError):
            # 👉 No GPU / no nvidia-smi: return empty info but not a crash
            return {
                "gpus": [],
                "error": "nvidia-smi not available on this machine",
            }

        lines = [l.strip() for l in output.splitlines() if l.strip()]

        gpus_info: List[dict] = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            gpu_idx, mem_used, mem_total = parts
            mem_used = float(mem_used)
            mem_total = float(mem_total)

            GPU_MEMORY_USED.labels(gpu_index=gpu_idx).set(mem_used)

            gpus_info.append(
                {
                    "gpu": gpu_idx,
                    "memory_used_MB": mem_used,
                    "memory_total_MB": mem_total,
                }
            )

        return {"gpus": gpus_info}
    finally:
        elapsed_req = time.perf_counter() - start_time
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed_req)



# ===== Entry point: start Prometheus metrics server + FastAPI =====
if __name__ == "__main__":
    import uvicorn
    from prometheus_client import start_http_server

    # Start Prometheus metrics HTTP server on port 8000
    start_http_server(8000)

    # Read student port from env, fallback to 8125
    port = int(os.getenv("STUDENT_PORT", "8125"))

    # IMPORTANT: pass the app object directly, not "main:app"
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)


