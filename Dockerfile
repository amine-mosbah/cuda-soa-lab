# Dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Your student port (can be overridden by env at runtime)
ENV STUDENT_PORT=8125

# Expose FastAPI port + Prometheus metrics port
EXPOSE 8125 8000

# Start app (starts Prometheus server on 8000 + uvicorn on STUDENT_PORT)
CMD ["python3", "main.py"]
