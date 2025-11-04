# 用带 CUDA 的基础镜像，避免装错 PyTorch
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 基础环境
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git ffmpeg libgl1 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 工作目录
WORKDIR /app
COPY requirements.txt /app/requirements.txt

# 升级 pip
RUN python3 -m pip install --upgrade pip

# 先按 CUDA12.1 安装匹配的 PyTorch（不要写在 requirements.txt 里）
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

# 其余依赖
RUN pip install --no-cache-dir -r /app/requirements.txt

# 复制源代码
COPY . /app

# 启动 RunPod handler
CMD ["python3", "-u", "runpod_handler.py"]

