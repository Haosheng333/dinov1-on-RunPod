FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# 需要 git（torch.hub 会用到），以及常用运行依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git ffmpeg libgl1 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 让权重缓存写到容器数据盘
ENV TORCH_HOME=/app/.cache/torch

WORKDIR /app
COPY requirements.txt /app/requirements.txt

# 升级 pip
RUN python3 -m pip install --upgrade pip

# 按 CUDA 12.1 安装匹配的 PyTorch 三件套（不要放到 requirements.txt 里）
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

# 其余依赖
RUN pip install --no-cache-dir -r /app/requirements.txt

# 拷贝代码
COPY . /app

# 启动 RunPod handler
CMD ["python3", "-u", "runpod_handler.py"]
