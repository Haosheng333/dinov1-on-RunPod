import os
import io
import requests
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T

import runpod

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 用官方仓库的 torch.hub 加载 DINO v1（vits16 或 vitb16）
# 可通过环境变量 MODEL_NAME 切换：dino_vits16 / dino_vitb16
MODEL_NAME = os.environ.get("MODEL_NAME", "dino_vits16")
REPO = "facebookresearch/dino:main"

# ----- 加载模型 -----
# 第一次运行会从 GitHub 下载，需 git；权重缓存到 $TORCH_HOME
model = torch.hub.load(REPO, MODEL_NAME)
model.eval().to(DEVICE)

# ----- 预处理（224） -----
transform = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])

def _load_image(url: str) -> Image.Image:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

@torch.inference_mode()
def _extract_embedding(img: Image.Image):
    x = transform(img).unsqueeze(0).to(DEVICE)  # [1,3,224,224]
    # 官方 hub 模型前向通常直接返回 [B, D] 的 CLS 表征
    feats = model(x)
    feats = F.normalize(feats, dim=-1)
    return feats.squeeze(0).float().cpu().tolist()

def handler(event):
    """
    输入:
      {"input": {"image_url": "https://.../img.jpg", "return_dims": true}}
    返回:
      {"embedding": [...], "dim": 384/768, "model": "dino_vits16", "device": "cuda"}
    """
    inp = event.get("input", {})
    url = inp.get("image_url")
    if not url:
        return {"error": "Missing 'image_url' in input."}

    try:
        img = _load_image(url)
        emb = _extract_embedding(img)
        out = {"embedding": emb, "model": MODEL_NAME, "device": DEVICE}
        if inp.get("return_dims"):
            out["dim"] = len(emb)
        return out
    except Exception as e:
        return {"error": str(e)}

# 启动 RunPod serverless
runpod.serverless.start({"handler": handler})
