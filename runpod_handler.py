import torch
import torchvision.transforms as T
from PIL import Image
import requests
from io import BytesIO
from dinov2.models.vision_transformer import vit_base

# 加载模型
model = vit_base()
model.eval()

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225))
])

def handler(event):
    """
    event["input"] 应该是一个包含图片 URL 的 JSON，例如：
    {"input": {"image_url": "https://..."}}
    """
    image_url = event["input"]["image_url"]
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = model(x).flatten().tolist()

    return {"embedding": features[:16]}  # 仅示例返回前16维
