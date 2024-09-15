import cv2
import torch
from torchvision.transforms import functional as F
import numpy as np
from einops import repeat
from mmflow.models.builder import build_flow_estimator
from mmengine.config import Config

# from mmflow.apis import inference_model !!!! do not import any of unsupported modules
from mmflow.models.decoders.gma_decoder import GMADecoder
from mmflow.models.flow_estimators.raft import RAFT

device = "cuda:0"
# Read an image
image = cv2.imread("resources/test_image.png")
# Map the image to [0, 1]
image = image.astype(np.float32)
image = F.to_tensor(image)
image = F.normalize(image, [127.5, 127.5, 127.5], [127.5, 127.5, 127.5])
# Initialize the FlowNet2SD model

cfg = Config.fromfile("resources/GMA/gma_8x2_50k_kitti2015_288x960.py")
ckpt = torch.load("resources/GMA/gma_8x2_50k_kitti2015_288x960.pth")

model = build_flow_estimator(cfg.model)
model.load_state_dict(ckpt["state_dict"])
model.to(device)

input_data = repeat(image, "c h w -> b c h w", b=30)
input_data = torch.cat([input_data, input_data], dim=1).to(device).contiguous()

# output = model(input_data, test_mode=True)
#
# print(output)

# inference
with torch.no_grad():
    if model.test_cfg is not None and model.test_cfg.get("iters") is not None:
        model.decoder.iters = model.test_cfg.get("iters")

    feat1, feat2, h_feat, cxt_feat = model.extract_feat(input_data)
    B, _, H, W = feat1.shape

    flow_init = torch.zeros((B, 2, H, W), device=feat1.device)

    results = model.decoder.forward(
        feat1=feat1,
        feat2=feat2,
        flow=flow_init,
        h=h_feat,
        cxt_feat=cxt_feat,
    )
print(results[-1].min())
print(results[-1].shape)
