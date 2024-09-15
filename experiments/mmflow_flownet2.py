import sys

from mmflow.models.encoders.flownet_encoder import FlowNetEncoder

sys.path.append("./src")
from typing import Dict

import cv2
import numpy as np
import torch
from einops import repeat
from mmengine.config import Config
from mmflow.models.builder import build_flow_estimator
from torch import Tensor, nn
from csi_flow_utils import flow_to_color
from csi_flow_utils.flow_utils import flow2img
from torchvision.transforms import functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# from mmflow.apis import inference_model !!!! do not import any of unsupported modules
from mmflow.models.flow_estimators.flownet import FlowNetS
from mmflow.models.encoders.flownet_encoder import FlowNetSDEncoder
# from mmflow.models.decoders.flownet_decoder


class FlowNetV2Wrapper(nn.Module):
    """
    A wrapper for LiteFlowNet model.
    This wrapper aims to run in evaluation mode. and allows the user to define how many decoder levels to use.
    So that won't waste time on the decoder levels that are not needed.
    """

    def __init__(self, cfg: str, ckpt: str, n_decoder_levels=None):
        """
        @param n_decoder_levels: The number of decoder levels to use. If None, all decoder levels are used.
        """
        super(FlowNetV2Wrapper, self).__init__()
        _cfg = Config.fromfile(cfg)
        _ckpt = torch.load(ckpt)
        self.n_decoder_levels = n_decoder_levels
        self.estimator = build_flow_estimator(_cfg.model)

    @torch.no_grad()
    def forward(self, imgs1: Tensor, imgs2: Tensor) -> Dict[str, Tensor]:
        """
        @param imgs1: Tensor of shape (b, c, h, w)
        @param imgs2: Tensor of shape (b, c, h, w)
        """
        imgs = torch.cat([imgs1, imgs2], dim=1)

        H, W = imgs.shape[2:]
        feat = self.estimator.encoder(imgs)

        flow_pred = dict()
        upfeat = None
        upflow = None

        for level in self.estimator.decoder.flow_levels[::-1]:
            if level == self.estimator.decoder.start_level:
                feat_ = feat[level]
            else:
                feat_ = torch.cat((feat[level], upfeat, upflow), dim=1)

            flow, upflow, upfeat = self.estimator.decoder.decoders[level](feat_)
            flow_pred[level] = flow * self.estimator.decoder.flow_div

        return flow_pred


device = "cuda:0"

images = []
for img in ["frame_0", "frame_1"]:
    # Read an image
    image = cv2.imread(f"outputs/{img}.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Map the image to [0, 1]
    image = image.astype(np.float32)
    image = F.to_tensor(image)
    # image = F.normalize(image, [0, 0, 0], [255, 255, 255])
    image = image / 255.0
    image.unsqueeze_(0)
    image = image.to(device)
    images.append(image)

model = FlowNetV2Wrapper(
    # "resources/litflownet2/liteflownet2_pre_M6S6_8x1_flyingchairs_320x448.py",
    # "resources/litflownet2/liteflownet2_pre_M6S6_8x1_flyingchairs_320x448.pth",
    "resources/flownet2/flownet2sd_8x1_slong_chairssdhom_384x448.py",
    "resources/flownet2/flownet2sd_8x1_slong_chairssdhom_384x448.pth",
    n_decoder_levels=None,
)
model.to(device)


output = model(images[1], images[0])
output = output["level2"]
print(output.max())
print(output[0])
plt.imshow(output[0].cpu().numpy()[0])
plt.savefig("outputs/flow_x.png")
plt.savefig("outputs/flow_y.png")

img = flow2img(output[0].cpu().numpy().transpose(1, 2, 0))
plt.imshow(img)
plt.savefig("outputs/flow.png")


# b 2 h w
# flow_vis = flow_to_color(output)
# # flow_vis = F.resize(flow_vis, [256, 256])
# plt.imshow(flow_vis[0].cpu().numpy().transpose(1, 2, 0))
# plt.savefig("outputs/flow.png")
# print(flow_vis.max())
# save_image(flow_vis[0], "outputs/flow_vis.jpg")
