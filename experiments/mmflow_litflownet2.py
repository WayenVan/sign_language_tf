import sys

sys.path.append("./src")
from typing import Dict

import cv2
import numpy as np
import torch
from einops import repeat
from mmengine.config import Config
from mmflow.models.builder import build_flow_estimator
from torch import Tensor, nn
from csi_flow_vis_torch import flow_to_color
from csi_flow_vis_torch.flow_utils import flow2img
from torchvision.transforms import functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# from mmflow.apis import inference_model !!!! do not import any of unsupported modules


class LiteFlowNetWrapper(nn.Module):
    """
    A wrapper for LiteFlowNet model.
    This wrapper aims to run in evaluation mode. and allows the user to define how many decoder levels to use.
    So that won't waste time on the decoder levels that are not needed.
    """

    def __init__(self, cfg: str, ckpt: str, n_decoder_levels=None):
        """
        @param n_decoder_levels: The number of decoder levels to use. If None, all decoder levels are used.
        """
        super(LiteFlowNetWrapper, self).__init__()
        _cfg = Config.fromfile(cfg)
        _ckpt = torch.load(ckpt)
        self.n_decoder_levels = n_decoder_levels
        self.estimator = build_flow_estimator(_cfg.model)

    @torch.no_grad()
    def forward(self, imgs: Tensor) -> Dict[str, Tensor]:
        img1, img2, feat1, feat2 = self.estimator.extract_feat(imgs)

        # return self.decoder.forward_test(img1, img2, feat1, feat2)
        # decoder prediction
        flow_pred = dict()

        upflow = None
        level_counter = 0
        for level in self.estimator.decoder.flow_levels[::-1]:
            _feat1 = self.estimator.decoder.decoders[level]["feat_layer"](feat1[level])
            _feat2 = self.estimator.decoder.decoders[level]["feat_layer"](feat2[level])
            h, w = _feat1.shape[2:]
            _img1, _img2 = (
                self.estimator.decoder._scale_img(img1, h, w),
                self.estimator.decoder._scale_img(img2, h, w),
            )

            flowM = self.estimator.decoder.decoders[level]["NetM"](
                _feat1,
                _feat2,
                upflow,
                self.estimator.decoder.multiplier[level],
            )
            flowS = self.estimator.decoder.decoders[level]["NetS"](
                _feat1,
                _feat2,
                flowM,
                self.estimator.decoder.multiplier[level],
            )
            if (
                level == self.estimator.decoder.end_level
                and not self.estimator.decoder.regularized_flow
            ):
                upflow = self.estimator.decoder.decoders[level]["upflow_layer"](flowS)
                flow_pred[level] = flowS
            else:
                rfeat = self.estimator.decoder.decoders[level]["rfeat_layer"](
                    feat1[level]
                )
                flowR = self.estimator.decoder.decoders[level]["NetR"](
                    _img1,
                    _img2,
                    rfeat,
                    flowS,
                    self.estimator.decoder.multiplier[level],
                )
                upflow = self.estimator.decoder.decoders[level]["upflow_layer"](flowR)
                flow_pred[level] = flowR

            level_counter += 1
            if (
                self.n_decoder_levels is not None
                and level_counter >= self.n_decoder_levels
            ):
                break

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
    image = F.normalize(image, [0, 0, 0], [255, 255, 255])
    images.append(image)

model = LiteFlowNetWrapper(
    # "resources/litflownet2/liteflownet2_pre_M6S6_8x1_flyingchairs_320x448.py",
    # "resources/litflownet2/liteflownet2_pre_M6S6_8x1_flyingchairs_320x448.pth",
    "resources/litflownet2/liteflownet2_pre_M3S3R3_8x1_flyingchairs_320x448.py",
    "resources/litflownet2/liteflownet2_pre_M3S3R3_8x1_flyingchairs_320x448.pth",
    n_decoder_levels=None,
)
model.to(device)

input_data = torch.cat(images, dim=0).to(device)
# input_data = repeat(input_data, "c h w -> b c h w", b=1)
input_data = input_data.unsqueeze(0)

output = model(input_data)
output = output["level3"] * 20.0
print(output.max())
print(output[0])
plt.imshow(output[0].cpu().numpy()[0])
plt.savefig("outputs/flow_x.png")
plt.savefig("outputs/flow_y.png")

img = flow2img(output[0].cpu().numpy().transpose(1, 2, 0))
cv2.imwrite("outputs/flow.png", img)


# b 2 h w
# flow_vis = flow_to_color(output)
# # flow_vis = F.resize(flow_vis, [256, 256])
# plt.imshow(flow_vis[0].cpu().numpy().transpose(1, 2, 0))
# plt.savefig("outputs/flow.png")
# print(flow_vis.max())
# save_image(flow_vis[0], "outputs/flow_vis.jpg")
