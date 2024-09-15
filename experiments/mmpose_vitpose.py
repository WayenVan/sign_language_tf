import sys

sys.path.append("./src")
from typing import Dict

import cv2
import numpy as np
import torch
from einops import repeat
from mmengine.config import Config
from torch import Tensor, nn
from torchvision.transforms import functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from warnings import warn
from mmpose.models.heads import HeatmapHead

from mmpose.apis import init_model


class ViTPoseWarpper(nn.Module):
    """
    A wrapper for LiteFlowNet model.
    This wrapper aims to run in evaluation mode. and allows the user to define how many decoder levels to use.
    So that won't waste time on the decoder levels that are not needed.
    """

    def __init__(self, cfg: str, ckpt: str):
        """
        @param n_decoder_levels: The number of decoder levels to use. If None, all decoder levels are used.
        """
        super(ViTPoseWarpper, self).__init__()
        _cfg = Config.fromfile(cfg)
        _ckpt = torch.load(ckpt)
        self.input_size = ()
        self.register_buffer("std", torch.tensor(_cfg.model.data_preprocessor.std))
        self.register_buffer("mean", torch.tensor(_cfg.model.data_preprocessor.mean))
        self.vitpose = init_model(_cfg, ckpt, device="cpu")
        self.freeze()

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """
        @param x: The input tensor of shape (B, C, H, W), should be normalized to [0, 1]
        @return: The output tensor of shape (B, 17, H//4, W//4)
        """
        assert (
            x.max().item() <= 1.0001
        ), "The input tensor should be normalized to [0, 1]"
        assert (
            x.min().item() >= -0.0001
        ), "The input tensor should be normalized to [0, 1]"

        # transer to [0, 255] for adapt the std and mean
        x = x * 255.0
        x = F.normalize(x, self.mean, self.std)
        return self.vitpose(x, None, "tensor")

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


device = "cuda:0"
# Read an image
image = cv2.imread("./outputs/frame_0.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Map the image to [0, 1]
image = image.astype(np.float32)
image = F.to_tensor(image)
image = F.resize(image, [224, 224])
image = image / 255.0
image.unsqueeze_(0)
image = image.to(device)

model = ViTPoseWarpper(
    "resources/ViTPose/td-hm_ViTPose-small_8xb64-210e_coco-256x192.py",
    "resources/ViTPose/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth",
)
model.to(device)


output = model(image)
print(output.shape)

plt.imshow(output[0, 0].cpu().numpy(), cmap="hot", interpolation="nearest")
plt.savefig("outputs/pose.png")
