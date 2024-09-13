if __name__ == "__main__":
    import sys

    sys.path.append("/root/projects/sign_language_transformer/src")

import torch
import numpy as np
from torch import nn
from torch import Tensor
from mmengine.config import Config
from mmengine.registry.utils import init_default_scope
from mmpose.apis import init_model
from csi_sign_language.utils.data import mapping_0_1
from einops import rearrange, repeat
from torch import Tensor
import torch.nn.functional as F
from mmpose.models import TopdownPoseEstimator
from torchvision import transforms
from torchvision.transforms.functional import normalize, resize


class MultiTaskDistillLoss(nn.Module):
    def __init__(
        self,
        dw_pose_cfg: str,
        dw_pose_ckpt: str,
        dw_pose_input_size: tuple[int, int],
        ctc_weight: float = 1.0,
        distill_weight: float = 1.0,
        temperature: float = 8.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        cfg = Config.fromfile(dw_pose_cfg)
        init_default_scope("mmpose")
        self.model = init_model(cfg, dw_pose_ckpt, device="cpu")

        self.ctc_weight = ctc_weight
        self.distill_weight = distill_weight

        self.mean = cfg.model.data_preprocessor.mean
        self.std = cfg.model.data_preprocessor.std
        self.simcc_split_ratio = cfg.model.head.simcc_split_ratio
        self.dw_pose_input_size = dw_pose_input_size
        self.temperature = temperature

        self._loss_ctc = nn.CTCLoss(blank=0, reduction="none")
        self._freeze_pose_model()

    def _freeze_pose_model(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def _preprocess(self, x):
        x = resize(x, list(self.dw_pose_input_size))
        return normalize(x, self.mean, self.std)

    def forward(self, outputs, input, input_length, target, target_length):
        loss = 0.0
        if self.ctc_weight > 0.0:
            out = nn.functional.log_softmax(outputs.out, dim=-1)
            ctc_loss = self._loss_ctc(
                out, target, outputs.t_length.cpu().int(), target_length.cpu().int()
            ).mean()
            loss += self.ctc_weight * ctc_loss

        if self.distill_weight > 0.0:
            distill_loss = self.distill_loss(
                outputs.encoder_out.simcc_out_x, outputs.encoder_out.simcc_out_y, input
            )
            loss += self.distill_weight * distill_loss
        return loss

    def distill_loss(self, out_x: Tensor, out_y: Tensor, input: Tensor):
        """
        @param out_x, out_y, x, y: [t b k l]
        @pararm input: [b c t h w]
        """
        T = out_x.shape[0]
        input = rearrange(input, "b c t h w -> (b t) c h w")
        target_x, target_y = self.pose_forward(input)
        out_x, out_y = (rearrange(a, "t b k l -> (b t) k l") for a in (out_x, out_y))

        # assertion
        assert target_x.shape == out_x.shape
        assert target_y.shape == out_y.shape

        # x, y are all logits here
        # apply termperature
        target_x, target_y = (a / self.temperature for a in (target_x, target_y))
        # log_softmax
        target_x, target_y = (
            nn.functional.softmax(a, dim=-1) for a in (target_x, target_y)
        )
        out_x, out_y = (nn.functional.log_softmax(a, dim=-1) for a in (out_x, out_y))

        return nn.functional.kl_div(out_x, target_x.detach()) + nn.functional.kl_div(
            out_y, target_y.detach(), log_target=True
        )

    @torch.no_grad()
    def pose_forward(self, input):
        """
        @param input: [b c h w]
        """
        input = self._preprocess(input)
        x, y = self.model(input, None, "tensor")
        return x, y


if __name__ == "__main__":
    ckpt = "/root/projects/sign_language_transformer/resources/dwpose-l/dw-ll_ucoco.pth"
    cfg = "/root/projects/sign_language_transformer/resources/dwpose-l/rtmpose-l_8xb64-270e_coco-ubody-wholebody-256x192.py"

    loss = MultiTaskDistillLoss(
        dw_pose_cfg=cfg,
        dw_pose_ckpt=ckpt,
        dw_pose_input_size=(256, 192),
        ctc_weight=1.0,
        distill_weight=1.0,
    )
    classes = 100
    B = 2
    T = 100
    L = 5
    H = 256
    W = 192
    K = 133

    # Dummy data for testing
    outputs = type(
        "Outputs",
        (object,),
        {
            "out": torch.randn(T // 4, B, classes),  # Example shape
            "t_length": torch.tensor([T // 4] * B),
            "simcc_out_x": torch.randn(T, B, K, W * 2),  # Example shape
            "simcc_out_y": torch.randn(T, B, K, H * 2),  # Example shape
        },
    )()
    input = torch.randn(B, 3, T, H, W)  # Example shape
    input_length = torch.tensor([T] * B)
    target = torch.randint(1, classes, (B, L))  # Example shape
    target_length = torch.tensor([L] * B)

    # Calculate loss
    loss_value = loss(outputs, input, input_length, target, target_length)
    print(f"Calculated loss: {loss_value.item()}")
