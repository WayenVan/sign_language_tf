import torch
import torch.nn as nn
from collections import namedtuple
from mmpretrain.models.backbones.resnet import ResNet
from mmpretrain.registry import MODELS
from mmengine import build_model_from_cfg
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmpose.models.heads import HeatmapHead
from einops import rearrange


if __name__ == "__main__":
    import sys

    sys.path.append("./src")
    from csi_sign_language.modules.resnet_distill.simcc import SimCCHead
else:
    from .simcc import SimCCHead


class ResnetDistEncoderV2(nn.Module):
    """
                                 |-> key_y
    resnet -> SimCCHead/RTMCCHead -> key_x
        -- -> CSLRhaed -> x, t_length
    """

    def __init__(
        self,
        cfg,
        ckpt: str,
        input_size: tuple[int, int],
        n_keypoints_simcc: int,
        n_keypoints_heatmap: int,
        simcc_x_samples: int,
        simcc_y_samples: int,
        drop_prob=0.1,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        cfg = Config.fromfile(cfg)
        model = build_model_from_cfg(cfg.model, MODELS)
        load_checkpoint(model, ckpt)
        self.resnet: ResNet = model.backbone

        ## CSLR bracnh
        self.dropout = nn.Dropout2d(drop_prob)
        self.gap = model.neck

        ## SimmCC bracnh
        # this is formed H, W, should be revers for the next
        self.feats_map_size: tuple[int, int] = (
            input_size[0] // 32,
            input_size[1] // 32,
        )
        self.pose_header = SimCCHead(
            in_channels=self.resnet.feat_dim,
            out_channels=n_keypoints_simcc,
            # W H
            input_size=(input_size[1], input_size[0]),
            # W H
            in_featuremap_size=(self.feats_map_size[1], self.feats_map_size[0]),
            simcc_x_samples=simcc_x_samples,
            simcc_y_samples=simcc_y_samples,
        )

        ## heatmap bracnh
        self.heatmap_header = HeatmapHead(
            in_channels=self.resnet.feat_dim,
            out_channels=n_keypoints_heatmap,
        )

    ResnetDistEncoderOut = namedtuple(
        "ResnetDistEncoderOut",
        ["out", "t_length", "simcc_out_x", "simcc_out_y", "heatmap"],
    )

    def forward(self, x, t_length):
        """
        @param x: (b, c, t, h, w)
        @param t_length: (b,)
        @return: out (b, c, t), simcc_out_x (t, b, k, l), simcc_out_y (t, b, k, l), t_length (b,)
        """
        T = int(x.size(2))
        x = rearrange(x, "n c t h w -> (n t) c h w")
        x = self.resnet_forward(x)

        # simcc_out
        simcc_out_x, simcc_out_y = self.pose_header(x)
        simcc_out_x, simcc_out_y = (
            rearrange(a, "(n t) k l -> t n k l", t=T)
            for a in [simcc_out_x, simcc_out_y]
        )

        # heatmap_out
        heatmap = self.heatmap_header(x)
        heatmap = rearrange(heatmap, "(n t) k h w -> t n k h w", t=T)

        out = self.gap(x)[-1]
        out = rearrange(out, "(n t) c -> n c t", t=T)
        return self.ResnetDistEncoderOut(
            out,
            t_length,
            simcc_out_x,
            simcc_out_y,
            heatmap,
        )

    def resnet_forward(self, x):
        if self.resnet.deep_stem:
            x = self.resnet.stem(x)
        else:
            x = self.resnet.conv1(x)
            x = self.resnet.norm1(x)
            x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        outs = []
        for i, layer_name in enumerate(self.resnet.res_layers):
            res_layer = getattr(self.resnet, layer_name)
            x = res_layer(x)

            if i == len(self.resnet.res_layers) - 1:
                x = self.dropout(x)
            if i in self.resnet.out_indices:
                outs.append(x)

        return tuple(outs)


if __name__ == "__main__":
    # a test for ResnetDistEncoder
    model = ResnetDistEncoder(
        cfg="/root/resources/resnet/resnet18.py",
        ckpt="/root/resources/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth",
        input_size=(224, 224),
        n_keypoints_simcc=144,
        n_keypoints_heatmap=17,
        simcc_x_samples=192 * 2,
        simcc_y_samples=256 * 2,
    )
    input_data = torch.randn(1, 3, 16, 224, 224)
    output = model(input_data, torch.tensor([16]))

    for k, v in output._asdict().items():
        print(k, v.shape)
