if __name__ == "__main__":
    import sys

    sys.path.append("src")
    from csi_sign_language.modules.resnet_focal_encoder_v2.resnet import ResNet
    from csi_sign_language.modules.resnet_focal_encoder_v2.components import (
        FocalAttention,
    )
else:
    from .resnet import ResNet
    from .components import FocalAttention

from einops.layers.torch import Reduce
from einops import rearrange
from collections import namedtuple, OrderedDict
import re
import torch
from torch import nn
from torchvision.transforms.functional import normalize


def build_resnet(arch):
    if arch == "resnet-18":
        return ResNet(depth=18, num_stages=4, out_indices=(0, 1, 2, 3))
    else:
        raise NotImplementedError(f"Architecture {arch} is not implemented")


# class MultiScaleFusion(nn.Module):
#     def __init__(self, embed_dims: List[int], *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#
#         self.token_projection = nn.ModuleList()
#         for i in range(len(embed_dims) - 1):
#             self.token_projection.append(nn.Linear(embed_dims[i], embed_dims[i + 1]))
#
#     def forward(self, tokens: Tuple[torch.Tensor]):
#         tokens = list(tokens)
#         for i in range(len(tokens) - 1):
#             tokens[i + 1] = tokens[i + 1] + self.token_projection[i](tokens[i])
#             # print('-'.join(str(A.shape) for A in tokens))
#         return tokens[-1]


class ResnetFocalEncoderV2(nn.Module):
    def __init__(
        self,
        arch,
        ckpt,
        focal_module_cfg={
            "embed_dims": [128, 256, 512],
            "heads": [8, 8, 8],
            "stage_index": [1, 2, 3],  # should be strictly increasing
        },
        drop_resnet=0.5,
        drop_hand=0.5,
        dropout_face=0.5,
        global_pooling="mean",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.resnet: ResNet = build_resnet(arch)
        self.dropout = nn.Dropout2d(drop_resnet)
        self.dropout_hand_token = nn.Dropout(drop_hand)
        self.dropout_face_token = nn.Dropout(dropout_face)

        self.gap = Reduce("b c h w -> b c", reduction=global_pooling)
        self.focal_attn_index = focal_module_cfg["stage_index"]

        ##assert strict increasing
        assert all(
            [
                focal_module_cfg["stage_index"][i]
                < focal_module_cfg["stage_index"][i + 1]
                for i in range(len(focal_module_cfg["stage_index"]) - 1)
            ]
        )

        self.focal_attentions = nn.ModuleDict()
        for i, stage_index in enumerate(focal_module_cfg["stage_index"]):
            embed_dims = focal_module_cfg["embed_dims"][i]
            heads = focal_module_cfg["heads"][i]
            feats_dims = self.resnet.out_channels[stage_index]
            self.focal_attentions[f"focal_attention_{stage_index}"] = FocalAttention(
                embed_dims=embed_dims,
                intoken_dims=focal_module_cfg["embed_dims"][i - 1]
                if i > 0
                else focal_module_cfg["embed_dims"][0],
                feats_dims=feats_dims,
                heads=heads,
            )
        # self.hand_fusion = MultiScaleFusion(focal_module_cfg["embed_dims"])
        # self.face_fusion = MultiScaleFusion(focal_module_cfg["embed_dims"])
        self.migrate_checkpoint(ckpt)

        # hand and face fusion
        self.hand_token = nn.Parameter(
            torch.randn(1, focal_module_cfg["embed_dims"][0]), requires_grad=True
        )
        self.face_token = nn.Parameter(
            torch.randn(1, focal_module_cfg["embed_dims"][0]), requires_grad=True
        )

    def migrate_checkpoint(self, ckpt):
        checkpoint = torch.load(ckpt, map_location="cpu")
        new_state_dict = OrderedDict()
        for key, value in checkpoint.items():
            if key.startswith("backbone."):
                new_key = re.sub("^backbone\.", "", key)
                new_state_dict[new_key] = value
        self.resnet.load_state_dict(new_state_dict, strict=False)
        del checkpoint

    ResnetFocalEncoderOut = namedtuple(
        "ResnetFocalEncoderOut", ["out", "t_length", "attn_weights"]
    )

    def forward(self, x, t_length):
        T = int(x.size(2))
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.pre_norm(x)
        x, hands_tokens, face_tokens, attn_weight = self.resnet_forward(x)
        x = self.dropout(x[-1])
        x = self.gap(x)
        x = rearrange(x, "(b t) c -> b c t", t=T)

        hand_token = hands_tokens[-1]
        face_token = face_tokens[-1]
        hand_token = self.dropout_hand_token(hand_token)
        face_token = self.dropout_face_token(face_token)

        hand_token, face_token = (
            rearrange(a, "(b t) c -> b c t", t=T) for a in (hand_token, face_token)
        )
        attn_weight = tuple(
            rearrange(a, "(b t) heads fh h w -> t b heads fh h w", t=T)
            for a in attn_weight
        )

        return self.ResnetFocalEncoderOut(
            # NOTE: see what happened when we remove GAP output from here, than added the
            # out=hand_token + x, t_length=t_length, attn_weights=attn_weight
            out=hand_token + x,
            t_length=t_length,
            attn_weights=attn_weight,
        )

    def pre_norm(self, x):
        x = normalize(
            x * 255.0, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
        )
        return x

    def resnet_forward(self, x):
        if self.resnet.deep_stem:
            x = self.resnet.stem(x)
        else:
            x = self.resnet.conv1(x)
            x = self.resnet.norm1(x)
            x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        outs = []
        hand_tokens = []
        face_tokens = []
        attn_weights = []

        hand_token = self.hand_token
        face_token = self.face_token
        for i, layer_name in enumerate(self.resnet.res_layers):
            res_layer = getattr(self.resnet, layer_name)
            x = res_layer(x)

            if i in self.focal_attn_index:
                hand_token, face_token, attn_weight = self.focal_attentions[
                    f"focal_attention_{i}"
                ](x, hand_token, face_token)
                hand_tokens.append(hand_token)
                face_tokens.append(face_token)
                attn_weights.append(attn_weight)

            if i == len(self.resnet.res_layers) - 1:
                x = self.dropout(x)
            if i in self.resnet.out_indices:
                outs.append(x)

        return tuple(outs), tuple(hand_tokens), tuple(face_tokens), tuple(attn_weights)


if __name__ == "__main__":
    model = ResnetFocalEncoderV2(
        "resnet-18", "resources/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth"
    ).to("cuda:1")
    x = torch.randn(2, 3, 16, 112, 112).to("cuda:1")
    t_length = torch.tensor([16, 16]).to("cuda:1")
    out = model(x, t_length)
    print(out.out.shape, out.t_length)
    for attn_weight in out.attn_weights:
        print(attn_weight.shape)
    # torch.Size([2, 512, 16]) tensor([16, 16]) torch.Size([2, 2, 8, 16, 16])
