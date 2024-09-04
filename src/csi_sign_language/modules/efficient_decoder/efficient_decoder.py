import torch
from collections import namedtuple
from torch import bucketize, nn
from einops import rearrange
from xformers.components.feedforward import MLP
from xformers.components.activations import Activation

if __name__ == "__main__":
    import sys

    sys.path.append("src")
    from csi_sign_language.modules.components.drop_path import DropPath
    from csi_sign_language.modules.efficient_decoder.efficient_attention import (
        # SparseAttention,
        BucketRandomAttention,
        # DiagonalMaskGenerator,
        # RandomBucketMaskGenerator,
    )

else:
    from ..components.drop_path import DropPath
    from .efficient_attention import (
        # SparseAttention,
        BucketRandomAttention,
        # DiagonalMaskGenerator,
        # RandomBucketMaskGenerator,
    )


class ConvHeader(nn.Module):
    """
    DPConv -> AvgPool(2) -> DPConv -> AvgPool(2)
    outputlenth = inputlength / 4
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        neck_channels,
        kernerl_size=5,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernerl_size,
                groups=in_channels,
                padding=kernerl_size // 2,
                bias=False,
            ),
            nn.Conv1d(in_channels, neck_channels, 1, padding=0, bias=False),
            nn.BatchNorm1d(neck_channels),
            nn.GELU(),
        )
        self.pool1 = nn.AvgPool1d(2)

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                neck_channels,
                neck_channels,
                kernerl_size,
                groups=neck_channels,
                padding=kernerl_size // 2,
                bias=False,
            ),
            nn.Conv1d(neck_channels, out_channels, 1, padding=0, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        self.pool2 = nn.AvgPool1d(2)

    def forward(self, x, t_length):
        # [t b c]
        x = rearrange(x, "t b c -> b c t")
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = rearrange(x, "b c t -> t b c")
        t_length = t_length // 4
        return x, t_length


class EfficientDecoderBlock(nn.Module):
    """
    Depth Convolution -> BN -> SparseAttention -> BN -> MLP -> BN
    """

    def __init__(
        self,
        dim,
        n_head,
        bucket_size,
        conv_kernel_size=3,
        ff_dropout=0.0,
        drop_path=0.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.conv = nn.Conv1d(
            dim, dim, conv_kernel_size, padding=1, bias=False, groups=dim
        )
        self.bn = nn.BatchNorm1d(dim)

        self.attn = BucketRandomAttention(
            dim,
            n_head,
            bucket_size=bucket_size,
        )

        self.ff = MLP(
            dim,
            ff_dropout,
            Activation.GeLU,
            hidden_layer_multiplier=2,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x, t_length):
        x = rearrange(x, "t b c -> b c t")
        x = self.conv(x) + self.drop_path(x)
        x = self.bn(x)
        x = rearrange(x, "b c t -> t b c")

        x = self.attn(x, x, x, key_length=t_length)[0] + self.drop_path(x)
        x = self.norm1(x)
        x = self.ff(x) + self.drop_path(x)
        x = self.norm2(x)
        return x


class EfficientDecoder(nn.Module):
    def __init__(
        self,
        input_dims,
        d_model,
        n_layers,
        n_head,
        n_classes,
        bucket_size,
        conv_kernel_head=5,
        conv_kernel_block=3,
        ff_dropout=0.0,
        drop_path=0.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.head = ConvHeader(input_dims, d_model, d_model, conv_kernel_head)

        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(
                EfficientDecoderBlock(
                    d_model,
                    n_head,
                    bucket_size,
                    conv_kernel_block,
                    ff_dropout,
                    drop_path,
                )
            )

        self.linear = nn.Linear(d_model, n_classes)

    EfficientDecoderOut = namedtuple("EfficientDecoderOut", ["out", "t_length"])

    def forward(self, x, t_length):
        """
        @param x: [t b c]
        """
        x, t_length = self.head(x, t_length)

        for block in self.blocks:
            x = block(x, t_length)
        x = self.linear(x)
        return self.EfficientDecoderOut(x, t_length)


if __name__ == "__main__":
    model = EfficientDecoder(50, 80, 3, 8, 1296, 4, 5, 3, 0.0, 0.0)
    x = torch.rand(20, 2, 50)
    t_length = torch.tensor([20, 20], dtype=torch.int64)
    out = model(x, t_length)
    print(out.out.shape, out.t_length)
