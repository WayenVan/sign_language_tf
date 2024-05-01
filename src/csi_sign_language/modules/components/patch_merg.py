from torch import nn
from mmpretrain.models.backbones.swin_transformer import PatchMerging, SwinBlock
from einops.layers.torch import Rearrange

class PatchMerge1D(nn.Module):
    
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size,
                 norm=True,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.sampler= nn.Sequential(
            Rearrange('n c t -> n c t ()'),
            nn.Unfold(
                kernel_size=(kernel_size, 1),
                stride=(kernel_size, 1),
            )
        )

        self.reduction = nn.Sequential(
            Rearrange('n c t -> n t c'),
            nn.LayerNorm(in_channels*kernel_size) if norm else nn.Identity(),
            nn.Linear(in_channels*kernel_size, out_channels, bias=False),
            Rearrange('n t c -> n c t')
        )

    def forward(self, x):
        x = self.sampler(x)
        x = self.reduction(x)
        return x 