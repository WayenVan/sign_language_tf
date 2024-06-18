import torch
from torch import nn
from einops import rearrange

class ConvStem(nn.Module):
    
    def __init__(self, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
    
    def forward(self, x):
        return self.conv(x)
    
class AdapterBlock(nn.Module):
    
    def __init__(
        self, 
        in_channels,
        feats_dim,
        out_channels,
        n_heads,
        down_sample = True,
        attn_dropout = 0.,
        conv_dropout = 0.,
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.linear = nn.Linear(feats_dim, in_channels)
        self.cross_attn = nn.MultiheadAttention(in_channels, n_heads, dropout=attn_dropout)
        self.layer_norm = nn.LayerNorm(in_channels)
        self.fast_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(conv_dropout)
        )
        if down_sample:
            self.down_sample = nn.MaxPool2d(2)
    
    def forward(self, x, f):
        """
        :param x: feature from last block [n, c, h, w]
        :param f: feature from vit block [n, d, l, s]
        """
        _, _, H, W = x.shape
        
        # f = nn.functional.interpolate(f, (H, W))
        f = rearrange(f, 'n d l s -> (l s) n d')
        f = self.linear(f)
        #[l*s, n, c]
        x = rearrange(x, 'n c h w -> (h w) n c')
        fused_x, _ = self.cross_attn(x, f, f)
        fused_x = self.layer_norm(x + fused_x)
        fused_x = rearrange(fused_x, '(h w) n c -> n c h w', h=H, w=W)
        
        conv_out = self.fast_conv(fused_x)

        if self.down_sample:
            return self.down_sample(conv_out)
        else:
            return conv_out