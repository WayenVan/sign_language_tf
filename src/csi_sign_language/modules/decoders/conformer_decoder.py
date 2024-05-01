from torch import nn
from einops import rearrange
from conformer import ConformerBlock, Conformer
import torch
from collections import namedtuple

class ConformerDecoder(nn.Module):
    
    def __init__(self, 
                 dim,
                 dim_head,
                 depth,
                 heads,
                 conv_kernel_size,
                 n_class,
                 attn_dropout=0.,
                 ff_dropout=0.,
                 conv_dropout=0.,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.blocks = nn.ModuleList([
            ConformerBlock(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                ff_mult=2,
                conv_kernel_size=conv_kernel_size,
                attn_dropout=attn_dropout,
                conv_dropout=conv_dropout,
                ff_dropout=ff_dropout
            ) for i in range(depth)
        ])
        self.linear = nn.Linear(dim, n_class)


    def forward(self, x, t_length):
        #[t n c] [n]
        t = int(x.size(0))
        x = rearrange(x, 't n c -> n t c')
        mask = self._make_video_mask(t_length, t)
        for block in self.blocks:
            x = block(x, mask)
        x = rearrange(x, 'n t c -> t n c')
        
        seq_out = x
        out = self.linear(x)

        Ret = namedtuple('ConformerDecoderOut', ['out', 't_length', 'seq_out'])
        return Ret(
            out=out,
            t_length=t_length,
            seq_out=seq_out
        )

    @staticmethod
    def _make_video_mask(video_length: torch.Tensor, temporal_dim):
        batch_size = video_length.size(dim=0)
        mask = torch.zeros(batch_size, temporal_dim)
        for idx in range(batch_size):
            mask[idx, :video_length[idx]] = 1
        return mask.bool().to(video_length.device)