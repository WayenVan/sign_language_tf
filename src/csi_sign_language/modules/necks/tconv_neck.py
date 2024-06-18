from ..components.tconv import *
from torch import nn
from collections import namedtuple
from einops import rearrange


class TemporalConvNeck(nn.Module):
    
    Ret = namedtuple('TemporalConvNeckOut', ['out', 't_length', 'feats'])
    def __init__(
        self, 
        in_channels,
        out_channels,
        bottle_channels,
        n_class,
        with_header = True,
        pooling = 'max',
        dropout= 0.1,
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.tconv= TemporalConv1D(in_channels, out_channels, bottle_channels, pooling=pooling, dropout=dropout)

        self.with_header = with_header
        if with_header:
            self.header = nn.Linear(out_channels, n_class)
        
    
    def forward(self, x, t_length):
        # [n c t]
        feats, t_length = self.tconv(x, t_length)
        
        x = rearrange(feats, 'n c t -> t n c')
        if self.with_header:
            out = self.header(x)
        else:
            out = None
        
        return self.Ret(
            out=out,
            t_length=t_length,
            feats=feats
        )