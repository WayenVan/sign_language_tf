from ..components.tconv import *
from torch import nn
from collections import namedtuple
from einops import rearrange
from einops.layers.torch import Reduce


class PoolNeck(nn.Module):
    
    Ret = namedtuple('TemporalConvNeckOut', ['out', 't_length', 'feats'])
    def __init__(
        self, 
        in_channels,
        n_class,
        kernel,
        with_header = True,
        type = 'max',
        dropout= 0.1,
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.kernel = kernel

        self.pool = nn.Sequential(
            self.build_pool(kernel, type),
            self.build_pool(kernel, type)
        )

        self.with_header = with_header
        if with_header:
            self.header = nn.Linear(in_channels, n_class)
    
    
    @staticmethod
    def build_pool(kernel, type):
        if type == 'max':
            return nn.MaxPool1d(kernel)
        if type == 'avg':
            return nn.AvgPool1d(kernel)
        
        
    
    def forward(self, x, t_length):
        # [n c t]
        t_length = (t_length // self.kernel) // self.kernel
        feats = self.pool(x)
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