
import torch.nn as nn
from ..components.tconv import *

from collections import namedtuple
from ..components.transformer import TransformerEncoder

class TransformerDecoder(nn.Module):
    
    def __init__(self, n_class, d_model, n_heads, n_layers, d_feedforward, freeze=False, dropout=0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tf = TransformerEncoder(d_model, d_feedforward, n_heads, n_layers, dropout=dropout)
        self.header = nn.Linear(d_model, n_class)
        self.freeze = freeze
        
    def forward(self, x, t_length):
        x = self.tf(x, t_length)
        seq_out = x
        x = self.header(x)
        
        ret = namedtuple('TransformerDecoderOut', ['out', 't_length', 'seq_out'])
        return ret(
            out = x,
            t_length = t_length,
            seq_out = seq_out
        )

    def train(self, mode: bool = True):
        super().train(mode)
        for p in self.parameters():
            p.requires_grad = not self.freeze