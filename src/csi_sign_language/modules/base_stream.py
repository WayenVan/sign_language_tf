import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from ...modules.bilstm import BiLSTMLayer
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from collections import namedtuple

from csi_sign_language.utils.misc import add_attributes


class BaseStream(nn.Module):

    ret = namedtuple('BaseStreamOut', ['out', 't_length', 'encoder_out'])

    def __init__(self, encoder, decoder, neck=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.rearrange = Rearrange('n c t -> t n c')
        self.decoder = decoder
        self.neck = neck

    def forward(self, x, t_length):
        """
        :param x: [n, c, t, h, w]
        """
        encoder_out = self.encoder(x, t_length)
        
        x = encoder_out.out
        t_length = encoder_out.t_length
        
        if self.neck is not None:
            x, t_length = self.neck(x, t_length)

        x = self.rearrange(x)
        decoder_out = self.decoder(x, t_length)

        return self.ret(
            out = decoder_out.out,
            t_length = decoder_out.t_length,
            encoder_out = encoder_out
        )
