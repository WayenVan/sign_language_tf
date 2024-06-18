import torch.nn as nn
from einops.layers.torch import Rearrange
from collections import namedtuple



class BaseStream(nn.Module):

    ret = namedtuple('BaseStreamOut', ['out', 't_length', 'encoder_out', 'neck_out'])

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
        
        neckout = None
        if self.neck is not None:
            neckout = self.neck(x, t_length)
            x = neckout.feats
            t_length = neckout.t_length

        x = self.rearrange(x)
        decoder_out = self.decoder(x, t_length)

        return self.ret(
            out = decoder_out.out,
            t_length = decoder_out.t_length,
            encoder_out = encoder_out,
            neck_out = neckout
        )
