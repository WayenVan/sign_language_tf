import torch.nn as nn
from einops.layers.torch import Rearrange
from collections import namedtuple


class MultiStageStream(nn.Module):
    """
    encoder -> <neck> -> decoder -> multi-stage modifier
    """

    def __init__(
        self, encoder, decoder, multi_stage_modifier, neck=None, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.rearrange = Rearrange("n c t -> t n c")
        self.decoder = decoder
        self.multi_stage_modifier = multi_stage_modifier
        self.neck = neck

    Out = namedtuple(
        "MultiStageStreamOut",
        [
            "out",
            "t_length",
            "encoder_out",
            "neck_out",
            "decoder_out",
            "multi_stage_out",
        ],
    )

    def forward(self, x, t_length):
        """
        :param x: [b, c, t, h, w]
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
        multi_stage_out = self.multi_stage_modifier(decoder_out.out, t_length)

        return self.Out(
            out=multi_stage_out.out,
            t_length=multi_stage_out.t_length,
            decoder_out=decoder_out,
            multi_stage_out=multi_stage_out,
            encoder_out=encoder_out,
            neck_out=neckout,
        )
