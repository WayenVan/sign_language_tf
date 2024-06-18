import torch
import torch.nn as nn
from einops import rearrange
from ...utils.misc import add_attributes

class TransformerEncoder(nn.Module):
    
    def __init__(
        self,
        d_model,
        d_feedforward,
        n_head,
        n_layers,
        dropout=0.1,
        ) -> None:
        super().__init__()
        add_attributes(self, locals())

        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward=d_feedforward, dropout=dropout)
        self.trans_decoder = nn.TransformerEncoder(encoder_layer, n_layers)
    
    def forward(self, x, seq_length):
        """
        :param x: [t, n, d]
        :param sequence_length: [n]
        """

        mask = self._make_video_mask(seq_length, x.size(dim=0))
        x = x + self.trans_decoder(x, src_key_padding_mask=mask)

        return x

    @staticmethod
    def _make_video_mask(video_length: torch.Tensor, temporal_dim):
        batch_size = video_length.size(dim=0)
        mask = torch.ones(batch_size, temporal_dim)
        for idx in range(batch_size):
            mask[idx, :video_length[idx]] = 0
        return mask.bool().to(video_length.device)