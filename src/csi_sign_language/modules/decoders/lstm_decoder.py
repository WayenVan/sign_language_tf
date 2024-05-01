from ..bilstm import BiLSTMLayer
from torch import nn
from einops import rearrange
from collections import namedtuple


class LSTMDecoder(nn.Module):
    
    LSTMDecoderOut = namedtuple('LSTMDecoderOut', ['out', 't_length', 'seq_out'])
    def __init__(self, 
                 input_size,
                 hidden_size,
                 n_layers,
                 n_class,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lstm = BiLSTMLayer(
            input_size,
            hidden_size=hidden_size,
            num_layers=n_layers
        )
        self.header = nn.Linear(hidden_size, n_class)
        
    def forward(self, x, t_length):
        #[t, n c], [n]
        seq_out = self.lstm(x, t_length)['predictions']
        x = self.header(seq_out)
        return self.LSTMDecoderOut(
            out=x,
            t_length=t_length,
            seq_out=seq_out
        )
