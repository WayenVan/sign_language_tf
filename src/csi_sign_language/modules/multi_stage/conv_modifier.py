import torch
import torch.nn as nn


class ConvModifier(nn.Module):
    def __init__(self, conv, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = conv

    def forward(self, x):
        x = self.conv(x)
        return x
