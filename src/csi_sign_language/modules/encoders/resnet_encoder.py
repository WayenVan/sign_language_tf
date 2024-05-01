import torch
from typing import List
import torch.nn as nn
import torch.nn.functional as F
from ..x3d import X3d
from ..externals.flownet2.models import FlowNet2SDConvDown, FlowNet2SD
from ..tconv import TemporalConv1D
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from collections import namedtuple

from mmpretrain.models.backbones.resnet import ResNet
from mmpretrain.registry import MODELS
import torch

from mmengine import load, build_model_from_cfg 
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from torch import nn

class ResnetEncoder(nn.Module):

    def __init__(self, cfg, ckpt, drop_prob=0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        cfg = Config.fromfile(cfg)
        model = build_model_from_cfg(cfg.model, MODELS)
        load_checkpoint(model, ckpt)
        self.resnet = model.backbone
        self.dropout = nn.Dropout2d(drop_prob)
        self.gap = model.neck
    
    def forward(self, x, t_length):
        T = int(x.size(2))
        x = rearrange(x, 'n c t h w -> (n t) c h w')
        x = self.resnet_forward(x)
        x = self.gap(x)[-1]
        x = rearrange(x, '(n t) c -> n c t', t=T)
        ret = namedtuple('ResnetEncoderOut', ['out', 't_length'])
        return ret(
            out=x,
            t_length=t_length,
        )

    def resnet_forward(self, x):
        if self.resnet.deep_stem:
            x = self.resnet.stem(x)
        else:
            x = self.resnet.conv1(x)
            x = self.resnet.norm1(x)
            x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        outs = []
        for i, layer_name in enumerate(self.resnet.res_layers):
            res_layer = getattr(self.resnet, layer_name)
            x = res_layer(x)

            if i == len(self.resnet.res_layers) - 1:
                x = self.dropout(x)
            if i in self.resnet.out_indices:
                outs.append(x)

        return tuple(outs)