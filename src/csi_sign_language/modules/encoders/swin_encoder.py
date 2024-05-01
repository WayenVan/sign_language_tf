from mmengine import Config

from mmpretrain.models.backbones.mobilevit import MobileViT
from mmpretrain.registry import MODELS
from mmengine import build_model_from_cfg
from mmengine.runner import load_checkpoint

from torch import nn
from einops import rearrange

from collections import namedtuple

class SwinEncoder(nn.Module):
    
    SwinEncoderOut = namedtuple('SwinEncoderOut', ['out', 't_length'])

    def __init__(
        self,
        cfg,
        ckpt,
        *args, **kwargs)-> None:
        super().__init__(*args, **kwargs)
        # default_scope.DefaultScope.overwrite_default_scope('mmpretrain')
        cfg = Config.fromfile(cfg)
        setattr(cfg.model.backbone, 'pad_small_map', True)
        model = build_model_from_cfg(cfg.model, MODELS)
        load_checkpoint(model, ckpt, map_location='cpu')
        self.swin = model.backbone
        self.gap = model.neck
        del model

    def forward(self, x, t_length):
        #n c t h w
        T = int(x.size(2))
        x = rearrange(x, 'n c t h w -> (n t) c h w')

        feats = self.swin(x)
        feats = self.gap(feats)
        x = feats[-1]
        x = rearrange(x, '(n t) c -> n c t', t=T)

        return self.SwinEncoderOut(
            out=x,
            t_length=t_length
        )
