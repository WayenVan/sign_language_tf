from mmengine import Config

from mmpretrain.models.backbones.mobilevit import MobileViT
from mmpretrain.registry import MODELS
from mmengine import build_model_from_cfg
from mmengine.runner import load_checkpoint

from torch import nn
from einops import rearrange

from collections import namedtuple

class MobiViTEncoder(nn.Module):
    
    MobileViTEncoderOut = namedtuple('MobileViTEncoderOut', ['out', 't_length'])

    def __init__(
        self,
        cfg,
        ckpt,
        *args, **kwargs)-> None:
        super().__init__(*args, **kwargs)
        cfg = Config.fromfile(cfg)

        self.cfg = cfg
        model = build_model_from_cfg(cfg.model, MODELS)
        load_checkpoint(model, ckpt, map_location='cpu')
        self.mbvit = model.backbone
        self.gap =model.neck
        del model

    
    def forward(self, x, t_length):
        #n c t h w
        T = int(x.size(2))
        x = rearrange(x, 'n c t h w -> (n t) c h w')

        
        feats = self.mbvit(x)
        feats = self.gap(feats)
        x = feats[-1]
        x = rearrange(x, '(n t) c -> n c t', t=T)

        return self.MobileViTEncoderOut(
            out=x,
            t_length=t_length
        )
