from mmpose.apis.inference import init_model
from mmengine import Config

from mmpretrain.models.backbones.vision_transformer import VisionTransformer
from torch import nn
import torch
from einops import rearrange

from csi_sign_language.utils.data import mapping_0_1
from collections import namedtuple


class VitPoseEncoder(nn.Module):
    
    def __init__(self, img_size, color_range, cfg_path, checkpoint, drop_path_rate, vit_pool_arch=None, freeze_vitpose=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        cfg = Config.fromfile(cfg_path)

        self.cfg = cfg
        self.color_range = color_range
        self.freeze_vitpose = freeze_vitpose
        self.register_buffer('std', torch.tensor(cfg.model.data_preprocessor.std))
        self.register_buffer('mean', torch.tensor(cfg.model.data_preprocessor.mean))

        vitpose = init_model(cfg, checkpoint, device='cpu')
        self.vit = vitpose.backbone
        # self.vit_head = vitpose.head
        del vitpose

        vit_out_channels = cfg.model.backbone.arch.embed_dims
        if vit_pool_arch is None:
            arch = {
                'embed_dims': vit_out_channels,
                'num_layers': 4,
                'num_heads': 8,
                'feedforward_channels': vit_out_channels * 2
            }
        else:
            arch = vit_pool_arch
        self.vitpool = VisionTransformer(
            arch = arch,
            img_size=(img_size + 4)//16,
            patch_size=2,
            in_channels=vit_out_channels,
            drop_path_rate=drop_path_rate,
            with_cls_token=True,
            out_type='cls_token'
        )
    
    
    def _data_preprocess(self, x):
        x = mapping_0_1(self.color_range, x)
        x = x * 255. #mapping to 0-255
        x = x.permute(0, 2, 3, 1)
        x = (x - self.mean) / self.std
        x = x.permute(0, 3, 1, 2)
        return x
    
    def forward(self, x, t_length):
        #n c t h w
        T = int(x.size(2))
        x = rearrange(x, 'n c t h w -> (n t) c h w')
        x = self._data_preprocess(x)
        
        feats = self.vit(x)
        x = self.vitpool(feats[-1])[-1]
        
        x = rearrange(x, '(n t) c -> n c t', t=T)

        VitPoseEncoderOut = namedtuple('VitPoseEncoderOut', ['out', 't_length'])
        return VitPoseEncoderOut(
            out=x,
            t_length=t_length
        )

    def train(self, mode: bool = True):
        super().train(mode)

        # for p in self.vit.parameters():
        #     p.requires_grad = not self.freeze_vitpose
        # for p in self.vit_head.parameters():
        #     p.requires_grad = not self.freeze_vitpose
            
        # if self.freeze_vitpose:
        #     self.vit.eval()
        #     self.vit_head.eval()
