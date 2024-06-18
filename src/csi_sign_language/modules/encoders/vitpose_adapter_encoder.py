from mmpose.apis.inference import init_model
from mmengine import Config

from mmpretrain.models.backbones.vision_transformer import VisionTransformer
from torch import nn
import torch
from einops import rearrange, reduce

from csi_sign_language.utils.data import mapping_0_1
from collections import namedtuple

from csi_sign_language.modules.adapters.conv_adapter import ConvStem, AdapterBlock


class VitPoseAdapterEncoder(nn.Module):
    
    VitPoseAdapterEncoderOut = namedtuple('VitPoseAdapterEncoderOut', ['out', 't_length'])

    def __init__(
        self, 
        color_range, 
        cfg_path, 
        checkpoint, 
        adapter_archs={
            'stem_channel': 32,
            'num_heads': 6,
            'channels': (32, 64, 128, 256),
            'vit_feats_indices': [1, 3, 5, 7],
        },
        dropout=0.5,
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        cfg = Config.fromfile(cfg_path)

        self.cfg = cfg
        self.color_range = color_range

        self.register_buffer('std', torch.tensor(cfg.model.data_preprocessor.std))
        self.register_buffer('mean', torch.tensor(cfg.model.data_preprocessor.mean))

        vitpose = init_model(cfg, checkpoint, device='cpu', cfg_options={
            'model': {'backbone':{'out_indices': adapter_archs['vit_feats_indices']}}
        })
        self.vit = vitpose.backbone
        del vitpose
        
        vit_out_channels = cfg.model.backbone.arch.embed_dims
        self.vit_feats_indices = adapter_archs['vit_feats_indices']
        self.adapter_stem = ConvStem(adapter_archs['channels'][0])
        self.adapter_blocks = nn.ModuleList()
        self.n_adapter_block = len(self.vit_feats_indices)
        self.adapter_dropout = nn.Dropout2d(dropout)
        for i in range(self.n_adapter_block):
            in_channels = adapter_archs['stem_channel'] if i == 0 else adapter_archs['channels'][i-1]
            self.adapter_blocks.append(
                AdapterBlock(
                    in_channels = in_channels,
                    out_channels = adapter_archs['channels'][i],
                    feats_dim = vit_out_channels,
                    n_heads = adapter_archs['num_heads'],
                )
            )
        self.freeze_vit()
            
    
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
        
        intermediate_feats = self.vit(x)

        #forward
        adapter_out = self.forward_adapter(x, intermediate_feats)

        x = reduce(adapter_out, 'n c h w -> n c', 'mean')
        x = rearrange(x, '(n t) c -> n c t', t=T)

        return self.VitPoseAdapterEncoderOut(
            out=x,
            t_length=t_length
        )
        
    def forward_adapter(self, x, feats):
        x = self.adapter_stem(x)
        for i in range(self.n_adapter_block):
            x = self.adapter_blocks[i](x, feats[i])
        x = self.adapter_dropout(x)
        return x

    def freeze_vit(self):
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.vit.eval()
