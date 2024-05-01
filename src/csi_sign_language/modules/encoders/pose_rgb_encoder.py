import torch
from mmpose.apis.inference import init_model
from mmengine import Config

from mmpretrain.models.backbones.vision_transformer import VisionTransformer
from mmpretrain.models.backbones.swin_transformer import SwinTransformer, SwinBlock
from mmpretrain.registry import MODELS
from mmengine import build_model_from_cfg
from mmengine.runner import load_checkpoint

from mmpose.models.heads import HeatmapHead
from torch import nn
import torch
from einops import rearrange

from csi_sign_language.utils.data import mapping_0_1
from collections import namedtuple
import copy
import ctypes
from einops.layers.torch import Reduce
# class Fusion(nn.Module):
    
#     def __init__(self, 
#                  rgb_channels, 
#                  heatmap_channels, 
#                  feature_map_size, 
#                  drop_path_rate) -> None:
#         super().__init__()
#         self.linear = nn.Linear(rgb_channels+heatmap_channels, 512)
#         self.gap = Reduce('n c h w -> n c', 'mean')
    
#     def forward(self, rgb_feats, pose_feats):
#         #n c h w
#         feats = torch.cat([self.gap(rgb_feats), self.gap(pose_feats)], dim=-1)
#         return self.linear(feats)

class Fusion(nn.Module):
    
    def __init__(self, 
                 rgb_channels, 
                 heatmap_channels, 
                 feature_map_size, 
                 drop_path_rate) -> None:
        super().__init__()
        arch = {
            'embed_dims': 512,
            'num_layers': 1,
            'num_heads': 8,
            'feedforward_channels': 512 * 2
        }

        self.vit = VisionTransformer(
            arch = arch,
            img_size=feature_map_size,
            patch_size=2,
            in_channels=heatmap_channels+rgb_channels,
            drop_path_rate=drop_path_rate,
            with_cls_token=True,
            out_type='cls_token'
        )
    
    def forward(self, rgb_feats, pose_feats):
        #n c h w
        feats = torch.cat([rgb_feats, pose_feats], dim=-3)
        return self.vit(feats)[-1]

class PoseProjection(nn.Module):
    
    def __init__(self, embd_dims, num_keypoints, drop_path_rate, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        arch = {
            'embed_dims': embd_dims,
            'depths': [1, 4, 1],
            'num_heads': [3, 6, 12]
        }
        self.swin = SwinTransformer(
            arch=arch,
            patch_size=2,
            drop_path_rate=drop_path_rate,
            in_channels=num_keypoints,
            out_indices=(len(arch['depths']) - 1, ),
            out_after_downsample=True,
            pad_small_map=True
        )
        
    def forward(self, x):
        return self.swin(x)[-1]
    

class SwinPoseEncoder(nn.Module):
    
    def __init__(self,
                 swin_cfg,
                 swin_ckpt,
                 num_keypoints,
                 drop_path_rate,
                 pose_embd_dims = 96,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # default_scope.DefaultScope.overwrite_default_scope('mmpretrain')
        cfg = Config.fromfile(swin_cfg)
        setattr(cfg.model.backbone, 'out_indices', (0, 1, 2, 3))
        setattr(cfg.model.backbone, 'pad_small_map', True)
        model = build_model_from_cfg(cfg.model, MODELS)
        load_checkpoint(model, swin_ckpt, map_location='cpu')

        self.swin = model.backbone
        self.swin.out_indices = list(range(self.swin.num_layers))
        del model

        self.pose_header = HeatmapHead(
            in_channels=self.swin.num_features[-2],
            out_channels=num_keypoints,
            deconv_out_channels=(256, 256),
            deconv_kernel_sizes=(4, 4),
        )
        self.pose_projection = PoseProjection(
            pose_embd_dims,
            num_keypoints,
            drop_path_rate,
        ) 
        self.fuse = Fusion(
            self.swin.num_features[-1],
            self.pose_projection.swin.num_features[-1],
            feature_map_size=64,
            drop_path_rate=drop_path_rate
        )

        
    def forward(self, x, t_length):

        #n c t h w
        N, C, T, H, W = x.shape
        x = rearrange(x, 'n c t h w -> (n t) c h w')

        swin_outputs = self.swin(x)
        rgb_feats = swin_outputs[-1]

        heatmap = self.pose_header((swin_outputs[-2], ))
        heatmap_feats = self.pose_projection(heatmap)

        #[(n t) c]
        fused_feats = self.fuse(rgb_feats, heatmap_feats)
        fused_feats = rearrange(fused_feats, '(n t) c -> n c t', n=N)
        
        Out = namedtuple('SwinPoseEncoderOut', ['out', 't_length', 'heatmap'])
        return Out(
            out = fused_feats,
            t_length = t_length,
            heatmap = rearrange(heatmap, '(n t) c h w -> n c t h w', n=N)
        )
        



