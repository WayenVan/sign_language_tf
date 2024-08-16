if __name__ == '__main__':
    import sys
    sys.path.append('/root/projects/sign_language_transformer/src')
from csi_sign_language.modules.heatmap_encoder.vision_transformer import VisionTransformer


from collections import OrderedDict, namedtuple
import torch
from torch import nn
import re
from einops import rearrange

def build_vit(arch):
    if arch == 'vit-base-p32':
        vit = VisionTransformer(
            arch='b',
            img_size=224,
            patch_size=32,
            drop_rate=0.1,
            init_cfg=[
                dict(
                    type='Kaiming',
                    layer='Conv2d',
                    mode='fan_in',
                    nonlinearity='linear')
            ],
            heatmap_header_cfg=dict(
                index=[2, 4, 6, 8, 10],
                kernel_size=3))
    elif arch == 'vit-base-p16':
        vit = VisionTransformer(
            arch='b',
            img_size=224,
            patch_size=16,
            drop_rate=0.1,
            init_cfg=[
                dict(
                    type='Kaiming',
                    layer='Conv2d',
                    mode='fan_in',
                    nonlinearity='linear')
            ],
            heatmap_header_cfg=dict(
                index=[2, 4, 6, 8, 10],
                kernel_size=5))
    else:
        raise ValueError(f'Unrecognized Vision Transformer architecture {arch}')
    return vit
    
class ViTHeatmapEncoder(nn.Module):
    
    def __init__(
        self, 
        arch,
        checkpoint = None,
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vit: VisionTransformer = build_vit(arch)

        if checkpoint is not None:
            self.migrate_checkpoint(checkpoint)

        

    ViTHeatmapEncoderOutput = namedtuple('ViTHeatmapEncoderOutput', ['out', 't_length', 'heatmaps'])
    def forward(self, x, t_length):
        N, C, T, H, W = x.shape
        x = rearrange(x, 'n c t h w -> (n t) c h w')

        feats, heatmap_out = self.vit(x)
        #heatmap_out: list of [n 2 h w] -> [n s 2 h w]
        heatmap_out = torch.stack(heatmap_out, dim=1)

        heatmap_out = rearrange(heatmap_out, '(n t) s d h w -> t n s d h w', t=T)
        out = rearrange(feats[-1], '(n t) c -> n c t', t=T)
        return self.ViTHeatmapEncoderOutput(out, t_length, heatmap_out)

        
    def migrate_checkpoint(self, ckpt):
        checkpoint = torch.load(ckpt, map_location='cpu')
        new_state_dict = OrderedDict()
        for key, value in checkpoint.items():
            if key.startswith('backbone.'):
                new_key = re.sub('^backbone\.', '', key)
                new_state_dict[new_key] = value
        self.vit.load_state_dict(new_state_dict, strict=False)
        del checkpoint

if __name__ == '__main__':
    model = ViTHeatmapEncoder('vit-base-p16', checkpoint='resources/vit-base/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth').to('cuda:1')
    x = torch.randn(1, 3, 300, 224, 224).to('cuda:1')
    with torch.no_grad():
        out = model(x, None)
    print(out[0].shape, out[1], out[2].shape)