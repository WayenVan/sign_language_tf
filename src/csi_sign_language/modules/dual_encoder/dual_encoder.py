from mmpose.models import CSPNeXt
from einops import rearrange, reduce
import torch
from collections import namedtuple
from torch import nn
from mmpose.apis.inference import init_model, init_default_scope
from mmengine import Config
from mmpose.models import TopdownPoseEstimator
from mmpose.models.backbones.cspnext import CSPNeXt
from mmcv.cnn.bricks.conv_module import ConvModule
# channel refer to SlowFast network

# in_channels, out_channels, num_blocks, add_identity, use_spp
my_archs = {
    'v1':(
        [4, 8, 3, True, False],
        [8, 16, 3, True, False],
        [16, 32, 6, True, False],
        [32, 64, 6, True, False],
    ),
    'v2':(
        [8, 16, 3, True, False],
        [16, 32, 6, True, False],
        [32, 64, 6, True, False],
        [64, 128, 3, True, False],
    ),
    'v3':(
        [16, 32, 3, True, False],
        [32, 64, 6, True, False],
        [64, 128, 6, True, False],
        [128, 256, 3, True, False],
    ),
}
def create_lightweit_cspnext(type):
    return CSPNeXt(
        arch_ovewrite=my_archs[type],
        expand_ratio=0.5,
        deepen_factor=1.,
        widen_factor=1.,
        out_indices=(1, 2, 3, 4),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
    )
    
class DownSampleFusedOut(nn.Module):
    
    def __init__(
        self, 
        in_channels,
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels

        self.downsample = nn.ModuleDict()
        for i in range(len(in_channels) - 1):
            self.downsample[f'downsample{i}'] = self.down_sample_conv(in_channels[i])
        
    
    def down_sample_conv(self, c_in):
        return ConvModule(
            c_in,
            c_in * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=dict(type='BN', momentum=0.1),
            bias=False
        )
    
    def forward(self, x):
        out = x[0]
        for i in range(len(self.in_channels) - 1):
            out = self.downsample[f'downsample{i}'](out) + x[i+1] 
        return out
    
class Fusion(nn.Module):
    
    def __init__(self, channel_pose, channel_rgp, act: nn.Module, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bn_rgp = nn.BatchNorm2d(channel_rgp)
        self.proj = nn.Conv2d(channel_rgp, channel_pose, 1, 1, 0)

    def forward(self, x_pose, x_rgb, padding_ratio):
        #n c h w
        _, _, H_dw, W_dw = x_pose.shape
        _, _, H, W= x_rgb.shape

        padding_size = int(H_dw * padding_ratio)
        x_pose = x_pose[:, :, :-padding_size]
        if H != H_dw or W != W_dw:
            x_pose = nn.functional.interpolate(x_pose, (H, W))

        return self.proj(self.bn_rgp(x_rgb)) + x_pose
    
    
class DualEncoder(nn.Module):
    
    def __init__(
        self, 
        dwpose_cfg,
        dwpose_ckpt,
        input_size,
        rgb_stream_arch = 'v1',
        dropout=0.1,
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        #load_dwpose
        cfg = Config.fromfile(dwpose_cfg)
        init_default_scope('mmpose')
        dwpose: TopdownPoseEstimator = init_model(cfg, dwpose_ckpt, device='cpu', cfg_options=dict(model=dict(backbone=dict(out_indices=(1, 2, 3, 4)))))
        self.dwpose_cspnext: CSPNeXt = dwpose.backbone
        self.dwpose_input_size = list(cfg.codec.input_size)[::-1]
        del dwpose
        
        self.input_size = input_size
        self.rgb_cspnext = create_lightweit_cspnext(rgb_stream_arch)
        
        self.out_dims_pose = [getattr(self.dwpose_cspnext, f'stage{i}')[-1].final_conv.out_channels for i in range(1, 5)]
        self.out_dims_rgb = [getattr(self.rgb_cspnext, f'stage{i}')[-1].final_conv.out_channels for i in range(1, 5)]
        
        self.fusions = nn.ModuleDict()
        for i in range(4):
            self.fusions[f'fusion{i}'] = Fusion(self.out_dims_pose[i], self.out_dims_rgb[i] ,act=nn.LeakyReLU())
        
        self.fused_scale_out = DownSampleFusedOut(self.out_dims_pose)
        self.drop = nn.Dropout2d(dropout)

        self._freeze_dwpose()
    
    def _freeze_dwpose(self):
        for p in self.dwpose_cspnext.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.dwpose_cspnext.eval()
        return

    
    DualEncoderOut = namedtuple('DualEncoderOut', ['out', 't_length'])
    def forward(self, x, t_length=None):
        # n c t h w
        N, _, _, _, _ = x.shape
        x = rearrange(x, 'n c t h w -> (n t) c h w')
        x_dw, padding_ratio_dw= self.padding_h_for_dwpose(x)
        x_rgb = x
        
        out_dw = self.dwpose_cspnext(x_dw)
        out_rgb = self.rgb_cspnext(x_rgb)
        
        out = []
        for i in range(4):
            out.append(self.fusions[f'fusion{i}'](out_dw[i], out_rgb[i], padding_ratio_dw))
        # out = self.fused_scale_out(out)
        out = out[-1]
        out = self.drop(out)
        out = reduce(out, 'n c h w -> n c', 'mean')
        out = rearrange(out, '(n t) c -> n c t', n=N)
        
        return self.DualEncoderOut(
            out=out,
            t_length=t_length
        )
        
        
    def padding_h_for_dwpose(self, x):
        # n c h w
        H_dw, W_dw = self.dwpose_input_size[0], self.dwpose_input_size[1]
        H, W = self.input_size[0], self.input_size[1]
        
        assert H_dw >= W_dw
        assert H == W
        assert W >= W_dw

        padding_size = (H_dw - W_dw)
        if W == W_dw:
            x = nn.functional.pad(x, (0, 0, 0, padding_size))
        elif W > W_dw:
            x = nn.functional.interpolate(x, (W_dw, W_dw))
            x = nn.functional.pad(x, (0, 0, 0, padding_size))

        assert x.shape[-2] == H_dw
        assert x.shape[-1] == W_dw

        return x, padding_size / H_dw
        
if __name__ == '__main__':
    model = DualEncoder(
        'resources/dwpose-l/rtmpose-l_8xb64-270e_coco-ubody-wholebody-256x192.py',
        'resources/dwpose-l/dw-ll_ucoco.pth',
        input_size=(224, 224)
    ).to('cuda:1')
    
    print(model.out_dims_rgb)
    print(model.dwpose_input_size)
    # model = create_lightweit_cspnext().to('cuda:1')
    input = torch.randn((2, 3, 100, 224, 224)).to('cuda:1')
    out = model(input)
    print(out[0].shape)