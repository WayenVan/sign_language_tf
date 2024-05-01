import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from einops import rearrange, repeat
from einops.layers.torch import Reduce

from csi_sign_language.utils.misc import add_attributes

class Conv_Pool_Proejction(nn.Module):

    def __init__(self, in_channels, out_channels, neck_channels, dropout=0.5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        add_attributes(self, locals())
        self.drop = nn.Dropout(p=dropout, inplace=False)
        self.project1 = self.make_projection_layer(in_channels, neck_channels)
        self.project2 = self.make_projection_layer(neck_channels, neck_channels)
        self.linear = nn.Conv3d(neck_channels, out_channels, kernel_size=1, padding=0)
        self.spatial_pool = nn.AdaptiveAvgPool3d(output_size=(None, 1, 1))
        self.flatten = nn.Flatten(-3)

    @staticmethod
    def make_projection_layer(in_channels, out_channels):
        return nn.Sequential(
            nn.AvgPool3d((4, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1)),
            nn.Conv3d(in_channels, out_channels,  kernel_size=1, stride=1),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x, video_length):
        # n, c, t, h, w
        # n, l
        x = self.drop(x)
        x = self.project1(x)
        x = self.project2(x)
        x = self.spatial_pool(x)
        x = self.linear(x)
        x = self.flatten(x)
        
        video_length = video_length//2//2
        return x, video_length

class Header(nn.Module):
    
    def __init__(self, input_channels, out_channels, bottleneck_channels, dropout=0.5, pool='max', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        add_attributes(self, locals())
        self.conv = nn.Sequential(
            nn.Conv3d(input_channels, bottleneck_channels, 1, padding=0, bias=False),
            nn.BatchNorm3d(bottleneck_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.dropout = nn.Dropout3d(dropout, inplace=False)
        self.pool = Reduce('n c t (h1 h2) (w1 w2) -> n c t h1 w1', pool, h1=1, w1=1)
        self.flatten = nn.Flatten(-3)
        self.fc = nn.Conv1d(bottleneck_channels, out_channels, 1)

    def forward(self, x, t_length):
        x = self.dropout(x)
        x = self.conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x, t_length
    

class HeaderTconv(nn.Module):

    def __init__(self, in_channels, out_channels, neck_channels, pool='max', n_downsample=2, dropout=0.5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        add_attributes(self, locals())
        self.drop = nn.Dropout3d(p=dropout, inplace=False)
        self.proj1 = nn.Conv3d(in_channels, neck_channels, kernel_size=1, padding=0)

        self.downsamples = nn.ModuleList([
            self.make_pool_cnn(neck_channels, neck_channels, pool) for i in range(n_downsample)
        ])
        self.proj2 = nn.Conv3d(neck_channels, out_channels, kernel_size=1, padding=0)
        self.spatial_pool = nn.AdaptiveAvgPool3d(output_size=(None, 1, 1))
        self.flatten = nn.Flatten(-3)

    @staticmethod
    def make_pool_cnn(in_channels, out_channels, pool='max'):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels,  kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d((2, 1, 1), stride=(2, 1, 1)) if pool=='max' else nn.AvgPool3d((2, 1, 1), stride=(2, 1, 1)),
        )

    def forward(self, x, video_length):
        # n, c, t, h, w
        # n, l
        x = self.drop(x)
        x = self.proj1(x)
        
        for down in self.downsamples:
            x = down(x)

        x = self.proj2(x)
        x = self.spatial_pool(x)
        x = self.flatten(x)

        video_length = video_length//(2*self.n_downsample)
        return x, video_length

class X3d(nn.Module):
    
    x3d_spec = dict(
        x3d_m=dict(
            channels=(192, 432),
            input_shape=(224, 224)
            ),
        x3d_s=dict(
            channels=(192, 432),
            input_shape=(160, 160)
            ),
    )

    def __init__(self, x3d_type='x3d_s', header=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        add_attributes(self, locals())

        self.input_size_spatial = self.x3d_spec[x3d_type]['input_shape']
        self.x3d_out_channels, self.conv_neck_channels = self.x3d_spec[x3d_type]['channels']

        x3d = torch.hub.load('facebookresearch/pytorchvideo', x3d_type, pretrained=True)
        self.move_x3d_layers(x3d)
        del x3d
        self.spec = self.x3d_spec[x3d_type]
        self.header = header

    def move_x3d_layers(self, x3d: nn.Module):
        blocks = x3d.blocks
        self.stem = copy.deepcopy(blocks[0])
        self.res_stages = nn.ModuleList(
            [copy.deepcopy(block) for block in blocks[1:-1]]
            )

    def forward(self, x, t_length):
        """
        :param x: [n, c, t, h, w]
        """
        N, C, T, H, W = x.shape
        # assert (H, W) == self.input_size_spatial, f"expect size {self.input_size_spatial}, got size ({H}, {W})"
        stages_out = []

        x = self.stem(x)
        stem_out = x

        for stage in self.res_stages:
            x = stage(x)
            stages_out.append(x)
        
        if self.header is not None:
            x, t_length= self.header(x, t_length)

        return x, t_length, stem_out, stages_out
    