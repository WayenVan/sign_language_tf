import torch
from einops import rearrange, reduce
from torch import nn
from .drop_path import DropPath

class FFN(nn.Module):
    
    def __init__(self, feature, neck_feature, act=nn.GELU, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ffn = nn.Sequential(
            nn.Linear(feature, neck_feature),
            act(),
            nn.Linear(neck_feature, feature)
        )
        
    def forward(self, x):
        return self.ffn(x)
    

class GlobalTemp(nn.Module):
    
    def __init__(self, feature, heads, dropout=0., *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attn = nn.MultiheadAttention(feature, heads, dropout)
        self.norm = nn.LayerNorm(feature)
    
    def forward(self, x, seq_length):
        # t n c
        x = self.norm(x)
        mask = self._make_video_mask(seq_length, x.size(dim=0))
        x, _ = self.attn(x, x, x, key_padding_mask=mask)
        return x
        
    @staticmethod
    def _make_video_mask(video_length: torch.Tensor, temporal_dim):
        batch_size = video_length.size(dim=0)
        mask = torch.ones(batch_size, temporal_dim)
        for idx in range(batch_size):
            mask[idx, :video_length[idx]] = 0
        return mask.bool().to(video_length.device)

class LocalTemp(nn.Module):
    
    def __init__(self, in_channels, kernel, padding, act=nn.GELU, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dwconv = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels, in_channels, kernel, padding=padding, groups=in_channels),
        )

    def forward(self, x):
        #x [t n c]
        x = rearrange(x, 't n c -> n c t')
        x = self.dwconv(x)
        x = rearrange(x, 'n c t -> t n c')
        return x

class Fusion(nn.Module):
    
    def __init__(self, embed_dim, heads, attn_drop=0., *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fusion = nn.MultiheadAttention(embed_dim, heads, dropout=attn_drop)
        self.norm_k = nn.LayerNorm(embed_dim)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)
    
    def forward(self, x, f_cls, f_p):
        """

        :param x: [t n c]
        :param f_cls: [n t c]
        :param f_p: [n c t]
        """
        T, _, _ = x.shape

        f_p = rearrange(f_p, 'n c t -> t n c')
        f_cls = rearrange(f_cls, 'n t c -> t n c')
        f_k = torch.stack((f_p, f_cls), dim=0)
        
        f_k = rearrange(f_k, 'k t n c -> k (t n) c')
        f_q = rearrange(x, '(q t) n c -> q (t n) c', q=1)
        
        f_k = self.norm_k(f_k)
        f_q = self.norm_q(f_q)
        
        out, _ = self.fusion(f_q, f_k, f_k)
        out = self.norm_out(out)
        out = rearrange(out, 'q (t n) c -> (q t) n c', t=T)
        return out
        

class FeatureMapSPModel(nn.Module):
    
    def __init__(self, channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bn = nn.BatchNorm3d(channels)
        self.dwconv = nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels)
        
    def forward(self, x):
        x = self.bn(x)
        return self.dwconv(x)

class Pooling(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.t_pooling_feats = nn.AvgPool3d((2, 1, 1))
        self.t_pooling_g = nn.AvgPool1d(2)
    
    def forward(self, f_vit, f_adapter, f_global, t_length):
        """

        :param f_vit: [n t (hw+1) d]
        :param f_adapter: [n c t h w]
        :param f_global: [t n c]
        :param t_length: [n]
        """
        
        f_global = rearrange(f_global, 't n c -> n c t')
        f_global = self.t_pooling_g(f_global)
        f_global = rearrange(f_global, 'n c t -> t n c')
        
        return self.t_pooling_feats(f_vit), self.t_pooling_feats(f_adapter), f_global, t_length//2

        
class AdapterBlock(nn.Module):
    
    def __init__(self, channel, type='local', drop_path=0., *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.ffn_f = FFN(channel, channel*2)
        self.map_projection = FeatureMapSPModel(channel)
    
        self.ffn_g = FFN(channel, channel*2)
        self.fusion = Fusion(channel, heads=8)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.t = type
        
        if type == 'local':
            self.temp_model = LocalTemp(channel, kernel=3, padding=1)
        elif type == 'global':
            self.temp_model = GlobalTemp(channel, heads=8)
        else:
            raise NotImplementedError('type need be local or global')

        
    def forward(self, f_vit, f_adapter, f_global, t_length):
        """
        :param f_vit: [n t (hw+1) d]
        :param f_adapter: [n c t h w]
        :param f_global: _description_
        :param t_length: _description_
        """
        _, _, _, H, W = f_adapter.shape

        f_cls = f_vit[:, :, 0]
        f_vit = f_vit[:, :, 1:]
        f_vit = rearrange(f_vit, 'n t (h w) c -> n c t h w', h=H, w=W)
        
        f_add = f_vit + f_adapter
        f_add = f_add + self.drop_path(self.map_projection(f_add))
        f_add = rearrange(f_add, 'n c t h w -> n t h w c')
        f_add = f_add + self.drop_path(self.ffn_f(f_add))
        f_add = rearrange(f_add, 'n t h w c -> n c t h w')
        
        f_p = reduce(f_add, 'n c t h w -> n c t', 'mean')
        f_global = self.fusion(f_global, f_cls, f_p)
        
        if self.t == 'local':
            f_global = f_global + self.drop_path(self.temp_model(f_global))
        if self.t == 'global':
            f_global = f_global + self.drop_path(self.temp_model(f_global, t_length))

        f_global = f_global + self.drop_path(self.ffn_g(f_global))
        
        return f_add, f_global

class Stem(nn.Module):
    
    def __init__(self, channels, target_size, act=nn.GELU, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(3, channels//4, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), 
            nn.Conv3d(channels//4, channels//2, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        self.st_conv = nn.Conv3d(channels//2, channels, (3, 3, 3), stride=(1, 2, 2), padding=1)
        self.act = act()
        self.norm = nn.BatchNorm3d(channels)
        self.target_size = target_size

    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.st_conv(x)
        x = self.act(x)
        x = self.norm(x)
        
        _, _, T, _, _ = x.shape
        x = rearrange(x, 'n c t h w -> (n t ) c h w')
        x = nn.functional.interpolate(x, (self.target_size[0], self.target_size[1]), mode='bilinear')
        x = rearrange(x, '(n t) c h w -> n c t h w', t=T)
        return x