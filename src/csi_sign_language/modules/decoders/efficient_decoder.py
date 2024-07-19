import torch
import random
from torch import nn
from einops import rearrange, einsum, repeat
from ..components.drop_path import DropPath
from ..components.masked_attention import make_diagonal_mask, make_random_mask_bucket, plot_mask
from ..components.my_scale_dot_product import MyScaledDotProduct
from xformers.components import MultiHeadDispatch
from xformers.components.attention.attention_mask import AttentionMask
from xformers.components.input_projection import InputProjection
from xformers.components.feedforward import MLP
from xformers.components.activations import Activation

class DiagonalMaskGenerator():
    
    def __init__(self, step) -> None:
        self.step = step
    
    def __call__(self, Lq, Lk, device):
        return make_diagonal_mask(Lq, Lk, device, k=self.step)

class RandomMaskGenerator():
    
    def __init__(self, bucket_size) -> None:
        self.bucket_size = bucket_size
    
    def __call__(self, Lq, Lk, device):
        return make_random_mask_bucket(Lq, Lk, device, bucket_size=self.bucket_size)

class SparseAttention(nn.Module):
    
    def __init__(
        self, 
        d_model,
        num_heads,
        mask_generator,
        *args,
        **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.attn = MultiHeadDispatch(
            dim_model=d_model,
            num_heads=num_heads,
            attention=MyScaledDotProduct(),
            use_rotary_embeddings=False
        )
        self.mask_generator = mask_generator
        
    def forward(self, q, k, v, key_length=None):
        #[t n c]
        import time
        t = time.time()
        N = q.shape[1]
        Lq = q.shape[0]
        Lk = k.shape[0]
        Lv = v.shape[0]
        assert Lk == Lv

        t1 = time.time()
        with torch.no_grad():
            mask = self.mask_generator(Lq, Lk, q.device)
            if key_length is not None:
                mask = repeat(mask, 'q k -> n q k', n=N).contiguous()
                for i in range(N):
                    mask[i, :, :key_length[i]] = 1
            mask = mask.bool()
        
        t2 = time.time()

        q, k, v = tuple(rearrange(x, 't n c -> n t c') for x in (q, k, v))
        with torch.cuda.device(q.device):
            t3 = time.time()
            y = self.attn(q, k, v, att_mask=mask)
            t4 = time.time()
        y = rearrange(y, 'n t c -> t n c')
        print(t1-t, t2-t1, t3-t2, t4-t3)
        return y
    

class BucketRandomAttention(nn.Module):
    
    def __init__(
        self, 
        d_model,
        num_heads,
        bucket_size,
        *args,
        **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.attn = nn.MultiheadAttention(
            d_model,
            num_heads,
        )
        self.bucket_size = bucket_size
    
    @staticmethod
    def split_list_into_groups(lst, group_size):
        return [lst[i:i + group_size] for i in range(0, len(lst), group_size)]
    
    @staticmethod
    def make_mask(length, index, device):
        N = length.shape[0]
        L = len(index)
        index = torch.tensor(index, dtype=torch.int64, device=device)
        mask = torch.zeros((N, L), device=device).bool()
        for i in range(N):
                mask[i, :] = (index >= length[i])

        return mask
        
    def forward(self, q, k, v, key_length=None):
        # [t n c]
        Lk = k.shape[0]
        groups = self.split_list_into_groups(list(range(Lk)), self.bucket_size)
        sampled_index = [random.choice(group) for group in groups]
        
        k = k[sampled_index, :, :]
        v = v[sampled_index, :, :]

        if key_length is not None:
            mask = self.make_mask(Lk, key_length, q.device)
        else:
            mask = None

        return self.attn(q, k, v, key_padding_mask=mask)


        
        
        

        
class ConvHeader(nn.Module):
    
    def __init__(
        self, 
        in_channels,
        out_channels,
        neck_channels,
        kernerl_size=5,
        drop_path=0.,
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernerl_size, groups=in_channels, padding=kernerl_size//2, bias=False),
            nn.Conv1d(in_channels, neck_channels, 1, padding=0, bias=False),
            nn.BatchNorm1d(neck_channels),
            nn.GeLU(inplace=True),
        )
        self.pool1 = nn.AvgPool1d(2)

        self.conv2 = nn.Sequential(
            nn.Conv1d(neck_channels, neck_channels, kernerl_size, groups=neck_channels, padding=kernerl_size//2, bias=False),
            nn.Conv1d(neck_channels, out_channels, 1, padding=0, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GeLU(inplace=True),
        )
        self.pool2 = nn.AvgPool1d(2)

        self.drop_path = DropPath(drop_path)
    
    def forward(self, x, t_length):
        #[t n c]
        x = rearrange(x, 't n c -> n t c')
        x = self.conv1(x) + self.drop_path(x)
        x = self.pool1(x)
        x = self.conv2(x) + self.drop_path(x)
        x = self.pool2(x)
        x = rearrange(x, 'n t c -> t n c')
        t_length = t_length // 4
        return x, t_length
    
    
    
    class EfficientDecoderBlock(nn.Module):
        
        def __init__(
            self, 
            dim,
            n_head,
            mask_generator,
            ff_dropout=0.,
            drop_path=0.,
            *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            
            self.conv = nn.Conv1d(dim, dim, 3, padding=1, bias=False, groups=dim),
            self.bn =  nn.BatchNorm1d(dim),

            self.atten = SparseAttention(dim, n_head, mask_generator)
            self.ff = MLP(
                dim,
                ff_dropout,
                Activation.GeLU,
                hidden_layer_multiplier=2,
            )
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            self.drop_path = DropPath(drop_path)

        def forward(self, x, t_length):
            y = self.conv(x) + self.drop_path(x)
            y = self.bn(y)
            y = self.atten(y, y, y, key_length=t_length) + self.drop_path(y)
            y = self.norm1(y)
            y = self.ff(y) + self.drop_path(y)
            y = self.norm2(y)
            return y
    
    
    