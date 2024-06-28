import torch
from collections import namedtuple
from torch import nn
import open_clip
from open_clip.transformer import VisionTransformer, _expand_token, Transformer
from open_clip import CLIP
from torchvision.transforms import Normalize
from einops import repeat, rearrange

from .adapter import *


class ClipAdapter(nn.Module):
    
    def __init__(
        self, 
        n_class,
        clip_model,
        pretrained,
        img_size=(224, 224),
        adapter_spec={
            'position': (2, 4, 6, 8, 10, 12),
            'type': ('L', 'L', 'L', 'G', 'G', 'G'),
            'drop_path': (0., 0., 0., 0.1, 0.1, 0.1)
        },
        pooling_spec=(0, 6),
        *args, 
        **kwargs) -> None:
        super().__init__(*args, **kwargs)
        clip = open_clip.create_model(clip_model, pretrained=pretrained)
        self.vit: VisionTransformer = clip.visual
        del clip
        
        self.embed_dim = int(len(self.vit.class_embedding))

        self.data_norm = Normalize(std=self.vit.preprocess_cfg['std'], mean=self.vit.preprocess_cfg['mean'])

        scale = self.embed_dim ** -0.5
        self.f_global = nn.Parameter(scale*torch.randn(self.embed_dim))

        
        feature_map_size = (self.vit.grid_size[0], self.vit.grid_size[1])
        self.adapter_stem = Stem(self.embed_dim, feature_map_size)
        self._create_adapters(adapter_spec)
        self._create_poolings(pooling_spec)
        
        self.adapter_proj = nn.Linear(self.embed_dim, self.vit.output_dim)
        self.header = nn.Linear(self.vit.output_dim, n_class)
    
    def _create_poolings(self, pooling_spec):
        self.poolings = nn.ModuleDict()
        for idx in pooling_spec:
            self.poolings[f'pooling_{idx}'] = Pooling()
        self.pooling_spec = pooling_spec

    def _create_adapters(self, spec):
        self.adapters = nn.ModuleDict()
        positions = spec['position']
        types = spec['type']
        drop_paths = spec['drop_path']
        
        for pos, type, d in list(zip(positions, types, drop_paths)):
            if type == 'L':
                t = 'local'
            elif type == 'G':
                t = 'global'
            else:
                raise NotImplementedError()

            self.adapters[f'adapter_{pos}'] = AdapterBlock(self.embed_dim, type=t, drop_path=d)
        
        self.adapter_positions = positions
            
    
    ClipAdapterOutput = namedtuple('ClipAdapterOutput', ('out', 't_length'))
    def forward(self, x, t_length):
        """
        :param x: [n c t h w]
        :param t_length: [n]
        """
        x = self.pre_proecess(x)
        f_vit = self.vit_stem_forward(rearrange(x, 'n c t h w -> (n t) c h w'))
        f_adapter = self.adapter_stem(x)
        
        f_vit, f_adapter, f_global, t_length = self.block_forward(
            f_vit,
            f_adapter,
            t_length
        )
        
        T, _, _ = f_global.shape
        out_feats_vit, _ = self.vit_post_forward(f_vit) 
        out_feats_vit = rearrange(out_feats_vit, '(n t) c -> t n c', t=T)
        
        #fusiong output
        out = self.adapter_proj(f_global)
        out = self.header(out_feats_vit+out)

        return self.ClipAdapterOutput(
            out=out,
            t_length=t_length
        )

    def pre_proecess(self, x):
        T = int(x.size(2))
        x = rearrange(x, 'n c t h w -> (n t) c h w')
        x = self.data_norm(x)
        x = rearrange(x, '(n t) c h w -> n c t h w', t=T)
        return x
        
    def vit_stem_forward(self, x):
        x = self.vit.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.vit.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.vit.positional_embedding.to(x.dtype)

        x = self.vit.patch_dropout(x)
        x = self.vit.ln_pre(x)

        return x

    def block_forward(self, f_vit, f_adapter,t_length):
        """
        :param f_vit: [(n t) (hw) c]
        :param f_adapter: [n c t h w]
        :param t_length: [n]
        """
        N,_,T,_,_ = f_adapter.shape
        f_vit = f_vit.permute(1, 0, 2)  # NLD -> LND

        f_global = repeat(self.f_global, 'd -> t n d', n=N, t=T)

        t: Transformer = self.vit.transformer
        for i in range(len(t.resblocks)+1):
            if t.grad_checkpointing and not torch.jit.is_scripting():
                raise NotImplementedError()
            else:
                
                f_vit = rearrange(f_vit, 'l (n t) c -> n t l c', t=T)
                if i in self.adapter_positions:
                    f_adapter, f_global = self.adapters[f'adapter_{i}'](f_vit, f_adapter, f_global, t_length)
                
                if i in self.pooling_spec:
                    f_vit, f_adapter, f_global, t_length = self.poolings[f'pooling_{i}'](f_vit, f_adapter, f_global, t_length) 

                _, T, _, _ = f_vit.shape
                f_vit = rearrange(f_vit, 'n t l c -> l (n t) c')

                if i < len(t.resblocks):
                    f_vit = t.resblocks[i](f_vit, attn_mask=None)

        f_vit = f_vit.permute(1, 0, 2)  # LND -> NLD

        return f_vit, f_adapter, f_global, t_length

    def vit_post_forward(self, x):
        if self.vit.attn_pool is not None:
            if self.vit.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.vit.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.vit.attn_pool(x)
                if self.vit.attn_pool_type == 'parallel':
                    pooled = self.vit.attn_pool_contrastive(x)
                else:
                    assert self.vit.attn_pool_type == 'cascade'
                    pooled = self.vit.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.vit.attn_pool(x)
                x = self.vit.ln_post(x)
                pooled, tokens = self.vit._global_pool(x)
        elif self.vit.final_ln_after_pool:
            pooled, tokens = self.vit._global_pool(x)
            pooled = self.vit.ln_post(pooled)

        else:
            x = self.vit.ln_post(x)
            pooled, tokens = self.vit._global_pool(x)

        if self.vit.proj is not None:
            pooled = pooled @ self.vit.proj

        return pooled, tokens

        
    
        
    

    