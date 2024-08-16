if __name__ == '__main__':
    import sys
    sys.path.append('src')
    from csi_sign_language.modules.resnet_focal_encoder.multihead_attention import ScaledDotProduct, MultiHeadDispatch
else:
    from .multihead_attention import MultiHeadDispatch, ScaledDotProduct
from torch import nn
import torch
from einops import rearrange
from xformers.components.feedforward.mlp import MLP

class FocalAttention(nn.Module):
    
    def __init__(
        self, 
        embed_dims,
        feats_dims,
        heads,
        qkv_bias=True,
        feed_forward_dropout= 0.,
         *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.feats_projection = nn.Conv2d(feats_dims, embed_dims, 1, 1, bias=True)
        self.attn = MultiHeadDispatch(
            embed_dims, 
            heads, 
            ScaledDotProduct()
        )

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

        self.hands_token = nn.Parameter(torch.randn(embed_dims), requires_grad=True)
        self.face_token = nn.Parameter(torch.randn(embed_dims), requires_grad=True)
        
        self.feedforward = MLP(
            dim_model=embed_dims,
            dropout=feed_forward_dropout,
            activation='gelu',
            hidden_layer_multiplier=2,
            bias=True
        )


    def forward(self, feats_map):
        # feats_map: B, C, H, W
        B, C, H, W = feats_map.shape
        feats_map = self.feats_projection(feats_map)

        hands_token = rearrange(self.hands_token, 'c -> () c')
        face_token = rearrange(self.face_token, 'c -> () c')

        keys = rearrange(feats_map, 'b c h w -> b (h w) c')
        querys = torch.stack([face_token, hands_token], dim=1)
        
        keys = self.norm1(keys)
        querys = self.norm2(querys)

        attn_out, attn_weight = self.attn(querys, keys, keys)
        attn_out = self.feedforward(attn_out)
        hands_token, face_token = torch.split(attn_out, 1, dim=1)
        hands_token, face_token = (hands_token.squeeze(1), face_token.squeeze(1))
        attn_weight = rearrange(attn_weight, 'b nh k (h w) -> b nh k h w', h=H, w=W)

        return hands_token, face_token, attn_weight
    
if __name__ == '__main__':
    feats_map = torch.randn(2, 512, 7, 7).to('cuda:1')
    model = FocalAttention(512, 512, 8).to('cuda:1')
    hands_token, face_token, attn_weight = model(feats_map)
    print(hands_token.shape, face_token.shape, attn_weight.shape)