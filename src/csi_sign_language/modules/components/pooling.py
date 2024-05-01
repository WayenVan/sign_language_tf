import torch
import torch.nn as nn
import torch.functional as F
from einops import einsum, rearrange

class NormPoolTemp(nn.Module):
    
    def __init__(self, kernel_size, stride) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.unfold = nn.Unfold(
            kernel_size=(kernel_size, 1),
            stride=(stride, 1))
    
    def forward(self, x: torch.Tensor):
        input = x
        # T, N, C, H, W
        T, _, C, H, W = x.shape
        x = rearrange(x, 't n c h w -> n c t (h w)')
        unforlded = self.unfold(x)
        unforlded = rearrange(
            unforlded, 
            'n (c k) (l h w) -> n c k l h w', 
            h=H, 
            l=(T-self.kernel_size)//self.stride+1,
            c=C,
            k=self.kernel_size)
        #N, C, K, L, (H, W)
        _, C, _, _, H, W = unforlded.shape
        norm_t = torch.linalg.vector_norm(
            rearrange(unforlded, 'n c k l h w -> n k l (c h w)'), 
            dim=-1)
        # norm_t: N, K, L
        score_t = torch.softmax(norm_t/(C * H * W)**0.5, 1)
        unforlded = rearrange(unforlded, 'n c k l h w -> n k l c h w')
        return einsum(unforlded, score_t, 'n k l ..., n k l -> l n ...')
    