import torch
import hydra
import sys
sys.path.append('src')
from hydra.utils import instantiate
from omegaconf import OmegaConf
from csi_sign_language.modules.clip_adapters.model import ClipAdapter
from csi_sign_language.modules.efficient_decoder.efficient_attention import SparseAttention, RandomMaskGenerator, DiagonalMaskGenerator, BucketRandomAttention
from xformers.components import MultiHeadDispatch
from xformers.components.attention import ScaledDotProduct
import torch.distributed
from torch import nn


def test_attn_mask():
    device = 'cuda:1'

    t = torch.randn(200, 2, 1024)
    l = torch.tensor([32, 32], dtype=torch.int64)

    m = SparseAttention(
        d_model=1024,
        num_heads=4,
        mask_generator=DiagonalMaskGenerator(step=4)
        # mask_generator=RandomMaskGenerator(bucket_size=4)
    )
    m2 = nn.MultiheadAttention(
        embed_dim=1024,
        num_heads=4,
    )
    m3 = MultiHeadDispatch(
        1024,
        4,
        attention=ScaledDotProduct(),
    )

    
    m4 = BucketRandomAttention(
        1024,
        4,
        4
    )

    #warm up 
    for i in range(8):
        attnetion_analysis(m4, t, t, t, device)

@torch.inference_mode()
def attnetion_analysis(module, k, q, v, device):
    import time
    module, k, q, v = (x.to(device) for x in (module, k, q, v))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    s = time.time()
    torch.cuda.synchronize(device)
    module(k, q, v)
    torch.cuda.synchronize(device)
    e = time.time()
    time = e - s

    max_memory = torch.cuda.max_memory_allocated(device) // 2 ** 20
    
    print(f'max memory: {max_memory}')
    print(f'time: {time}')

    
    
if __name__ == '__main__':
    test_attn_mask()