import torch
import hydra
import sys
sys.path.append('src')
from hydra.utils import instantiate
from omegaconf import OmegaConf
from csi_sign_language.modules.clip_adapters.model import ClipAdapter
from csi_sign_language.modules.vifi_clip.vifi_clip_encoder import ClipEncoder


def main():
    ckpt = 'resources/vifi_clip/vifi_clip_10_epochs_k400_full_finetuned.pth'
    device = 'cuda:1'
    m = ClipEncoder(
        ckpt,
        2,
        512,
        0,
        0,
        2
    ).to(device)
    
    x = torch.randn(1, 3, 100, 224, 224).to(device)
    t_length = torch.tensor([100, 100]).to(device)
    
    out = m(x, t_length)
    print(out)
    
    
if __name__ == '__main__':
    main()