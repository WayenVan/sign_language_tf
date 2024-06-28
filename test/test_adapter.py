import torch
import hydra
import sys
sys.path.append('src')
from hydra.utils import instantiate
from omegaconf import OmegaConf
from csi_sign_language.modules.clip_adapters.model import ClipAdapter


def main():
    x = torch.rand((1, 3, 200, 224, 224)).to('cuda:1')
    l = torch.tensor([80]).to('cuda:1')
    
    model = ClipAdapter(
        1296,
        'ViT-B-16',
        'openai',
        img_size=(224, 224),
        # adapter_spec={
        #     'position': (2, 4, 6, 8, 10, 12),
        #     'type': ('L', 'L', 'L', 'L', 'L', 'G'),
        #     'drop_path': (0., 0., 0., 0., 0., 0.),
        # },
        adapter_spec={
            'position': (3, 6, 9, 12),
            'type': ('L', 'L', 'L', 'G'),
            'drop_path': (0., 0., 0., 0.),
        },
        pooling_spec=(0, 6)
    )
    
    model.to('cuda:1')
    for name, p in model.named_parameters():
        print(name)
        
    
    
if __name__ == '__main__':
    main()