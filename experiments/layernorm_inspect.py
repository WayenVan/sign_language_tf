import hydra 
from hydra.utils import instantiate
from torch import nn
import sys
sys.path.append('src')

def main():
    
    hydra.initialize_config_dir('/home/jingyan/Documents/sign_language_rgb/configs')
    cfg = hydra.compose('run/train/vitpose_vitpool_tconv_trans_ddp.yaml')
    
    model: nn.Module= instantiate(cfg.model, vocab=[1,])
    for name, m in model.named_modules():
        print(name)
        print(m.__class__.__name__)

main()