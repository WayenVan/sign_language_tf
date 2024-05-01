
from omegaconf import OmegaConf
import torch
import sys
sys.path.append('src')
from hydra.utils import instantiate
from functools import partial

cfg = OmegaConf.load("configs/train/default.yaml")

dataloader = instantiate(cfg.data.train_loader)
model: torch.nn.Module = instantiate(cfg.model, vocab=dataloader.dataset.get_vocab()).cuda()
model = model.backbone.resnet
storage = {}
    

for name, m in model.named_modules():
    print(name)

