import torch
import hydra
import sys
sys.path.append('src')
from hydra.utils import instantiate
from omegaconf import OmegaConf

def test_hrnet_rnn():
    cfg = "configs/train/hrnet_rnn.yaml"
    cfg = OmegaConf.load(cfg)
    loader = instantiate(cfg.data.train_loader)
    model = instantiate(cfg.model, vocab=loader.dataset.get_vocab()).to(cfg.device)
    data = next(iter(loader))
    video = data['video'].to(cfg.device)
    lgt = data['video_length'].to(cfg.device)

    output = model(video, lgt)

    return



if __name__ == "__main__":
    test_hrnet_rnn()
    
    