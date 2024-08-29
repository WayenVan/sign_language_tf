import torch
import hydra
import sys

sys.path.append("src")
from hydra.utils import instantiate
import socket
from tqdm import tqdm


def test_dm():
    hydra.initialize_config_dir("/root/projects/sign_language_transformer/configs")
    cfg = hydra.compose("run/train/heatmapresv2")
    # cfg = hydra.compose('run/train/dual')
    print(socket.gethostname())
    # cfg.datamodule.num_workers = 0

    datamodule = instantiate(cfg.datamodule)
    train_loader = datamodule.train_dataloader()
    for i in tqdm(train_loader):
        pass
    return


if __name__ == "__main__":
    test_dm()
