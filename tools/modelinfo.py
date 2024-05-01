#! /usr/bin/env python3
import torch
from hydra.utils import instantiate
from hydra import compose, initialize_config_dir
import click
import os
from omegaconf import OmegaConf
import torch.nn as nn
import sys
sys.path.append('src')
from csi_sign_language.models.slr_model import SLRModel
import calflops

@click.option('--config', '-c', default='outputs/train_lightning/2024-04-13_19-54-02/config.yaml')
@click.option('-ckpt', '--checkpoint', default='outputs/train_lightning/2024-04-13_19-54-02/last.ckpt')
@click.command()
def main(config, checkpoint):
    cfg = OmegaConf.load(config)
    model = SLRModel.load_from_checkpoint(checkpoint, cfg=cfg, map_location='cpu', ctc_search_type='beam', strict=False).cuda()
    fake_data = torch.rand(1, 3, 150, 192, 192)
    fake_length = torch.tensor([150], dtype=torch.int64).cuda()
    
    flops, macs, params = calflops.calculate_flops(
        model= model,
        kwargs=dict(
            x = fake_data,
            t_length=fake_length
        ),
        print_results=True,
    )
    

if __name__ == '__main__':
    main()
    

