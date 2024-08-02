from omegaconf import OmegaConf
from hydra.utils import instantiate
import sys
import os
from matplotlib import pyplot as plt
import numpy as np
import hydra
import cv2
import tqdm
os.chdir('/home/jingyan/Documents/sign_language_rgb')
sys.path.append('src')

hydra.initialize_config_dir('/home/jingyan/Documents/sign_language_rgb/configs')
cfg = hydra.compose('run/train/vitpose_trans_lightning')
dm = instantiate(cfg.datamodule)
dataset = dm.train_dataloader().dataset


max_l = 0
for i, data in tqdm.tqdm(enumerate(dataset)):
    max_l = max(max_l, data['video'].shape[0])