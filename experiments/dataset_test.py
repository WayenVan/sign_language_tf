from omegaconf import OmegaConf
from hydra.utils import instantiate
import sys
import os
from matplotlib import pyplot as plt
import numpy as np
import hydra
import cv2
os.chdir('/home/jingyan/Documents/sign_language_rgb')
sys.path.append('src')

hydra.initialize_config_dir('/home/jingyan/Documents/sign_language_rgb/configs')
cfg = hydra.compose('run/train/vitpose_trans_lightning')
dm = instantiate(cfg.datamodule)
dataset = dm.train_dataloader().dataset
frame = dataset[2]['video'].numpy()
frame  = frame.transpose(0, 2, 3, 1)
# plt.imshow(frame[30])
print(frame[30].shape)
cv2.imwrite('resources/0.jpg', frame[30])