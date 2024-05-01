import torch 

p = torch.load('outputs/train_lightning/2024-04-10_18-53-03/last.ckpt', map_location='cpu')
a = 0

import sys
sys.path.append('src')
from csi_sign_language.modules.tconv import TemporalConv1D


TemporalConv1D(12, 12, 12, d=1.)