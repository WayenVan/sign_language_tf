import torch 
from collections import OrderedDict

from mmpretrain.models.backbones.vision_transformer import VisionTransformer
p = torch.load('resources/vifi_clip/vifi_clip_10_epochs_k400_full_finetuned.pth', map_location='cuda:1')
a = 0

import re
pattern = re.compile(r'^module.image_encoder.')
replacement = ''
new_dict = OrderedDict()

for key,value in p['model'].items():
    if pattern.match(key):
        new_key = pattern.sub(replacement, key)
        new_dict[new_key] = value



