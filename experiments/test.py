import torch 
from collections import OrderedDict

p = torch.load('resources/clip/clip-vit-base-p16_laion2b-pre_3rdparty_in1k-384px_20221220-558ed826.pth', map_location='cpu')
a = 0

import re
pattern = re.compile(r'^backbone.')
replacement = ''
new_dict = OrderedDict()

for key,value in p['state_dict'].items():
    if pattern.match(key):
        new_key = pattern.sub(replacement, key)
        new_dict[new_key] = value

import sys
sys.path.append('src')

